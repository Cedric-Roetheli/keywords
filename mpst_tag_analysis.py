"""
Analyze MPST tags vs TMDb hit status across years using kagglehub-loaded datasets.

- Loads MPST and TMDb via kagglehub (or user-specified local files).
- Merges on imdb_id, infers year from TMDb, defines hits by within-year metric percentile.
- Computes per-year tag overrepresentation (log-odds), JS divergence, and Jaccard turnover.
- Outputs tables and plots highlighting potential shock years.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import kagglehub
from kagglehub import KaggleDatasetAdapter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Defaults
DEFAULT_MPST = "cryptexcode/mpst-movie-plot-synopses-with-tags"
DEFAULT_TMDB = "asaniczka/tmdb-movies-dataset-2023-930k-movies"
IMDB_CANDIDATES = ["imdb_id", "imdbId", "imdbid", "imdb"]
YEAR_CANDIDATES = ["release_year", "year", "release_date"]
TAG_CANDIDATES = ["tags", "tag", "Tag", "Tags"]


# ----------------------------
# CLI parsing
# ----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MPST tags vs TMDb hits analysis.")
    parser.add_argument("--mpst-slug", default=DEFAULT_MPST)
    parser.add_argument("--tmdb-slug", default=DEFAULT_TMDB)
    parser.add_argument("--mpst-file", default=None, help="Optional path within MPST dataset or local file")
    parser.add_argument("--tmdb-file", default=None, help="Optional path within TMDb dataset or local file")
    parser.add_argument("--outdir", default="./outputs_mpst_analysis")
    parser.add_argument("--year-min", type=int, default=1970)
    parser.add_argument("--year-max", type=int, default=2023)
    parser.add_argument("--metric", default="vote_count", choices=["vote_count", "popularity", "revenue"])
    parser.add_argument("--hit-percentile", type=float, default=0.95)
    parser.add_argument("--min-tmdb-metric", type=float, default=50)
    parser.add_argument("--top-tags-k", type=int, default=15)
    parser.add_argument("--shock-years", default="2001,2008")
    parser.add_argument("--shock-window", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=123)
    return parser.parse_args()


# ----------------------------
# File handling
# ----------------------------
@dataclass
class DatasetInfo:
    slug: str
    paths: List[str]
    root_dir: Path


def auto_file_path(slug: str, preferred: Optional[str]) -> DatasetInfo:
    """
    Pick file(s) to load. If a user-specified existing path is given, use it directly.
    Otherwise, download via kagglehub, list files, and pick the largest with preferred extensions.
    """
    if preferred:
        cand = Path(preferred)
        if cand.exists():
            return DatasetInfo(slug=slug, paths=[str(cand.resolve())], root_dir=cand.parent)

    root = Path(kagglehub.dataset_download(slug))
    if preferred:
        cand = Path(preferred)
        if not cand.is_absolute():
            cand = root / cand
        if not cand.exists():
            raise FileNotFoundError(f"Preferred file {cand} not found for dataset {slug}")
        try:
            rel = cand.relative_to(root)
        except ValueError:
            rel = cand
        return DatasetInfo(slug=slug, paths=[str(rel)], root_dir=root)

    records = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if "partition.json" in name or name.endswith(".complete"):
                continue
            full = Path(dirpath) / name
            rel = full.relative_to(root)
            try:
                size = full.stat().st_size
            except OSError:
                size = 0
            records.append((str(rel), size))
    if not records:
        raise FileNotFoundError(f"No data files found in dataset {slug} at {root}")

    ext_priority = [".csv", ".parquet", ".json", ".tsv"]
    chosen = records
    for ext in ext_priority:
        subset = [(p, s) for p, s in records if p.lower().endswith(ext)]
        if subset:
            chosen = subset
            break
    chosen = sorted(chosen, key=lambda x: x[1], reverse=True)
    paths = [p for p, _ in chosen]
    return DatasetInfo(slug=slug, paths=paths, root_dir=root)


def load_kaggle_df(info: DatasetInfo, pandas_kwargs: Optional[dict] = None) -> pd.DataFrame:
    """
    Load one or more files with kagglehub.load_dataset. If absolute path exists, load via pandas directly.
    Concatenate if columns align; otherwise keep the first (largest) file.
    """
    pandas_kwargs = pandas_kwargs or {}
    frames: List[pd.DataFrame] = []
    base_cols = None
    for rel in info.paths:
        path_obj = Path(rel)
        if path_obj.is_absolute() and path_obj.exists():
            df = pd.read_csv(path_obj, **{k: v for k, v in pandas_kwargs.items()})
        else:
            try:
                df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, info.slug, rel, pandas_kwargs=pandas_kwargs)
            except TypeError:
                df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, info.slug, rel)
                if pandas_kwargs.get("usecols"):
                    df = df[pandas_kwargs["usecols"]]
        if base_cols is None:
            base_cols = list(df.columns)
            frames.append(df)
        else:
            if list(df.columns) == base_cols:
                frames.append(df)
            else:
                break
    if not frames:
        raise ValueError(f"Could not load any data from {info.slug}")
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


# ----------------------------
# Normalization utilities
# ----------------------------
def normalize_imdb(series: pd.Series) -> Tuple[pd.Series, int]:
    invalid = 0
    out = []
    for val in series:
        s = str(val).strip().lower().replace(" ", "")
        if not s or s in {"nan", "none"}:
            invalid += 1
            out.append(pd.NA)
            continue
        if s.isdigit():
            s = f"tt{s}"
        if not s.startswith("tt") and s[0].isdigit():
            s = f"tt{s}"
        if re.match(r"^tt\d+$", s):
            out.append(s)
        else:
            invalid += 1
            out.append(pd.NA)
    return pd.Series(out, dtype="string"), invalid


def extract_year_from_tmdb(df: pd.DataFrame) -> pd.Series:
    year = pd.Series([pd.NA] * len(df), dtype="Int64")
    for col in YEAR_CANDIDATES:
        if col in df.columns:
            if col == "release_date":
                vals = df[col].astype(str).str.extract(r"(\d{4})")[0]
                year = pd.to_numeric(vals, errors="coerce").astype("Int64")
            else:
                year = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            break
    return year


def detect_metric_column(columns: Sequence[str], preferred: str) -> str:
    order_map = {
        "vote_count": ["vote_count", "popularity", "revenue"],
        "popularity": ["popularity", "vote_count", "revenue"],
        "revenue": ["revenue", "vote_count", "popularity"],
    }
    for col in order_map.get(preferred, []):
        if col in columns:
            return col
    raise ValueError(f"No suitable metric column found. Available: {columns}")


def detect_imdb_column(df: pd.DataFrame) -> str:
    for c in IMDB_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"imdb_id column not found. Available: {df.columns.tolist()}")


def detect_tag_column(df: pd.DataFrame) -> str:
    for c in TAG_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"Tag column not found. Available: {df.columns.tolist()}")


def parse_tags(val) -> List[str]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    if isinstance(val, list):
        tokens = val
    else:
        text = str(val).strip()
        if not text:
            return []
        # Try JSON or literal list
        for loader in (json.loads, eval):
            try:
                parsed = loader(text)
                if isinstance(parsed, list):
                    tokens = parsed
                    break
            except Exception:
                tokens = None
        else:
            tokens = None
        if tokens is None:
            # delimiter-based
            if "|" in text:
                tokens = text.split("|")
            elif "," in text:
                tokens = text.split(",")
            else:
                tokens = [text]
    cleaned = []
    for t in tokens:
        if t is None:
            continue
        tok = str(t).strip().lower()
        tok = "_".join(tok.split())
        if tok:
            cleaned.append(tok)
    return cleaned


# ----------------------------
# TMDb hit computation
# ----------------------------
def compute_tmdb_hits(tmdb_df: pd.DataFrame, metric_col: str, args: argparse.Namespace) -> pd.DataFrame:
    df = tmdb_df.copy()
    df["year"] = extract_year_from_tmdb(df)
    imdb_col = detect_imdb_column(df)
    df["imdb_norm"], invalid = normalize_imdb(df[imdb_col])
    df = df.dropna(subset=["imdb_norm", "year", metric_col])
    if args.metric == "vote_count":
        df = df[df[metric_col] >= args.min_tmdb_metric]

    df = df[(df["year"] >= args.year_min) & (df["year"] <= args.year_max)]
    grouped = []
    for year, g in df.groupby("year"):
        if g.empty:
            continue
        pct = g[metric_col].rank(pct=True, method="max")
        g = g.assign(percentile=pct)
        g = g.assign(hit_tmdb=(g["percentile"] >= args.hit_percentile).astype(int))
        grouped.append(g[["imdb_norm", "year", metric_col, "percentile", "hit_tmdb"]])
    if not grouped:
        raise ValueError("No TMDb rows after filtering by year/metric.")
    return pd.concat(grouped, ignore_index=True)


# ----------------------------
# Per-year tag statistics
# ----------------------------
def log_odds(a_w: int, A: int, b_w: int, B: int, alpha: float = 0.5) -> float:
    return math.log((a_w + alpha) / (A - a_w + alpha)) - math.log((b_w + alpha) / (B - b_w + alpha))


def js_divergence(counter_a: Counter, counter_b: Counter) -> float:
    vocab = set(counter_a) | set(counter_b)
    if not vocab:
        return float("nan")
    a = np.array([counter_a.get(k, 0) for k in vocab], dtype=float)
    b = np.array([counter_b.get(k, 0) for k in vocab], dtype=float)
    sa, sb = a.sum(), b.sum()
    if sa == 0 or sb == 0:
        return float("nan")
    p = a / sa
    q = b / sb
    m = 0.5 * (p + q)
    eps = 1e-12
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    m = np.clip(m, eps, 1)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return float("nan")
    return len(set_a & set_b) / len(set_a | set_b) if (set_a | set_b) else float("nan")


def per_year_tag_stats(df: pd.DataFrame, top_k: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    yearly_rows = []
    summary_rows = []
    prev_top50 = None
    for year, g in df.groupby("year"):
        g_valid = g.dropna(subset=["mpst_hit"])
        hits = g_valid[g_valid["mpst_hit"] == 1]
        rest = g_valid[g_valid["mpst_hit"] == 0]
        n_top, n_rest = len(hits), len(rest)
        top_counter = Counter()
        rest_counter = Counter()
        for tags in hits["tags_parsed"]:
            top_counter.update(set(tags))
        for tags in rest["tags_parsed"]:
            rest_counter.update(set(tags))

        vocab = set(top_counter) | set(rest_counter)
        total_top = sum(top_counter.values())
        total_rest = sum(rest_counter.values())
        tag_scores = []
        for tag in vocab:
            ct_top = top_counter.get(tag, 0)
            ct_rest = rest_counter.get(tag, 0)
            if n_top == 0 or n_rest == 0:
                continue
            score = log_odds(ct_top, n_top, ct_rest, n_rest, alpha=0.5)
            p_top = ct_top / n_top if n_top else float("nan")
            p_rest = ct_rest / n_rest if n_rest else float("nan")
            delta = p_top - p_rest
            tag_scores.append((score, tag, ct_top, ct_rest, p_top, p_rest, delta))
        tag_scores.sort(key=lambda x: x[0], reverse=True)
        top_tags = tag_scores[:top_k]
        for score, tag, ct_top, ct_rest, p_top, p_rest, delta in top_tags:
            yearly_rows.append(
                {
                    "year": int(year),
                    "tag": tag,
                    "score_logodds": score,
                    "count_top": ct_top,
                    "count_rest": ct_rest,
                    "p_top": p_top,
                    "p_rest": p_rest,
                    "delta": delta,
                    "n_top": n_top,
                    "n_rest": n_rest,
                }
            )

        js = js_divergence(top_counter, rest_counter)
        current_top50 = {t for _, t, *_ in tag_scores[:50]}
        jac = jaccard(current_top50, prev_top50 or set()) if prev_top50 is not None else float("nan")
        summary_rows.append(
            {
                "year": int(year),
                "n_top": n_top,
                "n_rest": n_rest,
                "js_divergence": js,
                "jaccard_top50_prevyear": jac,
            }
        )
        prev_top50 = current_top50 if current_top50 else prev_top50
    yearly_df = pd.DataFrame(yearly_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values("year")
    return yearly_df, summary_df


# ----------------------------
# Shock window summary
# ----------------------------
def build_shock_window(summary_df: pd.DataFrame, shock_years: List[int], window: int) -> pd.DataFrame:
    rows = []
    for y in shock_years:
        window_years = list(range(y - window, y + window + 1))
        sub = summary_df[summary_df["year"].isin(window_years)]
        for _, r in sub.iterrows():
            rows.append(
                {
                    "window_id": f"{y}-{window}",
                    "year": int(r["year"]),
                    "js_divergence": r["js_divergence"],
                    "jaccard": r["jaccard_top50_prevyear"],
                    "n_top": r["n_top"],
                    "n_rest": r["n_rest"],
                }
            )
        if not sub.empty:
            rows.append(
                {
                    "window_id": f"{y}-{window}-mean",
                    "year": y,
                    "js_divergence": sub["js_divergence"].mean(),
                    "jaccard": sub["jaccard_top50_prevyear"].mean(),
                    "n_top": sub["n_top"].mean(),
                    "n_rest": sub["n_rest"].mean(),
                }
            )
    return pd.DataFrame(rows)


# ----------------------------
# Plotting
# ----------------------------
def plot_outputs(summary_df: pd.DataFrame, shock_years: List[int], window: int, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    def mark_shocks(ax):
        for y in shock_years:
            ax.axvline(y, color="gray", linestyle="--", alpha=0.6)
            ax.text(y, ax.get_ylim()[1], str(y), rotation=90, va="bottom", ha="right", fontsize=8, color="gray")

    if not summary_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(summary_df["year"], summary_df["js_divergence"], marker="o", label="JS divergence")
        mark_shocks(ax)
        ax.set_xlabel("Year")
        ax.set_ylabel("JS divergence")
        ax.set_title("JS divergence (hits vs rest) over time")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(outdir / "js_divergence_over_time_tags.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(summary_df["year"], summary_df["jaccard_top50_prevyear"], marker="o", color="tab:orange", label="Jaccard Top50")
        mark_shocks(ax)
        ax.set_xlabel("Year")
        ax.set_ylabel("Jaccard similarity (Top50 vs prev year)")
        ax.set_title("Year-over-year top tag set overlap")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(outdir / "jaccard_over_time_tags.png", dpi=150)
        plt.close(fig)

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(summary_df["year"], summary_df["js_divergence"], color="tab:blue", marker="o", label="JS divergence")
        ax1.set_ylabel("JS divergence", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2 = ax1.twinx()
        turnover = 1 - summary_df["jaccard_top50_prevyear"]
        ax2.plot(summary_df["year"], turnover, color="tab:red", marker="s", label="1 - Jaccard")
        ax2.set_ylabel("Theme turnover (1 - Jaccard)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        mark_shocks(ax1)
        ax1.set_xlabel("Year")
        ax1.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(outdir / "dual_axis_js_and_1minusjaccard_tags.png", dpi=150)
        plt.close(fig)

    # Shock window bars
    win_df = build_shock_window(summary_df, shock_years, window)
    if not win_df.empty:
        for y in shock_years:
            sub = win_df[(win_df["window_id"].str.startswith(f"{y}-")) & (~win_df["window_id"].str.endswith("mean"))]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(sub["year"].astype(int), sub["js_divergence"], color="tab:blue")
            ax.set_xlabel("Year")
            ax.set_ylabel("JS divergence")
            ax.set_title(f"Shock window JS divergence around {y} (±{window})")
            ax.grid(True, linestyle="--", alpha=0.5)
            fig.tight_layout()
            fig.savefig(outdir / f"shock_window_{y}_js.png", dpi=150)
            plt.close(fig)


# ----------------------------
# Outputs
# ----------------------------
def write_outputs(merged: pd.DataFrame, yearly_tags: pd.DataFrame, summary: pd.DataFrame, shock_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    merged_out = outdir / "mpst_tmdb_merged.parquet"
    try:
        merged.to_parquet(merged_out, index=False)
    except Exception:
        merged_out = outdir / "mpst_tmdb_merged.csv"
        merged.to_csv(merged_out, index=False)
    yearly_tags.to_csv(outdir / "yearly_top_tags.csv", index=False)
    summary.to_csv(outdir / "yearly_summary_tags.csv", index=False)
    shock_df.to_csv(outdir / "shock_window_summary.csv", index=False)


# ----------------------------
# Console summary
# ----------------------------
def console_summary(merged: pd.DataFrame, summary: pd.DataFrame, shock_years: List[int], shock_df: pd.DataFrame) -> None:
    total_mpst = len(merged)
    with_year = merged["year"].notna().sum()
    matched = merged["mpst_hit"].notna().sum()
    hits = merged[merged["mpst_hit"] == 1]
    rest = merged[merged["mpst_hit"] == 0]
    print("Coverage:")
    print(f"  MPST rows: {total_mpst}")
    print(f"  Matched with TMDb hit flag: {matched}")
    print(f"  With year: {with_year}")
    print(f"  Hits: {len(hits)}, Rest: {len(rest)}")
    for y in shock_years:
        row = summary[summary["year"] == y]
        prev_row = summary[summary["year"] == y - 1]
        next_row = summary[summary["year"] == y + 1]
        js = row.iloc[0]["js_divergence"] if not row.empty else float("nan")
        jac = row.iloc[0]["jaccard_top50_prevyear"] if not row.empty else float("nan")
        print(f"Shock year {y}: JS={js}, Jaccard={jac}")
        if not prev_row.empty:
            print(f"  Prev {y-1}: JS={prev_row.iloc[0]['js_divergence']}, Jaccard={prev_row.iloc[0]['jaccard_top50_prevyear']}")
        if not next_row.empty:
            print(f"  Next {y+1}: JS={next_row.iloc[0]['js_divergence']}, Jaccard={next_row.iloc[0]['jaccard_top50_prevyear']}")
    if not shock_df.empty:
        print("Shock window means:")
        means = shock_df[shock_df["window_id"].str.endswith("mean")]
        for _, r in means.iterrows():
            print(f"  {r['window_id']}: mean JS={r['js_divergence']:.4f}, mean Jaccard={r['jaccard']:.4f}")


# ----------------------------
# Main flow
# ----------------------------
def main() -> None:
    args = parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    outdir = Path(args.outdir)
    shock_years = [int(s) for s in args.shock_years.split(",") if s.strip()]

    # Load TMDb
    tmdb_info = auto_file_path(args.tmdb_slug, args.tmdb_file)
    # Load TMDb without strict usecols to avoid missing-column errors; we'll subset after load.
    tmdb_df = load_kaggle_df(tmdb_info, pandas_kwargs={"dtype": "string"})
    # Keep only relevant columns that actually exist
    tmdb_keep = [c for c in IMDB_CANDIDATES + YEAR_CANDIDATES + ["vote_count", "popularity", "revenue"] if c in tmdb_df.columns]
    tmdb_df = tmdb_df[tmdb_keep]
    metric_col = detect_metric_column(tmdb_df.columns, args.metric)
    print(f"Detected TMDb metric column: {metric_col}")
    tmdb_df[metric_col] = pd.to_numeric(tmdb_df[metric_col], errors="coerce")
    tmdb_hits = compute_tmdb_hits(tmdb_df, metric_col, args)

    # Load MPST
    mpst_info = auto_file_path(args.mpst_slug, args.mpst_file)
    mpst_df = load_kaggle_df(mpst_info, pandas_kwargs={"dtype": "string"})
    imdb_col_mpst = detect_imdb_column(mpst_df)
    tag_col = detect_tag_column(mpst_df)
    print(f"Detected MPST imdb column: {imdb_col_mpst}, tag column: {tag_col}")
    mpst_df["imdb_norm"], invalid_mpst = normalize_imdb(mpst_df[imdb_col_mpst])
    mpst_df["tags_parsed"] = mpst_df[tag_col].apply(parse_tags)

    # Merge
    merged = mpst_df.merge(tmdb_hits, on="imdb_norm", how="left", suffixes=("", "_tmdb"))
    merged["mpst_has_year"] = merged["year"].notna()
    merged["mpst_hit"] = merged["hit_tmdb"]
    merged = merged[(merged["year"] >= args.year_min) & (merged["year"] <= args.year_max)]
    merged_with_year = merged.dropna(subset=["year"]).copy()
    merged_with_year["year"] = merged_with_year["year"].astype(int)

    yearly_tags, summary_df = per_year_tag_stats(merged_with_year, args.top_tags_k)
    shock_df = build_shock_window(summary_df, shock_years, args.shock_window)

    write_outputs(merged_with_year, yearly_tags, summary_df, shock_df, outdir)
    plot_outputs(summary_df, shock_years, args.shock_window, outdir)

    console_summary(merged_with_year, summary_df, shock_years, shock_df)

    overall_scores = yearly_tags.groupby("tag")["score_logodds"].mean().sort_values(ascending=False).head(10)
    print("Top tags by average overrepresentation:")
    for tag, val in overall_scores.items():
        print(f"  {tag}: {val:.4f}")
    print(f"Outputs written to {outdir}")


if __name__ == "__main__":
    main()
