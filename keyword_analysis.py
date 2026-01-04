"""
CLI tool to compare yearly keyword distributions between top movies and the rest.

Features:
- Downloads a Kaggle TMDb movies dataset (assumes kaggle.json configured).
- Chunked two-pass processing to stay memory-safe.
- Robust keyword parsing across multiple formats.
- Per-year log-odds overrepresentation scores, JS divergence, and Jaccard rotation.
"""
from __future__ import annotations

import argparse
import ast
import heapq
import json
import math
import os
import random
import sys
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_DATASET = "asaniczka/tmdb-movies-dataset-2023-930k-movies"
DEFAULT_MIN_YEAR = 1970
DEFAULT_MAX_YEAR = 2024
DEFAULT_TOP_N = 20


# ----------------------------
# Argument parsing and setup
# ----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare keyword distributions for top movies vs the rest by year."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Kaggle dataset slug")
    parser.add_argument("--download-dir", default="./data", help="Where to download/unzip the dataset")
    parser.add_argument("--csv", default=None, help="CSV filename inside the dataset (auto-detect if omitted)")
    parser.add_argument("--year-min", type=int, default=DEFAULT_MIN_YEAR, help="Minimum release year")
    parser.add_argument("--year-max", type=int, default=DEFAULT_MAX_YEAR, help="Maximum release year")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Number of top movies per year")
    parser.add_argument(
        "--success-metric",
        choices=["revenue", "vote_count", "popularity"],
        default="revenue",
        help="Metric used to rank top movies (with fallbacks if missing)",
    )
    parser.add_argument("--min-vote-count", type=float, default=50, help="Minimum vote_count to include a movie (if column exists)")
    parser.add_argument("--min-keywords", type=int, default=1, help="Minimum parsed keywords required")
    parser.add_argument("--min-yearly-kw-count", type=int, default=5, help="Frequency floor: require keyword to appear at least this many times in a year (top+rest)")
    parser.add_argument("--parse-failure-sample", type=int, default=20, help="Sample size of raw keyword strings where parsing failed")
    parser.add_argument("--shock-years", default="2001,2008", help="Comma-separated years to highlight in report")
    parser.add_argument("--outdir", default="./outputs", help="Output directory for results")
    parser.add_argument("--chunksize", type=int, default=200_000, help="Rows per chunk for processing")
    return parser.parse_args()


# ----------------------------
# Kaggle dataset handling
# ----------------------------
def check_kaggle_credentials() -> bool:
    """Return True if Kaggle credentials are available and valid."""
    try:
        from kaggle import api  # type: ignore
    except ImportError:
        print("Error: kaggle package not installed. Install via `pip install kaggle`.", file=sys.stderr)
        return False

    kaggle_dir = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_dir.exists():
        print(f"Error: Kaggle credentials not found at {kaggle_dir}.", file=sys.stderr)
        print("Create the file with your API token from https://www.kaggle.com/settings/account", file=sys.stderr)
        return False

    try:
        api.authenticate()
        return True
    except Exception as exc:  # pragma: no cover - depends on env
        print(f"Error: Kaggle authentication failed: {exc}", file=sys.stderr)
        return False


def download_dataset(dataset_slug: str, download_dir: Path) -> None:
    """Download and unzip Kaggle dataset if not already present."""
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checking dataset in {download_dir} ...")

    # If any CSV already exists in download_dir, skip download.
    existing_csvs = list(download_dir.rglob("*.csv"))
    if existing_csvs:
        print(f"Found existing CSVs in {download_dir}, skipping download.")
        return

    if not check_kaggle_credentials():
        sys.exit(1)

    try:
        from kaggle import api  # type: ignore
    except ImportError:  # pragma: no cover
        print("kaggle package missing after credential check.", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading dataset {dataset_slug} ...")
    try:
        api.dataset_download_files(dataset_slug, path=str(download_dir), unzip=True)
    except Exception as exc:  # pragma: no cover - depends on env
        print(f"Dataset download failed: {exc}", file=sys.stderr)
        sys.exit(1)


def find_csv_file(download_dir: Path, csv_arg: Optional[str]) -> Path:
    """Locate the CSV file to read."""
    if csv_arg:
        csv_path = (download_dir / csv_arg) if not os.path.isabs(csv_arg) else Path(csv_arg)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file {csv_path} not found.")
        return csv_path

    candidates = sorted(download_dir.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV files found under {download_dir}.")

    # Choose the largest CSV assuming it's the main dataset.
    best = max(candidates, key=lambda p: p.stat().st_size)
    print(f"Auto-detected CSV: {best}")
    return best


# ----------------------------
# Column detection and helpers
# ----------------------------
@dataclass
class ColumnConfig:
    release_col: str
    success_col: str
    keyword_col: str
    vote_count_col: Optional[str]


def detect_columns(columns: Sequence[str], requested_metric: str) -> ColumnConfig:
    release_candidates = ["release_year", "release_date", "year"]
    keyword_candidates = ["keywords", "Keywords", "keyword_names", "tmdb_keywords"]

    release_col = next((c for c in release_candidates if c in columns), None)
    keyword_col = next((c for c in keyword_candidates if c in columns), None)
    vote_col = "vote_count" if "vote_count" in columns else None

    if requested_metric == "revenue":
        metric_order = ["revenue", "vote_count", "popularity"]
    elif requested_metric == "vote_count":
        metric_order = ["vote_count", "popularity", "revenue"]
    else:
        metric_order = ["popularity", "vote_count", "revenue"]
    success_col = next((c for c in metric_order if c in columns), None)

    missing = []
    if not release_col:
        missing.append("release date/year")
    if not keyword_col:
        missing.append("keywords")
    if not success_col:
        missing.append(f"success metric (tried: {', '.join(metric_order)})")

    if missing:
        raise ValueError(
            f"Required columns missing: {missing}. Available columns: {', '.join(columns)}"
        )

    return ColumnConfig(
        release_col=release_col,
        success_col=success_col,
        keyword_col=keyword_col,
        vote_count_col=vote_col,
    )


def read_columns(csv_path: Path) -> List[str]:
    df_head = pd.read_csv(csv_path, nrows=0)
    return df_head.columns.tolist()


def build_read_kwargs(config: ColumnConfig) -> dict:
    usecols = {config.release_col, config.success_col, config.keyword_col}
    if config.vote_count_col:
        usecols.add(config.vote_count_col)
    dtype = {
        config.keyword_col: "string",
        config.release_col: "string",
    }
    converters = {}
    for col in [config.success_col, config.vote_count_col]:
        if col:
            converters[col] = to_float
    return {"usecols": list(usecols), "dtype": dtype, "converters": converters, "low_memory": False}


# ----------------------------
# Parsing utilities
# ----------------------------
def normalize_keyword(token: str) -> str:
    token = token.strip().lower()
    token = "_".join(token.split())
    return token


def parse_keywords_field(value) -> Tuple[List[str], bool]:
    """Return (keywords, parse_ok)."""
    if value is None:
        return [], True
    try:
        if pd.isna(value):
            return [], True
    except Exception:
        pass
    if isinstance(value, float) and math.isnan(value):
        return [], True

    if isinstance(value, list):
        tokens = []
        for item in value:
            if isinstance(item, dict) and "name" in item:
                tokens.append(str(item["name"]))
            else:
                tokens.append(str(item))
        return _clean_tokens(tokens)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return [], True
        # JSON list of dicts
        for loader in (json.loads, ast.literal_eval):
            try:
                loaded = loader(text)
                if isinstance(loaded, list):
                    tokens = []
                    for item in loaded:
                        if isinstance(item, dict) and "name" in item:
                            tokens.append(str(item["name"]))
                        else:
                            tokens.append(str(item))
                    return _clean_tokens(tokens)
            except Exception:
                pass

        # Delimited strings
        if "|" in text:
            tokens = text.split("|")
            return _clean_tokens(tokens)
        if "," in text:
            tokens = text.split(",")
            return _clean_tokens(tokens)
        # Single token fallback
        return _clean_tokens([text])

    # Unknown type
    return [], False


def _clean_tokens(tokens: Iterable[str], parse_ok: bool = True) -> Tuple[List[str], bool]:
    cleaned: List[str] = []
    for tok in tokens:
        if tok is None:
            continue
        norm = normalize_keyword(str(tok))
        if norm:
            cleaned.append(norm)
    return cleaned, parse_ok


def extract_year(val: str) -> Optional[int]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    try:
        # Already a year number as string
        year_int = int(str(val)[:4])
        if year_int <= 0:
            return None
        return year_int
    except Exception:
        return None


def to_float(val) -> float:
    try:
        return float(val)
    except Exception:
        return float("nan")


# ----------------------------
# Core analysis
# ----------------------------
def determine_top_ids(
    csv_path: Path,
    config: ColumnConfig,
    args: argparse.Namespace,
    read_kwargs: dict,
) -> Dict[int, List[int]]:
    """First pass: find top-N row ids per year according to metric."""
    top_heaps: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
    rows_seen = 0
    reader = pd.read_csv(csv_path, chunksize=args.chunksize, **read_kwargs)
    col_idx = None

    for chunk_idx, chunk in enumerate(reader, start=1):
        if col_idx is None:
            col_idx = {c: i for i, c in enumerate(chunk.columns)}
        values = chunk.to_numpy()
        for offset, row in enumerate(values):
            row_id = rows_seen + offset
            year = extract_year(row[col_idx[config.release_col]])
            if year is None or year < args.year_min or year > args.year_max:
                continue

            keywords, parse_ok = parse_keywords_field(row[col_idx[config.keyword_col]])
            if len(keywords) < args.min_keywords:
                continue

            if config.vote_count_col:
                vote_val = to_float(row[col_idx[config.vote_count_col]])
                if math.isnan(vote_val) or vote_val < args.min_vote_count:
                    continue

            metric_val = to_float(row[col_idx[config.success_col]])
            if math.isnan(metric_val):
                continue

            heap = top_heaps[year]
            if len(heap) < args.top_n:
                heapq.heappush(heap, (metric_val, row_id))
            else:
                if metric_val > heap[0][0]:
                    heapq.heapreplace(heap, (metric_val, row_id))

        rows_seen += len(chunk)
        if chunk_idx % 5 == 0:
            print(f"[pass1] Processed {rows_seen:,} rows")

    top_ids_per_year: Dict[int, List[int]] = {}
    for year, heap in top_heaps.items():
        # Keep highest metrics
        top_rows = heapq.nlargest(args.top_n, heap)
        top_ids_per_year[year] = [row_id for _, row_id in top_rows]
    print(f"[pass1] Identified top rows for {len(top_ids_per_year)} years.")
    return top_ids_per_year


def aggregate_keywords(
    csv_path: Path,
    config: ColumnConfig,
    args: argparse.Namespace,
    read_kwargs: dict,
    top_ids_per_year: Dict[int, List[int]],
) -> Tuple[dict, List[Tuple[int, str]]]:
    """Second pass: count keywords for top vs rest, compute summaries."""
    top_sets = {year: set(ids) for year, ids in top_ids_per_year.items()}
    stats = {
        year: {
            "top_counter": Counter(),
            "rest_counter": Counter(),
            "n_top_movies": 0,
            "n_rest_movies": 0,
            "parse_failures": 0,
        }
        for year in range(args.year_min, args.year_max + 1)
    }
    failure_samples: List[Tuple[int, str]] = []
    failures_seen = 0
    sample_size = max(0, args.parse_failure_sample)

    rows_seen = 0
    reader = pd.read_csv(csv_path, chunksize=args.chunksize, **read_kwargs)
    col_idx = None

    for chunk_idx, chunk in enumerate(reader, start=1):
        if col_idx is None:
            col_idx = {c: i for i, c in enumerate(chunk.columns)}
        values = chunk.to_numpy()
        for offset, row in enumerate(values):
            row_id = rows_seen + offset
            year = extract_year(row[col_idx[config.release_col]])
            if year is None or year < args.year_min or year > args.year_max:
                continue

            raw_keywords_field = row[col_idx[config.keyword_col]]
            keywords, parse_ok = parse_keywords_field(raw_keywords_field)
            if not parse_ok:
                stats[year]["parse_failures"] += 1
                failures_seen += 1
                if sample_size > 0:
                    if len(failure_samples) < sample_size:
                        failure_samples.append((year, str(raw_keywords_field)))
                    else:
                        idx = random.randint(0, failures_seen - 1)
                        if idx < sample_size:
                            failure_samples[idx] = (year, str(raw_keywords_field))
            if len(keywords) < args.min_keywords:
                continue

            unique_keywords = set(keywords)
            if len(unique_keywords) < args.min_keywords:
                continue

            if config.vote_count_col:
                vote_val = to_float(row[col_idx[config.vote_count_col]])
                if math.isnan(vote_val) or vote_val < args.min_vote_count:
                    continue

            metric_val = to_float(row[col_idx[config.success_col]])
            if math.isnan(metric_val):
                continue

            group = "top_counter" if row_id in top_sets.get(year, set()) else "rest_counter"
            if group == "top_counter":
                stats[year]["n_top_movies"] += 1
            else:
                stats[year]["n_rest_movies"] += 1
            stats[year][group].update(unique_keywords)

        rows_seen += len(chunk)
        if chunk_idx % 5 == 0:
            print(f"[pass2] Processed {rows_seen:,} rows")

    return stats, failure_samples


def compute_log_odds(a_w: int, A: int, b_w: int, B: int, alpha: float = 0.5) -> float:
    return math.log((a_w + alpha) / (A - a_w + alpha)) - math.log((b_w + alpha) / (B - b_w + alpha))


def js_divergence(counter_a: Counter, counter_b: Counter) -> float:
    vocab = set(counter_a) | set(counter_b)
    if not vocab:
        return float("nan")
    a = np.array([counter_a.get(k, 0) for k in vocab], dtype=float)
    b = np.array([counter_b.get(k, 0) for k in vocab], dtype=float)
    sum_a, sum_b = a.sum(), b.sum()
    if sum_a == 0 or sum_b == 0:
        return float("nan")
    p = a / sum_a
    q = b / sum_b
    m = 0.5 * (p + q)
    # Avoid log(0) with small epsilon
    eps = 1e-12
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    m = np.clip(m, eps, 1)
    js = 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
    return float(js)


def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return float("nan")
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return float("nan")
    return inter / union


def build_outputs(stats: dict, args: argparse.Namespace, outdir: Path):
    yearly_rows = []
    summary_rows = []
    previous_top50: Optional[set] = None

    shock_years = []
    for part in args.shock_years.split(","):
        part = part.strip()
        if part:
            try:
                shock_years.append(int(part))
            except ValueError:
                pass

    js_values = []
    jaccard_values = []

    for year in range(args.year_min, args.year_max + 1):
        data = stats.get(year)
        if not data:
            continue
        top_counter = data["top_counter"]
        rest_counter = data["rest_counter"]
        total_top = sum(top_counter.values())
        total_rest = sum(rest_counter.values())

        # Compute log-odds and take top 30 keywords
        vocab = set(top_counter) | set(rest_counter)
        keyword_scores = []
        for kw in vocab:
            ct_top = top_counter.get(kw, 0)
            ct_rest = rest_counter.get(kw, 0)
            overall = ct_top + ct_rest
            if overall < args.min_yearly_kw_count:
                continue
            if total_top == 0 or total_rest == 0:
                score = float("nan")
                p_top = p_rest = delta = float("nan")
            else:
                score = compute_log_odds(ct_top, total_top, ct_rest, total_rest, alpha=0.5)
                p_top = ct_top / total_top
                p_rest = ct_rest / total_rest
                delta = p_top - p_rest
            keyword_scores.append(
                (score, kw, ct_top, ct_rest, p_top, p_rest, delta)
            )

        keyword_scores.sort(key=lambda x: (math.isnan(x[0]), -x[0] if not math.isnan(x[0]) else float("-inf")))
        top_keywords = keyword_scores[:30]
        for score, kw, ct_top, ct_rest, p_top, p_rest, delta in top_keywords:
            yearly_rows.append(
                {
                    "year": year,
                    "keyword": kw,
                    "score_logodds": score,
                    "count_top": ct_top,
                    "count_rest": ct_rest,
                    "p_top": p_top,
                    "p_rest": p_rest,
                    "delta": delta,
                }
            )

        js = js_divergence(top_counter, rest_counter)
        js_values.append((year, js))

        current_top50 = {kw for _, kw, *_ in keyword_scores[:50]}
        jaccard = jaccard_similarity(current_top50, previous_top50 or set()) if previous_top50 is not None else float("nan")
        jaccard_values.append((year, jaccard))
        previous_top50 = current_top50 if current_top50 else previous_top50

        summary_rows.append(
            {
                "year": year,
                "n_top": data["n_top_movies"],
                "n_rest": data["n_rest_movies"],
                "js_divergence": js,
                "jaccard_top50_prevyear": jaccard,
                "parse_failures": data["parse_failures"],
            }
        )

    # Save CSVs
    yearly_df = pd.DataFrame(yearly_rows)
    summary_df = pd.DataFrame(summary_rows)
    yearly_df.to_csv(outdir / "yearly_top_keywords.csv", index=False)
    summary_df.to_csv(outdir / "yearly_summary.csv", index=False)

    # Shock year report
    write_shock_report(outdir, summary_df, yearly_df, shock_years)

    # Optional plots
    try:
        import matplotlib.pyplot as plt  # type: ignore

        if not summary_df.empty:
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(summary_df["year"], summary_df["js_divergence"], marker="o")
            ax1.set_title("JS divergence over time")
            ax1.set_xlabel("Year")
            ax1.set_ylabel("JS divergence")
            fig1.tight_layout()
            fig1.savefig(outdir / "js_divergence.png", dpi=150)
            plt.close(fig1)

            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(summary_df["year"], summary_df["jaccard_top50_prevyear"], marker="o", color="orange")
            ax2.set_title("Jaccard of Top 50 keywords vs previous year")
            ax2.set_xlabel("Year")
            ax2.set_ylabel("Jaccard similarity")
            fig2.tight_layout()
            fig2.savefig(outdir / "jaccard_rotation.png", dpi=150)
            plt.close(fig2)
    except ImportError:
        print("matplotlib not installed; skipping plots.")


def write_shock_report(outdir: Path, summary_df: pd.DataFrame, yearly_df: pd.DataFrame, shock_years: List[int]) -> None:
    lines = []
    lines.append("Shock Year Diagnostics\n")

    if not summary_df.empty:
        top_js = summary_df.dropna(subset=["js_divergence"]).nlargest(5, "js_divergence")
        lines.append("Highest JS divergence years:")
        for _, row in top_js.iterrows():
            lines.append(f"  {int(row['year'])}: JS={row['js_divergence']:.4f}, top={int(row['n_top'])}, rest={int(row['n_rest'])}")
        lines.append("")

        low_jaccard = summary_df.dropna(subset=["jaccard_top50_prevyear"]).nsmallest(5, "jaccard_top50_prevyear")
        lines.append("Lowest Jaccard similarity to previous year (keyword rotation):")
        for _, row in low_jaccard.iterrows():
            lines.append(f"  {int(row['year'])}: Jaccard={row['jaccard_top50_prevyear']:.4f}")
        lines.append("")

    for y in shock_years:
        lines.append(f"Shock year {y}:")
        row = summary_df[summary_df["year"] == y]
        prev_row = summary_df[summary_df["year"] == y - 1]
        next_row = summary_df[summary_df["year"] == y + 1]
        if row.empty:
            lines.append("  No data.")
            continue
        js = row.iloc[0]["js_divergence"]
        jaccard = row.iloc[0]["jaccard_top50_prevyear"]
        lines.append(f"  JS divergence: {js:.4f} (prev-year jaccard={jaccard})")
        if not prev_row.empty:
            lines.append(f"  Prev year ({y-1}) JS: {prev_row.iloc[0]['js_divergence']:.4f}")
        if not next_row.empty:
            lines.append(f"  Next year ({y+1}) JS: {next_row.iloc[0]['js_divergence']:.4f}")
        # Show a few top keywords
        top_kw = yearly_df[yearly_df["year"] == y].nlargest(5, "score_logodds")
        if not top_kw.empty:
            kws = ", ".join(top_kw["keyword"].tolist())
            lines.append(f"  Top keywords: {kws}")
        lines.append("")

    report_path = outdir / "shock_year_report.txt"
    report_path.write_text("\n".join(lines))
    print(f"Wrote shock year report to {report_path}")


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    download_dir = Path(args.download_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    download_dataset(args.dataset, download_dir)
    csv_path = find_csv_file(download_dir, args.csv)

    columns = read_columns(csv_path)
    config = detect_columns(columns, args.success_metric)
    print(
        f"Using columns: year={config.release_col}, keywords={config.keyword_col}, "
        f"metric={config.success_col}, vote_count={config.vote_count_col}"
    )

    read_kwargs = build_read_kwargs(config)

    # Pass 1: find top row ids
    import heapq  # local import to avoid polluting namespace earlier

    top_ids_per_year = determine_top_ids(csv_path, config, args, read_kwargs)

    # Pass 2: aggregate keyword counts
    stats, failure_samples = aggregate_keywords(csv_path, config, args, read_kwargs, top_ids_per_year)

    # Outputs
    build_outputs(stats, args, outdir)
    total_failures = sum(data["parse_failures"] for data in stats.values())
    if total_failures:
        print(f"Parse failures observed: {total_failures}. Random sample (up to {len(failure_samples)}):")
        for year, raw in failure_samples:
            print(f"  year {year}: {raw}")
    else:
        print("No parse failures encountered.")
    print(f"Finished. Outputs written to {outdir}")


if __name__ == "__main__":
    main()
