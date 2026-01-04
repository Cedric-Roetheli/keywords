"""
Political keyword tier analysis using US domestic box office rankings (Top20 vs 21–100).

Inputs:
- TMDb metadata CSV (local)
- Mojo domestic box office CSV (local, with 'domestic' column)
Outputs: tier summaries, genre buckets, plots, and merge diagnostics.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from rapidfuzz import process, fuzz

    HAVE_RF = True
except ImportError:
    import difflib

    HAVE_RF = False

POLITICAL_PATTERNS = {
    "institutions_elections_law": re.compile(
        r"(^|_)(government|parliament|congress|senate|president|prime_minister|minister|election|vote|voting|campaign|politic(s|al)?|party|constitution|democracy|dictator(ship)?|regime|state|bureaucracy|public_policy|law|legal|court|judge|trial|supreme_court)($|_)",
        re.IGNORECASE,
    ),
    "war_security_intel": re.compile(
        r"(^|_)(war|world_war_(i|ii)|cold_war|army|military|soldier(s)?|veteran(s)?|battle|combat|invasion|occupation|terror(ism|ist)?|insurgent(s|cy)?|guerrilla|hostage|spy|espionage|cia|fbi|kgb|intelligence|surveillance|wiretap|secret_service|nuclear|missile(s)?|chemical_weapon(s)?|bioweapon(s)?)($|_)",
        re.IGNORECASE,
    ),
    "economy_finance_crisis": re.compile(
        r"(^|_)(econom(y|ic|ics)?|finance|financial|bank(s|ing)?|wall_street|stock_market|hedge_fund(s)?|invest(ment|or|ing)?|debt|credit|loan(s)?|mortgage(s)?|foreclosure|recession|depression|crisis|inflation|unemployment|austerity|bailout|budget|tax(es|ation)?|privatiz(e|ation)|nationaliz(e|ation))($|_)",
        re.IGNORECASE,
    ),
    "labor_collective_action": re.compile(
        r"(^|_)(labor|labour|worker(s)?|working_class|union(s)?|strike|protest|demonstration|riot(s)?|revolution|uprising|insurrection|coup|general_strike|activism|activist(s)?)($|_)",
        re.IGNORECASE,
    ),
    "inequality_corruption_elites": re.compile(
        r"(^|_)(corruption|bribe(ry)?|embezzle(ment)?|scandal|fraud|money_launder(ing)?|oligarch(s|y)?|inequal(ity|ities)|poverty|class_warfare|social_class|the_rich|billionaire(s)?|capitalis(m|t)|communis(m|t)|socialis(m|t)|corporate|corporation(s)?|big_business|monopoly|greed)($|_)",
        re.IGNORECASE,
    ),
    "migration_police_civilrights": re.compile(
        r"(^|_)(immigra(tion|nt|nts)|migrant(s)?|refugee(s)?|asylum|border|deport(ation)?|citizenship|civil_rights|human_rights|discriminat(e|ion)|racis(m|t)|apartheid|segregat(e|ion)|police|policing|cop(s)?|law_enforcement|prison|incarcerat(e|ion)|surveillance_state)($|_)",
        re.IGNORECASE,
    ),
}

KW_COLS = ["keywords", "keyword_names", "tmdb_keywords", "Keywords", "keyword"]
GENRE_COLS = ["genres", "genre_names", "Genres"]
ID_COLS = ["imdb_id", "id", "tmdb_id"]
TITLE_COLS = ["title", "original_title", "primary_title"]
BAD_TOKENS = {"<na>", "nan", "none", "null", ""}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Political keyword tiers using US domestic box office rankings.")
    p.add_argument("--tmdb-csv", required=True)
    p.add_argument("--mojo-csv", required=True)
    p.add_argument("--outdir", default="./outputs_us_market_mojo")
    p.add_argument("--year-min", type=int, default=1985)
    p.add_argument("--year-max", type=int, default=2023)
    p.add_argument("--top20", type=int, default=20)
    p.add_argument("--max-rank", type=int, default=100)
    p.add_argument("--filter-adult", action="store_true", default=True)
    p.add_argument("--runtime-min", type=float, default=40)
    p.add_argument("--min-vote-count", type=float, default=50)
    p.add_argument("--fuzzy-title-match", action="store_true", default=True)
    p.add_argument("--fuzzy-threshold", type=int, default=92)
    p.add_argument("--genre-bucket-mode", choices=["action_vs_nonaction", "actionwarthriller_vs_other"], default="action_vs_nonaction")
    p.add_argument("--shock-years", default="2001,2008")
    return p.parse_args()


def clean_money(val) -> float:
    if pd.isna(val):
        return math.nan
    s = str(val)
    s = s.replace("$", "").replace(",", "").strip()
    if not s:
        return math.nan
    try:
        return float(s)
    except Exception:
        return math.nan


def normalize_token(tok: str) -> str:
    tok = tok.strip().lower()
    tok = "_".join(tok.split())
    return tok


def parse_list_field(val) -> List[str]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    if isinstance(val, list):
        tokens = val
    else:
        text = str(val).strip()
        if not text:
            return []
        tokens = None
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(text)
                if isinstance(parsed, list):
                    tokens = []
                    for item in parsed:
                        if isinstance(item, dict) and "name" in item:
                            tokens.append(str(item["name"]))
                        else:
                            tokens.append(str(item))
                    break
            except Exception:
                tokens = None
        if tokens is None:
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
        nt = normalize_token(str(t))
        if nt and nt not in BAD_TOKENS:
            cleaned.append(nt)
    return cleaned


def parse_keywords(val) -> List[str]:
    return parse_list_field(val)


def parse_genres(val) -> List[str]:
    return parse_list_field(val)


def normalize_adult(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower()
    return s.isin(["true", "t", "1", "yes"])


def extract_year(row: pd.Series, year_col: Optional[str], date_col: Optional[str]) -> Optional[int]:
    if year_col and pd.notna(row.get(year_col)):
        try:
            return int(str(row[year_col])[:4])
        except Exception:
            pass
    if date_col and pd.notna(row.get(date_col)):
        m = re.match(r"(\d{4})", str(row[date_col]))
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


def detect_columns_tmdb(columns: List[str]) -> Tuple[str, str, Optional[str], Optional[str], str, Optional[str]]:
    kw_col = next((c for c in KW_COLS if c in columns), None)
    if kw_col is None:
        raise ValueError(f"No keyword column found. Available: {columns}")
    genre_col = next((c for c in GENRE_COLS if c in columns), None)
    id_col = next((c for c in ID_COLS if c in columns), None)
    if id_col is None:
        raise ValueError(f"No ID column found. Available: {columns}")
    title_col = next((c for c in TITLE_COLS if c in columns), None)
    year_col = "release_year" if "release_year" in columns else ("year" if "year" in columns else None)
    date_col = "release_date" if "release_date" in columns else None
    return kw_col, id_col, year_col, date_col, title_col, genre_col


def detect_columns_mojo(columns: List[str]) -> Tuple[str, Optional[str], str]:
    title_col = "title" if "title" in columns else next((c for c in columns if "title" in c.lower()), None)
    year_col = "year" if "year" in columns else ("release_year" if "release_year" in columns else None)
    if year_col is None and "release_date" in columns:
        year_col = "release_date"
    if title_col is None:
        raise ValueError("Mojo file missing title column")
    if "domestic" not in columns:
        raise ValueError("Mojo file missing 'domestic' column")
    imdb_col = "imdb_id" if "imdb_id" in columns else None
    return title_col, year_col, imdb_col


def normalize_title(title: str) -> str:
    if title is None:
        return ""
    s = str(title).lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = "_".join(s.split())
    s = s.replace("_the_", "_").replace("_a_", "_")
    return s


def classify_political(kw: str) -> bool:
    for pat in POLITICAL_PATTERNS.values():
        if pat.search(kw):
            return True
    return False


def classify_primary_group(kw: str) -> Optional[str]:
    for g in ["war_security_intel", "economy_finance_crisis", "institutions_elections_law", "migration_police_civilrights", "labor_collective_action", "inequality_corruption_elites"]:
        if POLITICAL_PATTERNS[g].search(kw):
            return g
    for name, pat in POLITICAL_PATTERNS.items():
        if pat.search(kw):
            return name
    return None


def build_rankings(mojo_path: Path, args: argparse.Namespace) -> pd.DataFrame:
    mojo = pd.read_csv(mojo_path)
    title_col, year_col, imdb_col = detect_columns_mojo(mojo.columns.tolist())
    mojo["domestic_clean"] = mojo["domestic"].apply(clean_money)
    if year_col == "release_date":
        mojo["year_clean"] = mojo["release_date"].astype(str).str.extract(r"(\d{4})")[0]
    else:
        mojo["year_clean"] = mojo[year_col]
    mojo["year_clean"] = pd.to_numeric(mojo["year_clean"], errors="coerce").astype("Int64")
    mojo = mojo[(mojo["year_clean"] >= args.year_min) & (mojo["year_clean"] <= args.year_max)]
    rows = []
    for year, g in mojo.groupby("year_clean"):
        g = g.dropna(subset=["domestic_clean"])
        g = g.sort_values("domestic_clean", ascending=False).head(args.max_rank)
        for idx, r in enumerate(g.itertuples(), start=1):
            rows.append(
                {
                    "year": int(year),
                    "rank": idx,
                    "mojo_title": getattr(r, title_col),
                    "title_norm": normalize_title(getattr(r, title_col)),
                    "domestic": r.domestic_clean,
                    "imdb_id": getattr(r, imdb_col) if imdb_col else None,
                    "source_row": r.Index,
                }
            )
    return pd.DataFrame(rows)


def fuzzy_match_by_year(rank_df: pd.DataFrame, tmdb_year_map: Dict[int, pd.DataFrame], threshold: int, use_fuzzy: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    matched_rows = []
    diagnostics = []
    for year, g in rank_df.groupby("year"):
        tmdb_year_df = tmdb_year_map.get(year)
        if tmdb_year_df is None or tmdb_year_df.empty:
            diagnostics.append({"year": year, "match_rate": 0, "note": "no tmdb rows"})
            continue
        tmdb_titles = tmdb_year_df["title_norm"].tolist()
        tmdb_lookup = dict(zip(tmdb_year_df["title_norm"], tmdb_year_df["tmdb_id"]))
        used_tmdb = set()
        matched = 0
        for row in g.itertuples():
            best_id = None
            best_score = -1
            if use_fuzzy and HAVE_RF:
                res = process.extract(normalize_title(row.mojo_title), tmdb_titles, scorer=fuzz.token_set_ratio, limit=3)
                if res:
                    cand, score, _ = res[0]
                    if score >= threshold:
                        best_score = score
                        best_id = tmdb_lookup[cand]
            else:
                # simple difflib fallback
                matches = difflib.get_close_matches(normalize_title(row.mojo_title), tmdb_titles, n=3, cutoff=threshold / 100)
                if matches:
                    cand = matches[0]
                    best_score = 100  # proxy
                    best_id = tmdb_lookup[cand]
            if best_id and best_id not in used_tmdb:
                matched += 1
                used_tmdb.add(best_id)
                matched_rows.append(
                    {
                        "year": year,
                        "rank": row.rank,
                        "domestic": row.domestic,
                        "mojo_title": row.mojo_title,
                        "tmdb_id": best_id,
                    }
                )
        rate = matched / len(g) if len(g) else 0
        diagnostics.append({"year": year, "match_rate": rate, "note": ""})
    return pd.DataFrame(matched_rows), pd.DataFrame(diagnostics)


def merge_rankings(rank_df: pd.DataFrame, tmdb_df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Try imdb_id merge first if available
    if "imdb_id" in tmdb_df.columns and rank_df["imdb_id"].notna().any():
        merged = rank_df.dropna(subset=["imdb_id"]).merge(tmdb_df, on="imdb_id", how="left", suffixes=("", "_tmdb"))
        matched = merged[~merged["tmdb_id"].isna()].copy()
        # unmatched with imdb try fuzzy
        remaining = rank_df[rank_df["imdb_id"].isna() | rank_df["imdb_id"].str.len().eq(0)]
    else:
        matched = pd.DataFrame()
        remaining = rank_df

    # Fuzzy for remaining
    tmdb_year_map = {y: g for y, g in tmdb_df.groupby("year")}
    fuzzy_matches, diag = fuzzy_match_by_year(remaining, tmdb_year_map, args.fuzzy_threshold, args.fuzzy_title_match)
    if not fuzzy_matches.empty:
        matched_fuzzy = fuzzy_matches.merge(tmdb_df, on=["year", "tmdb_id"], how="left")
        all_matched = pd.concat([matched, matched_fuzzy], ignore_index=True)
    else:
        all_matched = matched
    return all_matched, diag


def compute_movie_metrics(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        kws = parse_keywords(getattr(r, "keywords"))
        kw_set = set(kws)
        total_kw = len(kw_set)
        pol_set = set([kw for kw in kw_set if classify_political(kw)])
        pol_count = len(pol_set)
        pol_share = pol_count / total_kw if total_kw else math.nan
        genres = set(parse_genres(getattr(r, "genres"))) if "genres" in df.columns else set()
        if args.genre_bucket_mode == "action_vs_nonaction":
            bucket = "action" if "action" in genres else "non_action"
        else:
            bucket = "awt" if any(g in genres for g in ["action", "war", "thriller"]) else "other"
        rows.append(
            {
                "year": r.year,
                "rank": r.rank,
                "mojo_title": r.mojo_title,
                "tmdb_title": getattr(r, "title"),
                "domestic": r.domestic,
                "pol_count": pol_count,
                "pol_share": pol_share,
                "total_kw": total_kw,
                "any_pol": 1 if pol_count > 0 else 0,
                "bucket": bucket,
            }
        )
    return pd.DataFrame(rows)


def summarize_yearly(metrics_df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    bucket_rows = []
    for year, g in metrics_df.groupby("year"):
        top20 = g[g["rank"] <= args.top20]
        rest = g[(g["rank"] > args.top20) & (g["rank"] <= args.max_rank)]
        n20, nrest = len(top20), len(rest)
        rows.append(
            {
                "year": year,
                "n_top20": n20,
                "n_21_100": nrest,
                "mean_polkw_top20": top20["pol_count"].mean(),
                "mean_polkw_21_100": rest["pol_count"].mean(),
                "gap_polkw": top20["pol_count"].mean() - rest["pol_count"].mean(),
                "mean_polshare_top20": top20["pol_share"].mean(),
                "mean_polshare_21_100": rest["pol_share"].mean(),
                "gap_polshare": top20["pol_share"].mean() - rest["pol_share"].mean(),
                "mean_totalkw_top20": top20["total_kw"].mean(),
                "mean_totalkw_21_100": rest["total_kw"].mean(),
                "gap_totalkw": top20["total_kw"].mean() - rest["total_kw"].mean(),
                "share_any_top20": top20["any_pol"].mean(),
                "share_any_21_100": rest["any_pol"].mean(),
                "mean_domestic_top20": top20["domestic"].mean(),
                "mean_domestic_21_100": rest["domestic"].mean(),
                "median_domestic_top20": top20["domestic"].median(),
                "median_domestic_21_100": rest["domestic"].median(),
                "coverage_kw_top20": (top20["total_kw"] > 0).mean(),
                "coverage_kw_21_100": (rest["total_kw"] > 0).mean(),
            }
        )
        for bucket, gb in g.groupby("bucket"):
            tb = gb[gb["rank"] <= args.top20]
            rb = gb[(gb["rank"] > args.top20) & (gb["rank"] <= args.max_rank)]
            bucket_rows.append(
                {
                    "year": year,
                    "bucket": bucket,
                    "mean_polshare_top20": tb["pol_share"].mean(),
                    "mean_polshare_21_100": rb["pol_share"].mean(),
                    "gap_polshare": tb["pol_share"].mean() - rb["pol_share"].mean(),
                    "mean_polkw_top20": tb["pol_count"].mean(),
                    "mean_polkw_21_100": rb["pol_count"].mean(),
                    "gap_polkw": tb["pol_count"].mean() - rb["pol_count"].mean(),
                    "mean_totalkw_top20": tb["total_kw"].mean(),
                    "mean_totalkw_21_100": rb["total_kw"].mean(),
                }
            )
    return pd.DataFrame(rows).sort_values("year"), pd.DataFrame(bucket_rows).sort_values(["year", "bucket"])


def pooled_genre_shares(metrics_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    pooled = []
    for tier, df in [("top20", metrics_df[metrics_df["rank"] <= args.top20]), ("21_100", metrics_df[(metrics_df["rank"] > args.top20) & (metrics_df["rank"] <= args.max_rank)])]:
        counter = Counter()
        for genres in metrics_df.loc[df.index, "bucket"]:
            counter[genres] += 1
        total = sum(counter.values())
        for g, c in counter.items():
            pooled.append({"tier": tier, "genre": g, "share": c / total if total else 0, "count": c})
    return pd.DataFrame(pooled)


def make_plots(tier_df: pd.DataFrame, bucket_df: pd.DataFrame, pooled_df: pd.DataFrame, shock_years: List[int], outdir: Path):
    def mark(ax):
        for y in shock_years:
            ax.axvline(y, color="gray", linestyle="--", alpha=0.6)

    if not pooled_df.empty:
        top_genres = pooled_df.groupby("genre")["share"].sum().sort_values(ascending=False).head(15).index.tolist()
        fig, ax = plt.subplots(figsize=(8, 5))
        idx = np.arange(len(top_genres))
        width = 0.35
        for i, tier in enumerate(sorted(pooled_df["tier"].unique())):
            shares = [pooled_df[(pooled_df["tier"] == tier) & (pooled_df["genre"] == g)]["share"].sum() for g in top_genres]
            ax.bar(idx + i * width, shares, width, label=tier)
        ax.set_xticks(idx + width / 2)
        ax.set_xticklabels(top_genres, rotation=45, ha="right")
        ax.set_title("Genre composition: Top20 vs 21-100 (pooled)")
        ax.set_ylabel("Share")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "genre_composition_top20_vs_21_100_us.png", dpi=150)
        plt.close(fig)

    if not tier_df.empty:
        plot_specs = [
            (["mean_domestic_top20", "mean_domestic_21_100"], "domestic_top20_vs_21_100_over_time.png", "Mean domestic gross"),
            (["mean_polshare_top20", "mean_polshare_21_100"], "political_share_top20_vs_21_100_over_time.png", "Political share"),
            (["gap_polshare"], "political_share_gap_over_time.png", "Gap in political share"),
            (["mean_polkw_top20", "mean_polkw_21_100"], "political_kw_top20_vs_21_100_over_time.png", "Political keywords per film"),
            (["gap_polkw"], "political_kw_gap_over_time.png", "Gap in political keywords"),
            (["mean_totalkw_top20", "mean_totalkw_21_100"], "tagging_volume_top20_vs_21_100_over_time.png", "Total keywords per film"),
        ]
        for cols, fname, title in plot_specs:
            fig, ax = plt.subplots(figsize=(10, 5))
            for c in cols:
                ax.plot(tier_df["year"], tier_df[c], label=c)
            ax.set_title(title)
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")
            ax.grid(True, linestyle="--", alpha=0.5)
            mark(ax)
            ax.legend()
            fig.tight_layout()
            fig.savefig(outdir / fname, dpi=150)
            plt.close(fig)

    if not bucket_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for b in bucket_df["bucket"].unique():
            sub = bucket_df[bucket_df["bucket"] == b]
            ax.plot(sub["year"], sub["gap_polshare"], label=b)
        ax.set_title("Genre bucket gap (political share)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Gap")
        ax.grid(True, linestyle="--", alpha=0.5)
        mark(ax)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "genre_bucket_gap_polshare.png", dpi=150)
        plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    shock_years = [int(s) for s in args.shock_years.split(",") if s.strip()]

    # Load TMDb filtered universe
    tmdb = pd.read_csv(args.tmdb_csv, usecols=lambda c: True)
    kw_col, id_col, year_col, date_col, title_col, genre_col = detect_columns_tmdb(tmdb.columns.tolist())
    tmdb["year"] = tmdb.apply(lambda r: extract_year(r, year_col, date_col), axis=1)
    tmdb = tmdb[(tmdb["year"] >= args.year_min) & (tmdb["year"] <= args.year_max)]
    if "adult" in tmdb.columns and args.filter_adult:
        tmdb = tmdb[~normalize_adult(tmdb["adult"])]
    if "runtime" in tmdb.columns and args.runtime_min > 0:
        tmdb = tmdb[pd.to_numeric(tmdb["runtime"], errors="coerce") >= args.runtime_min]
    if "vote_count" in tmdb.columns and args.min_vote_count > 0:
        tmdb = tmdb[pd.to_numeric(tmdb["vote_count"], errors="coerce") >= args.min_vote_count]
    tmdb["title_norm"] = tmdb[title_col].astype(str).apply(normalize_title)
    tmdb = tmdb.rename(columns={id_col: "tmdb_id"})

    # Build rankings from Mojo
    ranks_df = build_rankings(Path(args.mojo_csv), args)
    ranks_df.to_csv(outdir / "us_domestic_rankings_top100.csv", index=False)

    # Merge
    matched_df, diag_df = merge_rankings(ranks_df, tmdb[["tmdb_id", "year", "title_norm", "title", kw_col, genre_col]], args)
    diag_df.to_csv(outdir / "merge_diagnostics.csv", index=False)

    # Compute metrics
    metrics_df = matched_df.rename(columns={kw_col: "keywords", genre_col: "genres"})
    metrics_df = compute_movie_metrics(metrics_df, args)

    # Sample
    sample_cols = ["year", "rank", "mojo_title", "tmdb_title", "domestic", "pol_count", "pol_share", "total_kw", "bucket"]
    metrics_df.head(200).to_csv(outdir / "merged_ranked_movies_sample.csv", index=False, columns=sample_cols)

    # Summaries
    tier_df, bucket_df = summarize_yearly(metrics_df, args)
    tier_df.to_csv(outdir / "yearly_tier_summary_us_mojo.csv", index=False)
    bucket_df.to_csv(outdir / "yearly_genre_bucket_summary_us_mojo.csv", index=False)

    # Pooled genre shares (using buckets as stand-in for genres)
    pooled_df = metrics_df.copy()
    pooled_df = pooled_df[["rank", "bucket"]]
    pooled_df["tier"] = np.where(pooled_df["rank"] <= args.top20, "top20", "21_100")
    pooled_counts = pooled_df.groupby(["tier", "bucket"]).size().reset_index(name="count")
    total_counts = pooled_counts.groupby("tier")["count"].transform("sum")
    pooled_counts["share"] = pooled_counts["count"] / total_counts
    pooled_counts.rename(columns={"bucket": "genre"}).to_csv(outdir / "pooled_genre_shares_top20_vs_21_100.csv", index=False)

    # Plots
    make_plots(tier_df, bucket_df, pooled_counts.rename(columns={"bucket": "genre"}), shock_years, outdir)

    # Console summary
    match_rate = len(metrics_df) / len(ranks_df) if len(ranks_df) else 0
    per_year_match = metrics_df.groupby("year").size() / ranks_df.groupby("year").size()
    print(f"Overall match rate: {match_rate:.3f}; per-year range: {per_year_match.min():.3f}-{per_year_match.max():.3f}")
    full_years = tier_df[(tier_df["n_top20"] >= args.top20) & (tier_df["n_21_100"] >= (args.max_rank - args.top20))].shape[0]
    print(f"Years with full tiers: {full_years}/{len(tier_df)}")
    print(f"Avg gaps: polshare={tier_df['gap_polshare'].mean():.3f}, polkw={tier_df['gap_polkw'].mean():.3f}, totalkw={tier_df['gap_totalkw'].mean():.3f}")
    pre = tier_df[tier_df["year"] < 2000]
    post = tier_df[tier_df["year"] >= 2000]
    if not pre.empty and not post.empty:
        print(f"Pre2000 gap_polshare mean: {pre['gap_polshare'].mean():.3f}, Post2000: {post['gap_polshare'].mean():.3f}")


if __name__ == "__main__":
    main()
