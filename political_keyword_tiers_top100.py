"""
Political keyword intensity comparisons between success tiers (Top20 vs 21–100) with normalization and genre robustness.
"""
from __future__ import annotations

import argparse
import ast
import heapq
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed patterns
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
BAD_TOKENS = {"<na>", "nan", "none", "null", ""}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Political keyword tiers (Top20 vs 21-100) with robustness checks.")
    p.add_argument("--tmdb-csv", required=True)
    p.add_argument("--outdir", default="./outputs_tiers_top100")
    p.add_argument("--year-min", type=int, default=1970)
    p.add_argument("--year-max", type=int, default=2023)
    p.add_argument("--metric", default="revenue", choices=["revenue", "vote_count", "popularity"])
    p.add_argument("--filter-adult", action="store_true", default=True)
    p.add_argument("--runtime-min", type=float, default=40)
    p.add_argument("--min-vote-count", type=float, default=50)
    p.add_argument("--chunksize", type=int, default=200_000)
    p.add_argument("--shock-years", default="2001,2008")
    p.add_argument("--max-rank", type=int, default=100)
    p.add_argument("--top20", type=int, default=20)
    p.add_argument("--top50", type=int, default=50)
    p.add_argument("--genre-bucket-mode", choices=["action_vs_nonaction", "actionwarthriller_vs_other"], default="action_vs_nonaction")
    return p.parse_args()


# ------------- helpers -------------
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
    return series.str.lower().isin(["true", "t", "1", "yes"]) if series.notna().any() else pd.Series(False, index=series.index)


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


def detect_columns(columns: List[str], metric_pref: str) -> Tuple[str, str, Optional[str], Optional[str], str, Optional[str]]:
    kw_col = next((c for c in KW_COLS if c in columns), None)
    if kw_col is None:
        raise ValueError(f"No keyword column found. Available: {columns}")
    genre_col = next((c for c in GENRE_COLS if c in columns), None)
    id_col = next((c for c in ID_COLS if c in columns), None)
    if id_col is None:
        raise ValueError(f"No ID column found. Available: {columns}")
    year_col = "release_year" if "release_year" in columns else ("year" if "year" in columns else None)
    date_col = "release_date" if "release_date" in columns else None
    order = {
        "revenue": ["revenue", "vote_count", "popularity"],
        "vote_count": ["vote_count", "popularity", "revenue"],
        "popularity": ["popularity", "vote_count", "revenue"],
    }[metric_pref]
    metric_col = next((c for c in order if c in columns), None)
    if metric_col is None:
        raise ValueError(f"No metric column found. Available: {columns}")
    return kw_col, id_col, year_col, date_col, metric_col, genre_col


def classify_political(kw: str) -> Tuple[bool, Optional[str]]:
    primary = None
    for g in ["war_security_intel", "economy_finance_crisis", "institutions_elections_law", "migration_police_civilrights", "labor_collective_action", "inequality_corruption_elites"]:
        if POLITICAL_PATTERNS[g].search(kw):
            primary = g
            break
    if primary:
        return True, primary
    for _, pat in POLITICAL_PATTERNS.items():
        if pat.search(kw):
            return True, None
    return False, None


# ------------- Pass 1 -------------
def build_rank_lists_pass1(csv_path: Path, cols: List[str], id_col: str, year_col: Optional[str], date_col: Optional[str], metric_col: str, args: argparse.Namespace) -> pd.DataFrame:
    usecols = [c for c in [id_col, year_col, date_col, metric_col, "adult", "runtime", "vote_count"] if c and c in cols]
    heaps = defaultdict(list)
    reader = pd.read_csv(csv_path, chunksize=args.chunksize, usecols=usecols, dtype="string", low_memory=False)
    for chunk in reader:
        if "adult" in chunk.columns and args.filter_adult:
            adult_bool = normalize_adult(chunk["adult"])
            chunk = chunk[~adult_bool]
        if "runtime" in chunk.columns and args.runtime_min > 0:
            rt = pd.to_numeric(chunk["runtime"], errors="coerce")
            chunk = chunk[rt >= args.runtime_min]
        if "vote_count" in chunk.columns and args.min_vote_count > 0:
            vc = pd.to_numeric(chunk["vote_count"], errors="coerce")
            chunk = chunk[vc >= args.min_vote_count]
        metric_vals = pd.to_numeric(chunk[metric_col], errors="coerce")
        if metric_col == "revenue":
            metric_vals = metric_vals.where(metric_vals > 0)
        chunk = chunk.assign(metric_val=metric_vals)
        chunk = chunk[metric_vals.notna()]
        for _, row in chunk.iterrows():
            year = extract_year(row, year_col, date_col)
            if year is None or year < args.year_min or year > args.year_max:
                continue
            metric = row["metric_val"]
            imdb = str(row[id_col]).strip()
            if not imdb:
                continue
            heap = heaps[year]
            if len(heap) < args.max_rank:
                heapq.heappush(heap, (metric, imdb))
            else:
                if metric > heap[0][0]:
                    heapq.heapreplace(heap, (metric, imdb))
    rows = []
    for year, heap in heaps.items():
        ranked = heapq.nlargest(len(heap), heap)
        ranked.sort(key=lambda x: x[0], reverse=True)
        for idx, (metric, imdb) in enumerate(ranked, start=1):
            rows.append({"year": year, "rank": idx, "id": imdb, "metric_value": metric})
    return pd.DataFrame(rows)


# ------------- Pass 2 -------------
def aggregate_pass2(
    csv_path: Path,
    cols: List[str],
    kw_col: str,
    genre_col: Optional[str],
    id_col: str,
    year_col: Optional[str],
    date_col: Optional[str],
    rank_map: Dict[int, Dict[str, int]],
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Counter], Dict[str, Counter], Dict[str, Counter], Dict[str, Counter]]:
    usecols = [c for c in [kw_col, genre_col, id_col, year_col, date_col, "adult", "runtime", "vote_count", args.metric] if c and c in cols]
    tier_sum_kw = {"top20": Counter(), "21_100": Counter(), "top50": Counter(), "51_100": Counter()}
    tier_sum_any = {"top20": Counter(), "21_100": Counter(), "top50": Counter(), "51_100": Counter()}
    tier_sum_polshare = {"top20": Counter(), "21_100": Counter(), "top50": Counter(), "51_100": Counter()}
    tier_sum_totalkw = {"top20": Counter(), "21_100": Counter(), "top50": Counter(), "51_100": Counter()}
    tier_sum_metric = {"top20": Counter(), "21_100": Counter(), "top50": Counter(), "51_100": Counter()}
    tier_metric_vals = {"top20": defaultdict(list), "21_100": defaultdict(list), "top50": defaultdict(list), "51_100": defaultdict(list)}
    tier_n = {"top20": Counter(), "21_100": Counter(), "top50": Counter(), "51_100": Counter()}
    # Genre buckets
    bucket_sum_kw = {"top20": Counter(), "21_100": Counter()}
    bucket_sum_polshare = {"top20": Counter(), "21_100": Counter()}
    bucket_sum_totalkw = {"top20": Counter(), "21_100": Counter()}
    bucket_n = {"top20": Counter(), "21_100": Counter()}
    # Pooled genres
    pooled_genre_counts = {"top20": Counter(), "21_100": Counter()}

    reader = pd.read_csv(csv_path, chunksize=args.chunksize, usecols=usecols, dtype="string", low_memory=False)
    for chunk in reader:
        if "adult" in chunk.columns and args.filter_adult:
            adult_bool = normalize_adult(chunk["adult"])
            chunk = chunk[~adult_bool]
        if "runtime" in chunk.columns and args.runtime_min > 0:
            rt = pd.to_numeric(chunk["runtime"], errors="coerce")
            chunk = chunk[rt >= args.runtime_min]
        if "vote_count" in chunk.columns and args.min_vote_count > 0:
            vc = pd.to_numeric(chunk["vote_count"], errors="coerce")
            chunk = chunk[vc >= args.min_vote_count]
        if args.metric in chunk.columns:
            mv = pd.to_numeric(chunk[args.metric], errors="coerce")
            if args.metric == "revenue":
                mv = mv.where(mv > 0)
            chunk = chunk.assign(metric_val=mv)
        else:
            chunk = chunk.assign(metric_val=np.nan)

        for _, row in chunk.iterrows():
            year = extract_year(row, year_col, date_col)
            if year is None or year < args.year_min or year > args.year_max:
                continue
            rmap = rank_map.get(year)
            if not rmap:
                continue
            movie_id = str(row[id_col]).strip()
            rank = rmap.get(movie_id)
            if rank is None or rank > args.max_rank:
                continue
            kws = parse_keywords(row.get(kw_col))
            pol_kws = []
            for kw in set(kws):
                is_pol, _ = classify_political(kw)
                if is_pol:
                    pol_kws.append(kw)
            pol_count = len(pol_kws)
            total_kw = len(set(kws))
            pol_share = pol_count / total_kw if total_kw else math.nan
            pol_any = 1 if pol_count > 0 else 0
            metric_val = row.get("metric_val")
            metric_num = float(metric_val) if pd.notna(metric_val) else math.nan
            # genres
            genres = parse_genres(row.get(genre_col)) if genre_col else []
            genres_set = set(genres)
            # pooled genres
            target_pooled = "top20" if rank <= args.top20 else "21_100"
            for g in genres_set:
                pooled_genre_counts[target_pooled][g] += 1

            # bucket
            bucket = None
            if args.genre_bucket_mode == "action_vs_nonaction":
                bucket = "action" if "action" in genres_set else "non_action"
            else:
                bucket = "awt" if any(x in genres_set for x in ["action", "war", "thriller"]) else "other"

            def update(tier_key):
                tier_sum_kw[tier_key][year] += pol_count
                tier_sum_any[tier_key][year] += pol_any
                if not math.isnan(pol_share):
                    tier_sum_polshare[tier_key][year] += pol_share
                tier_sum_totalkw[tier_key][year] += total_kw
                if not math.isnan(metric_num):
                    tier_sum_metric[tier_key][year] += metric_num
                    tier_metric_vals[tier_key][year].append(metric_num)
                tier_n[tier_key][year] += 1
                if bucket:
                    if tier_key in ["top20", "21_100"]:
                        bucket_sum_kw[tier_key][(year, bucket)] += pol_count
                        if not math.isnan(pol_share):
                            bucket_sum_polshare[tier_key][(year, bucket)] += pol_share
                        bucket_sum_totalkw[tier_key][(year, bucket)] += total_kw
                        bucket_n[tier_key][(year, bucket)] += 1

            if rank <= args.top20:
                update("top20")
            elif rank <= 100:
                update("21_100")
            if rank <= args.top50:
                update("top50")
            elif rank <= 100:
                update("51_100")

    # Build yearly summary
    years = sorted(set(tier_n["top20"].keys()) | set(tier_n["21_100"].keys()))
    rows = []
    for y in years:
        def mean_safe(sum_val, n_val):
            return sum_val[y] / n_val if n_val else 0

        n20 = tier_n["top20"][y]
        n80 = tier_n["21_100"][y]
        n50 = tier_n["top50"][y]
        n51_100 = tier_n["51_100"][y]
        rows.append(
            {
                "year": y,
                "n_top20": n20,
                "n_21_100": n80,
                "mean_polkw_top20": mean_safe(tier_sum_kw["top20"], n20),
                "mean_polkw_21_100": mean_safe(tier_sum_kw["21_100"], n80),
                "gap_polkw": mean_safe(tier_sum_kw["top20"], n20) - mean_safe(tier_sum_kw["21_100"], n80),
                "share_any_top20": mean_safe(tier_sum_any["top20"], n20),
                "share_any_21_100": mean_safe(tier_sum_any["21_100"], n80),
                "gap_any": mean_safe(tier_sum_any["top20"], n20) - mean_safe(tier_sum_any["21_100"], n80),
                "mean_polshare_top20": mean_safe(tier_sum_polshare["top20"], n20),
                "mean_polshare_21_100": mean_safe(tier_sum_polshare["21_100"], n80),
                "gap_polshare": mean_safe(tier_sum_polshare["top20"], n20) - mean_safe(tier_sum_polshare["21_100"], n80),
                "mean_totalkw_top20": mean_safe(tier_sum_totalkw["top20"], n20),
                "mean_totalkw_21_100": mean_safe(tier_sum_totalkw["21_100"], n80),
                "gap_totalkw": mean_safe(tier_sum_totalkw["top20"], n20) - mean_safe(tier_sum_totalkw["21_100"], n80),
                "mean_metric_top20": mean_safe(tier_sum_metric["top20"], n20),
                "mean_metric_21_100": mean_safe(tier_sum_metric["21_100"], n80),
                "median_metric_top20": np.median(tier_metric_vals["top20"][y]) if tier_metric_vals["top20"][y] else np.nan,
                "median_metric_21_100": np.median(tier_metric_vals["21_100"][y]) if tier_metric_vals["21_100"][y] else np.nan,
                "n_top50": n50,
                "n_51_100": n51_100,
                "mean_polkw_top50": mean_safe(tier_sum_kw["top50"], n50),
                "mean_polkw_51_100": mean_safe(tier_sum_kw["51_100"], n51_100),
                "gap_polkw_50": mean_safe(tier_sum_kw["top50"], n50) - mean_safe(tier_sum_kw["51_100"], n51_100),
                "share_any_top50": mean_safe(tier_sum_any["top50"], n50),
                "share_any_51_100": mean_safe(tier_sum_any["51_100"], n51_100),
            }
        )
    tier_df = pd.DataFrame(rows).sort_values("year")

    # genre buckets
    bucket_rows = []
    for (year, bucket), _ in bucket_sum_kw["top20"].items():
        n20b = bucket_n["top20"][(year, bucket)]
        n80b = bucket_n["21_100"][(year, bucket)]
        bucket_rows.append(
            {
                "year": year,
                "bucket": bucket,
                "mean_polkw_top20": bucket_sum_kw["top20"][(year, bucket)] / n20b if n20b else 0,
                "mean_polkw_21_100": bucket_sum_kw["21_100"][(year, bucket)] / n80b if n80b else 0,
                "gap_polkw": (bucket_sum_kw["top20"][(year, bucket)] / n20b if n20b else 0)
                - (bucket_sum_kw["21_100"][(year, bucket)] / n80b if n80b else 0),
                "mean_polshare_top20": bucket_sum_polshare["top20"][(year, bucket)] / n20b if n20b else 0,
                "mean_polshare_21_100": bucket_sum_polshare["21_100"][(year, bucket)] / n80b if n80b else 0,
                "gap_polshare": (bucket_sum_polshare["top20"][(year, bucket)] / n20b if n20b else 0)
                - (bucket_sum_polshare["21_100"][(year, bucket)] / n80b if n80b else 0),
                "mean_totalkw_top20": bucket_sum_totalkw["top20"][(year, bucket)] / n20b if n20b else 0,
                "mean_totalkw_21_100": bucket_sum_totalkw["21_100"][(year, bucket)] / n80b if n80b else 0,
            }
        )
    bucket_df = pd.DataFrame(bucket_rows).sort_values(["year", "bucket"])

    # pooled genres
    pooled_rows = []
    for tier in ["top20", "21_100"]:
        total_movies = sum(tier_n[tier].values())
        total_counts = sum(pooled_genre_counts[tier].values())
        if total_counts == 0:
            continue
        top_genres = pooled_genre_counts[tier].most_common()
        for g, c in top_genres:
            pooled_rows.append({"tier": tier, "genre": g, "count": c, "share": c / total_counts})
    pooled_df = pd.DataFrame(pooled_rows).sort_values(["tier", "share"], ascending=[True, False])
    return tier_df, bucket_df, pooled_df


# ------------- Plots -------------
def mark_shocks(ax, shock_years: List[int]):
    for y in shock_years:
        ax.axvline(y, color="gray", linestyle="--", alpha=0.6)
        ax.text(y, ax.get_ylim()[1], str(y), rotation=90, va="bottom", ha="right", fontsize=8, color="gray")


def make_genre_plots(pooled_df: pd.DataFrame, outdir: Path):
    if pooled_df.empty:
        return
    top_genres = pooled_df.groupby("genre")["share"].sum().sort_values(ascending=False).head(15).index.tolist()
    plot_df = pooled_df[pooled_df["genre"].isin(top_genres)]
    tiers = plot_df["tier"].unique()
    fig, ax = plt.subplots(figsize=(10, 6))
    idx = np.arange(len(top_genres))
    width = 0.35
    for i, tier in enumerate(sorted(tiers)):
        shares = [plot_df[(plot_df["tier"] == tier) & (plot_df["genre"] == g)]["share"].sum() for g in top_genres]
        ax.bar(idx + i * width, shares, width, label=tier)
    ax.set_xticks(idx + width / 2)
    ax.set_xticklabels(top_genres, rotation=45, ha="right")
    ax.set_ylabel("Share of genre counts")
    ax.set_title("Genre composition: Top20 vs 21-100 (pooled)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "genre_composition_top20_vs_21_100.png", dpi=150)
    plt.close(fig)


def make_timeseries_plots(tier_df: pd.DataFrame, bucket_df: pd.DataFrame, shock_years: List[int], outdir: Path):
    if tier_df.empty:
        return
    plot_specs = [
        (["mean_polkw_top20", "mean_polkw_21_100"], "political_kw_intensity_top20_vs_21_100.png", "Mean political keywords per film"),
        (["gap_polkw"], "political_kw_gap_top20_minus_21_100.png", "Gap: Top20 - 21-100 (political keywords)"),
        (["mean_polshare_top20", "mean_polshare_21_100"], "political_share_keywords_top20_vs_21_100.png", "Political share of keywords"),
        (["gap_polshare"], "political_share_keywords_gap_top20_minus_21_100.png", "Gap: Top20 - 21-100 (political share)"),
        (["mean_totalkw_top20", "mean_totalkw_21_100"], "tagging_volume_total_keywords_top20_vs_21_100.png", "Total keywords per film"),
        (["mean_metric_top20", "mean_metric_21_100"], "metric_top20_vs_21_100_over_time.png", "Mean metric"),
        (["median_metric_top20", "median_metric_21_100"], "metric_median_top20_vs_21_100_over_time.png", "Median metric"),
    ]
    for cols, fname, title in plot_specs:
        fig, ax = plt.subplots(figsize=(10, 5))
        for c in cols:
            ax.plot(tier_df["year"], tier_df[c], label=c)
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.5)
        mark_shocks(ax, shock_years)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=150)
        plt.close(fig)

    if not bucket_df.empty:
        buckets = bucket_df["bucket"].unique()
        fig, ax = plt.subplots(figsize=(10, 5))
        for b in buckets:
            sub = bucket_df[bucket_df["bucket"] == b]
            ax.plot(sub["year"], sub["gap_polkw"], label=b)
        ax.set_title("Genre-bucket gap (Top20 - 21-100)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Gap (mean political keywords)")
        ax.grid(True, linestyle="--", alpha=0.5)
        mark_shocks(ax, shock_years)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "genre_bucket_gap_polkw.png", dpi=150)
        plt.close(fig)


# ------------- Main -------------
def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    shock_years = [int(s) for s in args.shock_years.split(",") if s.strip()]

    head = pd.read_csv(args.tmdb_csv, nrows=0)
    cols = head.columns.tolist()
    kw_col, id_col, year_col, date_col, metric_col, genre_col = detect_columns(cols, args.metric)
    print(f"Detected columns: keywords={kw_col}, genres={genre_col}, id={id_col}, year={year_col or date_col}, metric={metric_col}")

    # Pass 1
    hits_df = build_rank_lists_pass1(Path(args.tmdb_csv), cols, id_col, year_col, date_col, metric_col, args)
    hits_df.to_csv(outdir / "hits_rank_lists_top100.csv", index=False)
    rank_map: Dict[int, Dict[str, int]] = defaultdict(dict)
    for _, r in hits_df.iterrows():
        rank_map[int(r["year"])][str(r["id"])] = int(r["rank"])

    # Pass 2
    tier_df, bucket_df, pooled_df = aggregate_pass2(
        Path(args.tmdb_csv), cols, kw_col, genre_col, id_col, year_col, date_col, rank_map, args
    )
    tier_df.to_csv(outdir / "yearly_tier_summary_top100.csv", index=False)
    bucket_df.to_csv(outdir / "yearly_genre_bucket_summary_top100.csv", index=False)
    pooled_df.to_csv(outdir / "pooled_genre_shares_top20_vs_21_100.csv", index=False)

    # Plots
    make_genre_plots(pooled_df, outdir)
    make_timeseries_plots(tier_df, bucket_df, shock_years, outdir)

    # Console summary
    full_years = tier_df[(tier_df["n_top20"] >= args.top20) & (tier_df["n_21_100"] >= (args.max_rank - args.top20))].shape[0]
    print(f"Years with full Top20 and 21-100: {full_years}/{len(tier_df)}")
    print(f"Avg gaps: polkw={tier_df['gap_polkw'].mean():.3f}, polshare={tier_df['gap_polshare'].mean():.3f}, totalkw={tier_df['gap_totalkw'].mean():.3f}")
    for y in shock_years:
        row = tier_df[tier_df["year"] == y]
        if not row.empty:
            print(f"Shock {y}: gap_polkw={row.iloc[0]['gap_polkw']:.3f}, gap_polshare={row.iloc[0]['gap_polshare']:.3f}")
            prev = tier_df[tier_df["year"] == y - 1]
            nxt = tier_df[tier_df["year"] == y + 1]
            if not prev.empty:
                print(f"  Prev {y-1}: gap_polkw={prev.iloc[0]['gap_polkw']:.3f}")
            if not nxt.empty:
                print(f"  Next {y+1}: gap_polkw={nxt.iloc[0]['gap_polkw']:.3f}")


if __name__ == "__main__":
    main()
