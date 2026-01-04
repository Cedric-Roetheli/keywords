"""
Political keyword time series for TMDb:
- Build political keyword list from candidate_keywords.csv via fixed regex.
- Two-pass TMDb processing to identify hits (top-N or percentile by metric) and then count political keywords for all vs hits.
- Normalize by yearly movie counts and within-political totals; aggregate to groups; plot trends and hit-vs-rest gaps.
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
from typing import Dict, Iterable, List, Optional, Tuple

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

GROUP_PRIORITY = [
    "economy_finance_crisis",
    "war_security_intel",
    "institutions_elections_law",
    "migration_police_civilrights",
    "labor_collective_action",
    "inequality_corruption_elites",
]

BAD_TOKENS = {"<na>", "nan", "none", "null", ""}
KEYWORD_COLUMNS = ["keywords", "keyword_names", "Keywords", "tmdb_keywords", "keyword"]
IMDB_COLUMNS = ["imdb_id", "imdbId", "imdbid", "imdb"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Political keyword normalized time series with hits vs rest.")
    parser.add_argument("--tmdb-csv", required=True)
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--outdir", default="./outputs_political_keywords")
    parser.add_argument("--year-min", type=int, default=1970)
    parser.add_argument("--year-max", type=int, default=2023)
    parser.add_argument("--top-x", type=int, default=15)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--filter-adult", action="store_true", default=True)
    parser.add_argument("--min-vote-count", type=float, default=0)
    parser.add_argument("--runtime-min", type=float, default=0)
    parser.add_argument("--use-movie-incidence", action="store_true", default=True)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--group-mode", choices=["primary", "multi"], default="primary")
    parser.add_argument("--shock-years", default="2001,2008")
    parser.add_argument("--shock-window", type=int, default=1)
    parser.add_argument("--hits-mode", choices=["topn", "percentile"], default="topn")
    parser.add_argument("--hits-n", type=int, default=20)
    parser.add_argument("--hits-percentile", type=float, default=0.95)
    parser.add_argument("--metric", choices=["revenue", "vote_count", "popularity"], default="revenue")
    parser.add_argument("--min-metric", type=float, default=0)
    return parser.parse_args()


# ----------------------------
# Candidate and political set
# ----------------------------
def normalize_token(tok: str) -> str:
    tok = tok.strip().lower()
    tok = "_".join(tok.split())
    return tok


def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["keyword"] = df["keyword"].astype(str).str.strip()
    df = df[~df["keyword"].str.lower().isin(BAD_TOKENS)]
    if "count" in df.columns:
        df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    else:
        df["count"] = 0
    if "cum_share" not in df.columns:
        df = df.sort_values("count", ascending=False)
        df["cum_count"] = df["count"].cumsum()
        total = df["count"].sum()
        df["cum_share"] = df["cum_count"] / total if total else 0
    return df


def build_political_dict(df: pd.DataFrame, coverage: float, min_count: int) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
    df = df[df["count"] >= min_count].copy()
    df = df.sort_values("count", ascending=False)
    df = df[df["cum_share"] <= coverage]
    rows = []
    mapping = {}
    for _, r in df.iterrows():
        kw = normalize_token(str(r["keyword"]))
        matched = []
        for name, pat in POLITICAL_PATTERNS.items():
            if pat.search(kw):
                matched.append(name)
        if not matched:
            continue
        primary = None
        for g in GROUP_PRIORITY:
            if g in matched:
                primary = g
                break
        rows.append(
            {
                "keyword": kw,
                "count": r["count"],
                "matched_groups": ";".join(matched),
                "primary_group": primary,
            }
        )
        mapping[kw] = {"matched_groups": matched, "primary_group": primary}
    pol_df = pd.DataFrame(rows).sort_values("count", ascending=False)
    return pol_df, mapping


# ----------------------------
# Parsing utilities
# ----------------------------
def parse_keywords(val) -> List[str]:
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


def extract_year(row: pd.Series, year_col: Optional[str], date_col: Optional[str]) -> Optional[int]:
    if year_col and pd.notna(row.get(year_col)):
        try:
            y = int(str(row[year_col])[:4])
            return y
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


def normalize_adult(series: pd.Series) -> pd.Series:
    return series.str.lower().isin(["true", "t", "1", "yes"]) if series.notna().any() else pd.Series(False, index=series.index)


def detect_columns(columns: List[str]) -> Tuple[str, Optional[str], Optional[str], str]:
    kw_col = None
    for cand in KEYWORD_COLUMNS:
        if cand in columns:
            kw_col = cand
            break
    if kw_col is None:
        raise ValueError(f"No keyword column found. Available columns: {columns}")
    imdb_col = None
    for cand in IMDB_COLUMNS:
        if cand in columns:
            imdb_col = cand
            break
    if imdb_col is None:
        raise ValueError(f"No imdb_id column found. Available columns: {columns}")
    year_col = "release_year" if "release_year" in columns else ("year" if "year" in columns else None)
    date_col = "release_date" if "release_date" in columns else None
    return kw_col, year_col, date_col, imdb_col


def detect_metric_column(columns: List[str], preferred: str) -> str:
    order_map = {
        "revenue": ["revenue", "vote_count", "popularity"],
        "vote_count": ["vote_count", "popularity", "revenue"],
        "popularity": ["popularity", "vote_count", "revenue"],
    }
    for col in order_map.get(preferred, []):
        if col in columns:
            return col
    raise ValueError(f"No metric column found. Available: {columns}")


# ----------------------------
# Pass 1: find hits
# ----------------------------
def find_hits(
    csv_path: Path,
    metric_col: str,
    imdb_col: str,
    year_col: Optional[str],
    date_col: Optional[str],
    args: argparse.Namespace,
    available_cols: List[str],
) -> Dict[int, Dict[str, float]]:
    usecols = [c for c in [metric_col, imdb_col, "release_year", "release_date", "year", "adult", "vote_count", "runtime"] if c in available_cols]
    heaps = defaultdict(list)  # year -> minheap of (metric, imdb)
    metrics_store = defaultdict(list)  # for percentile
    hits_per_year: Dict[int, Dict[str, float]] = defaultdict(dict)
    reader = pd.read_csv(csv_path, chunksize=args.chunksize, usecols=usecols, dtype="string", low_memory=False)
    for chunk in reader:
        if "adult" in chunk.columns and args.filter_adult:
            adult_bool = normalize_adult(chunk["adult"])
            chunk = chunk[~adult_bool]
        if "vote_count" in chunk.columns and args.min_vote_count > 0:
            vc = pd.to_numeric(chunk["vote_count"], errors="coerce")
            chunk = chunk[vc >= args.min_vote_count]
        if "runtime" in chunk.columns and args.runtime_min > 0:
            rt = pd.to_numeric(chunk["runtime"], errors="coerce")
            chunk = chunk[rt >= args.runtime_min]

        metric_vals = pd.to_numeric(chunk[metric_col], errors="coerce")
        chunk = chunk.assign(metric_val=metric_vals)
        chunk = chunk[metric_vals >= args.min_metric]

        for _, row in chunk.iterrows():
            year = extract_year(row, year_col, date_col)
            if year is None or year < args.year_min or year > args.year_max:
                continue
            metric = row["metric_val"]
            if pd.isna(metric):
                continue
            imdb = str(row[imdb_col]).strip()
            if not imdb:
                continue
            if args.hits_mode == "topn":
                heap = heaps[year]
                if len(heap) < args.hits_n:
                    heapq.heappush(heap, (metric, imdb))
                else:
                    if metric > heap[0][0]:
                        heapq.heapreplace(heap, (metric, imdb))
            else:
                metrics_store[year].append((metric, imdb))

    if args.hits_mode == "topn":
        for year, heap in heaps.items():
            for metric, imdb in heapq.nlargest(args.hits_n, heap):
                hits_per_year[year][imdb] = metric
    else:
        for year, pairs in metrics_store.items():
            if not pairs:
                continue
            values = [m for m, _ in pairs]
            thresh = np.quantile(values, args.hits_percentile)
            for m, imdb in pairs:
                if m >= thresh:
                    hits_per_year[year][imdb] = m

    # Save hits
    rows = [{"year": y, "imdb_id": imdb, "metric_value": m} for y, d in hits_per_year.items() for imdb, m in d.items()]
    hits_df = pd.DataFrame(rows)
    return hits_per_year, hits_df


# ----------------------------
# Pass 2: counts for all and hits
# ----------------------------
def process_tmdb_counts(
    csv_path: Path,
    kw_col: str,
    year_col: Optional[str],
    date_col: Optional[str],
    imdb_col: str,
    pol_map: Dict[str, Dict[str, str]],
    hits_per_year: Dict[int, Dict[str, float]],
    args: argparse.Namespace,
    available_cols: List[str],
) -> Tuple[Counter, Counter, Counter, Counter, Counter, Counter, Counter, Counter, int]:
    usecols = [c for c in [kw_col, imdb_col, "release_year", "release_date", "year", "adult", "vote_count", "runtime"] if c in available_cols]
    kw_all = Counter()
    kw_hit = Counter()
    grp_all = Counter()
    grp_hit = Counter()
    pol_total_all = Counter()
    pol_total_hit = Counter()
    n_all = Counter()
    n_hit = Counter()
    parse_failures = 0

    reader = pd.read_csv(csv_path, chunksize=args.chunksize, usecols=usecols, dtype="string", low_memory=False)
    for chunk in reader:
        if "adult" in chunk.columns and args.filter_adult:
            adult_bool = normalize_adult(chunk["adult"])
            chunk = chunk[~adult_bool]
        if "vote_count" in chunk.columns and args.min_vote_count > 0:
            vc = pd.to_numeric(chunk["vote_count"], errors="coerce")
            chunk = chunk[vc >= args.min_vote_count]
        if "runtime" in chunk.columns and args.runtime_min > 0:
            rt = pd.to_numeric(chunk["runtime"], errors="coerce")
            chunk = chunk[rt >= args.runtime_min]

        for _, row in chunk.iterrows():
            year = extract_year(row, year_col, date_col)
            if year is None or year < args.year_min or year > args.year_max:
                continue
            imdb = str(row[imdb_col]).strip()
            try:
                kws = parse_keywords(row.get(kw_col))
            except Exception:
                parse_failures += 1
                continue
            if not kws:
                continue
            pol_kws = [k for k in kws if k in pol_map]
            if not pol_kws:
                continue
            n_all[year] += 1
            is_hit = imdb in hits_per_year.get(year, {})
            if is_hit:
                n_hit[year] += 1
            if args.use_movie_incidence:
                pol_kws = list(set(pol_kws))
            for kw in pol_kws:
                kw_all[(year, kw)] += 1
                pol_total_all[year] += 1
                groups = pol_map[kw]["matched_groups"]
                primary = pol_map[kw]["primary_group"]
                if args.group_mode == "primary":
                    if primary:
                        grp_all[(year, primary)] += 1
                        if is_hit:
                            grp_hit[(year, primary)] += 1
                else:
                    for g in groups:
                        grp_all[(year, g)] += 1
                        if is_hit:
                            grp_hit[(year, g)] += 1
                if is_hit:
                    kw_hit[(year, kw)] += 1
                    pol_total_hit[year] += 1
    return kw_all, kw_hit, grp_all, grp_hit, pol_total_all, pol_total_hit, n_all, n_hit, parse_failures


# ----------------------------
# Build outputs
# ----------------------------
def build_rates(kw_counts: Counter, n_year: Counter, pol_total: Counter) -> pd.DataFrame:
    rows = [{"year": y, "keyword": kw, "count": c} for (y, kw), c in kw_counts.items()]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.groupby(["year", "keyword"], as_index=False)["count"].sum()
    denom_df = pd.DataFrame({"year": list(n_year.keys()), "N": list(n_year.values())})
    pol_df = pd.DataFrame({"year": list(pol_total.keys()), "pol_total": list(pol_total.values())})
    df = df.merge(denom_df, on="year", how="left").merge(pol_df, on="year", how="left")
    df["rate"] = df["count"] / df["N"]
    df["share_within_political"] = df["count"] / df["pol_total"]
    return df, denom_df, pol_df


def build_group_rates(grp_counts: Counter, n_year: Counter, pol_total: Counter) -> pd.DataFrame:
    rows = [{"year": y, "group": g, "count": c} for (y, g), c in grp_counts.items()]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.groupby(["year", "group"], as_index=False)["count"].sum()
    denom_df = pd.DataFrame({"year": list(n_year.keys()), "N": list(n_year.values())})
    pol_df = pd.DataFrame({"year": list(pol_total.keys()), "pol_total": list(pol_total.values())})
    df = df.merge(denom_df, on="year", how="left").merge(pol_df, on="year", how="left")
    df["group_rate"] = df["count"] / df["N"]
    df["group_share_within_political"] = df["count"] / df["pol_total"]
    return df


def build_hit_vs_rest(kw_all: pd.DataFrame, kw_hit: pd.DataFrame, n_all: pd.DataFrame, n_hit: pd.DataFrame) -> pd.DataFrame:
    rest_rows = []
    all_counts = kw_all.rename(columns={"count": "count_all", "rate": "rate_all"})
    hit_counts = kw_hit.rename(columns={"count": "count_hit", "rate": "rate_hit"})
    merged = all_counts.merge(hit_counts[["year", "keyword", "count_hit", "rate_hit"]], on=["year", "keyword"], how="left")
    merged = merged.fillna({"count_hit": 0, "rate_hit": 0})
    n_all_map = dict(zip(n_all["year"], n_all["N"]))
    n_hit_map = dict(zip(n_hit["year"], n_hit["N"]))
    merged["N_all"] = merged["year"].map(n_all_map)
    merged["N_hit"] = merged["year"].map(n_hit_map)
    merged["N_rest"] = merged["N_all"] - merged["N_hit"]
    merged["count_rest"] = merged["count_all"] - merged["count_hit"]
    merged["rate_rest"] = merged["count_rest"] / merged["N_rest"]
    merged["gap_rate"] = merged["rate_hit"] - merged["rate_rest"]
    merged["ratio_rate"] = merged["rate_hit"] / merged["rate_rest"].replace(0, np.nan)
    return merged


def build_group_hit_vs_rest(grp_all: pd.DataFrame, grp_hit: pd.DataFrame, n_all: pd.DataFrame, n_hit: pd.DataFrame) -> pd.DataFrame:
    merged = grp_all.rename(columns={"count": "count_all", "group_rate": "rate_all"})
    hit = grp_hit.rename(columns={"count": "count_hit", "group_rate": "rate_hit"})
    merged = merged.merge(hit[["year", "group", "count_hit", "rate_hit"]], on=["year", "group"], how="left").fillna(
        {"count_hit": 0, "rate_hit": 0}
    )
    n_all_map = dict(zip(n_all["year"], n_all["N"]))
    n_hit_map = dict(zip(n_hit["year"], n_hit["N"]))
    merged["N_all"] = merged["year"].map(n_all_map)
    merged["N_hit"] = merged["year"].map(n_hit_map)
    merged["N_rest"] = merged["N_all"] - merged["N_hit"]
    merged["count_rest"] = merged["count_all"] - merged["count_hit"]
    merged["rate_rest"] = merged["count_rest"] / merged["N_rest"]
    merged["gap_rate"] = merged["rate_hit"] - merged["rate_rest"]
    merged["ratio_rate"] = merged["rate_hit"] / merged["rate_rest"].replace(0, np.nan)
    return merged


# ----------------------------
# Plotting
# ----------------------------
def mark_shocks(ax, shock_years: List[int]):
    for y in shock_years:
        ax.axvline(y, color="gray", linestyle="--", alpha=0.6)
        ax.text(y, ax.get_ylim()[1], str(y), rotation=90, va="bottom", ha="right", fontsize=8, color="gray")


def plot_keyword_hits(kw_hit: pd.DataFrame, top_x: int, outdir: Path, shock_years: List[int]):
    if kw_hit.empty:
        return
    overall = kw_hit.groupby("keyword", as_index=False)["count"].sum().sort_values("count", ascending=False)
    top_keywords = overall.head(top_x)["keyword"].tolist()
    for col, fname, title in [
        ("rate", "topX_hits_rate_per_movie.png", "Top keywords among hits: rate per movie"),
        ("share_within_political", "topX_hits_share_within_political.png", "Top keywords among hits: share within political"),
    ]:
        pivot = kw_hit[kw_hit["keyword"].isin(top_keywords)].pivot(index="year", columns="keyword", values=col).fillna(0)
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot.plot(ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel(col)
        mark_shocks(ax, shock_years)
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=150)
        plt.close(fig)


def plot_group_hits(grp_hit: pd.DataFrame, outdir: Path, shock_years: List[int]):
    if grp_hit.empty:
        return
    for col, fname, title in [
        ("group_rate", "political_groups_hits_rate_per_movie.png", "Political groups among hits: rate per movie"),
        ("group_share_within_political", "political_groups_hits_share.png", "Political groups among hits: share within political"),
    ]:
        pivot = grp_hit.pivot(index="year", columns="group", values=col).fillna(0)
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot.plot(ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel(col)
        mark_shocks(ax, shock_years)
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=150)
        plt.close(fig)


def plot_group_gap(grp_gap: pd.DataFrame, outdir: Path, shock_years: List[int]):
    if grp_gap.empty:
        return
    pivot = grp_gap.pivot(index="year", columns="group", values="gap_rate").fillna(0)
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(ax=ax)
    ax.set_title("Group gap rate (hits - rest)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Gap rate")
    mark_shocks(ax, shock_years)
    fig.tight_layout()
    fig.savefig(outdir / "political_groups_hit_vs_rest_gap.png", dpi=150)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    shock_years = [int(s) for s in args.shock_years.split(",") if s.strip()]

    # Political dictionary
    candidates = load_candidates(Path(args.candidate_csv))
    pol_df, pol_map = build_political_dict(candidates, args.coverage, args.min_count)
    pol_df.to_csv(outdir / "political_keywords.csv", index=False)
    pol_df.head(200).to_csv(outdir / "political_keywords_top200.csv", index=False)
    total_mass = candidates["count"].sum()
    pol_mass = candidates[candidates["keyword"].isin(pol_df["keyword"])]["count"].sum()
    print(f"Political keywords: {len(pol_df)}")
    print(f"Mass share: {pol_mass}/{total_mass} = {pol_mass/total_mass if total_mass else 0:.4f}")

    # Detect columns
    head = pd.read_csv(args.tmdb_csv, nrows=0)
    columns = head.columns.tolist()
    kw_col, year_col, date_col, imdb_col = detect_columns(columns)
    metric_col = detect_metric_column(columns, args.metric)

    # Pass 1: hits
    hits_per_year, hits_df = find_hits(Path(args.tmdb_csv), metric_col, imdb_col, year_col, date_col, args, columns)
    hits_df.to_csv(outdir / "hits_imdb_ids.csv", index=False)
    print("Hits per year (sample):")
    for y in sorted(hits_per_year.keys())[:5]:
        print(f"  {y}: {len(hits_per_year[y])}")

    # Pass 2: counts
    kw_all_c, kw_hit_c, grp_all_c, grp_hit_c, pol_all_c, pol_hit_c, n_all_c, n_hit_c, parse_failures = process_tmdb_counts(
        Path(args.tmdb_csv), kw_col, year_col, date_col, imdb_col, pol_map, hits_per_year, args, columns
    )
    if parse_failures:
        print(f"Parse failures during keyword parsing: {parse_failures}")

    # Build rates
    kw_all, denom_all, pol_all = build_rates(kw_all_c, n_all_c, pol_all_c)
    kw_hit, denom_hit, pol_hit = build_rates(kw_hit_c, n_hit_c, pol_hit_c)
    kw_all.to_csv(outdir / "yearly_keyword_rates_all.csv", index=False)
    kw_hit.to_csv(outdir / "yearly_keyword_rates_hit.csv", index=False)
    denom_all.rename(columns={"N": "N_all"}).merge(pol_all, on="year", how="left").to_csv(
        outdir / "yearly_denominators_all.csv", index=False
    )
    denom_hit.rename(columns={"N": "N_hit"}).merge(pol_hit, on="year", how="left").to_csv(
        outdir / "yearly_denominators_hit.csv", index=False
    )

    grp_all = build_group_rates(grp_all_c, n_all_c, pol_all_c)
    grp_hit = build_group_rates(grp_hit_c, n_hit_c, pol_hit_c)
    grp_all.to_csv(outdir / "yearly_group_rates_all.csv", index=False)
    grp_hit.to_csv(outdir / "yearly_group_rates_hit.csv", index=False)

    # Hit vs rest
    kw_gap = build_hit_vs_rest(kw_all, kw_hit, denom_all, denom_hit)
    grp_gap = build_group_hit_vs_rest(grp_all, grp_hit, denom_all, denom_hit)
    kw_gap.to_csv(outdir / "yearly_keyword_hit_vs_rest.csv", index=False)
    grp_gap.to_csv(outdir / "yearly_group_hit_vs_rest.csv", index=False)

    # Plots
    plot_keyword_hits(kw_hit, args.top_x, outdir, shock_years)
    plot_group_hits(grp_hit, outdir, shock_years)
    plot_group_gap(grp_gap, outdir, shock_years)

    # Diagnostics
    if denom_hit["N"].lt(args.hits_n).any():
        warn_years = denom_hit[denom_hit["N"] < args.hits_n]["year"].tolist()
        print(f"Warning: years with fewer than hits-n movies counted: {warn_years}")
    if not denom_hit.empty:
        overall_rate_hit = pol_hit["pol_total"].sum() / denom_hit["N"].sum() if denom_hit["N"].sum() else 0
        print(f"Overall political rate among hits: {overall_rate_hit:.4f}")


if __name__ == "__main__":
    main()
