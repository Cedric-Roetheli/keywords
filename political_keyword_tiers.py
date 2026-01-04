"""
Compare political keyword intensity between success tiers within each year of TMDb.

Two-pass approach:
- Pass 1: build per-year rank lists (top max_rank) after theatrical filters.
- Pass 2: compute political keyword incidence for ranked movies and aggregate by tiers.
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

# Patterns
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
    "war_security_intel",
    "economy_finance_crisis",
    "institutions_elections_law",
    "migration_police_civilrights",
    "labor_collective_action",
    "inequality_corruption_elites",
]

BAD_TOKENS = {"<na>", "nan", "none", "null", ""}
KW_COLS = ["keywords", "keyword_names", "tmdb_keywords", "Keywords", "keyword"]
ID_COLS = ["imdb_id", "id", "tmdb_id"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Political keyword intensity by success tiers.")
    p.add_argument("--tmdb-csv", required=True)
    p.add_argument("--outdir", default="./outputs_tiers")
    p.add_argument("--year-min", type=int, default=1970)
    p.add_argument("--year-max", type=int, default=2023)
    p.add_argument("--metric", default="revenue", choices=["revenue", "vote_count", "popularity"])
    p.add_argument("--filter-adult", action="store_true", default=True)
    p.add_argument("--runtime-min", type=float, default=40)
    p.add_argument("--min-vote-count", type=float, default=50)
    p.add_argument("--chunksize", type=int, default=200_000)
    p.add_argument("--shock-years", default="2001,2008")
    p.add_argument("--max-rank", type=int, default=200)
    return p.parse_args()


# ---------------------------- utilities ----------------------------
def normalize_token(tok: str) -> str:
    tok = tok.strip().lower()
    tok = "_".join(tok.split())
    return tok


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


def detect_columns(columns: List[str], metric_pref: str) -> Tuple[str, str, Optional[str], Optional[str]]:
    kw_col = next((c for c in KW_COLS if c in columns), None)
    if kw_col is None:
        raise ValueError(f"No keyword column found. Available: {columns}")
    id_col = next((c for c in ID_COLS if c in columns), None)
    if id_col is None:
        raise ValueError(f"No ID column found. Available: {columns}")
    year_col = "release_year" if "release_year" in columns else ("year" if "year" in columns else None)
    date_col = "release_date" if "release_date" in columns else None
    # metric fallback
    order = {
        "revenue": ["revenue", "vote_count", "popularity"],
        "vote_count": ["vote_count", "popularity", "revenue"],
        "popularity": ["popularity", "vote_count", "revenue"],
    }[metric_pref]
    metric_col = next((c for c in order if c in columns), None)
    if metric_col is None:
        raise ValueError(f"No metric column found. Available: {columns}")
    return kw_col, id_col, year_col, date_col, metric_col


def classify_political(kw: str) -> Tuple[bool, Optional[str]]:
    for g in GROUP_PRIORITY:
        if POLITICAL_PATTERNS[g].search(kw):
            return True, g
    # if not priority matched, still check others for multi? requirement chooses primary by priority
    for name, pat in POLITICAL_PATTERNS.items():
        if pat.search(kw):
            return True, name
    return False, None


# ---------------------------- Pass 1 ----------------------------
def build_rank_lists_pass1(csv_path: Path, cols: List[str], kw_col: str, id_col: str, year_col: Optional[str], date_col: Optional[str], metric_col: str, args: argparse.Namespace) -> pd.DataFrame:
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
    hits_df = pd.DataFrame(rows)
    return hits_df


# ---------------------------- Pass 2 ----------------------------
def compute_intensity_pass2(
    csv_path: Path,
    cols: List[str],
    kw_col: str,
    id_col: str,
    year_col: Optional[str],
    date_col: Optional[str],
    ranked_map: Dict[int, Dict[str, int]],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Counter], Counter]:
    usecols = [c for c in [kw_col, id_col, year_col, date_col, "adult", "runtime", "vote_count"] if c and c in cols]
    sum_kw = {tier: Counter() for tier in ["top20", "21_200", "top50", "51_200"]}
    sum_any = {tier: Counter() for tier in ["top20", "21_200", "top50", "51_200"]}
    n_tier = {tier: Counter() for tier in ["top20", "21_200", "top50", "51_200"]}
    sum_group = {tier: Counter() for tier in ["top20", "21_200", "top50", "51_200"]}
    reader = pd.read_csv(csv_path, chunksize=args.chunksize, usecols=usecols, dtype="string", low_memory=False)
    parse_failures = 0
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

        for _, row in chunk.iterrows():
            year = extract_year(row, year_col, date_col)
            if year is None or year < args.year_min or year > args.year_max:
                continue
            rmap = ranked_map.get(year)
            if not rmap:
                continue
            movie_id = str(row[id_col]).strip()
            rank = rmap.get(movie_id)
            if rank is None or rank > args.max_rank:
                continue
            try:
                kws = parse_keywords(row.get(kw_col))
            except Exception:
                parse_failures += 1
                continue
            pol_kws = []
            group_set = defaultdict(int)
            for kw in set(kws):
                is_pol, group = classify_political(kw)
                if is_pol and group:
                    pol_kws.append(kw)
                    group_set[group] += 1
            pol_count = len(pol_kws)
            pol_any = 1 if pol_count > 0 else 0

            def update(tier):
                sum_kw[tier][year] += pol_count
                sum_any[tier][year] += pol_any
                n_tier[tier][year] += 1
                for g, c in group_set.items():
                    sum_group[tier][(year, g)] += c

            if rank <= 20:
                update("top20")
            elif 21 <= rank <= 200:
                update("21_200")
            if rank <= 50:
                update("top50")
            elif 51 <= rank <= 200:
                update("51_200")
    if parse_failures:
        print(f"Parse failures during pass2: {parse_failures}")
    return (sum_kw, sum_any, n_tier, sum_group), parse_failures


# ---------------------------- Outputs ----------------------------
def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def write_tables(sum_kw: Dict[str, Counter], sum_any: Dict[str, Counter], n_tier: Dict[str, Counter], sum_group: Dict[str, Counter], outdir: Path):
    years = sorted(set(list(n_tier["top20"].keys()) + list(n_tier["21_200"].keys()) + list(n_tier["top50"].keys()) + list(n_tier["51_200"].keys())))
    rows = []
    for y in years:
        n20 = n_tier["top20"][y]
        n180 = n_tier["21_200"][y]
        n50 = n_tier["top50"][y]
        n150 = n_tier["51_200"][y]
        rows.append(
            {
                "year": y,
                "n_top20": n20,
                "n_21_200": n180,
                "mean_kw_top20": safe_div(sum_kw["top20"][y], n20),
                "mean_kw_21_200": safe_div(sum_kw["21_200"][y], n180),
                "gap_kw_20_vs_21_200": safe_div(sum_kw["top20"][y], n20) - safe_div(sum_kw["21_200"][y], n180),
                "share_any_top20": safe_div(sum_any["top20"][y], n20),
                "share_any_21_200": safe_div(sum_any["21_200"][y], n180),
                "gap_any_20_vs_21_200": safe_div(sum_any["top20"][y], n20) - safe_div(sum_any["21_200"][y], n180),
                "n_top50": n50,
                "n_51_200": n150,
                "mean_kw_top50": safe_div(sum_kw["top50"][y], n50),
                "mean_kw_51_200": safe_div(sum_kw["51_200"][y], n150),
                "gap_kw_50_vs_51_200": safe_div(sum_kw["top50"][y], n50) - safe_div(sum_kw["51_200"][y], n150),
                "share_any_top50": safe_div(sum_any["top50"][y], n50),
                "share_any_51_200": safe_div(sum_any["51_200"][y], n150),
                "gap_any_50_vs_51_200": safe_div(sum_any["top50"][y], n50) - safe_div(sum_any["51_200"][y], n150),
            }
        )
    df = pd.DataFrame(rows).sort_values("year")
    df.to_csv(outdir / "yearly_tier_keyword_intensity.csv", index=False)

    # Groups
    group_rows = []
    groups = set(g for _, g in sum_group["top20"].keys()) | set(g for _, g in sum_group["21_200"].keys()) | set(g for _, g in sum_group["top50"].keys()) | set(g for _, g in sum_group["51_200"].keys())
    for y in years:
        for g in groups:
            group_rows.append(
                {
                    "year": y,
                    "group": g,
                    "mean_groupkw_top20": safe_div(sum_group["top20"][(y, g)], n_tier["top20"][y]),
                    "mean_groupkw_21_200": safe_div(sum_group["21_200"][(y, g)], n_tier["21_200"][y]),
                    "gap_groupkw_20_vs_21_200": safe_div(sum_group["top20"][(y, g)], n_tier["top20"][y]) - safe_div(
                        sum_group["21_200"][(y, g)], n_tier["21_200"][y]
                    ),
                    "mean_groupkw_top50": safe_div(sum_group["top50"][(y, g)], n_tier["top50"][y]),
                    "mean_groupkw_51_200": safe_div(sum_group["51_200"][(y, g)], n_tier["51_200"][y]),
                    "gap_groupkw_50_vs_51_200": safe_div(sum_group["top50"][(y, g)], n_tier["top50"][y]) - safe_div(
                        sum_group["51_200"][(y, g)], n_tier["51_200"][y]
                    ),
                }
            )
    df_group = pd.DataFrame(group_rows).sort_values(["year", "group"])
    df_group.to_csv(outdir / "yearly_tier_group_intensity.csv", index=False)
    return df, df_group


# ---------------------------- Plots ----------------------------
def mark_shocks(ax, shock_years: List[int]):
    for y in shock_years:
        ax.axvline(y, color="gray", linestyle="--", alpha=0.6)
        ax.text(y, ax.get_ylim()[1], str(y), rotation=90, va="bottom", ha="right", fontsize=8, color="gray")


def plot_results(tier_df: pd.DataFrame, group_df: pd.DataFrame, shock_years: List[int], outdir: Path):
    if tier_df.empty:
        return
    for cols, fname, title, ylabel in [
        (["mean_kw_top20", "mean_kw_21_200"], "mean_political_keywords_top20_vs_21_200.png", "Mean political keywords: Top20 vs 21-200", "Mean keywords per film"),
        (["gap_kw_20_vs_21_200"], "gap_political_keywords_20_vs_21_200.png", "Gap: Top20 - 21-200", "Gap"),
        (["share_any_top20", "share_any_21_200"], "share_any_top20_vs_21_200.png", "Share of films with any political keyword", "Share"),
        (["mean_kw_top50", "mean_kw_51_200"], "mean_political_keywords_top50_vs_51_200.png", "Mean political keywords: Top50 vs 51-200", "Mean keywords per film"),
        (["gap_kw_50_vs_51_200"], "gap_political_keywords_50_vs_51_200.png", "Gap: Top50 - 51-200", "Gap"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for c in cols:
            ax.plot(tier_df["year"], tier_df[c], label=c)
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        mark_shocks(ax, shock_years)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=150)
        plt.close(fig)

    if not group_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot = group_df.pivot(index="year", columns="group", values="gap_groupkw_20_vs_21_200").fillna(0)
        pivot.plot(ax=ax)
        ax.set_title("Group gap (Top20 - 21-200)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Gap")
        mark_shocks(ax, shock_years)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(outdir / "group_gap_20_vs_21_200.png", dpi=150)
        plt.close(fig)


# ---------------------------- Main ----------------------------
def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    shock_years = [int(s) for s in args.shock_years.split(",") if s.strip()]

    head = pd.read_csv(args.tmdb_csv, nrows=0)
    cols = head.columns.tolist()
    kw_col, id_col, year_col, date_col, metric_col = detect_columns(cols, args.metric)
    print(f"Detected columns: keywords={kw_col}, id={id_col}, year={year_col or date_col}, metric={metric_col}")

    # Pass 1
    hits_df = build_rank_lists_pass1(Path(args.tmdb_csv), cols, kw_col, id_col, year_col, date_col, metric_col, args)
    hits_df.to_csv(outdir / "hits_rank_lists.csv", index=False)
    ranked_map: Dict[int, Dict[str, int]] = defaultdict(dict)
    for _, r in hits_df.iterrows():
        ranked_map[int(r["year"])][str(r["id"])] = int(r["rank"])

    # Pass 2
    (sum_kw, sum_any, n_tier, sum_group), _ = compute_intensity_pass2(
        Path(args.tmdb_csv), cols, kw_col, id_col, year_col, date_col, ranked_map, args
    )

    # Outputs
    tier_df, group_df = write_tables(sum_kw, sum_any, n_tier, sum_group, outdir)
    plot_results(tier_df, group_df, shock_years, outdir)

    # Console summary
    full20 = tier_df[tier_df["n_top20"] >= 20].shape[0]
    full50 = tier_df[tier_df["n_top50"] >= 50].shape[0]
    print(f"Years with full Top20: {full20} / {len(tier_df)}; full Top50: {full50}")
    avg_gap20 = tier_df["gap_kw_20_vs_21_200"].mean()
    avg_top20 = tier_df["mean_kw_top20"].mean()
    avg_21_200 = tier_df["mean_kw_21_200"].mean()
    print(f"Avg mean_kw_top20: {avg_top20:.3f}, mean_kw_21_200: {avg_21_200:.3f}, gap: {avg_gap20:.3f}")
    if not tier_df.empty:
        top_gap_years = tier_df.sort_values("gap_kw_20_vs_21_200", ascending=False).head(10)
        print("Top 10 years by gap_kw_20_vs_21_200:")
        print(top_gap_years[["year", "gap_kw_20_vs_21_200"]])
    for y in shock_years:
        row = tier_df[tier_df["year"] == y]
        if not row.empty:
            print(f"Shock {y}: gap20={row.iloc[0]['gap_kw_20_vs_21_200']:.3f}, gap50={row.iloc[0]['gap_kw_50_vs_51_200']:.3f}")


if __name__ == "__main__":
    main()
