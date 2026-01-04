"""
Build a political keyword list from candidates, count their yearly frequency in TMDb, and plot trends.

Steps:
- Load candidate_keywords.csv, drop invalid tokens, apply fixed regex groups to flag political keywords.
- Save political keyword lists.
- Stream TMDb CSV, parse keywords, filter movies (adult/vote_count/runtime/year), and count political keywords by year.
- Plot top-X political keywords over time and total political keyword counts.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


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

BAD_TOKENS = {"<na>", "nan", "none", "null", ""}
KEYWORD_COLUMNS = ["keywords", "keyword_names", "Keywords", "tmdb_keywords", "keyword"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Political keyword trends over time from TMDb.")
    parser.add_argument("--tmdb-csv", required=True, help="Path to TMDb CSV (e.g., TMDB_movie_dataset_v11.csv)")
    parser.add_argument("--candidate-csv", required=True, help="Path to candidate_keywords.csv")
    parser.add_argument("--outdir", default="./outputs_political_keywords")
    parser.add_argument("--year-min", type=int, default=1970)
    parser.add_argument("--year-max", type=int, default=2023)
    parser.add_argument("--top-x", type=int, default=15)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--use-movie-incidence", action="store_true", default=True, help="Count each keyword at most once per movie")
    parser.add_argument("--filter-adult", action="store_true", default=True)
    parser.add_argument("--min-vote-count", type=float, default=0)
    parser.add_argument("--runtime-min", type=float, default=0)
    return parser.parse_args()


# ----------------------------
# Utilities
# ----------------------------
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


def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["keyword"] = df["keyword"].astype(str).str.strip()
    df = df[~df["keyword"].str.lower().isin(BAD_TOKENS)]
    if "count" in df.columns:
        df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    return df


def build_political_set(df: pd.DataFrame) -> Tuple[pd.DataFrame, set]:
    rows = []
    for _, r in df.iterrows():
        kw = normalize_token(str(r["keyword"]))
        matched = []
        for name, pat in POLITICAL_PATTERNS.items():
            if pat.search(kw):
                matched.append(name)
        if matched:
            rows.append({"keyword": kw, "count": r.get("count", 0), "matched_groups": ";".join(matched)})
    pol_df = pd.DataFrame(rows).sort_values("count", ascending=False)
    return pol_df, set(pol_df["keyword"].tolist())


def detect_columns(columns: List[str]) -> Tuple[str, Optional[str], Optional[str]]:
    kw_col = None
    for cand in KEYWORD_COLUMNS:
        if cand in columns:
            kw_col = cand
            break
    if kw_col is None:
        raise ValueError(f"No keyword column found. Available columns: {columns}")
    year_col = "release_year" if "release_year" in columns else ("year" if "year" in columns else None)
    date_col = "release_date" if "release_date" in columns else None
    return kw_col, year_col, date_col


def process_tmdb_chunks(
    csv_path: Path,
    kw_col: str,
    year_col: Optional[str],
    date_col: Optional[str],
    pol_set: set,
    args: argparse.Namespace,
    available_cols: List[str],
) -> Counter:
    usecols = [c for c in [kw_col, "release_year", "release_date", "year", "adult", "vote_count", "runtime"] if c in available_cols]
    counts = Counter()
    reader = pd.read_csv(csv_path, chunksize=args.chunksize, usecols=usecols, dtype="string", low_memory=False)
    for chunk in reader:
        # Filters
        if "adult" in chunk.columns and args.filter_adult:
            chunk = chunk[chunk["adult"].str.lower().isin(["false", "f", "0", "no", ""]) | chunk["adult"].isna()]
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
            kws = parse_keywords(row.get(kw_col))
            if not kws:
                continue
            pol_kws = [k for k in kws if k in pol_set]
            if not pol_kws:
                continue
            if args.use_movie_incidence:
                pol_kws = set(pol_kws)
            for kw in pol_kws:
                counts[(year, kw)] += 1
    return counts


def make_plots(yearly_df: pd.DataFrame, total_df: pd.DataFrame, top_df: pd.DataFrame, outdir: Path, top_x: int) -> None:
    if yearly_df.empty:
        return
    pivot = yearly_df.pivot(index="year", columns="keyword", values="count").fillna(0)
    pivot = pivot[top_df["keyword"].head(top_x)] if not top_df.empty else pivot

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(ax=ax)
    ax.set_title(f"Top {top_x} political keywords over time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(outdir / "top_political_keywords_over_time.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot.area(ax=ax, alpha=0.8)
    ax.set_title(f"Top {top_x} political keywords (stacked area)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(outdir / "top_political_keywords_area.png", dpi=150)
    plt.close(fig)

    if not total_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(total_df["year"], total_df["total_count"], marker="o")
        ax.set_title("Total political keyword counts per year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(outdir / "total_political_keyword_counts_over_time.png", dpi=150)
        plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Step A: political list
    candidates = load_candidates(Path(args.candidate_csv))
    pol_df, pol_set = build_political_set(candidates)
    pol_df.to_csv(outdir / "political_keywords.csv", index=False)
    pol_df.head(200).to_csv(outdir / "political_keywords_top.csv", index=False)

    total_mass = candidates["count"].sum()
    pol_mass = candidates[candidates["keyword"].isin(pol_set)]["count"].sum()
    print(f"Political keywords: {len(pol_set)}")
    print(f"Mass share: {pol_mass}/{total_mass} = {pol_mass/total_mass if total_mass else 0:.4f}")
    print("Top 20 political keywords:")
    print(pol_df.head(20)[["keyword", "count"]])

    # Step B: counts over time
    head = pd.read_csv(args.tmdb_csv, nrows=0)
    columns = head.columns.tolist()
    kw_col, year_col, date_col = detect_columns(columns)
    counts = process_tmdb_chunks(Path(args.tmdb_csv), kw_col, year_col, date_col, pol_set, args, columns)
    rows = [{"year": y, "keyword": kw, "count": c} for (y, kw), c in counts.items()]
    yearly_df = pd.DataFrame(rows)
    if not yearly_df.empty:
        yearly_df = yearly_df.groupby(["year", "keyword"], as_index=False)["count"].sum()
    yearly_df.to_csv(outdir / "yearly_political_keyword_counts.csv", index=False)

    total_df = yearly_df.groupby("year", as_index=False)["count"].sum().rename(columns={"count": "total_count"})
    total_df.to_csv(outdir / "total_political_counts_by_year.csv", index=False)

    top_df = yearly_df.groupby("keyword", as_index=False)["count"].sum().sort_values("count", ascending=False)
    top_keywords = top_df.head(args.top_x)["keyword"].tolist()
    years_top_counts = yearly_df[yearly_df["keyword"].isin(top_keywords)]
    if not years_top_counts.empty:
        years_top_counts = years_top_counts.pivot(index="year", columns="keyword", values="count").fillna(0)
    highest_years = total_df.sort_values("total_count", ascending=False).head(5)
    print("Years with highest total political keyword counts:")
    print(highest_years)

    # Step C: plots
    make_plots(yearly_df, total_df, top_df, outdir, args.top_x)


if __name__ == "__main__":
    main()
