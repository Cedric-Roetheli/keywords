"""
Political keyword tier analysis using US domestic box office rankings (clean version).

Build Top-100 per year from Mojo domestic grosses, merge to TMDb, apply year-quality filters,
and compare Top20 vs 21–100 political content (counts and shares), plus genre buckets.
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
    p = argparse.ArgumentParser(description="US domestic box office tier analysis with political keywords.")
    p.add_argument("--tmdb-csv", required=True)
    p.add_argument("--mojo-csv", required=True)
    p.add_argument("--mojo-fallback-csv", default=None, help="Optional fallback Mojo file to improve early-year coverage")
    p.add_argument("--fallback-year-max", type=int, default=1998, help="Use fallback file for years <= this (if provided)")
    p.add_argument("--outdir", default="./outputs_us_market_mojo_clean")
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


# ---------- helpers ----------
def clean_money(val) -> float:
    if pd.isna(val):
        return math.nan
    s = str(val).replace("$", "").replace(",", "").strip()
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
    return series.astype(str).str.lower().isin(["true", "t", "1", "yes"])


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


def detect_columns_mojo(columns: List[str]) -> Tuple[str, Optional[str], Optional[str], str]:
    title_col = "title" if "title" in columns else next((c for c in columns if "title" in c.lower()), None)
    year_col = None
    for cand in ["year", "release_year", "release_date", "movie_year", "Year"]:
        if cand in columns:
            year_col = cand
            break
    if year_col is None and "release_date" in columns:
        year_col = "release_date"
    imdb_col = "imdb_id" if "imdb_id" in columns else None
    domestic_col = None
    for cand in ["domestic", "Domestic", "Gross"]:
        if cand in columns:
            domestic_col = cand
            break
    if title_col is None or domestic_col is None:
        raise ValueError("Box office file must have title and domestic/gross columns")
    return title_col, year_col, imdb_col, domestic_col


def normalize_title(title: str) -> str:
    if title is None:
        return ""
    s = str(title).lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = "_".join(s.split())
    return s


def classify_political(kw: str) -> bool:
    for pat in POLITICAL_PATTERNS.values():
        if pat.search(kw):
            return True
    return False


def classify_primary_group(kw: str) -> Optional[str]:
    for g in [
        "war_security_intel",
        "institutions_elections_law",
        "economy_finance_crisis",
        "migration_police_civilrights",
        "labor_collective_action",
        "inequality_corruption_elites",
    ]:
        if POLITICAL_PATTERNS[g].search(kw):
            return g
    for name, pat in POLITICAL_PATTERNS.items():
        if pat.search(kw):
            return name
    return None


# ---------- rankings ----------
def load_mojo_file(path: Path, year_min: int, year_max: int) -> pd.DataFrame:
    mojo = pd.read_csv(path)
    title_col, year_col, imdb_col, domestic_col = detect_columns_mojo(mojo.columns.tolist())
    mojo["domestic_clean"] = mojo[domestic_col].apply(clean_money)
    if year_col == "release_date":
        mojo["year_clean"] = mojo["release_date"].astype(str).str.extract(r"(\d{4})")[0]
    else:
        mojo["year_clean"] = mojo[year_col]
    mojo["year_clean"] = pd.to_numeric(mojo["year_clean"], errors="coerce").astype("Int64")
    mojo = mojo[(mojo["year_clean"] >= year_min) & (mojo["year_clean"] <= year_max)]
    mojo["title_col"] = title_col
    mojo["imdb_col"] = imdb_col
    mojo["domestic_col"] = domestic_col
    return mojo


def build_rankings(mojo_path: Path, args: argparse.Namespace) -> pd.DataFrame:
    mojo_main = load_mojo_file(mojo_path, args.year_min, args.year_max)
    mojo_all = mojo_main.copy()
    if args.mojo_fallback_csv:
        fallback_df = load_mojo_file(Path(args.mojo_fallback_csv), args.year_min, args.fallback_year_max)
        mojo_all = pd.concat([mojo_main, fallback_df], ignore_index=True)

    rows = []
    for year, g in mojo_all.groupby("year_clean"):
        g = g.dropna(subset=["domestic_clean"])
        # unify columns
        title_col = g["title_col"].iloc[0]
        imdb_col = g["imdb_col"].iloc[0] if g["imdb_col"].notna().any() else None
        domestic_col = g["domestic_col"].iloc[0]
        # drop duplicates on normalized title keeping max domestic
        g = g.assign(title_norm=g[title_col].astype(str).apply(normalize_title))
        g = g.sort_values("domestic_clean", ascending=False).drop_duplicates(subset=["title_norm"], keep="first").head(args.max_rank)
        for idx, r in enumerate(g.itertuples(), start=1):
            rows.append(
                {
                    "year": int(year),
                    "rank": idx,
                    "mojo_title": getattr(r, title_col),
                    "title_norm": r.title_norm,
                    "domestic": r.domestic_clean,
                    "imdb_id": getattr(r, imdb_col) if imdb_col else None,
                    "source_row": r.Index,
                }
            )
    return pd.DataFrame(rows)


def fuzzy_match_by_year(rank_df: pd.DataFrame, tmdb_year_map: Dict[int, pd.DataFrame], threshold: int, use_fuzzy: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    matched_rows = []
    diag_rows = []
    for year, g in rank_df.groupby("year"):
        tmdb_year_df = tmdb_year_map.get(year)
        if tmdb_year_df is None or tmdb_year_df.empty:
            diag_rows.append({"year": year, "match_rate": 0, "note": "no tmdb rows"})
            continue
        tmdb_titles = tmdb_year_df["title_norm"].tolist()
        tmdb_lookup = dict(zip(tmdb_year_df["title_norm"], tmdb_year_df["tmdb_id"]))
        used = set()
        matched = 0
        for row in g.itertuples():
            best_id = None
            best_score = -1
            if use_fuzzy and HAVE_RF:
                res = process.extract(normalize_title(row.mojo_title), tmdb_titles, scorer=fuzz.token_set_ratio, limit=3)
                if res:
                    cand, score, _ = res[0]
                    if score >= threshold:
                        best_id = tmdb_lookup[cand]
                        best_score = score
            else:
                matches = difflib.get_close_matches(normalize_title(row.mojo_title), tmdb_titles, n=1, cutoff=threshold / 100)
                if matches:
                    cand = matches[0]
                    best_id = tmdb_lookup[cand]
                    best_score = threshold
            if best_id and best_id not in used:
                used.add(best_id)
                matched += 1
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
        diag_rows.append({"year": year, "match_rate": rate, "note": ""})
    return pd.DataFrame(matched_rows), pd.DataFrame(diag_rows)


def merge_rankings(rank_df: pd.DataFrame, tmdb_df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    matched = pd.DataFrame()
    diag = []
    # imdb merge if available
    if "imdb_id" in tmdb_df.columns and rank_df["imdb_id"].notna().any():
        direct = rank_df.dropna(subset=["imdb_id"]).merge(tmdb_df, on="imdb_id", how="left", suffixes=("", "_tmdb"))
        matched = direct[~direct["tmdb_id"].isna()].copy()
        diag.append({"year": "ALL", "match_rate": len(matched) / len(rank_df), "note": "imdb merge subset"})
        remaining = rank_df[rank_df["imdb_id"].isna() | rank_df["imdb_id"].eq("")]
    else:
        remaining = rank_df
    tmdb_year_map = {y: g for y, g in tmdb_df.groupby("year")}
    fuzzy_matches, diag_df = fuzzy_match_by_year(remaining, tmdb_year_map, args.fuzzy_threshold, args.fuzzy_title_match)
    if not fuzzy_matches.empty:
        fuzzy_merge = fuzzy_matches.merge(tmdb_df, on=["year", "tmdb_id"], how="left")
        matched = pd.concat([matched, fuzzy_merge], ignore_index=True)
    if diag_df is not None:
        diag.append(diag_df)
    diag_df_final = pd.concat(diag, ignore_index=True) if diag else pd.DataFrame()
    return matched, diag_df_final


# ---------- metrics ----------
def compute_movie_metrics(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        kw_set = set(parse_keywords(getattr(r, "keywords")))
        total_kw = len(kw_set)
        pol_groups = Counter()
        for kw in kw_set:
            g = classify_primary_group(kw)
            if g:
                pol_groups[g] += 1
        pol_count = sum(pol_groups.values())
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
                **{f"group_{g}": pol_groups.get(g, 0) for g in POLITICAL_PATTERNS.keys()},
            }
        )
    return pd.DataFrame(rows)


def summarize_yearly(metrics_df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    bucket_rows = []
    group_rows = []
    for year, g in metrics_df.groupby("year"):
        n_ranked = len(g)
        top20 = g[g["rank"] <= args.top20]
        rest = g[(g["rank"] > args.top20) & (g["rank"] <= args.max_rank)]
        n_top20 = len(top20)
        n_rest = len(rest)
        rows.append(
            {
                "year": year,
                "n_ranked_matched": n_ranked,
                "n_top20_matched": n_top20,
                "n_21_100_matched": n_rest,
                "mean_domestic_top20": top20["domestic"].mean(),
                "mean_domestic_21_100": rest["domestic"].mean(),
                "median_domestic_top20": top20["domestic"].median(),
                "median_domestic_21_100": rest["domestic"].median(),
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
            }
        )
        for group in POLITICAL_PATTERNS.keys():
            c_top = top20[f"group_{group}"].sum() if f"group_{group}" in top20.columns else 0
            c_rest = rest[f"group_{group}"].sum() if f"group_{group}" in rest.columns else 0
            pol_total_top = top20["pol_count"].sum()
            pol_total_rest = rest["pol_count"].sum()
            group_rows.append(
                {
                    "year": year,
                    "group": group,
                    "count_top20": c_top,
                    "count_21_100": c_rest,
                    "share_top20": c_top / pol_total_top if pol_total_top else np.nan,
                    "share_21_100": c_rest / pol_total_rest if pol_total_rest else np.nan,
                    "gap_share": (c_top / pol_total_top if pol_total_top else np.nan) - (c_rest / pol_total_rest if pol_total_rest else np.nan),
                }
            )
        for bucket, gb in g.groupby("bucket"):
            tb = gb[gb["rank"] <= args.top20]
            rb = gb[(gb["rank"] > args.top20) & (gb["rank"] <= args.max_rank)]
            bucket_rows.append(
                {
                    "year": year,
                    "bucket": bucket,
                    "n_top20": len(tb),
                    "n_21_100": len(rb),
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
    tier_df = pd.DataFrame(rows).sort_values("year")
    bucket_df = pd.DataFrame(bucket_rows).sort_values(["year", "bucket"])
    group_df = pd.DataFrame(group_rows).sort_values(["year", "group"])

    # pooled genre composition (use buckets as genre surrogate)
    pooled_rows = []
    for tier_label, tier_df_sub in [("top20", metrics_df[metrics_df["rank"] <= args.top20]), ("21_100", metrics_df[(metrics_df["rank"] > args.top20) & (metrics_df["rank"] <= args.max_rank)])]:
        counts = tier_df_sub["bucket"].value_counts()
        total = counts.sum()
        for gname, c in counts.items():
            pooled_rows.append({"tier": tier_label, "genre": gname, "share": c / total if total else 0, "count": c})
    pooled_df = pd.DataFrame(pooled_rows)
    return tier_df, bucket_df, pooled_df, group_df


def quality_filter(tier_df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    excluded = []
    keep_rows = []
    for r in tier_df.itertuples():
        reason = []
        if r.year == 2020:
            reason.append("drop_2020")
        if r.n_ranked_matched < 50:
            reason.append("ranked_matched_lt50")
        if r.n_top20_matched < 12:
            reason.append("top20_matched_lt12")
        if reason:
            excluded.append({"year": r.year, "n_ranked_matched": r.n_ranked_matched, "n_top20_matched": r.n_top20_matched, "reason": ";".join(reason)})
        else:
            keep_rows.append(r.Index)
    keep_df = tier_df.loc[keep_rows].reset_index(drop=True)
    excluded_df = pd.DataFrame(excluded)
    return keep_df, excluded_df


# ---------- plots ----------
DISPLAY_NAMES = {
    "war_security_intel": "War/Security/Intelligence",
    "institutions_elections_law": "Institutions/Law/Elections",
    "economy_finance_crisis": "Economy/Finance/Crisis",
    "migration_police_civilrights": "Migration/Police/Civil Rights",
    "labor_collective_action": "Labor/Collective Action",
    "inequality_corruption_elites": "Inequality/Corruption/Elites",
}


def make_plots(tier_df: pd.DataFrame, bucket_df: pd.DataFrame, pooled_df: pd.DataFrame, outdir: Path, group_df: Optional[pd.DataFrame] = None):
    if tier_df.empty:
        return
    plot_specs = [
        (["mean_domestic_top20", "mean_domestic_21_100"], "domestic_top20_vs_21_100_over_time.png", "Mean domestic gross"),
        (["median_domestic_top20", "median_domestic_21_100"], "domestic_median_top20_vs_21_100_over_time.png", "Median domestic gross"),
        (["mean_polkw_top20", "mean_polkw_21_100"], "polkw_mean_top20_vs_21_100_over_time.png", "Political keywords per film"),
        (["gap_polkw"], "polkw_gap_top20_minus_21_100.png", "Gap: political keywords"),
        (["mean_polshare_top20", "mean_polshare_21_100"], "polshare_mean_top20_vs_21_100_over_time.png", "Political share of keywords"),
        (["gap_polshare"], "polshare_gap_top20_minus_21_100.png", "Gap: political share"),
        (["mean_totalkw_top20", "mean_totalkw_21_100"], "totalkw_mean_top20_vs_21_100_over_time.png", "Total keywords per film"),
    ]
    for cols, fname, title in plot_specs:
        fig, ax = plt.subplots(figsize=(10, 5))
        for c in cols:
            ax.plot(tier_df["year"], tier_df[c], label=c)
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.5)
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
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "genre_bucket_polshare_gap.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        for b in bucket_df["bucket"].unique():
            sub = bucket_df[bucket_df["bucket"] == b]
            ax.plot(sub["year"], sub["gap_polkw"], label=b)
        ax.set_title("Genre bucket gap (political keywords)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Gap")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "genre_bucket_polkw_gap.png", dpi=150)
        plt.close(fig)

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
        ax.set_ylabel("Share")
        ax.set_title("Pooled genre composition: Top20 vs 21-100")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "pooled_genre_shares_top20_vs_21_100_clean.png", dpi=150)
        plt.close(fig)

    if group_df is not None and not group_df.empty:
        pooled_group = group_df.groupby("group")[["count_top20", "count_21_100"]].sum().reset_index()
        total_top = pooled_group["count_top20"].sum()
        total_rest = pooled_group["count_21_100"].sum()
        pooled_group["share_top20"] = pooled_group["count_top20"] / total_top if total_top else np.nan
        pooled_group["share_21_100"] = pooled_group["count_21_100"] / total_rest if total_rest else np.nan
        fig, ax = plt.subplots(figsize=(8, 5))
        idx = np.arange(len(pooled_group))
        width = 0.35
        ax.bar(idx - width / 2, pooled_group["share_top20"], width, label="Top20")
        ax.bar(idx + width / 2, pooled_group["share_21_100"], width, label="21-100")
        ax.set_xticks(idx)
        ax.set_xticklabels([DISPLAY_NAMES.get(g, g) for g in pooled_group["group"]], rotation=45, ha="right")
        ax.set_ylabel("Share of political keywords")
        ax.set_title("Pooled political group composition (Top20 vs 21-100)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "pooled_pol_groups_top20_vs_21_100.png", dpi=150)
        plt.close(fig)

        for tier_col, fname, title in [
            ("share_top20", "pol_group_shares_over_time_top20.png", "Political group shares over time (Top20)"),
            ("share_21_100", "pol_group_shares_over_time_21_100.png", "Political group shares over time (21-100)"),
        ]:
            fig, ax = plt.subplots(figsize=(10, 5))
            for g in group_df["group"].unique():
                sub = group_df[group_df["group"] == g]
                ax.plot(sub["year"], sub[tier_col], label=DISPLAY_NAMES.get(g, g))
            ax.set_title(title)
            ax.set_xlabel("Year")
            ax.set_ylabel("Share")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            fig.tight_layout()
            fig.savefig(outdir / fname, dpi=150)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        for g in group_df["group"].unique():
            sub = group_df[group_df["group"] == g]
            ax.plot(sub["year"], sub["gap_share"], label=DISPLAY_NAMES.get(g, g))
        ax.set_title("Political group share gaps (Top20 - 21-100)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Gap")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "pol_group_share_gaps_over_time.png", dpi=150)
        plt.close(fig)


# ---------- main ----------
def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # TMDb load and filter
    tmdb = pd.read_csv(args.tmdb_csv)
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

    # Mojo rankings
    ranks_df = build_rankings(Path(args.mojo_csv), args)
    ranks_df.to_csv(outdir / "us_domestic_rankings_top100.csv", index=False)

    # Merge
    tmdb_subset = tmdb[["tmdb_id", "year", "title_norm", "title", kw_col] + ([genre_col] if genre_col else [])]
    matched_df, diag_df = merge_rankings(ranks_df, tmdb_subset, args)
    diag_df.to_csv(outdir / "merge_diagnostics.csv", index=False)

    # Metrics
    metrics_df = matched_df.rename(columns={kw_col: "keywords"})
    if genre_col:
        metrics_df = metrics_df.rename(columns={genre_col: "genres"})
    metrics_df = compute_movie_metrics(metrics_df, args)

    # Summaries
    tier_df_raw, bucket_df_raw, pooled_df, group_df_raw = summarize_yearly(metrics_df, args)

    # Year quality filtering
    tier_df, excluded_df = quality_filter(tier_df_raw, args)
    excluded_df.to_csv(outdir / "excluded_years.csv", index=False)
    tier_df.to_csv(outdir / "yearly_tier_summary_us_mojo_clean.csv", index=False)

    # Filter bucket and pooled to kept years
    bucket_df = bucket_df_raw[bucket_df_raw["year"].isin(tier_df["year"])]
    pooled_df = pooled_df.copy()  # already pooled; keep as is
    bucket_df.to_csv(outdir / "yearly_genre_bucket_summary_us_mojo_clean.csv", index=False)
    group_df = group_df_raw[group_df_raw["year"].isin(tier_df["year"])]

    # Plots (clean years only)
    make_plots(tier_df, bucket_df, pooled_df, outdir, group_df)

    # Group shares outputs
    group_df.to_csv(outdir / "yearly_political_group_shares_us_mojo_clean.csv", index=False)
    pooled_group_rows = []
    for gname in POLITICAL_PATTERNS.keys():
        c_top = group_df[group_df["group"] == gname]["count_top20"].sum()
        c_rest = group_df[group_df["group"] == gname]["count_21_100"].sum()
        pol_top = group_df[group_df["group"] == gname]["count_top20"].sum()  # already c_top
        pol_rest = group_df[group_df["group"] == gname]["count_21_100"].sum()
        total_top = group_df[group_df["group"] == gname]["count_top20"].sum()
        total_rest = group_df[group_df["group"] == gname]["count_21_100"].sum()
        total_pol_top = group_df["count_top20"].sum()
        total_pol_rest = group_df["count_21_100"].sum()
        pooled_group_rows.append(
            {
                "group": gname,
                "pooled_count_top20": c_top,
                "pooled_count_21_100": c_rest,
                "pooled_share_top20": c_top / total_pol_top if total_pol_top else np.nan,
                "pooled_share_21_100": c_rest / total_pol_rest if total_pol_rest else np.nan,
                "pooled_gap_share": (c_top / total_pol_top if total_pol_top else np.nan) - (c_rest / total_pol_rest if total_pol_rest else np.nan),
            }
        )
    pd.DataFrame(pooled_group_rows).to_csv(outdir / "pooled_political_group_shares_us_mojo_clean.csv", index=False)

    # Console summary
    kept_years = tier_df["year"].tolist()
    print(f"Kept years ({len(kept_years)}): {kept_years[:5]} ... {kept_years[-5:] if kept_years else []}")
    if not excluded_df.empty:
        print("Excluded years reasons:")
        print(excluded_df["reason"].value_counts())
    print(f"Avg gaps (kept years): polshare={tier_df['gap_polshare'].mean():.3f}, polkw={tier_df['gap_polkw'].mean():.3f}, totalkw={tier_df['gap_totalkw'].mean():.3f}")
    domestic_diff = (tier_df["mean_domestic_top20"] - tier_df["mean_domestic_21_100"]).mean()
    print(f"Mean domestic difference (Top20-21_100): {domestic_diff:,.0f}")


if __name__ == "__main__":
    main()
