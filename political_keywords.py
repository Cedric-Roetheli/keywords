"""
Extract TMDb keywords, count frequencies, and flag political-related keywords.

Supports:
- Local CSV loading or kagglehub download (largest CSV).
- Streaming counts with chunked pandas reads.
- Rule-based political filtering via transparent regex patterns.
- Optional zero-shot classification refinement.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

BAD_TOKENS = {"<na>", "nan", "none", ""}

POLITICAL_PATTERNS = {
    # Institutions / elections / law
    "institutions_elections_law": re.compile(
        r"(^|_)(government|parliament|congress|senate|president|prime_minister|minister|election|vote|voting|campaign|politic(s|al)?|party|constitution|democracy|dictator(ship)?|regime|state|bureaucracy|public_policy|law|legal|court|judge|trial|supreme_court)($|_)",
        re.IGNORECASE
    ),

    # War / security / intelligence / surveillance
    "war_security_intel": re.compile(
        r"(^|_)(war|world_war_(i|ii)|cold_war|army|military|soldier(s)?|veteran(s)?|battle|combat|invasion|occupation|terror(ism|ist)?|insurgent(s|cy)?|guerrilla|hostage|spy|espionage|cia|fbi|kgb|intelligence|surveillance|wiretap|secret_service|nuclear|missile(s)?|chemical_weapon(s)?|bioweapon(s)?)($|_)",
        re.IGNORECASE
    ),

    # Economy / finance / crisis
    "economy_finance_crisis": re.compile(
        r"(^|_)(econom(y|ic|ics)?|finance|financial|bank(s|ing)?|wall_street|stock_market|hedge_fund(s)?|invest(ment|or|ing)?|debt|credit|loan(s)?|mortgage(s)?|foreclosure|recession|depression|crisis|inflation|unemployment|austerity|bailout|budget|tax(es|ation)?|privatiz(e|ation)|nationaliz(e|ation))($|_)",
        re.IGNORECASE
    ),

    # Labor / collective action / political conflict
    "labor_collective_action": re.compile(
        r"(^|_)(labor|labour|worker(s)?|working_class|union(s)?|strike|protest|demonstration|riot(s)?|revolution|uprising|insurrection|coup|general_strike|activism|activist(s)?)($|_)",
        re.IGNORECASE
    ),

    # Inequality / corruption / elites
    "inequality_corruption_elites": re.compile(
        r"(^|_)(corruption|bribe(ry)?|embezzle(ment)?|scandal|fraud|money_launder(ing)?|oligarch(s|y)?|inequal(ity|ities)|poverty|class_warfare|social_class|the_rich|billionaire(s)?|capitalis(m|t)|communis(m|t)|socialis(m|t)|corporate|corporation(s)?|big_business|monopoly|greed)($|_)",
        re.IGNORECASE
    ),

    # Migration / policing / civil rights / discrimination
    "migration_police_civilrights": re.compile(
        r"(^|_)(immigra(tion|nt|nts)|migrant(s)?|refugee(s)?|asylum|border|deport(ation)?|citizenship|civil_rights|human_rights|discriminat(e|ion)|racis(m|t)|apartheid|segregat(e|ion)|police|policing|cop(s)?|law_enforcement|prison|incarcerat(e|ion)|surveillance_state)($|_)",
        re.IGNORECASE
    ),
}


def political_rule(keyword: str) -> Tuple[bool, Optional[str]]:
    """Return (is_political, matched_group) using the predefined regex patterns."""
    if not keyword:
        return False, None
    kw = keyword.strip().lower()
    for group, pat in POLITICAL_PATTERNS.items():
        if pat.search(kw):
            return True, group
    return False, None


KEYWORD_COLUMNS = ["keywords", "keyword_names", "Keywords", "tmdb_keywords", "keyword"]


@dataclass
class DatasetInfo:
    slug: str
    paths: List[str]
    root_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract TMDb keywords and classify political-related terms.")
    parser.add_argument("--csv", default=None, help="Path to local TMDb CSV")
    parser.add_argument("--use-kagglehub", action="store_true", help="Download via kagglehub instead of local CSV")
    parser.add_argument("--kaggle-slug", default="asaniczka/tmdb-movies-dataset-2023-930k-movies")
    parser.add_argument("--outdir", default="./outputs_political_keywords")
    parser.add_argument("--kw-col", default=None, help="Keyword column name (auto-detect if missing)")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--coverage", type=float, default=0.95, help="Coverage share for classification subset")
    parser.add_argument("--min-count", type=int, default=5, help="Drop keywords below this count")
    parser.add_argument("--min-vote-count", type=float, default=50, help="Filter: vote_count >= this (if column exists)")
    parser.add_argument("--min-runtime", type=float, default=40, help="Filter: runtime >= this (if column exists)")
    parser.add_argument("--min-popularity", type=float, default=None, help="Filter: popularity >= this (if column exists)")
    parser.add_argument("--run-zeroshot", action="store_true", help="Run zero-shot classifier")
    parser.add_argument("--zeroshot-model", default="facebook/bart-large-mnli")
    parser.add_argument("--device", type=int, default=-1, help="-1 CPU, >=0 GPU device id")
    parser.add_argument("--max-keywords", type=int, default=None, help="Optional cap on keywords to classify")
    return parser.parse_args()


# ----------------------------
# File handling
# ----------------------------
def auto_kaggle_file(slug: str) -> DatasetInfo:
    import kagglehub
    root = Path(kagglehub.dataset_download(slug))
    records = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not name.lower().endswith(".csv"):
                continue
            full = Path(dirpath) / name
            rel = full.relative_to(root)
            try:
                size = full.stat().st_size
            except OSError:
                size = 0
            records.append((str(rel), size))
    if not records:
        raise FileNotFoundError(f"No CSV files found in dataset {slug}")
    chosen = max(records, key=lambda x: x[1])
    return DatasetInfo(slug=slug, paths=[chosen[0]], root_dir=root)


def load_iter(csv_path: Path, chunksize: int, usecols: List[str]) -> Iterable[pd.DataFrame]:
    return pd.read_csv(csv_path, chunksize=chunksize, usecols=usecols, dtype="string", low_memory=False)


def detect_kw_column(columns: List[str], provided: Optional[str]) -> str:
    if provided and provided in columns:
        return provided
    for cand in KEYWORD_COLUMNS:
        if cand in columns:
            return cand
    raise ValueError(f"No keyword column found. Available: {columns}")


# ----------------------------
# Keyword parsing
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
        # Try JSON and literal
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


# ----------------------------
# Core logic
# ----------------------------
def build_keyword_counts(reader: Iterable[pd.DataFrame], kw_col: str, args: argparse.Namespace) -> Tuple[Counter, int]:
    counter = Counter()
    parse_failures = 0
    for chunk in reader:
        if kw_col not in chunk.columns:
            raise ValueError(f"Keyword column {kw_col} missing in chunk")
        # Apply filters
        df = chunk
        if "adult" in df.columns:
            df = df[df["adult"].str.lower().isin(["false", "f", "0", "no", ""]) | df["adult"].isna()]
        if "vote_count" in df.columns and args.min_vote_count is not None:
            vc = pd.to_numeric(df["vote_count"], errors="coerce")
            df = df[vc >= args.min_vote_count]
        if "runtime" in df.columns and args.min_runtime is not None:
            rt = pd.to_numeric(df["runtime"], errors="coerce")
            df = df[rt >= args.min_runtime]
        if "popularity" in df.columns and args.min_popularity is not None:
            pop = pd.to_numeric(df["popularity"], errors="coerce")
            df = df[pop >= args.min_popularity]
        for val in df[kw_col]:
            try:
                kws = parse_keywords(val)
            except Exception:
                parse_failures += 1
                continue
            counter.update(kws)
    return counter, parse_failures


def build_cumshare_df(counter: Counter) -> pd.DataFrame:
    items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(items, columns=["keyword", "count"])
    df["cum_count"] = df["count"].cumsum()
    total = df["count"].sum()
    df["cum_share"] = df["cum_count"] / total if total else 0
    return df


def select_candidates(df: pd.DataFrame, coverage: float, min_count: int, max_keywords: Optional[int]) -> pd.DataFrame:
    filtered = df[df["count"] >= min_count].copy()
    filtered = filtered[filtered["cum_share"] <= coverage]
    if max_keywords is not None:
        filtered = filtered.head(max_keywords)
    return filtered


def classify_rule(df: pd.DataFrame, counter: Counter) -> pd.DataFrame:
    rows = []
    for kw in df["keyword"]:
        is_pol, rule = political_rule(kw)
        rows.append({"keyword": kw, "count": counter[kw], "is_political": is_pol, "matched_rule": rule})
    return pd.DataFrame(rows)


def run_zero_shot(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    from transformers import pipeline

    clf = pipeline("zero-shot-classification", model=args.zeroshot_model, device=args.device)
    labels = ["political economy / government", "not political"]
    records = []
    for _, row in df.iterrows():
        kw = row["keyword"].replace("_", " ")
        res = clf(kw, labels)
        label = res["labels"][0]
        score = float(res["scores"][0])
        zs_is_pol = label.startswith("political")
        records.append(
            {
                "keyword": row["keyword"],
                "count": row["count"],
                "rule_is_political": row["is_political"],
                "rule_matched_rule": row["matched_rule"],
                "zs_label": label,
                "zs_score": score,
                "zs_is_political": zs_is_pol,
            }
        )
    return pd.DataFrame(records)


def summarize_mass(counter: Counter, keywords: Iterable[str]) -> int:
    return sum(counter.get(k, 0) for k in keywords)


def plot_top(df: pd.DataFrame, title: str, outpath: Path, topn: int = 20) -> None:
    subset = df[df["is_political"]].sort_values("count", ascending=False).head(topn)
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(subset) + 2))
    subset = subset.iloc[::-1]
    ax.barh(subset["keyword"], subset["count"], color="tab:blue")
    ax.set_xlabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def write_report(outdir: Path, summary: Dict[str, str]) -> None:
    lines = ["# Political Keyword Extraction Report", ""]
    for k, v in summary.items():
        lines.append(f"- **{k}**: {v}")
    (outdir / "report.md").write_text("\n".join(lines))


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data path and column
    if args.use_kagglehub:
        info = auto_kaggle_file(args.kaggle_slug)
        import kagglehub
        csv_path = info.root_dir / info.paths[0]
    else:
        if not args.csv:
            raise ValueError("Provide --csv or use --use-kagglehub")
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} not found")

    # Detect keyword column
    head = pd.read_csv(csv_path, nrows=0)
    kw_col = detect_kw_column(head.columns.tolist(), args.kw_col)
    filter_cols = []
    for c in ["adult", "vote_count", "runtime", "popularity"]:
        if c in head.columns:
            filter_cols.append(c)
    usecols = list(dict.fromkeys([kw_col] + filter_cols))

    reader = load_iter(csv_path, args.chunksize, usecols)
    counter, parse_failures = build_keyword_counts(reader, kw_col, args)

    vocab_df = build_cumshare_df(counter)
    vocab_df.to_csv(outdir / "unique_keywords.csv", index=False, columns=["keyword", "count"])
    vocab_df.to_csv(outdir / "unique_keywords_with_cumshare.csv", index=False)

    total_occ = vocab_df["count"].sum()
    cov_stats = {}
    for level in [0.9, 0.95, 0.99]:
        cov_stats[level] = (vocab_df[vocab_df["cum_share"] <= level].shape[0])

    candidates = select_candidates(vocab_df, args.coverage, args.min_count, args.max_keywords)
    candidates.to_csv(outdir / "candidate_keywords.csv", index=False)

    rule_df = classify_rule(candidates, counter)
    rule_df.to_csv(outdir / "political_keywords_rulebased.csv", index=False)

    zs_df = None
    if args.run_zeroshot:
        zs_df = run_zero_shot(rule_df, args)
        zs_df.to_csv(outdir / "political_keywords_zeroshot.csv", index=False)
        strict = zs_df[(zs_df["rule_is_political"]) & (zs_df["zs_is_political"])].copy()
        broad = zs_df[(zs_df["rule_is_political"]) | (zs_df["zs_is_political"])].copy()
        strict.to_csv(outdir / "final_political_keywords_strict.csv", index=False)
        broad.to_csv(outdir / "final_political_keywords_broad.csv", index=False)
        plot_top(strict.rename(columns={"rule_is_political": "is_political"}), "Top political keywords (strict)", outdir / "top_political_keywords_strict.png")
    else:
        # If no zeroshot, final sets equal to rule-based
        rule_df.to_csv(outdir / "final_political_keywords_strict.csv", index=False)
        rule_df.to_csv(outdir / "final_political_keywords_broad.csv", index=False)
        zs_df = rule_df
        plot_top(rule_df, "Top political keywords (rule-based)", outdir / "top_political_keywords_rulebased.png")

    mass_rule = summarize_mass(counter, rule_df[rule_df["is_political"]]["keyword"])
    mass_total = sum(counter.values())
    summary = {
        "Total unique keywords": f"{len(counter):,}",
        "Total keyword occurrences": f"{mass_total:,}",
        "Parse failures": f"{parse_failures:,}",
        "Keywords covering 90%": cov_stats[0.9],
        "Keywords covering 95%": cov_stats[0.95],
        "Keywords covering 99%": cov_stats[0.99],
        "Rule-based political keywords": f"{rule_df[rule_df['is_political']].shape[0]:,}",
        "Rule-based mass share": f"{mass_rule / mass_total:.4f}" if mass_total else "NA",
    }
    if args.run_zeroshot and zs_df is not None:
        strict = zs_df[(zs_df["rule_is_political"]) & (zs_df["zs_is_political"])]
        broad = zs_df[(zs_df["rule_is_political"]) | (zs_df["zs_is_political"])]
        summary["Strict political keywords"] = f"{len(strict):,}"
        summary["Broad political keywords"] = f"{len(broad):,}"

    write_report(outdir, summary)

    # Diagnostics
    print("Summary stats:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    top_pol = rule_df[rule_df["is_political"]].sort_values("count", ascending=False).head(30)
    if not top_pol.empty:
        print("Top political keywords (rule-based):")
        for _, r in top_pol.iterrows():
            print(f"  {r['keyword']}: {r['count']}")


if __name__ == "__main__":
    main()
