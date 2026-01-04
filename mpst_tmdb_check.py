"""
Check coverage and imdb_id matches between MPST and TMDb Kaggle datasets using kagglehub.

Uses kagglehub.load_dataset with KaggleDatasetAdapter.PANDAS to load:
- MPST plot synopses dataset
- TMDb movie dataset

Outputs per-year counts for MPST and match statistics vs TMDb.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import kagglehub
from kagglehub import KaggleDatasetAdapter
from kagglehub import exceptions as kh_exceptions
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_MPST = "cryptexcode/mpst-movie-plot-synopses-with-tags"
DEFAULT_TMDB = "asaniczka/tmdb-movies-dataset-2023-930k-movies"


IMDB_CANDIDATES = ["imdb_id", "imdbId", "imdbid", "imdb"]
YEAR_CANDIDATES = ["release_year", "year"]
DATE_CANDIDATES = ["release_date", "released", "date"]
TMDB_YEAR_CANDIDATES = ["release_year", "year", "release_date"]


@dataclass
class DatasetInfo:
    slug: str
    paths: List[str]  # relative paths inside dataset
    root_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MPST and TMDb imdb_id coverage using kagglehub.")
    parser.add_argument("--mpst-slug", default=DEFAULT_MPST, help="MPST dataset slug")
    parser.add_argument("--tmdb-slug", default=DEFAULT_TMDB, help="TMDb dataset slug")
    parser.add_argument("--mpst-file", default=None, help="Optional MPST file path inside the dataset")
    parser.add_argument("--tmdb-file", default=None, help="Optional TMDb file path inside the dataset")
    parser.add_argument("--outdir", default="./outputs_mpst_check", help="Output directory")
    parser.add_argument("--year-min", type=int, default=None, help="Optional min year filter for MPST")
    parser.add_argument("--year-max", type=int, default=None, help="Optional max year filter for MPST")
    return parser.parse_args()


def list_dataset_files(root_dir: Path) -> List[Tuple[str, int]]:
    """List all files under root_dir, returning relative path and size."""
    records = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            full = Path(dirpath) / name
            rel = full.relative_to(root_dir)
            try:
                size = full.stat().st_size
            except OSError:
                size = 0
            records.append((str(rel), size))
    return records


def pick_file_path(slug: str, preferred: Optional[str]) -> DatasetInfo:
    """
    Pick file(s) to load for a dataset.
    If preferred is provided, use it (resolved relative to dataset root if needed).
    Otherwise, pick the largest file by priority: csv > parquet > json > tsv.
    If multiple files with the chosen extension exist, return all sorted by size (desc).
    """
    if preferred:
        candidate = Path(preferred)
        # If user supplied an existing local file, use it directly and bypass kagglehub.
        if candidate.exists():
            return DatasetInfo(slug=slug, paths=[str(candidate.resolve())], root_dir=candidate.parent)
    # Otherwise rely on kagglehub download
    root = Path(kagglehub.dataset_download(slug))
    if preferred:
        candidate = Path(preferred)
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"Preferred file {candidate} not found for dataset {slug}")
        try:
            rel = candidate.relative_to(root)
        except ValueError:
            rel = candidate
        return DatasetInfo(slug=slug, paths=[str(rel)], root_dir=root)

    records = list_dataset_files(root)
    # Drop obvious metadata
    records = [(p, s) for p, s in records if "partition.json" not in p and ".complete" not in p]
    if not records:
        # Force re-download if nothing useful
        kagglehub.dataset_download(slug, force_download=True)
        root = Path(kagglehub.dataset_download(slug))
        records = [(p, s) for p, s in list_dataset_files(root) if "partition.json" not in p and ".complete" not in p]
    if not records:
        raise FileNotFoundError(f"No files found in dataset {slug} at {root}")

    ext_priority = [".csv", ".parquet", ".json", ".tsv"]
    chosen_ext = None
    for ext in ext_priority:
        filtered = [(p, s) for p, s in records if p.lower().endswith(ext)]
        if filtered:
            chosen_ext = ext
            candidates = sorted(filtered, key=lambda x: x[1], reverse=True)
            break
    if not chosen_ext:
        # fallback to largest of all
        candidates = sorted(records, key=lambda x: x[1], reverse=True)

    paths = [p for p, _ in candidates]
    return DatasetInfo(slug=slug, paths=paths, root_dir=root)


def load_pandas_with_retry(slug: str, rel_path: str, pandas_kwargs: dict) -> pd.DataFrame:
    """Load a single file with retry on DataCorruptionError."""
    # If rel_path is absolute and exists locally, bypass kagglehub.
    if Path(rel_path).is_absolute() and Path(rel_path).exists():
        return pd.read_csv(rel_path, **{k: v for k, v in pandas_kwargs.items() if k != "usecols"})
    try:
        return kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, slug, rel_path, pandas_kwargs=pandas_kwargs)
    except TypeError:
        df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, slug, rel_path)
        if pandas_kwargs.get("usecols"):
            df = df[pandas_kwargs["usecols"]]
        return df
    except kh_exceptions.DataCorruptionError:
        print(f"Warning: data corruption detected for {slug}/{rel_path}, clearing cache and retrying...")
        clear_dataset_cache(slug)
        kagglehub.dataset_download(slug, force_download=True)
        return kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, slug, rel_path, pandas_kwargs=pandas_kwargs)


def clear_dataset_cache(slug: str) -> None:
    """Delete cached dataset files to avoid resume with corrupted partials."""
    cache_dir = Path.home() / ".cache" / "kagglehub" / "datasets"
    target = cache_dir / slug
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)


def load_dataset_auto(info: DatasetInfo, pandas_kwargs: Optional[dict] = None) -> pd.DataFrame:
    """
    Load one or multiple files from a dataset using kagglehub.load_dataset.
    If multiple files share identical columns, concatenate them; otherwise load only the largest.
    """
    pandas_kwargs = pandas_kwargs or {}
    loaded_frames = []
    base_columns = None
    for idx, rel_path in enumerate(info.paths):
        df = load_pandas_with_retry(info.slug, rel_path, pandas_kwargs)
        if base_columns is None:
            base_columns = list(df.columns)
            loaded_frames.append(df)
        else:
            if list(df.columns) == base_columns:
                loaded_frames.append(df)
            else:
                # Columns mismatch; keep only first (largest expected)
                if loaded_frames:
                    break
                loaded_frames.append(df)
                break
        # Only load multiple if columns match; otherwise stop
    if not loaded_frames:
        raise ValueError(f"Failed to load any files for dataset {info.slug}")
    if len(loaded_frames) == 1:
        return loaded_frames[0]
    return pd.concat(loaded_frames, ignore_index=True)


def normalize_imdb(series: pd.Series) -> Tuple[pd.Series, int]:
    """Normalize imdb ids and return cleaned series plus count of invalid values dropped."""
    invalid = 0
    cleaned = []
    for val in series:
        s = str(val).strip().lower()
        if not s or s in {"nan", "none"}:
            invalid += 1
            cleaned.append(pd.NA)
            continue
        s = s.replace(" ", "")
        if s.isdigit():
            s = f"tt{s}"
        if not s.startswith("tt") and s[0].isdigit():
            s = f"tt{s}"
        if re.match(r"^tt\d+$", s):
            cleaned.append(s)
        else:
            invalid += 1
            cleaned.append(pd.NA)
    return pd.Series(cleaned, dtype="string"), invalid


def extract_year_mpst(df: pd.DataFrame) -> Tuple[pd.Series, int]:
    """Extract release year from MPST dataframe (intrinsic only); return series and count of missing years."""
    year = pd.Series([pd.NA] * len(df), dtype="Int64")
    for col in YEAR_CANDIDATES:
        if col in df.columns:
            year = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            break
    else:
        for col in DATE_CANDIDATES:
            if col in df.columns:
                year_vals = df[col].astype(str).str.extract(r"(\d{4})")[0]
                year = pd.to_numeric(year_vals, errors="coerce").astype("Int64")
                break
        else:
            if "title" in df.columns:
                year_vals = df["title"].astype(str).str.extract(r"\((\d{4})\)")[0]
                year = pd.to_numeric(year_vals, errors="coerce").astype("Int64")
    missing_year = int(year.isna().sum())
    return year, missing_year


def extract_year_tmdb(df: pd.DataFrame) -> pd.Series:
    """Extract release year from TMDb dataframe."""
    year = pd.Series([pd.NA] * len(df), dtype="Int64")
    for col in TMDB_YEAR_CANDIDATES:
        if col in df.columns:
            if col in {"release_date", "released", "date"}:
                vals = df[col].astype(str).str.extract(r"(\d{4})")[0]
                year = pd.to_numeric(vals, errors="coerce").astype("Int64")
            else:
                year = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            break
    return year


def find_imdb_column(df: pd.DataFrame) -> Optional[str]:
    for col in IMDB_CANDIDATES:
        if col in df.columns:
            return col
    return None


def compute_tables(
    mpst_df: pd.DataFrame,
    tmdb_df: pd.DataFrame,
    year_min: Optional[int],
    year_max: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Compute per-year counts, match overall, and per-year match tables."""
    # Normalize imdb ids
    mpst_imdb_col = find_imdb_column(mpst_df)
    tmdb_imdb_col = find_imdb_column(tmdb_df)
    if not mpst_imdb_col or not tmdb_imdb_col:
        raise ValueError("imdb_id column not found in one of the datasets.")
    mpst_imdb_clean, mpst_invalid = normalize_imdb(mpst_df[mpst_imdb_col])
    tmdb_imdb_clean, tmdb_invalid = normalize_imdb(tmdb_df[tmdb_imdb_col])
    mpst_df = mpst_df.assign(imdb_norm=mpst_imdb_clean)
    tmdb_df = tmdb_df.assign(imdb_norm=tmdb_imdb_clean)
    mpst_df = mpst_df.dropna(subset=["imdb_norm"])
    tmdb_df = tmdb_df.dropna(subset=["imdb_norm"])

    # Extract year for MPST; if missing, map from TMDb by imdb_id
    mpst_year_intrinsic, missing_intrinsic = extract_year_mpst(mpst_df)
    tmdb_year = extract_year_tmdb(tmdb_df)
    tmdb_map = dict(zip(tmdb_df["imdb_norm"], tmdb_year))
    mapped_year = mpst_df["imdb_norm"].map(tmdb_map)
    combined_year = mpst_year_intrinsic.fillna(mapped_year)
    mapped_years = int(combined_year.notna().sum())
    missing_year = int(combined_year.isna().sum())
    mpst_df = mpst_df.assign(year=combined_year)
    if year_min is not None:
        mpst_df = mpst_df[mpst_df["year"].isna() | (mpst_df["year"] >= year_min)]
    if year_max is not None:
        mpst_df = mpst_df[mpst_df["year"].isna() | (mpst_df["year"] <= year_max)]

    # MPST per-year counts
    per_year = (
        mpst_df.dropna(subset=["year"])
        .groupby("year")
        .agg(n_movies=("imdb_norm", "count"), n_unique_imdb_id=("imdb_norm", "nunique"))
        .reset_index()
        .sort_values("year")
    )

    # Match stats
    mpst_unique = set(mpst_df["imdb_norm"].dropna().unique())
    tmdb_unique = set(tmdb_df["imdb_norm"].dropna().unique())
    matched_unique = mpst_unique & tmdb_unique

    overall = pd.DataFrame(
        [
            {
                "mpst_rows": len(mpst_df),
                "mpst_unique_imdb": len(mpst_unique),
                "tmdb_rows": len(tmdb_df),
                "tmdb_unique_imdb": len(tmdb_unique),
                "matched_unique_imdb": len(matched_unique),
                "match_rate_mpst_to_tmdb": len(matched_unique) / len(mpst_unique) if mpst_unique else 0.0,
                "match_rate_tmdb_to_mpst": len(matched_unique) / len(tmdb_unique) if tmdb_unique else 0.0,
            }
        ]
    )

    by_year_records = []
    for _, row in per_year.iterrows():
        yr = row["year"]
        mpst_year_ids = set(mpst_df.loc[mpst_df["year"] == yr, "imdb_norm"].unique())
        matched = mpst_year_ids & tmdb_unique
        by_year_records.append(
            {
                "year": int(yr),
                "mpst_unique_imdb": len(mpst_year_ids),
                "matched_unique_imdb": len(matched),
                "match_rate": len(matched) / len(mpst_year_ids) if mpst_year_ids else 0.0,
            }
        )
    if by_year_records:
        by_year = pd.DataFrame(by_year_records).sort_values("year")
    else:
        by_year = pd.DataFrame(columns=["year", "mpst_unique_imdb", "matched_unique_imdb", "match_rate"])

    meta = {
        "mpst_invalid_imdb": mpst_invalid,
        "tmdb_invalid_imdb": tmdb_invalid,
        "mpst_missing_year": missing_year,
        "mpst_with_year": mapped_years,
    }
    return per_year, overall, by_year, meta


def write_outputs(per_year: pd.DataFrame, overall: pd.DataFrame, by_year: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    per_year.to_csv(outdir / "mpst_movies_per_year.csv", index=False)
    overall.to_csv(outdir / "mpst_tmdb_match_overall.csv", index=False)
    by_year.to_csv(outdir / "mpst_tmdb_match_by_year.csv", index=False)


def plot_outputs(per_year: pd.DataFrame, by_year: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if not per_year.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(per_year["year"].astype(int), per_year["n_movies"], color="tab:blue")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of MPST movies")
        ax.set_title("MPST movies per year")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(outdir / "mpst_movies_per_year.png", dpi=150)
        plt.close(fig)

    if not by_year.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(by_year["year"], by_year["match_rate"], marker="o", color="tab:orange")
        ax.set_xlabel("Year")
        ax.set_ylabel("Match rate (MPST -> TMDb)")
        ax.set_title("MPST imdb_id match rate by year")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(outdir / "mpst_match_rate_by_year.png", dpi=150)
        plt.close(fig)


def print_summary(overall: pd.DataFrame, meta: dict) -> None:
    row = overall.iloc[0]
    print("Summary:")
    print(f"  MPST rows: {int(row['mpst_rows'])}, unique imdb: {int(row['mpst_unique_imdb'])}")
    print(f"  TMDb rows: {int(row['tmdb_rows'])}, unique imdb: {int(row['tmdb_unique_imdb'])}")
    print(f"  Matched unique imdb: {int(row['matched_unique_imdb'])}")
    print(f"  Match rate MPST -> TMDb: {row['match_rate_mpst_to_tmdb']:.4f}")
    print(f"  Match rate TMDb -> MPST: {row['match_rate_tmdb_to_mpst']:.4f}")
    print(f"  Invalid imdb dropped (MPST): {meta.get('mpst_invalid_imdb', 0)}")
    print(f"  Invalid imdb dropped (TMDb): {meta.get('tmdb_invalid_imdb', 0)}")
    print(f"  MPST rows with year: {meta.get('mpst_with_year', 0)}; missing year: {meta.get('mpst_missing_year', 0)}")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)

    # Load MPST
    mpst_info = pick_file_path(args.mpst_slug, args.mpst_file)
    mpst_df = load_dataset_auto(
        mpst_info,
        pandas_kwargs={"dtype": "string"},
    )

    # Load TMDb (only imdb-related columns)
    tmdb_info = pick_file_path(args.tmdb_slug, args.tmdb_file)
    tmdb_df = load_dataset_auto(
        tmdb_info,
        pandas_kwargs={"usecols": list(dict.fromkeys(IMDB_CANDIDATES + TMDB_YEAR_CANDIDATES)), "dtype": "string"},
    )

    per_year, overall, by_year, meta = compute_tables(mpst_df, tmdb_df, args.year_min, args.year_max)
    write_outputs(per_year, overall, by_year, outdir)
    plot_outputs(per_year, by_year, outdir)
    print_summary(overall, meta)
    print(f"Outputs written to {outdir}")


if __name__ == "__main__":
    main()
