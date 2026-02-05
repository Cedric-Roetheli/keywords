"""
Generate additional tier comparison plots:
- Distribution of yearly mean polshare by tier (Top20 vs 21-100).
- Pre/post-2002 comparison of the yearly polshare gap (Top20 - 21-100).

Inputs (expected):
- outputs_us_market_mojo_clean_boxoffice/yearly_tier_summary_us_mojo_clean.csv
  Required columns: year, mean_polshare_top20, mean_polshare_21_100 (or similar).

Outputs:
- outputs_us_market_mojo_clean_boxoffice/plots_tier_extra/
    tier_polshare_distribution_yearly_means.png
    tier_polshare_gap_pre_vs_post.png

Notes:
- Excludes year 2020.
- PRE: 1985-2002, POST: 2003-max_year.
- Polshare is plotted in percent; converts from fraction if needed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SUMMARY_PATH = BASE_DIR / "outputs_us_market_mojo_clean_boxoffice" / "yearly_tier_summary_us_mojo_clean.csv"
OUTDIR = BASE_DIR / "outputs_us_market_mojo_clean_boxoffice" / "plots_tier_extra"

PRE_START, PRE_END = 1985, 2002
POST_START = 2003


def bootstrap_ci(vals: np.ndarray, n_boot: int = 10000, alpha: float = 0.05) -> Tuple[float, float]:
    if len(vals) == 0:
        return np.nan, np.nan
    boots = np.random.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    low, high = np.percentile(boots, [100 * (alpha / 2), 100 * (1 - alpha / 2)])
    return float(low), float(high)


def normalize_percent(series: pd.Series) -> pd.Series:
    med = series.dropna().median()
    if med is None or np.isnan(med):
        return series
    if med <= 1.0:
        return series * 100.0
    return series


def load_summary() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Summary file not found: {SUMMARY_PATH}")
    df = pd.read_csv(SUMMARY_PATH)
    required = ["year", "mean_polshare_top20", "mean_polshare_21_100"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing in summary: {missing}")
    return df


def make_distribution_plot(df: pd.DataFrame, outpath: Path):
    fig, ax = plt.subplots(figsize=(7.2, 5))
    data = [df["polshare_top20"], df["polshare_21_100"]]
    tiers = ["Top 20", "21-100"]
    ns = [len(df["polshare_top20"].dropna()), len(df["polshare_21_100"].dropna())]
    labels = [f"{t} (n={n})" for t, n in zip(tiers, ns)]
    positions = [0, 1]

    ax.boxplot(
        data,
        positions=positions,
        widths=0.4,
        patch_artist=True,
        boxprops=dict(facecolor="#d9e6f2"),
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="gray"),
        capprops=dict(color="gray"),
    )

    rng = np.random.default_rng(123)
    for pos, vals in zip(positions, data):
        if len(vals) == 0:
            continue
        jitter = rng.normal(loc=pos, scale=0.03, size=len(vals))
        ax.scatter(jitter, vals, s=24, alpha=0.7, color="#1f77b4", edgecolor="none")

    for pos, vals in zip(positions, data):
        if len(vals) == 0:
            continue
        mean_val = np.mean(vals)
        low, high = bootstrap_ci(np.array(vals))
        ax.scatter(pos, mean_val, color="#d62728", marker="D", s=40, zorder=4)
        ax.vlines(pos, low, high, colors="#d62728", linewidth=1.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean polshare (%)", fontsize=12)
    ax.set_title("Political keyword intensity by performance tier (yearly means)", fontsize=15)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.text(
        0.99,
        -0.18,
        "Yearly Top-100 pooled within tier; 2020 excluded.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="dimgray",
    )
    ax.text(
        0.99,
        -0.26,
        "Points = yearly means",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="dimgray",
    )
    fig.tight_layout(pad=1.2)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def make_gap_plot(df: pd.DataFrame, outpath: Path, max_year: int):
    gap = df["polshare_top20"] - df["polshare_21_100"]
    df_gap = pd.DataFrame({"year": df["year"], "gap": gap})
    pre = df_gap[(df_gap["year"] >= PRE_START) & (df_gap["year"] <= PRE_END)]["gap"].dropna()
    post = df_gap[(df_gap["year"] >= POST_START) & (df_gap["year"] <= max_year)]["gap"].dropna()

    fig, ax = plt.subplots(figsize=(7.2, 5))
    data = [pre, post]
    labels = [f"{PRE_START}-{PRE_END} (n={len(pre)})", f"{POST_START}-{max_year} (n={len(post)})"]
    positions = [0, 1]

    ax.boxplot(
        data,
        positions=positions,
        widths=0.4,
        patch_artist=True,
        boxprops=dict(facecolor="#e7d9f2"),
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="gray"),
        capprops=dict(color="gray"),
    )

    rng = np.random.default_rng(123)
    for pos, vals in zip(positions, data):
        if len(vals) == 0:
            continue
        jitter = rng.normal(loc=pos, scale=0.03, size=len(vals))
        ax.scatter(jitter, vals, s=24, alpha=0.7, color="#1f77b4", edgecolor="none")

    for pos, vals in zip(positions, data):
        if len(vals) == 0:
            continue
        mean_val = np.mean(vals)
        low, high = bootstrap_ci(np.array(vals))
        ax.scatter(pos, mean_val, color="#d62728", marker="D", s=40, zorder=4)
        ax.vlines(pos, low, high, colors="#d62728", linewidth=1.6)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Gap in mean polshare (percentage points)", fontsize=12)
    ax.set_title("Tier gap in political keyword intensity before vs after 2002", fontsize=15)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.text(
        0.01,
        0.98,
        "Gap = mean polshare(Top 20) − mean polshare(21–100)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="dimgray",
    )
    ax.text(
        0.99,
        -0.18,
        "Gap computed from yearly mean polshare; 2020 excluded.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="dimgray",
    )
    fig.tight_layout(pad=1.2)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def main():
    try:
        df = load_summary()
        if "year" not in df.columns:
            raise ValueError("Missing 'year' column.")
        df["year"] = df["year"].astype(int)
        df = df[df["year"] != 2020]
        df = df.sort_values("year")
        max_year = df["year"].max()

        df["polshare_top20"] = normalize_percent(df["mean_polshare_top20"])
        df["polshare_21_100"] = normalize_percent(df["mean_polshare_21_100"])

        years_used = len(df)
        print(f"Year range used: {df['year'].min()}-{df['year'].max()} (2020 excluded), years={years_used}")

        OUTDIR.mkdir(parents=True, exist_ok=True)

        make_distribution_plot(
            df[["polshare_top20", "polshare_21_100"]],
            OUTDIR / "tier_polshare_distribution_yearly_means.png",
        )

        make_gap_plot(
            df[["year", "polshare_top20", "polshare_21_100"]],
            OUTDIR / "tier_polshare_gap_pre_vs_post.png",
            max_year=max_year,
        )

        print(f"Saved plots to {OUTDIR}")
    except Exception as exc:  # pragma: no cover
        print(f"Failed to generate tier extra plots: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
