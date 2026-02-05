"""
Generate thesis-ready overview plots for political keywords (pooled Top-100 US domestic box office).

Data sources (expected):
- outputs_us_market_mojo_clean_boxoffice/yearly_tier_summary_us_mojo_clean.csv
- outputs_us_market_mojo_clean_boxoffice/yearly_political_group_shares_us_mojo_clean.csv

Outputs:
- outputs_us_market_mojo_clean_boxoffice/plots_overview/
    overview_prevalence_political_any_top100.png
    overview_intensity_polshare_top100.png
    overview_heatmap_group_shares_top100.png
    overview_dualaxis_prevalence_and_intensity_top100.png (optional, created if both series are present)
    overview_top100_by_year.csv (year-level prevalence, polshare, group shares)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = BASE_DIR / "outputs_us_market_mojo_clean_boxoffice"
PLOT_DIR = DEFAULT_OUTDIR / "plots_overview"

TIER_SUMMARY_CANDIDATES = [
    "yearly_tier_summary_us_mojo_clean.csv",
    "yearly_tier_summary_us_mojo.csv",
]

GROUP_SUMMARY_CANDIDATES = [
    "yearly_political_group_shares_us_mojo_clean.csv",
    "yearly_political_group_shares_us_mojo.csv",
]

GROUP_ORDER = [
    "war_security_intel",
    "institutions_elections_law",
    "economy_finance_crisis",
    "migration_police_civilrights",
    "labor_collective_action",
    "inequality_corruption_elites",
]

GROUP_LABELS = {
    "war_security_intel": "War / security / intelligence",
    "institutions_elections_law": "Institutions / elections / law",
    "economy_finance_crisis": "Economy / finance / crisis",
    "migration_police_civilrights": "Migration / policing / civil rights",
    "labor_collective_action": "Labor / collective action",
    "inequality_corruption_elites": "Inequality / corruption / elites",
}


def find_first_existing(base: Path, candidates: List[str]) -> Path:
    for name in candidates:
        path = base / name
        if path.exists():
            return path
    raise FileNotFoundError(f"None of the candidate files found under {base}: {candidates}")


def load_tier_summary() -> pd.DataFrame:
    path = find_first_existing(DEFAULT_OUTDIR, TIER_SUMMARY_CANDIDATES)
    df = pd.read_csv(path)
    print(f"Using tier summary: {path}")
    required = [
        "year",
        "n_top20_matched",
        "n_21_100_matched",
        "share_any_top20",
        "share_any_21_100",
        "mean_polshare_top20",
        "mean_polshare_21_100",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Tier summary missing columns: {missing}")
    return df


def load_group_summary() -> pd.DataFrame:
    path = find_first_existing(DEFAULT_OUTDIR, GROUP_SUMMARY_CANDIDATES)
    df = pd.read_csv(path)
    print(f"Using group summary: {path}")
    required = ["year", "group", "count_top20", "count_21_100"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Group summary missing columns: {missing}")
    df = df[df["group"].isin(GROUP_ORDER)]
    df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)
    return df


def pooled_top100_from_tiers(tier_df: pd.DataFrame) -> pd.DataFrame:
    df = tier_df.copy()
    df["year"] = df["year"].astype(int)
    df = df[df["year"] != 2020]  # drop COVID anomaly
    df = df.sort_values("year")

    # compute pooled sample size and weighted prevalence/intensity
    df["n_total"] = df["n_top20_matched"] + df["n_21_100_matched"]
    df = df[df["n_total"] > 0]

    df["prevalence_any"] = (
        df["share_any_top20"] * df["n_top20_matched"]
        + df["share_any_21_100"] * df["n_21_100_matched"]
    ) / df["n_total"]

    df["mean_polshare"] = (
        df["mean_polshare_top20"] * df["n_top20_matched"]
        + df["mean_polshare_21_100"] * df["n_21_100_matched"]
    ) / df["n_total"]

    return df[["year", "prevalence_any", "mean_polshare", "n_total"]]


def pooled_group_shares(group_df: pd.DataFrame) -> pd.DataFrame:
    # combine counts across tiers per year
    agg = (
        group_df.groupby(["year", "group"])[["count_top20", "count_21_100"]]
        .sum()
        .reset_index()
    )
    agg["total_count"] = agg["count_top20"] + agg["count_21_100"]

    totals = agg.groupby("year")["total_count"].sum().rename("year_total")
    agg = agg.join(totals, on="year")
    agg = agg[agg["year"] != 2020]
    agg["share_total"] = agg["total_count"] / agg["year_total"].replace({0: np.nan})
    return agg[["year", "group", "share_total"]]


def make_line_plot(df: pd.DataFrame, ycol: str, ylabel: str, title: str, subtitle: str, outpath: Path, rolling: int = 0):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    years = df["year"]
    vals = df[ycol] * 100.0
    ax.plot(years, vals, marker="o", linewidth=1.8, label="Annual")
    if rolling and rolling > 1:
        roll = vals.rolling(rolling, center=True, min_periods=1).mean()
        ax.plot(years, roll, linestyle="--", linewidth=1.4, alpha=0.8, label=f"{rolling}-yr rolling")
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xticks(np.arange(years.min(), years.max() + 1, 5))
    if subtitle:
        ax.text(0.99, -0.18, subtitle, transform=ax.transAxes, ha="right", va="top", fontsize=9, color="dimgray")
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def make_dual_axis(df: pd.DataFrame, outpath: Path):
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    years = df["year"]
    prev = df["prevalence_any"] * 100.0
    polshare = df["mean_polshare"] * 100.0

    l1 = ax1.plot(years, prev, color="#1f77b4", marker="o", linewidth=1.8, label="Prevalence (≥1 political keyword)")
    ax1.set_ylabel("Prevalence (% of titles)", color="#1f77b4", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xlabel("Year", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    l2 = ax2.plot(years, polshare, color="#d62728", marker="s", linewidth=1.8, label="Mean polshare")
    ax2.set_ylabel("Mean polshare (%)", color="#d62728", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines = l1 + l2
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, fontsize=10, frameon=False, loc="upper left")
    ax1.set_title("Prevalence and intensity of political keywords (Top-100 pooled)", fontsize=15)
    ax1.text(
        0.99,
        -0.18,
        "polshare = political keywords / total keywords (unique, per film). 2020 excluded.",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="dimgray",
    )
    ax1.set_xticks(np.arange(years.min(), years.max() + 1, 5))
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def make_heatmap(group_shares: pd.DataFrame, outpath: Path):
    pivot = group_shares.pivot(index="group", columns="year", values="share_total").reindex(GROUP_ORDER)
    years = pivot.columns.astype(int)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_yticks(range(len(GROUP_ORDER)))
    ax.set_yticklabels([GROUP_LABELS[g] for g in GROUP_ORDER])
    step = max(1, len(years) // 12)
    xticks_idx = list(range(0, len(years), step))
    ax.set_xticks(xticks_idx)
    ax.set_xticklabels([years[i] for i in xticks_idx], rotation=45, ha="right")
    ax.set_title("Political keyword group composition (Top-100 pooled)", fontsize=15)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Share of political keywords", fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def write_overview_csv(summary_df: pd.DataFrame, group_shares: pd.DataFrame, outpath: Path):
    wide_groups = group_shares.pivot(index="year", columns="group", values="share_total").reindex(columns=GROUP_ORDER)
    merged = summary_df.join(wide_groups, on="year")
    merged.to_csv(outpath, index=False)
    print(f"Wrote overview CSV: {outpath}")


def main():
    try:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        tier_df = load_tier_summary()
        group_df = load_group_summary()

        pooled = pooled_top100_from_tiers(tier_df)
        pooled_groups = pooled_group_shares(group_df)

        if pooled.empty:
            raise ValueError("No pooled Top-100 rows after filtering; cannot plot.")

        yr_min, yr_max = pooled["year"].min(), pooled["year"].max()
        print(f"Pooled Top-100 years plotted: {yr_min}-{yr_max} (2020 excluded)")

        # Plots 1 & 2
        make_line_plot(
            pooled,
            ycol="prevalence_any",
            ylabel="Share of titles with ≥1 political keyword (%)",
            title="Prevalence of political keywords among annual Top-100 films",
            subtitle="Share of titles with ≥1 political keyword (TMDb keywords). 2020 excluded.",
            outpath=PLOT_DIR / "overview_prevalence_political_any_top100.png",
            rolling=5,
        )
        make_line_plot(
            pooled,
            ycol="mean_polshare",
            ylabel="Mean polshare (%)",
            title="Intensity of political keywords among annual Top-100 films",
            subtitle="polshare = political keywords / total keywords (unique, per film). 2020 excluded.",
            outpath=PLOT_DIR / "overview_intensity_polshare_top100.png",
            rolling=0,
        )

        # Dual axis (optional)
        make_dual_axis(
            pooled,
            outpath=PLOT_DIR / "overview_dualaxis_prevalence_and_intensity_top100.png",
        )

        # Heatmap
        make_heatmap(
            pooled_groups,
            outpath=PLOT_DIR / "overview_heatmap_group_shares_top100.png",
        )

        # Overview CSV
        write_overview_csv(
            pooled[["year", "prevalence_any", "mean_polshare"]],
            pooled_groups,
            PLOT_DIR / "overview_top100_by_year.csv",
        )

        print("Overview plots written to:", PLOT_DIR)
    except Exception as exc:  # pragma: no cover
        print(f"Failed to generate overview plots: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
