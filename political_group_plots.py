"""
Visualize political group composition from yearly group share outputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GROUP_ORDER = [
    "war_security_intel",
    "institutions_elections_law",
    "economy_finance_crisis",
    "migration_police_civilrights",
    "labor_collective_action",
    "inequality_corruption_elites",
]

GROUP_LABELS = {
    "war_security_intel": "War/Security",
    "institutions_elections_law": "Institutions/Law",
    "economy_finance_crisis": "Economy/Crisis",
    "migration_police_civilrights": "Migration/CivilRights",
    "labor_collective_action": "Labor/Collective",
    "inequality_corruption_elites": "Inequality/Corruption",
}

# filled at runtime in main so every plot shares identical group colors
GROUP_TO_COLOR: dict[str, str] = {}


def build_group_colors() -> dict[str, str]:
    """
    Create a deterministic color mapping per group using matplotlib's default
    color cycle. Ensures consistent colors across all plots.
    """
    base_colors: List[str] = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not base_colors:
        base_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    return {g: base_colors[i % len(base_colors)] for i, g in enumerate(GROUP_ORDER)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Political group composition plots.")
    p.add_argument("--yearly-groups", required=True)
    p.add_argument("--pooled-groups", required=True)
    p.add_argument("--tier-summary", default=None)
    p.add_argument("--outdir", default="./outputs_us_market_mojo_clean/plots_groups")
    p.add_argument("--year-min", type=int, default=1985)
    p.add_argument("--year-max", type=int, default=2023)
    p.add_argument("--min-polkw-top20", type=float, default=10)
    p.add_argument("--min-polkw-21_100", type=float, default=25)
    p.add_argument("--rolling", type=int, default=3)
    return p.parse_args()


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.yearly_groups)
    df = df[df["year"].between(args.year_min, args.year_max)]
    df = df[df["group"].isin(GROUP_ORDER)]
    df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)
    return df


def compute_masks(df: pd.DataFrame, min_top20: float, min_rest: float) -> pd.DataFrame:
    tot = df.groupby("year")[["count_top20", "count_21_100"]].sum().reset_index()
    tot = tot.rename(columns={"count_top20": "tot_polkw_top20", "count_21_100": "tot_polkw_21_100"})
    tot["mask_top20"] = tot["tot_polkw_top20"] >= min_top20
    tot["mask_21_100"] = tot["tot_polkw_21_100"] >= min_rest
    return tot


def apply_rolling(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if window <= 1:
        return df
    roll = (
        df.sort_values(["group", "year"])
        .groupby("group")[["share_top20", "share_21_100", "gap_share"]]
        .rolling(window, min_periods=1, center=True)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df[["share_top20", "share_21_100", "gap_share"]] = roll
    return df


def make_heatmap(df: pd.DataFrame, value_col: str, masks: pd.DataFrame, tier: str, outpath: Path):
    years = sorted(df["year"].unique())
    groups = GROUP_ORDER
    mat = np.full((len(groups), len(years)), np.nan)
    year_to_idx = {y: i for i, y in enumerate(years)}
    for _, r in df.iterrows():
        g_idx = groups.index(r["group"])
        y_idx = year_to_idx[r["year"]]
        mat[g_idx, y_idx] = r[value_col]
    if tier == "top20":
        mask_years = set(masks[~masks["mask_top20"]]["year"].tolist())
    elif tier == "21_100":
        mask_years = set(masks[~masks["mask_21_100"]]["year"].tolist())
    else:
        mask_years = set(masks[(~masks["mask_top20"]) | (~masks["mask_21_100"])]["year"].tolist())
    for y in mask_years:
        if y in year_to_idx:
            mat[:, year_to_idx[y]] = np.nan
    fig, ax = plt.subplots(figsize=(12, 5))
    if "gap" in value_col:
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm", origin="lower", vmin=-np.nanmax(np.abs(mat)), vmax=np.nanmax(np.abs(mat)))
    else:
        im = ax.imshow(mat, aspect="auto", cmap="viridis", origin="lower")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([GROUP_LABELS[g] for g in groups])
    ax.set_xticks(range(0, len(years), max(1, len(years) // 10)))
    ax.set_xticklabels([years[i] for i in range(0, len(years), max(1, len(years) // 10))])
    ax.set_title(f"{value_col} heatmap ({tier})")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def stacked_area(df: pd.DataFrame, value_col: str, tier_label: str, outpath: Path):
    years = sorted(df["year"].unique())
    groups = GROUP_ORDER
    mat = []
    for g in groups:
        series = df[df["group"] == g].set_index("year")[value_col].reindex(years).fillna(0)
        mat.append(series.values)
    mat = np.array(mat)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [GROUP_TO_COLOR[g] for g in groups]
    ax.stackplot(years, mat, labels=[GROUP_LABELS[g] for g in groups], colors=colors)
    ax.set_title(f"{value_col} stacked ({tier_label})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def pooled_stacked(pooled_df: pd.DataFrame, outpath: Path):
    groups = GROUP_ORDER
    fig, ax = plt.subplots(figsize=(6, 4))
    bottom_top = 0
    bottom_rest = 0
    for g in groups:
        row = pooled_df[pooled_df["group"] == g].iloc[0]
        st = row["pooled_share_top20"]
        sr = row["pooled_share_21_100"]
        color = GROUP_TO_COLOR[g]
        ax.bar("Top20", st, bottom=bottom_top, label=GROUP_LABELS[g], color=color)
        ax.bar("21-100", sr, bottom=bottom_rest, color=color)
        bottom_top += st
        bottom_rest += sr
    ax.set_ylabel("Share of political keywords")
    ax.set_title("Pooled political group composition")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_totals(masks: pd.DataFrame, outpath: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(masks["year"], masks["tot_polkw_top20"], label="Top20")
    ax.plot(masks["year"], masks["tot_polkw_21_100"], label="21-100")
    ax.set_title("Total political keyword counts by tier")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # build global color mapping once for all group-colored plots
    global GROUP_TO_COLOR
    GROUP_TO_COLOR = build_group_colors()
    print("Group color mapping:")
    for g in GROUP_ORDER:
        print(f"  {GROUP_LABELS[g]} -> {GROUP_TO_COLOR[g]}")

    df = load_data(args)
    pooled = pd.read_csv(args.pooled_groups)
    masks = compute_masks(df, args.min_polkw_top20, args.min_polkw_21_100)
    masks.to_csv(outdir / "group_heatmap_masking_summary.csv", index=False)

    df = apply_rolling(df, args.rolling)

    make_heatmap(df, "share_top20", masks, "top20", outdir / "heatmap_group_shares_top20.png")
    make_heatmap(df, "share_21_100", masks, "21_100", outdir / "heatmap_group_shares_21_100.png")
    make_heatmap(df, "gap_share", masks, "gap", outdir / "heatmap_group_share_gap.png")

    stacked_area(df, "share_top20", "Top20", outdir / "stacked_group_shares_top20.png")
    stacked_area(df, "share_21_100", "21-100", outdir / "stacked_group_shares_21_100.png")
    pooled_stacked(pooled, outdir / "pooled_pol_groups_top20_vs_21_100.png")
    plot_totals(masks, outdir / "total_polkw_counts_by_tier.png")


if __name__ == "__main__":
    main()
