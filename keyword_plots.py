"""
Plot utilities for TMDb keyword comparison outputs.

Loads yearly_summary.csv and yearly_top_keywords.csv and produces several figures:
- JS divergence over time (with shock-year markers)
- Jaccard similarity over time (Top50 vs previous year)
- Dual-axis overlay of JS divergence and 1 - Jaccard (theme turnover)
- Top keywords bar charts for selected years
- Shock windows grouped bars (shock year +/- 1)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot TMDb keyword divergence diagnostics.")
    parser.add_argument("--summary", default="yearly_summary.csv", help="Path to yearly_summary.csv")
    parser.add_argument("--topkw", default="yearly_top_keywords.csv", help="Path to yearly_top_keywords.csv")
    parser.add_argument("--outdir", default="./figures", help="Directory to save figures")
    parser.add_argument("--year-min", type=int, default=None, help="Optional min year filter")
    parser.add_argument("--year-max", type=int, default=None, help="Optional max year filter")
    parser.add_argument("--shock-years", default="2001,2008", help="Comma-separated shock years to highlight")
    parser.add_argument("--k", type=int, default=10, help="Top K keywords to plot per selected year")
    parser.add_argument(
        "--keyword-years",
        default="2001,2008",
        help="Comma-separated years to plot keyword bar charts for",
    )
    return parser.parse_args()


def parse_year_list(text: str) -> List[int]:
    years = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            years.append(int(part))
        except ValueError:
            continue
    return years


def load_data(summary_path: Path, topkw_path: Path, year_min: Optional[int], year_max: Optional[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean data, applying year filters."""
    summary = pd.read_csv(summary_path)
    topkw = pd.read_csv(topkw_path)

    summary["year"] = pd.to_numeric(summary["year"], errors="coerce").astype("Int64")
    topkw["year"] = pd.to_numeric(topkw["year"], errors="coerce").astype("Int64")
    summary["js_divergence"] = pd.to_numeric(summary.get("js_divergence", pd.NA), errors="coerce")
    summary["jaccard_top50_prevyear"] = pd.to_numeric(summary.get("jaccard_top50_prevyear", pd.NA), errors="coerce")
    topkw["score_logodds"] = pd.to_numeric(topkw.get("score_logodds", pd.NA), errors="coerce")

    if year_min is not None:
        summary = summary[summary["year"] >= year_min]
        topkw = topkw[topkw["year"] >= year_min]
    if year_max is not None:
        summary = summary[summary["year"] <= year_max]
        topkw = topkw[topkw["year"] <= year_max]

    summary = summary.sort_values("year")
    topkw = topkw.sort_values(["year", "score_logodds"], ascending=[True, False])
    return summary, topkw


def plot_js(df: pd.DataFrame, shock_years: Iterable[int], outdir: Path) -> None:
    data = df.dropna(subset=["js_divergence", "year"])
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data["year"], data["js_divergence"], marker="o", label="JS divergence")
    _mark_shocks(ax, shock_years, ymin=data["js_divergence"].min(), ymax=data["js_divergence"].max())
    ax.set_xlabel("Year")
    ax.set_ylabel("JS divergence")
    ax.set_title("JS divergence between Top-N and Rest keywords over time")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "js_divergence_over_time.png", dpi=150)
    plt.close(fig)


def plot_jaccard(df: pd.DataFrame, shock_years: Iterable[int], outdir: Path) -> None:
    data = df.dropna(subset=["jaccard_top50_prevyear", "year"])
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data["year"], data["jaccard_top50_prevyear"], marker="o", color="tab:orange", label="Jaccard Top50 vs prev year")
    _mark_shocks(ax, shock_years, ymin=data["jaccard_top50_prevyear"].min(), ymax=data["jaccard_top50_prevyear"].max())
    ax.set_xlabel("Year")
    ax.set_ylabel("Jaccard similarity (Top50)")
    ax.set_title("Year-over-year Top keyword set similarity")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "jaccard_top50_over_time.png", dpi=150)
    plt.close(fig)


def plot_dual_axis(df: pd.DataFrame, shock_years: Iterable[int], outdir: Path) -> None:
    data = df.dropna(subset=["js_divergence", "jaccard_top50_prevyear", "year"])
    if data.empty:
        return
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(data["year"], data["js_divergence"], color="tab:blue", marker="o", label="JS divergence")
    ax1.set_ylabel("JS divergence", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    turnover = 1 - data["jaccard_top50_prevyear"]
    ax2.plot(data["year"], turnover, color="tab:red", marker="s", label="1 - Jaccard (turnover)")
    ax2.set_ylabel("Theme turnover (1 - Jaccard)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    _mark_shocks(ax1, shock_years, ymin=min(data["js_divergence"].min(), turnover.min()), ymax=max(data["js_divergence"].max(), turnover.max()))

    fig.suptitle("Divergence and theme turnover over time")
    ax1.set_xlabel("Year")
    ax1.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(outdir / "divergence_and_turnover_dual_axis.png", dpi=150)
    plt.close(fig)


def plot_top_keywords(topkw: pd.DataFrame, summary: pd.DataFrame, years: Iterable[int], k: int, outdir: Path) -> None:
    for year in years:
        subset = topkw[topkw["year"] == year].head(k)
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(subset) + 2))
        subset = subset.iloc[::-1]  # reverse for horizontal bars top-down
        ax.barh(subset["keyword"], subset["score_logodds"], color="tab:blue")
        ax.set_xlabel("Log-odds score")
        ax.set_title(f"Top {len(subset)} keywords in {year}")
        summary_row = summary[summary["year"] == year]
        if not summary_row.empty:
            n_top = summary_row.iloc[0].get("n_top", "NA")
            n_rest = summary_row.iloc[0].get("n_rest", "NA")
            ax.text(0.95, 0.02, f"n_top={n_top}, n_rest={n_rest}", transform=ax.transAxes, ha="right", va="bottom")
        plt.tight_layout()
        fig.savefig(outdir / f"top_keywords_{year}.png", dpi=150)
        plt.close(fig)


def plot_shock_windows(df: pd.DataFrame, shock_years: Iterable[int], outdir: Path) -> None:
    records = []
    for y in shock_years:
        window_label = f"{y-1}-{y+1}"
        for yr in (y - 1, y, y + 1):
            row = df[df["year"] == yr]
            if row.empty:
                continue
            js_val = row.iloc[0]["js_divergence"]
            if pd.isna(js_val):
                continue
            records.append({"window": window_label, "year": yr, "js_divergence": js_val})
    if not records:
        return
    win_df = pd.DataFrame(records)
    windows = list(win_df["window"].unique())
    offsets = [-0.2, 0, 0.2]
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, yr_offset in enumerate([lambda y: y - 1, lambda y: y, lambda y: y + 1]):
        xs = []
        heights = []
        labels = []
        for i, w in enumerate(windows):
            base_year = int(w.split("-")[1]) - 1  # middle year
            target_year = yr_offset(base_year)
            row = win_df[(win_df["window"] == w) & (win_df["year"] == target_year)]
            xs.append(i + offsets[idx])
            heights.append(row["js_divergence"].iloc[0] if not row.empty else 0)
            labels.append(target_year)
        ax.bar(xs, heights, width=0.18, label=f"Year offset {['-1','0','+1'][idx]}")
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(windows)
    ax.set_ylabel("JS divergence")
    ax.set_title("Shock windows (shock year +/- 1)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(outdir / "shock_windows_js_divergence.png", dpi=150)
    plt.close(fig)


def _mark_shocks(ax, shock_years: Iterable[int], ymin: float, ymax: float) -> None:
    for y in shock_years:
        ax.axvline(y, color="gray", linestyle="--", alpha=0.6)
        ax.text(y, ymax, f"{y}", rotation=90, va="bottom", ha="right", fontsize=8, color="gray")


def print_console_summary(df: pd.DataFrame, shock_years: Iterable[int]) -> None:
    js_top = df.dropna(subset=["js_divergence"]).nlargest(5, "js_divergence")
    jaccard_bottom = df.dropna(subset=["jaccard_top50_prevyear"]).nsmallest(5, "jaccard_top50_prevyear")
    print("Top 5 JS divergence years:")
    for _, row in js_top.iterrows():
        print(f"  {int(row['year'])}: JS={row['js_divergence']:.4f}")
    print("Bottom 5 Jaccard similarity years (largest turnover):")
    for _, row in jaccard_bottom.iterrows():
        print(f"  {int(row['year'])}: Jaccard={row['jaccard_top50_prevyear']:.4f}")
    print("Shock year values:")
    for y in shock_years:
        row = df[df["year"] == y]
        if row.empty:
            print(f"  {y}: no data")
            continue
        js_val = row.iloc[0]["js_divergence"]
        jac_val = row.iloc[0]["jaccard_top50_prevyear"]
        js_str = f"{js_val:.4f}" if pd.notna(js_val) else "NA"
        jac_str = f"{jac_val:.4f}" if pd.notna(jac_val) else "NA"
        print(f"  {y}: JS={js_str}, Jaccard={jac_str}")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_df, topkw_df = load_data(Path(args.summary), Path(args.topkw), args.year_min, args.year_max)
    shock_years = parse_year_list(args.shock_years)
    keyword_years = parse_year_list(args.keyword_years)

    plot_js(summary_df, shock_years, outdir)
    plot_jaccard(summary_df, shock_years, outdir)
    plot_dual_axis(summary_df, shock_years, outdir)
    plot_top_keywords(topkw_df, summary_df, keyword_years, args.k, outdir)
    plot_shock_windows(summary_df, shock_years, outdir)
    print_console_summary(summary_df, shock_years)


if __name__ == "__main__":
    main()
