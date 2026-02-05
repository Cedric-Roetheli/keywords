"""
Generate structural break / level-shift plots for political keyword intensity (polshare)
using the pooled Top-100 overview CSV.

Input (expected):
- outputs_us_market_mojo_clean_boxoffice/plots_overview/overview_top100_by_year.csv
  Must contain: year, prevalence (optional), polshare (required), optionally totalkw.

Outputs:
- outputs_us_market_mojo_clean_boxoffice/plots_overview/break_plots/
    break_polshare_timeseries_with_regime_means.png
    break_polshare_distribution_pre_vs_post.png
    break_prevalence_timeseries_with_regime_means.png   (if prevalence present)
    break_totalkw_timeseries_with_regime_means.png      (if totalkw present)

Notes:
- Excludes 2020.
- PRE period: 1985–2002; POST: 2003–2024 (adapted to available data).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OVERVIEW = BASE_DIR / "outputs_us_market_mojo_clean_boxoffice" / "plots_overview" / "overview_top100_by_year.csv"
BREAK_DIR = BASE_DIR / "outputs_us_market_mojo_clean_boxoffice" / "plots_overview" / "break_plots"

PRE_START, PRE_END = 1985, 2002
POST_START, POST_END_DEFAULT = 2003, 2024


def find_overview() -> Path:
    if DEFAULT_OVERVIEW.exists():
        return DEFAULT_OVERVIEW
    matches = list(BASE_DIR.rglob("overview_top100_by_year.csv"))
    if not matches:
        raise FileNotFoundError("overview_top100_by_year.csv not found. Run make_overview_plots.py first.")
    return matches[0]


def detect_column(df: pd.DataFrame, keywords) -> Optional[str]:
    cols = list(df.columns)
    for c in cols:
        lc = c.lower()
        if all(k in lc for k in keywords):
            return c
    return None


def pick_polshare(df: pd.DataFrame) -> str:
    c = detect_column(df, ["polshare"])
    if c is None:
        raise ValueError("No polshare column found in overview CSV.")
    return c


def pick_prevalence(df: pd.DataFrame) -> Optional[str]:
    c = detect_column(df, ["prevalence"])
    if c:
        return c
    c = detect_column(df, ["political", "any"])
    return c


def pick_totalkw(df: pd.DataFrame) -> Optional[str]:
    c = detect_column(df, ["totalkw"])
    return c


def normalize_percent(series: pd.Series) -> pd.Series:
    """Return series in fractional terms (not percent). If median>2 assume already percent and divide by 100."""
    med = series.dropna().median()
    if med is None or np.isnan(med):
        return series
    if med > 2:
        return series / 100.0
    return series


def bootstrap_ci(vals: np.ndarray, n_boot: int = 10000, alpha: float = 0.05) -> Tuple[float, float]:
    if len(vals) == 0:
        return np.nan, np.nan
    boots = np.random.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    low, high = np.percentile(boots, [100 * (alpha / 2), 100 * (1 - alpha / 2)])
    return float(low), float(high)


def split_periods(df: pd.DataFrame, value_col: str):
    pre = df[(df["year"] >= PRE_START) & (df["year"] <= PRE_END)][["year", value_col]]
    # post_end is dynamic and set in main via a global; default to POST_END_DEFAULT for safety
    post_end = globals().get("POST_END_RUNTIME", POST_END_DEFAULT)
    post = df[(df["year"] >= POST_START) & (df["year"] <= post_end)][["year", value_col]]
    return pre, post


def plot_timeseries_with_means(
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    subtitle: str,
    pre_label: str,
    post_label: str,
    n_pre: int,
    n_post: int,
    outpath: Path,
):
    pre, post = split_periods(df, value_col)
    y_pct = df[value_col] * 100.0
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(df["year"], y_pct, marker="o", linewidth=1.8, color="#1f77b4")
    ax.axvline(2002.5, color="gray", linestyle="--", linewidth=1.2)

    pre_mean = pre[value_col].mean() * 100.0 if not pre.empty else np.nan
    post_mean = post[value_col].mean() * 100.0 if not post.empty else np.nan

    ax.axhline(pre_mean, color="#ff7f0e", linestyle="-", linewidth=1.4, label=f"Mean {pre_label}")
    ax.axhline(post_mean, color="#2ca02c", linestyle="-", linewidth=1.4, label=f"Mean {post_label}")

    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(np.arange(df["year"].min(), df["year"].max() + 1, 5))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10, frameon=False)
    sub = subtitle
    sub_extra = f"Pre: n={n_pre}, Post: n={n_post}; 2020 excluded."
    if sub:
        sub = f"{sub} | {sub_extra}"
    else:
        sub = sub_extra
    ax.text(0.99, -0.18, sub, transform=ax.transAxes, ha="right", va="top", fontsize=10, color="dimgray")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_distribution(
    pre: pd.Series,
    post: pd.Series,
    ylabel: str,
    title: str,
    pre_label: str,
    post_label: str,
    n_pre: int,
    n_post: int,
    outpath: Path,
):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    data = [pre * 100.0, post * 100.0]
    labels = [f"{pre_label} (n={n_pre})", f"{post_label} (n={n_post})"]
    positions = [0, 1]

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.4,
        patch_artist=True,
        boxprops=dict(facecolor="#d9e6f2"),
        medianprops=dict(color="black", linewidth=1.4),
        whiskerprops=dict(color="gray"),
        capprops=dict(color="gray"),
    )

    # jittered points
    rng = np.random.default_rng(123)
    for pos, vals in zip(positions, data):
        if len(vals) == 0:
            continue
        jitter = rng.normal(loc=pos, scale=0.03, size=len(vals))
        ax.scatter(jitter, vals, s=25, alpha=0.7, color="#1f77b4", edgecolor="none")

    # means + bootstrap CIs
    for pos, vals in zip(positions, data):
        if len(vals) == 0:
            continue
        mean_val = np.mean(vals)
        low, high = bootstrap_ci(np.array(vals), n_boot=10000)
        low *= 100.0 / 100.0  # noop for clarity
        high *= 100.0 / 100.0
        ax.scatter(pos, mean_val, color="#d62728", marker="D", s=40, zorder=4, label=None)
        ax.vlines(pos, low, high, colors="#d62728", linewidth=1.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def main():
    try:
        overview_path = find_overview()
        df = pd.read_csv(overview_path)
        print(f"Using overview file: {overview_path}")
        if "year" not in df.columns:
            raise ValueError("Overview CSV must contain 'year'.")
        df["year"] = df["year"].astype(int)
        df = df[df["year"] != 2020]
        df = df.sort_values("year")

        polshare_col = pick_polshare(df)
        prev_col = pick_prevalence(df)
        totalkw_col = pick_totalkw(df)

        df["polshare_frac"] = normalize_percent(df[polshare_col])
        if prev_col:
            df["prevalence_frac"] = normalize_percent(df[prev_col])
        if totalkw_col:
            df["totalkw_val"] = df[totalkw_col]

        # Drop rows missing polshare
        df = df.dropna(subset=["polshare_frac"])

        max_year = df["year"].max()
        pre_end_runtime = min(PRE_END, max_year)
        post_end_runtime = min(POST_END_DEFAULT, max_year)
        # set global used in split_periods
        globals()["POST_END_RUNTIME"] = post_end_runtime

        # Period splits
        pol_pre, pol_post = split_periods(df, "polshare_frac")
        print(
            f"Years available: {df['year'].min()}-{df['year'].max()} "
            f"(pre n={len(pol_pre)}, post n={len(pol_post)}, 2020 excluded)"
        )
        print(f"Detected max_year={max_year}; using post period {POST_START}-{post_end_runtime}")

        BREAK_DIR.mkdir(parents=True, exist_ok=True)

        pre_label = f"{PRE_START}-{pre_end_runtime}"
        post_label = f"{POST_START}-{post_end_runtime}"
        n_pre, n_post = len(pol_pre), len(pol_post)

        # Plot 1
        plot_timeseries_with_means(
            df,
            value_col="polshare_frac",
            ylabel="Mean polshare (%)",
            title="Political keyword intensity (polshare) and post-2002 level shift",
            subtitle="Top-100 pooled per year; polshare = political keywords / total keywords (unique); 2020 excluded.",
            pre_label=pre_label,
            post_label=post_label,
            n_pre=n_pre,
            n_post=n_post,
            outpath=BREAK_DIR / "break_polshare_timeseries_with_regime_means.png",
        )

        # Plot 2
        plot_distribution(
            pol_pre["polshare_frac"].dropna(),
            pol_post["polshare_frac"].dropna(),
            ylabel="Mean polshare (%)",
            title="Yearly distribution of political keyword intensity before vs after 2002",
            pre_label=pre_label,
            post_label=post_label,
            n_pre=n_pre,
            n_post=n_post,
            outpath=BREAK_DIR / "break_polshare_distribution_pre_vs_post.png",
        )

        # Optional prevalence timeseries
        if prev_col:
            prev_pre, prev_post = split_periods(df.dropna(subset=["prevalence_frac"]), "prevalence_frac")
            plot_timeseries_with_means(
                df.dropna(subset=["prevalence_frac"]),
                value_col="prevalence_frac",
                ylabel="Prevalence of ≥1 political keyword (%)",
                title="Prevalence of political keywords and post-2002 level shift",
                subtitle="Top-100 pooled per year; 2020 excluded.",
                pre_label=pre_label,
                post_label=post_label,
                n_pre=len(prev_pre),
                n_post=len(prev_post),
                outpath=BREAK_DIR / "break_prevalence_timeseries_with_regime_means.png",
            )

        # Optional totalkw timeseries
        if totalkw_col:
            plot_timeseries_with_means(
                df.dropna(subset=["totalkw_val"]),
                value_col="totalkw_val",
                ylabel="Mean total keywords (unique)",
                title="Total keyword volume and post-2002 shift",
                subtitle="Top-100 pooled per year; 2020 excluded.",
                pre_label=pre_label,
                post_label=post_label,
                n_pre=n_pre,
                n_post=n_post,
                outpath=BREAK_DIR / "break_totalkw_timeseries_with_regime_means.png",
            )

        print(f"Break plots written to {BREAK_DIR}")
    except Exception as exc:  # pragma: no cover
        print(f"Failed to generate break plots: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
