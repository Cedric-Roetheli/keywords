"""
Post-process Top20 vs 21–100 political keyword tier summary to produce adjusted metrics,
two-period comparisons, and plots for thesis motivation.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List
import statsmodels.api as sm
from scipy import stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Motivation/intro figures from tier summary.")
    p.add_argument("--tier-summary", required=True)
    p.add_argument("--outdir", default="./outputs_motivation_final")
    p.add_argument("--start-year", type=int, default=1985)
    p.add_argument("--end-year", type=int, default=2023)
    p.add_argument("--break-year", type=int, default=2000)
    p.add_argument("--shock-years", default="2001,2008")
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def load_data(path: Path, start_year: int, end_year: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [
        "year",
        "mean_polkw_top20",
        "mean_polkw_21_100",
        "mean_polshare_top20",
        "mean_polshare_21_100",
        "mean_totalkw_top20",
        "mean_totalkw_21_100",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "share_any_top20" in df.columns and "share_any_21_100" in df.columns:
        has_any = True
    else:
        has_any = False
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    df = df.dropna(subset=required)
    print(f"Retained years: {len(df)} (from {start_year}-{end_year})")
    return df, has_any


def compute_gaps(df: pd.DataFrame, has_any: bool) -> pd.DataFrame:
    df = df.copy()
    df["gap_polkw"] = df["mean_polkw_top20"] - df["mean_polkw_21_100"]
    df["gap_totalkw"] = df["mean_totalkw_top20"] - df["mean_totalkw_21_100"]
    df["gap_polshare"] = df["mean_polshare_top20"] - df["mean_polshare_21_100"]
    df["adj_ratio"] = (df["mean_polkw_top20"] / df["mean_totalkw_top20"]) - (
        df["mean_polkw_21_100"] / df["mean_totalkw_21_100"]
    )
    if has_any:
        df["gap_any"] = df["share_any_top20"] - df["share_any_21_100"]
    return df


def fit_adjustment_regression(df: pd.DataFrame) -> Tuple[pd.Series, sm.regression.linear_model.RegressionResultsWrapper]:
    X = sm.add_constant(df["gap_totalkw"])
    model = sm.OLS(df["gap_polkw"], X).fit(cov_type="HC1")
    resid = model.resid
    return resid, model


def bootstrap_ci(series: pd.Series, n_boot: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(series.dropna().values, size=(n_boot, len(series.dropna())), replace=True)
    means = samples.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def two_period_stats(df: pd.DataFrame, break_year: int, n_boot: int, seed: int, metrics: List[str]) -> pd.DataFrame:
    rows = []
    pre = df[df["year"] < break_year]
    post = df[df["year"] >= break_year]
    for m in metrics:
        pre_mean = pre[m].mean()
        post_mean = post[m].mean()
        pre_ci = bootstrap_ci(pre[m], n_boot, seed)
        post_ci = bootstrap_ci(post[m], n_boot, seed + 1)
        # Welch t-test
        tstat, pval = stats.ttest_ind(post[m].dropna(), pre[m].dropna(), equal_var=False)
        rows.append(
            {
                "metric": m,
                "mean_pre": pre_mean,
                "ci_pre_low": pre_ci[0],
                "ci_pre_high": pre_ci[1],
                "mean_post": post_mean,
                "ci_post_low": post_ci[0],
                "ci_post_high": post_ci[1],
                "diff_post_minus_pre": post_mean - pre_mean,
                "p_value": pval,
            }
        )
    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame, resid: pd.Series, model: sm.regression.linear_model.RegressionResultsWrapper, args: argparse.Namespace):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    shock_years = [int(s) for s in args.shock_years.split(",") if s.strip()]

    def mark_shocks(ax):
        ax.axvline(args.break_year, color="black", linestyle="--", alpha=0.7, label="break")
        for y in shock_years:
            ax.axvline(y, color="gray", linestyle="--", alpha=0.6)

    # A) gap_polkw and residual over time
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["year"], df["gap_polkw"], label="gap_polkw", marker="o")
    ax.plot(df["year"], resid, label="adjusted_gap_resid", marker="s")
    mark_shocks(ax)
    ax.set_title("Political keyword gap and adjusted residual over time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Gap")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(outdir / "gap_polkw_and_adjusted_over_time.png", dpi=150)
    plt.close(fig)

    # B) Scatter gap_totalkw vs gap_polkw with OLS line
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["gap_totalkw"], df["gap_polkw"], alpha=0.7)
    x_vals = np.linspace(df["gap_totalkw"].min(), df["gap_totalkw"].max(), 100)
    y_vals = model.params["const"] + model.params["gap_totalkw"] * x_vals
    ax.plot(x_vals, y_vals, color="red", label="OLS fit")
    ax.set_xlabel("gap_totalkw")
    ax.set_ylabel("gap_polkw")
    ax.set_title("Political gap vs tagging-volume gap")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "polkw_gap_vs_totalkw_gap_scatter.png", dpi=150)
    plt.close(fig)

    # C) Two-period error bars
    # This will be built in main after two-period stats

    # D) Rolling averages
    fig, ax = plt.subplots(figsize=(10, 5))
    roll = df.set_index("year")[["gap_polkw", "gap_polshare"]].rolling(3, min_periods=1).mean()
    roll.plot(ax=ax)
    mark_shocks(ax)
    ax.set_title("3-year rolling gaps")
    ax.set_xlabel("Year")
    ax.set_ylabel("Gap (rolling mean)")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(outdir / "rolling_gap_plots.png", dpi=150)
    plt.close(fig)


def plot_two_period_errorbars(summary_df: pd.DataFrame, outdir: Path, break_year: int):
    metrics = ["gap_polkw", "gap_totalkw", "gap_polshare", "adjusted_gap_resid", "adj_ratio"]
    summary_df = summary_df[summary_df["metric"].isin(metrics)]
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(summary_df))
    width = 0.35
    ax.errorbar(summary_df["mean_pre"], y_pos + width / 2, xerr=[summary_df["mean_pre"] - summary_df["ci_pre_low"], summary_df["ci_pre_high"] - summary_df["mean_pre"]], fmt="o", label="Pre")
    ax.errorbar(summary_df["mean_post"], y_pos - width / 2, xerr=[summary_df["mean_post"] - summary_df["ci_post_low"], summary_df["ci_post_high"] - summary_df["mean_post"]], fmt="o", label="Post")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary_df["metric"])
    ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_title(f"Pre ({'<' + str(break_year)}) vs Post ({'≥' + str(break_year)}) means with 95% CI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "two_period_errorbars_pre_post.png", dpi=150)
    plt.close(fig)


def breakpoint_sensitivity(df: pd.DataFrame, start: int = 1990, end: int = 2010) -> pd.DataFrame:
    rows = []
    for b in range(start, end + 1):
        pre = df[df["year"] < b]
        post = df[df["year"] >= b]
        diff = post["gap_polkw"].mean() - pre["gap_polkw"].mean()
        rows.append({"break_year": b, "diff_post_minus_pre": diff})
    return pd.DataFrame(rows)


def make_break_plot(break_df: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(break_df["break_year"], break_df["diff_post_minus_pre"], marker="o")
    ax.axvline(2000, color="black", linestyle="--", alpha=0.7)
    ax.set_title("Break-year sensitivity (gap_polkw)")
    ax.set_xlabel("Candidate break year")
    ax.set_ylabel("Diff post - pre")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(outdir / "break_year_sensitivity.png", dpi=150)
    plt.close(fig)


def write_report(df: pd.DataFrame, resid: pd.Series, model: sm.regression.linear_model.RegressionResultsWrapper, summary_df: pd.DataFrame, args: argparse.Namespace):
    outdir = Path(args.outdir)
    lines = []
    lines.append(f"# Motivation Summary")
    lines.append(f"- Years: {args.start_year}-{args.end_year}, retained {len(df)} years.")
    pre = df[df["year"] < args.break_year]
    post = df[df["year"] >= args.break_year]
    lines.append(f"- gap_polkw mean pre: {pre['gap_polkw'].mean():.3f}, post: {post['gap_polkw'].mean():.3f} (diff {post['gap_polkw'].mean() - pre['gap_polkw'].mean():.3f})")
    lines.append(f"- Adjusted gap (residual) mean pre: {pre['adjusted_gap_resid'].mean():.3f}, post: {post['adjusted_gap_resid'].mean():.3f}")
    lines.append("")
    lines.append("## Regression (gap_polkw ~ gap_totalkw)")
    lines.append(f"- alpha (const): {model.params.get('const', np.nan):.4f}")
    lines.append(f"- beta (gap_totalkw): {model.params.get('gap_totalkw', np.nan):.4f}")
    lines.append(f"- R^2: {model.rsquared:.4f}")
    pd.DataFrame({"alpha": [model.params.get("const", np.nan)], "beta_gap_totalkw": [model.params.get("gap_totalkw", np.nan)], "r2": [model.rsquared]}).to_csv(outdir / "regression_summary.csv", index=False)
    (outdir / "report.md").write_text("\n".join(lines))


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df, has_any = load_data(Path(args.tier_summary), args.start_year, args.end_year)
    df = compute_gaps(df, has_any)
    resid, model = fit_adjustment_regression(df)
    df["adjusted_gap_resid"] = resid
    df.to_csv(outdir / "yearly_adjusted_series.csv", index=False)

    metrics = ["gap_polkw", "gap_totalkw", "gap_polshare", "adjusted_gap_resid", "adj_ratio"]
    summary_df = two_period_stats(df, args.break_year, args.bootstrap, args.seed, metrics)
    summary_df.to_csv(outdir / "two_period_summary.csv", index=False)

    make_plots(df, resid, model, args)
    plot_two_period_errorbars(summary_df, outdir, args.break_year)

    break_df = breakpoint_sensitivity(df)
    make_break_plot(break_df, outdir)

    write_report(df, resid, model, summary_df, args)

    # Console summary
    print(summary_df)


if __name__ == "__main__":
    main()
