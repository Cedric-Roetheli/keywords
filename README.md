# Keywords & Political Content Analysis

This repo contains Python scripts and generated outputs for exploring TMDb movie metadata, political keyword incidence, and US domestic box office tiers. It includes end‑to‑end pipelines (Kaggle + local CSVs), diagnostics, and plotting utilities.

## Data inputs
- **TMDb metadata**: e.g., `data/TMDB_movie_dataset_v11.csv` (large). Kaggle slug `asaniczka/tmdb-movies-dataset-2023-930k-movies`.
- **MPST tags**: Kaggle slug `cryptexcode/mpst-movie-plot-synopses-with-tags` (used in MPST scripts).
- **US box office (domestic)**: `data/boxoffice_data_2024.csv` (preferred; richer early years) or `data/Mojo_budget_update.csv` (older Mojo scrape).
- Kaggle credentials are required for scripts that download via KaggleHub/CLI; local CSVs are used for the US‑box‑office pipelines.

## Core concepts & metrics
Per movie (computed with **movie‑incidence** on UNIQUE keywords):
- **total_kw_i**: number of unique TMDb keywords parsed for the movie.
- **political_kw_count_i** (`polkw`): number of unique keywords matching fixed regex political patterns.
- **political_share_i** (`polshare`): `political_kw_count_i / total_kw_i` (NaN if no keywords).
- **political_any_i**: indicator of at least one political keyword.
- **political_group_counts_i**: counts by primary political group (priority order: war_security_intel → institutions_elections_law → economy_finance_crisis → migration_police_civilrights → labor_collective_action → inequality_corruption_elites).

Year/tier aggregates (Top20 vs 21–100, etc.):
- **mean_polkw_*:** mean `political_kw_count_i` per movie in tier.
- **mean_polshare_*:** mean `political_share_i` per movie in tier (adjusts for tagging volume).
- **mean_totalkw_*:** mean `total_kw_i` per movie (tagging volume diagnostic).
- **share_any_*:** fraction of movies with `political_any_i=1`.
- **Group shares:** fraction of political keyword mass belonging to each political group.
- **Gaps:** differences between tiers, e.g., `gap_polshare = mean_polshare_top20 - mean_polshare_21_100`.

Political regex groups (underscore‑friendly, re.IGNORECASE):
1) war_security_intel  
2) institutions_elections_law  
3) economy_finance_crisis  
4) migration_police_civilrights  
5) labor_collective_action  
6) inequality_corruption_elites  

## Key scripts (runnable)
- **political_keyword_us_mojo_clean.py**  
  US domestic ranking pipeline (Top 1–100) using box office CSV + TMDb keywords; applies theatrical filters, fuzzy title matching fallback, year‑quality filters, tier summaries, genre buckets, political group composition, and plotting. Main outputs live in `outputs_us_market_mojo_clean_boxoffice/`.

- **political_group_plots.py**  
  Reads yearly + pooled political group share CSVs and renders heatmaps, stacked areas, pooled stacked bars, and totals diagnostics with consistent group colors.

- **political_keyword_tiers_top100.py / political_keyword_tiers.py / political_keyword_series.py / political_keyword_trends.py**  
  Earlier TMDb‑only pipelines for keyword trends, hit vs rest, and top‑N tiers (non‑US ranking). Use Kaggle TMDb CSV; produce yearly keyword tables, JS divergence, Jaccard, and plots.

- **political_keywords.py**  
  Builds political keyword vocab from TMDb keywords (regex rules + optional zero‑shot) and exports vocab/candidate lists.

- **mpst_tmdb_check.py / mpst_tag_analysis.py**  
  MPST tag matching to TMDb (coverage, imdb_id matching, per‑year stats).

- **keyword_analysis.py / keyword_plots.py / motivation_tiers_summary.py**  
  Legacy keyword pipelines and plotting utilities tied to the original top‑20 vs rest setup and motivation figures.

Each script has CLI help (`python script.py --help`) describing required/optional args.

## Main outputs (current focus)
`outputs_us_market_mojo_clean_boxoffice/` (latest US domestic analysis using `boxoffice_data_2024.csv`):
- `us_domestic_rankings_top100.csv`: cleaned per‑year ranks 1–100 by domestic gross.  
- `merge_diagnostics.csv`, `excluded_years.csv`: merge quality and year filters.  
- `yearly_tier_summary_us_mojo_clean.csv`: year‑level Top20 vs 21–100 metrics (polkw, polshare, totalkw, domestic means/medians, gaps).  
- `yearly_genre_bucket_summary_us_mojo_clean.csv`: same metrics within genre buckets (action vs non‑action or AWT vs other).  
- `yearly_political_group_shares_us_mojo_clean.csv`: group counts/shares per year and tier.  
- `pooled_political_group_shares_us_mojo_clean.csv`: pooled group composition across kept years.  
- `plots_groups/`: heatmaps, stacked area plots, pooled stacked bars, totals diagnostics with consistent group colors.  
- Additional PNGs: domestic vs time, political gaps, tagging volume, genre bucket gaps.
- `plots_overview/`: thesis-intro overview figures (Top-100 pooled, 2020 excluded):
  - `overview_prevalence_political_any_top100.png`: share of titles with ≥1 political keyword.
  - `overview_intensity_polshare_top100.png`: mean polshare (political keywords / total keywords).
  - `overview_heatmap_group_shares_top100.png`: political group composition heatmap (pooled Top-100).
  - `overview_dualaxis_prevalence_and_intensity_top100.png`: combined prevalence + intensity.
  - `overview_top100_by_year.csv`: year-level prevalence, polshare, and group shares used for plots.
  - `break_plots/`: structural break visuals (pre/post 2002) built from `overview_top100_by_year.csv`:
    - `break_polshare_timeseries_with_regime_means.png`: polshare time series with pre/post means and 2002.5 break line.
    - `break_polshare_distribution_pre_vs_post.png`: box/jitter + bootstrap CI comparing yearly polshare pre vs post 2002.
    - `break_prevalence_timeseries_with_regime_means.png`: prevalence time series with pre/post means (if prevalence present).
- `plots_tier_extra/`: supplemental tier comparisons (Top20 vs 21–100):
  - `tier_polshare_distribution_yearly_means.png`: box/jitter + mean/CI of yearly polshare (%) per tier.
  - `tier_polshare_gap_pre_vs_post.png`: box/jitter + mean/CI of yearly polshare gap (Top20 − 21–100) before vs after 2002.

### Plot guide (what each figure shows and data behind it)
Data sources noted per block; “polkw/polshare/totalkw” are defined in *Core concepts & metrics*.

**outputs_us_market_mojo_clean_boxoffice/** (TMDb + `boxoffice_data_2024.csv`, generated by `political_keyword_us_mojo_clean.py`; group plots via `political_group_plots.py`)  
- `domestic_top20_vs_21_100_over_time.png`, `domestic_median_top20_vs_21_100_over_time.png`: mean/median domestic gross by tier.  
- `polkw_mean_top20_vs_21_100_over_time.png`, `polkw_gap_top20_minus_21_100.png`: political keyword counts per film and tier gap.  
- `polshare_mean_top20_vs_21_100_over_time.png`, `polshare_gap_top20_minus_21_100.png`: normalized political share per film and tier gap.  
- `totalkw_mean_top20_vs_21_100_over_time.png`: tagging volume diagnostic.  
- `genre_bucket_polshare_gap.png`, `genre_bucket_polkw_gap.png`: gaps within genre buckets (action vs non‑action or AWT vs other).  
- `pooled_genre_shares_top20_vs_21_100_clean.png`: pooled genre incidence (Top20 vs 21–100).  
- `pooled_pol_groups_top20_vs_21_100.png`: pooled political group composition shares (consistent group colors).  
- `pol_group_shares_over_time_top20.png`, `pol_group_shares_over_time_21_100.png`: group share trends per tier.  
- `pol_group_share_gaps_over_time.png`: share differences (Top20 − 21–100) by group.  
- `plots_groups/heatmap_group_shares_top20.png`, `...21_100.png`, `heatmap_group_share_gap.png`: heatmaps of group shares and gaps (masked if political counts too low; masking summary in `plots_groups/group_heatmap_masking_summary.csv`).  
- `plots_groups/stacked_group_shares_top20.png`, `...21_100.png`: stacked area of group shares per tier.  
- `plots_groups/pooled_pol_groups_top20_vs_21_100.png`: pooled stacked bars with group colors.  
- `plots_groups/total_polkw_counts_by_tier.png`: total political keyword counts per tier used for masking.  
- Source CSVs: `yearly_tier_summary_us_mojo_clean.csv`, `yearly_genre_bucket_summary_us_mojo_clean.csv`, `yearly_political_group_shares_us_mojo_clean.csv`, `pooled_political_group_shares_us_mojo_clean.csv`.

**outputs_tiers_top100/** (TMDb only; `political_keyword_tiers_top100.py`)  
- `metric_*over_time*.png`: mean/median success metric by tier (Top20 vs 21–100).  
- `political_kw_intensity_*`, `political_kw_gap_*`, `political_share_keywords_*`: polkw/polshare trends and gaps.  
- `tagging_volume_total_keywords_*`: totalkw diagnostic.  
- `genre_*` plots: genre composition and bucket gaps.  
- Source tables: `hits_rank_lists_top100.csv`, `yearly_tier_summary_top100.csv`, `yearly_genre_bucket_summary_top100.csv`.

**outputs_tiers/** (TMDb only; `political_keyword_tiers.py`)  
- `mean_political_keywords_*`, `gap_political_keywords_*`, `share_any_*`, `group_gap_*`: polkw and prevalence across Top20 vs 21–200 style tiers.  
- Source tables: `hits_rank_lists.csv`, `yearly_tier_keyword_intensity.csv`, `yearly_tier_group_intensity.csv`.

**outputs_political_keywords_series_hits/** (`political_keyword_series.py` with hit detection)  
- `topX_hits_*`, `political_groups_hits_*`, `political_groups_hit_vs_rest_gap.png`: hit vs rest keyword/group rates and shares.  
- Source tables: `yearly_keyword_rates_all.csv`, `yearly_keyword_rates_hit.csv`, `yearly_keyword_hit_vs_rest.csv`, `yearly_group_rates_all.csv`, `yearly_group_rates_hit.csv`, `yearly_group_hit_vs_rest.csv`.

**outputs_political_keywords_series/** (`political_keyword_series.py` without hit split)  
- `topX_political_keywords_*`, `political_groups_*`, `total_political_keyword_counts_over_time.png`: overall political keyword incidence/rate/share over time.  
- `shock_window_*` CSVs: subsets for shock‑year windows.

**outputs_political_keywords_trends/** (`political_keyword_trends.py`)  
- `top_political_keywords_*`, `total_political_keyword_counts_over_time.png`: trend plots from the baseline political keyword vocab.  
- Source: `yearly_political_keyword_counts.csv`.

**outputs_political_keywords_filtered/** and **outputs_political_keywords/** (`political_keywords.py`)  
- `unique_keywords*.csv`, `candidate_keywords.csv`: vocabulary and coverage tables.  
- `political_keywords_rulebased.csv`, `final_political_keywords_*`: political vocab outputs.  
- `top_political_keywords_rulebased.png`: top political keyword bar chart.

**outputs_mpst_analysis/** (`mpst_tag_analysis.py` on MPST+TMDb)  
- `js_divergence_over_time_tags.png`, `jaccard_over_time_tags.png`, `dual_axis_js_and_1minusjaccard_tags.png`: hit vs rest MPST tag divergence/turnover.  
- `shock_window_*_js.png`, `shock_window_summary.csv`, `yearly_top_tags.csv`, `yearly_summary_tags.csv`: supporting tables/plots.

**outputs_mpst_check/** (`mpst_tmdb_check.py`)  
- `mpst_movies_per_year*.png/csv`: MPST coverage by year.  
- `mpst_tmdb_match_*`: imdb_id match diagnostics to TMDb.

**outputs/** (original TMDb keyword pipeline `keyword_analysis.py` + `keyword_plots.py`)  
- `yearly_top_keywords.csv`, `yearly_summary.csv`, `shock_year_report.txt`, `js_divergence.png`, `jaccard_rotation.png`: top‑20 vs rest keyword overrepresentation and divergence plots from the earliest run.

**outputs_motivation_final/** (`motivation_tiers_summary.py`)  
- `gap_polkw_and_adjusted_over_time.png`, `polkw_gap_vs_totalkw_gap_scatter.png`, `two_period_errorbars_pre_post.png`, `break_year_sensitivity.png`, `rolling_gap_plots.png`: motivation/intro figures using pre/post 2000 splits, residualized gaps, and rolling means.  
- Source tables: `two_period_summary.csv`, `yearly_adjusted_series.csv`, `regression_summary.csv`, `report.md`.

Other output folders (earlier experiments):
- `outputs_tiers_top100/`, `outputs_tiers/`, `outputs_political_keywords_series_hits/`, `outputs_political_keywords_series/`, `outputs_political_keywords_trends/`, `outputs_political_keywords_filtered/`, `outputs_political_keywords/`, `outputs_mpst_analysis/`, `outputs_mpst_check/`, `outputs_motivation_final/`, `outputs_us_market_mojo/`, `outputs_us_market_mojo_clean/`. These contain intermediate CSVs/plots for TMDb hit vs rest, political keyword vocab building, MPST matching, and motivation figures.

## Typical commands
- Latest US box office tier run (already executed):  
  ```bash
  python political_keyword_us_mojo_clean.py \
    --tmdb-csv ./data/TMDB_movie_dataset_v11.csv \
    --mojo-csv ./data/boxoffice_data_2024.csv \
    --outdir ./outputs_us_market_mojo_clean_boxoffice \
    --year-min 1985 --year-max 2023 \
    --runtime-min 40 --min-vote-count 50 --filter-adult \
    --fuzzy-title-match --fuzzy-threshold 92
  ```
- Group composition plots (consistent colors):  
  ```bash
  python political_group_plots.py \
    --yearly-groups ./outputs_us_market_mojo_clean_boxoffice/yearly_political_group_shares_us_mojo_clean.csv \
    --pooled-groups ./outputs_us_market_mojo_clean_boxoffice/pooled_political_group_shares_us_mojo_clean.csv \
    --tier-summary ./outputs_us_market_mojo_clean_boxoffice/yearly_tier_summary_us_mojo_clean.csv \
    --outdir ./outputs_us_market_mojo_clean_boxoffice/plots_groups \
    --year-min 1985 --year-max 2023 --min-polkw-top20 10 --min-polkw-21_100 25 --rolling 3
  ```
- Earlier TMDb‑only tiers (non‑US) example:  
  ```bash
  python political_keyword_tiers_top100.py --tmdb-csv ./data/TMDB_movie_dataset_v11.csv --outdir ./outputs_tiers_top100
  ```
- Political keyword vocab build example:  
  ```bash
  python political_keywords.py --csv ./data/TMDB_movie_dataset_v11.csv --outdir ./outputs_political_keywords_filtered
  ```

## Notes
- Data files in `data/` are ignored in git to avoid large binaries; place TMDb/box office CSVs there.
- Outputs (CSVs/PNGs) are versioned so figures and tables can be referenced directly.
- All keyword parsing uses robust handling of JSON/list/pipe/comma formats, normalizing to lowercase with underscores and dropping missing tokens (`<na>`, `nan`, `none`, `null`, empty).
