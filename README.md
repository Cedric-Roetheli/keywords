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
