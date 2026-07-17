# 🌦️ sunflow-scores

A Python framework for calculating scores of **solar irradiance (GHI)** nowcasts. Evaluates 15-minute temporal resolution satellite-based predictions against observations. 



## ✨ Key Features
* **Solar Nowcasting Validation:** Evaluates 15-minute temporal resolution satellite-based predictions against satellite observations.
* **Robust Meteorological Metrics:** Powered by the `scores` package, calculating continuous metrics (RMSE, MAE).
* **Dependency Management:** Uses `uv` for reproducible virtual environments.



## 🚀 Getting Started: Cloning & Setup
We use uv for dependency management. It is extremely fast and ensures that everyone on the team is running the exact same package versions.

Step 1: Install uv (One-time setup)
If you do not already have uv installed on your laptop, run this command:

`curl -LsSf https://astral.sh/uv/install.sh | sh `

Step 2: Clone the Repository
Download the project code from github to your laptop. Open your terminal and run:

`git clone https://github.com/dmidk/sunflow-scores.git`

`cd sunflow-scores`


Step 3: Build the Environment with uv
Instead of manually creating environments and installing packages, simply run:

`uv sync`
What this does: uv will read the pyproject.toml file, automatically download the correct Python version (if needed), create an isolated .venv folder, and install exact, locked versions of xarray, scores, dask, etc. It should take seconds.


## 🚀 Running the Validation
The main validator is `run_validation.py`. It takes a start date, an end date, the nowcast directory, and the observation directory.

### One day
Run a single validation day like this:
```bash
uv run python run_validation.py \
  --start 2025-01-02 \
  --end 2025-01-02 \
  --nwc-dir /path/to/sunflow_output/202501 \
  --obs-dir /path/to/satellite_GHI/202501 \
  --output-dir results
```

This writes a daily CSV such as `results/scores_20250102.csv`.

### A whole month
Because the inputs are organized in `YYYYMM` folders and the outputs are daily CSVs, the usual workflow is to loop over the days in a month.

Example for January 2025:
```bash
for day in $(seq -w 1 31); do
  uv run python run_validation.py \
    --start 2025-01-${day} \
    --end 2025-01-${day} \
    --nwc-dir /path/to/sunflow_output/202501 \
    --obs-dir /path/to/satellite_GHI/202501 \
    --output-dir results
done
```


### Restrict to a domain (bounding box)
By default the validation runs over the whole satellite/nowcast domain. To restrict it to a
geographic bounding box (for example only Denmark), pass `--bbox` as
`LON_MIN LAT_MIN LON_MAX LAT_MAX`:
```bash
uv run python run_validation.py \
  --start 2025-06-01 \
  --end 2025-06-03 \
  --nwc-dir /path/to/nowcasts \
  --obs-dir /path/to/observations \
  --bbox 7.5 54.5 15.5 58.0
```
The bounding box is applied identically to the nowcast and satellite observation grids, so the
fast same-grid alignment stays valid. Denmark's approximate box is `7.5 54.5 13.0 58.0`.

### Custom variable names
If your files use different variable names, pass them explicitly:
```bash
uv run python run_validation.py \
  --start 2025-01-02 \
  --end 2025-01-02 \
  --nwc-dir /path/to/nowcasts \
  --obs-dir /path/to/observations \
  --nowcast_ghi_var GHI_nowcast \
  --obs_ghi_var GHI_observation \
 
```

### Command-line arguments
| Argument | Description | Required | Default |
|---|---|---|---|
| `--start` | First nowcast initialization time. | Yes | |
| `--end` | Last nowcast initialization time. | Yes | |
| `--nwc-dir` | Directory containing the nowcast NetCDF files. | Yes | |
| `--obs-dir` | Directory containing the observation NetCDF files. | Yes | |
| `--output-dir` | Directory where the output score files are written. | No | `results` |
| `--bbox` | Restrict validation to a geographic bounding box: `LON_MIN LAT_MIN LON_MAX LAT_MAX` (e.g. Denmark: `7.5 54.5 13.0 58.0`). | No | none (whole domain) |
| `--nowcast_ghi_var` | GHI variable in the nowcast files. | No | `probabilistic_advection` |
| `--obs_ghi_var` | GHI variable in the observation files. | No | `sds` |
| `--obs_cs_ghi_var` | Clear-sky GHI variable in the observation files. | No | `sds_cs` |

### What the validation writes
The current pipeline writes one CSV per day:
`scores_YYYYMMDD.csv`

Each file contains by-init scores and a `lead_time_minutes` column, which is what the plotting tools consume.

## 📊 Plotting the results

All plotting functions support both flat directory structures (all `scores_*.csv` files in one folder) and monthly folder structures (e.g., `results/202501/`, `results/202502/`, etc.).

All plotting scripts are located in the `plotting/` directory. Run them with `uv run python plotting/<script>.py`.

### Daily plots
Use `plotting/plot_daily_scores.py` for one daily CSV or for a directory of daily CSVs.

Single day heatmap:
```bash
uv run python plotting/plot_daily_scores.py \
  --input results/scores_20250102.csv \
  --output-dir results/plots
```

Monthly summary over all daily CSVs in a directory:
```bash
uv run python plotting/plot_daily_scores.py \
  --input results \
  --summary \
  --output-dir results/plots
```

Monthly average heatmap by initialization time and lead time:
```bash
uv run python plotting/plot_daily_scores.py \
  --input results \
  --average-heatmap \
  --output-dir results/plots
```

You can also choose a metric with `--metric mae`, `--metric rmse`, or `--metric both`.

### Monthly plots
Use `plot_monthly_heatmaps.py` to write one summary plot and one averaged heatmap for each month found in a directory of daily CSVs.

Plot both summary and averaged heatmaps for both metrics:
```bash
uv run python plotting/plot_monthly_heatmaps.py \
  --input results \
  --summary \
  --heatmap \
  --metric both \
  --output-dir results/monthly_plots
```

Only summary MAE plots:
```bash
uv run python plotting/plot_monthly_heatmaps.py \
  --input results \
  --summary \
  --metric mae \
  --output-dir results/monthly_plots
```

Only averaged RMSE heatmaps:
```bash
uv run python plotting/plot_monthly_heatmaps.py \
  --input results \
  --heatmap \
  --metric rmse \
  --output-dir results/monthly_plots
```

The monthly script does not require every day of the month to be present. It will plot whatever daily CSVs exist for that month, which is useful when some days were skipped because there was no data.

### Per-month scores for a single lead time
Use `plot_leadtime_monthly.py` to extract only one lead time of choice (e.g. the
60-minute lead time) from every daily CSV and plot the mean score for that lead
time as one bar per calendar month.

```bash
uv run python plotting/plot_leadtime_monthly.py \
  --input results \
  --lead-time 60 \
  --metric both \
  --output-dir results/plots
```

The `--lead-time` value is given in minutes and must match a `lead_time_minutes`
value present in the CSVs (0, 15, 30, ...). With `--metric both` each month shows
two grouped bars (MAE and RMSE); use `--metric mae` or `--metric rmse` for a
single bar per month. The output is written to
`monthly_scores_leadtime_<N>min_<metric>.png`.

### Seasonal diurnal-cycle plot
Use `plot_seasonal_diurnal_cycles.py` to average the monthly diurnal-cycle curves into the four meteorological seasons.

Example for MAE only:
```bash
uv run python plotting/plot_seasonal_diurnal_cycles.py \
  --input results \
  --metric mae \
  --output-dir results/seasonal_plots
```

The script first computes each month's diurnal-cycle average, then averages those monthly curves within each season so the three months contribute equally.

### Lead-time curves (daily/monthly/yearly)
Use `plot_leadtime_curves.py` to plot scores across lead-time horizons (0, 15min, 30min, ..., 360min) for a given day, month, or year by averaging over all initialization times.

Single day:
```bash
uv run python plotting/plot_leadtime_curves.py \
  --input results \
  --date 2025-01-15 \
  --metric both \
  --output-dir results/plots
```

Multiple days (overlaid on same plot):
```bash
uv run python plotting/plot_leadtime_curves.py \
  --input results \
  --date 2025-01-15 2025-01-16 2025-01-17 \
  --metric both \
  --output-dir results/plots
```

Entire month:
```bash
uv run python plotting/plot_leadtime_curves.py \
  --input results \
  --month 2025-01 \
  --metric mae \
  --output-dir results/plots
```

Entire year:
```bash
uv run python plotting/plot_leadtime_curves.py \
  --input results \
  --year 2025 \
  --metric rmse \
  --output-dir results/plots
```

## 🔀 Two-model comparison workflow
This end-to-end workflow validates two model versions over the **same domain** for a date range,
then overlays their scores on shared comparison figures. It can be adapted for any pair of
models and any domain.

### 1. Run validation for each model version
Use the parameterized `run_validation.sh` script or the DMI-specific wrapper `run_validation_dmi.sh`.

**For external users** (local data directories):
```bash
export START_DATE=2025-01-01 END_DATE=2025-12-31
export NWC_DIR=/path/to/model_v1.0.0/nowcasts
export OBS_DIR=/path/to/observations
export OUTPUT_DIR=results/v1.0.0
export BBOX="7.5 54.5 15.5 58.0"  # Denmark
bash run_validation.sh

# Repeat for second model version
export NWC_DIR=/path/to/model_v1.0.1/nowcasts
export OUTPUT_DIR=results/v1.0.1
bash run_validation.sh
```


### 2. Compare the two models with `plot_model_comparison.py`
`plot_model_comparison.py` reads two (or more) labelled score directories and draws them on a
single figure. It has two modes.

**Lead-time line graph (0 → 360 min)** — one line per model:
```bash
uv run python plotting/plot_model_comparison.py \
  --inputs /path/to/scores/v1.0.0 /path/to/scores/v1.0.1 \
  --labels v1.0.0 v1.0.1 \
  --mode leadtime-line \
  --metric both \
  --output-dir plots
```
Writes `comparison_leadtime_line_<metric>.png`.

**Monthly bar chart at the 15- and 30-minute horizons** — grouped bars per model per month:
```bash
uv run python plotting/plot_model_comparison.py \
  --inputs /path/to/scores/v1.0.0 /path/to/scores/v1.0.1 \
  --labels v1.0.0 v1.0.1 \
  --mode monthly-bars \
  --lead-time 15,30 \
  --metric both \
  --output-dir plots
```
Writes one figure per requested lead time:
`comparison_monthly_leadtime_15min_<metric>.png` and
`comparison_monthly_leadtime_30min_<metric>.png`.

The `--inputs` and `--labels` lists must have the same number of entries. For `monthly-bars`,
each requested `--lead-time` must be present in the CSVs, otherwise a clear error naming the
missing lead time is raised. Coverage may differ between models (missing days/months); the
aggregation is a mean over available days, so both curves/bars still render for the data present.

## 📂 Project Structure
```text
sunflow-scores/
├── src/
│   └── sunflow_scores/
│       ├── __init__.py
│       ├── validator.py         # Core library: SatelliteNowcastLoader, SatelliteObservationLoader,
│       │                         # ScoreCalculator, GroundScoreCalculator
│       └── plot_utils.py        # Shared plotting utilities (CSV loading, metric columns, helpers)
├── plotting/
│   ├── plot_daily_scores.py         # Plot one day or a directory of daily CSVs (heatmaps)
│   ├── plot_monthly_heatmaps.py     # Plot monthly summaries / averaged heatmaps from daily CSVs
│   ├── plot_leadtime_monthly.py     # Plot per-month scores for a single chosen lead time
│   ├── plot_leadtime_curves.py      # Plot scores across lead-time horizons (0–360 min)
│   ├── plot_seasonal_diurnal_cycles.py  # Plot 4-season averages of monthly diurnal-cycle curves
│   └── plot_model_comparison.py     # Compare two model versions: lead-time line + monthly bars
├── tests/
│   ├── test_validation.py           # Validation pipeline tests
│   ├── test_model_comparison.py     # Model comparison logic tests
│   └── test_score_computation_resilience.py  # Corruption handling tests
├── run_validation.py            # Daily validation script: writes one scores_YYYYMMDD.csv per run
├── run_validation.sh            # Parameterized validation runner
├── pyproject.toml               # uv dependency definitions
├── uv.lock                      # Strictly locked dependency hashes
├── .gitignore                   # Excludes data files, results, generated plots
└── README.md
```

- Core library code lives under `src/sunflow_scores/`
- Plotting scripts are organized in `plotting/` for easy discovery and maintenance
- Run plotting scripts with `uv run python plotting/<script>.py`

## 🧪 Testing

Run the test suite with:
```bash
pytest tests/
```

Tests cover:
- Validation pipeline correctness (data loading, alignment, score computation)
- Model comparison logic (aggregation, filtering)
- Resilience to HDF5 file corruption and transient I/O errors


## 🤝 Contributing

Contributions are welcome. For bug reports or feature requests, open an issue on GitHub.
When submitting PRs:
- Include tests for new functionality
- Update the README if adding new scripts or CLI arguments
- Run `pytest` before submitting
- Ensure code follows the project's import/style patterns (see existing scripts)

## 📄 License

MIT. See LICENSE file for details.
