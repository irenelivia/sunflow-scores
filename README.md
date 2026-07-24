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
fast same-grid alignment stays valid. Denmark's approximate box is `7.5 54.5 15.5 58.0`.

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
| `--bbox` | Restrict validation to a geographic bounding box: `LON_MIN LAT_MIN LON_MAX LAT_MAX` (e.g. Denmark: `7.5 54.5 15.5 58.0`). | No | none (whole domain) |
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

## ☀️ Pyranometer point validation (DINI & nowcast vs ground truth)

In addition to grid-vs-grid validation, `run_validation_pyranometer.py` compares both
the DINI NWP forecast and the satellite nowcast against two DTU pyranometer stations
(Risoe, Lyngby) at their exact coordinates.

### Why this needs its own pipeline
The three GHI sources have different native time resolution and accumulation
conventions, reconciled in `src/sunflow_scores/time_alignment.py`:

| Source | Native resolution | Convention |
|---|---|---|
| Satellite nowcast | 15 min | `"12:00"` = avg[12:00, 12:15) — start-labelled |
| Pyranometer | 10s (Risoe) / 1min (Lyngby) | grouped to `"12:00"` = avg[12:00, 12:15) — start-labelled |
| DINI | hourly, cumulative-since-forecast-start | `"13:00"` (ref 12:00 + step 1h) = accumulated GHI over [12:00, 13:00) — end-labelled, 1h wide |

DINI's cumulative `grad` variable (J/m2) is de-accumulated by diffing consecutive
hourly steps, divided by 3600 to get an average W/m2 for that hour, then broadcast
across the four 15-minute start-labelled slots within it (DINI has no intra-hour
resolution, so all four get the same value).

DINI's grid is in its own projection (`dini_projection`), so station coordinates are
transformed with `pyproj`/`cartopy` before doing a nearest-cell `.sel()` — unlike the
satellite nowcast, which is already on a regular lat/lon grid and reuses the existing
`GroundScoreCalculator`'s nearest-grid-cell lookup unchanged.

Both forecasts also have a real-world **availability latency** before they're actually
downloadable: DINI takes ~3 hours (compute + data-transfer), the nowcast ~15-30
minutes. A `lead_time=0` value exists in the raw aligned data but was never actually
available at its own valid_time, so `run_validation_pyranometer.py` drops lead times
below each source's latency (`time_alignment.filter_usable_lead_times`,
`DINI_MIN_USABLE_LEAD_TIME` = 3h, `NOWCAST_MIN_USABLE_LEAD_TIME` = 30min) before
computing scores or saving the aligned NetCDFs — otherwise the "which model wins"
comparison would overstate both forecasts' real usefulness.

Clear-sky index (kt) scoring is intentionally out of scope for this pipeline for now,
since neither the raw pyranometer CSVs nor the DINI zarr carry a clear-sky GHI
variable; QC of the raw pyranometer signal (e.g. night-time dark-offset) is also left
to dedicated tooling such as [solarpy](https://github.com/AssessingSolar/solarpy)
rather than done here.

### Station registry
Station coordinates live in `src/sunflow_scores/stations.py`:
- **Risoe**: 55.694243°N, 12.101793°E
- **Lyngby**: 55.79064°N, 12.52505°E, 50m AMSL (DTU Building 119 rooftop)

### Running it
```bash
uv run python run_validation_pyranometer.py \
  --start 2025-06-01 --end 2025-06-07 \
  --pyranometer-dir /dmidata/projects/energivejr-data/pyranometers \
  --dini-path /dmidata/projects/energivejr-data/dini/consolidated/dini_sharded.zarr \
  --nwc-dir /dmidata/projects/energivejr-data/nowcasts/v1.0.0/202506 \
  --output-dir results/pyranometer
```

**Command-line options:**

Use `--mode nowcast` or `--mode dini` to run one comparison at a time (default: `both`).
This matters because **DINI point extraction is much slower than the nowcast
comparison for the same date range**: the zarr store's chunks each cover the *entire*
spatial grid (~24MB) but only a single `(init, step)` pair, so extracting even 2
station pixels still reads one full chunk per init/step — runtime scales with the
date range, not the number of stations. A progress bar reports chunk-read progress
so a long-running DINI extraction doesn't look stuck.

Restrict to a single station with `--station {risoe,lyngby}`:
```bash
uv run python run_validation_pyranometer.py --mode nowcast --station risoe \
  --start 2025-06-01 --end 2025-06-07 \
  --pyranometer-dir /dmidata/projects/energivejr-data/pyranometers \
  --nwc-dir /dmidata/projects/energivejr-data/nowcasts/v1.0.0/202506 \
  --output-dir results/pyranometer
```

Filter to daytime hours only (sunrise to sunset per station) with `--daytime-only`:
```bash
uv run python run_validation_pyranometer.py --daytime-only \
  --start 2025-06-01 --end 2025-06-07 \
  --pyranometer-dir /dmidata/projects/energivejr-data/pyranometers \
  --dini-path /dmidata/projects/energivejr-data/dini/consolidated/dini_sharded.zarr \
  --nwc-dir /dmidata/projects/energivejr-data/nowcasts/v1.0.0/202506 \
  --output-dir results/pyranometer
```

`--pyranometer-dir` must contain `risoe/` and `lyngby/` subdirectories with the raw
CSVs. For Lyngby, two raw sources from the same instrument are concatenated
automatically by date: `dtu_2025_MM.csv` for Jan-Mar 2025, and
`SR300_GHI_DTU_Lyngby_*.csv` from April 2025 onward.

Writes four CSVs to `--output-dir`:
- `dini_by_init.csv`, `dini_by_station.csv` — DINI vs pyranometer
- `nowcast_by_init.csv`, `nowcast_by_station.csv` — satellite nowcast vs pyranometer

The `_by_init` files carry a `lead_time_minutes` column (per-station-averaged scores
by initialization/lead time); the `_by_station` files give one overall MAE/RMSE per
station. The `*_aligned.nc` files carry the full (initialization_time, lead_time,
station_id) forecast/observation grid (not just the summary scores) and feed the two
plotting scripts below.

### Plotting: time series and lead-time skill comparison
```bash
# Overlay nowcast, DINI, and pyranometer GHI at a fixed lead time, one plot per station
uv run python plotting/plot_pyranometer_timeseries.py \
  --input-dir results/pyranometer \
  --lead-time-minutes 0 \
  --output-dir plots

# MAE/RMSE vs lead time for both sources, marking where DINI overtakes the nowcast
uv run python plotting/plot_pyranometer_leadtime_scores.py \
  --input-dir results/pyranometer \
  --metric both \
  --output-dir plots
```
`plot_pyranometer_timeseries.py` reads whichever of `dini_aligned.nc` /
`nowcast_aligned.nc` are present and needs at least one; `--station` selects specific
stations (default: all present). `plot_pyranometer_leadtime_scores.py` requires both
`dini_by_init.csv` and `nowcast_by_init.csv`, averages each metric over all
initialization times per lead time, and reports the crossover lead time (or that DINI
is already ahead at the shortest shared lead time, or that it never catches up within
the nowcast's own lead-time range — the nowcast tops out around 6h, DINI extends
further).

## 📂 Project Structure
```text
sunflow-scores/
├── src/
│   └── sunflow_scores/
│       ├── __init__.py
│       ├── validator.py         # Core library: SatelliteNowcastLoader, SatelliteObservationLoader,
│       │                         # ScoreCalculator, GroundScoreCalculator
│       ├── stations.py          # Pyranometer station registry (lat/lon/alt)
│       ├── time_alignment.py    # 15-min resampling + DINI de-accumulation label-convention rules
│       ├── pyranometer_loader.py  # RisoePyranometerLoader, LyngbyPyranometerLoader
│       ├── dini_loader.py       # DiniPointLoader: point extraction from the DINI zarr forecast
│       ├── point_score_calculator.py  # PointScoreCalculator: scores for two pre-extracted point series
│       └── plot_utils.py        # Shared plotting utilities (CSV loading, metric columns, helpers)
├── plotting/
│   ├── plot_daily_scores.py         # Plot one day or a directory of daily CSVs (heatmaps)
│   ├── plot_monthly_heatmaps.py     # Plot monthly summaries / averaged heatmaps from daily CSVs
│   ├── plot_leadtime_monthly.py     # Plot per-month scores for a single chosen lead time
│   ├── plot_leadtime_curves.py      # Plot scores across lead-time horizons (0–360 min)
│   ├── plot_seasonal_diurnal_cycles.py  # Plot 4-season averages of monthly diurnal-cycle curves
│   ├── plot_model_comparison.py     # Compare two model versions: lead-time line + monthly bars
│   ├── plot_pyranometer_timeseries.py     # Nowcast/DINI/pyranometer GHI time series overlay
│   └── plot_pyranometer_leadtime_scores.py  # MAE/RMSE vs lead time: DINI vs nowcast, with crossover
├── tests/
│   ├── test_validation.py           # Validation pipeline tests
│   ├── test_model_comparison.py     # Model comparison logic tests
│   ├── test_score_computation_resilience.py  # Corruption handling tests
│   ├── test_time_alignment.py       # 15-min resampling / DINI de-accumulation tests
│   ├── test_pyranometer_loader.py   # Risoe/Lyngby CSV loader tests (incl. Lyngby source cutover)
│   └── test_point_score_calculator.py  # Point alignment + MAE/RMSE tests
├── run_validation.py            # Daily validation script: writes one scores_YYYYMMDD.csv per run
├── run_validation_pyranometer.py  # Point validation: DINI & nowcast vs pyranometer ground truth
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

Contributions are welcome. For bug reports or feature suggestions, open an issue on GitHub.
