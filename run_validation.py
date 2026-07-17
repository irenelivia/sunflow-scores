"""
run_validation.py — Batch Solar Nowcast Validation
====================================================
Loads nowcasts and observations for a given date range, computes MAE and RMSE
per lead time, and writes the results to a CSV file.

Usage (satellite observations):
    uv run python run_validation.py --start 2026-02-01 --end 2026-02-28 \
        --nwc-dir /path/to/nowcasts --obs-dir /path/to/satellite_obs

Usage (ground station observations):
    uv run python run_validation.py --start 2026-02-01 --end 2026-02-28 \
        --nwc-dir /path/to/nowcasts --ground-obs /path/to/stations.csv

Output (written to --output-dir, default: ./results/):
    scores_YYYYMMDD.csv

Alignment modes (--align-mode):
    auto    – tries fast exact-time same-grid alignment first, falls back to
              general chunked-loop mode on any mismatch (default)
    fast    – exact-time same-grid; raises if grids differ
    general – chunked loop with nearest-time reindex; tolerates grid mismatch
"""

import argparse
import time
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr
import sys

project_root = Path(__file__).resolve().parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from sunflow_scores.validator import (
    SatelliteNowcastLoader,
    SatelliteObservationLoader,
    GroundObservationLoader,
    ScoreCalculator,
    GroundScoreCalculator,
    _open_with_retry,
    _compute_scores_per_init,
)

dask.config.set(scheduler="threads")


def _fmt_seconds(seconds: float) -> str:
    return f"{seconds:8.2f}s"


def _print_timing_summary(stage_times: dict[str, float]) -> None:
    print("Timing summary:")
    width = max(len(name) for name in stage_times)
    for name, seconds in stage_times.items():
        print(f"  {name.ljust(width)} : {_fmt_seconds(seconds)}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate solar nowcasts against satellite or ground observations."
    )
    parser.add_argument(
        "--start", required=True,
        help="First nowcast initialization time, e.g. 2026-02-01",
    )
    parser.add_argument(
        "--end", required=True,
        help="Last nowcast initialization time, e.g. 2026-02-28 23:45",
    )
    parser.add_argument(
        "--nwc-dir", required=True,
        help="Directory containing nowcast files",
    )

    # Observation source — mutually exclusive
    obs_group = parser.add_mutually_exclusive_group(required=True)
    obs_group.add_argument(
        "--obs-dir",
        help="Directory containing satellite observation files (NetCDF4_sds_*.nc)",
    )
    obs_group.add_argument(
        "--ground-obs",
        help="Path to ground station observations (CSV file, directory of CSVs, or NetCDF)",
    )

    parser.add_argument(
        "--output-dir", default="results",
        help="Directory to write output CSV files (default: ./results/)",
    )
    parser.add_argument(
        "--align-mode", default="auto", choices=["fast", "general", "auto"],
        help="Satellite alignment mode: fast | general | auto (default: auto). "
             "Ignored when --ground-obs is used.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=50,
        help="Chunk size for general alignment mode (default: 50)",
    )
    parser.add_argument(
        "--bbox", type=float, nargs=4, default=None,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help="Restrict validation to a geographic bounding box, given as "
             "lon_min lat_min lon_max lat_max (e.g. Denmark: 7.5 54.5 13.0 58.0). "
             "Applies to the nowcast and satellite observation grids.",
    )
    parser.add_argument("--nowcast_ghi_var",  type=str, default="probabilistic_advection",
                        help="Name of the GHI variable in the nowcast files")
    parser.add_argument("--obs_ghi_var",      type=str, default="sds",
                        help="Name of the GHI variable in the observation files")
    parser.add_argument("--obs_cs_ghi_var",   type=str, default="sds_cs",
                        help="Name of the clear-sky GHI variable in the observation files")
    parser.add_argument("--ground_obs_ghi_var",    type=str, default="ghi",
                        help="GHI column/variable name in ground observation files (default: ghi)")
    parser.add_argument("--ground_obs_cs_ghi_var", type=str, default="cs_ghi",
                        help="Clear-sky GHI column/variable name in ground obs (default: cs_ghi)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage_times: dict[str, float] = {}

    nwc_start = pd.Timestamp(args.start)
    nwc_end   = pd.Timestamp(args.end)

    # If date-only is given (midnight), interpret as full day.
    if nwc_start == nwc_start.normalize():
        nwc_start = nwc_start.normalize()
    if nwc_end == nwc_end.normalize():
        nwc_end = nwc_end.normalize() + pd.Timedelta(hours=23, minutes=45)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bbox = tuple(args.bbox) if args.bbox is not None else None

    date_tag = nwc_start.strftime("%Y%m%d")
    obs_mode = "ground" if args.ground_obs else "satellite"

    print(f"\n{'=' * 60}")
    print(f"  Validation run  [{obs_mode} observations]")
    print(f"  Nowcasts : {nwc_start}  →  {nwc_end}")
    print(f"  Output   : {output_dir}/")
    if bbox is not None:
        print(f"  Domain   : bbox lon=[{bbox[0]}, {bbox[2]}] lat=[{bbox[1]}, {bbox[3]}]")
    print(f"{'=' * 60}\n")

    def _skip_day(message: str) -> None:
        print(f"  SKIP: {message}")
        print("  Finished in 0.0 min")
        print(f"{'=' * 60}\n")

    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Load nowcasts
    # ------------------------------------------------------------------
    t_step = time.perf_counter()
    print("Step 1/4 — Loading nowcasts...")
    nwc_loader = SatelliteNowcastLoader(data_dir=args.nwc_dir, bbox=bbox)
    try:
        nowcast_ds = nwc_loader.load_data(nwc_start, nwc_end)
    except (ValueError, OSError) as exc:
        _skip_day(str(exc))
        return

    nwc_init_min, nwc_init_max, nwc_valid_min, nwc_valid_max = dask.compute(
        nowcast_ds.initialization_time.min(),
        nowcast_ds.initialization_time.max(),
        nowcast_ds.valid_time.min(),
        nowcast_ds.valid_time.max(),
    )
    print(f"  Loaded nowcast init range:       {nwc_init_min.values} to {nwc_init_max.values}")
    print(f"  Loaded nowcast valid_time range: {nwc_valid_min.values} to {nwc_valid_max.values}")
    print(f"  {nowcast_ds.sizes['initialization_time']} runs × "
          f"{nowcast_ds.sizes['lead_time']} lead steps loaded "
          f"({time.perf_counter() - t_step:.1f}s)\n")
    stage_times["step1/load_nowcasts"] = time.perf_counter() - t_step

    # ------------------------------------------------------------------
    # 2. Load observations
    # ------------------------------------------------------------------
    t_step = time.perf_counter()
    print("Step 2/4 — Loading observations...")

    obs_start = pd.Timestamp(nwc_valid_min.values) - pd.Timedelta(minutes=15)
    obs_end   = pd.Timestamp(nwc_valid_max.values) + pd.Timedelta(minutes=15)

    if obs_mode == "satellite":
        obs_loader = SatelliteObservationLoader(data_dir=args.obs_dir, bbox=bbox)
        try:
            obs_ds = obs_loader.load_data(obs_start, obs_end)
        except (ValueError, OSError) as exc:
            _skip_day(str(exc))
            return
        obs_min, obs_max = dask.compute(obs_ds.time.min(), obs_ds.time.max())
        print(f"  Loaded obs time range: {obs_min.values} to {obs_max.values}")
        print(f"  {obs_ds.sizes['time']} observation timesteps loaded "
              f"({time.perf_counter() - t_step:.1f}s)\n")
    else:
        ground_loader = GroundObservationLoader(
            data_path=args.ground_obs,
            ghi_var=args.ground_obs_ghi_var,
            cs_ghi_var=args.ground_obs_cs_ghi_var,
        )
        try:
            obs_ds = ground_loader.load_data(obs_start, obs_end)
        except (ValueError, OSError) as exc:
            _skip_day(str(exc))
            return
        print(f"  {obs_ds.sizes['station_id']} stations, "
              f"{obs_ds.sizes['time']} time steps loaded "
              f"({time.perf_counter() - t_step:.1f}s)\n")

    stage_times["step2/load_obs"] = time.perf_counter() - t_step

    # ------------------------------------------------------------------
    # 3. Align data
    # ------------------------------------------------------------------
    t_step = time.perf_counter()
    if obs_mode == "satellite":
        print(f"Step 3/4 — Aligning data (mode={args.align_mode})...")
        scorer = ScoreCalculator(
            nowcast_ds, obs_ds,
            nowcast_ghi_var=args.nowcast_ghi_var,
            obs_ghi_var=args.obs_ghi_var,
            obs_cs_ghi_var=args.obs_cs_ghi_var,
        )
        try:
            scorer.align_data(mode=args.align_mode, chunk_size=args.chunk_size)
        except (ValueError, OSError) as exc:
            _skip_day(str(exc))
            return
    else:
        print("Step 3/4 — Aligning ground observations to nowcast grid...")
        scorer = GroundScoreCalculator(
            nowcast_ds, obs_ds,
            nowcast_ghi_var=args.nowcast_ghi_var,
            obs_ghi_var=args.ground_obs_ghi_var,
            obs_cs_ghi_var=args.ground_obs_cs_ghi_var,
        )
        try:
            scorer.align_data()
        except (ValueError, RuntimeError, OSError) as exc:
            _skip_day(str(exc))
            return

    print(f"  Done ({time.perf_counter() - t_step:.1f}s)\n")
    stage_times["step3/align"] = time.perf_counter() - t_step

    # ------------------------------------------------------------------
    # 4. Compute scores and save
    # ------------------------------------------------------------------
    t_step = time.perf_counter()
    print("Step 4/4 — Calculating scores and saving...")

    mae_by_init_lazy  = scorer.calculate_mae_by_init()
    rmse_by_init_lazy = scorer.calculate_rmse_by_init()

    t_compute = time.perf_counter()
    try:
        mae_by_init, rmse_by_init = _compute_scores_per_init(mae_by_init_lazy, rmse_by_init_lazy)
    except (ValueError, OSError, AttributeError, RuntimeError, KeyError) as exc:
        _skip_day(str(exc))
        return
    stage_times["scores/compute"] = time.perf_counter() - t_compute

    scores_ds = xr.Dataset({"mae_by_init": mae_by_init, "rmse_by_init": rmse_by_init})

    out_path = output_dir / f"scores_{date_tag}.csv"

    print(f"  Writing by-init CSV to {out_path}")
    df = scores_ds.to_dataframe().reset_index()
    df["lead_time_minutes"] = (df["lead_time"] / np.timedelta64(1, "m")).astype("int32")

    # Keep output schema consistent: insert ensemble column if not present.
    if "ensemble" not in df.columns:
        ensemble_value = 0
        nowcast_var = nowcast_ds[args.nowcast_ghi_var]
        if "ensemble" in nowcast_var.coords:
            coord_values = np.asarray(nowcast_var.coords["ensemble"].values)
            if coord_values.size > 0:
                ensemble_value = int(coord_values[0])
        df.insert(1, "ensemble", ensemble_value)

    df = df.drop(columns=["lead_time"])
    df.to_csv(out_path, index=False)

    # For ground mode, also save per-station scores
    if obs_mode == "ground" and hasattr(scorer, "calculate_mae_by_station"):
        t_station = time.perf_counter()
        mae_by_station  = scorer.calculate_mae_by_station()
        rmse_by_station = scorer.calculate_rmse_by_station()
        station_scores  = xr.Dataset({
            "mae_by_station":  mae_by_station,
            "rmse_by_station": rmse_by_station,
        })
        station_path = output_dir / f"scores_by_station_{date_tag}.csv"
        station_scores.to_dataframe().reset_index().to_csv(station_path, index=False)
        print(f"  Per-station scores → {station_path}")
        stage_times["scores/station_csv"] = time.perf_counter() - t_station

    print(f"  ALL METRICS → {out_path}")
    print(f"  ({time.perf_counter() - t_step:.1f}s)\n")
    stage_times["step4/scores_and_save"] = time.perf_counter() - t_step

    total = time.perf_counter() - t0
    stage_times["run/total"] = total
    _print_timing_summary(stage_times)

    print(f"{'=' * 60}")
    print(f"  Finished in {total / 60:.1f} min")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
