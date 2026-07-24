"""
Point validation: DINI and the satellite nowcast against pyranometer ground
truth (Risoe, Lyngby), for a given date range.

Use --mode to run one comparison at a time -- useful because DINI point
extraction reads one full spatial zarr chunk (~24MB) per (init, step)
regardless of how few stations are requested, so it can be much slower than
the nowcast comparison for the same date range (see DiniPointLoader).

Scores exclude lead times shorter than each forecast's real-world
availability latency (DINI: 3h, nowcast: 30min) -- see
sunflow_scores.time_alignment.filter_usable_lead_times.

Writes to --output-dir, depending on --mode:
    dini_by_init.csv, dini_by_station.csv, dini_aligned.nc         (DINI vs pyranometer)
    nowcast_by_init.csv, nowcast_by_station.csv, nowcast_aligned.nc (satellite nowcast vs pyranometer)

The *_aligned.nc files hold the full (initialization_time, lead_time, station_id)
forecast/observation series (not just the summary scores), consumed by
plotting/plot_pyranometer_timeseries.py and plotting/plot_pyranometer_leadtime_scores.py.

Examples:
    # both comparisons
    uv run python run_validation_pyranometer.py \\
        --start 2025-06-01 --end 2025-06-07 \\
        --pyranometer-dir /dmidata/projects/energivejr-data/pyranometers \\
        --dini-path /dmidata/projects/energivejr-data/dini/consolidated/dini_sharded.zarr \\
        --nwc-dir /dmidata/projects/energivejr-data/nowcasts/v1.0.0/202506 \\
        --output-dir results/pyranometer

    # nowcast only (fast, no --dini-path needed)
    uv run python run_validation_pyranometer.py --mode nowcast \\
        --start 2025-06-01 --end 2025-06-07 \\
        --pyranometer-dir /dmidata/projects/energivejr-data/pyranometers \\
        --nwc-dir /dmidata/projects/energivejr-data/nowcasts/v1.0.0/202506 \\
        --output-dir results/pyranometer

    # DINI only (no --nwc-dir needed)
    uv run python run_validation_pyranometer.py --mode dini \\
        --start 2025-06-01 --end 2025-06-07 \\
        --pyranometer-dir /dmidata/projects/energivejr-data/pyranometers \\
        --dini-path /dmidata/projects/energivejr-data/dini/consolidated/dini_sharded.zarr \\
        --output-dir results/pyranometer
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import pandas as pd
import xarray as xr
from astral import Observer
from astral.sun import sun

project_root = Path(__file__).resolve().parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from sunflow_scores import (
    DINI_MIN_USABLE_LEAD_TIME,
    NOWCAST_MIN_USABLE_LEAD_TIME,
    DiniPointLoader,
    GroundScoreCalculator,
    LyngbyPyranometerLoader,
    PointScoreCalculator,
    RisoePyranometerLoader,
    SatelliteNowcastLoader,
    filter_usable_lead_times,
)
from sunflow_scores.stations import STATIONS

STATION_IDS = ["risoe", "lyngby"]

STATION_LOADERS = {
    "risoe": RisoePyranometerLoader,
    "lyngby": LyngbyPyranometerLoader,
}


def _by_init_to_frame(mae: xr.DataArray, rmse: xr.DataArray) -> pd.DataFrame:
    """Flatten (initialization_time, lead_time) MAE/RMSE DataArrays into a tidy long DataFrame."""
    df = xr.Dataset({"mae": mae, "rmse": rmse}).to_dataframe().reset_index()
    df["lead_time_minutes"] = df["lead_time"] / pd.Timedelta(minutes=1)
    return df


def _by_station_to_frame(mae: xr.DataArray, rmse: xr.DataArray) -> pd.DataFrame:
    return xr.Dataset({"mae": mae, "rmse": rmse}).to_dataframe().reset_index()


def filter_to_daytime(data: xr.Dataset, station_ids: list[str]) -> xr.Dataset:
    """Filter aligned data to only daytime hours (sunrise to sunset for each station)."""
    import numpy as np
    import datetime

    times = pd.to_datetime(data.coords["initialization_time"].values)
    daytime_mask = np.zeros(len(times), dtype=bool)

    for station_id in station_ids:
        meta = STATIONS[station_id]
        observer = Observer(latitude=meta["lat"], longitude=meta["lon"])

        daily_sunrise_sunset = {}
        for date in times.normalize().unique():
            sun_info = sun(observer, date=date.date())
            sr = sun_info["sunrise"].replace(tzinfo=None)
            ss = sun_info["sunset"].replace(tzinfo=None)
            daily_sunrise_sunset[date] = (sr, ss)

        for i, t in enumerate(times):
            t_date = pd.Timestamp(t).normalize()
            sr, ss = daily_sunrise_sunset[t_date]
            t_naive = pd.Timestamp(t).replace(tzinfo=None)
            if sr <= t_naive <= ss:
                daytime_mask[i] = True

    return data.isel(initialization_time=daytime_mask)


def load_pyranometer_obs(pyranometer_dir: str, start_date: str, end_date: str, station_ids: list[str]) -> xr.Dataset:
    base = Path(pyranometer_dir)
    loaded = [
        STATION_LOADERS[station_id](base / station_id).load_data(start_date, end_date)
        for station_id in station_ids
    ]
    if len(loaded) == 1:
        return loaded[0]
    return xr.concat(loaded, dim="station_id")


def run_dini(args, ground_obs: xr.Dataset, output_dir: Path, station_ids: list[str], daytime_only: bool = False) -> None:
    print("\nDINI vs pyranometer...")
    dini_data = DiniPointLoader(args.dini_path, station_ids).load_data(args.start, args.end)
    dini_calc = PointScoreCalculator(dini_data, ground_obs)
    dini_calc.align_data()
    dini_calc.aligned_data = filter_usable_lead_times(dini_calc.aligned_data, DINI_MIN_USABLE_LEAD_TIME)
    if daytime_only:
        dini_calc.aligned_data = filter_to_daytime(dini_calc.aligned_data, station_ids)
    aligned = dini_calc.aligned_data
    dini_by_init = _by_init_to_frame(dini_calc.calculate_mae_by_init(), dini_calc.calculate_rmse_by_init())
    dini_by_station = _by_station_to_frame(dini_calc.calculate_mae_by_station(), dini_calc.calculate_rmse_by_station())
    dini_by_init.to_csv(output_dir / "dini_by_init.csv", index=False)
    dini_by_station.to_csv(output_dir / "dini_by_station.csv", index=False)
    # engine="h5netcdf" avoids loading netCDF4-python's separately-bundled
    # libhdf5 in a process that has already loaded h5netcdf/h5py for reads
    # (SatelliteNowcastLoader, etc.) -- mixing the two causes HDF5 errors on
    # write and segfaults on later opens.
    aligned.to_netcdf(output_dir / "dini_aligned.nc", engine="h5netcdf")


def run_nowcast(args, ground_obs: xr.Dataset, output_dir: Path, station_ids: list[str], daytime_only: bool = False) -> None:
    print("\nSatellite nowcast vs pyranometer...")
    nowcast_data = SatelliteNowcastLoader(args.nwc_dir).load_data(args.start, args.end)
    nowcast_calc = GroundScoreCalculator(
        nowcast_data, ground_obs, nowcast_ghi_var=args.nowcast_ghi_var, obs_ghi_var="ghi",
    )
    nowcast_calc.align_data()
    nowcast_calc.aligned_data = filter_usable_lead_times(nowcast_calc.aligned_data, NOWCAST_MIN_USABLE_LEAD_TIME)
    if daytime_only:
        nowcast_calc.aligned_data = filter_to_daytime(nowcast_calc.aligned_data, station_ids)
    aligned = nowcast_calc.aligned_data
    nowcast_by_init = _by_init_to_frame(nowcast_calc.calculate_mae_by_init(), nowcast_calc.calculate_rmse_by_init())
    nowcast_by_station = _by_station_to_frame(
        nowcast_calc.calculate_mae_by_station(), nowcast_calc.calculate_rmse_by_station()
    )
    nowcast_by_init.to_csv(output_dir / "nowcast_by_init.csv", index=False)
    nowcast_by_station.to_csv(output_dir / "nowcast_by_station.csv", index=False)
    # aligned_data carries the full (init, lead_time, station) grid for plotting;
    # values are already eager (loaded during align_data), so this is a plain write.
    # engine="h5netcdf" -- see comment in run_dini() re: mixing HDF5 libraries.
    aligned.to_netcdf(output_dir / "nowcast_aligned.nc", engine="h5netcdf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Point validation against pyranometer ground truth")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--pyranometer-dir", type=str, required=True,
                         help="Directory containing risoe/ and lyngby/ subdirectories")
    parser.add_argument("--station", choices=STATION_IDS, default=None,
                         help="Restrict validation to a single pyranometer station (default: both)")
    parser.add_argument("--daytime-only", action="store_true", default=False,
                         help="Filter to daytime hours only (sunrise to sunset)")
    parser.add_argument("--mode", choices=["both", "nowcast", "dini"], default="both",
                         help="Which comparison(s) to run (default: both)")
    parser.add_argument("--dini-path", type=str, default=None,
                         help="Path to the DINI zarr store (required for --mode both/dini)")
    parser.add_argument("--nwc-dir", type=str, default=None,
                         help="Directory with satellite nowcast files (required for --mode both/nowcast)")
    parser.add_argument("--output-dir", type=str, default="results/pyranometer")
    parser.add_argument("--nowcast_ghi_var", type=str, default="probabilistic_advection")
    args = parser.parse_args()

    if args.mode in ("both", "dini") and not args.dini_path:
        parser.error("--dini-path is required for --mode both/dini")
    if args.mode in ("both", "nowcast") and not args.nwc_dir:
        parser.error("--nwc-dir is required for --mode both/nowcast")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    station_ids = [args.station] if args.station else STATION_IDS

    print("Loading pyranometer observations...")
    ground_obs = load_pyranometer_obs(args.pyranometer_dir, args.start, args.end, station_ids)

    if args.mode in ("both", "dini"):
        run_dini(args, ground_obs, output_dir, station_ids, args.daytime_only)
    if args.mode in ("both", "nowcast"):
        run_nowcast(args, ground_obs, output_dir, station_ids, args.daytime_only)

    print(f"\nDone. Wrote scores to {output_dir}")


if __name__ == "__main__":
    main()
