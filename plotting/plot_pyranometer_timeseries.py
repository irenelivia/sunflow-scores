#!/usr/bin/env python3
"""Plot GHI time series: satellite nowcast, DINI, and pyranometer observation.

Reads the aligned NetCDF files written by run_validation_pyranometer.py
(dini_aligned.nc and/or nowcast_aligned.nc) and overlays both forecasts
against the pyranometer observation at a fixed lead time, one figure per
station.

Example
-------
    uv run python plotting/plot_pyranometer_timeseries.py \
        --input-dir results/pyranometer \
        --lead-time-minutes 0 \
        --output-dir plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot nowcast/DINI/pyranometer GHI time series at a fixed lead time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory with dini_aligned.nc and/or nowcast_aligned.nc (from run_validation_pyranometer.py).",
    )
    parser.add_argument("--output-dir", default="plots", help="Directory where plots will be written.")
    parser.add_argument(
        "--lead-time-minutes", type=float, default=0,
        help="Lead time (minutes) to hold fixed; nearest available lead time is used per source.",
    )
    parser.add_argument(
        "--station", nargs="+", default=None,
        help="Station id(s) to plot (default: every station present in the aligned data).",
    )
    return parser.parse_args()


def _series_at_lead_time(ds: xr.Dataset, var: str, lead_time_minutes: float, station: str) -> pd.Series:
    """Fixed-lead-time slice -> pd.Series indexed by valid_time for one station."""
    # method="nearest" must apply only to lead_time (a real-valued index); applying it
    # to station_id too (a string index) breaks pandas' nearest-neighbor distance calc.
    sel = ds.sel(station_id=station).sel(
        lead_time=pd.Timedelta(minutes=lead_time_minutes), method="nearest",
    )
    valid_time = pd.DatetimeIndex(sel["valid_time"].values)
    return pd.Series(sel[var].values, index=valid_time).sort_index()


def plot_station(
    station: str,
    dini_ds: xr.Dataset | None,
    nowcast_ds: xr.Dataset | None,
    lead_time_minutes: float,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)

    obs_plotted = False
    if nowcast_ds is not None:
        nwc = _series_at_lead_time(nowcast_ds, "nowcast", lead_time_minutes, station)
        ax.plot(nwc.index, nwc.values, label="Satellite nowcast", color="C0", linewidth=1.5)
        obs = _series_at_lead_time(nowcast_ds, "observation", lead_time_minutes, station)
        ax.plot(obs.index, obs.values, label="Pyranometer (obs)", color="k", linewidth=1.2, alpha=0.8)
        obs_plotted = True

    if dini_ds is not None:
        dini = _series_at_lead_time(dini_ds, "forecast", lead_time_minutes, station)
        ax.plot(dini.index, dini.values, label="DINI", color="C1", linewidth=1.5)
        if not obs_plotted:
            obs = _series_at_lead_time(dini_ds, "observation", lead_time_minutes, station)
            ax.plot(obs.index, obs.values, label="Pyranometer (obs)", color="k", linewidth=1.2, alpha=0.8)

    ax.set_xlabel("Valid time")
    ax.set_ylabel("GHI (W/m$^2$)")
    ax.set_title(f"{station} -- lead time {lead_time_minutes:.0f} min")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"timeseries_{station}_lead{int(lead_time_minutes)}min.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)

    dini_path = input_dir / "dini_aligned.nc"
    nowcast_path = input_dir / "nowcast_aligned.nc"
    # engine="h5netcdf" matches how run_validation_pyranometer.py wrote these files;
    # mixing engines (netCDF4 vs h5netcdf) in one process can corrupt HDF5 state.
    dini_ds = xr.open_dataset(dini_path, engine="h5netcdf") if dini_path.exists() else None
    nowcast_ds = xr.open_dataset(nowcast_path, engine="h5netcdf") if nowcast_path.exists() else None

    if dini_ds is None and nowcast_ds is None:
        raise FileNotFoundError(
            f"Neither dini_aligned.nc nor nowcast_aligned.nc found in {input_dir}."
        )

    available_stations = set()
    if dini_ds is not None:
        available_stations.update(dini_ds["station_id"].values.tolist())
    if nowcast_ds is not None:
        available_stations.update(nowcast_ds["station_id"].values.tolist())

    stations = args.station if args.station else sorted(available_stations)

    for station in stations:
        plot_station(station, dini_ds, nowcast_ds, args.lead_time_minutes, Path(args.output_dir))


if __name__ == "__main__":
    main()
