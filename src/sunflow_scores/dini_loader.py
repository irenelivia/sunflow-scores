"""
Point extraction from the DINI NWP GHI forecast (dini_sharded.zarr) at
pyranometer station coordinates.

DINI's 'grad' variable is cumulative-since-forecast-start radiant energy
(J/m2) on a regular grid in its own projection (dini_projection). Extraction
picks the nearest grid cell per station in that projection, then
`time_alignment.deaccumulate_and_expand_to_15min` converts the cumulative
hourly steps into start-labelled, 15-minute average GHI (W/m2) so DINI is
directly comparable with the pyranometer and satellite-nowcast series.
"""

from __future__ import annotations

import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd
import xarray as xr
from cartopy import crs as ccrs
from dask.diagnostics import ProgressBar
from pyproj import CRS, Transformer

from .stations import STATIONS
from .time_alignment import deaccumulate_and_expand_to_15min


class DiniPointLoader:
    """
    Loads DINI GHI forecasts and extracts point time series at pyranometer
    station coordinates.

    Parameters
    ----------
    data_path : str
        Path to the DINI zarr store (e.g. dini_sharded.zarr).
    station_ids : list[str]
        Station keys present in `sunflow_scores.stations.STATIONS` to extract.
    grad_var : str
        Name of the cumulative GHI variable (default 'grad').
    """

    def __init__(self, data_path: str, station_ids: list[str], grad_var: str = "grad"):
        self.data_path = data_path
        self.station_ids = station_ids
        self.grad_var = grad_var

    def _station_xy(self, ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
        """Transform each station's (lon, lat) into the DINI projection's (x, y)."""
        wkt = ds["dini_projection"].attrs["crs_wkt"]
        dini_proj = ccrs.Projection(CRS.from_wkt(wkt))
        transformer = Transformer.from_crs(ccrs.PlateCarree(), dini_proj, always_xy=True)

        lons = [STATIONS[sid]["lon"] for sid in self.station_ids]
        lats = [STATIONS[sid]["lat"] for sid in self.station_ids]
        x, y = transformer.transform(lons, lats)
        return np.asarray(x), np.asarray(y)

    def load_data(self, start_date, end_date) -> xr.Dataset:
        """
        Returns an xr.Dataset with dims (initialization_time, lead_time, station_id)
        and variable 'ghi' (W/m2, start-labelled 15-minute average), matching the
        same convention as SatelliteNowcastLoader's output.
        """
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

        # chunks={} forces dask-backed (parallelisable) reads; without it xarray's
        # zarr backend falls back to a plain lazy, single-threaded array.
        ds = xr.open_dataset(self.data_path, chunks={})
        ds = ds.sel(forecast_reference_time=slice(start_date, end_date))
        n_init = ds.sizes.get("forecast_reference_time", 0)
        if n_init == 0:
            raise ValueError(f"No DINI forecasts found in [{start_date}, {end_date}].")

        x, y = self._station_xy(ds)
        point = ds[[self.grad_var]].sel(
            x=xr.DataArray(x, dims="station_id"),
            y=xr.DataArray(y, dims="station_id"),
            method="nearest",
        )
        point = point.assign_coords(station_id=self.station_ids)

        # Each zarr chunk covers the FULL spatial grid but only one
        # (forecast_reference_time, step) pair (~24MB/chunk), so point
        # extraction still reads one full chunk per (init, step) regardless
        # of how few stations are requested -- runtime scales with the date
        # range, not with the number of stations. Threaded + progress bar so
        # long ranges are visibly progressing rather than appearing to hang.
        n_chunks = n_init * ds.sizes.get("step", 1)
        print(
            f"  Extracting DINI grad at {len(self.station_ids)} station(s): "
            f"{n_chunks} (init, step) chunks to read (~24MB each) from the network mount..."
        )
        with ProgressBar():
            grad = point[self.grad_var].compute(scheduler="threads", num_workers=8)
        ghi = deaccumulate_and_expand_to_15min(grad, step_dim="step")
        ghi.name = "ghi"

        ghi = ghi.rename({"forecast_reference_time": "initialization_time"})
        valid_time = ghi["initialization_time"] + ghi["lead_time"]

        out = ghi.to_dataset()
        out = out.assign_coords(
            valid_time=(("initialization_time", "lead_time"), valid_time.data)
        )
        print(
            f"  Loaded DINI: {out.sizes['initialization_time']} init times x "
            f"{out.sizes['lead_time']} lead times x {out.sizes['station_id']} stations."
        )
        return out
