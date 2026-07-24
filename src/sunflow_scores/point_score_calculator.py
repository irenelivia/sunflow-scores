"""
Score calculator for forecasts that are already point-extracted (e.g. DINI,
via DiniPointLoader) against point-based ground station observations (e.g.
pyranometers, via RisoePyranometerLoader / LyngbyPyranometerLoader).

This differs from validator.GroundScoreCalculator in that it does not need to
find the nearest grid cell per station -- both inputs already carry a
'station_id' dimension. Use GroundScoreCalculator instead when the forecast
side is still on a regular lat/lon grid (e.g. the satellite nowcast).
"""

from __future__ import annotations

import numpy as np
import xarray as xr


class PointScoreCalculator:
    """
    Aligns a point-extracted forecast and point-based ground observations by
    valid_time and station, and computes MAE / RMSE.

    Parameters
    ----------
    forecast_data : xr.Dataset
        dims (initialization_time, lead_time, station_id), with a 2D
        'valid_time' coordinate (initialization_time, lead_time). Output of
        DiniPointLoader.load_data().
    ground_obs : xr.Dataset
        dims (time, station_id). Output of a pyranometer loader's load_data().
    forecast_ghi_var, obs_ghi_var : str
    """

    def __init__(
        self,
        forecast_data: xr.Dataset,
        ground_obs: xr.Dataset,
        forecast_ghi_var: str = "ghi",
        obs_ghi_var: str = "ghi",
    ):
        self.forecast_data = forecast_data
        self.ground_obs = ground_obs
        self.forecast_ghi_var = forecast_ghi_var
        self.obs_ghi_var = obs_ghi_var
        self.aligned_data: xr.Dataset | None = None

    def align_data(self) -> xr.Dataset:
        """
        Match forecast and observation stations, then align by valid_time.
        Valid_times with no matching observation are left as NaN (ground
        station data commonly has gaps from outages).
        """
        common_stations = [
            s for s in self.forecast_data["station_id"].values
            if s in set(self.ground_obs["station_id"].values)
        ]
        if not common_stations:
            raise ValueError("No overlapping station_id between forecast and ground observations.")

        fcst = self.forecast_data.sel(station_id=common_stations)
        obs = self.ground_obs.sel(station_id=common_stations)

        valid_time = fcst["valid_time"]
        flat_valid_times = xr.DataArray(
            np.asarray(valid_time.values).reshape(-1), dims="sample"
        )
        selected = obs.reindex(time=flat_valid_times.values).rename({"time": "sample"})

        n_init = fcst.sizes["initialization_time"]
        n_lead = fcst.sizes["lead_time"]
        n_station = len(common_stations)
        dims = ("initialization_time", "lead_time", "station_id")
        coords = {
            "initialization_time": fcst["initialization_time"],
            "lead_time": fcst["lead_time"],
            "station_id": common_stations,
        }

        obs_data = selected[self.obs_ghi_var].transpose("sample", "station_id").values.reshape(
            (n_init, n_lead, n_station)
        )

        self.aligned_data = xr.Dataset(
            {
                "forecast": (dims, fcst[self.forecast_ghi_var].values),
                "observation": (dims, obs_data),
            },
            coords=coords,
        ).assign_coords(valid_time=(("initialization_time", "lead_time"), valid_time.data))

        n_missing = int(np.isnan(obs_data).sum())
        print(
            f"  Point alignment complete: {n_station} station(s), "
            f"{n_missing}/{obs_data.size} valid_time steps missing an observation."
        )
        return self.aligned_data

    def calculate_mae_by_init(self) -> xr.DataArray:
        """MAE for each (initialization_time, lead_time) averaged over stations."""
        if self.aligned_data is None:
            raise RuntimeError("Call .align_data() first.")
        abs_err = np.abs(self.aligned_data["forecast"] - self.aligned_data["observation"])
        out = abs_err.mean(dim="station_id", skipna=True)
        out.name = "mae_by_init"
        return out

    def calculate_rmse_by_init(self) -> xr.DataArray:
        """RMSE for each (initialization_time, lead_time) averaged over stations."""
        if self.aligned_data is None:
            raise RuntimeError("Call .align_data() first.")
        sq_err = (self.aligned_data["forecast"] - self.aligned_data["observation"]) ** 2
        out = np.sqrt(sq_err.mean(dim="station_id", skipna=True))
        out.name = "rmse_by_init"
        return out

    def calculate_mae_by_station(self) -> xr.DataArray:
        """MAE for each station averaged over all (initialization_time, lead_time) pairs."""
        if self.aligned_data is None:
            raise RuntimeError("Call .align_data() first.")
        abs_err = np.abs(self.aligned_data["forecast"] - self.aligned_data["observation"])
        out = abs_err.mean(dim=["initialization_time", "lead_time"], skipna=True)
        out.name = "mae_by_station"
        return out

    def calculate_rmse_by_station(self) -> xr.DataArray:
        """RMSE for each station averaged over all (initialization_time, lead_time) pairs."""
        if self.aligned_data is None:
            raise RuntimeError("Call .align_data() first.")
        sq_err = (self.aligned_data["forecast"] - self.aligned_data["observation"]) ** 2
        out = np.sqrt(sq_err.mean(dim=["initialization_time", "lead_time"], skipna=True))
        out.name = "rmse_by_station"
        return out
