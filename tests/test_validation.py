import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import pytest

# Make the package module available for testing
from sunflow_scores.validator import ScoreCalculator, SatelliteNowcastLoader, SatelliteObservationLoader

@pytest.fixture
def dummy_data(tmp_path: Path) -> tuple[Path, Path]:
    """
    Creates small, temporary NetCDF files for testing.

    Returns a tuple of paths to the dummy nowcast and observation directories.
    """
    nwc_dir = tmp_path / "nwc"
    obs_dir = tmp_path / "obs"
    nwc_dir.mkdir()
    obs_dir.mkdir()

    # --- Create Dummy Observation Data ---
    # Shared coordinates
    lat = np.arange(55, 56, 0.1)
    lon = np.arange(10, 11, 0.1)
    time_obs = pd.date_range("2025-01-01T12:00", "2025-01-01T15:00", freq="15min")

    # Create a dummy observation file
    obs_data = xr.Dataset(
        {
            "GHI": (("time", "lat", "lon"), np.full((len(time_obs), len(lat), len(lon)), 100.0)),
            "CLEARSKY_GHI": (("time", "lat", "lon"), np.full((len(time_obs), len(lat), len(lon)), 200.0)),
        },
        coords={"time": time_obs, "lat": lat, "lon": lon},
    )
    obs_data.to_netcdf(obs_dir / "obs_20250101.nc")


    # --- Create Dummy Nowcast Data ---
    # Nowcast data has an extra lead_time dimension
    lead_time = pd.to_timedelta(range(0, 61, 15), unit="min")
    time_nwc_init = pd.date_range("2025-01-01T12:00", "2025-01-01T14:00", freq="15min")

    # Create a separate nowcast file for each initialization time
    for init_time in time_nwc_init:
        # Create slightly different data for each file
        # The dummy data should have a 'time' dimension, which the preprocessor will convert to 'lead_time'
        valid_times = init_time + lead_time
        nwc_ghi = np.random.rand(len(valid_times), len(lat), len(lon)) * (150 + init_time.hour)
        nwc_data = xr.Dataset(
            {
                "GHI": (("time", "lat", "lon"), nwc_ghi),
            },
            coords={"time": valid_times, "lat": lat, "lon": lon},
        )
        # The file name must match the loader's expectation
        fname = f"SolarNowcast_{init_time.strftime('%Y%m%d%H%M')}.nc"
        nwc_data.to_netcdf(nwc_dir / fname)

    return nwc_dir, obs_dir


def test_score_calculator_runs(dummy_data):
    """
    Tests that the ScoreCalculator can be initialized and run without errors.
    """
    nwc_dir, obs_dir = dummy_data
    # The date range must match the dummy data we created
    start_date = "2025-01-01T12:00"
    end_date = "2025-01-01T14:00"

    # Load the dummy data using the loaders
    nwc_loader = SatelliteNowcastLoader(data_dir=nwc_dir)
    obs_loader = SatelliteObservationLoader(data_dir=obs_dir)
    nwc_data = nwc_loader.load_data(start_date, end_date)
    obs_data = obs_loader.load_data(start_date, end_date)

    # Initialize the calculator with the loaded data
    calculator = ScoreCalculator(
        nwc_data=nwc_data,
        obs_data=obs_data,
        nowcast_ghi_var="GHI",
        obs_ghi_var="GHI",
        obs_cs_ghi_var="CLEARSKY_GHI",
    )

    # Align data
    calculator.align_data()
    assert calculator.aligned_data is not None
    assert "time" in calculator.aligned_data.dims
    assert "lead_time" in calculator.aligned_data.dims

    # Calculate one score to check for errors
    mae = calculator.calculate_mae(data=calculator.aligned_data, groupby_time_of_day=False)
    assert mae is not None
    assert "lat" in mae.dims
    assert "lon" in mae.dims
    assert "lead_time" in mae.dims
    assert not np.isnan(mae.values).any()
