import numpy as np
import pandas as pd
import xarray as xr

from sunflow_scores.point_score_calculator import PointScoreCalculator


def _forecast_dataset():
    init = pd.to_datetime(["2025-06-01T00:00", "2025-06-01T01:00"])
    lead = pd.to_timedelta([0, 15], unit="min")
    stations = ["risoe", "lyngby"]

    ghi = np.array(
        [
            [[100.0, 200.0], [110.0, 210.0]],
            [[120.0, 220.0], [130.0, 230.0]],
        ]
    )  # (init, lead, station)

    valid_time = np.array(
        [[t + lt for lt in lead] for t in init], dtype="datetime64[ns]"
    )

    return xr.Dataset(
        {"ghi": (("initialization_time", "lead_time", "station_id"), ghi)},
        coords={
            "initialization_time": init,
            "lead_time": lead,
            "station_id": stations,
            "valid_time": (("initialization_time", "lead_time"), valid_time),
        },
    )


def _ground_obs_dataset():
    times = pd.to_datetime(
        ["2025-06-01T00:00", "2025-06-01T00:15", "2025-06-01T01:00", "2025-06-01T01:15"]
    )
    ghi = np.array(
        [
            [90.0, 190.0],
            [100.0, 200.0],
            [110.0, 210.0],
            [120.0, 220.0],
        ]
    )  # (time, station)
    return xr.Dataset(
        {"ghi": (("time", "station_id"), ghi)},
        coords={"time": times, "station_id": ["risoe", "lyngby"]},
    )


def test_align_data_matches_by_valid_time_and_station():
    calc = PointScoreCalculator(_forecast_dataset(), _ground_obs_dataset())
    aligned = calc.align_data()

    # forecast[init=00:00, lead=0min, risoe] = 100.0, obs at valid_time 00:00 risoe = 90.0
    assert aligned["forecast"].sel(
        initialization_time="2025-06-01T00:00", lead_time=pd.Timedelta(0), station_id="risoe"
    ).item() == 100.0
    assert aligned["observation"].sel(
        initialization_time="2025-06-01T00:00", lead_time=pd.Timedelta(0), station_id="risoe"
    ).item() == 90.0


def test_calculate_mae_rmse_by_station():
    calc = PointScoreCalculator(_forecast_dataset(), _ground_obs_dataset())
    calc.align_data()

    mae_by_station = calc.calculate_mae_by_station()
    rmse_by_station = calc.calculate_rmse_by_station()

    # Every forecast/obs pair here differs by exactly 10.0 W/m2
    np.testing.assert_allclose(mae_by_station.values, [10.0, 10.0])
    np.testing.assert_allclose(rmse_by_station.values, [10.0, 10.0])


def test_calculate_mae_rmse_by_init_averages_over_stations():
    calc = PointScoreCalculator(_forecast_dataset(), _ground_obs_dataset())
    calc.align_data()

    mae_by_init = calc.calculate_mae_by_init()
    assert mae_by_init.sizes == {"initialization_time": 2, "lead_time": 2}
    np.testing.assert_allclose(mae_by_init.values, [[10.0, 10.0], [10.0, 10.0]])


def test_missing_observation_becomes_nan_not_error():
    obs = _ground_obs_dataset().sel(time=slice("2025-06-01T00:00", "2025-06-01T00:00"))
    calc = PointScoreCalculator(_forecast_dataset(), obs)
    aligned = calc.align_data()

    # valid_time 01:00 / 01:15 have no matching observation -> NaN, not a raised error
    assert np.isnan(
        aligned["observation"].sel(
            initialization_time="2025-06-01T01:00", lead_time=pd.Timedelta(0), station_id="risoe"
        ).item()
    )
