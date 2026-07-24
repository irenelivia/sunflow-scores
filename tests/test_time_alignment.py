import numpy as np
import pandas as pd
import xarray as xr

from sunflow_scores.time_alignment import (
    DINI_MIN_USABLE_LEAD_TIME,
    NOWCAST_MIN_USABLE_LEAD_TIME,
    deaccumulate_and_expand_to_15min,
    filter_usable_lead_times,
    resample_high_freq_to_15min,
)


def test_resample_high_freq_to_15min_start_label():
    # 1-minute samples; first 15-min bin [00:00, 00:15) should average to 1.0,
    # second bin [00:15, 00:30) to 2.0.
    times = pd.date_range("2025-01-01T00:00", periods=30, freq="1min")
    values = [1.0] * 15 + [2.0] * 15
    df = pd.DataFrame({"time": times, "ghi": values})

    out = resample_high_freq_to_15min(df, time_col="time", value_col="ghi")

    assert list(out["time"]) == [
        pd.Timestamp("2025-01-01T00:00"),
        pd.Timestamp("2025-01-01T00:15"),
    ]
    assert out["ghi"].tolist() == [1.0, 2.0]


def test_resample_high_freq_to_15min_ignores_nan():
    times = pd.date_range("2025-01-01T00:00", periods=4, freq="1min")
    df = pd.DataFrame({"time": times, "ghi": [1.0, np.nan, 3.0, np.nan]})

    out = resample_high_freq_to_15min(df, time_col="time", value_col="ghi")

    assert out["ghi"].iloc[0] == 2.0  # mean of 1.0 and 3.0, NaNs skipped


def test_deaccumulate_and_expand_to_15min():
    # Cumulative-since-start grad: hour 1 accumulates 10*3600 J/m2 (avg 10 W/m2),
    # hour 2 accumulates 20*3600 J/m2 (avg 20 W/m2).
    steps = pd.to_timedelta([0, 1, 2], unit="h").values
    grad = xr.DataArray(
        [0.0, 10 * 3600, 10 * 3600 + 20 * 3600],
        dims=["step"],
        coords={"step": steps},
    )

    out = deaccumulate_and_expand_to_15min(grad, step_dim="step")

    assert out.sizes["lead_time"] == 8  # 2 hours x 4 quarters
    np.testing.assert_allclose(out.values, [10.0] * 4 + [20.0] * 4)

    expected_lead_times = pd.to_timedelta(
        [0, 15, 30, 45, 60, 75, 90, 105], unit="min"
    ).values
    np.testing.assert_array_equal(out["lead_time"].values, expected_lead_times)


def test_deaccumulate_and_expand_to_15min_requires_two_steps():
    grad = xr.DataArray([0.0], dims=["step"], coords={"step": [np.timedelta64(0, "h")]})
    try:
        deaccumulate_and_expand_to_15min(grad, step_dim="step")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_filter_usable_lead_times_drops_short_dini_lead_times():
    lead_time = pd.to_timedelta([0, 60, 120, 180, 240], unit="min")
    da = xr.DataArray([1, 2, 3, 4, 5], dims=["lead_time"], coords={"lead_time": lead_time})

    out = filter_usable_lead_times(da, DINI_MIN_USABLE_LEAD_TIME)

    np.testing.assert_array_equal(
        out["lead_time"].values, pd.to_timedelta([180, 240], unit="min").values
    )
    assert out.values.tolist() == [4, 5]


def test_filter_usable_lead_times_drops_short_nowcast_lead_times():
    lead_time = pd.to_timedelta([0, 15, 30, 45], unit="min")
    da = xr.DataArray([1, 2, 3, 4], dims=["lead_time"], coords={"lead_time": lead_time})

    out = filter_usable_lead_times(da, NOWCAST_MIN_USABLE_LEAD_TIME)

    np.testing.assert_array_equal(
        out["lead_time"].values, pd.to_timedelta([30, 45], unit="min").values
    )
