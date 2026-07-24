"""
Shared time-labelling / resampling rules for reconciling the three GHI sources
used in point (pyranometer) validation, each with a different native
accumulation and labelling convention:

    Satellite nowcast : "12:00" already means avg[12:00, 12:15)  (start-label)
    Pyranometer        : raw samples accumulated over each sample interval;
                          grouping "12:00" = avg[12:00, 12:15)   (start-label)
    DINI               : 'grad' is cumulative-since-forecast-start; the step
                          ending at "13:00" (ref 12:00 + 1h) represents the
                          accumulated energy over [12:00, 13:00), i.e. an
                          END-label convention one hour wide.

DINI values are therefore de-accumulated (diffed across consecutive hourly
steps) and re-labelled to the same start-label, 15-minute convention as the
other two sources before any comparison is made.

Both forecasts also have a real-world *availability latency* -- the delay
between an initialization time and the moment that forecast is actually
downloadable -- which limits which lead times represent a genuinely
available forecast at the time it would be used operationally:

    DINI     : ~3 hours (computation + data-transfer time)
    Nowcast  : ~15-30 minutes

A "lead_time=0" DINI or nowcast value is a real number in the aligned data,
but it was NOT actually available at its own valid_time -- comparing skill
below these latencies would overstate how useful either forecast really is.
`filter_usable_lead_times` drops those unusable short lead times before
scoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

DINI_MIN_USABLE_LEAD_TIME = pd.Timedelta(hours=3)
NOWCAST_MIN_USABLE_LEAD_TIME = pd.Timedelta(minutes=30)  # conservative end of the 15-30min range


def filter_usable_lead_times(
    data: xr.Dataset | xr.DataArray, min_lead_time: pd.Timedelta, lead_dim: str = "lead_time"
) -> xr.Dataset | xr.DataArray:
    """Drop lead times shorter than the forecast's real-world availability latency."""
    return data.sel({lead_dim: slice(min_lead_time, None)})


def resample_high_freq_to_15min(
    df: pd.DataFrame, time_col: str, value_col: str
) -> pd.DataFrame:
    """
    Group high-frequency samples (e.g. 10s or 1min pyranometer readings) into
    15-minute start-labelled bins, e.g. bin "12:00" = mean of samples in
    [12:00, 12:15).

    NaN samples are ignored by the mean (pandas default), so gaps in the raw
    data degrade rather than poison a bin.
    """
    bin_start = df[time_col].dt.floor("15min")
    out = (
        df.assign(**{time_col: bin_start})
        .groupby(time_col, as_index=False)[value_col]
        .mean()
        .sort_values(time_col)
        .reset_index(drop=True)
    )
    return out


def deaccumulate_and_expand_to_15min(
    grad: xr.DataArray, step_dim: str = "step"
) -> xr.DataArray:
    """
    Convert DINI's cumulative-since-forecast-start 'grad' (J/m2) into
    start-labelled, 15-minute-resolution average GHI (W/m2).

    Steps are assumed hourly. For each pair of consecutive steps (k-1, k):
      - the incremental accumulation over [step[k-1], step[k]) is
        grad.isel(step=k) - grad.isel(step=k-1)
      - dividing by 3600s gives the average W/m2 over that hour
      - DINI has no intra-hour resolution, so this single average is
        broadcast to the four 15-minute start-labelled sub-slots within
        that hour: step[k-1], step[k-1]+15min, step[k-1]+30min, step[k-1]+45min

    The first step (index 0, typically step=0h with grad=0) has no preceding
    interval and is dropped from the output.

    Returns a DataArray with `step_dim` replaced by a new 'lead_time'
    dimension at 15-minute resolution, one quarter the length of the
    original hourly deltas.
    """
    steps = grad[step_dim].values
    if len(steps) < 2:
        raise ValueError("Need at least two step values to de-accumulate DINI data.")

    hourly_avg = grad.diff(step_dim) / 3600.0  # W/m2, one value per (step[k-1], step[k]) window
    step_axis = hourly_avg.get_axis_num(step_dim)

    quarter_offsets = np.array([0, 15, 30, 45], dtype="timedelta64[m]").astype("timedelta64[ns]")
    window_starts = steps[:-1]  # step[k-1] for each hourly window
    lead_times = np.repeat(window_starts, 4) + np.tile(quarter_offsets, len(window_starts))

    expanded_values = np.repeat(hourly_avg.values, 4, axis=step_axis)

    other_dims = [d for d in hourly_avg.dims if d != step_dim]
    other_coords = {k: v for k, v in hourly_avg.coords.items() if step_dim not in v.dims}

    result = xr.DataArray(
        expanded_values,
        dims=[d if d != step_dim else "lead_time" for d in hourly_avg.dims],
        coords={**other_coords, "lead_time": lead_times},
        name=grad.name,
    )
    return result
