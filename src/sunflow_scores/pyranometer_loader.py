"""
Loaders for the two DTU pyranometer stations (Risoe, Lyngby).

Each raw source has its own file layout and native sampling rate; both are
resampled to the shared 15-minute, start-labelled convention described in
`time_alignment.py` before being exposed as an xr.Dataset(time, station_id),
the same shape produced by `validator.GroundObservationLoader.load_data()`.

Note: raw pyranometer readings can be negative at night (dark-offset /
thermal-signal artefacts) or otherwise need quality control; that is
intentionally out of scope here and left to dedicated QC tooling (e.g. the
`solarpy` package) further up the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from .stations import STATIONS
from .time_alignment import resample_high_freq_to_15min


def _to_station_dataset(df: pd.DataFrame, station_id: str) -> xr.Dataset:
    """Build the (time, station_id) xr.Dataset shape shared with GroundObservationLoader."""
    meta = STATIONS[station_id]
    return xr.Dataset(
        {"ghi": (("time", "station_id"), df["ghi"].values[:, None])},
        coords={
            "time": pd.DatetimeIndex(df["time"]),
            "station_id": [station_id],
            "lat": ("station_id", [meta["lat"]]),
            "lon": ("station_id", [meta["lon"]]),
        },
    )


class RisoePyranometerLoader:
    """
    Loads DTU Risoe pyranometer CSVs (10-second samples).

    Expected columns: TmStamp, Global_Horizontal_Pyr (GHI, W/m2).
    File naming: DTU_PV_Weather_B130_main_weather_data_*.csv
    """

    STATION_ID = "risoe"

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_data(self, start_date, end_date) -> xr.Dataset:
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

        files = sorted(self.data_dir.glob("DTU_PV_Weather_B130_main_weather_data_*.csv"))
        if not files:
            raise ValueError(f"No Risoe pyranometer files found in {self.data_dir}")

        frames = []
        for f in files:
            df = pd.read_csv(f, usecols=["TmStamp", "Global_Horizontal_Pyr"])
            df["time"] = pd.to_datetime(df["TmStamp"])
            df["ghi"] = df["Global_Horizontal_Pyr"]
            frames.append(df[["time", "ghi"]])

        raw = pd.concat(frames, ignore_index=True)
        raw = raw[(raw["time"] >= start_date) & (raw["time"] <= end_date)]
        if raw.empty:
            raise ValueError(f"No Risoe observations in [{start_date}, {end_date}].")

        binned = resample_high_freq_to_15min(raw, time_col="time", value_col="ghi")
        print(f"  Loaded Risoe: {len(raw)} raw samples -> {len(binned)} 15-min bins.")
        return _to_station_dataset(binned, self.STATION_ID)


class LyngbyPyranometerLoader:
    """
    Loads DTU Lyngby pyranometer CSVs (1-minute samples).

    Two raw sources from the same instrument, sent separately by the data
    provider, are concatenated by date range:
      - dtu_2025_MM.csv                          (Time(utc), GHI, ...)  Jan-Mar 2025
      - SR300_GHI_DTU_Lyngby_2025_04_2026_05.csv (time, SR300_*_GHI_Wm2) Apr 2025 onward
    """

    STATION_ID = "lyngby"
    CUTOVER = pd.Timestamp("2025-04-01")

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def _load_dtu_files(self) -> pd.DataFrame:
        files = sorted(self.data_dir.glob("dtu_2025_*.csv"))
        frames = []
        for f in files:
            df = pd.read_csv(f, usecols=["Time(utc)", "GHI"])
            df["time"] = pd.to_datetime(df["Time(utc)"])
            df["ghi"] = df["GHI"]
            frames.append(df[["time", "ghi"]])
        if not frames:
            return pd.DataFrame(columns=["time", "ghi"])
        return pd.concat(frames, ignore_index=True)

    def _load_sr300_files(self) -> pd.DataFrame:
        files = sorted(self.data_dir.glob("SR300_GHI_DTU_Lyngby_*.csv"))
        frames = []
        for f in files:
            df = pd.read_csv(f)
            ghi_col = next(c for c in df.columns if c != "time")
            df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
            df["ghi"] = df[ghi_col]
            frames.append(df[["time", "ghi"]])
        if not frames:
            return pd.DataFrame(columns=["time", "ghi"])
        return pd.concat(frames, ignore_index=True)

    def load_data(self, start_date, end_date) -> xr.Dataset:
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

        segments = []
        if start_date < self.CUTOVER:
            dtu = self._load_dtu_files()
            segments.append(dtu[dtu["time"] < self.CUTOVER])
        if end_date >= self.CUTOVER:
            sr300 = self._load_sr300_files()
            segments.append(sr300[sr300["time"] >= self.CUTOVER])

        if not segments:
            raise ValueError(f"No Lyngby source selected for range [{start_date}, {end_date}].")

        raw = pd.concat(segments, ignore_index=True).sort_values("time")
        raw = raw[(raw["time"] >= start_date) & (raw["time"] <= end_date)]
        if raw.empty:
            raise ValueError(f"No Lyngby observations in [{start_date}, {end_date}].")

        binned = resample_high_freq_to_15min(raw, time_col="time", value_col="ghi")
        print(f"  Loaded Lyngby: {len(raw)} raw samples -> {len(binned)} 15-min bins.")
        return _to_station_dataset(binned, self.STATION_ID)
