from pathlib import Path

import pandas as pd
import pytest

from sunflow_scores.pyranometer_loader import LyngbyPyranometerLoader, RisoePyranometerLoader


@pytest.fixture
def risoe_dir(tmp_path: Path) -> Path:
    d = tmp_path / "risoe"
    d.mkdir()
    times = pd.date_range("2025-01-01T00:00", periods=180, freq="10s")
    df = pd.DataFrame(
        {
            "TmStamp": times.strftime("%Y-%m-%d %H:%M:%S.0000000"),
            "Global_Horizontal_Pyr": 5.0,
            "Diffuse_Horizontal": 1.0,
            "Direct_Normal": 1.0,
        }
    )
    df.to_csv(d / "DTU_PV_Weather_B130_main_weather_data_2025.csv", index=False)
    return d


@pytest.fixture
def lyngby_dir(tmp_path: Path) -> Path:
    d = tmp_path / "lyngby"
    d.mkdir()

    # dtu_2025_MM.csv covers Jan-Mar (pre-cutover)
    times_dtu = pd.date_range("2025-03-31T23:00", periods=90, freq="1min")
    df_dtu = pd.DataFrame(
        {
            "Time(utc)": times_dtu.strftime("%Y-%m-%d %H:%M:%S"),
            "GHI": 3.0,
            "DNI": 0.0,
            "DHI": 0.0,
        }
    )
    df_dtu.to_csv(d / "dtu_2025_03.csv", index=False)

    # SR300 file covers Apr onward (post-cutover)
    times_sr300 = pd.date_range("2025-04-01T00:00", periods=60, freq="1min")
    df_sr300 = pd.DataFrame(
        {
            "time": times_sr300.strftime("%Y-%m-%d %H:%M:%S+00:00"),
            "SR300_45389_GHI_Wm2": 7.0,
        }
    )
    df_sr300.to_csv(d / "SR300_GHI_DTU_Lyngby_2025_04_2026_05.csv", index=False)

    return d


def test_risoe_loader_resamples_to_15min(risoe_dir):
    loader = RisoePyranometerLoader(risoe_dir)
    ds = loader.load_data("2025-01-01", "2025-01-02")

    assert ds["station_id"].values.tolist() == ["risoe"]
    assert (ds["ghi"].values == 5.0).all()
    assert ds.sizes["time"] == 2  # 180 x 10s samples = 30min -> two 15-min bins


def test_lyngby_loader_concatenates_across_cutover(lyngby_dir):
    loader = LyngbyPyranometerLoader(lyngby_dir)
    ds = loader.load_data("2025-03-31T23:00", "2025-04-01T01:00")

    values = ds["ghi"].values.ravel()
    times = pd.DatetimeIndex(ds["time"].values)

    pre_cutover = values[times < LyngbyPyranometerLoader.CUTOVER]
    post_cutover = values[times >= LyngbyPyranometerLoader.CUTOVER]

    assert len(pre_cutover) > 0 and (pre_cutover == 3.0).all()
    assert len(post_cutover) > 0 and (post_cutover == 7.0).all()


def test_lyngby_loader_only_dtu_source_before_cutover(lyngby_dir):
    loader = LyngbyPyranometerLoader(lyngby_dir)
    ds = loader.load_data("2025-03-31T23:00", "2025-03-31T23:59")

    assert (ds["ghi"].values == 3.0).all()
