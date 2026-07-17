"""Tests for plot_model_comparison.py using small synthetic score CSVs.

No server/network data is required: two synthetic model directories are built
with the same schema written by run_validation.py, then both plot modes are
exercised end-to-end and the output PNGs are asserted to exist.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Load plot_model_comparison.py (a top-level script, not a package module).
_SCRIPT = Path(__file__).resolve().parent.parent / "plot_model_comparison.py"
_spec = importlib.util.spec_from_file_location("plot_model_comparison", _SCRIPT)
pmc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pmc)


LEAD_TIMES = list(range(0, 361, 15))  # 0 .. 360 min


def _write_synthetic_model(model_dir: Path, days, base_bias: float) -> None:
    """Write one scores_YYYYMMDD.csv per day for a synthetic model."""
    model_dir.mkdir(parents=True, exist_ok=True)
    for day in days:
        rows = []
        init_time = pd.Timestamp(day) + pd.Timedelta(hours=12)
        for lt in LEAD_TIMES:
            valid_time = init_time + pd.Timedelta(minutes=lt)
            mae = base_bias + lt * 0.1
            rmse = mae * 1.4
            rows.append(
                {
                    "initialization_time": init_time,
                    "ensemble": 0,
                    "valid_time": valid_time,
                    "lead_time_minutes": lt,
                    "mae_by_init": mae,
                    "rmse_by_init": rmse,
                }
            )
        df = pd.DataFrame(rows)
        tag = pd.Timestamp(day).strftime("%Y%m%d")
        df.to_csv(model_dir / f"scores_{tag}.csv", index=False)


@pytest.fixture
def two_models(tmp_path: Path):
    """Two model dirs spanning two calendar months (Jan and Feb 2025)."""
    days = pd.to_datetime(
        ["2025-01-05", "2025-01-20", "2025-02-05", "2025-02-20"]
    )
    dir_a = tmp_path / "v1.0.0"
    dir_b = tmp_path / "v1.0.1"
    _write_synthetic_model(dir_a, days, base_bias=10.0)
    _write_synthetic_model(dir_b, days, base_bias=8.0)
    return dir_a, dir_b


def test_leadtime_line_mode(two_models, tmp_path):
    dir_a, dir_b = two_models
    out_dir = tmp_path / "plots"

    curves = {
        "v1.0.0": pmc.collect_leadtime_curve(dir_a, "both"),
        "v1.0.1": pmc.collect_leadtime_curve(dir_b, "both"),
    }
    # Both models cover the full 0..360 lead-time range.
    for curve in curves.values():
        assert list(curve.index) == LEAD_TIMES

    out_path = pmc.plot_leadtime_line(curves, "both", out_dir)
    assert out_path.exists()
    assert out_path.name == "comparison_leadtime_line_both.png"


def test_monthly_bars_mode_15_and_30(two_models, tmp_path):
    dir_a, dir_b = two_models
    out_dir = tmp_path / "plots"

    produced = []
    for lead_time in (15, 30):
        monthly = {
            "v1.0.0": pmc.collect_monthly_for_leadtime(dir_a, lead_time, "both"),
            "v1.0.1": pmc.collect_monthly_for_leadtime(dir_b, lead_time, "both"),
        }
        # Two months present in the synthetic data.
        for df in monthly.values():
            assert list(df.index) == ["202501", "202502"]
        out_path = pmc.plot_monthly_bars(monthly, lead_time, "both", out_dir)
        assert out_path.exists()
        produced.append(out_path.name)

    assert "comparison_monthly_leadtime_15min_both.png" in produced
    assert "comparison_monthly_leadtime_30min_both.png" in produced


def test_missing_lead_time_raises(two_models):
    dir_a, _ = two_models
    with pytest.raises(ValueError, match="999"):
        pmc.collect_monthly_for_leadtime(dir_a, 999, "mae")
