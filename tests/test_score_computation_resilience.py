"""
Tests for _compute_scores_per_init's resilience to h5netcdf/h5py corruption.

Verifies that when a single (initialization_time, lead_time) cell fails to
compute due to a simulated h5netcdf-corruption-like error, only that specific
cell ends up NaN — the rest of the day (other lead times of the same init,
and other inits) must still be computed normally, not wiped out.
"""
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask import delayed

from sunflow_scores.validator import _compute_scores_per_init


def _build_lazy_array(n_init: int, n_lead: int, bad_cells: set[tuple[int, int]], corrupt: bool = True) -> xr.DataArray:
    """
    Build a lazy (initialization_time, lead_time) DataArray where the cells in
    *bad_cells* raise an h5netcdf-corruption-like KeyError when computed
    (if *corrupt* is True) or a plain ValueError otherwise (to simulate a
    genuine, non-corruption bug that must still propagate).
    """
    rows = []
    for i in range(n_init):
        row_blocks = []
        for l in range(n_lead):
            if (i, l) in bad_cells:
                if corrupt:
                    def _raise_corrupt():
                        raise KeyError("h5netcdf: invalid dataset identifier (_h5ds corruption)")
                    task = delayed(_raise_corrupt)()
                else:
                    def _raise_generic():
                        raise ValueError("some unrelated real bug")
                    task = delayed(_raise_generic)()
                block = da.from_delayed(task, shape=(), dtype="float32")
            else:
                value = np.array(float(i * 100 + l), dtype="float32")
                block = da.from_array(value)
            row_blocks.append(block)
        rows.append(da.stack(row_blocks))
    data = da.stack(rows)

    inits = np.array(
        [np.datetime64("2025-01-01") + np.timedelta64(i, "D") for i in range(n_init)]
    )
    leads = np.array([np.timedelta64(l * 15, "m") for l in range(n_lead)])

    return xr.DataArray(
        data,
        dims=["initialization_time", "lead_time"],
        coords={"initialization_time": inits, "lead_time": leads},
        name="mae_by_init",
    )


def test_single_bad_lead_time_only_nans_that_cell():
    """A corrupt lead_time within an otherwise-good init must not wipe out the whole init."""
    n_init, n_lead = 2, 3
    bad_cells = {(0, 1)}  # init 0, lead 1 is corrupt

    mae_lazy = _build_lazy_array(n_init, n_lead, bad_cells)
    rmse_lazy = _build_lazy_array(n_init, n_lead, bad_cells)

    mae_result, rmse_result = _compute_scores_per_init(mae_lazy, rmse_lazy)

    mae_values = mae_result.values
    rmse_values = rmse_result.values

    # Only the single corrupt cell should be NaN.
    assert np.isnan(mae_values[0, 1])
    assert np.isnan(rmse_values[0, 1])

    # All other cells (including the other lead times of the SAME init) must
    # be computed normally, not filled with NaN.
    for i in range(n_init):
        for l in range(n_lead):
            if (i, l) == (0, 1):
                continue
            assert not np.isnan(mae_values[i, l]), f"cell ({i},{l}) unexpectedly NaN"
            assert mae_values[i, l] == pytest.approx(float(i * 100 + l))


def test_fully_corrupt_init_is_entirely_nan():
    """If every lead_time of an init is corrupt, the whole init is NaN (as before)."""
    n_init, n_lead = 2, 3
    bad_cells = {(1, 0), (1, 1), (1, 2)}  # all lead times of init 1 are corrupt

    mae_lazy = _build_lazy_array(n_init, n_lead, bad_cells)
    rmse_lazy = _build_lazy_array(n_init, n_lead, bad_cells)

    mae_result, rmse_result = _compute_scores_per_init(mae_lazy, rmse_lazy)

    assert np.isnan(mae_result.values[1, :]).all()
    assert not np.isnan(mae_result.values[0, :]).any()


def test_non_corruption_error_still_propagates():
    """A genuine (non-corruption) exception must not be silently swallowed as NaN."""
    n_init, n_lead = 1, 2
    bad_cells = {(0, 0)}

    mae_lazy = _build_lazy_array(n_init, n_lead, bad_cells, corrupt=False)
    rmse_lazy = _build_lazy_array(n_init, n_lead, bad_cells, corrupt=False)

    with pytest.raises(ValueError, match="some unrelated real bug"):
        _compute_scores_per_init(mae_lazy, rmse_lazy)
