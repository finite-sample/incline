"""Tests for the time axis.

Every estimator derives its x from here, so a mistake in this module is a
mistake in all of them. That is not hypothetical: the derivation used to be
copied across fifteen call sites, and one copy returned a pandas ``Index``
instead of an ndarray, which crashed ``local_polynomial_trend`` on any
``DatetimeIndex`` -- the package's most common input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from incline.axis import REGULARITY_TOLERANCE, TimeAxis


def test_datetime_index_becomes_days_from_the_start():
    """Daily stamps give unit spacing measured in days."""
    axis = TimeAxis.from_index(pd.date_range("2020-01-01", periods=5))
    assert isinstance(axis.x, np.ndarray)
    assert axis.x.dtype == np.float64
    np.testing.assert_allclose(axis.x, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert axis.delta == 1.0
    assert axis.unit == "days"


def test_derived_axis_is_a_real_ndarray():
    """Regression: it used to be a pandas Index, which has no .reshape.

    Downstream code does array arithmetic on this, so the type is part of the
    contract rather than an implementation detail.
    """
    axis = TimeAxis.from_index(pd.date_range("2020-01-01", periods=5))
    assert hasattr(axis.x, "reshape")
    assert axis.x.reshape(-1, 1).shape == (5, 1)


def test_sub_daily_stamps_keep_their_fractional_spacing():
    """An hourly series is 1/24 of a day apart, not one unit."""
    axis = TimeAxis.from_index(pd.date_range("2020-01-01", periods=4, freq="6h"))
    assert axis.delta == pytest.approx(0.25)


@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_integer_and_float_time_columns_agree(dtype):
    """Regression: an integer time column must not be read as positional.

    ``np.int64`` is not a subclass of ``int``, so a naive isinstance check
    missed it. With a column stepping by 10, falling back to the positional
    index turns a slope of 2.0 into 20.0.
    """
    times = (np.arange(20) * 10).astype(dtype)
    axis = TimeAxis.from_frame(pd.DataFrame({"t": times}), time_column="t")
    assert axis.delta == pytest.approx(10.0)
    np.testing.assert_allclose(axis.x, np.arange(20) * 10.0)


def test_time_column_wins_over_the_index():
    """An explicit column is an instruction, not a hint."""
    frame = pd.DataFrame(
        {"t": np.arange(0, 50, 10, dtype=float)},
        index=pd.date_range("2020-01-01", periods=5),
    )
    assert TimeAxis.from_frame(frame, time_column="t").delta == pytest.approx(10.0)
    assert TimeAxis.from_frame(frame).delta == pytest.approx(1.0)


def test_plain_numeric_index_is_used_as_is():
    """A RangeIndex needs no conversion."""
    axis = TimeAxis.from_index(pd.RangeIndex(6))
    np.testing.assert_allclose(axis.x, np.arange(6.0))
    assert axis.unit == "index"


def test_positional_axis_is_unit_spaced():
    """The convenience constructor used throughout the tests."""
    axis = TimeAxis.positional(7)
    np.testing.assert_allclose(axis.x, np.arange(7.0))
    assert axis.n == 7
    assert axis.span == 6.0


def test_uniform_sampling_is_regular():
    """A grid has no spacing variation."""
    axis = TimeAxis.positional(30)
    assert axis.spacing_cv == 0.0
    assert axis.is_regular


def test_uneven_sampling_is_flagged_irregular():
    """Gaps must be visible to methods that assume a grid."""
    axis = TimeAxis.from_index(
        pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-09", "2020-01-10"])
    )
    assert axis.spacing_cv > REGULARITY_TOLERANCE
    assert not axis.is_regular


def test_grid_methods_warn_on_irregular_sampling():
    """Savitzky-Golay is wrong off a grid, so it should say so."""
    axis = TimeAxis.from_index(
        pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-20", "2020-01-21"])
    )
    with pytest.warns(UserWarning, match="uniform sampling"):
        axis.require_regular("Savitzky-Golay")


def test_out_of_order_time_is_refused():
    """Sorting is the caller's job, and silently proceeding hides the error."""
    with pytest.raises(ValueError, match="strictly increasing"):
        TimeAxis.from_index(pd.DatetimeIndex(["2020-01-05", "2020-01-01"]))


def test_duplicate_timestamps_are_refused():
    """Zero spacing has no meaningful derivative."""
    with pytest.raises(ValueError, match="strictly increasing"):
        TimeAxis.from_frame(pd.DataFrame({"t": [1.0, 1.0, 2.0]}), time_column="t")


def test_non_finite_time_is_refused():
    """A NaN in the axis would poison every downstream computation."""
    with pytest.raises(ValueError, match="non-finite"):
        TimeAxis.from_frame(pd.DataFrame({"t": [0.0, np.nan, 2.0]}), time_column="t")


def test_single_point_axis_has_a_usable_delta():
    """One observation has no spacing, so fall back to unit rather than fail."""
    axis = TimeAxis.positional(1)
    assert axis.n == 1
    assert axis.delta == 1.0
    assert axis.span == 0.0


def test_key_distinguishes_axes_and_repeats_for_equal_ones():
    """Operator caching is keyed on this, so it must not collide."""
    first = TimeAxis.positional(10)
    same = TimeAxis.positional(10)
    other = TimeAxis.positional(11)
    assert first.key() == same.key()
    assert first.key() != other.key()
    assert hash(first.key()) == hash(same.key())


def test_period_index_is_a_time_axis():
    """Monthly and quarterly series are ordinarily indexed by period.

    Rejecting them sent numpy a Period object and surfaced
    "float() argument must be ... not 'Period'", when the index carries
    perfectly good time information.
    """
    axis = TimeAxis.from_index(pd.period_range("2020-01", periods=6, freq="M"))
    assert axis.unit == "days"
    # January to February is 31 days; the axis is in days from the start.
    assert axis.x[1] == pytest.approx(31.0)
    assert axis.n == 6


def test_quarterly_period_index_spacing():
    """The same, at a coarser frequency."""
    axis = TimeAxis.from_index(pd.period_range("2020Q1", periods=4, freq="Q"))
    assert axis.x[1] == pytest.approx(91.0)


@pytest.mark.parametrize(
    ("label", "index"),
    [
        ("strings", pd.Index(["a", "b", "c", "d"])),
        ("categorical", pd.CategoricalIndex(list("abcd"))),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_a_non_time_index_says_what_to_do(label, index):
    """An index with no time information must fail with a usable message.

    It previously reached numpy and produced "could not convert string to
    float: 'a'", which names neither the cause nor the remedy.
    """
    del label
    with pytest.raises(ValueError, match="time_column"):
        TimeAxis.from_index(index)
