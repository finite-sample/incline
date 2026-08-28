"""Tests for the scale sweep.

The property that matters is calibration on null data. The previous SiZer
carried its own variance formulas and the spline branch flagged roughly 90% of
pure noise as trending; the module docstring said so. SiZer no longer computes
standard errors at all, so the map inherits whatever the smoother's calibration
is -- and that is what these tests pin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from incline.process import StateSpace
from incline.sizer import SiZer, SiZerMap, sizer_analysis, trend_with_sizer
from incline.smoothers import LocalPolynomial, Loess, NaiveDifference, SavitzkyGolay

N = 120


def series(seed: int = 0, trend: bool = True) -> pd.DataFrame:
    """A test series, with or without an underlying trend."""
    x = np.arange(N, dtype=float)
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 1.0, N)
    if trend:
        values = 0.05 * x + 2 * np.sin(x / 15) + rng.normal(0, 0.4, N)
    return pd.DataFrame({"value": values}, index=pd.date_range("2020-01-01", periods=N))


def test_sweep_returns_a_populated_map():
    """The sweep fills a scale-by-position grid."""
    result = sizer_analysis(series(), n_scales=8)
    assert isinstance(result, SiZerMap)
    assert result.significance.shape == (8, N)
    assert result.derivative.shape == result.standard_error.shape == (8, N)
    assert set(np.unique(result.significance)) <= {-1, 0, 1}
    assert np.all(np.diff(result.scales) > 0)


@pytest.mark.parametrize("smoother", [NaiveDifference(), StateSpace()])
def test_sweep_refuses_an_estimator_without_a_scale_knob(smoother):
    """A scale map cannot repeat one unchanged fit under several labels."""
    with pytest.raises(ValueError, match="no smoothing scale"):
        SiZer(smoother=smoother, n_scales=3, simultaneous=False).fit(series())


def test_map_renders_as_a_long_frame():
    """One row per (scale, point), for plotting outside matplotlib."""
    frame = sizer_analysis(series(), n_scales=5).to_frame()
    assert list(frame.columns) == [
        "x",
        "scale",
        "derivative",
        "derivative_standard_error",
        "significance",
    ]
    assert len(frame) == 5 * N


@pytest.mark.parametrize(
    "smoother",
    [
        pytest.param(LocalPolynomial(degree=2), id="local_poly"),
        pytest.param(SavitzkyGolay(degree=3), id="sgolay"),
    ],
)
@pytest.mark.parametrize("simultaneous", [False, True])
def test_pure_noise_is_not_flagged_as_trending(smoother, simultaneous):
    """The headline regression: a null series must stay mostly unflagged.

    Pointwise flags should land near the nominal 5%; a whole-curve band should
    be far stricter. The old spline branch reported ~90% here.
    """
    rates = []
    for seed in range(5):
        result = SiZer(smoother=smoother, n_scales=8, simultaneous=simultaneous).fit(
            series(seed=100 + seed, trend=False)
        )
        rates.append(float((result.significance != 0).mean()))

    flagged = float(np.mean(rates))
    assert flagged < (0.02 if simultaneous else 0.12), (
        f"flagged {flagged:.1%} of cells on pure noise"
    )


def test_a_real_trend_is_found():
    """Calibration must not have been bought by flagging nothing."""
    result = sizer_analysis(series(trend=True), n_scales=8)
    assert (result.significance == 1).mean() > 0.2


def test_whole_curve_band_is_stricter_than_pointwise():
    """The multiplicity correction can only remove flags, never add them."""
    data = series(trend=True)
    pointwise = SiZer(n_scales=6, simultaneous=False).fit(data)
    whole = SiZer(n_scales=6, simultaneous=True).fit(data)
    assert (whole.significance != 0).sum() <= (pointwise.significance != 0).sum()


def test_bootstrapped_smoother_refuses_an_unavailable_simultaneous_band():
    """A requested correction must not silently change to pointwise inference."""
    with pytest.raises(ValueError, match="simultaneous"):
        SiZer(smoother=Loess(), n_scales=3, simultaneous=True).fit(series())


def test_native_smoother_refuses_an_unavailable_simultaneous_band():
    """SiZer must not turn an incompatible request into an all-blank map."""
    from incline import GaussianProcess

    with pytest.raises(ValueError, match="simultaneous"):
        SiZer(
            smoother=GaussianProcess(optimize=False, n_restarts=0),
            n_scales=3,
            simultaneous=True,
        ).fit(series())


def test_persistent_regions_require_agreement_across_scales():
    """Demanding more consecutive scales can only shrink the flagged set."""
    result = sizer_analysis(series(trend=True), n_scales=10)
    lenient = result.significant_regions(min_persistence=2)
    strict = result.significant_regions(min_persistence=8)

    def covered(regions):
        return sum(end - start for start, end in regions["increasing"])

    assert covered(strict) <= covered(lenient)


def test_explicit_scales_are_used_verbatim():
    """A caller who names the scales gets those scales."""
    wanted = np.array([0.05, 0.15, 0.4])
    result = SiZer(scales=wanted, simultaneous=False).fit(series())
    np.testing.assert_allclose(result.scales, wanted)


def test_short_series_is_refused():
    """Four points cannot support a scale sweep."""
    tiny = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="at least 5"):
        SiZer().fit(tiny)


def test_sweep_accepts_a_datetime_index_and_a_time_column():
    """Both time sources must give the same answer on the same series."""
    x = np.arange(N, dtype=float)
    rng = np.random.default_rng(3)
    values = 0.05 * x + rng.normal(0, 0.3, N)

    by_index = SiZer(n_scales=4, simultaneous=False).fit(
        pd.DataFrame({"value": values}, index=pd.date_range("2020-01-01", periods=N))
    )
    by_column = SiZer(n_scales=4, simultaneous=False).fit(
        pd.DataFrame({"value": values, "t": x}), time_column="t"
    )
    np.testing.assert_allclose(by_index.derivative, by_column.derivative, atol=1e-9)


def test_trend_with_sizer_attaches_persistence_columns():
    """The convenience wrapper keeps the estimator's schema and adds to it."""
    result = trend_with_sizer(series(trend=True), n_scales=6)
    for column in (
        "derivative_value",
        "derivative_standard_error",
        "uncertainty_method",
        "sizer_significance",
        "persistent_increasing",
        "persistent_decreasing",
    ):
        assert column in result.columns
    assert len(result) == N


def test_trend_with_sizer_uses_one_uncertainty_configuration():
    """The displayed trend and scale map must use the same requested noise model."""
    from incline.api import estimate

    data = series(trend=True)
    smoother = SavitzkyGolay(window_length=15)
    result = trend_with_sizer(
        data,
        smoother=smoother,
        n_scales=3,
        noise="ar1",
        simultaneous=False,
        confidence_level=0.9,
        random_state=7,
    )
    expected = estimate(
        smoother,
        data,
        with_uncertainty=True,
        noise="ar1",
        simultaneous=False,
        confidence_level=0.9,
        random_state=7,
    )
    np.testing.assert_allclose(
        result["derivative_standard_error"], expected.standard_error
    )


def test_bootstrapped_sizer_is_reproducible_from_one_random_state():
    """One public seed must control every bootstrap in the scale sweep."""
    config = SiZer(
        smoother=Loess(),
        n_scales=2,
        simultaneous=False,
        n_bootstrap=10,
        random_state=17,
    )
    first = config.fit(series())
    second = config.fit(series())
    np.testing.assert_array_equal(first.standard_error, second.standard_error)


def test_plot_returns_a_figure():
    """The map renders without a display attached."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = sizer_analysis(series(), n_scales=4).plot()
    assert figure is not None
    plt.close(figure)


@pytest.mark.parametrize("figsize", [(0, 8), (12, np.inf), ("wide", 8)])
def test_plot_rejects_invalid_figure_sizes(figsize):
    """Matplotlib should not be the first layer to diagnose an invalid size."""
    result = sizer_analysis(series(), n_scales=4)
    with pytest.raises(ValueError, match="figsize"):
        result.plot(figsize=figsize)


def test_missing_values_are_refused_rather_than_silently_swept():
    """Regression: the finite count was checked and then the raw series used.

    A NaN propagated through every smoother at every scale, so the map came back
    entirely blank -- which reads as "nothing is trending here" rather than
    "nothing could be computed".
    """
    values = series()["value"].to_numpy(copy=True)
    values[17] = np.nan
    frame = pd.DataFrame(
        {"value": values}, index=pd.date_range("2020-01-01", periods=N)
    )
    with pytest.raises(ValueError, match="missing value"):
        sizer_analysis(frame)
