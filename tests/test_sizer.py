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

from incline.sizer import SiZer, SiZerMap, sizer_analysis, trend_with_sizer
from incline.smoothers import InterpolatingSpline, LocalPolynomial, SavitzkyGolay


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
    assert result.derivative.shape == result.se.shape == (8, N)
    assert set(np.unique(result.significance)) <= {-1, 0, 1}
    assert np.all(np.diff(result.scales) > 0)


def test_map_renders_as_a_long_frame():
    """One row per (scale, point), for plotting outside matplotlib."""
    frame = sizer_analysis(series(), n_scales=5).to_frame()
    assert list(frame.columns) == [
        "x",
        "scale",
        "derivative",
        "derivative_se",
        "significance",
    ]
    assert len(frame) == 5 * N


@pytest.mark.parametrize(
    "smoother",
    [
        pytest.param(LocalPolynomial(degree=2), id="local_poly"),
        pytest.param(SavitzkyGolay(polyorder=3), id="sgolay"),
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


def test_bootstrapped_smoother_reports_pointwise_rather_than_pretending():
    """A whole-curve band needs an exact operator; saying so beats faking it."""
    with pytest.warns(UserWarning, match="whole-curve"):
        result = SiZer(
            smoother=InterpolatingSpline(), n_scales=3, simultaneous=True
        ).fit(series())
    assert result.simultaneous is False


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
        "derivative_se",
        "se_method",
        "sizer_significance",
        "persistent_increasing",
        "persistent_decreasing",
    ):
        assert column in result.columns
    assert len(result) == N


def test_plot_returns_a_figure():
    """The map renders without a display attached."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = sizer_analysis(series(), n_scales=4).plot()
    assert figure is not None
    plt.close(figure)
