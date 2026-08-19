"""Tests for the synthetic-series generators.

These generators are the ground truth the calibration suite measures against, so
a silent error here would quietly invalidate every coverage number in
``test_calibration.py``. Two properties matter most: the closed-form derivatives
really are the derivatives, and ``noise_std`` means the same thing for every
noise process.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from incline.axis import TimeAxis
from incline.simulate import (
    ExponentialTrend,
    NoiseGenerator,
    PolynomialTrend,
    SinusoidalTrend,
    StepTrend,
    generate_time_series,
    standard_test_functions,
)
from incline.smoothers import SavitzkyGolay

SMOOTH_TRENDS = [
    pytest.param(PolynomialTrend([1.0, -2.0, 0.5]), id="polynomial"),
    pytest.param(SinusoidalTrend(amplitude=2.0, frequency=0.1), id="sinusoidal"),
    pytest.param(ExponentialTrend(scale=1.5, rate=0.2), id="exponential"),
]


@pytest.mark.parametrize("trend", SMOOTH_TRENDS)
@pytest.mark.parametrize("order", [1, 2])
def test_closed_form_derivatives_match_numerical_ones(trend, order):
    """The analytic derivative must agree with a finite difference."""
    x = np.linspace(0.5, 6.0, 40)
    step = 1e-5
    analytic = trend.derivative(x, order)
    if order == 1:
        numerical = (trend(x + step) - trend(x - step)) / (2 * step)
    else:
        numerical = (trend(x + step) - 2 * trend(x) + trend(x - step)) / step**2

    scale = max(float(np.max(np.abs(numerical))), 1e-6)
    assert np.max(np.abs(analytic - numerical)) / scale < 1e-4


def test_polynomial_derivative_past_its_degree_is_exactly_zero():
    """Differentiating a line twice gives zero, not floating-point dust."""
    line = PolynomialTrend([0.0, 1.0])
    x = np.linspace(0, 5, 10)
    assert np.all(line.derivative(x, 2) == 0.0)
    assert np.all(line.derivative(x, 5) == 0.0)


def test_step_trend_selects_the_right_segment():
    """Breakpoints are left-closed, and the last value extends past the end."""
    step = StepTrend(breakpoints=[3.0, 7.0], values=[1.0, 3.0, 2.0])
    got = step(np.array([0.0, 3.0, 5.0, 7.0, 9.0]))
    assert got.tolist() == [1.0, 1.0, 3.0, 3.0, 2.0]
    assert np.all(step.derivative(np.array([0.0, 5.0])) == 0.0)


def test_step_trend_rejects_too_few_values():
    """A segment without a value is a construction error, not a silent clip."""
    with pytest.raises(ValueError, match="one value per breakpoint"):
        StepTrend(breakpoints=[1.0, 2.0], values=[1.0])


@pytest.mark.parametrize(
    ("label", "draw"),
    [
        ("white", lambda rng: NoiseGenerator.white(60000, 0.5, rng)),
        ("ar1_moderate", lambda rng: NoiseGenerator.ar1(60000, 0.7, 0.5, rng)),
        ("ar1_strong", lambda rng: NoiseGenerator.ar1(60000, 0.9, 0.5, rng)),
        ("seasonal", lambda rng: NoiseGenerator.seasonal(60000, 12, 0.5, 0.8, rng)),
    ],
)
def test_noise_std_is_the_marginal_standard_deviation(label, draw):
    """Every noise type must honour ``std`` as its marginal spread.

    Previously it did not: AR(1) treated it as the innovation standard
    deviation, so at phi=0.9 the series was 2.3x wider than white noise at the
    same setting, and the seasonal generator ignored the argument entirely.
    Comparing methods across noise types then compared different noise levels.
    """
    del label
    assert draw(np.random.default_rng(0)).std() == pytest.approx(0.5, rel=0.05)


def test_ar1_has_the_requested_autocorrelation():
    """The dependence, not just the spread, must come out as asked."""
    noise = NoiseGenerator.ar1(60000, 0.7, 1.0, np.random.default_rng(1))
    centered = noise - noise.mean()
    lag_one = float(np.sum(centered[1:] * centered[:-1]) / np.sum(centered**2))
    assert lag_one == pytest.approx(0.7, abs=0.02)


def test_ar1_rejects_a_nonstationary_coefficient():
    """phi outside the unit circle has no marginal variance to scale to."""
    with pytest.raises(ValueError, match="stationarity"):
        NoiseGenerator.ar1(10, phi=1.0)


def test_generators_do_not_touch_the_global_random_state():
    """Simulating must not perturb a caller's own seeded stream.

    The previous implementation called ``np.random.seed`` internally, so
    generating a series silently reseeded the global generator underneath
    whatever else was running.
    """
    np.random.seed(1234)
    expected = np.random.random()

    np.random.seed(1234)
    NoiseGenerator.white(50, 1.0, 99)
    NoiseGenerator.ar1(50, 0.7, 1.0, 99)
    generate_time_series(PolynomialTrend([0.0, 1.0]), random_state=7)
    assert np.random.random() == expected


def test_generation_is_reproducible_from_a_seed():
    """The same seed gives the same series; a different one does not."""
    first, deriv_first = generate_time_series(
        SinusoidalTrend(), n_points=50, random_state=5
    )
    same, deriv_same = generate_time_series(
        SinusoidalTrend(), n_points=50, random_state=5
    )
    other, _ = generate_time_series(SinusoidalTrend(), n_points=50, random_state=6)

    pd.testing.assert_frame_equal(first, same)
    np.testing.assert_array_equal(deriv_first, deriv_same)
    assert not np.allclose(first["value"], other["value"])


def test_regular_series_carries_a_datetime_index():
    """The default shape is what the estimators take without a time column."""
    frame, derivative = generate_time_series(
        PolynomialTrend([0.0, 2.0]), n_points=40, random_state=3
    )
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert list(frame.columns) == ["value", "true_value", "noise"]
    assert len(derivative) == 40


def test_irregular_series_carries_an_explicit_time_column():
    """Irregular sampling cannot be a DatetimeIndex, so x is a column."""
    frame, _ = generate_time_series(
        PolynomialTrend([0.0, 2.0]),
        n_points=40,
        irregular_spacing=True,
        random_state=3,
    )
    assert "time" in frame.columns
    assert frame["time"].is_monotonic_increasing
    assert frame["time"].diff().iloc[1:].std() > 0


def test_missing_data_blanks_values_but_keeps_the_truth():
    """Only the observed column goes missing; the truth stays for scoring."""
    frame, _ = generate_time_series(
        PolynomialTrend([0.0, 1.0]),
        n_points=200,
        missing_data_prob=0.2,
        random_state=3,
    )
    assert 0 < frame["value"].isna().sum() < 200
    assert frame["true_value"].notna().all()


def test_unknown_noise_type_is_refused():
    """A typo must not silently fall through to some default."""
    with pytest.raises(ValueError, match="Unknown noise type"):
        generate_time_series(PolynomialTrend([0.0, 1.0]), noise_type="pink")


def test_standard_functions_expose_names_and_derivatives():
    """The bundled set is usable as-is by the calibration suite."""
    functions = standard_test_functions()
    assert len(functions) >= 4
    x = np.linspace(0, 10, 20)
    for function in functions:
        assert isinstance(function.name, str)
        assert function.name
        assert function(x).shape == x.shape
        assert function.derivative(x, 1).shape == x.shape


def test_the_datetime_index_matches_the_x_the_derivative_is_stated_on():
    """Regression: the index stepped by days while x stepped by x_range/n.

    The frame carried a daily DatetimeIndex regardless of ``x_range``, so an
    estimator read a spacing of one day while the returned true derivative was
    per unit of x. Over (0, 10) with 100 points those differ by a factor of ten,
    which silently rescaled every calibration measurement taken against it.
    """
    trend = PolynomialTrend([0.0, 2.0])
    frame, truth = generate_time_series(
        trend, n_points=100, x_range=(0.0, 10.0), noise_std=0.0, random_state=0
    )
    axis = TimeAxis.from_index(frame.index)
    np.testing.assert_allclose(axis.x, np.linspace(0.0, 10.0, 100), atol=1e-9)
    np.testing.assert_allclose(
        np.gradient(frame["value"].to_numpy(), axis.x), truth, rtol=1e-6
    )


def test_the_stated_derivative_is_recovered_per_unit_of_the_index():
    """An estimator fed the frame must land on the stated derivative."""
    frame, _ = generate_time_series(
        PolynomialTrend([0.0, 2.0]),
        n_points=80,
        x_range=(0.0, 8.0),
        noise_std=0.0,
        random_state=0,
    )
    estimate = SavitzkyGolay(window_length=11).fit(
        TimeAxis.from_index(frame.index), frame["value"].to_numpy(), order=1
    )
    assert float(np.median(estimate.derivative)) == pytest.approx(2.0, rel=1e-6)
