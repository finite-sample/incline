"""Tests for seasonal decomposition.

The schema tests carry the most weight. The previous implementation returned
``derivative_value`` from one code path and ``trend_derivative_value`` from
another, so what a caller had to index depended on whether seasonality happened
to be detected. Several tests below exist purely to pin that down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import incline.seasonal as seasonal_module
from incline.noise import AR1, IID, Given, Heteroskedastic, NoiseFit
from incline.process import GaussianProcess
from incline.seasonal import (
    DECOMPOSITION_COLUMNS,
    Seasonality,
    deseasonalize,
    detect_seasonality,
    moving_average_decompose,
    stl_decompose,
    trend_with_deseasonalization,
)
from incline.smoothers import SavitzkyGolay, Smoother, SmoothingSpline

N = 200


def seasonal_series(period: int = 12, seed: int = 0) -> pd.DataFrame:
    """A trending series with a clear cycle."""
    t = np.arange(N, dtype=float)
    rng = np.random.default_rng(seed)
    values = 0.05 * t + 3 * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.4, N)
    return pd.DataFrame({"value": values}, index=pd.date_range("2020-01-01", periods=N))


def plain_series(seed: int = 1) -> pd.DataFrame:
    """A trending series with no cycle."""
    t = np.arange(N, dtype=float)
    values = 0.05 * t + np.random.default_rng(seed).normal(0, 0.4, N)
    return pd.DataFrame({"value": values}, index=pd.date_range("2020-01-01", periods=N))


def dependent_seasonal_series(n: int = 120) -> pd.DataFrame:
    """A seasonal trend observed with known AR(1) noise."""
    t = np.arange(n, dtype=float)
    phi = 0.75
    standard_deviation = 0.5
    rng = np.random.default_rng(22)
    errors = np.empty(n)
    errors[0] = rng.normal(scale=standard_deviation)
    innovation_scale = standard_deviation * np.sqrt(1 - phi**2)
    for index in range(1, n):
        errors[index] = phi * errors[index - 1] + rng.normal(scale=innovation_scale)
    values = 0.025 * t + 2 * np.sin(2 * np.pi * t / 12) + errors
    return pd.DataFrame(
        {"value": values},
        index=pd.date_range("2010-01-31", periods=n, freq="ME"),
    )


def test_detects_a_planted_cycle():
    """A 12-point cycle should be found at 12."""
    found = detect_seasonality(seasonal_series(period=12))
    assert isinstance(found, Seasonality)
    assert found.seasonal
    assert found.period == 12
    assert 0 < found.strength <= 1


def test_reports_no_cycle_when_there_is_none():
    """A trend without a cycle must not manufacture one."""
    assert detect_seasonality(plain_series()).seasonal is False


def test_detection_survives_a_series_too_short_to_analyze():
    """Five points cannot support any detector."""
    tiny = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    assert detect_seasonality(tiny).seasonal is False


@pytest.mark.filterwarnings("ignore:Period .* is unusable:UserWarning")
@pytest.mark.parametrize(
    ("label", "decompose"),
    [
        ("auto_with_cycle", lambda: deseasonalize(seasonal_series())),
        ("auto_without_cycle", lambda: deseasonalize(plain_series())),
        ("explicit_stl", lambda: deseasonalize(seasonal_series(), method="stl")),
        ("explicit_simple", lambda: deseasonalize(seasonal_series(), method="simple")),
        ("odd_period", lambda: moving_average_decompose(seasonal_series(), period=7)),
        ("even_period", lambda: moving_average_decompose(seasonal_series(), period=12)),
    ],
)
def test_every_route_returns_the_same_schema(label, decompose):
    """One schema, whichever path ran.

    This is the regression that motivated the rewrite: callers had to know
    which branch fired before they knew what to index.
    """
    del label
    result = decompose()
    for column in DECOMPOSITION_COLUMNS:
        assert column in result.columns, f"missing {column}"
    assert result["deseasonalized"].notna().all()
    assert len(result) == N


def test_components_reconstruct_the_observed_series():
    """trend + seasonal + residual must add back up."""
    result = deseasonalize(seasonal_series())
    total = (
        result["trend_component"]
        + result["seasonal_component"]
        + result["residual_component"]
    )
    np.testing.assert_allclose(total, result["value"], atol=1e-8)


def test_deseasonalizing_removes_the_cycle():
    """The seasonal swing should be gone from the adjusted series."""
    data = seasonal_series(period=12)
    result = deseasonalize(data, method="stl")

    # Compare the amplitude at the seasonal frequency, not the total spread,
    # since the trend dominates the variance either way.
    def cycle_amplitude(values):
        t = np.arange(len(values))
        wave = np.exp(-2j * np.pi * t / 12)
        return float(np.abs(np.sum((values - np.mean(values)) * wave)) / len(values))

    assert cycle_amplitude(result["deseasonalized"]) < 0.3 * cycle_amplitude(
        data["value"]
    )


def test_no_cycle_leaves_the_series_untouched():
    """With nothing to remove, the adjusted series is the original."""
    data = plain_series()
    result = deseasonalize(data)
    assert result["decomposition_method"].iloc[0] == "none"
    np.testing.assert_allclose(result["deseasonalized"], data["value"])


def test_unknown_method_is_refused():
    """A typo must not silently pick a default."""
    with pytest.raises(ValueError, match="Unknown decomposition method"):
        deseasonalize(seasonal_series(), method="magic")


@pytest.mark.parametrize("period", [7, 12])
def test_moving_average_handles_odd_and_even_periods(period):
    """Regression: the odd-period path wrote into a read-only rolling view."""
    # The series keeps its own default cycle on purpose: what varies here is
    # the decomposition window, odd against even. Threading period into the
    # data too would change both sides at once.
    series = seasonal_series()  # preen: allow-dropped-arg
    result = moving_average_decompose(series, period=period)
    assert result["trend_component"].notna().all()
    assert result["deseasonalized"].notna().all()


@pytest.mark.parametrize("period", [7, 12])
def test_moving_average_fills_the_trailing_edge(period):
    """Regression: the tail fill must read the last computed value.

    Index ``-half`` is the first element of the slice being assigned, so
    reading it propagated NaN across the entire tail.
    """
    # Same as above: the window varies, the series does not.
    series = seasonal_series()  # preen: allow-dropped-arg
    trend = moving_average_decompose(series, period=period)["trend_component"]
    assert trend.notna().all()
    assert np.isfinite(trend.iloc[-1])
    assert np.isfinite(trend.iloc[0])


def test_explicit_stl_refuses_an_unusable_period():
    """An explicit method request must not silently run another estimator."""
    with pytest.raises(ValueError, match="unusable"):
        stl_decompose(seasonal_series(), period=999)


def test_auto_decomposition_may_fall_back_for_an_unusable_period():
    """Fallback remains valid when the caller delegated method selection."""
    with pytest.warns(UserWarning, match="unusable"):
        result = deseasonalize(seasonal_series(), method="auto", period=999)
    assert result["decomposition_method"].iloc[0] == "none"


def test_trend_with_deseasonalization_survives_an_unusable_period():
    """Regression: downstream indexing of 'deseasonalized' used to raise."""
    with pytest.warns(UserWarning, match="unusable"):
        result = trend_with_deseasonalization(
            seasonal_series(), method="auto", period=999
        )
    assert "derivative_value" in result.columns
    assert result["derivative_value"].notna().sum() > N // 2


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(seasonal_series(), id="cyclic"),
        pytest.param(plain_series(), id="flat"),
    ],
)
def test_trend_with_deseasonalization_has_one_schema(data):
    """Both paths emit derivative_value, never trend_derivative_value."""
    result = trend_with_deseasonalization(data, SavitzkyGolay(window_length=21))
    assert "derivative_value" in result.columns
    assert "trend_derivative_value" not in result.columns
    for column in DECOMPOSITION_COLUMNS:
        assert column in result.columns


def test_trend_with_deseasonalization_forwards_uncertainty_options():
    """The wrapper must pass with_uncertainty= through rather than swallowing it."""
    result = trend_with_deseasonalization(
        seasonal_series(),
        SavitzkyGolay(window_length=21),
        with_uncertainty=True,
        n_bootstrap=30,
        random_state=1,
    )
    assert result["derivative_standard_error"].notna().any()


def test_uncertainty_accounts_for_the_seasonal_fit_by_default():
    """Asking for a standard error must not hand back a knowingly narrow one.

    The seasonal component was estimated from the same data, so an interval
    conditional on it covers 0.917 against a nominal 0.95. The wrapper
    bootstraps the whole pipeline instead, which is why uncertainty_method says so.
    """
    result = trend_with_deseasonalization(
        seasonal_series(),
        SavitzkyGolay(window_length=21),
        with_uncertainty=True,
        n_bootstrap=30,
        random_state=1,
    )
    assert result["uncertainty_method"].iloc[0] == "pipeline_bootstrap"
    assert result["noise_model"].iloc[0].startswith("iid(")
    assert result["confidence_level"].eq(0.95).all()
    assert not result["simultaneous"].any()


def test_no_bootstrap_cost_when_no_standard_error_is_asked_for():
    """The pipeline bootstrap is only paid for when it is wanted."""
    result = trend_with_deseasonalization(
        seasonal_series(), SavitzkyGolay(window_length=21)
    )
    assert result["derivative_standard_error"].isna().all()
    assert result["uncertainty_method"].iloc[0] is None
    assert result["confidence_level"].isna().all()
    assert not result["simultaneous"].any()


def test_it_works_with_any_smoother():
    """Taking a Smoother rather than a name is the point of the rewrite."""
    data = seasonal_series()
    for smoother in (SavitzkyGolay(window_length=15), SmoothingSpline(penalty=1e4)):
        result = trend_with_deseasonalization(data, smoother)
        assert result["derivative_method"].iloc[0] == smoother.name


def test_the_original_values_are_preserved():
    """The observed column must survive; only the estimate is of the adjusted."""
    data = seasonal_series()
    result = trend_with_deseasonalization(data, SavitzkyGolay(window_length=21))
    np.testing.assert_allclose(result["value"], data["value"])


def test_pipeline_bootstrap_honors_confidence_level():
    """Regression: the pipeline percentiles were hard-coded to 2.5/97.5.

    A requested 50% interval came back as a 95% one, which silently changes
    every significance decision downstream.
    """
    data = seasonal_series()
    smoother = SavitzkyGolay(window_length=21)
    narrow = trend_with_deseasonalization(
        data,
        smoother,
        with_uncertainty=True,
        confidence_level=0.50,
        n_bootstrap=40,
        random_state=1,
    )
    wide = trend_with_deseasonalization(
        data,
        smoother,
        with_uncertainty=True,
        confidence_level=0.99,
        n_bootstrap=40,
        random_state=1,
    )
    narrow_width = float(
        (narrow["derivative_ci_upper"] - narrow["derivative_ci_lower"]).mean()
    )
    wide_width = float(
        (wide["derivative_ci_upper"] - wide["derivative_ci_lower"]).mean()
    )
    assert wide_width > 2.0 * narrow_width, (
        f"50% width {narrow_width:.5f} vs 99% width {wide_width:.5f}"
    )
    assert narrow["confidence_level"].eq(0.50).all()
    assert wide["confidence_level"].eq(0.99).all()


def test_pipeline_bootstrap_rejects_unimplemented_simultaneous_bands():
    """Pointwise pipeline percentiles must not be labeled simultaneous."""
    with pytest.raises(ValueError, match="does not support simultaneous"):
        trend_with_deseasonalization(
            seasonal_series(),
            SavitzkyGolay(window_length=21),
            with_uncertainty=True,
            simultaneous=True,
            n_bootstrap=20,
            random_state=1,
        )


@pytest.mark.parametrize("n_bootstrap", [0, 1, 1.5, True])
def test_pipeline_bootstrap_validates_its_own_replicate_count(n_bootstrap):
    """The wrapper's named count must reach the shared estimator contract."""
    with pytest.raises(ValueError, match="n_bootstrap"):
        trend_with_deseasonalization(
            seasonal_series(),
            SavitzkyGolay(window_length=21),
            with_uncertainty=True,
            n_bootstrap=n_bootstrap,
            random_state=1,
        )


def test_pipeline_owns_uncertainty_without_running_an_inner_interval(monkeypatch):
    """No conditional interval should be computed and then overwritten."""

    def reject_inner_uncertainty(*args, **kwargs):
        del args, kwargs
        raise AssertionError(
            "the seasonal point and refits must not attach uncertainty"
        )

    monkeypatch.setattr(Smoother, "_attach_uncertainty", reject_inner_uncertainty)
    result = trend_with_deseasonalization(
        seasonal_series(),
        SavitzkyGolay(window_length=21),
        with_uncertainty=True,
        n_bootstrap=12,
        random_state=1,
    )
    assert result["uncertainty_method"].eq("pipeline_bootstrap").all()


def test_ar1_noise_reaches_fixed_smoother_pipeline_uncertainty(monkeypatch):
    """Dependence must widen the outer bootstrap, not break every refit."""
    original_draws = NoiseFit.gaussian_draws
    draw_sizes = []

    def tracked_draws(self, *args, **kwargs):
        draw_sizes.append(args[1])
        return original_draws(self, *args, **kwargs)

    monkeypatch.setattr(NoiseFit, "gaussian_draws", tracked_draws)
    data = dependent_seasonal_series()
    common = {
        "method": "stl",
        "period": 12,
        "with_uncertainty": True,
        "n_bootstrap": 80,
        "random_state": 9,
    }
    independent = trend_with_deseasonalization(
        data,
        SavitzkyGolay(window_length=21),
        noise=IID(standard_deviation=0.5),
        **common,
    )
    dependent = trend_with_deseasonalization(
        data,
        SavitzkyGolay(window_length=21),
        noise=AR1(phi=0.75, standard_deviation=0.5),
        **common,
    )
    interior = slice(20, -20)
    independent_error = float(
        independent["derivative_standard_error"].iloc[interior].median()
    )
    dependent_error = float(
        dependent["derivative_standard_error"].iloc[interior].median()
    )
    assert dependent["uncertainty_method"].eq("pipeline_bootstrap").all()
    assert dependent["noise_model"].iloc[0].startswith("ar1(phi=0.750")
    assert dependent_error > 1.3 * independent_error
    assert draw_sizes == [80]


def test_heteroskedastic_noise_reaches_pipeline_uncertainty():
    """A stated change in scale must appear in the reported standard errors."""
    data = dependent_seasonal_series()
    midpoint = len(data) // 2
    scale = np.r_[np.full(midpoint, 0.15), np.full(len(data) - midpoint, 0.8)]
    result = trend_with_deseasonalization(
        data,
        SavitzkyGolay(window_length=21),
        method="stl",
        period=12,
        with_uncertainty=True,
        noise=Heteroskedastic(standard_deviation=scale),
        n_bootstrap=80,
        random_state=9,
    )
    quiet_error = float(
        result["derivative_standard_error"].iloc[15 : midpoint - 10].median()
    )
    noisy_error = float(
        result["derivative_standard_error"].iloc[midpoint + 10 : -15].median()
    )
    assert result["noise_model"].eq("heteroskedastic").all()
    assert noisy_error > 3 * quiet_error


def test_given_covariance_is_preserved_by_pipeline_bootstrap():
    """Off-diagonal covariance must affect the whole-pipeline interval."""
    data = dependent_seasonal_series(n=72)
    positions = np.arange(len(data))
    independent_covariance = 0.4**2 * np.eye(len(data))
    dependent_covariance = 0.4**2 * 0.8 ** np.abs(
        np.subtract.outer(positions, positions)
    )
    common = {
        "method": "stl",
        "period": 12,
        "with_uncertainty": True,
        "n_bootstrap": 50,
        "random_state": 5,
    }
    independent = trend_with_deseasonalization(
        data,
        SavitzkyGolay(window_length=15),
        noise=Given(independent_covariance),
        **common,
    )
    dependent = trend_with_deseasonalization(
        data,
        SavitzkyGolay(window_length=15),
        noise=Given(dependent_covariance),
        **common,
    )
    independent_error = float(
        independent["derivative_standard_error"].iloc[12:-12].median()
    )
    dependent_error = float(
        dependent["derivative_standard_error"].iloc[12:-12].median()
    )
    assert dependent["noise_model"].eq("given_covariance").all()
    assert dependent_error > 1.1 * independent_error


def test_adaptive_spline_keeps_noise_for_point_and_pipeline_fits(monkeypatch):
    """Covariance-aware penalty selection must survive the outer bootstrap."""
    original_draws = NoiseFit.gaussian_draws
    draw_sizes = []

    def tracked_draws(self, *args, **kwargs):
        draw_sizes.append(args[1])
        return original_draws(self, *args, **kwargs)

    monkeypatch.setattr(NoiseFit, "gaussian_draws", tracked_draws)
    result = trend_with_deseasonalization(
        dependent_seasonal_series(n=72),
        SmoothingSpline(),
        method="stl",
        period=12,
        with_uncertainty=True,
        noise=AR1(phi=0.6, standard_deviation=0.5),
        n_bootstrap=8,
        random_state=9,
    )
    assert result["uncertainty_method"].eq("pipeline_bootstrap").all()
    assert result["noise_model"].iloc[0].startswith("ar1(phi=0.600")
    assert np.isfinite(result["generalized_penalty"]).all()
    assert draw_sizes == [8]


def test_native_model_still_rejects_external_pipeline_noise():
    """The wrapper must not hide an option the selected model cannot use."""
    with pytest.raises(ValueError, match="models noise internally"):
        trend_with_deseasonalization(
            seasonal_series(),
            GaussianProcess(optimize=False),
            with_uncertainty=True,
            noise="ar1",
            n_bootstrap=2,
            random_state=1,
        )


@pytest.mark.parametrize("route", ["residual", "parametric"])
def test_every_failed_pipeline_refit_raises_instead_of_falling_back(monkeypatch, route):
    """A broken outer bootstrap must never return a conditional interval."""
    original = seasonal_module.deseasonalize
    calls = 0

    def fail_after_initial_decomposition(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original(*args, **kwargs)
        raise ValueError("planted decomposition failure")

    monkeypatch.setattr(
        seasonal_module, "deseasonalize", fail_after_initial_decomposition
    )
    noise = Given(0.4**2 * np.eye(N)) if route == "parametric" else None
    with pytest.raises(
        RuntimeError, match="seasonal pipeline bootstrap failed for every replicate"
    ):
        trend_with_deseasonalization(
            seasonal_series(),
            SavitzkyGolay(window_length=21),
            with_uncertainty=True,
            noise=noise,
            n_bootstrap=3,
            random_state=1,
        )


def test_pure_noise_is_not_uniformly_significant():
    """Regression: a degenerate bootstrap made every point significant.

    When the decomposition leaves residuals with no spread -- which happens on a
    short or nearly deterministic series -- every resample reproduced the same
    curve, the standard error came out at 1e-18, and the pipeline reported 100%
    of points as significantly trending on pure noise.
    """
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {"value": rng.normal(0, 1.0, 96)},
        index=pd.date_range("2020-01-01", periods=96, freq="ME"),
    )
    result = trend_with_deseasonalization(
        frame,
        smoother=SmoothingSpline(),
        with_uncertainty=True,
        n_bootstrap=40,
        random_state=0,
    )
    rate = float(result["significant_trend"].mean())
    assert rate < 0.5, f"{rate:.0%} of a pure-noise series flagged as trending"
    assert float(result["derivative_standard_error"].median()) > 1e-6


def test_a_constant_series_reports_no_trend():
    """The degenerate case itself: zero residual spread must not mean certainty."""
    frame = pd.DataFrame(
        {"value": np.full(60, 4.0)},
        index=pd.date_range("2020-01-01", periods=60, freq="ME"),
    )
    # A flat series has no residual spread to resample, so no standard error
    # can be bootstrapped. That warning is the substance of this test.
    with pytest.warns(UserWarning, match="no estimable noise level"):
        result = trend_with_deseasonalization(
            frame,
            smoother=SmoothingSpline(),
            with_uncertainty=True,
            n_bootstrap=20,
            random_state=0,
        )
    assert not result["significant_trend"].any()
