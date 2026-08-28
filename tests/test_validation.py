"""Public arguments fail at the boundary when their values have no meaning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from incline import (
    SMOOTHERS,
    GaussianProcess,
    L1TrendFilter,
    LocalPolynomial,
    Loess,
    SavitzkyGolay,
    SiZer,
    SiZerMap,
    SmoothingSpline,
    StateSpace,
    TimeAxis,
    build,
    generate_time_series,
    trending,
)
from incline.noise import AR1, IID, Given, Heteroskedastic, NoiseFit, NoiseModel
from incline.simulate import NoiseGenerator, PolynomialTrend

AXIS = TimeAxis.positional(20)
VALUES = np.linspace(0.0, 1.0, AXIS.n)


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, -0.1, 1.1, np.nan, np.inf])
def test_confidence_level_must_be_strictly_between_zero_and_one(confidence_level):
    """Invalid probabilities must not become NaN or infinite intervals."""
    with pytest.raises(ValueError, match="confidence_level"):
        SavitzkyGolay(window_length=7).fit(
            AXIS,
            VALUES,
            confidence_level=confidence_level,
        )


@pytest.mark.parametrize("n_bootstrap", [-1, 0, 1, 1.5, True])
def test_bootstrap_count_must_be_an_integer_large_enough_to_measure_spread(
    n_bootstrap,
):
    """One or fewer draws cannot estimate a sampling standard deviation."""
    with pytest.raises(ValueError, match="n_bootstrap"):
        Loess(robust=True).fit(
            AXIS,
            VALUES,
            with_uncertainty=True,
            n_bootstrap=n_bootstrap,
        )


@pytest.mark.parametrize(
    ("noise_class", "kwargs"),
    [
        (IID, {"standard_deviation": -1.0}),
        (IID, {"standard_deviation": np.nan}),
        (IID, {"standard_deviation": np.inf}),
        (AR1, {"standard_deviation": -1.0}),
        (Heteroskedastic, {"standard_deviation": np.array([0.2, -0.1])}),
        (Heteroskedastic, {"standard_deviation": np.array([0.2, np.nan])}),
    ],
)
def test_noise_scales_must_be_finite_and_nonnegative(noise_class, kwargs):
    """A scale cannot be negative or non-finite."""
    with pytest.raises(ValueError, match="standard_deviation"):
        noise_class(**kwargs)


@pytest.mark.parametrize("phi", [-1.0, 1.0, -1.2, 1.2, np.nan, np.inf])
def test_ar1_coefficient_must_describe_a_stationary_process(phi):
    """An AR(1) covariance exists only strictly inside the unit circle."""
    with pytest.raises(ValueError, match="phi"):
        AR1(phi=phi)


@pytest.mark.parametrize(
    "covariance",
    [
        np.ones((2, 3)),
        np.array([[1.0, np.nan], [np.nan, 1.0]]),
        np.array([[1.0, 0.0], [0.5, 1.0]]),
        np.array([[1.0, 2.0], [2.0, 1.0]]),
    ],
)
def test_given_covariance_must_be_a_finite_symmetric_psd_matrix(covariance):
    """Invalid covariance matrices must not yield clipped zero variances."""
    with pytest.raises(ValueError, match="covariance"):
        Given(covariance)


def test_positive_semidefinite_covariance_is_allowed():
    """Perfect correlation is singular but still a valid covariance."""
    covariance = np.ones((2, 2))
    fitted = Given(covariance).estimate(np.array([0.0, 1.0]), TimeAxis.positional(2))
    np.testing.assert_array_equal(fitted.explicit, covariance)


@pytest.mark.parametrize(
    ("n", "n_draws", "match"),
    [
        (0, 2, "n"),
        (-1, 2, "n"),
        (1.5, 2, "n"),
        (True, 2, "n"),
        (2, 0, "n_draws"),
        (2, -1, "n_draws"),
        (2, 1.5, "n_draws"),
        (2, True, "n_draws"),
    ],
)
def test_gaussian_draw_counts_must_be_positive_integers(n, n_draws, match):
    """Invalid draw dimensions fail before allocating an array."""
    with pytest.raises(ValueError, match=match):
        NoiseFit(standard_deviation=1.0).gaussian_draws(n, n_draws)


def test_gaussian_draw_dimension_must_match_a_supplied_covariance():
    """A fitted covariance cannot be silently truncated or expanded."""
    fit = Given(np.eye(2)).estimate(np.ones(2), TimeAxis.positional(2))
    with pytest.raises(ValueError, match="explicit covariance"):
        fit.gaussian_draws(3, 2)


def test_gaussian_draws_reproduce_the_fitted_ar1_covariance():
    """The parametric bootstrap generator must preserve variance and lag covariance."""
    fit = AR1(phi=-0.4, standard_deviation=0.7).estimate(VALUES, AXIS)
    first = fit.gaussian_draws(AXIS.n, 50_000, random_state=723)
    second = fit.gaussian_draws(AXIS.n, 50_000, random_state=723)

    np.testing.assert_array_equal(first, second)
    assert float(np.var(first)) == pytest.approx(0.7**2, rel=0.02)
    assert float(np.mean(first[:, 1:] * first[:, :-1])) == pytest.approx(
        -0.4 * 0.7**2,
        rel=0.03,
    )


@pytest.mark.parametrize(
    ("noise", "label"),
    [
        (IID(standard_deviation=0.2), "iid("),
        (AR1(phi=0.0, standard_deviation=0.2), "ar1(phi=0.000"),
        (
            Heteroskedastic(standard_deviation=np.full(AXIS.n, 0.2)),
            "heteroskedastic",
        ),
        (Given(np.eye(AXIS.n)), "given_covariance"),
    ],
)
@pytest.mark.parametrize("bias_correct", [False, True])
def test_noise_provenance_names_the_actual_covariance_structure(
    noise, label, bias_correct
):
    """Distinct covariance models must not all be reported as iid."""
    estimate = SavitzkyGolay(window_length=7).fit(
        AXIS,
        VALUES,
        with_uncertainty=True,
        noise=noise,
        bias_correct=bias_correct,
    )
    assert estimate.provenance.noise.startswith(label)


def test_irrelevant_noise_cannot_block_a_native_posterior():
    """Reject the incompatible option before trying to fit that noise model."""

    class ExplodingNoise(NoiseModel):
        def estimate(self, y, axis):
            raise AssertionError("an irrelevant noise model was fitted")

    smoother = GaussianProcess(optimize=False, n_restarts=0)
    with pytest.raises(ValueError, match="noise"):
        smoother.fit(
            AXIS, np.sin(AXIS.x / 3), with_uncertainty=True, noise=ExplodingNoise()
        )


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: SavitzkyGolay(window_length=0), "window_length"),
        (lambda: SavitzkyGolay(window_length=4), "window_length"),
        (lambda: SavitzkyGolay(degree=-1), "degree"),
        (lambda: SavitzkyGolay(window_length=3, degree=3), "degree"),
        (lambda: LocalPolynomial(bandwidth=0), "bandwidth"),
        (lambda: LocalPolynomial(bandwidth=1.1), "bandwidth"),
        (lambda: LocalPolynomial(degree=1.5), "degree"),
        (lambda: LocalPolynomial(kernel="triangle"), "Unknown kernel"),
        (lambda: Loess(span=0), "span"),
        (lambda: Loess(span=1.1), "span"),
        (lambda: SmoothingSpline(penalty=-1), "penalty"),
        (lambda: L1TrendFilter(), "exactly one"),
        (
            lambda: L1TrendFilter(penalty=1, penalty_fraction=0.5),
            "exactly one",
        ),
        (lambda: L1TrendFilter(penalty=-1), "penalty"),
        (lambda: L1TrendFilter(penalty_fraction=1.1), "penalty_fraction"),
        (lambda: L1TrendFilter(penalty=1, difference_order=0), "difference_order"),
        (lambda: L1TrendFilter(penalty=1, max_iter=0), "max_iter"),
        (lambda: L1TrendFilter(penalty=1, tolerance=0), "tolerance"),
    ],
)
def test_smoother_constructor_domains_are_enforced(build, match):
    """Invalid smoother settings fail when the object is constructed."""
    with pytest.raises(ValueError, match=match):
        build()


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: GaussianProcess(kernel="triangle"), "Unknown kernel"),
        (lambda: GaussianProcess(amplitude=0), "amplitude"),
        (lambda: GaussianProcess(length_scale=0), "length_scale"),
        (lambda: GaussianProcess(noise_level=-1), "noise_level"),
        (lambda: GaussianProcess(n_restarts=-1), "n_restarts"),
        (lambda: GaussianProcess(optimize=1), "optimize"),
        (lambda: StateSpace(seasonal_period=1), "seasonal_period"),
    ],
)
def test_process_constructor_domains_are_enforced(build, match):
    """Probability-model settings fail before reaching upstream libraries."""
    with pytest.raises(ValueError, match=match):
        build()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("scales", np.array([]), "scales"),
        ("scales", np.array([0.2, 0.1]), "scales"),
        ("scales", np.array([0.1, 1.1]), "scales"),
        ("derivative", np.zeros((2, 19)), "derivative"),
        ("standard_error", np.full((2, 20), -1.0), "standard_error"),
        ("significance", np.full((2, 20), 2), "significance"),
        ("confidence_level", 1.0, "confidence_level"),
        ("simultaneous", 1, "simultaneous"),
        ("smoother_name", "", "smoother_name"),
    ],
)
def test_sizer_map_rejects_inconsistent_results(field, value, match):
    """A malformed public result object must fail before plotting or reshaping."""
    options = {
        "axis": AXIS,
        "scales": np.array([0.1, 0.2]),
        "derivative": np.zeros((2, AXIS.n)),
        "standard_error": np.ones((2, AXIS.n)),
        "significance": np.zeros((2, AXIS.n), dtype=int),
        "confidence_level": 0.95,
        "simultaneous": False,
        "smoother_name": "local_poly",
    }
    options[field] = value
    with pytest.raises(ValueError, match=match):
        SiZerMap(**options)


@pytest.mark.parametrize("name", sorted(SMOOTHERS))
@pytest.mark.parametrize("scale", [0, -0.1, 1.1, np.nan, np.inf, True])
def test_every_smoother_enforces_the_shared_scale_domain(name, scale):
    """Scale setters reject invalid fractions instead of clipping or ignoring them."""
    smoother = build(name, penalty_fraction=0.2) if name == "l1_filter" else build(name)
    with pytest.raises(ValueError, match="scale"):
        smoother.with_scale(scale, AXIS)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.ones((2, 10)), "one-dimensional"),
        (np.array([]), "at least one"),
    ],
)
def test_fit_requires_a_nonempty_one_dimensional_series(values, match):
    """Array-shape failures should not surface later as misleading math errors."""
    with pytest.raises(ValueError, match=match):
        SavitzkyGolay(window_length=7).fit(
            TimeAxis.positional(max(1, len(values))), values
        )


def test_derivative_order_rejects_boolean_values():
    """True is an integer in Python but not a meaningful derivative request."""
    with pytest.raises(ValueError, match="derivative_order"):
        SavitzkyGolay(window_length=7).fit(AXIS, VALUES, derivative_order=True)


@pytest.mark.parametrize(
    "config",
    [
        {"scales": np.array([])},
        {"scales": np.array([0.1, -0.2])},
        {"scales": np.array([0.2, 0.1])},
        {"n_scales": 0},
        {"scale_range": (0.5, 0.1)},
        {"min_persistence": 0},
    ],
)
def test_sizer_domains_are_enforced(config):
    """Invalid scale grids and persistence thresholds never make empty maps."""
    if "min_persistence" in config:
        result = SiZer(n_scales=3, simultaneous=False).fit(
            pd.DataFrame({"value": np.arange(20.0)})
        )
        with pytest.raises(ValueError, match="min_persistence"):
            result.significant_regions(**config)
    else:
        with pytest.raises(ValueError, match="must"):
            SiZer(simultaneous=False, **config)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_length": 0},
        {"confidence_level": 1.0},
        {"aggregation": "average"},
        {"weighting": "linear", "aggregation": "last"},
        {"ids": ["ignored"]},
    ],
)
def test_ranking_domains_and_irrelevant_arguments_are_enforced(kwargs):
    """Ranking options must be meaningful for the selected input and summary."""
    estimate = SavitzkyGolay(window_length=7).fit(AXIS, VALUES)
    with pytest.raises((TypeError, ValueError)):
        trending({"series": estimate}, **kwargs)


@pytest.mark.parametrize(
    "call",
    [
        lambda: NoiseGenerator.white(-1),
        lambda: NoiseGenerator.white(10, standard_deviation=-1),
        lambda: NoiseGenerator.seasonal(10, period=0),
        lambda: NoiseGenerator.seasonal(10, seasonal_fraction=1.1),
        lambda: generate_time_series(PolynomialTrend([0.0]), n_points=0),
        lambda: generate_time_series(PolynomialTrend([0.0]), x_range=(1.0, 0.0)),
        lambda: generate_time_series(PolynomialTrend([0.0]), irregular_spacing=1),
    ],
)
def test_simulation_domains_are_enforced(call):
    """Synthetic truth generators reject impossible configurations."""
    with pytest.raises(ValueError, match="must"):
        call()
