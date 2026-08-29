"""Tests for the smoothers and the shared fit machinery.

Most behavior lives on the base class, so most of these are parametrized over
the whole registry: whatever gets added later has to satisfy them too.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from incline.axis import TimeAxis
from incline.noise import AR1, IID
from incline.result import TrendEstimate
from incline.smoothers import (
    SMOOTHERS,
    L1TrendFilter,
    LocalPolynomial,
    Loess,
    NaiveDifference,
    SavitzkyGolay,
    SmoothingSpline,
    build,
)

N = 100
AXIS = TimeAxis.positional(N)


def noisy(seed: int = 0, slope: float = 0.05) -> np.ndarray:
    """A gently curved series with noise."""
    rng = np.random.default_rng(seed)
    return slope * AXIS.x + np.sin(AXIS.x / 12) + rng.normal(0, 0.25, N)


ALL_NAMES = sorted(SMOOTHERS)
SCALABLE_NAMES = [name for name in ALL_NAMES if name not in {"kalman", "naive"}]
LINEAR = [
    pytest.param(SavitzkyGolay(window_length=15), id="sgolay"),
    pytest.param(NaiveDifference(), id="naive"),
    pytest.param(LocalPolynomial(bandwidth=0.2, degree=2), id="local_poly"),
    pytest.param(SmoothingSpline(penalty=1e4), id="smoothing_spline_fixed"),
    pytest.param(Loess(span=0.3, robust=False), id="loess_plain"),
]
NONLINEAR = [
    pytest.param(SmoothingSpline(penalty=None), id="smoothing_spline_gcv"),
    pytest.param(Loess(span=0.3, robust=True), id="loess_robust"),
    pytest.param(L1TrendFilter(penalty_fraction=0.2), id="l1"),
]


def build_for_test(name: str):
    """Build registry entries with an explicit L1 smoothing choice."""
    if name == "l1_filter":
        return build(name, penalty_fraction=0.2)
    return build(name)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_registry_names_match_their_classes(name):
    """build() must return the smoother the registry is keyed by."""
    smoother = build_for_test(name)
    assert smoother.name == name
    assert isinstance(smoother, SMOOTHERS[name])


def test_unknown_smoother_name_lists_the_alternatives():
    """A typo should tell you what was available."""
    with pytest.raises(ValueError, match="Unknown method"):
        build("savitsky_golay")


@pytest.mark.parametrize("smoother", LINEAR)
def test_linear_smoothers_declare_themselves_linear(smoother):
    """The declaration drives which uncertainty route runs."""
    assert smoother.is_linear


@pytest.mark.parametrize("smoother", NONLINEAR)
def test_nonlinear_smoothers_declare_themselves_nonlinear(smoother):
    """Exact standard errors must never be offered to these."""
    assert not smoother.is_linear


@pytest.mark.parametrize("smoother", NONLINEAR)
def test_bias_correction_is_refused_for_nonlinear_smoothers(smoother):
    """The correction is operator composition, so it needs an operator."""
    with pytest.raises(ValueError, match="not a linear smoother"):
        smoother.fit(AXIS, noisy(), derivative_order=1, bias_correct=True)


@pytest.mark.parametrize("smoother", NONLINEAR)
def test_bootstrap_smoothers_refuse_simultaneous_bands(smoother):
    """A pointwise bootstrap interval must not masquerade as a whole-curve band."""
    with pytest.raises(ValueError, match=r"bootstrap.*simultaneous"):
        smoother.fit(
            AXIS,
            noisy(),
            derivative_order=1,
            with_uncertainty=True,
            simultaneous=True,
            n_bootstrap=20,
        )


def test_simultaneous_requires_uncertainty():
    """A whole-curve request cannot be silently ignored when intervals are off."""
    with pytest.raises(ValueError, match="requires with_uncertainty=True"):
        SavitzkyGolay().fit(AXIS, noisy(), simultaneous=True)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_estimate_recovers_a_straight_line(name):
    """Every smoother must get a constant slope right on noiseless data."""
    y = 2.0 * AXIS.x + 1.0
    estimate = build_for_test(name).fit(AXIS, y, derivative_order=1)
    interior = estimate.derivative[10:-10]
    assert np.allclose(interior, 2.0, atol=0.05), f"{name} got {interior[:3]}"


@pytest.mark.parametrize("name", ALL_NAMES)
def test_fit_without_se_leaves_uncertainty_honestly_empty(name):
    """Absent uncertainty is NaN plus a None label, never a missing column."""
    estimate = build_for_test(name).fit(AXIS, noisy(), derivative_order=1)
    assert isinstance(estimate, TrendEstimate)
    assert estimate.standard_error is None
    assert estimate.has_uncertainty is False
    assert not estimate.significant.any()
    assert estimate.provenance.uncertainty_method is None


@pytest.mark.parametrize("name", ALL_NAMES)
def test_fit_with_se_labels_its_route(name):
    """Every smoother reports how it got its standard error."""
    smoother = build_for_test(name)
    estimate = smoother.fit(
        AXIS, noisy(), derivative_order=1, with_uncertainty=True, n_bootstrap=40
    )
    assert estimate.standard_error is not None
    assert np.all(estimate.standard_error[np.isfinite(estimate.standard_error)] >= 0)
    if smoother.has_native_posterior:
        expected = "native"
    elif smoother.is_linear:
        expected = "operator"
    else:
        expected = "bootstrap"
    assert estimate.provenance.uncertainty_method == expected


@pytest.mark.parametrize("smoother", LINEAR)
def test_operators_are_cached_and_reused(smoother):
    """The operator depends on the axis, not the data, so it should be reused."""
    first_smooth, first_deriv = smoother.operators(AXIS, 1)
    second_smooth, second_deriv = smoother.operators(AXIS, 1)
    assert first_deriv is second_deriv
    assert first_smooth is second_smooth


@pytest.mark.parametrize("smoother", LINEAR)
def test_probed_operator_reproduces_the_estimator(smoother):
    """The probe is only meaningful if L @ y really is the estimate."""
    y = noisy(3)
    _, operator = smoother.operators(AXIS, 1)
    direct = smoother.evaluate(AXIS, y, 1).derivative
    finite = np.isfinite(direct)
    np.testing.assert_allclose((operator @ y)[finite], direct[finite], atol=1e-8)


@pytest.mark.parametrize("penalty", [None, 12.5])
def test_smoothing_spline_matches_scipys_reference(penalty):
    """The wrapper must reproduce SciPy's values and analytic derivative."""
    from scipy.interpolate import make_smoothing_spline

    y = noisy(37)
    reference = make_smoothing_spline(AXIS.x, y, lam=penalty)
    actual = SmoothingSpline(penalty=penalty).evaluate(AXIS, y, 1)

    np.testing.assert_allclose(actual.values, reference(AXIS.x), rtol=0, atol=0)
    np.testing.assert_allclose(
        actual.derivative,
        reference(AXIS.x, nu=1),
        rtol=0,
        atol=0,
    )


def test_smoothing_spline_tunes_for_correlated_errors():
    """AR(1)-aware tuning must beat iid GCV on a known smooth signal."""
    from incline.simulate import NoiseGenerator

    x = AXIS.x
    truth = 2 * np.sin(x / 15) + 0.03 * x
    derivative = 2 * np.cos(x / 15) / 15 + 0.03
    observed = truth + NoiseGenerator.ar1(
        N,
        phi=0.7,
        standard_deviation=0.4,
        random_state=0,
    )

    ordinary = SmoothingSpline().fit(AXIS, observed, derivative_order=1)
    correlated = SmoothingSpline().fit(
        AXIS,
        observed,
        derivative_order=1,
        noise=AR1(phi=0.7, standard_deviation=0.4),
    )

    interior = slice(10, -10)
    ordinary_rmse = np.sqrt(
        np.mean((ordinary.derivative[interior] - derivative[interior]) ** 2)
    )
    correlated_rmse = np.sqrt(
        np.mean((correlated.derivative[interior] - derivative[interior]) ** 2)
    )
    assert correlated_rmse < 0.4 * ordinary_rmse
    assert correlated.provenance.noise is not None
    assert correlated.provenance.params["selected_penalty"] > 0


def test_correlated_smoothing_spline_preserves_its_linear_null_space():
    """Covariance whitening must not penalize a line."""
    values = 0.7 - 0.4 * AXIS.x

    estimate = SmoothingSpline().fit(
        AXIS,
        values,
        derivative_order=1,
        noise=AR1(phi=0.7, standard_deviation=0.4),
    )

    np.testing.assert_allclose(estimate.values, values, rtol=0, atol=2e-8)
    np.testing.assert_allclose(estimate.derivative, -0.4, rtol=0, atol=2e-8)


def test_correlated_smoothing_spline_matches_penalized_gls():
    """The selected fit must solve the stated penalized GLS problem."""
    from incline.smoothers import _natural_spline_penalty

    values = noisy(19)
    noise_fit = AR1(phi=0.6, standard_deviation=0.3).estimate(values, AXIS)
    covariance = noise_fit.covariance(N)
    estimate = SmoothingSpline().fit(
        AXIS,
        values,
        derivative_order=1,
        noise=AR1(phi=0.6, standard_deviation=0.3),
    )
    penalty = estimate.provenance.params["selected_penalty"]
    precision = np.linalg.inv(covariance)
    expected = np.linalg.solve(
        precision + penalty * _natural_spline_penalty(AXIS),
        precision @ values,
    )

    np.testing.assert_allclose(estimate.values, expected, rtol=2e-9, atol=2e-9)


SMOOTHING_LINEAR = [p for p in LINEAR if p.id != "naive"]


@pytest.mark.parametrize("smoother", SMOOTHING_LINEAR)
def test_correlated_noise_widens_a_smoother_interval(smoother):
    """For an averaging estimator, dependence inflates the variance.

    A smoother averages neighbors, and positively correlated neighbors carry
    less independent information than the count suggests, so the interval must
    grow. Differencing estimators behave the opposite way; see below.
    """
    y = noisy(5)
    independent = smoother.fit(
        AXIS, y, derivative_order=1, with_uncertainty=True, noise=IID()
    )
    correlated = smoother.fit(
        AXIS, y, derivative_order=1, with_uncertainty=True, noise=AR1(phi=0.7)
    )
    assert np.nanmean(correlated.standard_error) > np.nanmean(
        independent.standard_error
    )


def test_correlated_noise_narrows_a_differencing_interval():
    """For a difference, positive autocorrelation cancels rather than accumulates.

    The central difference is ``(y[i+1] - y[i-1]) / 2h``, whose variance is
    ``sigma**2 (1 - rho_2) / 2h**2``. Correlated neighbors partly cancel in the
    subtraction, so the correct AR(1) interval is *narrower* than the
    independent one -- the reverse of every smoothing method here. Pinning it
    guards against someone "fixing" the sign by assuming dependence always
    widens.

    Sigma is stated rather than estimated on both sides. Estimating it under
    each model separately compares two different processes: imposing phi=0.7 on
    genuinely independent data implies a marginal variance about four times
    larger, which is a fact about noise estimation and not about differencing.
    """
    y = noisy(5)
    naive = NaiveDifference()
    phi = 0.7
    independent = naive.fit(
        AXIS,
        y,
        derivative_order=1,
        with_uncertainty=True,
        noise=IID(standard_deviation=0.25),
    )
    correlated = naive.fit(
        AXIS,
        y,
        derivative_order=1,
        with_uncertainty=True,
        noise=AR1(phi=phi, standard_deviation=0.25),
    )
    ratio = float(
        np.nanmean(correlated.standard_error) / np.nanmean(independent.standard_error)
    )
    assert ratio == pytest.approx(math.sqrt(1 - phi**2), rel=0.02), (
        f"AR(1)/iid width ratio {ratio:.3f}, expected {math.sqrt(1 - phi**2):.3f}"
    )


@pytest.mark.parametrize("phi", [0.0, 0.5, 0.9])
def test_a_stated_autocorrelation_rescales_the_noise_level(phi):
    """Regression: sigma was left at a value calibrated for a different phi.

    Sigma is recovered by dividing the second difference's observed variance by
    its theoretical value at a given phi, and that divisor falls from 6 to 0.9
    across the grid. Reusing one sigma across phi therefore misstates the noise
    level by up to an order of magnitude -- and ``AR1(phi=0.9)`` returned the
    same 1.417 as ``AR1(phi=0.0)``.
    """
    from incline.noise import estimate_ar1

    y = noisy(5)
    fitted_phi, sigma = estimate_ar1(y, phi)
    assert fitted_phi == phi
    # A more strongly correlated process has to have a larger marginal spread
    # to leave the same second-difference variance behind.
    _, baseline = estimate_ar1(y, 0.0)
    assert sigma >= baseline
    if phi > 0:
        assert sigma > baseline * 1.2


def test_second_derivative_finds_curvature():
    """Order 2 must actually track the second derivative, not repeat the first."""
    y = 0.5 * AXIS.x**2
    for smoother in (
        SavitzkyGolay(window_length=21),
        LocalPolynomial(bandwidth=0.3, degree=2),
    ):
        estimate = smoother.fit(AXIS, y, derivative_order=2)
        interior = estimate.derivative[15:-15]
        assert np.allclose(interior, 1.0, atol=0.05), smoother.name


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (
            np.arange(8, dtype=float),
            np.array([-0.05, 0.26, 0.57, 0.88, 1.19, 1.5, 1.75, 2.0]),
        ),
        (
            np.array([0.0, 0.4, 1.1, 2.0, 3.7, 4.0, 6.2, 9.0]),
            np.array(
                [
                    0.071651778766,
                    0.225655949454,
                    0.495163248156,
                    0.841672632202,
                    1.339042609481,
                    1.426813781941,
                    1.875,
                    1.825,
                ]
            ),
        ),
    ],
    ids=["regular", "irregular"],
)
def test_l1_filter_matches_cvxpy_reference(x, expected):
    """The primal fit must match an independent convex-program solution."""
    y = np.array([0.2, -0.1, 0.4, 1.2, 0.9, 1.8, 2.0, 1.7])
    axis = TimeAxis(x=x, delta=float(np.median(np.diff(x))), unit="index")
    estimate = L1TrendFilter(penalty=0.35, tolerance=1e-11, max_iter=1000).fit(axis, y)
    np.testing.assert_allclose(estimate.values, expected, rtol=0, atol=2e-9)


def test_zero_l1_penalty_returns_the_observations_exactly():
    """At lambda zero the objective is minimized by the unaltered data."""
    y = noisy(31)
    estimate = L1TrendFilter(penalty=0).fit(AXIS, y)
    np.testing.assert_array_equal(estimate.values, y)


def test_unit_l1_penalty_fraction_reaches_the_polynomial_null_space():
    """The saturation endpoint is analytic, not optimizer-tolerance dependent."""
    x = np.array([0.0, 0.4, 1.1, 2.0, 3.7, 4.0, 6.2, 9.0])
    axis = TimeAxis(x=x, delta=float(np.median(np.diff(x))), unit="index")
    y = np.array([0.2, -0.1, 0.4, 1.2, 0.9, 1.8, 2.0, 1.7])
    estimate = L1TrendFilter(penalty_fraction=1.0).fit(axis, y)
    linear_projection = np.polynomial.polynomial.polyval(
        x, np.polynomial.polynomial.polyfit(x, y, deg=1)
    )
    np.testing.assert_allclose(estimate.values, linear_projection, atol=2e-8)
    assert estimate.provenance.params["resolved_penalty"] == pytest.approx(
        estimate.provenance.params["maximum_penalty"]
    )
    assert estimate.provenance.params["optimizer_iterations"] == 0

    above_endpoint = L1TrendFilter(
        penalty=2 * estimate.provenance.params["maximum_penalty"]
    ).fit(axis, y)
    np.testing.assert_allclose(above_endpoint.values, linear_projection, atol=2e-8)
    assert above_endpoint.provenance.params["optimizer_iterations"] == 0


def test_absolute_l1_penalty_has_no_fake_normalized_scale():
    """A data-dependent normalized value cannot be inferred from the axis."""
    assert L1TrendFilter(penalty=1.0).scale_of(AXIS) is None


def test_l1_filter_refuses_an_unidentifiable_difference_order():
    """The penalized order must leave at least one finite difference."""
    with pytest.raises(ValueError, match="requires more than"):
        L1TrendFilter(penalty=1.0, difference_order=3).fit(
            TimeAxis.positional(3), np.arange(3, dtype=np.float64)
        )


def test_l1_filter_reports_optimizer_nonconvergence():
    """Hitting the iteration cap must not silently return an unfinished fit."""
    y = np.sin(AXIS.x / 3) + 0.2 * np.cos(1.7 * AXIS.x)
    with pytest.raises(RuntimeError, match=r"did not converge.*max_iter=1"):
        L1TrendFilter(penalty_fraction=0.5, max_iter=1).fit(AXIS, y)


def test_l1_filter_requires_an_explicit_penalty_choice():
    """The registry cannot smuggle an arbitrary absolute penalty into a fit."""
    with pytest.raises(ValueError, match="exactly one"):
        build("l1_filter")


@pytest.mark.parametrize("name", SCALABLE_NAMES)
def test_scale_round_trips_through_the_normalized_knob(name):
    """with_scale and scale_of must be consistent enough to sweep with."""
    smoother = build_for_test(name)
    rescaled = smoother.with_scale(0.25, AXIS)
    assert isinstance(rescaled, type(smoother))
    recovered = rescaled.scale_of(AXIS)
    assert 0 < recovered <= 1.0


@pytest.mark.parametrize("name", SCALABLE_NAMES)
def test_a_wider_scale_smooths_more(name):
    """The scale knob must be monotone or a bandwidth sweep is meaningless."""
    smoother = build_for_test(name)
    y = noisy(11)
    narrow = smoother.with_scale(0.05, AXIS).fit(AXIS, y, derivative_order=1)
    wide = smoother.with_scale(0.4, AXIS).fit(AXIS, y, derivative_order=1)
    assert np.nanstd(wide.derivative) < np.nanstd(narrow.derivative), smoother.name


@pytest.mark.parametrize("name", ["naive", "kalman"])
def test_methods_without_a_scale_refuse_a_scale_sweep(name):
    """SiZer must not relabel one unchanged fit as several smoothing scales."""
    smoother = build_for_test(name)
    assert smoother.scale_of(AXIS) is None
    with pytest.raises(ValueError, match="no smoothing scale"):
        smoother.with_scale(0.25, AXIS)


@pytest.mark.parametrize("name", ["gp", "smoothing_spline"])
def test_adaptive_smoothers_do_not_report_a_placeholder_scale(name):
    """A scale selected from the data is unknown before fitting, not 0.2."""
    smoother = build_for_test(name)
    assert smoother.scale_of(AXIS) is None
    assert smoother.with_scale(0.25, AXIS).scale_of(AXIS) == pytest.approx(0.25)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_length_mismatch_is_refused(name):
    """A y of the wrong length is a caller error worth naming."""
    with pytest.raises(ValueError, match="axis has"):
        build_for_test(name).fit(AXIS, np.zeros(N - 1), derivative_order=1)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_constant_series_has_zero_slope_and_claims_nothing(name):
    """A flat line trends nowhere, and must not manufacture a trend from dust.

    A constant series has no estimable noise level, so the honest output is a
    slope of zero and no significance claim. Getting this wrong is subtle: the
    point estimate and the standard error both land around 1e-16, and an
    interval of [1e-16, 3e-16] does technically exclude zero.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimate = build_for_test(name).fit(
            AXIS,
            np.full(N, 3.0),
            derivative_order=1,
            with_uncertainty=True,
            n_bootstrap=20,
        )
    interior = estimate.derivative[10:-10]
    assert np.allclose(interior[np.isfinite(interior)], 0.0, atol=1e-6)
    assert not estimate.significant.any()


@pytest.mark.parametrize("smoother", SMOOTHING_LINEAR)
def test_bias_correction_widens_and_recentres(smoother):
    """Correcting bias trades interval width for centering; both should show."""
    y = noisy(13)
    plain = smoother.fit(AXIS, y, derivative_order=1, with_uncertainty=True)
    corrected = smoother.fit(
        AXIS, y, derivative_order=1, with_uncertainty=True, bias_correct=True
    )
    assert corrected.provenance.bias_corrected
    assert np.nanmean(corrected.standard_error) >= np.nanmean(plain.standard_error)


def test_bias_correction_is_refused_without_a_smoothing_scale():
    """Finite differencing has no smoothing bias for a pilot fit to estimate."""
    with pytest.raises(ValueError, match=r"bias correction.*no smoothing scale"):
        NaiveDifference().fit(AXIS, noisy(), derivative_order=1, bias_correct=True)


def test_pilot_scale_requires_bias_correction():
    """A pilot setting must not be accepted when no pilot fit will run."""
    with pytest.raises(ValueError, match="requires bias_correct=True"):
        SavitzkyGolay().fit(AXIS, noisy(), pilot_scale=0.1)


def test_noise_model_reaches_the_bootstrap(recwarn):
    """Regression: `noise=` was ignored by every bootstrapped smoother.

    The fitted noise model was computed and then discarded, and the bootstrap
    re-derived its own scale from the data, so an explicit sigma had no effect
    on the reported uncertainty at all.
    """
    del recwarn
    y = noisy(21)
    smoother = SmoothingSpline()
    small = smoother.fit(
        AXIS,
        y,
        derivative_order=1,
        with_uncertainty=True,
        noise=IID(standard_deviation=0.3),
        n_bootstrap=40,
        random_state=1,
    )
    large = smoother.fit(
        AXIS,
        y,
        derivative_order=1,
        with_uncertainty=True,
        noise=IID(standard_deviation=3.0),
        n_bootstrap=40,
        random_state=1,
    )
    ratio = float(np.nanmean(large.standard_error) / np.nanmean(small.standard_error))
    assert 5.0 < ratio < 20.0, (
        f"a tenfold noise level changed the standard error by {ratio:.2f}x"
    )


def test_adaptive_spline_heteroskedastic_intervals_are_calibrated():
    """Intervals must calibrate at both ends of a changing noise scale."""
    from incline.noise import Heteroskedastic

    ramp = np.linspace(0.1, 2.0, N)
    truth = 0.7 - 0.4 * AXIS.x
    noise = Heteroskedastic(standard_deviation=ramp)
    rng = np.random.default_rng(4817)
    points = (20, 80)
    estimates = np.empty((60, len(points)))
    errors = np.empty_like(estimates)
    covered = np.empty_like(estimates, dtype=bool)

    for replicate in range(60):
        observed = truth + rng.normal(size=N) * ramp
        fitted = SmoothingSpline().fit(
            AXIS,
            observed,
            derivative_order=1,
            with_uncertainty=True,
            noise=noise,
            n_bootstrap=40,
            random_state=int(rng.integers(1 << 30)),
        )
        estimates[replicate] = fitted.derivative[list(points)]
        errors[replicate] = fitted.standard_error[list(points)]
        covered[replicate] = (fitted.ci_lower[list(points)] <= -0.4) & (
            fitted.ci_upper[list(points)] >= -0.4
        )

    bias = estimates.mean(axis=0) + 0.4
    ratios = errors.mean(axis=0) / estimates.std(axis=0, ddof=1)
    coverage = covered.mean(axis=0)
    assert np.all(np.abs(bias) < 0.01)
    assert np.all((ratios > 0.75) & (ratios < 1.3))
    assert np.all((coverage > 0.85) & (coverage <= 1.0))


def test_grid_methods_warn_on_an_irregular_axis():
    """Regression: TimeAxis.require_regular existed but was never called.

    A Savitzky-Golay filter applied to an unevenly spaced series scales every
    derivative by one median step, so callers got silently wrong per-time
    slopes with no warning at all.
    """
    from incline.axis import TimeAxis as Axis

    uneven = Axis._build(np.sort(np.random.default_rng(1).uniform(0, 100, 60)), "index")
    y = 2.0 * uneven.x
    with pytest.raises(ValueError, match="uniform sampling"):
        SavitzkyGolay(window_length=11).fit(uneven, y, derivative_order=1)


def test_irregular_grid_is_refused_for_naive_differences():
    """A median-step stencil is also wrong when neighboring steps differ."""
    from incline.axis import TimeAxis as Axis

    uneven = Axis._build(np.array([0.0, 1.0, 4.0, 5.0, 9.0]), "index")
    y = 2.0 * uneven.x
    with pytest.raises(ValueError, match="uniform sampling"):
        NaiveDifference().fit(uneven, y)


def test_local_polynomial_rejects_a_negative_bandwidth():
    """A negative bandwidth must not behave identically to its absolute value."""
    with pytest.raises(ValueError, match="bandwidth"):
        LocalPolynomial(bandwidth=-0.2)


def test_grid_methods_stay_quiet_on_a_regular_axis():
    """The warning must not fire on the ordinary case."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SavitzkyGolay(window_length=11).fit(AXIS, noisy(3), derivative_order=1)


def test_only_grid_methods_declare_the_requirement():
    """A spline or LOESS handles irregular spacing natively and must not warn."""
    assert SavitzkyGolay.requires_regular_grid
    assert NaiveDifference.requires_regular_grid
    assert not LocalPolynomial.requires_regular_grid
    assert not SmoothingSpline.requires_regular_grid


def test_adaptive_spline_ar1_intervals_are_calibrated():
    """AR(1) estimation and adaptive smoothing must give calibrated intervals."""
    from incline.noise import AR1
    from incline.simulate import NoiseGenerator

    rng = np.random.default_rng(91821)
    truth = 0.7 - 0.4 * AXIS.x
    smoother = SmoothingSpline()

    estimates, errors, covered = [], [], []
    for _ in range(60):
        y = truth + NoiseGenerator.ar1(N, 0.7, 0.4, rng)
        fitted = smoother.fit(
            AXIS,
            y,
            derivative_order=1,
            with_uncertainty=True,
            noise=AR1(),
            n_bootstrap=40,
            random_state=int(rng.integers(1 << 30)),
        )
        midpoint = N // 2
        estimates.append(fitted.derivative[midpoint])
        errors.append(fitted.standard_error[midpoint])
        covered.append(fitted.ci_lower[midpoint] <= -0.4 <= fitted.ci_upper[midpoint])

    bias = float(np.mean(estimates) + 0.4)
    ratio = float(np.mean(errors) / np.std(estimates, ddof=1))
    coverage = float(np.mean(covered))
    assert abs(bias) < 0.02
    assert 0.8 < ratio < 1.3
    assert 0.85 < coverage <= 1.0


def test_adaptive_spline_is_equivariant_to_a_stated_noise_scale():
    """Scaling a linear signal's deviations and covariance must scale inference."""
    from incline.noise import AR1
    from incline.simulate import NoiseGenerator

    truth = 0.7 - 0.4 * AXIS.x
    unit_noise = NoiseGenerator.ar1(N, 0.5, 1.0, random_state=31)
    smoother = SmoothingSpline()
    small = smoother.fit(
        AXIS,
        truth + 0.3 * unit_noise,
        derivative_order=1,
        with_uncertainty=True,
        noise=AR1(standard_deviation=0.3, phi=0.5),
        n_bootstrap=40,
        random_state=1,
    )
    large = smoother.fit(
        AXIS,
        truth + 3.0 * unit_noise,
        derivative_order=1,
        with_uncertainty=True,
        noise=AR1(standard_deviation=3.0, phi=0.5),
        n_bootstrap=40,
        random_state=1,
    )
    np.testing.assert_allclose(
        large.derivative + 0.4,
        10.0 * (small.derivative + 0.4),
        rtol=0,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        large.standard_error,
        10.0 * small.standard_error,
        rtol=0,
        atol=5e-9,
    )


def test_a_series_shorter_than_the_window_says_so():
    """Regression: Savitzky-Golay raised from deep inside scipy on short input.

    ``savgol_filter`` reported "degree must be less than window_length",
    naming neither the series length nor the remedy. Below the polynomial order
    there is no window that could work, so the estimator has to say that.
    """
    short = TimeAxis.positional(4)
    with pytest.raises(ValueError, match="at least"):
        SavitzkyGolay(window_length=21).fit(short, np.arange(4.0), derivative_order=2)


def test_a_series_shorter_than_the_explicit_window_is_refused():
    """An explicit window is a contract, not a suggestion to clamp silently."""
    axis = TimeAxis.positional(9)
    with pytest.raises(ValueError, match="window_length=41"):
        SavitzkyGolay(window_length=41).fit(axis, np.arange(9.0), derivative_order=1)


@pytest.mark.parametrize("name", ["gp", "kalman"])
def test_a_native_posterior_says_when_an_argument_does_not_apply(name):
    """Regression: noise= and simultaneous= must not be accepted and ignored.

    Neither smoother exposes a linear operator, so its interval comes from its
    own posterior. A warning followed by a different analysis still returns a
    result under assumptions the caller did not choose.
    """
    smoother = build_for_test(name)
    with pytest.raises(ValueError, match="noise"):
        smoother.fit(
            AXIS, noisy(), derivative_order=1, with_uncertainty=True, noise=AR1()
        )

    with pytest.raises(ValueError, match="simultaneous"):
        smoother.fit(
            AXIS, noisy(), derivative_order=1, with_uncertainty=True, simultaneous=True
        )


def test_a_noise_model_that_does_apply_passes_without_a_warning():
    """The warning has to be about applicability, not about the argument."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SavitzkyGolay(window_length=15).fit(
            AXIS, noisy(), derivative_order=1, with_uncertainty=True, noise=AR1()
        )


def test_the_operator_cache_is_bounded_by_memory_not_by_entries():
    """Regression: the bound counted entries, each holding two n-by-n matrices.

    At n = 2000 an entry is 64 MB, so an allowance of 64 entries reached roughly
    4 GB before evicting anything -- a multi-scale sweep on a long series was
    enough to exhaust memory.
    """
    from incline.smoothers import _OPERATOR_CACHE, OPERATOR_CACHE_BYTES, _cache_bytes

    _OPERATOR_CACHE.clear()
    try:
        axis = TimeAxis.positional(400)
        for window in range(5, 200, 2):
            SavitzkyGolay(window_length=window).operators(axis, 1)
        assert _cache_bytes() <= OPERATOR_CACHE_BYTES

        wanted = SavitzkyGolay(window_length=11).operators(axis, 1)[1].copy()
        for window in range(5, 200, 2):
            SavitzkyGolay(window_length=window).operators(axis, 1)
        np.testing.assert_allclose(
            wanted, SavitzkyGolay(window_length=11).operators(axis, 1)[1]
        )
    finally:
        _OPERATOR_CACHE.clear()


@pytest.mark.parametrize("smoother", LINEAR)
def test_whole_curve_band_is_wider_than_pointwise(smoother):
    """Correcting for testing every point can only widen the band."""
    y = noisy(7)
    pointwise = smoother.fit(AXIS, y, derivative_order=1, with_uncertainty=True)
    whole = smoother.fit(
        AXIS,
        y,
        derivative_order=1,
        with_uncertainty=True,
        simultaneous=True,
        random_state=0,
    )
    width_point = np.nanmean(pointwise.ci_upper - pointwise.ci_lower)
    width_whole = np.nanmean(whole.ci_upper - whole.ci_lower)
    assert width_whole > width_point
    assert whole.provenance.simultaneous


@pytest.mark.parametrize("name", ALL_NAMES)
def test_wider_confidence_gives_a_wider_interval(name):
    """confidence_level must reach the interval, not get defaulted away."""
    y = noisy(9)
    narrow = build_for_test(name).fit(
        AXIS,
        y,
        derivative_order=1,
        with_uncertainty=True,
        confidence_level=0.5,
        n_bootstrap=40,
        random_state=1,
    )
    wide = build_for_test(name).fit(
        AXIS,
        y,
        derivative_order=1,
        with_uncertainty=True,
        confidence_level=0.99,
        n_bootstrap=40,
        random_state=1,
    )
    assert np.nanmean(wide.ci_upper - wide.ci_lower) > np.nanmean(
        narrow.ci_upper - narrow.ci_lower
    )


@pytest.mark.parametrize("name", ALL_NAMES)
def test_unsupported_derivative_order_is_refused(name):
    """Asking for a derivative a method cannot give must raise, not guess."""
    smoother = build_for_test(name)
    unsupported = max(smoother.supported_orders) + 1
    with pytest.raises(ValueError, match=r"order|derivative"):
        smoother.fit(AXIS, noisy(), derivative_order=unsupported)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_missing_values_are_refused_by_every_smoother(name):
    """A gapped series has no honest uncertainty, so it is refused at the door.

    Every route to a standard error gives a wrong answer on a gapped series
    rather than failing: the noise estimators splice across the gap, the block
    bootstrap draws blocks spanning it, the L1 penalty fraction lands on NaN and
    falls back to an absolute penalty, and the linearity check compares an
    all-NaN product, finds nothing finite and returns without checking.
    """
    y = noisy()
    y[40] = np.nan
    with pytest.raises(ValueError, match="missing value"):
        build_for_test(name).fit(AXIS, y, derivative_order=1, with_uncertainty=True)


def test_the_refusal_names_the_count_and_the_remedy():
    """An error that does not say what to do just relocates the problem."""
    y = noisy()
    y[10:15] = np.nan
    with pytest.raises(ValueError, match="missing value") as raised:
        SavitzkyGolay(window_length=15).fit(AXIS, y, derivative_order=1)
    message = str(raised.value)
    assert "5 missing values" in message
    assert "interpolate" in message
    assert "dropna" in message

    y_one = noisy()
    y_one[10] = np.nan
    with pytest.raises(ValueError, match="missing value") as single:
        SavitzkyGolay(window_length=15).fit(AXIS, y_one, derivative_order=1)
    assert "1 missing value;" in str(single.value), str(single.value)


def test_a_gap_can_no_longer_inflate_the_noise_estimate():
    """Regression: dropping a gap and differencing across it reads as noise.

    On a trending series a 20-point gap took the estimated noise level from
    0.286 to 1.782 and the AR(1) scale to 8.11 with a spurious phi of 0.93.
    Standard errors six times too wide report a real trend as insignificant,
    which is worse than refusing, because the caller cannot tell.
    """
    from incline.noise import estimate_ar1, rice_sigma

    rng = np.random.default_rng(0)
    axis = TimeAxis.positional(200)
    truth = 0.3
    y = 2.0 * axis.x + rng.normal(0, truth, 200)
    assert rice_sigma(y) == pytest.approx(truth, rel=0.15)

    gapped = y.copy()
    gapped[100:120] = np.nan
    # The estimators themselves are still gap-blind; what changed is that no
    # public path can reach them with a gap.
    assert rice_sigma(gapped) > 4 * truth
    assert estimate_ar1(gapped)[1] > 4 * truth
    with pytest.raises(ValueError, match="missing value"):
        SavitzkyGolay(window_length=15).fit(
            axis, gapped, derivative_order=1, with_uncertainty=True
        )


def test_an_infinite_value_is_refused_too():
    """isfinite, not isnan -- an overflow is just as unusable as a gap."""
    y = noisy()
    y[20] = np.inf
    with pytest.raises(ValueError, match="missing value"):
        SavitzkyGolay(window_length=15).fit(AXIS, y, derivative_order=1)
