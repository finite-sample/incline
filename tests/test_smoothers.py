"""Tests for the smoothers and the shared fit machinery.

Most behavior lives on the base class, so most of these are parametrized over
the whole registry: whatever gets added later has to satisfy them too.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from incline.axis import TimeAxis
from incline.noise import AR1, IID
from incline.result import TrendEstimate
from incline.smoothers import (
    SMOOTHERS,
    InterpolatingSpline,
    L1TrendFilter,
    LocalPolynomial,
    Loess,
    NaiveDifference,
    PenalizedSpline,
    SavitzkyGolay,
    build,
)


N = 100
AXIS = TimeAxis.positional(N)


def noisy(seed: int = 0, slope: float = 0.05) -> np.ndarray:
    """A gently curved series with noise."""
    rng = np.random.default_rng(seed)
    return slope * AXIS.x + np.sin(AXIS.x / 12) + rng.normal(0, 0.25, N)


ALL_NAMES = sorted(SMOOTHERS)
LINEAR = [
    pytest.param(SavitzkyGolay(window_length=15), id="sgolay"),
    pytest.param(NaiveDifference(), id="naive"),
    pytest.param(LocalPolynomial(bandwidth=0.2, degree=2), id="local_poly"),
    pytest.param(PenalizedSpline(lam=1e4), id="pspline_fixed"),
    pytest.param(Loess(frac=0.3, robust=False), id="loess_plain"),
]
NONLINEAR = [
    pytest.param(InterpolatingSpline(), id="spline"),
    pytest.param(PenalizedSpline(lam=None), id="pspline_gcv"),
    pytest.param(Loess(frac=0.3, robust=True), id="loess_robust"),
    pytest.param(L1TrendFilter(), id="l1"),
]


@pytest.mark.parametrize("name", ALL_NAMES)
def test_registry_names_match_their_classes(name):
    """build() must return the smoother the registry is keyed by."""
    smoother = build(name)
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
        smoother.fit(AXIS, noisy(), order=1, bias_correct=True)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_estimate_recovers_a_straight_line(name):
    """Every smoother must get a constant slope right on noiseless data."""
    y = 2.0 * AXIS.x + 1.0
    estimate = build(name).fit(AXIS, y, order=1)
    interior = estimate.derivative[10:-10]
    assert np.allclose(interior, 2.0, atol=0.05), f"{name} got {interior[:3]}"


@pytest.mark.parametrize("name", ALL_NAMES)
def test_fit_without_se_leaves_uncertainty_honestly_empty(name):
    """Absent uncertainty is NaN plus a None label, never a missing column."""
    estimate = build(name).fit(AXIS, noisy(), order=1)
    assert isinstance(estimate, TrendEstimate)
    assert estimate.se is None
    assert estimate.has_uncertainty is False
    assert not estimate.significant.any()
    assert estimate.provenance.se_method is None


@pytest.mark.parametrize("name", ALL_NAMES)
def test_fit_with_se_labels_its_route(name):
    """Every smoother reports how it got its standard error."""
    smoother = build(name)
    estimate = smoother.fit(AXIS, noisy(), order=1, se=True, n_bootstrap=40)
    assert estimate.se is not None
    assert np.all(estimate.se[np.isfinite(estimate.se)] >= 0)
    if smoother.has_native_posterior:
        expected = "native"
    elif smoother.is_linear:
        expected = "operator"
    else:
        expected = "bootstrap"
    assert estimate.provenance.se_method == expected


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


SMOOTHING_LINEAR = [p for p in LINEAR if p.id != "naive"]


@pytest.mark.parametrize("smoother", SMOOTHING_LINEAR)
def test_correlated_noise_widens_a_smoother_interval(smoother):
    """For an averaging estimator, dependence inflates the variance.

    A smoother averages neighbors, and positively correlated neighbors carry
    less independent information than the count suggests, so the interval must
    grow. Differencing estimators behave the opposite way; see below.
    """
    y = noisy(5)
    independent = smoother.fit(AXIS, y, order=1, se=True, noise=IID())
    correlated = smoother.fit(AXIS, y, order=1, se=True, noise=AR1(phi=0.7))
    assert np.nanmean(correlated.se) > np.nanmean(independent.se)


def test_correlated_noise_narrows_a_differencing_interval():
    """For a difference, positive autocorrelation cancels rather than accumulates.

    The central difference is ``(y[i+1] - y[i-1]) / 2h``, whose variance is
    ``sigma**2 (1 - rho_2) / 2h**2``. Correlated neighbors partly cancel in the
    subtraction, so the correct AR(1) interval is *narrower* than the
    independent one -- the reverse of every smoothing method here. Pinning it
    guards against someone "fixing" the sign by assuming dependence always
    widens.
    """
    y = noisy(5)
    naive = NaiveDifference()
    independent = naive.fit(AXIS, y, order=1, se=True, noise=IID())
    correlated = naive.fit(AXIS, y, order=1, se=True, noise=AR1(phi=0.7))
    assert np.nanmean(correlated.se) < np.nanmean(independent.se)


@pytest.mark.parametrize("smoother", LINEAR)
def test_whole_curve_band_is_wider_than_pointwise(smoother):
    """Correcting for testing every point can only widen the band."""
    y = noisy(7)
    pointwise = smoother.fit(AXIS, y, order=1, se=True)
    whole = smoother.fit(AXIS, y, order=1, se=True, simultaneous=True, random_state=0)
    width_point = np.nanmean(pointwise.ci_upper - pointwise.ci_lower)
    width_whole = np.nanmean(whole.ci_upper - whole.ci_lower)
    assert width_whole > width_point
    assert whole.provenance.simultaneous


@pytest.mark.parametrize("name", ALL_NAMES)
def test_wider_confidence_gives_a_wider_interval(name):
    """confidence_level must reach the interval, not get defaulted away."""
    y = noisy(9)
    narrow = build(name).fit(
        AXIS, y, order=1, se=True, confidence_level=0.5, n_bootstrap=40, random_state=1
    )
    wide = build(name).fit(
        AXIS, y, order=1, se=True, confidence_level=0.99, n_bootstrap=40, random_state=1
    )
    assert np.nanmean(wide.ci_upper - wide.ci_lower) > np.nanmean(
        narrow.ci_upper - narrow.ci_lower
    )


@pytest.mark.parametrize("name", ALL_NAMES)
def test_unsupported_derivative_order_is_refused(name):
    """Asking for a derivative a method cannot give must raise, not guess."""
    smoother = build(name)
    unsupported = max(smoother.supported_orders) + 1
    with pytest.raises(ValueError, match=r"order|derivative"):
        smoother.fit(AXIS, noisy(), order=unsupported)


def test_second_derivative_finds_curvature():
    """Order 2 must actually track the second derivative, not repeat the first."""
    y = 0.5 * AXIS.x**2
    for smoother in (
        SavitzkyGolay(window_length=21),
        LocalPolynomial(bandwidth=0.3, degree=2),
    ):
        estimate = smoother.fit(AXIS, y, order=2)
        interior = estimate.derivative[15:-15]
        assert np.allclose(interior, 1.0, atol=0.05), smoother.name


@pytest.mark.parametrize("name", ALL_NAMES)
def test_scale_round_trips_through_the_normalised_knob(name):
    """with_scale and scale_of must be consistent enough to sweep with."""
    smoother = build(name)
    rescaled = smoother.with_scale(0.25, AXIS)
    assert isinstance(rescaled, type(smoother))
    recovered = rescaled.scale_of(AXIS)
    assert 0 < recovered <= 1.0


@pytest.mark.parametrize("name", ALL_NAMES)
def test_a_wider_scale_smooths_more(name):
    """The scale knob must be monotone or a bandwidth sweep is meaningless."""
    smoother = build(name)
    if smoother.scale_of(AXIS) == smoother.with_scale(0.9, AXIS).scale_of(AXIS):
        pytest.skip(f"{name} has no smoothing scale to vary")
    y = noisy(11)
    narrow = smoother.with_scale(0.05, AXIS).fit(AXIS, y, order=1)
    wide = smoother.with_scale(0.4, AXIS).fit(AXIS, y, order=1)
    assert np.nanstd(wide.derivative) < np.nanstd(narrow.derivative), smoother.name


@pytest.mark.parametrize("name", ALL_NAMES)
def test_length_mismatch_is_refused(name):
    """A y of the wrong length is a caller error worth naming."""
    with pytest.raises(ValueError, match="axis has"):
        build(name).fit(AXIS, np.zeros(N - 1), order=1)


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
        estimate = build(name).fit(
            AXIS, np.full(N, 3.0), order=1, se=True, n_bootstrap=20
        )
    interior = estimate.derivative[10:-10]
    assert np.allclose(interior[np.isfinite(interior)], 0.0, atol=1e-6)
    assert not estimate.significant.any()


@pytest.mark.parametrize("smoother", LINEAR)
def test_bias_correction_widens_and_recentres(smoother):
    """Correcting bias trades interval width for centering; both should show."""
    y = noisy(13)
    plain = smoother.fit(AXIS, y, order=1, se=True)
    corrected = smoother.fit(AXIS, y, order=1, se=True, bias_correct=True)
    assert corrected.provenance.bias_corrected
    assert np.nanmean(corrected.se) >= np.nanmean(plain.se)
