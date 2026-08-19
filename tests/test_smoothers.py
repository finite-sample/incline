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

    Sigma is stated rather than estimated on both sides. Estimating it under
    each model separately compares two different processes: imposing phi=0.7 on
    genuinely independent data implies a marginal variance about four times
    larger, which is a fact about noise estimation and not about differencing.
    """
    y = noisy(5)
    naive = NaiveDifference()
    phi = 0.7
    independent = naive.fit(AXIS, y, order=1, se=True, noise=IID(sigma=0.25))
    correlated = naive.fit(AXIS, y, order=1, se=True, noise=AR1(phi=phi, sigma=0.25))
    ratio = float(np.nanmean(correlated.se) / np.nanmean(independent.se))
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


def test_noise_model_reaches_the_bootstrap(recwarn):
    """Regression: `noise=` was ignored by every bootstrapped smoother.

    The fitted noise model was computed and then discarded, and the bootstrap
    re-derived its own scale from the data, so an explicit sigma had no effect
    on the reported uncertainty at all.
    """
    del recwarn
    y = noisy(21)
    smoother = InterpolatingSpline()
    small = smoother.fit(
        AXIS,
        y,
        order=1,
        se=True,
        noise=IID(sigma=0.3),
        n_bootstrap=40,
        random_state=1,
    )
    large = smoother.fit(
        AXIS,
        y,
        order=1,
        se=True,
        noise=IID(sigma=3.0),
        n_bootstrap=40,
        random_state=1,
    )
    ratio = float(np.nanmean(large.se) / np.nanmean(small.se))
    assert 5.0 < ratio < 20.0, (
        f"a tenfold noise level changed the standard error by {ratio:.2f}x"
    )


def test_heteroskedastic_noise_reaches_the_bootstrap():
    """A varying scale must shape a bootstrapped interval too, not just an exact one."""
    from incline.noise import Heteroskedastic

    ramp = np.linspace(0.1, 2.0, N)
    rng = np.random.default_rng(23)
    y = 0.05 * AXIS.x + rng.normal(0, 1, N) * ramp
    estimate = InterpolatingSpline().fit(
        AXIS,
        y,
        order=1,
        se=True,
        noise=Heteroskedastic(sigma=ramp),
        n_bootstrap=60,
        random_state=2,
    )
    quiet = float(np.nanmean(estimate.se[10:35]))
    loud = float(np.nanmean(estimate.se[-35:-10]))
    assert loud > 2.0 * quiet, f"quiet end {quiet:.4f}, noisy end {loud:.4f}"


def test_grid_methods_warn_on_an_irregular_axis():
    """Regression: TimeAxis.require_regular existed but was never called.

    A Savitzky-Golay filter applied to an unevenly spaced series scales every
    derivative by one median step, so callers got silently wrong per-time
    slopes with no warning at all.
    """
    from incline.axis import TimeAxis as Axis

    uneven = Axis._build(np.sort(np.random.default_rng(1).uniform(0, 100, 60)), "index")
    y = 2.0 * uneven.x
    with pytest.warns(UserWarning, match="uniform sampling"):
        SavitzkyGolay(window_length=11).fit(uneven, y, order=1)


def test_grid_methods_stay_quiet_on_a_regular_axis():
    """The warning must not fire on the ordinary case."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SavitzkyGolay(window_length=11).fit(AXIS, noisy(3), order=1)


def test_only_grid_methods_declare_the_requirement():
    """A spline or LOESS handles irregular spacing natively and must not warn."""
    assert SavitzkyGolay.requires_regular_grid
    assert NaiveDifference.requires_regular_grid
    assert not LocalPolynomial.requires_regular_grid
    assert not InterpolatingSpline.requires_regular_grid


def test_an_estimated_ar1_scale_does_not_inflate_the_bootstrap():
    """Regression: honoring `noise=` must not mean adopting every estimate.

    Passing estimate_ar1's sigma to the bootstrap looks principled and measures
    badly. That estimator exists to pair with a full AR(1) covariance in
    `propagate`; used alone it runs about 1.4x high, and on top of an already
    over-dispersed block bootstrap it produced standard errors 2.8x the
    estimator's real spread. An estimated scalar scale is therefore left to the
    bootstrap's own difference-based estimate, which its rescaling was
    calibrated against.
    """
    from incline.noise import AR1
    from incline.simulate import NoiseGenerator

    rng = np.random.default_rng(5)
    truth = 0.7 - 0.4 * AXIS.x
    smoother = InterpolatingSpline()

    estimates, errors = [], []
    for _ in range(40):
        y = truth + NoiseGenerator.ar1(N, 0.7, 0.4, rng)
        fitted = smoother.fit(
            AXIS,
            y,
            order=1,
            se=True,
            noise=AR1(),
            n_bootstrap=40,
            random_state=int(rng.integers(1 << 30)),
        )
        estimates.append(fitted.derivative[N // 2])
        errors.append(fitted.se[N // 2])

    ratio = float(np.mean(errors) / np.std(estimates))
    assert ratio < 1.6, f"the bootstrap reports {ratio:.2f}x the estimator's spread"
    # Bounded below as well. This assertion used to carry only the upper bound,
    # so it could not fail in the direction that matters: a standard error that
    # is too *small* is what makes intervals under-cover and significance claims
    # wrong, and nothing here forbade it.
    #
    # The floor is 0.5 rather than something near 1 because the measurement says
    # so. Under AR(1) noise this bootstrap runs low -- 0.59, 0.61 and 0.83 across
    # three seeds, and 0.72 with coverage 0.817 against a nominal 0.95 over 120
    # replicates. That is a real shortfall, not a tolerance to widen until it
    # passes. Compare test_bootstrap_uncertainty.py, where the same ratio under
    # iid noise averages 0.95 to 1.02. This is a regression guard on a
    # known-imperfect number, not a claim that the number is right.
    assert ratio > 0.5, (
        f"the bootstrap reports {ratio:.2f}x the estimator's spread, so its "
        "intervals are far too narrow"
    )


def test_a_stated_scale_is_still_adopted_by_the_bootstrap():
    """The review finding stays fixed: an explicit sigma must reach the interval."""
    from incline.noise import AR1

    y = noisy(31)
    smoother = InterpolatingSpline()
    small = smoother.fit(
        AXIS,
        y,
        order=1,
        se=True,
        noise=AR1(sigma=0.3, phi=0.5),
        n_bootstrap=40,
        random_state=1,
    )
    large = smoother.fit(
        AXIS,
        y,
        order=1,
        se=True,
        noise=AR1(sigma=3.0, phi=0.5),
        n_bootstrap=40,
        random_state=1,
    )
    assert np.nanmean(large.se) > 5.0 * np.nanmean(small.se)


def test_a_series_shorter_than_the_window_says_so():
    """Regression: Savitzky-Golay raised from deep inside scipy on short input.

    ``savgol_filter`` reported "polyorder must be less than window_length",
    naming neither the series length nor the remedy. Below the polynomial order
    there is no window that could work, so the estimator has to say that.
    """
    short = TimeAxis.positional(4)
    with pytest.raises(ValueError, match="at least"):
        SavitzkyGolay(window_length=21).fit(short, np.arange(4.0), order=2)


def test_a_short_series_still_fits_when_a_window_exists():
    """The clamp still applies wherever the polynomial order allows one."""
    axis = TimeAxis.positional(9)
    estimate = SavitzkyGolay(window_length=41).fit(axis, np.arange(9.0), order=1)
    assert np.all(np.isfinite(estimate.derivative))


@pytest.mark.parametrize("name", ["gp", "kalman"])
def test_a_native_posterior_says_when_an_argument_does_not_apply(name):
    """Regression: noise= and simultaneous= were accepted and then ignored.

    Neither smoother exposes a linear operator, so its interval comes from its
    own posterior. Silently discarding the caller's noise model let a series be
    reported under assumptions nobody chose.
    """
    smoother = build(name)
    with pytest.warns(UserWarning, match="posterior"):
        result = smoother.fit(AXIS, noisy(), order=1, se=True, noise=AR1())
    assert result.provenance.noise is None

    with pytest.warns(UserWarning, match="whole-curve"):
        pointwise = smoother.fit(AXIS, noisy(), order=1, se=True, simultaneous=True)
    assert not pointwise.provenance.simultaneous


def test_a_noise_model_that_does_apply_passes_without_a_warning():
    """The warning has to be about applicability, not about the argument."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SavitzkyGolay(window_length=15).fit(
            AXIS, noisy(), order=1, se=True, noise=AR1()
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
        build(name).fit(AXIS, y, order=1, se=True)


def test_the_refusal_names_the_count_and_the_remedy():
    """An error that does not say what to do just relocates the problem."""
    y = noisy()
    y[10:15] = np.nan
    with pytest.raises(ValueError, match="missing value") as raised:
        SavitzkyGolay(window_length=15).fit(AXIS, y, order=1)
    message = str(raised.value)
    assert "5 missing values" in message
    assert "interpolate" in message
    assert "dropna" in message

    y_one = noisy()
    y_one[10] = np.nan
    with pytest.raises(ValueError, match="missing value") as single:
        SavitzkyGolay(window_length=15).fit(AXIS, y_one, order=1)
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
        SavitzkyGolay(window_length=15).fit(axis, gapped, order=1, se=True)


def test_an_infinite_value_is_refused_too():
    """isfinite, not isnan -- an overflow is just as unusable as a gap."""
    y = noisy()
    y[20] = np.inf
    with pytest.raises(ValueError, match="missing value"):
        SavitzkyGolay(window_length=15).fit(AXIS, y, order=1)
