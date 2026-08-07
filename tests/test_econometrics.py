"""Does each estimator do what an estimator is supposed to do?

Everything else in the suite checks plumbing -- schemas, shapes, that nothing
crashes. This file checks the statistics. Four properties, each building on the
one before:

1. **Exactness.** When the truth lies inside an estimator's approximation space,
   it reproduces the truth with no bias at all. Deterministic, so this is
   settled to machine precision rather than to within Monte Carlo error.
2. **Unbiasedness.** Add noise, and the estimator's mean must still sit on the
   truth -- tested as a t statistic against its own sampling variability.
3. **Coverage.** With bias eliminated by construction, a 95% interval must
   contain the true derivative 95% of the time. This is the property a standard
   error actually claims, and it is only cleanly testable because step 1 holds.
4. **Size and power.** Under no trend, ``significant_trend`` must fire at about
   the nominal rate; under a real trend it must fire more often as the signal
   grows. The old SiZer spline branch flagged ~90% of pure noise, which is what
   a size failure looks like.

Why polynomials. Every linear smoother here reproduces polynomials of its own
order exactly, so choosing a truth inside the span makes the bias identically
zero and lets coverage be measured against the *true* derivative. Against a
general truth, smoothing bias dominates and coverage collapses for reasons that
have nothing to do with whether the variance is right -- which is a fact about
bandwidth selection, not a defect, and is documented in ``docs/limitations.md``.

Gaussian processes and state-space models are deliberately absent from the
unbiasedness and coverage gates. They are shrinkage estimators, biased by
construction; the right guarantee for them is that credible intervals cover when
the data come from their prior, which is not yet implemented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from incline import api
from incline import smoothers as sm
from incline.axis import TimeAxis
from tests._statistics import (
    DEEP_REPS,
    FAST_REPS,
    assert_coverage,
    assert_proportion,
    assert_unbiased,
    binomial_band,
    monte_carlo,
)


N = 120
AXIS = TimeAxis.positional(N)
SIGMA = 0.5
POINT = 60  # one interior point, well away from the boundary
INTERIOR = slice(25, -25)

TIERS = [
    pytest.param(FAST_REPS, id="fast"),
    pytest.param(DEEP_REPS, id="deep", marks=pytest.mark.slow),
]


def polynomial(coefficients: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """The values and first derivative of a polynomial on the shared axis."""
    x = AXIS.x
    degree = len(coefficients) - 1
    values = sum(coefficients[k] * x**k for k in range(degree + 1))
    derivative = sum(k * coefficients[k] * x ** (k - 1) for k in range(1, degree + 1))
    return np.asarray(values, dtype=float), np.asarray(derivative, dtype=float)


# Each estimator paired with a polynomial that lies inside its span. A local
# polynomial of degree d reproduces degree-d truths; a penalized spline whose
# penalty is on curvature reproduces straight lines, since those incur no
# penalty at all.
IN_SPAN = [
    pytest.param(
        sm.LocalPolynomial(bandwidth=0.25, degree=2),
        [0.7, -0.4, 0.015],
        id="local_poly_d2_quadratic",
    ),
    pytest.param(
        sm.LocalPolynomial(bandwidth=0.25, degree=1),
        [0.7, -0.4],
        id="local_poly_d1_linear",
    ),
    pytest.param(
        sm.SavitzkyGolay(window_length=21, polyorder=3),
        [0.7, -0.4, 0.015, 2e-4],
        id="sgolay_p3_cubic",
    ),
    pytest.param(
        sm.SavitzkyGolay(window_length=21, polyorder=2),
        [0.7, -0.4, 0.015],
        id="sgolay_p2_quadratic",
    ),
    pytest.param(sm.NaiveDifference(), [0.7, -0.4], id="naive_linear"),
    pytest.param(
        sm.Loess(frac=0.3, degree=1, robust=False), [0.7, -0.4], id="loess_d1_linear"
    ),
    pytest.param(sm.PenalizedSpline(lam=1e3), [0.7, -0.4], id="pspline_linear"),
]


# ---------------------------------------------------------------------------
# 1. Exactness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("smoother", "coefficients"), IN_SPAN)
def test_truth_in_span_is_reproduced_exactly(smoother, coefficients):
    """No noise, truth inside the span: the bias must be zero, not merely small.

    This is the sharpest unbiasedness statement available and it costs one fit.
    If it fails, every Monte Carlo result below is measuring the wrong thing.
    """
    values, derivative = polynomial(coefficients)
    estimated = smoother.evaluate(AXIS, values, 1).derivative

    scale = float(np.nanmax(np.abs(derivative[INTERIOR])))
    error = float(np.nanmax(np.abs(estimated[INTERIOR] - derivative[INTERIOR])))
    assert error / scale < 1e-9, f"relative bias {error / scale:.2e}"


def test_a_truth_outside_the_span_is_not_reproduced():
    """The exactness test above must be capable of failing.

    A degree-1 local polynomial cannot reproduce a quadratic's slope, and if
    this passed it would mean the check is vacuous.
    """
    values, derivative = polynomial([0.7, -0.4, 0.05])
    estimated = (
        sm.LocalPolynomial(bandwidth=0.3, degree=1).evaluate(AXIS, values, 1).derivative
    )
    scale = float(np.nanmax(np.abs(derivative[INTERIOR])))
    error = float(np.nanmax(np.abs(estimated[INTERIOR] - derivative[INTERIOR])))
    assert error / scale > 1e-6


# ---------------------------------------------------------------------------
# 2. Unbiasedness under noise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reps", TIERS)
@pytest.mark.parametrize(("smoother", "coefficients"), IN_SPAN)
def test_estimator_is_unbiased_under_noise(smoother, coefficients, reps, capsys):
    """Adding noise must not shift the estimator's mean off the truth."""
    values, derivative = polynomial(coefficients)
    result = monte_carlo(smoother, AXIS, values, derivative, POINT, reps, sigma=SIGMA)
    with capsys.disabled():
        print("  " + result.summary(f"unbiased/{smoother.name}"))
    assert_unbiased(result, smoother.name)


# ---------------------------------------------------------------------------
# 3. Coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reps", TIERS)
@pytest.mark.parametrize(("smoother", "coefficients"), IN_SPAN)
def test_interval_covers_the_true_derivative_at_nominal(
    smoother, coefficients, reps, capsys
):
    """A 95% interval must contain the truth 95% of the time.

    With bias eliminated by construction this is a clean statement about the
    standard error, gated against the binomial spread rather than a hand-picked
    tolerance.
    """
    values, derivative = polynomial(coefficients)
    result = monte_carlo(smoother, AXIS, values, derivative, POINT, reps, sigma=SIGMA)
    with capsys.disabled():
        print("  " + result.summary(f"coverage/{smoother.name}"))
    assert_coverage(result, 0.95, smoother.name)
    assert 0.75 < result.se_ratio < 1.35, (
        f"{smoother.name}: reported SE is {result.se_ratio:.3f}x the true spread"
    )


@pytest.mark.parametrize("reps", TIERS)
def test_a_deliberately_wrong_standard_error_fails_the_coverage_gate(reps):
    """The gate must be able to fail, or it proves nothing.

    Halving a correct standard error should drive coverage far below nominal
    and trip the binomial band.
    """
    values, derivative = polynomial([0.7, -0.4])
    result = monte_carlo(
        sm.SavitzkyGolay(window_length=21, polyorder=2),
        AXIS,
        values,
        derivative,
        POINT,
        reps,
        sigma=SIGMA,
    )
    z = 1.959963985
    halved = np.abs(result.estimates - result.truth) <= z * (result.standard_errors / 2)
    low, _ = binomial_band(0.95, reps)
    assert float(np.mean(halved)) < low


# ---------------------------------------------------------------------------
# 4. Size and power
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reps", TIERS)
@pytest.mark.parametrize(("smoother", "coefficients"), IN_SPAN)
def test_size_under_no_trend(smoother, coefficients, reps, capsys):
    """With a flat truth, a 95% interval must exclude zero about 5% of the time.

    Over-rejection here is what a too-small standard error looks like from the
    user's side: apparent trends that are not there.
    """
    del coefficients
    flat = np.full(N, 3.0)
    zero = np.zeros(N)
    result = monte_carlo(smoother, AXIS, flat, zero, POINT, reps, sigma=SIGMA)
    with capsys.disabled():
        print("  " + result.summary(f"size/{smoother.name}"))
    assert_proportion(result.rejection_rate, reps, 0.05, f"{smoother.name} size")


@pytest.mark.parametrize("reps", TIERS)
def test_power_rises_with_the_signal(reps, capsys):
    """A stronger trend must be detected more often, and a strong one always."""
    smoother = sm.SavitzkyGolay(window_length=21, polyorder=2)
    rates = []
    for slope in (0.0, 0.02, 0.05, 0.20):
        values, derivative = polynomial([0.0, slope])
        result = monte_carlo(
            smoother, AXIS, values, derivative, POINT, reps, sigma=SIGMA
        )
        rates.append(result.rejection_rate)
    with capsys.disabled():
        print(f"  power/sgolay slopes 0/.02/.05/.20 -> {[round(r, 3) for r in rates]}")

    assert rates == sorted(rates), f"power must not decrease with signal: {rates}"
    assert rates[0] < 0.15, f"size at zero slope is {rates[0]:.3f}"
    assert rates[-1] > 0.95, f"power at a strong trend is only {rates[-1]:.3f}"


# ---------------------------------------------------------------------------
# Noise misspecification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reps", TIERS)
def test_assuming_independence_under_autocorrelation_undercovers(reps, capsys):
    """The cost of the default assumption when it is wrong, stated as a number.

    Under AR(1) with phi=0.7 the independent-noise assumption reports standard
    errors around a third of their true size, and a nominal 95% interval covers
    well under half the time.
    """
    values, derivative = polynomial([0.7, -0.4])
    smoother = sm.SavitzkyGolay(window_length=21, polyorder=2)
    result = monte_carlo(
        smoother,
        AXIS,
        values,
        derivative,
        POINT,
        reps,
        sigma=SIGMA,
        phi=0.7,
        noise="iid",
    )
    with capsys.disabled():
        print("  " + result.summary("misspecified/iid-under-ar1"))
    assert result.se_ratio < 0.6
    assert result.coverage < 0.75


@pytest.mark.parametrize("reps", TIERS)
def test_modelling_the_autocorrelation_repairs_most_of_it(reps, capsys):
    """noise='ar1' recovers the variance, though not quite to nominal.

    The AR coefficient is itself estimated from a finite series, so the interval
    inherits that uncertainty. Coverage lands around 0.8 rather than 0.95 --
    much better than 0.42, and honest about not being perfect.
    """
    values, derivative = polynomial([0.7, -0.4])
    smoother = sm.SavitzkyGolay(window_length=21, polyorder=2)
    result = monte_carlo(
        smoother,
        AXIS,
        values,
        derivative,
        POINT,
        reps,
        sigma=SIGMA,
        phi=0.7,
        noise="ar1",
    )
    with capsys.disabled():
        print("  " + result.summary("modeled/ar1-under-ar1"))
    assert 0.7 < result.se_ratio < 1.5
    assert result.coverage > 0.7


# ---------------------------------------------------------------------------
# Regressions that are not about calibration
# ---------------------------------------------------------------------------


def test_gaussian_process_standard_error_is_not_absurd():
    """Regression: the reported variance was once 3,614x too large.

    The old implementation differenced the posterior mean at a hundredth of the
    sample spacing and summed the endpoint variances as though independent, so
    the standard error scaled with 1/dx, the intervals swallowed every plausible
    value, and the significance flag never fired.
    """
    x = np.linspace(0, 10, 120)
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {"value": 0.5 * x + np.sin(x) + rng.normal(0, 0.3, 120), "t": x}
    )

    result = api.gp_trend(frame, time_column="t", se=True)
    truth = 0.5 + np.cos(x)

    assert result["derivative_se"].median() < 1.0
    assert np.corrcoef(result["derivative_value"], truth)[0, 1] > 0.9
    covered = (
        (result["derivative_ci_lower"] <= truth)
        & (truth <= result["derivative_ci_upper"])
    ).mean()
    assert 0.8 <= covered <= 1.0
    assert result["significant_trend"].mean() > 0.5
