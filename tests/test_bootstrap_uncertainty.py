"""Is the bootstrap path's uncertainty any good?

Every Monte Carlo test in this suite ran on the *operator* path: `IN_SPAN` in
test_econometrics.py lists only linear smoothers. The two smoothers that reach
their uncertainty by resampling -- `SmoothingSpline` with GCV and
`L1TrendFilter` --
had no coverage test at all. The 0.76 -> 0.975 figures in `incline.uncertainty`'s
module docstring are recorded measurements, not assertions: nothing in the suite
would notice if they regressed.

Unbiasedness is the wrong property for these two and is deliberately not tested
here, for the same reason `gp` and `kalman` are excluded from
test_econometrics.py. Both are adaptive: the L1 filter's soft-threshold step is
what produces changepoints, and the spline's knot placement and smoothing budget
are both functions of the data. Shrinkage toward smoothness is the design, not a
defect.

What is testable is the uncertainty, and "coverage" is not one question but
three. Coverage against the truth mixes the interval's *width*, its *centring*,
and the estimator's *bias*, and only the first two are the bootstrap's job. The
tests below separate them, hardest-to-fake first:

1. **se/sd** -- does the bootstrap recover the estimator's own sampling spread?
   Pure variance check, independent of any bias.
2. **Coverage against E[f-hat]** -- does the interval cover the thing it is
   actually centred on? Pure construction check. Must hold everywhere, including
   where the estimator is badly biased.
3. **Coverage against the truth, away from features** -- 1 and 2 plus bias, in a
   region where bias is small.
4. **Average coverage across the function** -- Nychka's (1988) framing for
   smoothing splines: pointwise coverage varies a lot, dipping where bias is a
   large share of error, but the average across the curve is close to nominal.
   Measured here, the pointwise se/sd ratio scatters over 0.84-1.18 while its
   average sits at 0.95-1.02, so the average is both the right quantity and the
   stable one.

Coverage where bias dominates is *characterised* rather than gated at nominal --
see the kink test at the bottom. Gating it at 0.95 would assert something untrue.

**The unit of replication is the replicate, never the point.** Whether the
interval covers at index 40 and whether it covers at index 41 are nearly the same
event: both come from one curve fitted to one dataset. Pooling 71 points and 100
replicates into 7100 binomials would multiply the apparent sample size by 71 while
adding almost no information. So the across-the-function test forms one number per
replicate and takes its Monte Carlo error from the spread of those.
"""

from __future__ import annotations

import numpy as np
import pytest

from incline.axis import TimeAxis
from incline.simulate import NoiseGenerator
from incline.smoothers import build
from tests._statistics import (
    DEEP_REPS,
    FAST_REPS,
    GATE_SIGMAS,
    MonteCarloResult,
    assert_se_calibrated,
    binomial_band,
)

N = 120
SIGMA = 0.5
# Adaptive spline selection needs more draws to stabilize the lower tail of its
# pointwise calibration ratios. The L1 filter is stable with fewer draws and is
# substantially more expensive to refit.
BOOTSTRAP_REPLICATES = {"smoothing_spline": 120, "l1_filter": 80}
# Away from both boundaries, where every smoother is a different estimator.
INTERIOR = np.arange(25, 96)
POINT = 60

AXIS = TimeAxis.positional(N)
_T = np.arange(N, dtype=float)

# A smooth truth inside the span of every smoother here, so bias is small and the
# tests below are about the uncertainty rather than about approximation error.
SMOOTH = 0.02 * (_T - 60) ** 2 / 100 + 0.05 * _T
SMOOTH_DERIVATIVE = 0.04 * (_T - 60) / 100 + 0.05

BOOTSTRAP_SMOOTHERS = ["smoothing_spline", "l1_filter"]

TIERS = [
    pytest.param(FAST_REPS, id="fast"),
    pytest.param(DEEP_REPS, id="deep", marks=pytest.mark.slow),
]


def _study(name: str, reps: int, seed0: int):
    """Refit a smoother over many noise draws and keep the whole curve.

    Args:
        name: Registered smoother name.
        reps: Replicates.
        seed0: First seed; replicate ``i`` uses ``seed0 + i``, so a study can be
            reproduced one replicate at a time and two studies with different
            ``seed0`` are independent.

    Returns:
        Tuple of ``(estimates, errors, lower, upper)``, each ``(reps, n)``.
    """
    estimates = np.empty((reps, N))
    errors = np.empty((reps, N))
    lower = np.empty((reps, N))
    upper = np.empty((reps, N))

    for i in range(reps):
        rng = np.random.default_rng(seed0 + i)
        smoother = (
            build(name, penalty_fraction=0.2) if name == "l1_filter" else build(name)
        )
        fitted = smoother.fit(
            AXIS,
            SMOOTH + NoiseGenerator.white(N, SIGMA, rng),
            derivative_order=1,
            with_uncertainty=True,
            n_bootstrap=BOOTSTRAP_REPLICATES[name],
            # Seed the resampling too, not just the noise draw. Without this the
            # bootstrap runs off an unseeded generator and the whole study is
            # irreproducible: the same input gave se[60] of 0.4411 and then
            # 0.4756 on two consecutive calls. A gate whose value moves between
            # runs cannot distinguish a regression from the dice.
            random_state=seed0 + i,
        )
        estimates[i] = fitted.derivative
        errors[i] = fitted.standard_error
        lower[i] = fitted.ci_lower
        upper[i] = fitted.ci_upper

    return estimates, errors, lower, upper


def _expected_derivative(name: str, reps: int, seed0: int) -> float:
    """Estimate the smoother's expectation from independent point fits.

    The expectation is a target for a separate bootstrap interval study. It
    needs independent noisy datasets, but not bootstrap intervals of its own.

    Args:
        name: Registered smoother name.
        reps: Independent point-estimate replicates.
        seed0: First seed.

    Returns:
        Mean derivative estimate at ``POINT``.
    """
    estimates = np.empty(reps)
    for i in range(reps):
        rng = np.random.default_rng(seed0 + i)
        smoother = (
            build(name, penalty_fraction=0.2) if name == "l1_filter" else build(name)
        )
        fitted = smoother.fit(
            AXIS,
            SMOOTH + NoiseGenerator.white(N, SIGMA, rng),
            derivative_order=1,
        )
        estimates[i] = fitted.derivative[POINT]
    return float(estimates.mean())


@pytest.fixture(scope="module")
def studies():
    """One study per smoother, reused across the tests that share it.

    Returns:
        Callable taking ``(name, reps)`` and returning the cached study.
    """
    cache: dict[tuple[str, int, int], tuple] = {}

    def get(name: str, reps: int, seed0: int = 5000):
        key = (name, reps, seed0)
        if key not in cache:
            cache[key] = _study(name, reps, seed0)
        return cache[key]

    return get


@pytest.fixture(scope="module")
def expectations():
    """Cache independently estimated point-estimator expectations.

    Returns:
        Callable taking ``(name, reps)`` and returning the expected derivative.
    """
    cache: dict[tuple[str, int, int], float] = {}

    def get(name: str, reps: int, seed0: int = 90000) -> float:
        key = (name, reps, seed0)
        if key not in cache:
            cache[key] = _expected_derivative(name, reps, seed0)
        return cache[key]

    return get


# --------------------------------------------------------------------------
# 1. Does the bootstrap recover the estimator's own spread?
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", BOOTSTRAP_SMOOTHERS)
@pytest.mark.parametrize("reps", TIERS)
def test_the_bootstrap_standard_error_matches_the_estimators_spread(
    name, reps, studies
):
    """se/sd must be near one, checked in **both** directions.

    The only calibration test the bootstrap path had was one-sided -- it bounded
    over-dispersion at 1.6x and let a standard error be arbitrarily small.
    Too-small is the direction that matters: it is the one that makes intervals
    under-cover and significance claims wrong.

    Measured, averaged over the interior: 0.946 for the spline and 1.017 for the
    L1 filter. Pointwise the ratio scatters over 0.84-1.18, which is why the
    average carries the tight gate and the pointwise check is loose.

    Args:
        name: Smoother under test.
        reps: Replicates for this tier.
        studies: Cached-study fixture.
    """
    estimates, errors, _, _ = studies(name, reps)

    spread = estimates[:, INTERIOR].std(axis=0, ddof=1)
    reported = errors[:, INTERIOR].mean(axis=0)
    ratios = reported / spread

    assert np.all(ratios > 0.7), (
        f"{name}: the bootstrap understates the spread at some points "
        f"(min ratio {ratios.min():.3f})"
    )
    assert np.all(ratios < 1.4), (
        f"{name}: the bootstrap overstates the spread at some points "
        f"(max ratio {ratios.max():.3f})"
    )
    average = float(ratios.mean())
    assert 0.85 < average < 1.15, (
        f"{name}: reported standard errors average {average:.3f} times the "
        "estimator's actual spread across the interior"
    )


# --------------------------------------------------------------------------
# 2. Does the interval cover what it is centred on?
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", BOOTSTRAP_SMOOTHERS)
@pytest.mark.parametrize("reps", TIERS)
def test_the_interval_covers_the_value_it_is_centred_on(
    name, reps, studies, expectations
):
    """Coverage of E[f-hat], which isolates construction from bias.

    An adaptive smoother is biased wherever the truth has a feature it must
    smooth over, and no interval built from the data alone knows that. Asking
    whether the interval covers its own expectation removes bias from the
    question entirely and leaves only: is the width right and is the centring
    right. That must hold everywhere.

    E[f-hat] is estimated from an **independent** block of replicates. Using the
    same replicates would centre the target by construction and make the test
    easier than it looks.

    Measured: 0.961 for the spline, 0.985 for the L1 filter.

    Args:
        name: Smoother under test.
        reps: Replicates for this tier.
        studies: Cached-study fixture.
        expectations: Independently estimated point-estimator expectations.
    """
    _, _, lower, upper = studies(name, reps)
    pseudo_truth = expectations(name, reps)

    covered = (lower[:, POINT] <= pseudo_truth) & (pseudo_truth <= upper[:, POINT])
    band = binomial_band(0.95, reps)
    rate = float(covered.mean())

    assert rate >= band[0], (
        f"{name}: the interval covers its own expectation only {rate:.3f} of the "
        f"time, below the {band[0]:.3f} floor for a nominal 0.95 over {reps} "
        "replicates -- the interval is too narrow or is not centred on the "
        "estimator"
    )


# --------------------------------------------------------------------------
# 3 and 4. Coverage of the truth, pointwise and across the function.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", BOOTSTRAP_SMOOTHERS)
@pytest.mark.parametrize("reps", TIERS)
def test_coverage_of_the_truth_away_from_features(name, reps, studies):
    """On a smooth truth, bias is small, so coverage should reach nominal.

    This is the first test in the file whose failure could be the *estimator's*
    fault rather than the interval's, which is why it comes after the two that
    cannot be.

    Args:
        name: Smoother under test.
        reps: Replicates for this tier.
        studies: Cached-study fixture.
    """
    _, _, lower, upper = studies(name, reps)
    target = SMOOTH_DERIVATIVE[POINT]

    covered = (lower[:, POINT] <= target) & (target <= upper[:, POINT])
    band = binomial_band(0.95, reps)
    rate = float(covered.mean())

    assert rate >= band[0], (
        f"{name}: covers the truth {rate:.3f} of the time at a point where bias "
        f"is small, below the {band[0]:.3f} floor over {reps} replicates"
    )


@pytest.mark.parametrize("name", BOOTSTRAP_SMOOTHERS)
@pytest.mark.parametrize("reps", TIERS)
def test_average_coverage_across_the_function(name, reps, studies):
    """Nychka's across-the-function coverage, with the replicate as the unit.

    Pointwise coverage of a smoother varies along the curve -- it dips where the
    bias is a large share of the error -- while the average over the curve sits
    near nominal, because the average posterior variance tracks the average
    squared error. That makes the average the number worth gating.

    One fraction per replicate, so the Monte Carlo error comes from the spread
    across replicates. Pooling 71 points x 100 replicates into 7100 binomials
    would inflate the apparent sample size by a factor of 71 and make a badly
    calibrated interval look precisely measured.

    Measured: 0.965 for the spline (MC SE 0.0025), 0.984 for the L1 filter
    (MC SE 0.0032). Both sit above nominal: the percentile bootstrap refits the
    smoother on every resample, so re-selection of knots widens the interval.
    Conservative, and stated rather than hidden.

    Args:
        name: Smoother under test.
        reps: Replicates for this tier.
        studies: Cached-study fixture.
    """
    _, _, lower, upper = studies(name, reps)
    target = SMOOTH_DERIVATIVE[INTERIOR]

    per_replicate = (
        (lower[:, INTERIOR] <= target) & (target <= upper[:, INTERIOR])
    ).mean(axis=1)

    average = float(per_replicate.mean())
    mc_se = float(per_replicate.std(ddof=1) / np.sqrt(reps))

    assert average >= 0.90, (
        f"{name}: average coverage across the function is {average:.3f} "
        f"(MC SE {mc_se:.4f}) against a nominal 0.95"
    )
    assert average <= 0.999, (
        f"{name}: average coverage is {average:.3f}, so the intervals are "
        "effectively vacuous"
    )


# --------------------------------------------------------------------------
# 5. Where it breaks, characterised rather than gated at nominal.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("reps", TIERS)
def test_l1_regularization_bias_near_a_slope_change(reps):
    """Separate regularization bias from bootstrap SE calibration.

    On a piecewise-linear truth whose slope jumps from 0.02 to 0.20, the L1
    filter's explicit 0.2 penalty fraction shrinks both slopes toward one
    another. The derivative does not exist at the kink itself, so this test uses
    fixed points ten observations to its left and right, where the truth is
    unambiguously 0.02 and 0.20.

    The bootstrap standard error should still match the estimator's sampling
    spread. Coverage of the scientific truth should not be advertised as
    nominal, however: residual resampling around the fitted trend cannot recover
    regularization bias. A separate smooth-truth test supplies the positive
    coverage gate.

    Args:
        reps: Replicates for this tier.
    """
    kink = 60
    truth = np.where(kink > _T, 0.02 * _T, 0.02 * kink + 0.20 * (_T - kink))
    points = np.array([50, 70])
    targets = np.array([0.02, 0.20])
    estimates = np.empty((reps, len(points)))
    standard_errors = np.empty_like(estimates)
    lowers = np.empty_like(estimates)
    uppers = np.empty_like(estimates)
    for i in range(reps):
        rng = np.random.default_rng(8000 + i)
        fitted = build("l1_filter", penalty_fraction=0.2).fit(
            AXIS,
            truth + NoiseGenerator.white(N, SIGMA, rng),
            derivative_order=1,
            with_uncertainty=True,
            n_bootstrap=BOOTSTRAP_REPLICATES["l1_filter"],
            random_state=8000 + i,
        )
        estimates[i] = fitted.derivative[points]
        standard_errors[i] = fitted.standard_error[points]
        lowers[i] = fitted.ci_lower[points]
        uppers[i] = fitted.ci_upper[points]

    for column, (point, target) in enumerate(zip(points, targets, strict=True)):
        study = MonteCarloResult(
            estimates=estimates[:, column],
            standard_errors=standard_errors[:, column],
            covered=((lowers[:, column] <= target) & (target <= uppers[:, column])),
            rejected=None,
            truth=float(target),
        )
        label = f"L1 filter at x={point} around a slope change"
        assert abs(study.bias_t) >= GATE_SIGMAS, (
            f"{label}: the documented regularization bias is no longer resolved "
            f"({study.bias:+.6f}, {study.bias_t:+.2f} Monte Carlo SEs); reassess "
            "the limitation rather than preserving this expectation"
        )
        assert_se_calibrated(study, label)
        _, poor_coverage_ceiling = binomial_band(0.05, reps)
        assert study.coverage <= poor_coverage_ceiling, (
            f"{label}: coverage improved to {study.coverage:.3f}, above the "
            f"{poor_coverage_ceiling:.3f} ceiling for a 0.05 claim; reassess the "
            "documented limitation"
        )
