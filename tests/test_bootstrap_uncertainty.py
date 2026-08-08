"""Is the bootstrap path's uncertainty any good?

Every Monte Carlo test in this suite ran on the *operator* path: `IN_SPAN` in
test_econometrics.py lists only linear smoothers. The two smoothers that reach
their uncertainty by resampling -- `InterpolatingSpline` and `L1TrendFilter` --
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
    binomial_band,
)


N = 120
SIGMA = 0.5
# Enough resamples for the bootstrap's own noise to sit well under the sampling
# spread being measured, and few enough that the file runs in about a minute:
# 80 costs 33 ms a fit for the spline and 114 ms for the L1 filter.
N_BOOTSTRAP = 80
# Away from both boundaries, where every smoother is a different estimator.
INTERIOR = np.arange(25, 96)
POINT = 60

AXIS = TimeAxis.positional(N)
_T = np.arange(N, dtype=float)

# A smooth truth inside the span of every smoother here, so bias is small and the
# tests below are about the uncertainty rather than about approximation error.
SMOOTH = 0.02 * (_T - 60) ** 2 / 100 + 0.05 * _T
SMOOTH_DERIVATIVE = 0.04 * (_T - 60) / 100 + 0.05

BOOTSTRAP_SMOOTHERS = ["spline", "l1_filter"]

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
        fitted = build(name).fit(
            AXIS,
            SMOOTH + NoiseGenerator.white(N, SIGMA, rng),
            order=1,
            se=True,
            n_bootstrap=N_BOOTSTRAP,
            # Seed the resampling too, not just the noise draw. Without this the
            # bootstrap runs off an unseeded generator and the whole study is
            # irreproducible: the same input gave se[60] of 0.4411 and then
            # 0.4756 on two consecutive calls. A gate whose value moves between
            # runs cannot distinguish a regression from the dice.
            random_state=seed0 + i,
        )
        estimates[i] = fitted.derivative
        errors[i] = fitted.se
        lower[i] = fitted.ci_lower
        upper[i] = fitted.ci_upper

    return estimates, errors, lower, upper


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
def test_the_interval_covers_the_value_it_is_centred_on(name, reps, studies):
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
    """
    _, _, lower, upper = studies(name, reps)
    independent, _, _, _ = studies(name, reps, seed0=90000)
    pseudo_truth = independent[:, POINT].mean()

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
def test_the_l1_filter_undercovers_at_a_slope_change(reps):
    """A known limitation, pinned so it cannot get quietly worse.

    On a piecewise-linear truth whose slope jumps from 0.02 to 0.20, the L1
    filter's default penalty smooths the corner. The derivative at the corner is
    biased by about -0.09, roughly 14 Monte Carlo standard errors, and coverage
    of the true slope falls to 0.67.

    The interval is not at fault, and this test exists partly to say so: at the
    same point the standard error is about 0.94 of the estimator's real spread
    and the interval covers its own expectation essentially always. The failure
    is bias, which no interval computed from the data alone can know about.

    So the gate is a floor at the measured level, not at nominal. Asserting 0.95
    here would be asserting something untrue about the estimator. Compare
    `InterpolatingSpline`, which places knots adaptively and reaches 0.96 at the
    same corner -- the difference between the two is the point.

    Args:
        reps: Replicates for this tier.
    """
    kink = 60
    truth = np.where(_T < kink, 0.02 * _T, 0.02 * kink + 0.20 * (_T - kink))
    target = 0.20

    covered = 0
    for i in range(reps):
        rng = np.random.default_rng(8000 + i)
        fitted = build("l1_filter").fit(
            AXIS,
            truth + NoiseGenerator.white(N, SIGMA, rng),
            order=1,
            se=True,
            n_bootstrap=N_BOOTSTRAP,
            random_state=8000 + i,
        )
        covered += (
            float(fitted.ci_lower[kink]) <= target <= float(fitted.ci_upper[kink])
        )

    rate = covered / reps
    assert 0.5 <= rate <= 0.85, (
        f"coverage at the slope change is {rate:.3f}; it was 0.67 when this was "
        "written. Above the range means the penalty or the interval changed for "
        "the better and the note above needs rewriting; below means it got worse."
    )
