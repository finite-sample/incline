"""Do the Bayesian estimators' credible intervals mean what they say?

Gaussian processes and state-space models are shrinkage estimators. They are
biased by construction -- pulling the fit toward the prior is the entire point --
so the unbiasedness and frequentist-coverage gates in ``test_econometrics.py``
do not apply and would fail for reasons that are not defects.

The guarantee that *does* apply is Bayesian: **when the data really do come from
the model's prior, a 95% credible interval must contain the truth 95% of the
time.** That is exactly checkable by simulation, and it is a strong test,
because it exercises the posterior mean and the posterior variance jointly. A
sign error in the derivative kernel, a missing covariance term, or a variance
off by a constant all break it.

This is what caught the old Gaussian process implementation, whose reported
standard error was 3,614x too large, and it is what would have caught the
state-space hyperparameter inflation that could be 1e8 too large.
"""

from __future__ import annotations

import numpy as np
import pytest

from incline.axis import TimeAxis
from incline.process import GaussianProcess, StateSpace
from tests._statistics import (
    DEEP_REPS,
    FAST_REPS,
    assert_count_rate,
    draw_from_gp_prior,
    draw_from_local_linear_trend,
)

TIERS = [
    pytest.param(FAST_REPS, id="fast"),
    pytest.param(DEEP_REPS, id="deep", marks=pytest.mark.slow),
]

POINT = 40


@pytest.mark.parametrize("reps", TIERS)
def test_gaussian_process_credible_intervals_cover_under_its_prior(reps, capsys):
    """Draw f and f' jointly from the GP prior; the interval must cover at 95%.

    Every hyperparameter is stated exactly: ``optimize=False`` stops them being
    re-fitted and ``standardize=False`` keeps them in the data's own units, so
    the prior being conditioned on is the prior the data came from. That makes
    this a real Bayesian calibration check rather than an approximate one, and
    it exercises the posterior mean and variance jointly.

    This covered 0.90 before the response scaling was made optional, because a
    supplied ``noise_level`` then meant something other than what was asked for.
    """
    n, amplitude, length_scale, noise_sd = 80, 1.0, 8.0, 0.3
    axis = TimeAxis.positional(n)
    rng = np.random.default_rng(0)

    hits = 0
    widths = []
    for _ in range(reps):
        values, derivative = draw_from_gp_prior(axis.x, amplitude, length_scale, rng)
        observed = values + rng.normal(0, noise_sd, n)

        estimate = GaussianProcess(
            kernel="rbf",
            amplitude=amplitude,
            length_scale=length_scale,
            noise_level=noise_sd**2,
            optimize=False,
            standardize=False,
        ).fit(axis, observed, order=1, se=True)

        hits += bool(
            estimate.ci_lower[POINT] <= derivative[POINT] <= estimate.ci_upper[POINT]
        )
        widths.append(float(estimate.se[POINT]))

    with capsys.disabled():
        print(
            f"  gp/prior-calibration coverage={hits / reps:.3f} "
            f"mean_se={np.mean(widths):.4f} reps={reps}"
        )
    assert_count_rate(hits, reps, 0.95, "gp credible-interval coverage")


@pytest.mark.parametrize("reps", TIERS)
def test_state_space_credible_intervals_cover_under_its_own_process(reps, capsys):
    """Simulate the local linear trend model, then check its slope interval.

    Unlike the Gaussian process case the variances are re-estimated on every
    replicate rather than held at the truth, so this measures the realistic
    situation -- and it does not reach nominal. Measured: **coverage 0.80** for
    a nominal 0.95, with the reported standard error at 0.77 of the actual
    estimation error. The intervals are conditional on the fitted variances and
    that estimation error is not propagated, so they are about a quarter too
    narrow.

    This is a documented limitation rather than a bug, so the gate is set at
    the measured level. What it still catches is regression in either
    direction: coverage collapsing further, or an interval inflated until it
    covers everything, which is the failure that has actually occurred here.
    """
    n = 120
    axis = TimeAxis.positional(n)
    rng = np.random.default_rng(1)

    hits = 0
    errors = []
    reported = []
    for _ in range(reps):
        observed, slope = draw_from_local_linear_trend(n, 0.5, 0.05, 0.01, rng)
        estimate = StateSpace().fit(axis, observed, order=1, se=True)

        hits += bool(
            estimate.ci_lower[POINT] <= slope[POINT] <= estimate.ci_upper[POINT]
        )
        errors.append(float(estimate.derivative[POINT]) - float(slope[POINT]))
        reported.append(float(estimate.se[POINT]))

    coverage = hits / reps
    root_mean_square = float(np.sqrt(np.mean(np.square(errors))))
    ratio = float(np.mean(reported)) / root_mean_square

    with capsys.disabled():
        print(
            f"  kalman/prior-calibration coverage={coverage:.3f} "
            f"se/rmse={ratio:.3f} reps={reps}"
        )

    assert coverage > 0.70, f"credible intervals cover only {coverage:.3f}"
    # The regression that motivated this test: an inflation heuristic drove
    # this ratio to 3e7 while coverage stayed high, because a vacuous interval
    # covers everything.
    assert 0.5 < ratio < 3.0, (
        f"reported standard error is {ratio:.3g}x the actual estimation error"
    )


@pytest.mark.parametrize("reps", TIERS)
def test_a_wrong_length_scale_breaks_gp_calibration(reps):
    """The calibration gate must be able to fail.

    A length scale far *longer* than the truth makes the prior derivative
    variance too small -- it scales as amplitude over length squared -- so the
    posterior is overconfident and coverage drops. A shorter one errs the other
    way and would not test anything.
    """
    n, amplitude, length_scale, noise_sd = 80, 1.0, 8.0, 0.3
    axis = TimeAxis.positional(n)
    rng = np.random.default_rng(2)

    hits = 0
    for _ in range(reps):
        values, derivative = draw_from_gp_prior(axis.x, amplitude, length_scale, rng)
        observed = values + rng.normal(0, noise_sd, n)
        estimate = GaussianProcess(
            kernel="rbf",
            amplitude=amplitude,
            length_scale=64.0,
            noise_level=noise_sd**2,
            optimize=False,
            standardize=False,
        ).fit(axis, observed, order=1, se=True)
        hits += bool(
            estimate.ci_lower[POINT] <= derivative[POINT] <= estimate.ci_upper[POINT]
        )

    assert hits / reps < 0.90, (
        f"a badly misspecified prior still covered {hits / reps:.3f} of the time"
    )
