"""Does the estimator get better with more data, and does the knob work?

Two families of property that a single sample size cannot reveal.

**Consistency.** As the sample grows and the bandwidth shrinks appropriately,
bias and variance both go to zero. An estimator can be perfectly calibrated at
one n and still be wrong -- an error in how the bandwidth translates to the time
axis, for instance, shows up as an RMSE that stops improving.

**The bias-variance tradeoff.** Widening the bandwidth must lower the variance
and raise the bias, monotonically and in that direction. This is what pins the
*sign* of the scale knob: an inverted map still produces plausible-looking
numbers at any single setting, and would sail through every other test in the
suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from incline import smoothers as sm
from incline.axis import TimeAxis
from incline.simulate import NoiseGenerator
from tests._statistics import DEEP_REPS, FAST_REPS


SIGMA = 0.4

TIERS = [
    pytest.param(FAST_REPS, id="fast"),
    pytest.param(DEEP_REPS, id="deep", marks=pytest.mark.slow),
]


def sampling_behaviour(smoother, n, reps, sigma=SIGMA, seed=3, frequency=3.0):
    """Bias, spread and mean reported error of the mid-series slope.

    The mean function is held fixed on [0, 1] while n grows, so more data means
    a denser sample of the same curve -- which is what consistency is about.
    ``frequency`` sets how much curvature that curve has, which is what the
    bias-variance tests vary.
    """
    x = np.linspace(0.0, 1.0, n)
    axis = TimeAxis._build(x, "index")
    truth = np.sin(frequency * x)
    true_slope = frequency * np.cos(frequency * x)
    point = n // 2

    rng = np.random.default_rng(seed)
    estimates = np.empty(reps)
    reported = np.empty(reps)
    for i in range(reps):
        fitted = smoother.fit(
            axis, truth + NoiseGenerator.white(n, sigma, rng), order=1, se=True
        )
        estimates[i] = fitted.derivative[point]
        reported[i] = fitted.se[point]

    bias = float(estimates.mean() - true_slope[point])
    spread = float(estimates.std())
    return {
        "bias": abs(bias),
        "sd": spread,
        "rmse": float(np.hypot(bias, spread)),
        "reported_se": float(reported.mean()),
    }


# The bandwidth must shrink with n or the bias never disappears. n^(-1/7) is the
# usual rate for estimating a first derivative with a local quadratic.
def shrinking_bandwidth(n: int, base: float = 0.35, reference: int = 50) -> float:
    """A bandwidth schedule that lets both bias and variance vanish."""
    return base * (n / reference) ** (-1 / 7)


# Sample sizes are scaled by tier: a local polynomial costs O(n) solves per
# replicate, so n=400 at full replicates is minutes of work. The decline in
# RMSE is already unmistakable over the shorter ladder.
FAST_SIZES = [40, 80, 160]
DEEP_SIZES = [50, 100, 200, 400]


def sizes_for(reps: int) -> list[int]:
    """The sample-size ladder appropriate to the tier being run."""
    return DEEP_SIZES if reps > FAST_REPS else FAST_SIZES


@pytest.mark.parametrize("reps", TIERS)
def test_rmse_declines_with_sample_size(reps, capsys):
    """More data must mean a better estimate."""
    sizes = sizes_for(reps)
    results = []
    for n in sizes:
        smoother = sm.LocalPolynomial(bandwidth=shrinking_bandwidth(n), degree=2)
        results.append(sampling_behaviour(smoother, n, reps))

    errors = [r["rmse"] for r in results]
    with capsys.disabled():
        print(
            "  consistency/local_poly rmse "
            + " ".join(f"n={n}:{e:.4f}" for n, e in zip(sizes, errors, strict=True))
        )

    assert errors[-1] < 0.85 * errors[0], (
        f"RMSE barely improved from n={sizes[0]} to n={sizes[-1]}: {errors}"
    )
    # A log-log fit is more robust to Monte Carlo wobble than pairwise ordering.
    slope = np.polyfit(np.log(sizes), np.log(errors), 1)[0]
    assert slope < -0.15, f"RMSE is not shrinking with n (log-log slope {slope:.3f})"


@pytest.mark.parametrize("reps", TIERS)
def test_both_bias_and_variance_shrink(reps):
    """Consistency needs both terms to vanish, not one to trade against the other."""
    sizes = sizes_for(reps)
    small = sampling_behaviour(
        sm.LocalPolynomial(bandwidth=shrinking_bandwidth(sizes[0]), degree=2),
        sizes[0],
        reps,
    )
    large = sampling_behaviour(
        sm.LocalPolynomial(bandwidth=shrinking_bandwidth(sizes[-1]), degree=2),
        sizes[-1],
        reps,
    )
    assert large["sd"] < small["sd"], "the variance did not fall with n"
    assert large["bias"] <= small["bias"] + 0.05, "the bias grew with n"


@pytest.mark.parametrize("reps", TIERS)
def test_standard_errors_stay_calibrated_as_n_grows(reps, capsys):
    """Calibration at one sample size does not imply calibration at another."""
    sizes = sizes_for(reps)
    ratios = []
    for n in sizes:
        smoother = sm.LocalPolynomial(bandwidth=shrinking_bandwidth(n), degree=2)
        result = sampling_behaviour(smoother, n, reps)
        ratios.append(result["reported_se"] / result["sd"])

    with capsys.disabled():
        print(
            "  consistency/se-ratio "
            + " ".join(f"n={n}:{r:.3f}" for n, r in zip(sizes, ratios, strict=True))
        )
    for n, ratio in zip(sizes, ratios, strict=True):
        assert 0.75 < ratio < 1.35, (
            f"at n={n} the reported SE is {ratio:.3f}x the spread"
        )


BANDWIDTHS = [0.08, 0.15, 0.25, 0.40]


@pytest.mark.parametrize("reps", TIERS)
def test_wider_bandwidth_lowers_variance(reps, capsys):
    """Averaging over more points must reduce the spread, monotonically.

    This pins the direction of the scale knob. An inverted map produces
    perfectly plausible numbers at any single bandwidth.
    """
    spreads = [
        sampling_behaviour(sm.LocalPolynomial(bandwidth=bw, degree=2), 120, reps)["sd"]
        for bw in BANDWIDTHS
    ]
    with capsys.disabled():
        print(
            "  tradeoff/sd "
            + " ".join(
                f"h={h}:{s:.4f}" for h, s in zip(BANDWIDTHS, spreads, strict=True)
            )
        )
    assert spreads == sorted(spreads, reverse=True), (
        f"variance is not decreasing in bandwidth: {spreads}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("reps", TIERS)
def test_wider_bandwidth_raises_bias(reps, capsys):
    """The other half of the tradeoff, measured where bias is unambiguous.

    A gentle truth is nearly quadratic, so a local quadratic reproduces it at
    any width and the bias stays at the level of Monte Carlo noise. Curvature is
    what a wide window cannot follow, so the truth here oscillates twice across
    the span.
    """
    biases = [
        sampling_behaviour(
            sm.LocalPolynomial(bandwidth=bw, degree=2), 120, reps, frequency=8.0
        )["bias"]
        for bw in BANDWIDTHS
    ]
    with capsys.disabled():
        print(
            "  tradeoff/bias "
            + " ".join(
                f"h={h}:{b:.3f}" for h, b in zip(BANDWIDTHS, biases, strict=True)
            )
        )
    assert biases == sorted(biases), f"bias is not increasing in bandwidth: {biases}"
    assert biases[-1] > 3 * biases[0]


@pytest.mark.slow
@pytest.mark.parametrize("reps", TIERS)
def test_the_best_bandwidth_tracks_the_curvature_of_the_truth(reps, capsys):
    """A wigglier truth must want a narrower window. That is the whole tradeoff.

    Stated this way rather than as "the RMSE curve is U-shaped", because where
    the optimum falls depends on the truth: for a nearly quadratic mean the best
    bandwidth is the widest on offer, and for a strongly curved one it is the
    narrowest. What must hold in general is that the optimum *moves*, and in
    the right direction.
    """
    grid = [0.05, 0.10, 0.20, 0.35, 0.60]

    def best_bandwidth(frequency):
        errors = [
            sampling_behaviour(
                sm.LocalPolynomial(bandwidth=bw, degree=2),
                200,
                reps,
                frequency=frequency,
            )["rmse"]
            for bw in grid
        ]
        return grid[int(np.argmin(errors))], errors

    smooth_best, smooth_errors = best_bandwidth(3.0)
    wiggly_best, wiggly_errors = best_bandwidth(12.0)

    with capsys.disabled():
        print(f"  tradeoff/best-h smooth={smooth_best} wiggly={wiggly_best}")
        print("    smooth rmse " + " ".join(f"{e:.3f}" for e in smooth_errors))
        print("    wiggly rmse " + " ".join(f"{e:.3f}" for e in wiggly_errors))

    assert wiggly_best < smooth_best, (
        f"a curvier truth chose a wider bandwidth ({wiggly_best}) than a "
        f"smooth one ({smooth_best})"
    )
