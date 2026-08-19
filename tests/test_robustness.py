"""What happens when the Gaussian, constant-variance assumption is wrong?

Two very different answers, which is the point of separating them.

**Non-Gaussian noise barely matters.** A smoother's derivative is a weighted sum
of many observations, so the central limit theorem does the work: the sampling
distribution is close to Gaussian whatever the noise distribution, and the
variance formula only ever needed the second moment. Heavy tails, skew and
Laplace noise all cover at close to nominal.

**Non-constant variance matters a great deal, and hides.** Assuming one noise
level when it varies gets the average right and the two ends wrong in opposite
directions. Measured at the midpoint alone it looks perfect. Every test here
therefore checks several positions.
"""

from __future__ import annotations

import numpy as np
import pytest

from incline import smoothers as sm
from incline.axis import TimeAxis
from incline.noise import Heteroskedastic, local_sigma
from tests._statistics import DEEP_REPS, FAST_REPS, assert_proportion

N = 160
AXIS = TimeAxis.positional(N)
TRUTH = 0.7 - 0.4 * AXIS.x  # linear, so in span: bias is exactly zero
TRUE_SLOPE = -0.4
SIGMA = 0.5
SMOOTHER = sm.SavitzkyGolay(window_length=21, polyorder=2)

TIERS = [
    pytest.param(FAST_REPS, id="fast"),
    pytest.param(DEEP_REPS, id="deep", marks=pytest.mark.slow),
]

# Noise families, all scaled to the same marginal standard deviation so that
# only the *shape* of the distribution differs.
SHAPES = {
    "gaussian": lambda rng: rng.normal(0, SIGMA, N),
    "student_t3": lambda rng: rng.standard_t(3, N) * SIGMA / np.sqrt(3.0),
    "student_t5": lambda rng: rng.standard_t(5, N) * SIGMA / np.sqrt(5 / 3),
    "exponential_skewed": lambda rng: (rng.exponential(1.0, N) - 1.0) * SIGMA,
    "laplace": lambda rng: rng.laplace(0, SIGMA / np.sqrt(2), N),
}


def coverage_at(points, reps, draw, noise="iid", seed=4):
    """Coverage and SE-to-spread ratio at each of several positions."""
    rng = np.random.default_rng(seed)
    estimates = {p: [] for p in points}
    reported = {p: [] for p in points}
    hits = dict.fromkeys(points, 0)

    for _ in range(reps):
        fitted = SMOOTHER.fit(AXIS, TRUTH + draw(rng), order=1, se=True, noise=noise)
        for p in points:
            estimates[p].append(fitted.derivative[p])
            reported[p].append(fitted.se[p])
            hits[p] += bool(fitted.ci_lower[p] <= TRUE_SLOPE <= fitted.ci_upper[p])

    return {
        p: {
            "coverage": hits[p] / reps,
            "ratio": float(np.mean(reported[p]) / np.std(estimates[p])),
        }
        for p in points
    }


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("reps", TIERS)
def test_coverage_survives_non_gaussian_noise(shape, reps, capsys):
    """Heavy tails and skew must not break the interval.

    Only the second moment enters the variance, and the estimator averages
    enough points for its own distribution to be near-Gaussian regardless.
    """
    result = coverage_at([80], reps, SHAPES[shape])[80]
    with capsys.disabled():
        print(
            f"  robustness/{shape:19s} coverage={result['coverage']:.3f} "
            f"SE/MC={result['ratio']:.3f}"
        )
    assert_proportion(result["coverage"], reps, 0.95, f"{shape} coverage")


@pytest.mark.parametrize("reps", TIERS)
def test_heavy_tails_do_not_bias_the_estimate(reps):
    """A symmetric heavy-tailed distribution must not shift the estimate."""
    rng = np.random.default_rng(5)
    estimates = np.array(
        [
            SMOOTHER.fit(AXIS, TRUTH + SHAPES["student_t3"](rng), order=1).derivative[
                80
            ]
            for _ in range(reps)
        ]
    )
    error = estimates.std() / np.sqrt(reps)
    assert abs(estimates.mean() - TRUE_SLOPE) < 3 * error


# Noise whose standard deviation rises fivefold across the series.
RAMP = np.linspace(0.2, 1.0, N)
POSITIONS = [25, 80, 135]


@pytest.mark.parametrize("reps", TIERS)
def test_assuming_constant_variance_fails_at_the_ends(reps, capsys):
    """The failure the midpoint hides.

    With one noise level fitted to a series whose scale varies fivefold, the
    interval is far too wide where the data are quiet and too narrow where they
    are noisy -- while looking perfectly calibrated in between.
    """
    results = coverage_at(POSITIONS, reps, lambda rng: rng.normal(0, 1, N) * RAMP)
    with capsys.disabled():
        for p in POSITIONS:
            print(
                f"  hetero/iid-assumed point={p:3d} sd={RAMP[p]:.2f} "
                f"coverage={results[p]['coverage']:.3f} "
                f"SE/MC={results[p]['ratio']:.3f}"
            )

    quiet, noisy = results[POSITIONS[0]], results[POSITIONS[-1]]
    assert quiet["ratio"] > 1.4, "the quiet end should be over-covered"
    assert noisy["ratio"] < 0.85, "the noisy end should be under-covered"
    assert noisy["coverage"] < 0.92


@pytest.mark.parametrize("reps", TIERS)
def test_modelling_the_varying_scale_repairs_it(reps, capsys):
    """noise=Heteroskedastic() should restore coverage at every position."""
    results = coverage_at(
        POSITIONS, reps, lambda rng: rng.normal(0, 1, N) * RAMP, noise=Heteroskedastic()
    )
    with capsys.disabled():
        for p in POSITIONS:
            print(
                f"  hetero/modeled  point={p:3d} sd={RAMP[p]:.2f} "
                f"coverage={results[p]['coverage']:.3f} "
                f"SE/MC={results[p]['ratio']:.3f}"
            )
    for p in POSITIONS:
        assert 0.75 < results[p]["ratio"] < 1.35, (
            f"at point {p} the reported SE is {results[p]['ratio']:.3f}x the spread"
        )
        assert_proportion(
            results[p]["coverage"], reps, 0.95, f"heteroskedastic point {p}"
        )


def test_local_sigma_tracks_a_changing_scale():
    """The local estimator must follow the ramp, not average it away."""
    rng = np.random.default_rng(6)
    estimated = local_sigma(TRUTH + rng.normal(0, 1, N) * RAMP, window=31)
    assert estimated.shape == (N,)
    # Compare ends rather than pointwise: the estimator is deliberately smooth.
    assert estimated[-30:].mean() > 2.5 * estimated[:30].mean()
    assert np.corrcoef(estimated, RAMP)[0, 1] > 0.8


def test_local_sigma_matches_the_global_one_when_variance_is_constant():
    """Allowing for variation must not cost accuracy when there is none."""
    from incline.noise import rice_sigma

    rng = np.random.default_rng(7)
    y = TRUTH + rng.normal(0, SIGMA, N)
    assert local_sigma(y).mean() == pytest.approx(rice_sigma(y), rel=0.15)


def test_supplied_scale_must_match_the_series():
    """A per-point sigma of the wrong length is a caller error worth naming."""
    with pytest.raises(ValueError, match="one value per observation"):
        Heteroskedastic(sigma=np.ones(5)).estimate(np.ones(N), AXIS)


def test_heteroskedastic_is_reachable_by_name():
    """noise='heteroskedastic' should work like the other shorthand names."""
    from incline.noise import resolve_noise

    assert isinstance(resolve_noise("heteroskedastic"), Heteroskedastic)
