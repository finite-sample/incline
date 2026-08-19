"""Incline's Monte Carlo helpers, on top of simcheck.

The generic machinery -- ``MonteCarloResult``, the binomial gates, the replicate
runner -- lives in `simcheck <https://github.com/finite-sample/simcheck>`_, which
was extracted from this file. What stays here is the part that is genuinely about
incline: driving a ``Smoother`` over a ``TimeAxis``, and the two generating
processes whose true derivative is known exactly.

**Coverage is measured at fixed points, not pooled across the curve.** Whether
the interval covers at x=40 and whether it covers at x=41 are almost the same
event -- they come from one smoothed curve fitted to one dataset. Pooling them
would multiply the apparent sample size by the number of points while adding
almost no information, and would make a badly miscalibrated estimator look
precisely measured. One point per binomial keeps the arithmetic honest.

**Gates are binomial, not hand-picked.** ``coverage > 0.85`` is arbitrary: too
loose at ten thousand replicates, too tight at fifty. simcheck derives the
tolerance from the replicate count, so the same assertion adapts -- it loosens
automatically in the fast tier and tightens in the deep one, and states in its
failure message how far outside the band the result fell.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from simcheck import (
    GATE_SIGMAS,
    Estimate,
    MonteCarloResult,
    assert_count_rate,
    assert_coverage,
    assert_proportion,
    assert_se_calibrated,
    assert_unbiased,
    binomial_band,
)
from simcheck import monte_carlo as _run_study

from incline.simulate import NoiseGenerator

__all__ = [
    "DEEP_REPS",
    "FAST_REPS",
    "GATE_SIGMAS",
    "MonteCarloResult",
    "assert_count_rate",
    "assert_coverage",
    "assert_proportion",
    "assert_se_calibrated",
    "assert_unbiased",
    "binomial_band",
    "draw_from_gp_prior",
    "draw_from_local_linear_trend",
    "monte_carlo",
]


if TYPE_CHECKING:
    from incline.axis import TimeAxis
    from incline.smoothers import Smoother

# Replicate counts for the two tiers. The deep count can be raised in CI
# without touching the tests: the binomial gate tightens on its own.
FAST_REPS = 100
DEEP_REPS = int(os.environ.get("INCLINE_MC_REPS", "400"))


# The old `monte_carlo` drew every replicate from one shared generator, so
# replicate 400 depended on every draw before it. simcheck's runner spawns an
# independent generator per replicate instead, which is why the wrapper below is
# a translation layer rather than a loop.
def monte_carlo(
    smoother: Smoother,
    axis: TimeAxis,
    truth: npt.NDArray[np.float64],
    true_derivative: npt.NDArray[np.float64],
    point: int,
    reps: int,
    sigma: float = 0.5,
    phi: float = 0.0,
    seed: int = 4,
    **fit_kwargs,
) -> MonteCarloResult:
    """Refit a smoother over many noise draws and record what it does.

    Args:
        smoother: The estimator under test.
        axis: The shared time axis.
        truth: The true mean function on that axis.
        true_derivative: Its true first derivative.
        point: Index of the interior point to measure at.
        reps: Number of replicates.
        sigma: Marginal noise standard deviation.
        phi: AR(1) coefficient; zero for independent noise.
        seed: Seed for the replicate stream.
        **fit_kwargs: Forwarded to ``smoother.fit``, e.g. ``noise='ar1'``.

    Returns:
        The recorded sampling behaviour.
    """
    n = axis.n
    target = float(true_derivative[point])

    def replicate(rng: np.random.Generator) -> Estimate:
        noise = (
            NoiseGenerator.white(n, sigma, rng)
            if phi == 0.0
            else NoiseGenerator.ar1(n, phi, sigma, rng)
        )
        fitted = smoother.fit(axis, truth + noise, order=1, se=True, **fit_kwargs)
        has_interval = fitted.ci_lower is not None and fitted.ci_upper is not None
        return Estimate(
            value=float(fitted.derivative[point]),
            standard_error=(
                float(fitted.se[point]) if fitted.se is not None else float("nan")
            ),
            lower=float(fitted.ci_lower[point]) if has_interval else None,
            upper=float(fitted.ci_upper[point]) if has_interval else None,
            rejected=bool(fitted.significant[point]),
        )

    return _run_study(replicate, truth=target, reps=reps, seed=seed)


def draw_from_gp_prior(
    x: npt.NDArray[np.float64],
    amplitude: float,
    length_scale: float,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Draw a function and its derivative jointly from a squared-exponential prior.

    A Gaussian process and its derivative are jointly Gaussian, with covariance

        [[ k(x, x')      dk/dx'(x, x')  ]
         [ dk/dx(x, x')  d2k/dxdx'(x,x') ]]

    Sampling both together is what makes a Bayesian calibration test possible:
    it produces a truth whose derivative is known exactly rather than
    approximated, and which really did come from the model's prior.

    Args:
        x: Points to draw at.
        amplitude: Prior signal variance.
        length_scale: Prior length scale.
        rng: Source of randomness.

    Returns:
        Tuple of (function values, first derivative values).
    """
    separation = x[:, None] - x[None, :]
    decay = np.exp(-0.5 * (separation / length_scale) ** 2)

    value_value = amplitude * decay
    derivative_value = amplitude * decay * (-separation / length_scale**2)
    derivative_derivative = (
        amplitude / length_scale**2 * decay * (1 - (separation / length_scale) ** 2)
    )

    n = len(x)
    joint = np.block(
        [
            [value_value, derivative_value.T],
            [derivative_value, derivative_derivative],
        ]
    ) + 1e-8 * np.eye(2 * n)
    sample = np.linalg.cholesky(joint) @ rng.standard_normal(2 * n)
    return sample[:n], sample[n:]


def draw_from_local_linear_trend(
    n: int,
    observation_sd: float,
    level_sd: float,
    slope_sd: float,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Simulate the local linear trend model's own generating process.

    Args:
        n: Series length.
        observation_sd: Measurement noise.
        level_sd: Level disturbance.
        slope_sd: Slope disturbance.
        rng: Source of randomness.

    Returns:
        Tuple of (observed series, true slope at each point).
    """
    level = np.empty(n)
    slope = np.empty(n)
    level[0], slope[0] = 0.0, 0.05
    for t in range(1, n):
        slope[t] = slope[t - 1] + rng.normal(0, slope_sd)
        level[t] = level[t - 1] + slope[t - 1] + rng.normal(0, level_sd)
    return level + rng.normal(0, observation_sd, n), slope
