"""Monte Carlo machinery for the econometric tests.

Two decisions here shape every test that uses this module.

**Coverage is measured at fixed points, not pooled across the curve.** Whether
the interval covers at x=40 and whether it covers at x=41 are almost the same
event -- they come from one smoothed curve fitted to one dataset. Pooling them
would multiply the apparent sample size by the number of points while adding
almost no information, and would make a badly miscalibrated estimator look
precisely measured. One point per binomial keeps the arithmetic honest.

**Gates are binomial, not hand-picked.** An assertion like ``coverage > 0.85``
is arbitrary: too loose at ten thousand replicates, too tight at fifty. The
tolerance here is derived from the replicate count, so the same assertion
adapts -- it loosens automatically in the fast tier and tightens in the deep
one, and it states in its failure message how far outside the band the result
fell.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from incline.simulate import NoiseGenerator


if TYPE_CHECKING:
    from incline.axis import TimeAxis
    from incline.smoothers import Smoother

# Replicate counts for the two tiers. The deep count can be raised in CI
# without touching the tests: the binomial gate tightens on its own.
FAST_REPS = 100
DEEP_REPS = int(os.environ.get("INCLINE_MC_REPS", "400"))

# How many binomial standard errors a result may sit from nominal before it
# counts as miscalibrated. Three is loose enough that a correct estimator
# essentially never trips it, and tight enough to catch a 20% error in a
# standard error at the deep replicate count.
GATE_SIGMAS = 3.0


@dataclass(frozen=True)
class MonteCarloResult:
    """Sampling behavior of an estimator at one point.

    Attributes:
        estimates: The estimate from each replicate.
        standard_errors: The reported standard error from each replicate.
        covered: Whether each replicate's interval contained the truth.
        rejected: Whether each replicate flagged a significant trend.
        truth: The true derivative at the point.
    """

    estimates: npt.NDArray[np.float64]
    standard_errors: npt.NDArray[np.float64]
    covered: npt.NDArray[np.bool_]
    rejected: npt.NDArray[np.bool_]
    truth: float

    @property
    def reps(self) -> int:
        """Number of replicates."""
        return len(self.estimates)

    @property
    def bias(self) -> float:
        """Mean deviation of the estimate from the truth."""
        return float(np.mean(self.estimates) - self.truth)

    @property
    def sampling_sd(self) -> float:
        """The estimator's actual spread across replicates."""
        return float(np.std(self.estimates, ddof=1))

    @property
    def reported_se(self) -> float:
        """The standard error the estimator claims, on average."""
        return float(np.mean(self.standard_errors))

    @property
    def se_ratio(self) -> float:
        """Claimed standard error over actual spread. One means calibrated."""
        return self.reported_se / self.sampling_sd if self.sampling_sd else np.nan

    @property
    def bias_t(self) -> float:
        """Bias in units of its own standard error.

        The Monte Carlo estimate of the mean has standard error
        ``sampling_sd / sqrt(reps)``, so this is a t statistic for the null
        that the estimator is unbiased.
        """
        if not self.sampling_sd:
            return 0.0
        return self.bias / (self.sampling_sd / np.sqrt(self.reps))

    @property
    def coverage(self) -> float:
        """Fraction of replicates whose interval contained the truth."""
        return float(np.mean(self.covered))

    @property
    def rejection_rate(self) -> float:
        """Fraction of replicates that flagged a significant trend."""
        return float(np.mean(self.rejected))

    def summary(self, label: str) -> str:
        """A one-line report, for printing a table of results."""
        return (
            f"{label:34s} bias={self.bias:+9.5f} t={self.bias_t:+6.2f} "
            f"SE/MC={self.se_ratio:5.3f} cover={self.coverage:.3f} "
            f"reject={self.rejection_rate:.3f}"
        )


def binomial_band(
    nominal: float, reps: int, sigmas: float = GATE_SIGMAS
) -> tuple[float, float]:
    """The interval a well-calibrated rate should land in.

    Args:
        nominal: The rate the estimator claims, e.g. 0.95 for coverage.
        reps: Number of replicates.
        sigmas: How many binomial standard errors of slack to allow.

    Returns:
        Lower and upper bounds, clipped to [0, 1].
    """
    spread = sigmas * np.sqrt(nominal * (1 - nominal) / reps)
    return max(0.0, nominal - spread), min(1.0, nominal + spread)


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
    """Refit an estimator over many noise draws and record what it does.

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
        The recorded sampling behavior.
    """
    rng = np.random.default_rng(seed)
    n = axis.n
    target = float(true_derivative[point])

    estimates = np.empty(reps)
    errors = np.empty(reps)
    covered = np.zeros(reps, dtype=bool)
    rejected = np.zeros(reps, dtype=bool)

    for i in range(reps):
        noise = (
            NoiseGenerator.white(n, sigma, rng)
            if phi == 0.0
            else NoiseGenerator.ar1(n, phi, sigma, rng)
        )
        fitted = smoother.fit(axis, truth + noise, order=1, se=True, **fit_kwargs)
        estimates[i] = fitted.derivative[point]
        errors[i] = fitted.se[point] if fitted.se is not None else np.nan
        if fitted.ci_lower is not None and fitted.ci_upper is not None:
            covered[i] = fitted.ci_lower[point] <= target <= fitted.ci_upper[point]
        rejected[i] = bool(fitted.significant[point])

    return MonteCarloResult(estimates, errors, covered, rejected, target)


def assert_unbiased(result: MonteCarloResult, label: str = "") -> None:
    """Fail if the estimator's mean is distinguishable from the truth.

    Args:
        result: A completed Monte Carlo study.
        label: Included in the failure message.

    Raises:
        AssertionError: If the bias t statistic exceeds the gate.
    """
    assert abs(result.bias_t) < GATE_SIGMAS, (
        f"{label}: bias {result.bias:+.6f} is {result.bias_t:+.2f} standard "
        f"errors from zero over {result.reps} replicates "
        f"(sampling sd {result.sampling_sd:.5f})"
    )


def assert_rate(
    successes: int | float,
    reps: int,
    nominal: float,
    label: str = "",
    sigmas: float = GATE_SIGMAS,
) -> None:
    """Fail if an observed rate is inconsistent with the claimed one.

    Args:
        successes: Number of successes, or the rate itself.
        reps: Number of replicates.
        nominal: The claimed rate.
        label: Included in the failure message.
        sigmas: Slack, in binomial standard errors.

    Raises:
        AssertionError: If the observed rate falls outside the band.
    """
    observed = successes / reps if successes > 1 else float(successes)
    low, high = binomial_band(nominal, reps, sigmas)
    assert low <= observed <= high, (
        f"{label}: observed rate {observed:.3f} outside the {sigmas:g}-sigma "
        f"band [{low:.3f}, {high:.3f}] for a nominal {nominal:.3f} "
        f"over {reps} replicates"
    )


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


def assert_coverage(
    result: MonteCarloResult, nominal: float = 0.95, label: str = ""
) -> None:
    """Fail if interval coverage is inconsistent with the nominal level.

    Args:
        result: A completed Monte Carlo study.
        nominal: The claimed confidence level.
        label: Included in the failure message.

    Raises:
        AssertionError: If coverage falls outside the binomial band.
    """
    assert_rate(result.coverage, result.reps, nominal, f"{label} coverage")
