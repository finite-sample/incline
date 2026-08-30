"""Strategies for quantifying a derivative estimate's uncertainty.

Which strategy applies is a property of the *smoother*, not of its name:

``operator``
    The estimate is a fixed linear map ``L`` of the data, so its sampling
    variance is exactly ``diag(L Sigma L')``. No asymptotics, no resampling.
    Recovering ``L`` costs one evaluation per observation.
``bootstrap``
    The estimate is not linear -- the smoother chooses knots, iterates robust
    weights, or solves an L1 problem -- so the operator does not exist and the
    sampling distribution has to be simulated.
``native``
    The smoother is a probability model (Gaussian process, state space) and
    already carries a posterior variance. Asking it is better than probing it.

The functions here take plain callables rather than smoother objects, so this
module sits underneath :mod:`incline.smoothers` and nothing imports backwards.

Why the residual bootstrap rescales
-----------------------------------
Residuals from a smoother are shrunk by ``(I - S)``: the fit has already
absorbed part of the noise. Resampling them directly understates the noise, and
therefore understates uncertainty. Rescaling the residuals to a difference-based
noise estimate preserves a target noise level that does not depend on the
smoother being evaluated.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .noise import rice_sigma

if TYPE_CHECKING:
    from collections.abc import Callable

    from .noise import NoiseFit

# Relative tolerance for calling an estimator linear. Probing is exact up to
# floating point, so genuine linear smoothers land near 1e-15; the nonlinear
# ones miss by factors of derivative_order 1.
LINEARITY_TOLERANCE = 1e-8

# Draws used to find a simultaneous band's critical value.
SIMULTANEOUS_DRAWS = 2000


def bootstrap_block_size(n: int) -> int:
    """Choose the standard cube-root block length for dependent resampling.

    Args:
        n: Number of observations.

    Returns:
        Block length, with at least two observations.
    """
    return max(2, round(n ** (1 / 3)))


def probe_operator(
    evaluate: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    n: int,
) -> npt.NDArray[np.float64]:
    """Recover the matrix of a linear estimator by probing it.

    Applies ``evaluate`` to each standard basis vector. If the estimator is
    linear, the results are its columns.

    Args:
        evaluate: Maps a response vector to an estimate of the same length.
        n: Number of observations.

    Returns:
        The operator, shape (n, n).
    """
    operator = np.empty((n, n), dtype=np.float64)
    basis = np.zeros(n, dtype=np.float64)
    for j in range(n):
        basis[j] = 1.0
        operator[:, j] = evaluate(basis)
        basis[j] = 0.0
    return operator


def verify_linearity(
    operator: npt.NDArray[np.float64],
    evaluate: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    y: npt.NDArray[np.float64],
    label: str,
) -> None:
    """Check that a probed operator actually reproduces the estimator.

    A smoother that selects knots, reweights robustly, or solves a penalized
    L1 problem is not linear, and its probed "operator" is meaningless. This
    guard makes it impossible for such a smoother to be handed exact standard
    errors by mistake.

    Args:
        operator: The probed operator.
        evaluate: The estimator that was probed.
        y: Observed values to test against.
        label: Smoother name, used in the error message.

    Raises:
        ValueError: If the operator does not reproduce the estimator.
    """
    direct = np.asarray(evaluate(y), dtype=float)
    predicted = operator @ y
    finite = np.isfinite(direct) & np.isfinite(predicted)
    if not np.any(finite):
        return

    # The error to judge is relative to the size of the arithmetic, not to the
    # size of the answer. A constant series has a derivative of exactly zero
    # everywhere, so scaling by the output alone divides float dust by nothing
    # and reports a gross nonlinearity. ||L||_inf * max|y| bounds the
    # cancellation that forming L @ y can suffer, which is the right yardstick.
    row_sum = float(np.max(np.sum(np.abs(operator), axis=1)))
    reference = max(
        float(np.nanmax(np.abs(direct[finite]))),
        row_sum * float(np.max(np.abs(y))),
        1e-12,
    )
    error = float(np.nanmax(np.abs(direct[finite] - predicted[finite]))) / reference
    if error > LINEARITY_TOLERANCE:
        raise ValueError(
            f"{label} is not a linear smoother (probe mismatch {error:.2e}), "
            f"so it has no exact operator variance. Use a bootstrap instead."
        )


def operator_variance(
    operator: npt.NDArray[np.float64], noise: NoiseFit
) -> npt.NDArray[np.float64]:
    """Exact sampling variance of a linear estimate.

    Args:
        operator: The estimator's linear operator.
        noise: The fitted noise process.

    Returns:
        Variance at each point.
    """
    return np.maximum(noise.propagate(operator), 0.0)


def bias_corrected_operator(
    operator: npt.NDArray[np.float64],
    pilot_smoother: npt.NDArray[np.float64],
    pilot_derivative: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compose a bias-corrected derivative operator.

    The estimate ``L y`` has expectation ``L m``, where ``m`` is the true mean
    vector -- not ``m'``. The gap is smoothing bias. Estimating ``m`` with a
    less-smoothed pilot fit ``S_p y`` gives a bias estimate
    ``L S_p y - L_p y``, and subtracting it yields

        L_bc = L - L S_p + L_p

    which is still a linear operator. The exact-variance machinery therefore
    applies unchanged, and the correction costs no new theory -- only width,
    empirically around a factor of five.

    Args:
        operator: Derivative operator at the working scale.
        pilot_smoother: Smoothing operator of the pilot fit.
        pilot_derivative: Derivative operator of the pilot fit.

    Returns:
        The bias-corrected derivative operator.
    """
    return operator - operator @ pilot_smoother + pilot_derivative


def simultaneous_critical_value(
    operator: npt.NDArray[np.float64],
    noise: NoiseFit,
    standard_errors: npt.NDArray[np.float64],
    confidence_level: float,
    random_state: int | np.random.Generator | None = None,
) -> float:
    """Critical value for a band that covers the whole curve at once.

    A pointwise 95% interval fails somewhere along a 200-point curve far more
    than 5% of the time. This simulates the joint distribution implied by
    ``L Sigma L'`` and returns the quantile of the maximum standardized
    deviation, which is the multiplier a simultaneous band needs.

    Args:
        operator: The estimator's linear operator.
        noise: The fitted noise process.
        standard_errors: Pointwise standard errors.
        confidence_level: Desired simultaneous coverage.
        random_state: Seed or Generator for the simulation.

    Returns:
        The critical multiplier.
    """
    rng = np.random.default_rng(random_state)
    n = operator.shape[0]

    covariance = operator @ noise.covariance(n) @ operator.T
    covariance = (covariance + covariance.T) / 2
    # Clamp roundoff-scale negative eigenvalues on near-singular maps.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    factor = eigenvectors * np.sqrt(eigenvalues)

    draws = factor @ rng.standard_normal((n, SIMULTANEOUS_DRAWS))
    safe = np.where(standard_errors > 0, standard_errors, np.inf)
    maxima = np.max(np.abs(draws) / safe[:, None], axis=0)
    return float(np.quantile(maxima, confidence_level))


def residual_bootstrap(
    y: npt.NDArray[np.float64],
    fitted: npt.NDArray[np.float64],
    refit: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    n_bootstrap: int = 200,
    confidence_level: float = 0.95,
    block_size: int | None = None,
    random_state: int | np.random.Generator | None = None,
    scale: npt.NDArray[np.float64] | float | None = None,
) -> tuple[
    npt.NDArray[np.float64] | None,
    npt.NDArray[np.float64] | None,
    npt.NDArray[np.float64] | None,
]:
    """Simulate the sampling distribution of a nonlinear smoother's derivative.

    Resamples residuals and refits on the **original** time axis. Resampling
    ``(time, value)`` pairs instead -- as the previous implementation did, by
    drawing blocks of rows and then reattaching the original timestamps --
    destroys the trend being estimated, so the resampling distribution is not
    centered on the estimator. On a clean linear trend of slope 0.101 that
    produced the interval [-0.298, 0.577].

    Residuals are rescaled to a difference-based noise estimate before
    resampling; see the module docstring.

    Percentile intervals are returned rather than pivotal ones: refitting a
    smoother to already-smooth fitted values shifts the bootstrap distribution
    slightly, and the pivotal reflection doubles that shift (measured coverage
    0.534 against the percentile interval's 0.975).

    Args:
        y: Observed values.
        fitted: Smoothed values from the estimator on the original data.
        refit: Maps a resampled response to a derivative estimate.
        n_bootstrap: Number of replicates.
        confidence_level: Confidence level for the percentile interval.
        block_size: When set, residuals are resampled in contiguous blocks of
            this length, preserving short-range dependence.
        random_state: Seed or Generator.
        scale: Noise standard deviation to resample at, either one value or one
            per observation. Comes from the caller's noise model; estimated
            from the series when None.

    Returns:
        Tuple of (se, ci_lower, ci_upper), or (None, None, None) when the
        series has no estimable noise.

    Raises:
        RuntimeError: If every refit fails or returns the wrong shape.
    """
    rng = np.random.default_rng(random_state)

    y = np.asarray(y, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    n = len(y)

    residuals = y - fitted
    finite = np.isfinite(residuals)
    if not np.any(finite):
        return None, None, None
    residuals = residuals[finite]
    residuals = residuals - residuals.mean()

    # The scale to resample at comes from the caller's noise model when there is
    # one. Recomputing it here regardless, as this once did, makes `noise=` a
    # no-op for every smoother that is bootstrapped rather than probed: an
    # explicit IID(standard_deviation=...), Heteroskedastic or Given had no
    # effect at all.
    target = (
        np.full(n, rice_sigma(y))
        if scale is None
        else np.broadcast_to(np.asarray(scale, dtype=float), (n,))
    )

    # A series whose second differences vanish carries no information about its
    # own noise. Resampling residuals of size 1e-16 would still return a spread
    # of 1e-16, and a point estimate of the same magnitude then "excludes zero".
    # No standard error is the honest answer; zero is not.
    magnitude = float(np.max(np.abs(y))) or 1.0
    if float(np.mean(target)) <= magnitude * 1e-12:
        warnings.warn(
            "The series has no estimable noise level, so no standard error "
            "can be bootstrapped.",
            stacklevel=2,
        )
        return None, None, None

    # Standardize the residuals so that scaling by `target` below gives each
    # point exactly the spread the noise model asks for. That is what lets a
    # varying scale be honored as well as a constant one.
    spread = float(residuals.std())
    if spread > 1e-12:
        residuals = residuals / spread

    replicates: list[npt.NDArray[np.float64]] = []
    failures = 0
    first_failure: Exception | None = None
    for _ in range(n_bootstrap):
        drawn = _draw_residuals(residuals, n, block_size, rng) * target
        try:
            estimate = np.asarray(refit(fitted + drawn), dtype=float)
        except Exception as exc:  # one bad replicate must not abort the rest
            failures += 1
            if first_failure is None:
                first_failure = exc
            continue
        if len(estimate) == n:
            replicates.append(estimate)
        else:
            failures += 1

    if not replicates:
        message = "Every bootstrap replicate failed; no interval can be computed."
        raise RuntimeError(message) from first_failure

    if failures:
        warnings.warn(
            f"{failures} of {n_bootstrap} bootstrap replicates failed.",
            stacklevel=2,
        )

    draws = np.asarray(replicates)
    alpha = 1 - confidence_level
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        se = np.nanstd(draws, axis=0)
        lower = np.nanpercentile(draws, 100 * alpha / 2, axis=0)
        upper = np.nanpercentile(draws, 100 * (1 - alpha / 2), axis=0)

    return se, lower, upper


def parametric_bootstrap(
    fitted: npt.NDArray[np.float64],
    refit: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    noise: NoiseFit,
    n_bootstrap: int = 200,
    confidence_level: float = 0.95,
    random_state: int | np.random.Generator | None = None,
) -> tuple[
    npt.NDArray[np.float64] | None,
    npt.NDArray[np.float64] | None,
    npt.NDArray[np.float64] | None,
]:
    """Refit over Gaussian draws from a fitted covariance model.

    Unlike a residual block bootstrap, this preserves the complete covariance
    supplied to the estimator. It is used when the covariance participates in
    selecting the smoother or when a caller supplies an arbitrary covariance
    that residual resampling cannot preserve. The simulated errors are
    Gaussian; the covariance alone does not specify a fuller distribution.

    Args:
        fitted: Smoothed values from the original fit.
        refit: Maps a simulated response to a derivative estimate.
        noise: Fitted covariance model to simulate from.
        n_bootstrap: Number of replicates.
        confidence_level: Confidence level for the percentile interval.
        random_state: Seed or Generator.

    Returns:
        Tuple of ``(standard_error, ci_lower, ci_upper)``.

    Raises:
        RuntimeError: If every refit fails or returns the wrong shape.
    """
    fitted = np.asarray(fitted, dtype=np.float64)
    errors = noise.gaussian_draws(len(fitted), n_bootstrap, random_state)
    replicates: list[npt.NDArray[np.float64]] = []
    failures = 0
    first_failure: Exception | None = None
    for error in errors:
        try:
            estimate = np.asarray(refit(fitted + error), dtype=np.float64)
        except Exception as exc:  # one bad replicate must not abort the rest
            failures += 1
            if first_failure is None:
                first_failure = exc
            continue
        if estimate.shape == fitted.shape:
            replicates.append(estimate)
        else:
            failures += 1

    if not replicates:
        message = (
            "Every parametric bootstrap replicate failed; no interval can be computed."
        )
        raise RuntimeError(message) from first_failure
    if failures:
        warnings.warn(
            f"{failures} of {n_bootstrap} parametric bootstrap replicates failed.",
            stacklevel=2,
        )

    draws = np.asarray(replicates)
    alpha = 1.0 - confidence_level
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        standard_error = np.nanstd(draws, axis=0)
        lower = np.nanpercentile(draws, 100 * alpha / 2, axis=0)
        upper = np.nanpercentile(draws, 100 * (1 - alpha / 2), axis=0)
    return standard_error, lower, upper


def _draw_residuals(
    residuals: npt.NDArray[np.float64],
    n: int,
    block_size: int | None,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Resample residuals, in blocks when dependence must be preserved."""
    if block_size is None or block_size <= 1:
        return rng.choice(residuals, size=n, replace=True)

    available = len(residuals)
    block_size = min(block_size, available)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, available - block_size + 1, size=n_blocks)
    return np.concatenate([residuals[s : s + block_size] for s in starts])[:n]
