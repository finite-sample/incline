"""Noise models for the residual process.

A derivative estimate's sampling variance is ``diag(L Sigma L')`` where ``L`` is
the estimator's linear operator and ``Sigma`` the covariance of the noise. This
module supplies ``Sigma``.

Estimating it is the subtle part. The obvious approach -- fit the smoother, take
the residuals, read off their autocorrelation -- **does not work**. A smoother
removes the low-frequency content of the noise along with the trend, so residual
autocorrelation is badly attenuated: on AR(1) data with phi=0.7 that route
estimates phi=0.21 and the resulting intervals are half the width they should be.

Both estimators here are therefore *difference based*. They are computed from
differences of the raw series, never from the smoother's residuals, so nothing
the smoother does can bias them. Differencing annihilates the trend and leaves
the noise, which is exactly the separation required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.linalg import matmul_toeplitz


if TYPE_CHECKING:
    from .axis import TimeAxis

# Second-difference filter. Annihilates any locally linear trend, so what
# survives is noise. w[a] * w[b] products drive the theoretical autocovariance.
_SECOND_DIFF = np.array([1.0, -2.0, 1.0])

# Lags matched when identifying the AR(1) coefficient. Beyond lag 3 the
# empirical autocovariance of a second difference is mostly estimation error.
_MATCH_LAGS = 4

_PHI_GRID = np.linspace(0.0, 0.95, 192)


def _second_difference(y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Second difference of a series, with the mean removed."""
    d = y[2:] - 2 * y[1:-1] + y[:-2]
    return d - d.mean()


def _theoretical_acov(phi: float, lag: int) -> float:
    """Autocovariance of the second difference of a unit-variance AR(1)."""
    return float(
        sum(
            _SECOND_DIFF[a] * _SECOND_DIFF[b] * phi ** abs(lag + a - b)
            for a in range(3)
            for b in range(3)
        )
    )


def _empirical_acov(d: npt.NDArray[np.float64], n_lags: int) -> npt.NDArray[np.float64]:
    """Sample autocovariance of ``d`` at lags 0..n_lags-1."""
    return np.array([float(np.mean(d[k:] * d[: len(d) - k])) for k in range(n_lags)])


def rice_sigma(y: npt.NDArray[np.float64]) -> float:
    """Estimate the noise standard deviation from second differences.

    The Gasser-Sroka-Jennen-Steinmetz estimator. For a smooth mean function the
    second difference has variance ``6 * sigma**2``, so ``sigma**2`` is the
    mean squared second difference over six.

    Curvature in the mean inflates this slightly, which makes it mildly
    conservative -- the preferred direction of error for an interval width.

    Args:
        y: Observed values, assumed roughly equally spaced.

    Returns:
        Estimated noise standard deviation, or 0.0 for fewer than 3 points.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 3:
        return 0.0
    return float(np.sqrt(np.mean(_second_difference(y) ** 2) / 6.0))


def estimate_ar1(
    y: npt.NDArray[np.float64], phi: float | None = None
) -> tuple[float, float]:
    """Estimate AR(1) noise parameters without reference to any smoother.

    Matches the sample autocovariance of the series' second difference against
    the AR(1) theoretical values over a grid of ``phi``. Because the second
    difference annihilates a locally linear trend, what is being matched is
    noise structure rather than signal.

    Measured against known truth (n=200, sinusoidal trend): phi_hat of
    0.107 / 0.454 / 0.680 for true phi of 0.0 / 0.4 / 0.7.

    Args:
        y: Observed values.
        phi: Use this autocorrelation instead of identifying one, and scale
            sigma to match it. Sigma is derived by dividing the observed
            autocovariance by its theoretical value *at a particular phi*, so a
            sigma computed for one phi does not describe the process at
            another: over the grid that divisor ranges from 6 down to 0.9, an
            order of magnitude.

    Returns:
        Tuple of (phi, sigma), where sigma is the marginal standard deviation
        of the noise process.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < _MATCH_LAGS + 3:
        return (phi or 0.0), rice_sigma(y)

    d = _second_difference(y)
    empirical = _empirical_acov(d, _MATCH_LAGS)
    if empirical[0] <= 0:
        return (phi or 0.0), 0.0

    if phi is None:
        theoretical = np.array(
            [[_theoretical_acov(p, k) for k in range(_MATCH_LAGS)] for p in _PHI_GRID]
        )
        # Match the *shape* of the autocovariance; the level fixes sigma below.
        normalized = theoretical / theoretical[:, [0]]
        target = empirical / empirical[0]
        phi = float(
            _PHI_GRID[np.argmin(((normalized[:, 1:] - target[1:]) ** 2).sum(1))]
        )

    variance = empirical[0] / _theoretical_acov(phi, 0)
    return phi, float(np.sqrt(max(variance, 0.0)))


def local_sigma(
    y: npt.NDArray[np.float64], window: int = 25
) -> npt.NDArray[np.float64]:
    """Estimate a noise level that varies across the series.

    Applies the same second-difference logic as :func:`rice_sigma` inside a
    rolling window, so the estimate stays free of the smoother and free of the
    trend while being allowed to change.

    Args:
        y: Observed values.
        window: Number of second differences averaged at each point. Wider is
            steadier but slower to follow a change in scale.

    Returns:
        Estimated noise standard deviation at each point.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return np.full(n, rice_sigma(y))

    squared = np.empty(n)
    d2 = (y[2:] - 2 * y[1:-1] + y[:-2]) ** 2 / 6.0
    # Each second difference is centered on its middle point; the ends reuse the
    # nearest interior value rather than inventing one.
    squared[1:-1] = d2
    squared[0] = d2[0]
    squared[-1] = d2[-1]

    half = max(window // 2, 1)
    padded = np.pad(squared, half, mode="edge")
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    smoothed = np.convolve(padded, kernel, mode="valid")[:n]
    return np.sqrt(np.maximum(smoothed, 0.0))


@dataclass(frozen=True)
class NoiseFit:
    """A noise process fitted to a series.

    Attributes:
        sigma: Marginal standard deviation of the noise.
        phi: AR(1) coefficient. Zero means independent.
        explicit: A caller-supplied covariance matrix, used verbatim when set.
        sigma_vector: Per-point standard deviations, when the scale varies.
        scale_is_stated: Whether the scale came from the caller or from a
            plain scalar estimate. Decides whether the bootstrap adopts it.
    """

    sigma: float
    phi: float = 0.0
    explicit: npt.NDArray[np.float64] | None = None
    sigma_vector: npt.NDArray[np.float64] | None = None
    scale_is_stated: bool = False

    def bootstrap_scale(self, n: int) -> npt.NDArray[np.float64] | None:
        """The scale a residual bootstrap should resample at, if this fixes one.

        Returns None when the scale is a plain scalar this module estimated,
        leaving the bootstrap to use its own difference-based estimate -- which
        is the one its rescaling was calibrated against.

        The distinction is not fussiness. Feeding :func:`estimate_ar1`'s sigma
        to the bootstrap looks principled and measures badly: that estimator
        exists to pair with a full AR(1) covariance in ``propagate``, and used
        alone it ran about 1.4x high, which on top of an already over-dispersed
        block bootstrap gave standard errors 2.8x the estimator's real spread.

        Args:
            n: Number of observations.

        Returns:
            Standard deviation at each point, or None to let the bootstrap
            estimate its own.
        """
        if self.explicit is not None:
            return np.sqrt(np.maximum(np.diag(self.explicit), 0.0))
        if self.sigma_vector is not None:
            return np.asarray(self.sigma_vector, dtype=np.float64)
        if self.scale_is_stated:
            return np.full(n, self.sigma, dtype=np.float64)
        return None

    def covariance(self, n: int) -> npt.NDArray[np.float64]:
        """Materialise the n x n noise covariance.

        Args:
            n: Number of observations.

        Returns:
            The covariance matrix.
        """
        if self.explicit is not None:
            return self.explicit
        if self.sigma_vector is not None:
            return np.diag(self.sigma_vector**2)
        lags = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
        return self.sigma**2 * self.phi**lags

    def propagate(self, operator: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Push the noise through a linear operator.

        Computes ``diag(L Sigma L')``, the sampling variance of ``L y``.

        Args:
            operator: The estimator's linear operator ``L``, shape (n, n).

        Returns:
            Variance at each point, shape (n,).
        """
        n = operator.shape[0]

        if self.explicit is not None:
            return np.einsum(
                "ij,jk,ik->i", operator, self.explicit, operator, optimize=True
            )

        if self.sigma_vector is not None:
            # Diagonal but not constant: each column carries its own variance.
            return np.sum(operator**2 * self.sigma_vector**2, axis=1)

        if self.phi == 0.0:
            # Sigma is diagonal, so no matrix product is needed.
            return self.sigma**2 * np.sum(operator**2, axis=1)

        # Sigma is symmetric Toeplitz; exploit that rather than forming it.
        band = self.sigma**2 * self.phi ** np.arange(n)
        sigma_lt = np.asarray(
            matmul_toeplitz((band, band), operator.T), dtype=np.float64
        )
        return np.sum(operator * sigma_lt.T, axis=1)


class NoiseModel(ABC):
    """Base class for noise specifications."""

    @abstractmethod
    def estimate(self, y: npt.NDArray[np.float64], axis: TimeAxis) -> NoiseFit:
        """Fit the noise process to a series.

        Args:
            y: Observed values.
            axis: The series' time axis.

        Returns:
            The fitted noise process.
        """


@dataclass(frozen=True)
class IID(NoiseModel):
    """Independent noise with constant variance.

    The default. Sharpest intervals when it holds; on AR(1) data with phi=0.7
    it reports standard errors 71% of their true size, so reach for
    :class:`AR1` when dependence is plausible.

    Attributes:
        sigma: Fixed noise level. Estimated from the data when None.
    """

    sigma: float | None = None

    def estimate(self, y: npt.NDArray[np.float64], axis: TimeAxis) -> NoiseFit:
        """Estimate the noise level, or use the supplied one."""
        del axis
        stated = self.sigma is not None
        return NoiseFit(
            sigma=self.sigma if stated else rice_sigma(y),
            scale_is_stated=stated,
        )


@dataclass(frozen=True)
class AR1(NoiseModel):
    """First-order autoregressive noise.

    Attributes:
        phi: Fixed autocorrelation. Estimated from the data when None.
        sigma: Fixed marginal noise level. Estimated when None.
    """

    phi: float | None = None
    sigma: float | None = None

    def estimate(self, y: npt.NDArray[np.float64], axis: TimeAxis) -> NoiseFit:
        """Estimate phi and sigma, or use the supplied values."""
        del axis
        # Pass the caller's phi in, so sigma is rescaled to match it rather
        # than left at a value calibrated for a different autocorrelation.
        phi_hat, sigma_hat = estimate_ar1(y, self.phi)
        return NoiseFit(
            sigma=self.sigma if self.sigma is not None else sigma_hat,
            phi=phi_hat,
            scale_is_stated=self.sigma is not None,
        )


@dataclass(frozen=True)
class Heteroskedastic(NoiseModel):
    """Independent noise whose scale changes across the series.

    Assuming one constant noise level when the scale varies gets the *average*
    right and everything else wrong, in opposite directions at the two ends.
    Measured on a series whose noise standard deviation rises from 0.2 to 1.0,
    with a nominal 95% interval: coverage 1.000 and a standard error 2.0x too
    large in the quiet stretch, coverage 0.828 and a standard error 0.70x too
    small in the noisy one. The midpoint looks perfect, which is what makes it
    easy to miss.

    Attributes:
        sigma: Per-point standard deviations. Estimated locally when None.
        window: Points averaged by the local estimator.
    """

    sigma: npt.NDArray[np.float64] | None = None
    window: int = 25

    def estimate(self, y: npt.NDArray[np.float64], axis: TimeAxis) -> NoiseFit:
        """Estimate the local noise level, or use the supplied one."""
        del axis
        if self.sigma is not None:
            scale = np.asarray(self.sigma, dtype=float)
            if scale.shape != (len(y),):
                raise ValueError(
                    f"sigma must have one value per observation "
                    f"({len(y)}), got {scale.shape}"
                )
        else:
            scale = local_sigma(y, self.window)
        return NoiseFit(
            sigma=float(np.mean(scale)), sigma_vector=scale, scale_is_stated=True
        )


@dataclass(frozen=True)
class Given(NoiseModel):
    """A caller-supplied noise covariance, used exactly as provided.

    Attributes:
        covariance: The n x n noise covariance matrix.
    """

    covariance: npt.NDArray[np.float64]

    def estimate(self, y: npt.NDArray[np.float64], axis: TimeAxis) -> NoiseFit:
        """Return the supplied covariance, checking it matches the series."""
        del axis
        cov = np.asarray(self.covariance, dtype=float)
        if cov.shape != (len(y), len(y)):
            raise ValueError(f"covariance must be {(len(y), len(y))}, got {cov.shape}")
        return NoiseFit(sigma=float(np.sqrt(np.mean(np.diag(cov)))), explicit=cov)


def resolve_noise(spec: NoiseModel | str | None) -> NoiseModel:
    """Turn a user-facing noise argument into a NoiseModel.

    Args:
        spec: A NoiseModel, the string ``'iid'`` or ``'ar1'``, or None for the
            default.

    Returns:
        The corresponding NoiseModel.

    Raises:
        ValueError: If the string is not a recognized noise model.
    """
    if spec is None:
        return IID()
    if isinstance(spec, NoiseModel):
        return spec
    match spec:
        case "iid":
            return IID()
        case "ar1":
            return AR1()
        case "heteroskedastic":
            return Heteroskedastic()
        case _:
            raise ValueError(
                f"Unknown noise model {spec!r}; use 'iid', 'ar1' or 'heteroskedastic'"
            )
