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
from typing import TYPE_CHECKING, Literal

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


def _validate_sigma(name: str, value: float | None) -> None:
    """Validate a scalar noise standard deviation."""
    if value is None:
        return
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be a finite non-negative scalar")


def _validate_covariance(value: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return a finite, symmetric, positive-semidefinite covariance matrix."""
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError("covariance must be real")
    try:
        covariance = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("covariance must be a numeric matrix") from exc
    if (
        covariance.ndim != 2
        or covariance.shape[0] != covariance.shape[1]
        or covariance.shape[0] == 0
    ):
        raise ValueError("covariance must be a non-empty square matrix")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must contain only finite values")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance must be symmetric")

    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    tolerance = 100 * np.finfo(float).eps * covariance.shape[0] * scale
    if float(eigenvalues[0]) < -tolerance:
        raise ValueError("covariance must be positive semidefinite")
    return covariance


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
    second difference has variance ``6 * standard_deviation**2``, so the
    noise variance is the mean squared second difference over six.

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
            standard_deviation to match it. Sigma is derived by dividing the observed
            autocovariance by its theoretical value *at a particular phi*, so a
            standard_deviation computed for one phi does not describe the process at
            another: over the grid that divisor ranges from 6 down to 0.9, an
            order of magnitude.

    Returns:
        Tuple of (phi, standard_deviation), where standard_deviation is the
        marginal standard deviation of the noise process.
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
        # Match the autocovariance shape; its level fixes the scale below.
        normalized = theoretical / theoretical[:, [0]]
        target = empirical / empirical[0]
        phi = float(
            _PHI_GRID[np.argmin(((normalized[:, 1:] - target[1:]) ** 2).sum(1))]
        )

    variance = empirical[0] / _theoretical_acov(phi, 0)
    return phi, float(np.sqrt(max(variance, 0.0)))


def local_sigma(
    y: npt.NDArray[np.float64], window_length: int = 25
) -> npt.NDArray[np.float64]:
    """Estimate a noise level that varies across the series.

    Applies the same second-difference logic as :func:`rice_sigma` inside a
    rolling window, so the estimate stays free of the smoother and free of the
    trend while being allowed to change.

    Args:
        y: Observed values.
        window_length: Number of second differences averaged at each point. Wider is
            steadier but slower to follow a change in scale.

    Returns:
        Estimated noise standard deviation at each point.

    Raises:
        ValueError: If ``y`` is not one-dimensional or ``window_length`` is
            not an odd integer of at least three.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if (
        isinstance(window_length, (bool, np.bool_))
        or not isinstance(window_length, (int, np.integer))
        or window_length < 3
        or window_length % 2 == 0
    ):
        raise ValueError("window_length must be an odd integer of at least 3")
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

    half = max(window_length // 2, 1)
    padded = np.pad(squared, half, mode="edge")
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    smoothed = np.convolve(padded, kernel, mode="valid")[:n]
    return np.sqrt(np.maximum(smoothed, 0.0))


@dataclass(frozen=True)
class NoiseFit:
    """A noise process fitted to a series.

    Attributes:
        standard_deviation: Marginal standard deviation of the noise.
        phi: AR(1) coefficient. Zero means independent.
        explicit: A caller-supplied covariance matrix, used verbatim when set.
        standard_deviation_vector: Per-point standard deviations, when the scale varies.
        scale_is_stated: Whether the scale came from the caller or from a
            plain scalar estimate. Decides whether the bootstrap adopts it.
        structure: Covariance structure used to produce the fit.
    """

    standard_deviation: float
    phi: float = 0.0
    explicit: npt.NDArray[np.float64] | None = None
    standard_deviation_vector: npt.NDArray[np.float64] | None = None
    scale_is_stated: bool = False
    structure: Literal["iid", "ar1", "heteroskedastic", "given"] = "iid"

    def __post_init__(self) -> None:
        """Validate the fitted covariance representation."""
        _validate_sigma("standard_deviation", self.standard_deviation)
        if (
            isinstance(self.phi, (bool, np.bool_))
            or not isinstance(self.phi, (int, float, np.integer, np.floating))
            or not np.isfinite(self.phi)
            or not -1.0 < self.phi < 1.0
        ):
            raise ValueError("phi must be finite and strictly between -1 and 1")
        if self.standard_deviation_vector is not None:
            vector = np.asarray(self.standard_deviation_vector, dtype=float)
            if (
                vector.ndim != 1
                or not np.all(np.isfinite(vector))
                or np.any(vector < 0)
            ):
                raise ValueError(
                    "standard_deviation_vector must be finite, nonnegative and "
                    "one-dimensional"
                )
            object.__setattr__(self, "standard_deviation_vector", vector)
        if self.explicit is not None:
            object.__setattr__(self, "explicit", _validate_covariance(self.explicit))
        if not isinstance(self.scale_is_stated, (bool, np.bool_)):
            raise ValueError("scale_is_stated must be boolean")
        if self.structure not in {"iid", "ar1", "heteroskedastic", "given"}:
            raise ValueError("structure must name a supported covariance structure")

    def bootstrap_scale(self, n: int) -> npt.NDArray[np.float64] | None:
        """The scale a residual bootstrap should resample at, if this fixes one.

        A caller-supplied scale is preserved. A scalar estimated for a complete
        covariance model is not reused as an independent residual scale; the
        residual bootstrap estimates its compatible target directly.

        Args:
            n: Number of observations.

        Returns:
            Standard deviation at each point, or None to let the bootstrap
            estimate its own.
        """
        if self.explicit is not None:
            return np.sqrt(np.maximum(np.diag(self.explicit), 0.0))
        if self.standard_deviation_vector is not None:
            return np.asarray(self.standard_deviation_vector, dtype=np.float64)
        if self.scale_is_stated:
            return np.full(n, self.standard_deviation, dtype=np.float64)
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
        if self.standard_deviation_vector is not None:
            return np.diag(self.standard_deviation_vector**2)
        lags = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
        return self.standard_deviation**2 * self.phi**lags

    def gaussian_draws(
        self,
        n: int,
        n_draws: int,
        random_state: int | np.random.Generator | None = None,
    ) -> npt.NDArray[np.float64]:
        """Draw Gaussian noise with this fitted covariance.

        Args:
            n: Number of observations per draw.
            n_draws: Number of independent draws.
            random_state: Seed or Generator.

        Returns:
            Array with shape ``(n_draws, n)``.

        Raises:
            ValueError: If a count is invalid or does not match the fitted
                covariance dimension.
        """
        for name, value in (("n", n), ("n_draws", n_draws)):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if self.explicit is not None and self.explicit.shape != (n, n):
            raise ValueError(
                f"explicit covariance must be {(n, n)}, got {self.explicit.shape}"
            )
        if (
            self.standard_deviation_vector is not None
            and self.standard_deviation_vector.shape != (n,)
        ):
            raise ValueError(
                "standard_deviation_vector must have one value per observation "
                f"({n}), got {self.standard_deviation_vector.shape}"
            )
        rng = np.random.default_rng(random_state)
        if self.explicit is not None:
            eigenvalues, eigenvectors = np.linalg.eigh(self.explicit)
            factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))
            return rng.standard_normal((n_draws, n)) @ factor.T
        if self.standard_deviation_vector is not None:
            return rng.standard_normal((n_draws, n)) * self.standard_deviation_vector
        if self.phi == 0.0:
            return rng.normal(
                scale=self.standard_deviation,
                size=(n_draws, n),
            )

        draws = np.empty((n_draws, n), dtype=np.float64)
        draws[:, 0] = rng.normal(scale=self.standard_deviation, size=n_draws)
        innovation_scale = self.standard_deviation * np.sqrt(1.0 - self.phi**2)
        for index in range(1, n):
            draws[:, index] = self.phi * draws[:, index - 1] + rng.normal(
                scale=innovation_scale,
                size=n_draws,
            )
        return draws

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

        if self.standard_deviation_vector is not None:
            # Diagonal but not constant: each column carries its own variance.
            return np.sum(operator**2 * self.standard_deviation_vector**2, axis=1)

        if self.phi == 0.0:
            # Sigma is diagonal, so no matrix product is needed.
            return self.standard_deviation**2 * np.sum(operator**2, axis=1)

        # Sigma is symmetric Toeplitz; exploit that rather than forming it.
        band = self.standard_deviation**2 * self.phi ** np.arange(n)
        sigma_lt = np.asarray(
            matmul_toeplitz((band, band), operator.T), dtype=np.float64
        )
        return np.sum(operator * sigma_lt.T, axis=1)


def describe_noise_fit(noise: NoiseFit) -> str:
    """Format a fitted noise model for result provenance.

    Args:
        noise: Fitted noise process.

    Returns:
        Stable user-facing description of the covariance used.
    """
    if noise.structure == "ar1":
        return (
            f"ar1(phi={noise.phi:.3f}, "
            f"standard_deviation={noise.standard_deviation:.4g})"
        )
    if noise.structure == "heteroskedastic":
        return "heteroskedastic"
    if noise.structure == "given":
        return "given_covariance"
    return f"iid(standard_deviation={noise.standard_deviation:.4g})"


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
        standard_deviation: Fixed noise level. Estimated from the data when None.
    """

    standard_deviation: float | None = None

    def __post_init__(self) -> None:
        """Validate the stated noise scale."""
        _validate_sigma("standard_deviation", self.standard_deviation)

    def estimate(self, y: npt.NDArray[np.float64], axis: TimeAxis) -> NoiseFit:
        """Estimate the noise level, or use the supplied one."""
        del axis
        return NoiseFit(
            standard_deviation=rice_sigma(y)
            if self.standard_deviation is None
            else self.standard_deviation,
            scale_is_stated=self.standard_deviation is not None,
            structure="iid",
        )


@dataclass(frozen=True)
class AR1(NoiseModel):
    """First-order autoregressive noise.

    Attributes:
        phi: Fixed autocorrelation. Estimated from the data when None.
        standard_deviation: Fixed marginal noise level. Estimated when None.
    """

    phi: float | None = None
    standard_deviation: float | None = None

    def __post_init__(self) -> None:
        """Validate the stationary AR(1) parameter domain."""
        if self.phi is not None and (
            isinstance(self.phi, (bool, np.bool_))
            or not isinstance(self.phi, (int, float, np.integer, np.floating))
            or not np.isfinite(self.phi)
            or not -1.0 < float(self.phi) < 1.0
        ):
            raise ValueError("phi must be finite and strictly between -1 and 1")
        _validate_sigma("standard_deviation", self.standard_deviation)

    def estimate(self, y: npt.NDArray[np.float64], axis: TimeAxis) -> NoiseFit:
        """Estimate phi and standard_deviation, or use the supplied values."""
        del axis
        # Pass the caller's phi in, so standard_deviation is rescaled to match it rather
        # than left at a value calibrated for a different autocorrelation.
        phi_hat, sigma_hat = estimate_ar1(y, self.phi)
        return NoiseFit(
            standard_deviation=self.standard_deviation
            if self.standard_deviation is not None
            else sigma_hat,
            phi=phi_hat,
            scale_is_stated=self.standard_deviation is not None,
            structure="ar1",
        )


@dataclass(frozen=True)
class Heteroskedastic(NoiseModel):
    """Independent noise whose scale changes across the series.

    A constant noise model cannot represent changing local scale: it tends to
    overstate uncertainty in quieter regions and understate it in noisier ones.

    Attributes:
        standard_deviation: Per-point standard deviations. Estimated locally when None.
        window_length: Points averaged by the local estimator.
    """

    standard_deviation: npt.NDArray[np.float64] | None = None
    window_length: int = 25

    def __post_init__(self) -> None:
        """Validate a stated per-observation noise scale."""
        if (
            isinstance(self.window_length, (bool, np.bool_))
            or not isinstance(self.window_length, (int, np.integer))
            or self.window_length < 3
            or self.window_length % 2 == 0
        ):
            raise ValueError("window_length must be an odd integer of at least 3")
        if self.standard_deviation is None:
            return
        scale = np.asarray(self.standard_deviation)
        if (
            scale.ndim != 1
            or np.issubdtype(scale.dtype, np.bool_)
            or not np.issubdtype(scale.dtype, np.number)
            or not np.all(np.isfinite(scale))
            or np.any(scale < 0.0)
        ):
            raise ValueError(
                "standard_deviation must be a finite non-negative one-dimensional array"
            )

    def estimate(self, y: npt.NDArray[np.float64], axis: TimeAxis) -> NoiseFit:
        """Estimate the local noise level, or use the supplied one."""
        del axis
        if self.standard_deviation is not None:
            scale = np.asarray(self.standard_deviation, dtype=float)
            if not np.all(np.isfinite(scale)) or np.any(scale < 0.0):
                raise ValueError(
                    "standard_deviation must contain finite non-negative values"
                )
            if scale.shape != (len(y),):
                raise ValueError(
                    f"standard_deviation must have one value per observation "
                    f"({len(y)}), got {scale.shape}"
                )
        else:
            scale = local_sigma(y, self.window_length)
        return NoiseFit(
            standard_deviation=float(np.mean(scale)),
            standard_deviation_vector=scale,
            scale_is_stated=True,
            structure="heteroskedastic",
        )


@dataclass(frozen=True)
class Given(NoiseModel):
    """A caller-supplied noise covariance, used exactly as provided.

    Attributes:
        covariance: The n x n noise covariance matrix.
    """

    covariance: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate the supplied covariance independently of series length."""
        _validate_covariance(self.covariance)

    def estimate(self, y: npt.NDArray[np.float64], axis: TimeAxis) -> NoiseFit:
        """Return the supplied covariance, checking it matches the series."""
        del axis
        cov = _validate_covariance(self.covariance)
        if cov.shape != (len(y), len(y)):
            raise ValueError(f"covariance must be {(len(y), len(y))}, got {cov.shape}")
        return NoiseFit(
            standard_deviation=float(np.sqrt(np.mean(np.diag(cov)))),
            explicit=cov,
            structure="given",
        )


def resolve_noise(spec: NoiseModel | str | None) -> NoiseModel:
    """Turn a user-facing noise argument into a NoiseModel.

    Args:
        spec: A NoiseModel, ``'iid'``, ``'ar1'``, ``'heteroskedastic'``, or
            None for the default IID estimator.

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
