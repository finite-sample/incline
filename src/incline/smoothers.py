"""Smoothers, and the single implementation of fitting one.

Every method in the package answers the same question -- given a noisy series,
what is the derivative of the underlying smooth curve -- and differs only in how
it smooths. :class:`Smoother` is that shared shape, and :meth:`Smoother.fit`
is written once, here, rather than eleven times across the package.

The important declaration a smoother makes is :attr:`Smoother.is_linear`. If the
derivative is a fixed linear map of the data then its sampling variance is exact
and cheap; if not, it has to be simulated. Probing settles the question rather
than intuition: a smoothing spline whose penalty is selected by generalized
cross-validation is data-dependent, and LOESS is linear only with robust
reweighting switched off -- which is not its default.

Scale
-----
Methods parameterize smoothing incompatibly: a window in points, a fraction of
the sample, a bandwidth in x units, a roughness penalty. :meth:`with_scale`
maps a single normalized knob -- the fraction of the series span the smoother
looks across -- onto each. One concept serves multi-scale analysis, parameter
selection, and the documentation's slider.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import numpy.typing as npt
from scipy.interpolate import make_interp_spline, make_smoothing_spline
from scipy.linalg import cholesky, eigh, solve_banded, solve_triangular
from scipy.optimize import lsq_linear, minimize_scalar
from scipy.signal import savgol_coeffs, savgol_filter
from scipy.sparse import csc_matrix
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.nonparametric.smoothers_lowess import lowess

from .noise import NoiseModel, resolve_noise
from .result import Provenance, TrendEstimate
from .uncertainty import (
    bias_corrected_operator,
    operator_variance,
    parametric_bootstrap,
    residual_bootstrap,
    simultaneous_critical_value,
    verify_linearity,
)

if TYPE_CHECKING:
    from typing import Self

    from .axis import TimeAxis
    from .noise import NoiseFit

# Operators depend only on (smoother, axis, derivative_order), so they survive across
# fits. A handful of entries covers a multi-scale sweep or a simulation study.
#
# The bound is in bytes rather than entries because each entry holds two n x n
# float64 matrices: at n = 2000 that is 64 MB apiece, so a 64-entry allowance
# would have reached about 4 GB before evicting anything.
OPERATOR_CACHE_BYTES = 512 * 1024 * 1024
_OPERATOR_CACHE: dict[Any, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}
_SPLINE_PENALTY_CACHE: dict[Any, npt.NDArray[np.float64]] = {}

SMOOTHERS: dict[str, type[Smoother]] = {}


def _is_integer(value: object) -> bool:
    """Whether a value is an integer but not a boolean."""
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _finite_number(value: object) -> bool:
    """Whether a value is a finite real scalar but not a boolean."""
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and bool(np.isfinite(value))
    )


def _validate_scale(scale: float) -> float:
    """Return a normalized smoothing scale after enforcing the shared domain."""
    if not _finite_number(scale) or not 0.0 < float(scale) <= 1.0:
        raise ValueError("scale must be finite and in (0, 1]")
    return float(scale)


def _noise_label(noise: NoiseFit) -> str:
    """Describe a fitted scalar noise model for result provenance."""
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


def _cache_bytes() -> int:
    """Total memory held by the operator cache."""
    return sum(a.nbytes + b.nbytes for a, b in _OPERATOR_CACHE.values())


def register[S: type[Smoother]](cls: S) -> S:
    """Add a smoother to the registry keyed by its name.

    Typed generically rather than as ``type[Smoother] -> type[Smoother]``:
    the upcast erased every subclass's synthesized dataclass ``__init__``,
    so pyright rejected each keyword constructor call in the package.

    Args:
        cls: The smoother class.

    Returns:
        The same class, so this can be used as a decorator.
    """
    SMOOTHERS[cls.name] = cls
    return cls


@dataclass(frozen=True)
class Evaluation:
    """A smoother's output before any uncertainty is attached.

    Attributes:
        values: The smoothed series.
        derivative: The derivative of the smooth, per unit of the time axis.
        params: Data-dependent settings selected while fitting.
    """

    values: npt.NDArray[np.float64]
    derivative: npt.NDArray[np.float64]
    params: dict[str, Any] | None = None


class Smoother(ABC):
    """Base class for every trend estimator.

    Subclasses implement :meth:`evaluate` and :meth:`with_scale` and declare
    whether they are linear. Everything else -- uncertainty, bias correction,
    simultaneous bands, result assembly -- is inherited.
    """

    name: ClassVar[str]
    linear: ClassVar[bool] = False
    has_native_posterior: ClassVar[bool] = False
    supported_orders: ClassVar[frozenset[int]] = frozenset({1, 2})
    # Whether the method's arithmetic assumes evenly spaced observations. A
    # convolution filter cannot be applied to an irregular axis and produce a
    # correct per-time derivative; it scales everything by one median step.
    requires_regular_grid: ClassVar[bool] = False

    @property
    def is_linear(self) -> bool:
        """Whether the derivative is a fixed linear map of the data.

        A property rather than the class attribute alone because some
        smoothers are linear only in certain configurations.
        """
        return self.linear

    @abstractmethod
    def evaluate(
        self, axis: TimeAxis, y: npt.NDArray[np.float64], derivative_order: int
    ) -> Evaluation:
        """Smooth a series and differentiate the smooth.

        Args:
            axis: The time axis.
            y: Observed values.
            derivative_order: Derivative order.

        Returns:
            The smoothed values and the derivative.
        """

    def evaluate_with_noise(
        self,
        axis: TimeAxis,
        y: npt.NDArray[np.float64],
        derivative_order: int,
        noise: NoiseFit | None,
    ) -> Evaluation:
        """Evaluate, allowing an adaptive smoother to use a fitted covariance.

        Args:
            axis: The time axis.
            y: Observed values.
            derivative_order: Derivative order.
            noise: Fitted covariance, when the caller supplied or requested one.

        Returns:
            The smoothed values and derivative.
        """
        del noise
        return self.evaluate(axis, y, derivative_order)

    @abstractmethod
    def with_scale(self, scale: float, axis: TimeAxis) -> Self:
        """Return a copy smoothing across ``scale`` of the series span.

        Args:
            scale: Fraction of the span in (0, 1].
            axis: The time axis, for converting to native units.

        Returns:
            A new smoother.
        """

    @abstractmethod
    def scale_of(self, axis: TimeAxis) -> float | None:
        """Report this smoother's current scale as a fraction of the span.

        Args:
            axis: The time axis.

        Returns:
            The scale in (0, 1], or None when the configuration has no fixed
            data-independent scale.
        """

    def params(self) -> dict[str, Any]:
        """Smoother-specific settings, recorded on the result."""
        return {}

    def analytic_operators(
        self, axis: TimeAxis, derivative_order: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
        """State the smoothing and derivative operators directly, if known.

        A linear smoother's operators are already implicit in the arithmetic it
        performs, so a smoother that fits a local system per point can emit its
        operator rows from the same solve. Doing so turns an O(n) probe into a
        constant factor. Returning None falls back to probing.

        Args:
            axis: The time axis.
            derivative_order: Derivative order.

        Returns:
            Tuple of (smoothing operator, derivative operator), or None.
        """
        del axis, derivative_order
        return None

    def operators(
        self, axis: TimeAxis, derivative_order: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The smoothing and derivative operators for this configuration.

        Cached: an operator depends on the smoother's settings, the axis and
        the derivative order, but never on the observed values. Repeated fits
        over the same grid -- a multi-scale sweep, a Monte Carlo study -- reuse
        the work.

        Args:
            axis: The time axis.
            derivative_order: Derivative order.

        Returns:
            Tuple of (smoothing operator, derivative operator).
        """
        cache_key = (self, axis.key(), derivative_order)
        cached = _OPERATOR_CACHE.get(cache_key)
        if cached is not None:
            return cached

        analytic = self.analytic_operators(axis, derivative_order)
        if analytic is not None:
            smoothing, derivative = analytic
        else:
            smoothing, derivative = self._probe_operators(axis, derivative_order)

        size = smoothing.nbytes + derivative.nbytes
        if size <= OPERATOR_CACHE_BYTES:
            while _OPERATOR_CACHE and _cache_bytes() + size > OPERATOR_CACHE_BYTES:
                _OPERATOR_CACHE.pop(next(iter(_OPERATOR_CACHE)))
            _OPERATOR_CACHE[cache_key] = (smoothing, derivative)
        return smoothing, derivative

    def _probe_operators(
        self, axis: TimeAxis, derivative_order: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Recover both operators by evaluating on each basis vector."""
        smoothing = np.empty((axis.n, axis.n), dtype=np.float64)
        derivative = np.empty((axis.n, axis.n), dtype=np.float64)
        basis = np.zeros(axis.n, dtype=np.float64)
        for j in range(axis.n):
            basis[j] = 1.0
            evaluation = self.evaluate(axis, basis, derivative_order)
            smoothing[:, j] = evaluation.values
            derivative[:, j] = evaluation.derivative
            basis[j] = 0.0
        return smoothing, derivative

    def native_posterior(
        self,
        axis: TimeAxis,
        y: npt.NDArray[np.float64],
        derivative_order: int,
        confidence_level: float,
    ) -> (
        tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64] | None,
            npt.NDArray[np.float64] | None,
        ]
        | None
    ):
        """Uncertainty from the smoother's own probability model.

        Overridden by smoothers that are probability models and therefore
        already know their posterior variance.

        Args:
            axis: The time axis.
            y: Observed values.
            derivative_order: Derivative derivative_order.
            confidence_level: Confidence level.

        Returns:
            Tuple of (se, ci_lower, ci_upper), or None if unsupported.
        """
        del axis, y, derivative_order, confidence_level
        return None

    def fit(
        self,
        axis: TimeAxis,
        y: npt.NDArray[np.float64],
        derivative_order: int = 1,
        with_uncertainty: bool = False,
        noise: NoiseModel | str | None = None,
        bias_correct: bool = False,
        simultaneous: bool = False,
        confidence_level: float = 0.95,
        pilot_scale: float | None = None,
        n_bootstrap: int = 200,
        random_state: int | np.random.Generator | None = None,
    ) -> TrendEstimate:
        """Estimate the trend and, optionally, its uncertainty.

        Args:
            axis: The time axis.
            y: Observed values.
            derivative_order: Derivative derivative_order.
            with_uncertainty: Whether to compute standard errors. Off by default because
                the exact route costs one smoother evaluation per observation.
            noise: Noise model, or ``'iid'`` / ``'ar1'``.
            bias_correct: Subtract an estimate of smoothing bias using a
                less-smoothed pilot fit. Linear smoothers only.
            simultaneous: Return a band covering the whole curve at once
                rather than pointwise intervals. Fixed linear smoothers only.
            confidence_level: Confidence level for intervals.
            pilot_scale: Scale of the pilot fit used for bias correction.
                Defaults to a third of this smoother's scale.
            n_bootstrap: Replicates, for nonlinear smoothers.
            random_state: Seed or Generator.

        Returns:
            The estimate.

        Raises:
            ValueError: If an argument is outside its domain, the derivative
                derivative_order is unsupported, bias correction is requested for a
                nonlinear smoother, or an uncertainty option is incompatible
                with a native posterior.
        """
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1:
            raise ValueError(f"y must be a one-dimensional series, got shape {y.shape}")
        if y.size == 0:
            raise ValueError("y must contain at least one observation")
        if not _is_integer(derivative_order) or derivative_order < 0:
            raise ValueError("derivative_order must be a nonnegative integer")
        for name, value in (
            ("with_uncertainty", with_uncertainty),
            ("bias_correct", bias_correct),
            ("simultaneous", simultaneous),
        ):
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
        if (
            isinstance(confidence_level, (bool, np.bool_))
            or not isinstance(confidence_level, (int, float, np.integer, np.floating))
            or not np.isfinite(confidence_level)
            or not 0.0 < float(confidence_level) < 1.0
        ):
            raise ValueError(
                "confidence_level must be finite and strictly between 0 and 1"
            )
        if (
            isinstance(n_bootstrap, (bool, np.bool_))
            or not isinstance(n_bootstrap, (int, np.integer))
            or int(n_bootstrap) < 2
        ):
            raise ValueError("n_bootstrap must be an integer of at least 2")
        if pilot_scale is not None and (
            not _finite_number(pilot_scale) or not 0.0 < pilot_scale <= 1.0
        ):
            raise ValueError("pilot_scale must be finite and in (0, 1]")
        if pilot_scale is not None and not bias_correct:
            raise ValueError("pilot_scale requires bias_correct=True")
        if simultaneous and not with_uncertainty:
            raise ValueError("simultaneous=True requires with_uncertainty=True")
        if derivative_order not in self.supported_orders:
            raise ValueError(
                f"{self.name} supports derivative orders "
                f"{sorted(self.supported_orders)}, got {derivative_order}"
            )
        if len(y) != axis.n:
            raise ValueError(f"y has {len(y)} values but the axis has {axis.n} points")
        missing = int(np.sum(~np.isfinite(y)))
        if missing:
            # Refused here, once, for the same reason TimeAxis refuses a
            # non-finite time: every route to uncertainty silently gives the
            # wrong answer on a gapped series rather than failing.
            #
            # The noise estimators drop the gap and second-difference across
            # it, which reads the resulting jump as noise -- on a trending
            # series with a 20-point gap that took the scale from 0.286 to 1.782
            # and the AR(1) scale to 8.11, so real trends are reported as
            # insignificant. The block bootstrap draws blocks spanning the
            # gap. A fractional L1 penalty cannot be normalized against a
            # series containing NaN. `local_sigma` convolves the NaN over 27
            # neighbors. And `verify_linearity`
            # compares an all-NaN `operator @ y`, finds nothing finite and
            # returns without checking, so a wrong operator passes.
            #
            # Estimating on the observed subset against its true irregular
            # axis would be defensible, but silently reporting a number that
            # is wrong by 6x is not.
            raise ValueError(
                f"{self.name} cannot estimate a derivative from a series with "
                f"{missing} missing value{'' if missing == 1 else 's'}; "
                f"interpolate or drop {'it' if missing == 1 else 'them'} first "
                f"(df['value'].interpolate(), or df.dropna())."
            )
        if self.requires_regular_grid:
            # Once per fit, not once per probe evaluation.
            axis.require_regular(self.name)

        if with_uncertainty and self.has_native_posterior:
            if noise is not None:
                raise ValueError(
                    f"{self.name} carries its own posterior; noise is unavailable"
                )
            if simultaneous:
                raise ValueError(
                    f"{self.name} has no simultaneous whole-curve posterior band"
                )
        if with_uncertainty and simultaneous and not self.is_linear:
            raise ValueError(
                f"{self.name} uses bootstrap uncertainty and does not support "
                "simultaneous whole-curve bands"
            )

        if bias_correct:
            return self._fit_bias_corrected(
                axis,
                y,
                derivative_order,
                with_uncertainty,
                noise,
                simultaneous,
                confidence_level,
                pilot_scale,
                random_state,
            )

        noise_model = None
        noise_fit = None
        if not self.has_native_posterior and (noise is not None or with_uncertainty):
            noise_model = resolve_noise(noise)
            noise_fit = noise_model.estimate(y, axis)

        evaluation = self.evaluate_with_noise(
            axis,
            y,
            derivative_order,
            noise_fit,
        )
        noise_label = _noise_label(noise_fit) if noise_fit is not None else None
        estimate = self._assemble(
            axis,
            y,
            derivative_order,
            evaluation,
            noise_label=noise_label,
        )
        if not with_uncertainty:
            return estimate

        return self._attach_uncertainty(
            estimate,
            axis,
            y,
            derivative_order,
            noise_model,
            noise_fit,
            simultaneous,
            confidence_level,
            n_bootstrap,
            random_state,
        )

    def _assemble(
        self,
        axis: TimeAxis,
        y: npt.NDArray[np.float64],
        derivative_order: int,
        evaluation: Evaluation,
        bias_corrected: bool = False,
        noise_label: str | None = None,
    ) -> TrendEstimate:
        """Wrap an evaluation in a TrendEstimate."""
        del y
        return TrendEstimate(
            axis=axis,
            values=evaluation.values,
            derivative=evaluation.derivative,
            derivative_order=derivative_order,
            provenance=Provenance(
                method=self.name,
                noise=noise_label,
                bias_corrected=bias_corrected,
                params={**self.params(), **(evaluation.params or {})},
            ),
        )

    def _attach_uncertainty(
        self,
        estimate: TrendEstimate,
        axis: TimeAxis,
        y: npt.NDArray[np.float64],
        derivative_order: int,
        noise_model: NoiseModel | None,
        noise_fit: NoiseFit | None,
        simultaneous: bool,
        confidence_level: float,
        n_bootstrap: int,
        random_state: int | np.random.Generator | None,
    ) -> TrendEstimate:
        """Compute uncertainty by whichever strategy this smoother supports."""
        native = self.native_posterior(axis, y, derivative_order, confidence_level)
        if native is not None:
            standard_errors, lower, upper = native
            return estimate.with_uncertainty(
                standard_error=standard_errors,
                uncertainty_method="native",
                ci_lower=lower,
                ci_upper=upper,
                confidence_level=confidence_level,
            )

        if noise_model is None or noise_fit is None:
            raise RuntimeError("uncertainty requires a fitted noise model")
        noise_label = _noise_label(noise_fit)

        if self.is_linear:
            _, operator = self.operators(axis, derivative_order)
            verify_linearity(
                operator,
                lambda v: self.evaluate(axis, v, derivative_order).derivative,
                y,
                self.name,
            )
            standard_errors = np.sqrt(operator_variance(operator, noise_fit))
            multiplier = None
            if simultaneous:
                multiplier = simultaneous_critical_value(
                    operator,
                    noise_fit,
                    standard_errors,
                    confidence_level,
                    random_state,
                )
            return self._interval_from_se(
                estimate,
                standard_errors,
                "operator",
                confidence_level,
                noise_label,
                multiplier,
                simultaneous,
            )

        standard_errors, lower, upper = self.bootstrap_uncertainty(
            estimate=estimate,
            axis=axis,
            y=y,
            derivative_order=derivative_order,
            noise_model=noise_model,
            noise_fit=noise_fit,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        if standard_errors is None:
            return estimate
        return estimate.with_uncertainty(
            standard_error=standard_errors,
            uncertainty_method="bootstrap",
            ci_lower=lower,
            ci_upper=upper,
            confidence_level=confidence_level,
            noise=noise_label,
        )

    def bootstrap_uncertainty(
        self,
        estimate: TrendEstimate,
        axis: TimeAxis,
        y: npt.NDArray[np.float64],
        derivative_order: int,
        noise_model: NoiseModel,
        noise_fit: NoiseFit,
        confidence_level: float,
        n_bootstrap: int,
        random_state: int | np.random.Generator | None,
    ) -> tuple[
        npt.NDArray[np.float64] | None,
        npt.NDArray[np.float64] | None,
        npt.NDArray[np.float64] | None,
    ]:
        """Bootstrap a nonlinear smoother, preserving short-range dependence.

        Args:
            estimate: Original fitted result.
            axis: Time axis.
            y: Observed values.
            derivative_order: Derivative order.
            noise_model: Noise model specification.
            noise_fit: Fitted noise covariance.
            confidence_level: Confidence level.
            n_bootstrap: Number of replicates.
            random_state: Seed or Generator.

        Returns:
            Standard errors and percentile interval bounds.
        """
        del noise_model
        block = _block_size(axis.n) if noise_fit.phi else None
        return residual_bootstrap(
            y=y,
            fitted=estimate.values,
            refit=lambda v: self.evaluate(axis, v, derivative_order).derivative,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            block_size=block,
            random_state=random_state,
            scale=noise_fit.bootstrap_scale(axis.n),
        )

    @staticmethod
    def _interval_from_se(
        estimate: TrendEstimate,
        standard_errors: npt.NDArray[np.float64],
        uncertainty_method: str,
        confidence_level: float,
        noise_label: str,
        multiplier: float | None,
        simultaneous: bool,
    ) -> TrendEstimate:
        """Build an interval, widening it when a simultaneous band is asked for."""
        if multiplier is None:
            return estimate.with_uncertainty(
                standard_error=standard_errors,
                uncertainty_method=uncertainty_method,
                confidence_level=confidence_level,
                noise=noise_label,
            )
        return estimate.with_uncertainty(
            standard_error=standard_errors,
            uncertainty_method=uncertainty_method,
            ci_lower=estimate.derivative - multiplier * standard_errors,
            ci_upper=estimate.derivative + multiplier * standard_errors,
            confidence_level=confidence_level,
            noise=noise_label,
            simultaneous=simultaneous,
        )

    def _fit_bias_corrected(
        self,
        axis: TimeAxis,
        y: npt.NDArray[np.float64],
        derivative_order: int,
        with_uncertainty: bool,
        noise: NoiseModel | str | None,
        simultaneous: bool,
        confidence_level: float,
        pilot_scale: float | None,
        random_state: int | np.random.Generator | None,
    ) -> TrendEstimate:
        """Estimate with the smoothing bias subtracted off.

        Only defined for linear smoothers, where the corrected estimator is
        itself a linear operator and so keeps exact variances.
        """
        if not self.is_linear:
            raise ValueError(
                f"{self.name} is not a linear smoother, so its bias cannot be "
                f"corrected by operator composition. Undersmooth instead by "
                f"passing a smaller scale."
            )

        if pilot_scale is None:
            current_scale = self.scale_of(axis)
            if current_scale is None:
                raise ValueError(
                    f"bias correction does not apply to {self.name}, which has "
                    "no smoothing scale"
                )
            pilot_scale = max(current_scale / 3.0, 3.0 / axis.n)
        pilot = self.with_scale(pilot_scale, axis)

        _, derivative_operator = self.operators(axis, derivative_order)
        pilot_smoothing, pilot_derivative = pilot.operators(axis, derivative_order)
        corrected = bias_corrected_operator(
            derivative_operator, pilot_smoothing, pilot_derivative
        )

        evaluation = self.evaluate(axis, y, derivative_order)
        estimate = TrendEstimate(
            axis=axis,
            values=evaluation.values,
            derivative=corrected @ y,
            derivative_order=derivative_order,
            provenance=Provenance(
                method=self.name, bias_corrected=True, params=self.params()
            ),
        )
        if not with_uncertainty:
            return estimate

        noise_fit = resolve_noise(noise).estimate(y, axis)
        noise_label = _noise_label(noise_fit)
        standard_errors = np.sqrt(operator_variance(corrected, noise_fit))
        multiplier = (
            simultaneous_critical_value(
                corrected,
                noise_fit,
                standard_errors,
                confidence_level,
                random_state,
            )
            if simultaneous
            else None
        )
        return self._interval_from_se(
            estimate,
            standard_errors,
            "operator",
            confidence_level,
            noise_label,
            multiplier,
            simultaneous,
        )


def _block_size(n: int) -> int:
    """Block length for a dependent bootstrap, the usual n^(1/3) rule."""
    return max(2, round(n ** (1 / 3)))


def _odd(value: int, minimum: int) -> int:
    """Nearest odd integer at or above ``minimum``."""
    value = max(int(value), minimum)
    return value if value % 2 == 1 else value + 1


# --------------------------------------------------------------------------
# Linear smoothers: exact variance available
# --------------------------------------------------------------------------


@register
@dataclass(frozen=True)
class SavitzkyGolay(Smoother):
    """Savitzky-Golay filter: local polynomial least squares on a fixed window.

    A fixed convolution, so the derivative operator is available in closed
    form as well as by probing -- the two agree to 1e-15, which makes this the
    natural cross-check on the probe machinery.

    Attributes:
        window_length: Odd number of points in the window.
        degree: Degree of the local polynomial.
    """

    name: ClassVar[str] = "sgolay"
    linear: ClassVar[bool] = True
    supported_orders: ClassVar[frozenset[int]] = frozenset({0, 1, 2, 3})
    requires_regular_grid: ClassVar[bool] = True

    window_length: int = 15
    degree: int = 3

    def __post_init__(self) -> None:
        """Validate the fixed-window polynomial."""
        if not _is_integer(self.window_length) or self.window_length < 1:
            raise ValueError("window_length must be a positive integer")
        if self.window_length % 2 == 0:
            raise ValueError("window_length must be odd")
        if not _is_integer(self.degree) or self.degree < 0:
            raise ValueError("degree must be a nonnegative integer")
        if self.degree >= self.window_length:
            raise ValueError("degree must be less than window_length")

    def evaluate(
        self, axis: TimeAxis, y: npt.NDArray[np.float64], derivative_order: int
    ) -> Evaluation:
        """Filter the series and differentiate."""
        if derivative_order > self.degree:
            raise ValueError(
                f"sgolay with degree={self.degree} cannot estimate derivative "
                f"order {derivative_order}"
            )
        window = self._window_for(axis.n)
        # asarray: scipy annotates savgol_filter with an array union; the
        # float64 input guarantees float64 output, at no copy.
        return Evaluation(
            values=np.asarray(
                savgol_filter(
                    y,
                    window_length=window,
                    polyorder=self.degree,
                    delta=axis.delta,
                ),
                dtype=np.float64,
            ),
            derivative=np.asarray(
                savgol_filter(
                    y,
                    window_length=window,
                    polyorder=self.degree,
                    deriv=derivative_order,
                    delta=axis.delta,
                ),
                dtype=np.float64,
            ),
        )

    def _window_for(self, n: int) -> int:
        """The filter width to use on a series of ``n`` points.

        Args:
            n: Number of observations.

        Returns:
            The configured window.

        Raises:
            ValueError: If the series is too short to support the filter.
        """
        if n < self.window_length:
            raise ValueError(
                f"Savitzky-Golay with window_length={self.window_length} needs at "
                f"least {self.window_length} observations, got {n}"
            )
        return self.window_length

    def closed_form_se(
        self, axis: TimeAxis, derivative_order: int, standard_deviation: float
    ) -> npt.NDArray[np.float64]:
        """Interior standard error from the filter coefficients directly.

        Args:
            axis: The time axis.
            derivative_order: Derivative derivative_order.
            standard_deviation: Noise standard deviation.

        Returns:
            The constant interior standard error, broadcast over the series.
        """
        coefficients = savgol_coeffs(
            self._window_for(axis.n),
            self.degree,
            deriv=derivative_order,
            delta=axis.delta,
        )
        return np.full(axis.n, standard_deviation * float(np.linalg.norm(coefficients)))

    def with_scale(self, scale: float, axis: TimeAxis) -> Self:
        """Set the window to ``scale`` of the sample."""
        scale = _validate_scale(scale)
        return replace(self, window_length=_odd(round(scale * axis.n), self.degree + 2))

    def scale_of(self, axis: TimeAxis) -> float:
        """Window length as a fraction of the sample."""
        return min(self.window_length / axis.n, 1.0)

    def params(self) -> dict[str, Any]:
        """Report the window and polynomial degree."""
        return {"window_length": self.window_length, "degree": self.degree}


@register
@dataclass(frozen=True)
class NaiveDifference(Smoother):
    """Central finite differences on the raw series.

    The estimator the package exists to argue against: it does no smoothing,
    so it inherits the noise directly. Kept because the comparison is the
    point, and because its exact variance makes that comparison quantitative.
    """

    name: ClassVar[str] = "naive"
    linear: ClassVar[bool] = True
    supported_orders: ClassVar[frozenset[int]] = frozenset({1})
    requires_regular_grid: ClassVar[bool] = True

    def evaluate(
        self, axis: TimeAxis, y: npt.NDArray[np.float64], derivative_order: int
    ) -> Evaluation:
        """Average the forward and backward difference at each point."""
        del derivative_order
        backward = np.full(axis.n, np.nan)
        forward = np.full(axis.n, np.nan)
        backward[1:] = (y[1:] - y[:-1]) / axis.delta
        forward[:-1] = (y[1:] - y[:-1]) / axis.delta
        with np.errstate(invalid="ignore"):
            derivative = np.nanmean(np.vstack([backward, forward]), axis=0)
        return Evaluation(values=y.copy(), derivative=derivative)

    def analytic_operators(
        self, axis: TimeAxis, derivative_order: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
        """The difference stencil, written down directly."""
        del derivative_order
        n = axis.n
        derivative = np.zeros((n, n), dtype=np.float64)
        step = axis.delta
        if n == 1:
            return np.eye(1), np.full((1, 1), np.nan)
        # Interior points average the forward and backward difference, which
        # telescopes to the central difference over two steps.
        for i in range(n):
            if i == 0:
                derivative[i, 0], derivative[i, 1] = -1 / step, 1 / step
            elif i == n - 1:
                derivative[i, n - 2], derivative[i, n - 1] = -1 / step, 1 / step
            else:
                derivative[i, i - 1] = -0.5 / step
                derivative[i, i + 1] = 0.5 / step
        return np.eye(n, dtype=np.float64), derivative

    def with_scale(self, scale: float, axis: TimeAxis) -> Self:
        """Reject a scale because finite differencing has no bandwidth."""
        _validate_scale(scale)
        del axis
        raise ValueError("naive differencing has no smoothing scale")

    def scale_of(self, axis: TimeAxis) -> None:
        """Return None because finite differencing has no bandwidth."""
        del axis


@register
@dataclass(frozen=True)
class LocalPolynomial(Smoother):
    """Local polynomial regression with kernel weights.

    Fits a weighted least squares polynomial around every point and reads the
    derivative off the coefficients.

    Attributes:
        bandwidth: Kernel width as a fraction of the series span.
        degree: Degree of the local polynomial.
        kernel: ``'gaussian'``, ``'epanechnikov'`` or ``'uniform'``.
    """

    name: ClassVar[str] = "local_poly"
    linear: ClassVar[bool] = True

    bandwidth: float = 0.2
    degree: int = 2
    kernel: str = "gaussian"

    def __post_init__(self) -> None:
        """Validate the local regression configuration."""
        if not _finite_number(self.bandwidth) or not 0.0 < self.bandwidth <= 1.0:
            raise ValueError("bandwidth must be finite and in (0, 1]")
        if (
            isinstance(self.degree, (bool, np.bool_))
            or not isinstance(self.degree, (int, np.integer))
            or self.degree < 0
        ):
            raise ValueError("degree must be a nonnegative integer")
        if self.kernel not in {"gaussian", "epanechnikov", "uniform"}:
            raise ValueError(
                f"Unknown kernel {self.kernel!r}; use 'gaussian', "
                "'epanechnikov' or 'uniform'"
            )

    def _weights(
        self, distances: npt.NDArray[np.float64], h: float
    ) -> npt.NDArray[np.float64]:
        """Kernel weights at the given distances."""
        u = distances / h
        match self.kernel:
            case "gaussian":
                return np.exp(-0.5 * u**2)
            case "epanechnikov":
                return np.maximum(0.0, 0.75 * (1 - u**2))
            case "uniform":
                return (np.abs(u) <= 1).astype(float)
            case _:
                raise ValueError(f"Unknown kernel: {self.kernel}")

    def evaluate(
        self, axis: TimeAxis, y: npt.NDArray[np.float64], derivative_order: int
    ) -> Evaluation:
        """Fit a weighted local polynomial at every point."""
        if derivative_order > self.degree:
            raise ValueError(
                f"local_poly with degree={self.degree} cannot estimate a "
                f"derivative of derivative_order {derivative_order}; raise the degree"
            )
        n = axis.n
        h = self.bandwidth * np.ptp(axis.x)
        values = np.full(n, np.nan)
        derivative = np.full(n, np.nan)
        features = PolynomialFeatures(degree=self.degree, include_bias=True)

        for i in range(n):
            centered = axis.x - axis.x[i]
            weights = self._weights(np.abs(centered), h)
            mask = weights > 1e-10
            if int(np.sum(mask)) < self.degree + 1:
                raise ValueError(
                    "bandwidth leaves too few weighted observations for the degree"
                )
            design = features.fit_transform(centered[mask].reshape(-1, 1))
            w = weights[mask]
            try:
                xtw = design.T * w
                coefficients = np.linalg.solve(xtw @ design, xtw @ y[mask])
            except np.linalg.LinAlgError:
                raise ValueError(
                    "local polynomial system is singular; increase bandwidth or "
                    "lower degree"
                ) from None
            values[i] = coefficients[0]
            # The Taylor coefficient carries a derivative-order factorial.
            derivative[i] = coefficients[derivative_order] * float(
                math.factorial(derivative_order)
            )

        return Evaluation(values=values, derivative=derivative)

    def analytic_operators(
        self, axis: TimeAxis, derivative_order: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
        """Emit the operator rows from the same weighted least squares solve.

        Each point's estimate is ``e_k' (X'WX)^-1 X'W y``; the row vector in
        front of ``y`` is that point's operator row. Building it costs one
        extra solve against an identity block rather than n full evaluations.
        """
        if derivative_order > self.degree:
            return None
        n = axis.n
        h = self.bandwidth * np.ptp(axis.x)
        smoothing = np.zeros((n, n), dtype=np.float64)
        derivative = np.zeros((n, n), dtype=np.float64)
        features = PolynomialFeatures(degree=self.degree, include_bias=True)
        scale = float(math.factorial(derivative_order))

        for i in range(n):
            centered = axis.x - axis.x[i]
            weights = self._weights(np.abs(centered), h)
            mask = weights > 1e-10
            if int(np.sum(mask)) < self.degree + 1:
                raise ValueError(
                    "bandwidth leaves too few weighted observations for the degree"
                )
            design = features.fit_transform(centered[mask].reshape(-1, 1))
            w = weights[mask]
            xtw = design.T * w
            try:
                # Rows of (X'WX)^-1 X'W map y directly onto the coefficients.
                projection = np.linalg.solve(xtw @ design, xtw)
            except np.linalg.LinAlgError:
                raise ValueError(
                    "local polynomial system is singular; increase bandwidth or "
                    "lower degree"
                ) from None
            smoothing[i, mask] = projection[0]
            derivative[i, mask] = projection[derivative_order] * scale

        return smoothing, derivative

    def with_scale(self, scale: float, axis: TimeAxis) -> Self:
        """Set the kernel bandwidth to ``scale`` of the span."""
        del axis
        return replace(self, bandwidth=_validate_scale(scale))

    def scale_of(self, axis: TimeAxis) -> float:
        """Bandwidth as a fraction of the span."""
        del axis
        return self.bandwidth

    def params(self) -> dict[str, Any]:
        """Report bandwidth, degree and kernel."""
        return {
            "bandwidth": self.bandwidth,
            "degree": self.degree,
            "kernel": self.kernel,
        }


@register
@dataclass(frozen=True)
class Loess(Smoother):
    """Local-linear LOWESS smoothing and numerical differentiation.

    Linear only when ``robust`` is off. The robust variant reweights according
    to the residuals it just computed, which makes the map depend on the data
    and routes uncertainty to the bootstrap. The implementation delegates the
    smooth to statsmodels' reference LOWESS implementation, which is local
    linear, then differentiates the returned smooth on its actual time axis.

    Attributes:
        span: Fraction of the sample in each local regression.
        robust: Whether to run LOWESS's robustifying iterations.

    Note:
        LOWESS is due to William S. Cleveland (1979),
        https://doi.org/10.1080/01621459.1979.10481038. The smooth is computed
        by :func:`statsmodels.nonparametric.smoothers_lowess.lowess`.
    """

    name: ClassVar[str] = "loess"
    supported_orders: ClassVar[frozenset[int]] = frozenset({0, 1})

    span: float = 0.3
    robust: bool = True

    def __post_init__(self) -> None:
        """Validate the local regression configuration."""
        if not _finite_number(self.span) or not 0.0 < self.span <= 1.0:
            raise ValueError("span must be finite and in (0, 1]")
        if not isinstance(self.robust, (bool, np.bool_)):
            raise ValueError("robust must be boolean")

    @property
    def is_linear(self) -> bool:
        """Linear exactly when robust reweighting is disabled."""
        return not self.robust

    def evaluate(
        self, axis: TimeAxis, y: npt.NDArray[np.float64], derivative_order: int
    ) -> Evaluation:
        """Smooth with LOWESS, then differentiate the returned curve."""
        smoothed = np.asarray(
            lowess(
                y,
                axis.x,
                frac=self.span,
                it=3 if self.robust else 0,
                delta=0.0,
                return_sorted=False,
            ),
            dtype=np.float64,
        )
        if derivative_order == 0:
            derivative = smoothed.copy()
        else:
            if axis.n < 2:
                raise ValueError(
                    "loess needs at least two observations to differentiate"
                )
            derivative = np.gradient(smoothed, axis.x, edge_order=1)
        return Evaluation(values=smoothed, derivative=derivative)

    def with_scale(self, scale: float, axis: TimeAxis) -> Self:
        """Set the LOWESS fraction."""
        del axis
        return replace(self, span=_validate_scale(scale))

    def scale_of(self, axis: TimeAxis) -> float:
        """The LOWESS fraction is already a scale."""
        del axis
        return self.span

    def params(self) -> dict[str, Any]:
        """Report span and robustness."""
        return {"span": self.span, "robust": self.robust}


def _natural_spline_penalty(axis: TimeAxis) -> npt.NDArray[np.float64]:
    """Integrated squared-curvature matrix for values at the observed knots."""
    cached = _SPLINE_PENALTY_CACHE.get(axis.key())
    if cached is not None:
        return cached

    n = axis.n
    if n < 5:
        raise ValueError(f"smoothing_spline needs at least 5 observations, got {n}")
    spacing = np.diff(axis.x)
    interior = n - 2
    columns = np.arange(interior)
    rows = np.concatenate([columns, columns + 1, columns + 2])
    column_indices = np.tile(columns, 3)
    coefficients = np.concatenate(
        [
            1.0 / spacing[:-1],
            -(1.0 / spacing[:-1] + 1.0 / spacing[1:]),
            1.0 / spacing[1:],
        ]
    )
    divided_differences = csc_matrix(
        (coefficients, (rows, column_indices)),
        shape=(n, interior),
    )

    roughness_banded = np.zeros((3, interior), dtype=np.float64)
    roughness_banded[1] = (spacing[:-1] + spacing[1:]) / 3.0
    if interior > 1:
        roughness_banded[0, 1:] = spacing[1:-1] / 6.0
        roughness_banded[2, :-1] = spacing[1:-1] / 6.0
    solved = solve_banded(
        (1, 1),
        roughness_banded,
        divided_differences.T.toarray(),
        check_finite=False,
    )
    penalty = np.asarray(divided_differences @ solved, dtype=np.float64)
    penalty = (penalty + penalty.T) / 2.0

    if penalty.nbytes <= OPERATOR_CACHE_BYTES:
        while (
            _SPLINE_PENALTY_CACHE
            and sum(value.nbytes for value in _SPLINE_PENALTY_CACHE.values())
            + penalty.nbytes
            > OPERATOR_CACHE_BYTES
        ):
            _SPLINE_PENALTY_CACHE.pop(next(iter(_SPLINE_PENALTY_CACHE)))
        _SPLINE_PENALTY_CACHE[axis.key()] = penalty
    return penalty


def _is_homoskedastic_iid(noise: NoiseFit) -> bool:
    """Whether a covariance is a scalar multiple of the identity."""
    if noise.standard_deviation == 0.0:
        return True
    if noise.explicit is not None:
        diagonal = np.diag(noise.explicit)
        return bool(
            np.all(noise.explicit == np.diag(diagonal))
            and np.all(diagonal == diagonal[0])
        )
    if noise.standard_deviation_vector is not None:
        return bool(
            np.all(
                noise.standard_deviation_vector == noise.standard_deviation_vector[0]
            )
        )
    return noise.phi == 0.0


def _gml_smoothing_spline(
    axis: TimeAxis,
    y: npt.NDArray[np.float64],
    derivative_order: int,
    covariance: npt.NDArray[np.float64],
) -> Evaluation:
    """Fit a natural smoothing spline by covariance-aware GML."""
    penalty = _natural_spline_penalty(axis)
    try:
        factor = np.tril(cholesky(covariance, lower=True, check_finite=False))
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "automatic smoothing-spline tuning needs a positive-definite "
            "noise covariance; supply an explicit penalty for a singular one"
        ) from exc

    transformed_penalty = factor.T @ penalty @ factor
    transformed_penalty = (transformed_penalty + transformed_penalty.T) / 2.0
    eigenvalues, eigenvectors = eigh(transformed_penalty, check_finite=False)

    # A natural cubic spline has the two-dimensional unpenalized span {1, x}.
    # The remaining n - 2 eigenvalues carry curvature and must be positive.
    positive_values = eigenvalues[2:]
    tolerance = (
        np.finfo(np.float64).eps * axis.n * max(float(np.max(np.abs(eigenvalues))), 1.0)
    )
    if positive_values[0] <= tolerance:
        raise ValueError(
            "the time axis and covariance do not identify a stable smoothing "
            "spline; supply an explicit penalty"
        )

    whitened = solve_triangular(
        factor,
        y,
        lower=True,
        check_finite=False,
    )
    coordinates = eigenvectors.T @ whitened
    penalized_coordinates = coordinates[2:]
    geometric_mean = float(np.exp(np.mean(np.log(positive_values))))
    normalized_values = positive_values / geometric_mean
    epsilon = np.finfo(np.float64).eps
    lower_bound = float(np.log(np.sqrt(epsilon)))
    upper_bound = -lower_bound

    def objective(log_scaled_penalty: float) -> float:
        scaled = np.exp(log_scaled_penalty) * normalized_values
        inverse_variance = scaled / (1.0 + scaled)
        residual_sum = float(np.dot(penalized_coordinates**2, inverse_variance))
        if residual_sum <= 0.0:
            return np.inf
        return float(
            len(positive_values) * np.log(residual_sum / len(positive_values))
            + np.sum(np.log1p(1.0 / scaled))
        )

    if np.linalg.norm(penalized_coordinates) <= np.linalg.norm(whitened) * np.sqrt(
        epsilon
    ):
        log_scaled_penalty = upper_bound
    else:
        optimum = minimize_scalar(
            objective,
            bounds=(lower_bound, upper_bound),
            method="bounded",
        )
        if not optimum.success or not np.isfinite(optimum.fun):
            raise ValueError(
                "generalized maximum-likelihood smoothing-penalty selection failed"
            )
        log_scaled_penalty = float(optimum.x)

    selected_penalty = float(np.exp(log_scaled_penalty) / geometric_mean)
    shrinkage = 1.0 + selected_penalty * positive_values
    coordinates[2:] /= shrinkage
    fitted_values = factor @ (eigenvectors @ coordinates)
    spline = make_interp_spline(
        axis.x,
        fitted_values,
        k=3,
        bc_type="natural",
    )
    return Evaluation(
        values=np.asarray(fitted_values, dtype=np.float64),
        derivative=np.asarray(
            spline(axis.x, nu=derivative_order),
            dtype=np.float64,
        ),
        params={
            "selected_penalty": selected_penalty,
            "selection_method": "gml",
        },
    )


@register
@dataclass(frozen=True)
class SmoothingSpline(Smoother):
    """Cubic smoothing spline with a second-derivative roughness penalty.

    With independent errors this wraps
    :func:`scipy.interpolate.make_smoothing_spline`, SciPy's implementation of
    Woltring's generalized cross-validation algorithm. With a non-diagonal or
    heteroskedastic covariance it selects the penalty by generalized maximum
    likelihood and solves the corresponding penalized generalized least-squares
    problem. A fixed penalty makes the smoother linear; selecting it from the
    data makes the fit nonlinear.

    Attributes:
        penalty: Roughness penalty. Chosen by generalized cross-validation when
            None -- which makes the fit data-dependent and therefore nonlinear.

    Note:
        The GCV algorithm is due to Herman J. Woltring (1986),
        https://doi.org/10.1016/0141-1195(86)90098-7. The correlated-error
        penalized-GLS and GML formulation follows Diggle and Hutchinson (1989),
        https://doi.org/10.1111/j.1467-842X.1989.tb00510.x, and Wang (1998),
        https://doi.org/10.1080/01621459.1998.10474115. Incline selects the
        penalty conditional on the fitted :class:`~incline.noise.NoiseModel`;
        it does not jointly optimize the covariance and penalty. The independent
        fit and analytic differentiation delegate to
        :func:`scipy.interpolate.make_smoothing_spline`.
    """

    name: ClassVar[str] = "smoothing_spline"
    supported_orders: ClassVar[frozenset[int]] = frozenset({0, 1, 2, 3})

    penalty: float | None = None

    def __post_init__(self) -> None:
        """Validate the optional roughness penalty."""
        if self.penalty is not None and (
            not _finite_number(self.penalty) or self.penalty < 0
        ):
            raise ValueError("penalty must be finite and nonnegative")

    @property
    def is_linear(self) -> bool:
        """Linear when the penalty is fixed, not cross-validated."""
        return self.penalty is not None

    def evaluate(
        self, axis: TimeAxis, y: npt.NDArray[np.float64], derivative_order: int
    ) -> Evaluation:
        """Fit the smoothing spline and differentiate it."""
        if axis.n < 5:
            raise ValueError(
                f"smoothing_spline needs at least 5 observations, got {axis.n}"
            )
        spline = make_smoothing_spline(axis.x, y, lam=self.penalty)
        return Evaluation(
            values=np.asarray(spline(axis.x), dtype=np.float64),
            derivative=np.asarray(
                spline(axis.x, nu=derivative_order), dtype=np.float64
            ),
        )

    def evaluate_with_noise(
        self,
        axis: TimeAxis,
        y: npt.NDArray[np.float64],
        derivative_order: int,
        noise: NoiseFit | None,
    ) -> Evaluation:
        """Use covariance-aware GML when an adaptive fit has dependent errors."""
        if self.penalty is not None or noise is None or _is_homoskedastic_iid(noise):
            return self.evaluate(axis, y, derivative_order)
        return _gml_smoothing_spline(
            axis,
            y,
            derivative_order,
            noise.covariance(axis.n),
        )

    def bootstrap_uncertainty(
        self,
        estimate: TrendEstimate,
        axis: TimeAxis,
        y: npt.NDArray[np.float64],
        derivative_order: int,
        noise_model: NoiseModel,
        noise_fit: NoiseFit,
        confidence_level: float,
        n_bootstrap: int,
        random_state: int | np.random.Generator | None,
    ) -> tuple[
        npt.NDArray[np.float64] | None,
        npt.NDArray[np.float64] | None,
        npt.NDArray[np.float64] | None,
    ]:
        """Repeat covariance fitting and GML selection in every Gaussian draw."""
        if self.penalty is not None or _is_homoskedastic_iid(noise_fit):
            return super().bootstrap_uncertainty(
                estimate,
                axis,
                y,
                derivative_order,
                noise_model,
                noise_fit,
                confidence_level,
                n_bootstrap,
                random_state,
            )

        def refit(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            refitted_noise = noise_model.estimate(values, axis)
            return self.evaluate_with_noise(
                axis,
                values,
                derivative_order,
                refitted_noise,
            ).derivative

        return parametric_bootstrap(
            fitted=estimate.values,
            refit=refit,
            noise=noise_fit,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=random_state,
        )

    def with_scale(self, scale: float, axis: TimeAxis) -> Self:
        """Set the penalty from an equivalent bandwidth.

        For a cubic smoothing spline the equivalent kernel width behaves like
        ``(penalty / n) ** (1/4)``, so a target width of ``scale * span`` implies
        ``penalty = n * (scale * span) ** 4``.
        """
        scale = _validate_scale(scale)
        width = scale * (np.ptp(axis.x) or 1.0)
        return replace(self, penalty=float(axis.n * width**4))

    def scale_of(self, axis: TimeAxis) -> float | None:
        """Invert the bandwidth-to-penalty map."""
        if self.penalty is None:
            return None
        span = float(np.ptp(axis.x)) or 1.0
        return float(min((self.penalty / axis.n) ** 0.25 / span, 1.0))

    def params(self) -> dict[str, Any]:
        """Report the penalty."""
        return {
            "penalty": self.penalty,
            "selection_method": "gcv" if self.penalty is None else "fixed",
        }


@register
@dataclass(frozen=True)
class L1TrendFilter(Smoother):
    """L1 trend filtering: piecewise-polynomial fit with sparse kinks.

    This implements the estimator of Kim et al. (2009),
    https://doi.org/10.1137/070690274, and its arbitrary-input extension from
    Tibshirani (2014), https://doi.org/10.1214/13-AOS1189. The convex problem is
    solved through its box-constrained least-squares dual using SciPy's
    ``lsq_linear``; Incline constructs the grid-aware penalty operator and maps
    the dual solution back to the fitted trend.

    The L1 penalty is nonlinear in the data -- that is what produces sparse
    kinks -- so uncertainty is bootstrapped.

    The penalized difference order and the reported derivative order are
    separate settings; the previous implementation used one for both.

    Attributes:
        penalty: Absolute penalty on the differences. Larger means fewer kinks.
            Exactly one of ``penalty`` and ``penalty_fraction`` is required.
        penalty_fraction: Fraction of the smallest penalty that collapses the
            fit to a polynomial of degree ``difference_order - 1``. Exactly one
            of ``penalty`` and ``penalty_fraction`` is required.
        difference_order: Order of the penalized difference. Two gives a
            piecewise-linear trend, the usual choice.
        max_iter: Bounded least-squares iteration cap.
        tolerance: Optimizer convergence tolerance.
    """

    name: ClassVar[str] = "l1_filter"

    penalty: float | None = None
    penalty_fraction: float | None = None
    difference_order: int = 2
    max_iter: int = 1000
    tolerance: float = 1e-8

    def __post_init__(self) -> None:
        """Validate the optimization and penalty settings."""
        if (self.penalty is None) == (self.penalty_fraction is None):
            raise ValueError(
                "exactly one of penalty and penalty_fraction must be provided"
            )
        if self.penalty is not None and (
            not _finite_number(self.penalty) or self.penalty < 0
        ):
            raise ValueError("penalty must be finite and nonnegative")
        if self.penalty_fraction is not None and (
            not _finite_number(self.penalty_fraction)
            or not 0.0 <= self.penalty_fraction <= 1.0
        ):
            raise ValueError("penalty_fraction must be finite and between 0 and 1")
        if not _is_integer(self.difference_order) or self.difference_order < 1:
            raise ValueError("difference_order must be a positive integer")
        if not _is_integer(self.max_iter) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not _finite_number(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")

    def _differences(self, axis: TimeAxis) -> npt.NDArray[np.float64]:
        """Construct Tibshirani's divided-difference operator on the axis."""
        if self.difference_order >= axis.n:
            raise ValueError(
                f"difference_order={self.difference_order} requires more than "
                f"{self.difference_order} observations; axis has {axis.n}"
            )

        differences = np.diff(np.eye(axis.n), axis=0)
        for order in range(1, self.difference_order):
            spacings = axis.x[order:] - axis.x[:-order]
            weights = order / spacings
            differences = np.diff(np.eye(axis.n - order), axis=0) @ (
                weights[:, None] * differences
            )
        return differences

    @staticmethod
    def _maximum_penalty(
        differences: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> float:
        """Return the first penalty whose solution lies in ``null(D)``."""
        dual_unconstrained = np.linalg.solve(
            differences @ differences.T, differences @ y
        )
        return float(np.max(np.abs(dual_unconstrained), initial=0.0))

    def _resolved_penalty(
        self, differences: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> tuple[float, float]:
        """Resolve an absolute or fractional penalty against its exact scale."""
        maximum = self._maximum_penalty(differences, y)
        if self.penalty is not None:
            return self.penalty, maximum
        fraction = self.penalty_fraction
        if fraction is None:
            raise RuntimeError("L1 penalty configuration changed after validation")
        return fraction * maximum, maximum

    def evaluate(
        self, axis: TimeAxis, y: npt.NDArray[np.float64], derivative_order: int
    ) -> Evaluation:
        """Solve the L1 trend filtering problem, then difference the fit."""
        differences = self._differences(axis)
        penalty, maximum_penalty = self._resolved_penalty(differences, y)

        iterations = 0
        optimality = 0.0
        if penalty == 0 or maximum_penalty == 0:
            trend = y.copy()
        else:
            solved = lsq_linear(
                differences.T,
                y,
                bounds=(-penalty, penalty),
                tol=self.tolerance,
                lsq_solver="exact",
                max_iter=self.max_iter,
            )
            if not solved.success:
                raise RuntimeError(
                    "L1 trend-filter optimization did not converge within "
                    f"max_iter={self.max_iter}: {solved.message}"
                )
            trend = y - differences.T @ solved.x
            iterations = solved.nit
            optimality = float(solved.optimality)

        derivative = np.gradient(trend, axis.x, edge_order=1)
        for _ in range(derivative_order - 1):
            derivative = np.gradient(derivative, axis.x, edge_order=1)
        return Evaluation(
            values=trend,
            derivative=derivative,
            params={
                "resolved_penalty": penalty,
                "maximum_penalty": maximum_penalty,
                "optimizer_iterations": iterations,
                "optimizer_optimality": optimality,
            },
        )

    def with_scale(self, scale: float, axis: TimeAxis) -> Self:
        """Set the penalty as a fraction of its saturating value."""
        del axis
        return replace(
            self,
            penalty=None,
            penalty_fraction=_validate_scale(scale),
        )

    def scale_of(self, axis: TimeAxis) -> float | None:
        """Return the configured relative scale when one exists."""
        del axis
        if self.penalty_fraction is None:
            return None
        return self.penalty_fraction

    def params(self) -> dict[str, Any]:
        """Report the configured penalty and optimizer settings."""
        return {
            "penalty": self.penalty,
            "penalty_fraction": self.penalty_fraction,
            "difference_order": self.difference_order,
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
        }


def build(name: str, **kwargs: Any) -> Smoother:
    """Construct a smoother by name.

    Replaces the string ``match`` that previously lived in ``estimate_trend``
    and needed a new branch for every method.

    Args:
        name: Registered smoother name.
        **kwargs: Passed to the smoother's constructor.

    Returns:
        The smoother.

    Raises:
        ValueError: If the name is not registered.
    """
    if name not in SMOOTHERS:
        raise ValueError(f"Unknown method {name!r}; available: {sorted(SMOOTHERS)}")
    return SMOOTHERS[name](**kwargs)
