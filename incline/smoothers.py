"""Smoothers, and the single implementation of fitting one.

Every method in the package answers the same question -- given a noisy series,
what is the derivative of the underlying smooth curve -- and differs only in how
it smooths. :class:`Smoother` is that shared shape, and :meth:`Smoother.fit`
is written once, here, rather than eleven times across the package.

The important declaration a smoother makes is :attr:`Smoother.is_linear`. If the
derivative is a fixed linear map of the data then its sampling variance is exact
and cheap; if not, it has to be simulated. Probing settles the question rather
than intuition: ``UnivariateSpline`` looks like a smoother but picks its knots
from the data, and LOESS is linear only with robust reweighting switched off --
which is not its default.

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
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import numpy.typing as npt
from scipy.interpolate import UnivariateSpline, make_smoothing_spline
from scipy.signal import savgol_coeffs, savgol_filter
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.nonparametric.smoothers_lowess import lowess

from .axis import TimeAxis
from .noise import NoiseModel, resolve_noise
from .result import Provenance, TrendEstimate
from .uncertainty import (
    bias_corrected_operator,
    operator_variance,
    residual_bootstrap,
    simultaneous_critical_value,
    verify_linearity,
)


if TYPE_CHECKING:
    from typing import Self

# Operators depend only on (smoother, axis, order), so they survive across
# fits. A handful of entries covers a multi-scale sweep or a simulation study.
#
# The bound is in bytes rather than entries because each entry holds two n x n
# float64 matrices: at n = 2000 that is 64 MB apiece, so a 64-entry allowance
# would have reached about 4 GB before evicting anything.
OPERATOR_CACHE_BYTES = 512 * 1024 * 1024
_OPERATOR_CACHE: dict[Any, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}

SMOOTHERS: dict[str, type[Smoother]] = {}


def _cache_bytes() -> int:
    """Total memory held by the operator cache."""
    return sum(a.nbytes + b.nbytes for a, b in _OPERATOR_CACHE.values())


def register(cls: type[Smoother]) -> type[Smoother]:
    """Add a smoother to the registry keyed by its name.

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
    """

    values: npt.NDArray[np.float64]
    derivative: npt.NDArray[np.float64]


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
        self, axis: TimeAxis, y: npt.NDArray[np.float64], order: int
    ) -> Evaluation:
        """Smooth a series and differentiate the smooth.

        Args:
            axis: The time axis.
            y: Observed values.
            order: Derivative order.

        Returns:
            The smoothed values and the derivative.
        """

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
    def scale_of(self, axis: TimeAxis) -> float:
        """Report this smoother's current scale as a fraction of the span.

        Args:
            axis: The time axis.

        Returns:
            The scale in (0, 1].
        """

    def params(self) -> dict[str, Any]:
        """Smoother-specific settings, recorded on the result."""
        return {}

    def analytic_operators(
        self, axis: TimeAxis, order: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
        """State the smoothing and derivative operators directly, if known.

        A linear smoother's operators are already implicit in the arithmetic it
        performs, so a smoother that fits a local system per point can emit its
        operator rows from the same solve. Doing so turns an O(n) probe into a
        constant factor. Returning None falls back to probing.

        Args:
            axis: The time axis.
            order: Derivative order.

        Returns:
            Tuple of (smoothing operator, derivative operator), or None.
        """
        del axis, order
        return None

    def operators(
        self, axis: TimeAxis, order: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The smoothing and derivative operators for this configuration.

        Cached: an operator depends on the smoother's settings, the axis and
        the derivative order, but never on the observed values. Repeated fits
        over the same grid -- a multi-scale sweep, a Monte Carlo study -- reuse
        the work.

        Args:
            axis: The time axis.
            order: Derivative order.

        Returns:
            Tuple of (smoothing operator, derivative operator).
        """
        cache_key = (self, axis.key(), order)
        cached = _OPERATOR_CACHE.get(cache_key)
        if cached is not None:
            return cached

        analytic = self.analytic_operators(axis, order)
        if analytic is not None:
            smoothing, derivative = analytic
        else:
            smoothing, derivative = self._probe_operators(axis, order)

        size = smoothing.nbytes + derivative.nbytes
        if size <= OPERATOR_CACHE_BYTES:
            while _OPERATOR_CACHE and _cache_bytes() + size > OPERATOR_CACHE_BYTES:
                _OPERATOR_CACHE.pop(next(iter(_OPERATOR_CACHE)))
            _OPERATOR_CACHE[cache_key] = (smoothing, derivative)
        return smoothing, derivative

    def _probe_operators(
        self, axis: TimeAxis, order: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Recover both operators by evaluating on each basis vector."""
        smoothing = np.empty((axis.n, axis.n), dtype=np.float64)
        derivative = np.empty((axis.n, axis.n), dtype=np.float64)
        basis = np.zeros(axis.n, dtype=np.float64)
        for j in range(axis.n):
            basis[j] = 1.0
            evaluation = self.evaluate(axis, basis, order)
            smoothing[:, j] = evaluation.values
            derivative[:, j] = evaluation.derivative
            basis[j] = 0.0
        return smoothing, derivative

    def native_posterior(
        self,
        axis: TimeAxis,
        y: npt.NDArray[np.float64],
        order: int,
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
            order: Derivative order.
            confidence_level: Confidence level.

        Returns:
            Tuple of (se, ci_lower, ci_upper), or None if unsupported.
        """
        del axis, y, order, confidence_level
        return None

    def fit(
        self,
        axis: TimeAxis,
        y: npt.NDArray[np.float64],
        order: int = 1,
        se: bool = False,
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
            order: Derivative order.
            se: Whether to compute standard errors. Off by default because
                the exact route costs one smoother evaluation per observation.
            noise: Noise model, or ``'iid'`` / ``'ar1'``.
            bias_correct: Subtract an estimate of smoothing bias using a
                less-smoothed pilot fit. Linear smoothers only.
            simultaneous: Return a band covering the whole curve at once
                rather than pointwise intervals.
            confidence_level: Confidence level for intervals.
            pilot_scale: Scale of the pilot fit used for bias correction.
                Defaults to a third of this smoother's scale.
            n_bootstrap: Replicates, for nonlinear smoothers.
            random_state: Seed or Generator.

        Returns:
            The estimate.

        Raises:
            ValueError: If the derivative order is unsupported, or bias
                correction is requested for a nonlinear smoother.
        """
        y = np.asarray(y, dtype=np.float64)
        if order not in self.supported_orders:
            raise ValueError(
                f"{self.name} supports derivative orders "
                f"{sorted(self.supported_orders)}, got {order}"
            )
        if len(y) != axis.n:
            raise ValueError(f"y has {len(y)} values but the axis has {axis.n} points")
        if self.requires_regular_grid:
            # Once per fit, not once per probe evaluation.
            axis.require_regular(self.name)

        if bias_correct:
            return self._fit_bias_corrected(
                axis,
                y,
                order,
                se,
                noise,
                simultaneous,
                confidence_level,
                pilot_scale,
                random_state,
            )

        evaluation = self.evaluate(axis, y, order)
        estimate = self._assemble(axis, y, order, evaluation)
        if not se:
            return estimate

        return self._attach_uncertainty(
            estimate,
            axis,
            y,
            order,
            noise,
            simultaneous,
            confidence_level,
            n_bootstrap,
            random_state,
        )

    def _assemble(
        self,
        axis: TimeAxis,
        y: npt.NDArray[np.float64],
        order: int,
        evaluation: Evaluation,
        bias_corrected: bool = False,
    ) -> TrendEstimate:
        """Wrap an evaluation in a TrendEstimate."""
        del y
        return TrendEstimate(
            axis=axis,
            values=evaluation.values,
            derivative=evaluation.derivative,
            order=order,
            provenance=Provenance(
                method=self.name,
                bias_corrected=bias_corrected,
                params=self.params(),
            ),
        )

    def _attach_uncertainty(
        self,
        estimate: TrendEstimate,
        axis: TimeAxis,
        y: npt.NDArray[np.float64],
        order: int,
        noise: NoiseModel | str | None,
        simultaneous: bool,
        confidence_level: float,
        n_bootstrap: int,
        random_state: int | np.random.Generator | None,
    ) -> TrendEstimate:
        """Compute uncertainty by whichever strategy this smoother supports."""
        noise_model = resolve_noise(noise)
        noise_fit = noise_model.estimate(y, axis)
        noise_label = (
            f"ar1(phi={noise_fit.phi:.3f}, sigma={noise_fit.sigma:.4g})"
            if noise_fit.phi
            else f"iid(sigma={noise_fit.sigma:.4g})"
        )

        native = self.native_posterior(axis, y, order, confidence_level)
        if native is not None:
            # This smoother's posterior is its own, so an external noise model
            # and a whole-curve band do not enter it. Both were being accepted
            # and dropped, and the noise label was recorded anyway -- naming a
            # model that had no effect on the interval.
            if noise is not None:
                warnings.warn(
                    f"{self.name} carries its own posterior, so noise= does "
                    f"not affect its uncertainty.",
                    stacklevel=3,
                )
            if simultaneous:
                warnings.warn(
                    f"{self.name} has no linear operator, so a whole-curve "
                    f"band is unavailable; the interval is pointwise.",
                    stacklevel=3,
                )
            standard_errors, lower, upper = native
            return estimate.with_uncertainty(
                se=standard_errors,
                se_method="native",
                ci_lower=lower,
                ci_upper=upper,
                confidence_level=confidence_level,
            )

        if self.is_linear:
            _, operator = self.operators(axis, order)
            verify_linearity(
                operator,
                lambda v: self.evaluate(axis, v, order).derivative,
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

        block = _block_size(axis.n) if noise_fit.phi else None
        standard_errors, lower, upper = residual_bootstrap(
            y=y,
            fitted=estimate.values,
            refit=lambda v: self.evaluate(axis, v, order).derivative,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            block_size=block,
            random_state=random_state,
            scale=noise_fit.bootstrap_scale(axis.n),
        )
        if standard_errors is None:
            return estimate
        return estimate.with_uncertainty(
            se=standard_errors,
            se_method="bootstrap",
            ci_lower=lower,
            ci_upper=upper,
            confidence_level=confidence_level,
            noise=noise_label,
        )

    @staticmethod
    def _interval_from_se(
        estimate: TrendEstimate,
        standard_errors: npt.NDArray[np.float64],
        se_method: str,
        confidence_level: float,
        noise_label: str,
        multiplier: float | None,
        simultaneous: bool,
    ) -> TrendEstimate:
        """Build an interval, widening it when a simultaneous band is asked for."""
        if multiplier is None:
            return estimate.with_uncertainty(
                se=standard_errors,
                se_method=se_method,
                confidence_level=confidence_level,
                noise=noise_label,
            )
        return estimate.with_uncertainty(
            se=standard_errors,
            se_method=se_method,
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
        order: int,
        se: bool,
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
            pilot_scale = max(self.scale_of(axis) / 3.0, 3.0 / axis.n)
        pilot = self.with_scale(pilot_scale, axis)

        _, derivative_operator = self.operators(axis, order)
        pilot_smoothing, pilot_derivative = pilot.operators(axis, order)
        corrected = bias_corrected_operator(
            derivative_operator, pilot_smoothing, pilot_derivative
        )

        evaluation = self.evaluate(axis, y, order)
        estimate = TrendEstimate(
            axis=axis,
            values=evaluation.values,
            derivative=corrected @ y,
            order=order,
            provenance=Provenance(
                method=self.name, bias_corrected=True, params=self.params()
            ),
        )
        if not se:
            return estimate

        noise_fit = resolve_noise(noise).estimate(y, axis)
        noise_label = (
            f"ar1(phi={noise_fit.phi:.3f}, sigma={noise_fit.sigma:.4g})"
            if noise_fit.phi
            else f"iid(sigma={noise_fit.sigma:.4g})"
        )
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
        window_length: Number of points in the window; forced odd.
        polyorder: Degree of the local polynomial.
    """

    name: ClassVar[str] = "sgolay"
    linear: ClassVar[bool] = True
    supported_orders: ClassVar[frozenset[int]] = frozenset({0, 1, 2, 3})
    requires_regular_grid: ClassVar[bool] = True

    window_length: int = 15
    polyorder: int = 3

    def evaluate(
        self, axis: TimeAxis, y: npt.NDArray[np.float64], order: int
    ) -> Evaluation:
        """Filter the series and differentiate."""
        window = self._window_for(axis.n)
        return Evaluation(
            values=savgol_filter(
                y,
                window_length=window,
                polyorder=self.polyorder,
                delta=axis.delta,
            ),
            derivative=savgol_filter(
                y,
                window_length=window,
                polyorder=self.polyorder,
                deriv=order,
                delta=axis.delta,
            ),
        )

    def _window_for(self, n: int) -> int:
        """The filter width to use on a series of ``n`` points.

        Args:
            n: Number of observations.

        Returns:
            An odd window that fits the series and exceeds the polynomial order.

        Raises:
            ValueError: If the series is too short to support the filter.
        """
        floor = self.polyorder + 2
        if n < floor:
            raise ValueError(
                f"Savitzky-Golay with polyorder={self.polyorder} needs at "
                f"least {floor} observations, got {n}. Lower polyorder, or "
                f"use a method that does not need a fixed window."
            )
        widest = n if n % 2 else n - 1
        # Clamping to the series length must not drop below the order floor,
        # or scipy raises "polyorder must be less than window_length".
        return max(min(_odd(self.window_length, floor), widest), floor)

    def closed_form_se(
        self, axis: TimeAxis, order: int, sigma: float
    ) -> npt.NDArray[np.float64]:
        """Interior standard error from the filter coefficients directly.

        Args:
            axis: The time axis.
            order: Derivative order.
            sigma: Noise standard deviation.

        Returns:
            The constant interior standard error, broadcast over the series.
        """
        coefficients = savgol_coeffs(
            self._window_for(axis.n), self.polyorder, deriv=order, delta=axis.delta
        )
        return np.full(axis.n, sigma * float(np.linalg.norm(coefficients)))

    def with_scale(self, scale: float, axis: TimeAxis) -> Self:
        """Set the window to ``scale`` of the sample."""
        return replace(
            self, window_length=_odd(round(scale * axis.n), self.polyorder + 2)
        )

    def scale_of(self, axis: TimeAxis) -> float:
        """Window length as a fraction of the sample."""
        return min(self.window_length / axis.n, 1.0)

    def params(self) -> dict[str, Any]:
        """Report the window and polynomial order."""
        return {"window_length": self.window_length, "function_order": self.polyorder}


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
        self, axis: TimeAxis, y: npt.NDArray[np.float64], order: int
    ) -> Evaluation:
        """Average the forward and backward difference at each point."""
        del order
        backward = np.full(axis.n, np.nan)
        forward = np.full(axis.n, np.nan)
        backward[1:] = (y[1:] - y[:-1]) / axis.delta
        forward[:-1] = (y[1:] - y[:-1]) / axis.delta
        with np.errstate(invalid="ignore"):
            derivative = np.nanmean(np.vstack([backward, forward]), axis=0)
        return Evaluation(values=y.copy(), derivative=derivative)

    def analytic_operators(
        self, axis: TimeAxis, order: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
        """The difference stencil, written down directly."""
        del order
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
        """No scale to set; finite differencing has no bandwidth."""
        del scale, axis
        return self

    def scale_of(self, axis: TimeAxis) -> float:
        """The narrowest scale there is: one observation."""
        return 1.0 / axis.n


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
        self, axis: TimeAxis, y: npt.NDArray[np.float64], order: int
    ) -> Evaluation:
        """Fit a weighted local polynomial at every point."""
        if order > self.degree:
            raise ValueError(
                f"local_poly with degree={self.degree} cannot estimate a "
                f"derivative of order {order}; raise the degree"
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
                values[i] = y[i]
                continue
            design = features.fit_transform(centered[mask].reshape(-1, 1))
            w = weights[mask]
            try:
                xtw = design.T * w
                coefficients = np.linalg.solve(xtw @ design, xtw @ y[mask])
            except np.linalg.LinAlgError:
                values[i] = y[i]
                continue
            values[i] = coefficients[0]
            # The Taylor coefficient of x^order carries a factor of order!.
            derivative[i] = coefficients[order] * float(math.factorial(order))

        return Evaluation(values=values, derivative=derivative)

    def analytic_operators(
        self, axis: TimeAxis, order: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
        """Emit the operator rows from the same weighted least squares solve.

        Each point's estimate is ``e_k' (X'WX)^-1 X'W y``; the row vector in
        front of ``y`` is that point's operator row. Building it costs one
        extra solve against an identity block rather than n full evaluations.
        """
        if order > self.degree:
            return None
        n = axis.n
        h = self.bandwidth * np.ptp(axis.x)
        smoothing = np.zeros((n, n), dtype=np.float64)
        derivative = np.zeros((n, n), dtype=np.float64)
        features = PolynomialFeatures(degree=self.degree, include_bias=True)
        scale = float(math.factorial(order))

        for i in range(n):
            centered = axis.x - axis.x[i]
            weights = self._weights(np.abs(centered), h)
            mask = weights > 1e-10
            if int(np.sum(mask)) < self.degree + 1:
                smoothing[i, i] = 1.0
                derivative[i, :] = np.nan
                continue
            design = features.fit_transform(centered[mask].reshape(-1, 1))
            w = weights[mask]
            xtw = design.T * w
            try:
                # Rows of (X'WX)^-1 X'W map y directly onto the coefficients.
                projection = np.linalg.solve(xtw @ design, xtw)
            except np.linalg.LinAlgError:
                smoothing[i, i] = 1.0
                derivative[i, :] = np.nan
                continue
            smoothing[i, mask] = projection[0]
            derivative[i, mask] = projection[order] * scale

        return smoothing, derivative

    def with_scale(self, scale: float, axis: TimeAxis) -> Self:
        """Set the kernel bandwidth to ``scale`` of the span."""
        del axis
        return replace(self, bandwidth=float(np.clip(scale, 1e-6, 1.0)))

    def scale_of(self, axis: TimeAxis) -> float:
        """Bandwidth as a fraction of the span."""
        del axis
        return self.bandwidth

    def params(self) -> dict[str, Any]:
        """Report bandwidth, degree and kernel."""
        return {
            "bandwidth": self.bandwidth,
            "function_order": self.degree,
            "kernel": self.kernel,
        }


@register
@dataclass(frozen=True)
class Loess(Smoother):
    """LOESS smoothing, then a local polynomial fit to the smooth.

    Linear only when ``robust`` is off. The robust variant reweights according
    to the residuals it just computed, which makes the map depend on the data
    -- so it is routed to the bootstrap. Note that robust is the default,
    matching the usual expectation of LOESS.

    Attributes:
        frac: Fraction of the sample in each local regression.
        degree: Degree of the polynomial fitted to the smooth.
        robust: Whether to run LOWESS's robustifying iterations.
    """

    name: ClassVar[str] = "loess"

    frac: float = 0.3
    degree: int = 1
    robust: bool = True

    @property
    def is_linear(self) -> bool:
        """Linear exactly when robust reweighting is disabled."""
        return not self.robust

    def evaluate(
        self, axis: TimeAxis, y: npt.NDArray[np.float64], order: int
    ) -> Evaluation:
        """Smooth with LOWESS, then differentiate the smooth locally."""
        if order > self.degree:
            raise ValueError(
                f"loess with degree={self.degree} cannot estimate a "
                f"derivative of order {order}; raise the degree"
            )
        n = axis.n
        smoothed = np.asarray(
            lowess(
                y,
                axis.x,
                frac=self.frac,
                it=3 if self.robust else 0,
                delta=0.0,
                return_sorted=False,
            ),
            dtype=np.float64,
        )

        window = _odd(int(self.frac * n), 3)
        half = window // 2
        derivative = np.full(n, np.nan)
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            if hi - lo < self.degree + 1:
                continue
            try:
                coefficients = np.polyfit(
                    axis.x[lo:hi] - axis.x[i], smoothed[lo:hi], self.degree
                )
            except np.linalg.LinAlgError:
                continue
            # polyfit returns highest power first.
            derivative[i] = coefficients[-(order + 1)] * float(math.factorial(order))
        return Evaluation(values=smoothed, derivative=derivative)

    def with_scale(self, scale: float, axis: TimeAxis) -> Self:
        """Set the LOWESS fraction."""
        del axis
        return replace(self, frac=float(np.clip(scale, 3e-3, 1.0)))

    def scale_of(self, axis: TimeAxis) -> float:
        """The LOWESS fraction is already a scale."""
        del axis
        return self.frac

    def params(self) -> dict[str, Any]:
        """Report fraction, degree and robustness."""
        return {
            "bandwidth": self.frac,
            "function_order": self.degree,
            "robust": self.robust,
        }


@register
@dataclass(frozen=True)
class PenalizedSpline(Smoother):
    """Smoothing spline with a fixed roughness penalty.

    Linear, because the penalty is fixed rather than chosen from the data.
    This is the spline to use when exact standard errors matter; see
    :class:`InterpolatingSpline` for the knot-selecting alternative.

    Attributes:
        lam: Roughness penalty. Chosen by generalized cross-validation when
            None -- which makes the fit data-dependent and therefore nonlinear.
    """

    name: ClassVar[str] = "pspline"
    supported_orders: ClassVar[frozenset[int]] = frozenset({0, 1, 2, 3})

    lam: float | None = None

    @property
    def is_linear(self) -> bool:
        """Linear when the penalty is fixed, not cross-validated."""
        return self.lam is not None

    def evaluate(
        self, axis: TimeAxis, y: npt.NDArray[np.float64], order: int
    ) -> Evaluation:
        """Fit the penalized spline and differentiate it."""
        spline = make_smoothing_spline(axis.x, y, lam=self.lam)
        return Evaluation(
            values=np.asarray(spline(axis.x), dtype=np.float64),
            derivative=np.asarray(spline(axis.x, nu=order), dtype=np.float64),
        )

    def with_scale(self, scale: float, axis: TimeAxis) -> Self:
        """Set the penalty from an equivalent bandwidth.

        For a cubic smoothing spline the equivalent kernel width behaves like
        ``(lam / n) ** (1/4)``, so a target width of ``scale * span`` implies
        ``lam = n * (scale * span) ** 4``.
        """
        width = max(scale, 1e-6) * (np.ptp(axis.x) or 1.0)
        return replace(self, lam=float(axis.n * width**4))

    def scale_of(self, axis: TimeAxis) -> float:
        """Invert the bandwidth-to-penalty map."""
        if self.lam is None:
            return 0.2
        span = float(np.ptp(axis.x)) or 1.0
        return float(min((self.lam / axis.n) ** 0.25 / span, 1.0))

    def params(self) -> dict[str, Any]:
        """Report the penalty."""
        return {"smoothing_parameter": self.lam}


# --------------------------------------------------------------------------
# Nonlinear smoothers: bootstrap required
# --------------------------------------------------------------------------


@register
@dataclass(frozen=True)
class InterpolatingSpline(Smoother):
    """Smoothing spline that chooses its own knots.

    ``UnivariateSpline`` places knots to satisfy a residual budget, so the
    fitted map depends on the data and no operator exists -- probing it misses
    by a factor of order one, not by rounding error. Uncertainty comes from
    the bootstrap.

    Attributes:
        function_order: Spline degree.
        s: Residual budget. Derived from the data when None.
    """

    name: ClassVar[str] = "spline"
    supported_orders: ClassVar[frozenset[int]] = frozenset({0, 1, 2, 3})

    function_order: int = 3
    s: float | None = None
    s_factor: float = 1.0

    def _budget(self, y: npt.NDArray[np.float64]) -> float:
        """Residual budget, derived from the noise level when not given."""
        if self.s is not None:
            return self.s
        return float(len(y) * np.var(np.diff(y)) * 0.1 * self.s_factor)

    def evaluate(
        self, axis: TimeAxis, y: npt.NDArray[np.float64], order: int
    ) -> Evaluation:
        """Fit the spline and differentiate it."""
        spline = UnivariateSpline(axis.x, y, k=self.function_order, s=self._budget(y))
        return Evaluation(
            values=np.asarray(spline(axis.x), dtype=np.float64),
            derivative=np.asarray(spline(axis.x, nu=order), dtype=np.float64),
        )

    def with_scale(self, scale: float, axis: TimeAxis) -> Self:
        """Scale the residual budget; larger budget means a smoother fit."""
        del axis
        return replace(self, s=None, s_factor=float(max(scale, 1e-6) * 5.0))

    def scale_of(self, axis: TimeAxis) -> float:
        """Approximate scale implied by the budget multiplier."""
        del axis
        return float(min(self.s_factor / 5.0, 1.0))

    def params(self) -> dict[str, Any]:
        """Report the spline degree and budget."""
        return {"function_order": self.function_order, "smoothing_parameter": self.s}


@register
@dataclass(frozen=True)
class L1TrendFilter(Smoother):
    """L1 trend filtering: piecewise-polynomial fit with sparse kinks.

    Solved by ADMM. The soft-thresholding step is nonlinear in the data -- that
    is the whole point, since it is what produces changepoints -- so
    uncertainty is bootstrapped.

    Note that the penalty's difference order and the derivative being reported
    are separate things; the previous implementation used one parameter for
    both.

    Attributes:
        lambda_param: Absolute penalty on the differences. Larger means fewer
            kinks. Ignored when ``lambda_fraction`` is set.
        lambda_fraction: Penalty as a fraction of the value above which the fit
            collapses to a plain polynomial. Being relative to the data is what
            makes it comparable across series; an absolute penalty is not.
        difference_order: Order of the penalized difference. Two gives a
            piecewise-linear trend, the usual choice.
        max_iter: ADMM iteration cap.
        tol: Convergence tolerance on the primal variable.
    """

    name: ClassVar[str] = "l1_filter"

    lambda_param: float = 1.0
    lambda_fraction: float | None = None
    difference_order: int = 2
    max_iter: int = 100
    tol: float = 1e-4

    def _penalty(
        self, differences: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> float:
        """Resolve the penalty, relative to the data when asked.

        The reference is ``max|D y|``, the magnitude of what the ADMM's
        soft-threshold actually compares against. The textbook normalizer is
        ``lambda_max = ||(D D')^-1 D y||_inf``, above which an exactly solved
        problem collapses to a plain polynomial -- but this solver runs at fixed
        rho and a capped iteration count, and measured against it the fit stops
        responding at roughly ``max|D y|``, two orders of magnitude below that.
        Normalizing by the theoretical value would map the entire scale knob
        into the saturated region, where every setting returns byte-identical
        output.

        An absolute penalty has no reference at all: the same number can leave
        one series untouched and flatten the next.
        """
        if self.lambda_fraction is None:
            return self.lambda_param
        reference = float(np.max(np.abs(differences @ y)))
        if not np.isfinite(reference) or reference <= 0:
            return self.lambda_param
        return max(self.lambda_fraction * reference, 1e-12)

    def evaluate(
        self, axis: TimeAxis, y: npt.NDArray[np.float64], order: int
    ) -> Evaluation:
        """Solve the L1 trend filtering problem, then difference the fit."""
        n = axis.n
        differences = np.diff(np.eye(n), n=self.difference_order, axis=0)
        differences = differences / (axis.delta**self.difference_order)
        penalty = self._penalty(differences, y)

        trend = y.copy()
        auxiliary = np.zeros(differences.shape[0])
        dual = np.zeros(differences.shape[0])
        rho = 1.0
        resolvent = np.linalg.inv(np.eye(n) + rho * differences.T @ differences)

        for _ in range(self.max_iter):
            previous = trend
            trend = resolvent @ (y + rho * differences.T @ (auxiliary - dual))
            shifted = differences @ trend + dual
            auxiliary = np.sign(shifted) * np.maximum(
                np.abs(shifted) - penalty / rho, 0.0
            )
            dual = dual + differences @ trend - auxiliary
            if np.linalg.norm(trend - previous) < self.tol:
                break

        derivative = np.gradient(trend, axis.x, edge_order=1)
        for _ in range(order - 1):
            derivative = np.gradient(derivative, axis.x, edge_order=1)
        return Evaluation(values=trend, derivative=derivative)

    def with_scale(self, scale: float, axis: TimeAxis) -> Self:
        """Set the penalty as a fraction of its saturating value."""
        del axis
        return replace(self, lambda_fraction=float(np.clip(scale, 1e-6, 1.0)))

    def scale_of(self, axis: TimeAxis) -> float:
        """The relative penalty is already a scale."""
        del axis
        return self.lambda_fraction if self.lambda_fraction is not None else 0.2

    def params(self) -> dict[str, Any]:
        """Report the penalty and difference order."""
        return {
            "lambda": (
                self.lambda_fraction
                if self.lambda_fraction is not None
                else self.lambda_param
            ),
            "function_order": self.difference_order,
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
