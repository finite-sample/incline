"""Smoothers that are probability models, and so already know their own variance.

A Gaussian process and a state-space model do not need to be probed or
bootstrapped: each carries a posterior covariance that *is* the answer. Asking
for it is both cheaper and more accurate than reconstructing it from outside.

The previous Gaussian process implementation did neither. It differenced the
posterior mean at a step of one hundredth of the sample spacing and then added
the two endpoint variances as though they were independent:

    dy_var = (std_plus**2 + std_minus**2) / (2 * dx) ** 2

Two points a thousandth of a length scale apart have posterior correlation
essentially one, so almost all of that variance cancels in reality and none of
it cancelled in the formula. The reported standard error came out 3,614 times
too large, the intervals swallowed every plausible value, and the significance
flag fired on 0% of a series with an obvious trend. Differentiating the kernel
instead is exact, has no step size, and costs one solve.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import numpy.typing as npt
from scipy.linalg import cho_solve
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel

from .smoothers import Evaluation, Smoother, register

if TYPE_CHECKING:
    from typing import Self

    from .axis import TimeAxis

# How many fitted models to remember. `fit` asks for the point estimate and the
# posterior separately, so without this every call fits twice.
_FIT_CACHE_LIMIT = 4
_FIT_CACHE: dict[Any, Any] = {}

# Mean-square differentiability of the Matern family: a process with smoothness
# nu has m derivatives exactly when nu > m. Asking for more is not a numerical
# difficulty, it is a derivative that does not exist.
MATERN_ORDERS = {"matern32": 1, "matern52": 2, "rbf": 3}


def _remember(key: Any, value: Any) -> Any:
    """Store a fitted model, bounding the cache."""
    if len(_FIT_CACHE) >= _FIT_CACHE_LIMIT:
        _FIT_CACHE.clear()
    _FIT_CACHE[key] = value
    return value


@dataclass(frozen=True)
class _KernelParts:
    """The pieces of a fitted kernel needed to differentiate it.

    Attributes:
        amplitude: Signal variance.
        length_scale: Characteristic length scale.
        noise: White-noise variance. Excluded from every derivative, because
            the derivative of interest belongs to the latent function, not to
            the measurement error.
        family: Which covariance function.
    """

    amplitude: float
    length_scale: float
    noise: float
    family: str


def _decompose(kernel: Any, family: str) -> _KernelParts:
    """Pull amplitude, length scale and noise out of a fitted kernel.

    Args:
        kernel: The fitted sklearn kernel.
        family: The requested kernel family.

    Returns:
        Its parts.

    Raises:
        ValueError: If the kernel is not the expected product-plus-noise shape.
    """
    try:
        signal, white = kernel.k1, kernel.k2
        return _KernelParts(
            amplitude=float(signal.k1.constant_value),
            length_scale=float(signal.k2.length_scale),
            noise=float(white.noise_level),
            family=family,
        )
    except AttributeError as exc:
        raise ValueError(
            f"Cannot differentiate kernel {kernel!r}: expected "
            f"Constant * Stationary + White"
        ) from exc


def _cross_derivative(
    parts: _KernelParts, r: npt.NDArray[np.float64], order: int
) -> npt.NDArray[np.float64]:
    """``d^order/dx^order`` of the covariance, as a function of ``r = x - x'``.

    Args:
        parts: The kernel's parameters.
        r: Signed separations.
        order: Derivative order.

    Returns:
        The derivative at each separation.

    Raises:
        ValueError: If the kernel family is unknown.
    """
    amplitude, length = parts.amplitude, parts.length_scale
    match parts.family:
        case "rbf":
            base = amplitude * np.exp(-0.5 * (r / length) ** 2)
            match order:
                case 0:
                    return base
                case 1:
                    return base * (-r / length**2)
                case 2:
                    return base * (r**2 / length**4 - 1 / length**2)
                case _:
                    raise ValueError(f"rbf derivative order {order} not implemented")
        case "matern32":
            a = np.sqrt(3.0) / length
            decay = np.exp(-a * np.abs(r))
            match order:
                case 0:
                    return amplitude * (1 + a * np.abs(r)) * decay
                case 1:
                    return -amplitude * a**2 * r * decay
                case _:
                    raise ValueError("matern32 supports first derivatives only")
        case "matern52":
            a = np.sqrt(5.0) / length
            decay = np.exp(-a * np.abs(r))
            match order:
                case 0:
                    return amplitude * (1 + a * np.abs(r) + a**2 * r**2 / 3) * decay
                case 1:
                    return -amplitude * (a**2 / 3) * r * (1 + a * np.abs(r)) * decay
                case 2:
                    return (
                        -amplitude
                        * (a**2 / 3)
                        * decay
                        * (1 + a * np.abs(r) - a**2 * r**2)
                    )
                case _:
                    raise ValueError("matern52 supports up to second derivatives")
        case _:
            raise ValueError(f"Unknown kernel family {parts.family!r}")


def _prior_derivative_variance(parts: _KernelParts, order: int) -> float:
    """``d^2order/dx^order dx'^order`` of the covariance at zero separation.

    This is the prior variance of the derivative -- the value the posterior
    variance decays toward away from the data.

    Args:
        parts: The kernel's parameters.
        order: Derivative order.

    Returns:
        The prior variance.

    Raises:
        ValueError: If the derivative does not exist for this kernel.
    """
    amplitude, length = parts.amplitude, parts.length_scale
    match (parts.family, order):
        case ("rbf", 0):
            return amplitude
        case ("rbf", 1):
            return amplitude / length**2
        case ("rbf", 2):
            return 3 * amplitude / length**4
        case ("matern32", 0) | ("matern52", 0):
            return amplitude
        case ("matern32", 1):
            return 3 * amplitude / length**2
        case ("matern52", 1):
            return 5 * amplitude / (3 * length**2)
        case ("matern52", 2):
            # d4k/dr4 at the origin. Matern-5/2 is twice mean-square
            # differentiable (nu > 2), so this limit exists, though the |r|^5
            # term makes a finite difference converge to it only slowly.
            return 25 * amplitude / length**4
        case _:
            raise ValueError(
                f"{parts.family} is not {order}-times mean-square "
                f"differentiable, so that derivative has no variance"
            )


@register
@dataclass(frozen=True)
class GaussianProcess(Smoother):
    """Gaussian process regression with an exact derivative posterior.

    Attributes:
        kernel: ``'rbf'``, ``'matern32'`` or ``'matern52'``. The Matern
            smoothness caps the derivative order: 3/2 is once differentiable,
            5/2 twice.
        amplitude: Prior variance of the signal. Optimized from a starting
            value of 1 when None.
        length_scale: Initial length scale. Optimized from the data when None.
        noise_level: Noise variance. A starting value when ``optimize`` is on,
            the value used when it is off.
        n_restarts: Restarts for the hyperparameter optimizer.
        optimize: Whether to fit the kernel hyperparameters by marginal
            likelihood. When False the kernel is used exactly as given.
        standardize: Whether to rescale the response before fitting. Decides
            what units ``amplitude`` and ``noise_level`` are in.

    Note:
        The three settings are separable and each does one thing: ``optimize``
        decides whether the hyperparameters are learned, ``standardize`` decides
        what units they are expressed in, and ``amplitude`` is one of them.

        With ``standardize=True`` the response is centered *and* scaled, so
        ``amplitude`` and ``noise_level`` are in units of the series' own
        standard deviation. That is the right default when they are being
        learned, and it keeps the fit well conditioned.

        With ``standardize=False`` the response is only centered, so both are in
        the data's own units and a prior can be stated exactly. Centering is
        kept either way because the prior mean is zero and real series are not;
        a constant shift does not affect any derivative. Scaling is what
        obscured the units: it made a supplied ``noise_level`` mean something
        other than what was asked for, and measured against data drawn from a
        known prior, credible intervals covered 0.90 instead of 0.95.

    Note:
        ``optimize`` exists because sklearn re-fits the hyperparameters on every
        call regardless of ``n_restarts_optimizer`` -- that setting controls how
        many restarts an optimization gets, not whether one happens. Passing a
        ``length_scale`` therefore only chose a starting point, and the fitted
        scale came out the same whatever was asked for, which silently made
        :meth:`with_scale` a no-op and a multi-scale sweep a single scale
        repeated. ``with_scale`` now turns optimization off so the scale it sets
        is the scale that gets used.
    """

    name: ClassVar[str] = "gp"
    has_native_posterior: ClassVar[bool] = True
    supported_orders: ClassVar[frozenset[int]] = frozenset({0, 1, 2})

    kernel: str = "rbf"
    amplitude: float | None = None
    length_scale: float | None = None
    noise_level: float = 0.1
    n_restarts: int = 5
    optimize: bool = True
    standardize: bool = True

    def _fitted(self, axis: TimeAxis, y: npt.NDArray[np.float64]) -> Any:
        """Fit the GP, reusing the fit when the same data comes back.

        Returns the fitted model together with the mean removed from the
        response, which the posterior has to add back for order zero.
        """
        key = (self, axis.key(), y.tobytes())
        cached = _FIT_CACHE.get(key)
        if cached is not None:
            return cached

        length = (
            self.length_scale
            if self.length_scale is not None
            else max(float(np.ptp(axis.x)) / 5.0, 1e-6)
        )
        signal = ConstantKernel(self.amplitude if self.amplitude is not None else 1.0)
        match self.kernel:
            case "rbf":
                shape = signal * RBF(length_scale=length)
            case "matern32":
                shape = signal * Matern(length_scale=length, nu=1.5)
            case "matern52":
                shape = signal * Matern(length_scale=length, nu=2.5)
            case _:
                raise ValueError(f"Unknown kernel type: {self.kernel}")

        # Without standardization the response is still centered by hand: the
        # prior mean is zero, so an uncentred series would be shrunk toward
        # zero rather than toward its own level. Only the scaling is dropped.
        offset = 0.0 if self.standardize else float(np.mean(y))

        model = GaussianProcessRegressor(
            kernel=shape + WhiteKernel(noise_level=self.noise_level),
            alpha=1e-10,
            n_restarts_optimizer=self.n_restarts if self.optimize else 0,
            # None is documented sklearn API for "keep the kernel fixed";
            # the inferred signature only admits str.
            optimizer="fmin_l_bfgs_b" if self.optimize else None,  # pyright: ignore[reportArgumentType]
            normalize_y=self.standardize,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(axis.x.reshape(-1, 1), y - offset)
        # Deliberately stashed on the sklearn estimator under an incline-
        # namespaced name so _posterior can recover the centring offset from
        # the cached model without a parallel cache keyed the same way.
        model._incline_offset = offset  # noqa: SLF001  # pyright: ignore[reportAttributeAccessIssue]
        return _remember(key, model)

    def _posterior(
        self, axis: TimeAxis, y: npt.NDArray[np.float64], order: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Posterior mean and standard deviation of the ``order``-th derivative.

        The derivative of a Gaussian process is a Gaussian process, with
        covariance given by differentiating the kernel. Both the mean and the
        variance follow from that, exactly.
        """
        model = self._fitted(axis, y)
        parts = _decompose(model.kernel_, self.kernel)

        if order > MATERN_ORDERS[self.kernel]:
            raise ValueError(
                f"kernel {self.kernel!r} supports derivative orders up to "
                f"{MATERN_ORDERS[self.kernel]}, got {order}"
            )

        train = model.X_train_.ravel()
        separation = axis.x[:, None] - train[None, :]
        cross = _cross_derivative(parts, separation, order)

        # With standardization the fit was on rescaled targets, so undo it.
        # sklearn stores this as a bare float when normalize_y is on but as
        # array([1.]) when it is off, and float() raises on an array.
        spread = float(np.ravel(np.asarray(getattr(model, "_y_train_std", 1.0)))[0])
        mean = (cross @ model.alpha_).ravel() * spread
        if order == 0:
            center = np.ravel(np.asarray(getattr(model, "_y_train_mean", 0.0)))[0]
            mean = mean + float(center) + float(getattr(model, "_incline_offset", 0.0))

        # Var = k''(0) - k'(x,X) K^-1 k'(X,x), using the stored Cholesky.
        solved = cho_solve((model.L_, True), cross.T)
        explained = np.einsum("ij,ji->i", cross, solved)
        variance = _prior_derivative_variance(parts, order) - explained
        return mean, np.sqrt(np.maximum(variance, 0.0)) * spread

    def evaluate(
        self, axis: TimeAxis, y: npt.NDArray[np.float64], order: int
    ) -> Evaluation:
        """Posterior mean of the smooth and of its derivative."""
        values, _ = self._posterior(axis, y, 0)
        derivative, _ = self._posterior(axis, y, order)
        return Evaluation(values=values, derivative=derivative)

    def native_posterior(
        self,
        axis: TimeAxis,
        y: npt.NDArray[np.float64],
        order: int,
        confidence_level: float,
    ) -> tuple[npt.NDArray[np.float64], None, None]:
        """Exact posterior standard deviation of the derivative."""
        del confidence_level
        _, standard_error = self._posterior(axis, y, order)
        return standard_error, None, None

    def with_scale(self, scale: float, axis: TimeAxis) -> Self:
        """Fix the length scale to ``scale`` of the series span.

        Optimization is switched off, or the scale just set would immediately
        be optimized away and every scale in a sweep would return the same fit.
        """
        span = float(np.ptp(axis.x)) or 1.0
        return replace(
            self,
            length_scale=max(scale, 1e-6) * span,
            n_restarts=0,
            optimize=False,
        )

    def scale_of(self, axis: TimeAxis) -> float:
        """Length scale as a fraction of the span."""
        span = float(np.ptp(axis.x)) or 1.0
        if self.length_scale is None:
            return 0.2
        return float(min(self.length_scale / span, 1.0))

    def params(self) -> dict[str, Any]:
        """Report the kernel and length scale."""
        return {"kernel": self.kernel, "length_scale": self.length_scale}


@register
@dataclass(frozen=True)
class StateSpace(Smoother):
    """Local linear trend, optionally with seasonal and damping.

    The slope is a state, so its smoothed variance is a diagonal entry of the
    smoother covariance and needs no extra machinery. This replaces a
    hand-rolled Kalman filter and a BFGS call whose inverse Hessian was
    discarded; statsmodels supplies both the state covariance and the
    hyperparameter covariance.

    Attributes:
        seasonal_periods: Length of a seasonal cycle, or None.

    Note:
        The intervals are **conditional on the fitted variances**. Those
        variances were estimated from the same data, and that estimation error
        is not propagated, so the intervals are somewhat narrow -- most visibly
        on short series.

        A correction for it was tried and removed. It scaled the interval by the
        relative standard error of each variance parameter, ``bse / |param|``,
        which is undefined at the boundary -- and variances land exactly on zero
        routinely, whenever a component is not needed. Measured over 40 fits, the
        median inflation factor was 1e5 and the maximum 3e8, turning a standard
        error of 0.017 into 4.6e6. A correction that can be eight orders of
        magnitude wrong is worse than the bias it was meant to remove.

        There is likewise no damped-trend option. The previous implementation
        accepted one and forwarded it to statsmodels, which has no such
        parameter and ignored it, so the setting did nothing.
    """

    name: ClassVar[str] = "kalman"
    has_native_posterior: ClassVar[bool] = True
    supported_orders: ClassVar[frozenset[int]] = frozenset({1})

    seasonal_periods: int | None = None

    def _fitted(self, axis: TimeAxis, y: npt.NDArray[np.float64]) -> Any:
        """Fit the unobserved-components model, reusing an identical fit."""
        key = (self, axis.key(), y.tobytes())
        cached = _FIT_CACHE.get(key)
        if cached is not None:
            return cached

        from statsmodels.tsa.statespace.structural import UnobservedComponents

        model = UnobservedComponents(
            y,
            # A model-name string is documented statsmodels API; the inferred
            # signature only admits the bool default.
            level="local linear trend",  # pyright: ignore[reportArgumentType]
            seasonal=self.seasonal_periods,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(disp=False)
        return _remember(key, result)

    def _slope(
        self, axis: TimeAxis, y: npt.NDArray[np.float64]
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """Smoothed level, slope, and the slope's standard error."""
        result = self._fitted(axis, y)
        states = result.smoothed_state
        covariance = result.smoothed_state_cov

        level = np.asarray(states[0], dtype=np.float64)
        # The slope state advances per observation; the axis may not.
        slope = np.asarray(states[1], dtype=np.float64) / axis.delta
        variance = np.asarray(covariance[1, 1, :], dtype=np.float64)
        standard_error = np.sqrt(np.maximum(variance, 0.0)) / axis.delta

        return level, slope, standard_error

    def evaluate(
        self, axis: TimeAxis, y: npt.NDArray[np.float64], order: int
    ) -> Evaluation:
        """Smoothed level and slope."""
        del order
        level, slope, _ = self._slope(axis, y)
        return Evaluation(values=level, derivative=slope)

    def native_posterior(
        self,
        axis: TimeAxis,
        y: npt.NDArray[np.float64],
        order: int,
        confidence_level: float,
    ) -> tuple[npt.NDArray[np.float64], None, None]:
        """Standard error of the smoothed slope state."""
        del order, confidence_level
        _, _, standard_error = self._slope(axis, y)
        return standard_error, None, None

    def with_scale(self, scale: float, axis: TimeAxis) -> Self:
        """No direct scale knob; smoothing follows from the fitted variances."""
        del scale, axis
        return self

    def scale_of(self, axis: TimeAxis) -> float:
        """Nominal scale, since smoothing is chosen by likelihood."""
        del axis
        return 0.2

    def params(self) -> dict[str, Any]:
        """Report the model configuration."""
        return {"seasonal_periods": self.seasonal_periods}
