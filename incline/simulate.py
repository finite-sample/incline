"""Synthetic series with known derivatives, for testing estimators against truth.

A standard error is a claim about repeated sampling, so checking one means
simulating from a trend whose derivative you already know. These generators
supply that ground truth.

``noise_std`` means the same thing for every noise type here: the **marginal**
standard deviation of the noise process. That is worth stating because it was
previously not true -- the AR(1) generator treated it as the innovation standard
deviation, so the noise it produced was a factor of ``1/sqrt(1 - phi**2)`` larger
than the white-noise generator's at the same setting (1.4x at phi=0.7, 2.3x at
phi=0.9), and the seasonal generator ignored the argument altogether. Comparing
methods across noise types under those definitions compared different noise
levels.

Every generator takes ``random_state`` and draws from its own Generator. Nothing
here touches the global numpy random state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd


if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "ExponentialTrend",
    "NoiseGenerator",
    "PolynomialTrend",
    "SinusoidalTrend",
    "StepTrend",
    "TrendFunction",
    "generate_time_series",
    "standard_test_functions",
]


class TrendFunction(ABC):
    """A trend whose derivatives are known in closed form."""

    @abstractmethod
    def __call__(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the trend.

        Args:
            x: Points to evaluate at.

        Returns:
            Trend values.
        """

    @abstractmethod
    def derivative(
        self, x: npt.NDArray[np.float64], order: int = 1
    ) -> npt.NDArray[np.float64]:
        """Evaluate a derivative of the trend.

        Args:
            x: Points to evaluate at.
            order: Derivative order.

        Returns:
            Derivative values.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """A short human-readable label."""


class PolynomialTrend(TrendFunction):
    """A polynomial trend.

    Attributes:
        coefficients: ``[a0, a1, a2, ...]`` for ``a0 + a1*x + a2*x**2 + ...``
    """

    def __init__(self, coefficients: Sequence[float]) -> None:
        """Store the coefficients in ascending power order."""
        self.coefficients = np.asarray(coefficients, dtype=np.float64)

    def __call__(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the polynomial."""
        return np.polyval(self.coefficients[::-1], np.asarray(x, dtype=np.float64))

    def derivative(
        self, x: npt.NDArray[np.float64], order: int = 1
    ) -> npt.NDArray[np.float64]:
        """Differentiate the polynomial ``order`` times."""
        x = np.asarray(x, dtype=np.float64)
        if order == 0:
            return self(x)
        # polyder works in descending order, which is why the reversals bracket it.
        derived = np.polyder(self.coefficients[::-1], order)
        if len(derived) == 0:
            return np.zeros_like(x)
        return np.polyval(derived, x)

    @property
    def name(self) -> str:
        """Label including the degree."""
        return f"Polynomial(deg={len(self.coefficients) - 1})"


class SinusoidalTrend(TrendFunction):
    """A sinusoid, whose derivatives cycle through sine and cosine.

    Attributes:
        amplitude: Peak amplitude.
        frequency: Cycles per unit x.
        phase: Phase offset in radians.
    """

    def __init__(
        self, amplitude: float = 1.0, frequency: float = 1.0, phase: float = 0.0
    ) -> None:
        """Store the sinusoid's parameters."""
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

    def _angle(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The sinusoid's argument at ``x``."""
        return 2 * np.pi * self.frequency * np.asarray(x, dtype=np.float64) + self.phase

    def __call__(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the sinusoid."""
        return self.amplitude * np.sin(self._angle(x))

    def derivative(
        self, x: npt.NDArray[np.float64], order: int = 1
    ) -> npt.NDArray[np.float64]:
        """Differentiate, using the period-4 cycle of sine's derivatives."""
        angle = self._angle(x)
        scale = self.amplitude * (2 * np.pi * self.frequency) ** order
        match order % 4:
            case 0:
                return scale * np.sin(angle)
            case 1:
                return scale * np.cos(angle)
            case 2:
                return -scale * np.sin(angle)
            case _:
                return -scale * np.cos(angle)

    @property
    def name(self) -> str:
        """Label including amplitude and frequency."""
        return f"Sinusoidal(A={self.amplitude}, f={self.frequency})"


class ExponentialTrend(TrendFunction):
    """Exponential growth or decay.

    Attributes:
        scale: Value at ``x = 0``.
        rate: Growth rate.
    """

    def __init__(self, scale: float = 1.0, rate: float = 0.1) -> None:
        """Store the scale and rate."""
        self.scale = scale
        self.rate = rate

    def __call__(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the exponential."""
        return self.scale * np.exp(self.rate * np.asarray(x, dtype=np.float64))

    def derivative(
        self, x: npt.NDArray[np.float64], order: int = 1
    ) -> npt.NDArray[np.float64]:
        """Every derivative is the function times a power of the rate."""
        return self.scale * (self.rate**order) * np.exp(self.rate * np.asarray(x))

    @property
    def name(self) -> str:
        """Label including scale and rate."""
        return f"Exponential(scale={self.scale}, rate={self.rate})"


class StepTrend(TrendFunction):
    """A piecewise-constant trend.

    Useful precisely because no smoother can represent it: the derivative is
    zero everywhere except at the jumps, where it does not exist, so every
    method trades a spike of bias for its smoothness.

    Attributes:
        breakpoints: Ascending x positions at which the value changes.
        values: The value on each segment; one more than the breakpoints, or
            equal in length to treat the last as extending to infinity.
    """

    def __init__(self, breakpoints: Sequence[float], values: Sequence[float]) -> None:
        """Store the segment boundaries and their values."""
        self.breakpoints = np.asarray(breakpoints, dtype=np.float64)
        self.values = np.asarray(values, dtype=np.float64)
        if len(self.values) < len(self.breakpoints):
            raise ValueError(
                f"need at least one value per breakpoint, got "
                f"{len(self.values)} values for {len(self.breakpoints)} breakpoints"
            )

    def __call__(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Select each point's segment value."""
        x = np.asarray(x, dtype=np.float64)
        # searchsorted maps each x to its segment index in one pass.
        segment = np.searchsorted(self.breakpoints, x, side="left")
        segment = np.clip(segment, 0, len(self.values) - 1)
        return self.values[segment]

    def derivative(
        self, x: npt.NDArray[np.float64], order: int = 1
    ) -> npt.NDArray[np.float64]:
        """Zero away from the jumps, where the derivative does not exist."""
        del order
        return np.zeros_like(np.asarray(x, dtype=np.float64))

    @property
    def name(self) -> str:
        """Label including the number of jumps."""
        return f"Step(n_breaks={len(self.breakpoints)})"


class NoiseGenerator:
    """Noise processes, all parameterized by their marginal standard deviation."""

    @staticmethod
    def white(
        n: int, std: float = 1.0, random_state: int | np.random.Generator | None = None
    ) -> npt.NDArray[np.float64]:
        """Independent Gaussian noise.

        Args:
            n: Number of points.
            std: Marginal standard deviation.
            random_state: Seed or Generator.

        Returns:
            The noise series.
        """
        return np.random.default_rng(random_state).normal(0, std, n)

    @staticmethod
    def ar1(
        n: int,
        phi: float = 0.7,
        std: float = 1.0,
        random_state: int | np.random.Generator | None = None,
    ) -> npt.NDArray[np.float64]:
        """First-order autoregressive noise, started in its stationary state.

        ``std`` is the marginal standard deviation, so the innovation variance
        is scaled by ``1 - phi**2`` and the first draw comes from the stationary
        distribution. Without both, the series would be a different size from
        white noise at the same ``std`` and would drift for its first few dozen
        points.

        Args:
            n: Number of points.
            phi: Autocorrelation at lag one, strictly inside (-1, 1).
            std: Marginal standard deviation.
            random_state: Seed or Generator.

        Returns:
            The noise series.

        Raises:
            ValueError: If ``phi`` is not inside (-1, 1).
        """
        if not -1.0 < phi < 1.0:
            raise ValueError(f"phi must be inside (-1, 1) for stationarity, got {phi}")
        rng = np.random.default_rng(random_state)
        innovation = std * np.sqrt(1 - phi**2)
        noise = np.empty(n, dtype=np.float64)
        if n == 0:
            return noise
        noise[0] = rng.normal(0, std)
        draws = rng.normal(0, innovation, n)
        for i in range(1, n):
            noise[i] = phi * noise[i - 1] + draws[i]
        return noise

    @staticmethod
    def seasonal(
        n: int,
        period: int = 12,
        std: float = 1.0,
        seasonal_fraction: float = 0.8,
        random_state: int | np.random.Generator | None = None,
    ) -> npt.NDArray[np.float64]:
        """A deterministic cycle plus white noise, scaled to ``std`` overall.

        Args:
            n: Number of points.
            period: Length of one cycle in observations.
            std: Marginal standard deviation of the combined series.
            seasonal_fraction: Share of the variance carried by the cycle.
            random_state: Seed or Generator.

        Returns:
            The noise series.
        """
        rng = np.random.default_rng(random_state)
        # A sine of amplitude a has variance a**2 / 2, hence the sqrt(2).
        seasonal_sd = std * np.sqrt(seasonal_fraction)
        residual_sd = std * np.sqrt(1 - seasonal_fraction)
        cycle = seasonal_sd * np.sqrt(2) * np.sin(2 * np.pi * np.arange(n) / period)
        return cycle + rng.normal(0, residual_sd, n)


def generate_time_series(
    trend_function: TrendFunction,
    n_points: int = 100,
    x_range: tuple[float, float] = (0.0, 10.0),
    noise_type: str = "white",
    noise_std: float = 0.1,
    irregular_spacing: bool = False,
    missing_data_prob: float = 0.0,
    random_state: int | np.random.Generator | None = None,
    **noise_kwargs: Any,
) -> tuple[pd.DataFrame, npt.NDArray[np.float64]]:
    """Simulate a series from a known trend.

    Args:
        trend_function: The true trend.
        n_points: Number of observations.
        x_range: Span of x values.
        noise_type: ``'white'``, ``'ar1'`` or ``'seasonal'``.
        noise_std: Marginal standard deviation of the noise, whichever type.
        irregular_spacing: Draw x uniformly rather than on a grid. The result
            then carries a numeric ``time`` column instead of a DatetimeIndex.
        missing_data_prob: Fraction of values blanked to NaN.
        random_state: Seed or Generator.
        **noise_kwargs: Extra arguments for the noise process, such as ``phi``
            for AR(1) or ``period`` for seasonal.

    Returns:
        Tuple of (frame with ``value``/``true_value``/``noise``, true first
        derivative at each point).

    Raises:
        ValueError: If the noise type is unknown.
    """
    rng = np.random.default_rng(random_state)

    if irregular_spacing:
        x = np.sort(rng.uniform(x_range[0], x_range[1], n_points))
    else:
        x = np.linspace(x_range[0], x_range[1], n_points)

    true_values = trend_function(x)
    true_derivative = trend_function.derivative(x, order=1)

    match noise_type:
        case "white":
            noise = NoiseGenerator.white(n_points, noise_std, rng)
        case "ar1":
            noise = NoiseGenerator.ar1(
                n_points, noise_kwargs.get("phi", 0.7), noise_std, rng
            )
        case "seasonal":
            noise = NoiseGenerator.seasonal(
                n_points,
                noise_kwargs.get("period", 12),
                noise_std,
                noise_kwargs.get("seasonal_fraction", 0.8),
                rng,
            )
        case _:
            raise ValueError(
                f"Unknown noise type {noise_type!r}; use 'white', 'ar1' or 'seasonal'"
            )

    observed = true_values + noise

    if irregular_spacing:
        frame = pd.DataFrame(
            {
                "time": x,
                "value": observed,
                "true_value": true_values,
                "noise": noise,
            }
        )
    else:
        frame = pd.DataFrame(
            {"value": observed, "true_value": true_values, "noise": noise},
            index=pd.date_range("2020-01-01", periods=n_points, freq="D"),
        )

    if missing_data_prob > 0:
        frame.loc[rng.random(n_points) < missing_data_prob, "value"] = np.nan

    return frame, true_derivative


def standard_test_functions() -> list[TrendFunction]:
    """A spread of trends covering the shapes estimators handle differently.

    Returns:
        Linear, quadratic, sinusoidal, exponential and step trends.
    """
    return [
        PolynomialTrend([0.0, 1.0]),
        PolynomialTrend([0.0, 0.0, 0.5]),
        SinusoidalTrend(amplitude=2.0, frequency=0.1),
        ExponentialTrend(scale=1.0, rate=0.2),
        StepTrend(breakpoints=[3.0, 7.0], values=[1.0, 3.0, 2.0]),
    ]
