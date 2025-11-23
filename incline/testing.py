"""Comprehensive simulation and testing framework for trend estimation methods.

This module provides tools for generating synthetic time series with known
derivatives and benchmarking different trend estimation methods.
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# Check for optional dependencies
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from scipy import stats

    HAS_SCIPY_STATS = True
except ImportError:
    HAS_SCIPY_STATS = False


@dataclass
class SimulationResult:
    """Container for simulation results."""

    method: str
    mse_derivative: float
    bias_derivative: float
    coverage_95: float
    coverage_90: float
    mean_ci_width: float
    computation_time: float
    parameters: dict[str, Any]


class TrendFunction(ABC):
    """Abstract base class for trend functions with known derivatives."""

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the function at points x."""
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray, order: int = 1) -> np.ndarray:
        """Evaluate the derivative at points x."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the function."""
        pass


class PolynomialTrend(TrendFunction):
    """Polynomial trend function."""

    def __init__(self, coefficients: list[float]):
        """Initialize PolynomialTrend.

        Parameters
        ----------
        coefficients : list
            Polynomial coefficients [a0, a1, a2, ...] for a0 + a1*x + a2*x^2 + ...
        """
        self.coefficients = np.array(coefficients)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.polyval(self.coefficients[::-1], x)

    def derivative(self, x: np.ndarray, order: int = 1) -> np.ndarray:
        if order == 0:
            return self(x)

        # Start with original coefficients [a0, a1, a2, ...] for a0 + a1*x + a2*x^2 + ...
        deriv_coeffs = self.coefficients.copy()

        # Apply derivative operation 'order' times
        for _ in range(order):
            if len(deriv_coeffs) <= 1:
                return np.zeros_like(x)

            # For polynomial a0 + a1*x + a2*x^2 + ..., derivative is a1 + 2*a2*x + ...
            # Multiply each coefficient by its power and remove constant term
            new_coeffs = []
            for i in range(1, len(deriv_coeffs)):
                new_coeffs.append(deriv_coeffs[i] * i)
            deriv_coeffs = np.array(new_coeffs)

        if len(deriv_coeffs) == 0:
            return np.zeros_like(x)

        return np.polyval(deriv_coeffs[::-1], x)

    @property
    def name(self) -> str:
        return f"Polynomial(deg={len(self.coefficients) - 1})"


class SinusoidalTrend(TrendFunction):
    """Sinusoidal trend function."""

    def __init__(
        self, amplitude: float = 1.0, frequency: float = 1.0, phase: float = 0.0
    ):
        """Initialize SinusoidalTrend.

        Parameters
        ----------
        amplitude : float
            Amplitude of the sinusoid
        frequency : float
            Frequency of the sinusoid
        phase : float
            Phase shift of the sinusoid
        """
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.amplitude * np.sin(2 * np.pi * self.frequency * x + self.phase)

    def derivative(self, x: np.ndarray, order: int = 1) -> np.ndarray:
        if order == 1:
            return (
                self.amplitude
                * 2
                * np.pi
                * self.frequency
                * np.cos(2 * np.pi * self.frequency * x + self.phase)
            )
        elif order == 2:
            return (
                -self.amplitude
                * (2 * np.pi * self.frequency) ** 2
                * np.sin(2 * np.pi * self.frequency * x + self.phase)
            )
        else:
            # General case using pattern
            factor = (2 * np.pi * self.frequency) ** order
            if order % 4 == 0:
                return (
                    self.amplitude
                    * factor
                    * np.sin(2 * np.pi * self.frequency * x + self.phase)
                )
            elif order % 4 == 1:
                return (
                    self.amplitude
                    * factor
                    * np.cos(2 * np.pi * self.frequency * x + self.phase)
                )
            elif order % 4 == 2:
                return (
                    -self.amplitude
                    * factor
                    * np.sin(2 * np.pi * self.frequency * x + self.phase)
                )
            else:  # order % 4 == 3
                return (
                    -self.amplitude
                    * factor
                    * np.cos(2 * np.pi * self.frequency * x + self.phase)
                )

    @property
    def name(self) -> str:
        return f"Sinusoidal(A={self.amplitude}, f={self.frequency})"


class ExponentialTrend(TrendFunction):
    """Exponential trend function."""

    def __init__(self, scale: float = 1.0, rate: float = 0.1):
        """Initialize ExponentialTrend.

        Parameters
        ----------
        scale : float
            Scale factor
        rate : float
            Exponential growth rate
        """
        self.scale = scale
        self.rate = rate

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.scale * np.exp(self.rate * x)

    def derivative(self, x: np.ndarray, order: int = 1) -> np.ndarray:
        return self.scale * (self.rate**order) * np.exp(self.rate * x)

    @property
    def name(self) -> str:
        return f"Exponential(scale={self.scale}, rate={self.rate})"


class StepTrend(TrendFunction):
    """Piecewise constant (step) trend function."""

    def __init__(self, breakpoints: list[float], values: list[float]):
        """Initialize StepTrend.

        Parameters
        ----------
        breakpoints : list[float]
            X-coordinates where the function changes value
        values : list[float]
            Values for each piecewise constant segment
        """
        self.breakpoints = np.array(breakpoints)
        self.values = np.array(values)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        result = np.zeros_like(x)
        for i, (bp, val) in enumerate(zip(self.breakpoints, self.values, strict=False)):
            if i == 0:
                mask = x <= bp
            else:
                mask = (x > self.breakpoints[i - 1]) & (x <= bp)
            result[mask] = val

        # Handle points beyond last breakpoint
        if len(x) > 0:
            result[x > self.breakpoints[-1]] = self.values[-1]

        return result

    def derivative(self, x: np.ndarray, order: int = 1) -> np.ndarray:
        if order == 1:
            # Derivative is zero except at breakpoints (where it's undefined)
            return np.zeros_like(x)
        else:
            return np.zeros_like(x)

    @property
    def name(self) -> str:
        return f"Step(n_breaks={len(self.breakpoints)})"


class NoiseGenerator:
    """Generator for different types of noise processes."""

    @staticmethod
    def white_noise(
        n: int, std: float = 1.0, random_state: int | None = None
    ) -> np.ndarray:
        """Generate white noise."""
        if random_state is not None:
            np.random.seed(random_state)
        return np.random.normal(0, std, n)

    @staticmethod
    def ar1_noise(
        n: int, phi: float = 0.7, std: float = 1.0, random_state: int | None = None
    ) -> np.ndarray:
        """Generate AR(1) noise."""
        if random_state is not None:
            np.random.seed(random_state)

        errors = np.random.normal(0, std, n)
        y = np.zeros(n)
        y[0] = errors[0]

        for i in range(1, n):
            y[i] = phi * y[i - 1] + errors[i]

        return y

    @staticmethod
    def seasonal_noise(
        n: int,
        period: int = 12,
        amplitude: float = 1.0,
        random_state: int | None = None,
    ) -> np.ndarray:
        """Generate noise with seasonal component."""
        if random_state is not None:
            np.random.seed(random_state)

        t = np.arange(n)
        seasonal = amplitude * np.sin(2 * np.pi * t / period)
        white = np.random.normal(0, 0.5, n)

        return seasonal + white


def generate_time_series(
    trend_function: TrendFunction,
    n_points: int = 100,
    x_range: tuple[float, float] = (0, 10),
    noise_type: str = "white",
    noise_std: float = 0.1,
    irregular_spacing: bool = False,
    missing_data_prob: float = 0.0,
    random_state: int | None = None,
    **noise_kwargs,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate synthetic time series with known trend and derivative.

    Parameters
    ----------
    trend_function : TrendFunction
        Function defining the true trend
    n_points : int
        Number of time points
    x_range : tuple
        Range of x values (start, end)
    noise_type : str
        Type of noise ('white', 'ar1', 'seasonal')
    noise_std : float
        Standard deviation of noise
    irregular_spacing : bool
        Whether to use irregular time spacing
    missing_data_prob : float
        Probability of missing data points
    random_state : int, optional
        Random seed for reproducibility
    **noise_kwargs
        Additional arguments for noise generation

    Returns:
    -------
    df : pd.DataFrame
        Generated time series data
    true_derivative : np.ndarray
        True derivative values
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate time points
    if irregular_spacing:
        # Generate irregular spacing with some structure
        x = np.sort(np.random.uniform(x_range[0], x_range[1], n_points))
    else:
        x = np.linspace(x_range[0], x_range[1], n_points)

    # Generate true function values and derivatives
    true_values = trend_function(x)
    true_derivative = trend_function.derivative(x, order=1)

    # Generate noise
    if noise_type == "white":
        noise = NoiseGenerator.white_noise(n_points, noise_std, random_state)
    elif noise_type == "ar1":
        phi = noise_kwargs.get("phi", 0.7)
        noise = NoiseGenerator.ar1_noise(n_points, phi, noise_std, random_state)
    elif noise_type == "seasonal":
        period = noise_kwargs.get("period", 12)
        amplitude = noise_kwargs.get("amplitude", noise_std)
        noise = NoiseGenerator.seasonal_noise(n_points, period, amplitude, random_state)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    # Add noise to function values
    observed_values = true_values + noise

    # Create time index
    if irregular_spacing:
        # Use x values directly
        time_index = x
        time_column = "time"
    else:
        # Create datetime index
        time_index = pd.date_range("2020-01-01", periods=n_points, freq="D")
        time_column = None

    # Create DataFrame
    if time_column:
        df = pd.DataFrame(
            {
                "time": time_index,
                "value": observed_values,
                "true_value": true_values,
                "noise": noise,
            }
        )
    else:
        df = pd.DataFrame(
            {"value": observed_values, "true_value": true_values, "noise": noise},
            index=time_index,
        )

    # Introduce missing data
    if missing_data_prob > 0:
        missing_mask = np.random.random(n_points) < missing_data_prob
        df.loc[missing_mask, "value"] = np.nan

    return df, true_derivative


def benchmark_method(
    method_name: str,
    method_function: Callable,
    df: pd.DataFrame,
    true_derivative: np.ndarray,
    time_column: str | None = None,
    confidence_level: float = 0.95,
    **method_kwargs,
) -> SimulationResult:
    """Benchmark a single trend estimation method.

    Parameters
    ----------
    method_name : str
        Name of the method
    method_function : callable
        Function that estimates trends
    df : pd.DataFrame
        Time series data
    true_derivative : np.ndarray
        True derivative values
    time_column : str, optional
        Time column name
    confidence_level : float
        Confidence level for coverage testing
    **method_kwargs
        Additional arguments for the method

    Returns:
    -------
    SimulationResult
        Benchmark results
    """
    import time

    # Time the method
    start_time = time.time()

    try:
        if time_column:
            result = method_function(df, time_column=time_column, **method_kwargs)
        else:
            result = method_function(df, **method_kwargs)

        computation_time = time.time() - start_time

        # Extract derivatives
        estimated_derivative = result["derivative_value"].values

        # Remove NaN values for comparison
        valid_mask = ~(np.isnan(estimated_derivative) | np.isnan(true_derivative))
        if not np.any(valid_mask):
            raise ValueError("No valid derivative estimates")

        est_valid = estimated_derivative[valid_mask]
        true_valid = true_derivative[valid_mask]

        # Calculate MSE and bias
        mse = np.mean((est_valid - true_valid) ** 2)
        bias = np.mean(est_valid - true_valid)

        # Check for confidence intervals
        coverage_95 = np.nan
        coverage_90 = np.nan
        mean_ci_width = np.nan

        if (
            "derivative_ci_lower" in result.columns
            and "derivative_ci_upper" in result.columns
        ):
            ci_lower = result["derivative_ci_lower"].values[valid_mask]
            ci_upper = result["derivative_ci_upper"].values[valid_mask]

            # Calculate coverage
            in_ci_95 = (true_valid >= ci_lower) & (true_valid <= ci_upper)
            coverage_95 = np.mean(in_ci_95)

            # For 90% CI (approximate from 95% CI)
            ci_width = ci_upper - ci_lower
            # This is 1, but keeping structure
            stats.norm.ppf(0.95) / stats.norm.ppf(0.95)
            coverage_90 = coverage_95  # Simplified for now

            mean_ci_width = np.mean(ci_width)

        return SimulationResult(
            method=method_name,
            mse_derivative=mse,
            bias_derivative=bias,
            coverage_95=coverage_95,
            coverage_90=coverage_90,
            mean_ci_width=mean_ci_width,
            computation_time=computation_time,
            parameters=method_kwargs,
        )

    except Exception as e:
        warnings.warn(f"Method {method_name} failed: {e}", stacklevel=2)
        return SimulationResult(
            method=method_name,
            mse_derivative=np.inf,
            bias_derivative=np.inf,
            coverage_95=np.nan,
            coverage_90=np.nan,
            mean_ci_width=np.nan,
            computation_time=np.inf,
            parameters=method_kwargs,
        )


def run_comprehensive_benchmark(
    trend_functions: list[TrendFunction],
    noise_types: list[str] | None = None,
    noise_levels: list[float] | None = None,
    n_points_list: list[int] | None = None,
    n_replications: int = 50,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Run comprehensive benchmark across multiple scenarios.

    Parameters
    ----------
    trend_functions : list
        List of TrendFunction objects to test
    noise_types : list
        Types of noise to test
    noise_levels : list
        Noise standard deviations to test
    n_points_list : list
        Sample sizes to test
    n_replications : int
        Number of replications per scenario
    random_state : int, optional
        Random seed

    Returns:
    -------
    pd.DataFrame
        Comprehensive benchmark results
    """
    from .advanced import l1_trend_filter, loess_trend
    from .trend import naive_trend, sgolay_trend, spline_trend

    # Define methods to test
    if n_points_list is None:
        n_points_list = [50, 100, 200]
    if noise_levels is None:
        noise_levels = [0.1, 0.5, 1.0]
    if noise_types is None:
        noise_types = ["white", "ar1"]
    methods = {
        "naive": naive_trend,
        "spline": spline_trend,
        "sgolay": sgolay_trend,
    }

    # Add advanced methods if available
    try:
        methods["loess"] = loess_trend
        methods["l1_filter"] = l1_trend_filter
    except ImportError:
        pass

    results = []

    # Set up scenarios
    scenarios = []
    for trend_func in trend_functions:
        for noise_type in noise_types:
            for noise_level in noise_levels:
                for n_points in n_points_list:
                    scenarios.append(
                        {
                            "trend_function": trend_func,
                            "noise_type": noise_type,
                            "noise_level": noise_level,
                            "n_points": n_points,
                        }
                    )

    len(scenarios) * len(methods) * n_replications
    scenario_count = 0

    for scenario in scenarios:
        for method_name, method_func in methods.items():
            for rep in range(n_replications):
                scenario_count += 1

                if scenario_count % 100 == 0:
                    pass

                # Generate data
                rep_seed = random_state + rep if random_state else None
                df, true_deriv = generate_time_series(
                    trend_function=scenario["trend_function"],
                    n_points=scenario["n_points"],
                    noise_type=scenario["noise_type"],
                    noise_std=scenario["noise_level"],
                    random_state=rep_seed,
                )

                # Determine method parameters
                method_kwargs = {}
                if method_name == "sgolay":
                    # Adjust window size for smaller datasets
                    window_length = min(15, scenario["n_points"] // 3)
                    if window_length % 2 == 0:
                        window_length -= 1
                    method_kwargs["window_length"] = max(5, window_length)

                # Benchmark method
                result = benchmark_method(
                    method_name, method_func, df, true_deriv, **method_kwargs
                )

                # Store results
                result_dict = {
                    "trend_function": scenario["trend_function"].name,
                    "noise_type": scenario["noise_type"],
                    "noise_level": scenario["noise_level"],
                    "n_points": scenario["n_points"],
                    "replication": rep,
                    "method": result.method,
                    "mse_derivative": result.mse_derivative,
                    "bias_derivative": result.bias_derivative,
                    "coverage_95": result.coverage_95,
                    "computation_time": result.computation_time,
                }

                results.append(result_dict)

    return pd.DataFrame(results)


def plot_benchmark_results(results_df: pd.DataFrame, metric: str = "mse_derivative"):
    """Plot benchmark results."""
    if not HAS_MATPLOTLIB:
        return

    # Aggregate results
    agg_results = (
        results_df.groupby(
            ["trend_function", "noise_type", "noise_level", "n_points", "method"]
        )[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Create plots
    _fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    # Plot 1: MSE vs noise level
    for i, trend_func in enumerate(agg_results["trend_function"].unique()):
        if i >= len(axes):
            break

        subset = agg_results[agg_results["trend_function"] == trend_func]

        for method in subset["method"].unique():
            method_data = subset[subset["method"] == method]
            axes[i].errorbar(
                method_data["noise_level"],
                method_data["mean"],
                yerr=method_data["std"],
                label=method,
                marker="o",
            )

        axes[i].set_xlabel("Noise Level")
        axes[i].set_ylabel(metric)
        axes[i].set_title(f"{trend_func}")
        axes[i].legend()
        axes[i].set_yscale("log")

    plt.tight_layout()
    plt.show()


# Predefined test cases
def get_standard_test_functions() -> list[TrendFunction]:
    """Get a standard set of test functions for benchmarking."""
    return [
        PolynomialTrend([0, 1]),  # Linear
        PolynomialTrend([0, 0, 1]),  # Quadratic
        PolynomialTrend([0, 1, 0, 0.1]),  # Cubic
        SinusoidalTrend(amplitude=2.0, frequency=0.5),  # Smooth sinusoid
        ExponentialTrend(scale=1.0, rate=0.05),  # Exponential growth
        StepTrend([2, 5, 8], [1, 3, 2, 4]),  # Step function
    ]
