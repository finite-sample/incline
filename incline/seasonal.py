"""Seasonal decomposition and detrending for trend estimation.

This module provides tools for separating seasonal components from time series
before trend estimation, which is crucial for obtaining meaningful derivatives
in data with strong seasonal patterns.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd


# Check for optional dependencies
try:
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import acf

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from scipy import signal
    from scipy.fft import fft, fftfreq

    HAS_SCIPY_SIGNAL = True
except ImportError:
    HAS_SCIPY_SIGNAL = False


def detect_seasonality(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    max_period: int | None = None,
) -> dict[str, Any]:
    """Detect seasonal patterns in time series data.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data
    column_value : str
        Value column name
    time_column : str, optional
        Time column name
    max_period : int, optional
        Maximum period to search for seasonality

    Returns:
    -------
    dict
        Dictionary with seasonality information:
        - 'seasonal': bool, whether seasonality detected
        - 'period': int, estimated seasonal period
        - 'strength': float, strength of seasonality (0-1)
        - 'method': str, detection method used
    """
    y = df[column_value].values
    n = len(y)

    if max_period is None:
        max_period = min(n // 3, 365)  # Up to yearly for daily data

    # Method 1: Autocorrelation-based detection
    if HAS_STATSMODELS and n > 2 * max_period:
        try:
            # Compute autocorrelation function
            autocorr = acf(y, nlags=max_period, fft=True)

            # Find peaks in autocorrelation (excluding lag 0)
            peaks = []
            for lag in range(2, len(autocorr)):
                if (
                    lag < len(autocorr) - 1
                    and autocorr[lag] > autocorr[lag - 1]
                    and autocorr[lag] > autocorr[lag + 1]
                    and autocorr[lag] > 0.3
                ):  # Threshold for significance
                    peaks.append((lag, autocorr[lag]))

            if peaks:
                # Take the strongest peak
                best_period, strength = max(peaks, key=lambda x: x[1])
                return {
                    "seasonal": True,
                    "period": best_period,
                    "strength": strength,
                    "method": "autocorrelation",
                }
        except Exception as e:
            warnings.warn(
                f"Autocorrelation seasonality detection failed: {e}", stacklevel=2
            )

    # Method 2: FFT-based spectral analysis
    if HAS_SCIPY_SIGNAL and n > 20:
        try:
            # Remove trend first
            y_detrended = signal.detrend(y)

            # Apply FFT
            fft_vals = fft(y_detrended)
            freqs = fftfreq(n)

            # Find dominant frequency (excluding DC component)
            power = np.abs(fft_vals[1 : n // 2])
            max_freq_idx = np.argmax(power) + 1

            if freqs[max_freq_idx] > 0:
                period = int(1 / freqs[max_freq_idx])
                if 2 <= period <= max_period:
                    # Estimate strength from relative power
                    total_power = np.sum(power)
                    peak_power = power[max_freq_idx - 1]
                    strength = peak_power / total_power

                    if strength > 0.1:  # Threshold for significance
                        return {
                            "seasonal": True,
                            "period": period,
                            "strength": strength,
                            "method": "fft",
                        }
        except Exception as e:
            warnings.warn(f"FFT seasonality detection failed: {e}", stacklevel=2)

    # Method 3: Simple variance-based detection
    # Check if grouping by potential periods reduces variance
    best_reduction = 0
    best_period_var: int | None = None
    original_var = np.var(y)

    for period in range(2, min(max_period + 1, n // 3)):
        try:
            # Group by period and compute within-group variance
            groups = [y[i::period] for i in range(period)]
            within_var = np.mean(
                [np.var(group) if len(group) > 1 else original_var for group in groups]
            )
            variance_reduction = 1 - within_var / original_var

            if variance_reduction > best_reduction:
                best_reduction = variance_reduction
                best_period_var = period
        except Exception as e:
            warnings.warn(
                f"Variance-based seasonality check failed for period {period}: {e}",
                stacklevel=2,
            )
            continue

    if best_reduction > 0.3:  # Threshold for significance
        return {
            "seasonal": True,
            "period": best_period_var,
            "strength": best_reduction,
            "method": "variance",
        }

    # No significant seasonality detected
    return {"seasonal": False, "period": None, "strength": 0.0, "method": "none"}


def stl_decompose(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    period: int | None = None,
    seasonal: int = 7,
    trend: int | None = None,
    robust: bool = True,
) -> pd.DataFrame:
    """Perform STL (Seasonal and Trend decomposition using Loess) decomposition.

    STL is a versatile and robust method for decomposing time series into
    seasonal, trend, and remainder components.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data
    column_value : str
        Value column name
    time_column : str, optional
        Time column name
    period : int, optional
        Seasonal period (auto-detected if None)
    seasonal : int
        Length of seasonal smoother (odd number)
    trend : int, optional
        Length of trend smoother
    robust : bool
        Use robust fitting

    Returns:
    -------
    pd.DataFrame
        Original data plus trend, seasonal, and residual components
    """
    if not HAS_STATSMODELS:
        raise ImportError(
            "statsmodels is required for STL decomposition. "
            "Install with: pip install statsmodels"
        )

    y = df[column_value].values
    n = len(y)

    # Auto-detect period if not provided
    if period is None:
        seasonality_info = detect_seasonality(df, column_value, time_column)
        if seasonality_info["seasonal"]:
            period = seasonality_info["period"]
        else:
            # Assume no strong seasonality, use a default
            period = min(12, n // 4)  # Monthly for shorter series

    if period < 2 or period >= n // 2:
        warnings.warn(
            f"Invalid period {period}, using simple trend estimation", stacklevel=2
        )
        # Fall back to simple trend estimation
        from .trend import spline_trend

        return spline_trend(df, column_value, time_column)

    # Ensure seasonal smoother is appropriate
    if seasonal % 2 == 0:
        seasonal += 1
    seasonal = max(seasonal, period + (period % 2 == 0))

    # Set trend smoother if not provided
    if trend is None:
        trend = int(1.5 * period / (1 - 1.5 / seasonal))
        if trend % 2 == 0:
            trend += 1

    try:
        # Create pandas Series with appropriate index
        if isinstance(df.index, pd.DatetimeIndex):
            ts = pd.Series(y, index=df.index)
        else:
            ts = pd.Series(y)

        # Perform STL decomposition
        stl = STL(ts, seasonal=seasonal, trend=trend, period=period, robust=robust)
        decomposition = stl.fit()

        # Create output dataframe
        odf = df.copy()
        odf["trend_component"] = decomposition.trend.values
        odf["seasonal_component"] = decomposition.seasonal.values
        odf["residual_component"] = decomposition.resid.values
        odf["deseasonalized"] = decomposition.trend.values + decomposition.resid.values

        # Add decomposition metadata
        odf["period"] = period
        odf["decomposition_method"] = "stl"

        return odf

    except Exception as e:
        warnings.warn(
            f"STL decomposition failed: {e}, using fallback method", stacklevel=2
        )
        return simple_deseasonalize(df, column_value, time_column, period)


def simple_deseasonalize(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    period: int | None = None,
) -> pd.DataFrame:
    """Simple seasonal adjustment using moving averages.

    This is a fallback method when statsmodels is not available or
    STL decomposition fails.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data
    column_value : str
        Value column name
    time_column : str, optional
        Time column name
    period : int, optional
        Seasonal period (auto-detected if None)

    Returns:
    -------
    pd.DataFrame
        Original data plus deseasonalized components
    """
    y = df[column_value].values
    n = len(y)

    # Auto-detect period if not provided
    if period is None:
        seasonality_info = detect_seasonality(df, column_value, time_column)
        if seasonality_info["seasonal"]:
            period = seasonality_info["period"]
        else:
            period = min(12, n // 4)

    if period < 2 or period >= n // 2:
        # No meaningful seasonality
        odf = df.copy()
        odf["trend_component"] = y
        odf["seasonal_component"] = np.zeros(n)
        odf["residual_component"] = np.zeros(n)
        odf["deseasonalized"] = y
        odf["period"] = period
        odf["decomposition_method"] = "none"
        return odf

    # Estimate trend using centered moving average
    window = period
    if window % 2 == 0:
        # For even periods, use weighted average
        trend = np.full(n, np.nan)
        half_window = window // 2

        for i in range(half_window, n - half_window):
            # Weighted moving average for even periods
            start_idx = i - half_window
            end_idx = i + half_window
            data_slice = y[start_idx : end_idx + 1]
            weights = np.ones(len(data_slice))
            weights[0] = weights[-1] = 0.5
            trend[i] = np.average(data_slice, weights=weights)
    else:
        # Simple centered moving average for odd periods
        trend = pd.Series(y).rolling(window=window, center=True).mean().values

    # Fill in missing trend values at edges
    trend[: period // 2] = trend[period // 2]
    trend[-(period // 2) :] = trend[-(period // 2)]

    # Estimate seasonal component
    detrended = y - trend
    seasonal = np.zeros(n)

    # Average each seasonal position
    for season_idx in range(period):
        indices = np.arange(season_idx, n, period)
        if len(indices) > 0:
            seasonal_mean = np.nanmean(detrended[indices])
            seasonal[indices] = seasonal_mean

    # Ensure seasonal component sums to zero over each period
    for start_idx in range(0, n - period + 1, period):
        end_idx = start_idx + period
        seasonal_period = seasonal[start_idx:end_idx]
        adjustment = np.mean(seasonal_period)
        seasonal[start_idx:end_idx] -= adjustment

    # Compute residuals
    residual = y - trend - seasonal
    deseasonalized = y - seasonal

    # Create output dataframe
    odf = df.copy()
    odf["trend_component"] = trend
    odf["seasonal_component"] = seasonal
    odf["residual_component"] = residual
    odf["deseasonalized"] = deseasonalized
    odf["period"] = period
    odf["decomposition_method"] = "simple"

    return odf


def trend_with_deseasonalization(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    trend_method: str = "spline",
    decomposition_method: str = "auto",
    period: int | None = None,
    **trend_kwargs,
) -> pd.DataFrame:
    """Estimate trends after seasonal decomposition.

    This function first removes seasonal components, then estimates trends
    on the deseasonalized data. This produces more reliable trend estimates
    for seasonal time series.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data
    column_value : str
        Value column name
    time_column : str, optional
        Time column name
    trend_method : str
        Trend estimation method ('spline', 'sgolay', 'loess', etc.)
    decomposition_method : str
        Decomposition method ('auto', 'stl', 'simple')
    period : int, optional
        Seasonal period (auto-detected if None)
    **trend_kwargs
        Additional arguments for trend estimation

    Returns:
    -------
    pd.DataFrame
        Results with seasonal decomposition and trend estimates
    """
    # First, check if seasonality exists
    seasonality_info = detect_seasonality(df, column_value, time_column)

    if not seasonality_info["seasonal"] and decomposition_method == "auto":
        # No seasonality detected, proceed with direct trend estimation
        from .advanced import estimate_trend, loess_trend
        from .trend import sgolay_trend, spline_trend

        if trend_method == "spline":
            trend_result = spline_trend(df, column_value, time_column, **trend_kwargs)
        elif trend_method == "sgolay":
            trend_result = sgolay_trend(df, column_value, time_column, **trend_kwargs)
        elif trend_method == "loess":
            trend_result = loess_trend(df, column_value, time_column, **trend_kwargs)
        else:
            trend_result = estimate_trend(
                df, column_value, time_column, trend_method, **trend_kwargs
            )

        # Add seasonality information
        trend_result["seasonality_detected"] = seasonality_info["seasonal"]
        trend_result["seasonality_strength"] = seasonality_info["strength"]
        return trend_result

    # Perform seasonal decomposition
    if decomposition_method == "auto":
        if HAS_STATSMODELS:
            decomp_result = stl_decompose(df, column_value, time_column, period)
        else:
            decomp_result = simple_deseasonalize(df, column_value, time_column, period)
    elif decomposition_method == "stl":
        decomp_result = stl_decompose(df, column_value, time_column, period)
    elif decomposition_method == "simple":
        decomp_result = simple_deseasonalize(df, column_value, time_column, period)
    else:
        raise ValueError(f"Unknown decomposition method: {decomposition_method}")

    # Estimate trend on deseasonalized data
    deseasonalized_df = df.copy()
    deseasonalized_df[column_value] = decomp_result["deseasonalized"]

    from .trend import sgolay_trend, spline_trend

    if trend_method == "spline":
        trend_result = spline_trend(
            deseasonalized_df, column_value, time_column, **trend_kwargs
        )
    elif trend_method == "sgolay":
        trend_result = sgolay_trend(
            deseasonalized_df, column_value, time_column, **trend_kwargs
        )
    elif trend_method == "loess":
        try:
            from .advanced import loess_trend

            trend_result = loess_trend(
                deseasonalized_df, column_value, time_column, **trend_kwargs
            )
        except ImportError:
            warnings.warn("LOESS not available, using spline instead", stacklevel=2)
            trend_result = spline_trend(
                deseasonalized_df, column_value, time_column, **trend_kwargs
            )
    else:
        try:
            from .advanced import estimate_trend

            trend_result = estimate_trend(
                deseasonalized_df,
                column_value,
                time_column,
                trend_method,
                **trend_kwargs,
            )
        except ImportError:
            warnings.warn(
                f"Method {trend_method} not available, using spline instead",
                stacklevel=2,
            )
            trend_result = spline_trend(
                deseasonalized_df, column_value, time_column, **trend_kwargs
            )

    # Combine decomposition and trend results
    final_result = decomp_result.copy()

    # Add trend estimation results
    for col in [
        "smoothed_value",
        "derivative_value",
        "derivative_method",
        "function_order",
        "derivative_order",
    ]:
        if col in trend_result.columns:
            final_result[f"trend_{col}"] = trend_result[col]

    # Copy other trend-specific columns
    for col in trend_result.columns:
        if col not in final_result.columns and col != column_value:
            final_result[col] = trend_result[col]

    final_result["seasonality_detected"] = seasonality_info["seasonal"]
    final_result["seasonality_strength"] = seasonality_info["strength"]

    return final_result


# Convenience function for pipeline integration
def deseasonalize_pipeline(
    decomposition_method: str = "auto", period: int | None = None, **decomp_kwargs: Any
) -> Any:
    """Create a deseasonalization pipeline for use with pandas pipe.

    Parameters
    ----------
    decomposition_method : str
        Decomposition method to use
    period : int, optional
        Seasonal period
    **decomp_kwargs
        Additional decomposition arguments

    Returns:
    -------
    callable
        Function that can be used with pandas.DataFrame.pipe()

    Examples:
    --------
    >>> result = df.pipe(deseasonalize_pipeline('stl')).pipe(estimate_trend, 'spline')
    """

    def _deseasonalize(
        df: pd.DataFrame, column_value: str = "value", time_column: str | None = None
    ) -> pd.DataFrame:
        if decomposition_method == "stl":
            return stl_decompose(df, column_value, time_column, period, **decomp_kwargs)
        elif decomposition_method == "simple":
            return simple_deseasonalize(df, column_value, time_column, period)
        else:
            # Auto-select
            seasonality_info = detect_seasonality(df, column_value, time_column)
            if seasonality_info["seasonal"] and HAS_STATSMODELS:
                return stl_decompose(
                    df, column_value, time_column, period, **decomp_kwargs
                )
            else:
                return simple_deseasonalize(df, column_value, time_column, period)

    return _deseasonalize
