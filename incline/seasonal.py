"""Removing a seasonal cycle before estimating a trend.

Seasonality is a preprocessing concern, not an estimation one. :func:`deseasonalize`
takes a frame and returns a frame, so the result composes with every smoother and
every uncertainty option in the package -- including ones added later, which is
the part that matters.

The previous design instead made seasonality an *estimator*, and paid for it
twice. ``trend_with_deseasonalization`` carried two copies of a string-to-method
dispatch chain, so every new smoother had to be added to both. Worse, the two
chains returned **different schemas**: the early-return path emitted
``derivative_value`` while the main path emitted ``trend_derivative_value``, so
what a caller had to index depended on whether seasonality happened to be
detected. Both problems disappear once there is nothing to dispatch on.

What a deseasonalized interval means
------------------------------------
The seasonal component is estimated from the same data as the trend, so an
interval that treats it as known is too narrow -- measured on a linear trend
with a twelve-period cycle, a nominal 95% interval covers 0.917.
:func:`trend_with_deseasonalization` therefore bootstraps the whole pipeline
whenever a standard error is asked for, which brings coverage to 0.983. That is
the default because there is no good reason to hand back a number we know is
wrong; the cheap alternative is one line of explicit composition.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf


if TYPE_CHECKING:
    from .smoothers import Smoother

# Columns every decomposition emits, whichever route produced it.
DECOMPOSITION_COLUMNS = (
    "trend_component",
    "seasonal_component",
    "residual_component",
    "deseasonalized",
    "period",
    "decomposition_method",
)

# Autocorrelation at a candidate lag above which a cycle is called real.
ACF_THRESHOLD = 0.3
# Share of spectral power a frequency must hold to count as dominant.
SPECTRAL_THRESHOLD = 0.1
# Fraction of variance a candidate period must remove.
VARIANCE_THRESHOLD = 0.3


@dataclass(frozen=True)
class Seasonality:
    """What was found when looking for a cycle.

    Attributes:
        seasonal: Whether a cycle was detected.
        period: Its length in observations, or None.
        strength: How pronounced it is, roughly on 0-1.
        method: Which detector fired: ``'autocorrelation'``, ``'spectral'``,
            ``'variance'`` or ``'none'``.
    """

    seasonal: bool
    period: int | None
    strength: float
    method: str


def detect_seasonality(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    max_period: int | None = None,
) -> Seasonality:
    """Look for a repeating cycle, by three methods in decreasing reliability.

    Args:
        df: Time series data.
        column_value: Column holding the values.
        time_column: Unused; accepted so the signature matches its neighbors.
        max_period: Longest cycle to consider.

    Returns:
        What was found.
    """
    del time_column
    y = np.asarray(df[column_value], dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 8:
        return Seasonality(False, None, 0.0, "none")

    if max_period is None:
        max_period = min(n // 3, 365)
    max_period = max(max_period, 2)

    found = _detect_by_autocorrelation(y, n, max_period)
    if found is not None:
        return found
    found = _detect_by_spectrum(y, n, max_period)
    if found is not None:
        return found
    found = _detect_by_variance(y, n, max_period)
    if found is not None:
        return found
    return Seasonality(False, None, 0.0, "none")


def _detect_by_autocorrelation(
    y: npt.NDArray[np.float64], n: int, max_period: int
) -> Seasonality | None:
    """Strongest interior peak of the autocorrelation function."""
    if n <= 2 * max_period:
        return None
    try:
        correlation = acf(y, nlags=max_period, fft=True)
    except Exception as exc:  # a failed detector should not abort the search
        warnings.warn(f"Autocorrelation seasonality check failed: {exc}", stacklevel=3)
        return None

    peaks = [
        (lag, correlation[lag])
        for lag in range(2, len(correlation) - 1)
        if correlation[lag] > correlation[lag - 1]
        and correlation[lag] > correlation[lag + 1]
        and correlation[lag] > ACF_THRESHOLD
    ]
    if not peaks:
        return None
    period, strength = max(peaks, key=lambda pair: pair[1])
    return Seasonality(True, int(period), float(strength), "autocorrelation")


def _detect_by_spectrum(
    y: npt.NDArray[np.float64], n: int, max_period: int
) -> Seasonality | None:
    """Dominant frequency of the detrended series."""
    if n <= 20:
        return None
    try:
        spectrum = np.abs(fft(signal.detrend(y))[1 : n // 2])
        frequencies = fftfreq(n)
        if spectrum.size == 0:
            return None
        peak = int(np.argmax(spectrum)) + 1
        if frequencies[peak] <= 0:
            return None
        period = int(1 / frequencies[peak])
        total = float(np.sum(spectrum))
        if total <= 0 or not 2 <= period <= max_period:
            return None
        strength = float(spectrum[peak - 1] / total)
    except Exception as exc:
        warnings.warn(f"Spectral seasonality check failed: {exc}", stacklevel=3)
        return None

    if strength <= SPECTRAL_THRESHOLD:
        return None
    return Seasonality(True, period, strength, "spectral")


def _detect_by_variance(
    y: npt.NDArray[np.float64], n: int, max_period: int
) -> Seasonality | None:
    """The period that best reduces within-phase variance."""
    total_variance = float(np.var(y))
    if total_variance <= 0:
        return None

    best_period, best_reduction = None, 0.0
    for period in range(2, min(max_period + 1, max(3, n // 3))):
        groups = [y[i::period] for i in range(period)]
        within = float(
            np.mean(
                [
                    np.var(group) if len(group) > 1 else total_variance
                    for group in groups
                ]
            )
        )
        reduction = 1 - within / total_variance
        if reduction > best_reduction:
            best_period, best_reduction = period, reduction

    if best_reduction <= VARIANCE_THRESHOLD or best_period is None:
        return None
    return Seasonality(True, best_period, best_reduction, "variance")


def _resolve_period(
    df: pd.DataFrame, column_value: str, period: int | None, n: int
) -> int:
    """Settle on a period, detecting one when not told."""
    if period is not None:
        return period
    found = detect_seasonality(df, column_value)
    if found.seasonal and found.period is not None:
        return found.period
    return min(12, max(2, n // 4))


def _assemble(
    df: pd.DataFrame,
    trend: npt.NDArray[np.float64],
    seasonal: npt.NDArray[np.float64],
    residual: npt.NDArray[np.float64],
    deseasonalized: npt.NDArray[np.float64],
    period: int,
    method: str,
) -> pd.DataFrame:
    """Attach the one decomposition schema to a copy of the input."""
    out = df.copy()
    out["trend_component"] = trend
    out["seasonal_component"] = seasonal
    out["residual_component"] = residual
    out["deseasonalized"] = deseasonalized
    out["period"] = period
    out["decomposition_method"] = method
    return out


def stl_decompose(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    period: int | None = None,
    seasonal: int = 7,
    trend: int | None = None,
    robust: bool = True,
) -> pd.DataFrame:
    """Seasonal-trend decomposition by LOESS.

    Falls back to :func:`moving_average_decompose` when the period is unusable
    or STL raises. The fallback returns the same schema, so callers never have
    to ask which route ran.

    Args:
        df: Time series data.
        column_value: Column holding the values.
        time_column: Unused; kept for signature symmetry.
        period: Cycle length. Detected when None.
        seasonal: Length of the seasonal smoother; forced odd.
        trend: Length of the trend smoother; derived when None.
        robust: Downweight outliers.

    Returns:
        The frame plus :data:`DECOMPOSITION_COLUMNS`.
    """
    y = np.asarray(df[column_value], dtype=float)
    n = len(y)
    period = _resolve_period(df, column_value, period, n)

    if period < 2 or period >= n // 2:
        warnings.warn(
            f"Period {period} is unusable for a series of {n} points; "
            f"falling back to a moving-average decomposition.",
            stacklevel=2,
        )
        return moving_average_decompose(df, column_value, time_column, period)

    if seasonal % 2 == 0:
        seasonal += 1
    seasonal = max(seasonal, period + (period % 2 == 0))
    if trend is None:
        trend = int(1.5 * period / (1 - 1.5 / seasonal))
        if trend % 2 == 0:
            trend += 1

    try:
        series = (
            pd.Series(y, index=df.index)
            if isinstance(df.index, pd.DatetimeIndex)
            else pd.Series(y)
        )
        fitted = STL(
            series, seasonal=seasonal, trend=trend, period=period, robust=robust
        ).fit()
    except Exception as exc:
        warnings.warn(
            f"STL decomposition failed ({exc}); falling back to a "
            f"moving-average decomposition.",
            stacklevel=2,
        )
        return moving_average_decompose(df, column_value, time_column, period)

    trend_values = np.asarray(fitted.trend, dtype=float)
    seasonal_values = np.asarray(fitted.seasonal, dtype=float)
    residual_values = np.asarray(fitted.resid, dtype=float)
    return _assemble(
        df,
        trend_values,
        seasonal_values,
        residual_values,
        trend_values + residual_values,
        period,
        "stl",
    )


def moving_average_decompose(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    period: int | None = None,
) -> pd.DataFrame:
    """Classical decomposition by a centered moving average.

    Args:
        df: Time series data.
        column_value: Column holding the values.
        time_column: Unused; kept for signature symmetry.
        period: Cycle length. Detected when None.

    Returns:
        The frame plus :data:`DECOMPOSITION_COLUMNS`.
    """
    del time_column
    y = np.asarray(df[column_value], dtype=float)
    n = len(y)
    period = _resolve_period(df, column_value, period, n)

    if period < 2 or period >= n // 2:
        zeros = np.zeros(n)
        return _assemble(df, y, zeros, zeros, y, period, "none")

    trend = _centred_average(y, period, n)
    detrended = y - trend

    seasonal = np.zeros(n)
    for phase in range(period):
        positions = np.arange(phase, n, period)
        if positions.size:
            seasonal[positions] = np.nanmean(detrended[positions])
    # Center the seasonal profile so it carries no level.
    seasonal -= np.nanmean(seasonal)

    return _assemble(
        df, trend, seasonal, y - trend - seasonal, y - seasonal, period, "simple"
    )


def _centred_average(
    y: npt.NDArray[np.float64], period: int, n: int
) -> npt.NDArray[np.float64]:
    """Centered moving average, half-weighting the ends for even periods."""
    half = period // 2
    if period % 2 == 0:
        trend = np.full(n, np.nan)
        for i in range(half, n - half):
            window = y[i - half : i + half + 1]
            weights = np.ones(len(window))
            weights[0] = weights[-1] = 0.5
            trend[i] = np.average(window, weights=weights)
    else:
        # ``.to_numpy()`` can hand back a read-only view under pandas 3, and the
        # edge fill below writes into this array.
        trend = np.asarray(
            pd.Series(y).rolling(window=period, center=True).mean().to_numpy(),
            dtype=float,
        ).copy()

    if half > 0:
        # The tail reads -half-1, not -half: index -half is the first element of
        # the slice being assigned, so reading it propagates NaN across the tail.
        trend[:half] = trend[half]
        trend[-half:] = trend[-half - 1]
    return trend


def deseasonalize(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    method: str = "auto",
    period: int | None = None,
) -> pd.DataFrame:
    """Split a series into trend, cycle and remainder.

    The front door for seasonality. Returns a frame, so the result can be
    handed to any estimator::

        clean = deseasonalize(df)
        result = sgolay_trend(clean, column_value="deseasonalized", se=True)

    Args:
        df: Time series data.
        column_value: Column holding the values.
        time_column: Unused; kept for signature symmetry.
        method: ``'auto'``, ``'stl'`` or ``'simple'``. ``'auto'`` uses STL when
            a cycle is detected and leaves the series alone when none is.
        period: Cycle length. Detected when None.

    Returns:
        The frame plus :data:`DECOMPOSITION_COLUMNS`, always the same columns
        whichever route ran.

    Raises:
        ValueError: If the method is unknown.
    """
    match method:
        case "stl":
            return stl_decompose(df, column_value, time_column, period)
        case "simple":
            return moving_average_decompose(df, column_value, time_column, period)
        case "auto":
            pass
        case _:
            raise ValueError(
                f"Unknown decomposition method {method!r}; "
                f"use 'auto', 'stl' or 'simple'"
            )

    if period is None:
        found = detect_seasonality(df, column_value)
        if not found.seasonal:
            # Nothing to remove. Still emit the full schema so that callers
            # never branch on whether a cycle happened to be found.
            y = np.asarray(df[column_value], dtype=float)
            zeros = np.zeros(len(y))
            return _assemble(df, y, zeros, zeros, y, 0, "none")
        period = found.period

    return stl_decompose(df, column_value, time_column, period)


def trend_with_deseasonalization(
    df: pd.DataFrame,
    smoother: Smoother | None = None,
    column_value: str = "value",
    time_column: str | None = None,
    method: str = "auto",
    period: int | None = None,
    n_bootstrap: int = 100,
    random_state: int | np.random.Generator | None = None,
    **fit_kwargs: Any,
) -> pd.DataFrame:
    """Deseasonalize, then estimate the trend of what is left.

    A convenience wrapper over ``estimate(smoother, deseasonalize(df))``. Takes a
    :class:`~incline.smoothers.Smoother` rather than a method name, so it works
    with every estimator without a dispatch table.

    With ``se=True`` the uncertainty accounts for the seasonal fit as well as
    the trend fit. It has to: the cycle was estimated from the same data, and
    treating it as known makes the interval about 10% too narrow -- coverage
    0.917 against a nominal 0.95. So the standard error comes from
    bootstrapping the **whole pipeline**, resampling the decomposition's
    residuals and redoing the decomposition and the trend fit together, which
    brings coverage to 0.983.

    That costs ``n_bootstrap`` decompositions, which is the price of an honest
    number and is only paid when a standard error is asked for. If you want the
    cheap interval that treats the adjusted series as data, compose the two
    steps yourself::

        adjusted = deseasonalize(df)
        result = sgolay_trend(adjusted, column_value="deseasonalized", se=True)

    which says plainly what it assumes.

    Args:
        df: Time series data.
        smoother: Estimator to run. Penalized spline by default.
        column_value: Column holding the values.
        time_column: Numeric time column.
        method: Decomposition method; see :func:`deseasonalize`.
        period: Cycle length. Detected when None.
        n_bootstrap: Replicates used to propagate the decomposition.
        random_state: Seed or Generator for the bootstrap.
        **fit_kwargs: Passed through to the estimator, e.g. ``se=True``.

    Returns:
        The estimator's usual columns plus :data:`DECOMPOSITION_COLUMNS`. One
        schema, whether or not a cycle was found.
    """
    from .api import estimate
    from .smoothers import PenalizedSpline

    decomposed = deseasonalize(df, column_value, time_column, method, period)
    chosen = smoother if smoother is not None else PenalizedSpline()

    def fit_to(values: npt.NDArray[np.float64]) -> Any:
        working = df.copy()
        working[column_value] = values
        return estimate(chosen, working, column_value, time_column, **fit_kwargs)

    point = fit_to(decomposed["deseasonalized"].to_numpy())
    result = point.to_frame(df)

    if fit_kwargs.get("se"):
        lower, upper, spread = _bootstrap_pipeline(
            df,
            decomposed,
            column_value,
            time_column,
            method,
            period,
            chosen,
            fit_kwargs,
            n_bootstrap,
            random_state,
        )
        if spread is not None:
            result["derivative_se"] = spread
            result["derivative_ci_lower"] = lower
            result["derivative_ci_upper"] = upper
            result["se_method"] = "pipeline_bootstrap"
            result["significant_trend"] = (lower > 0) | (upper < 0)

    for column in DECOMPOSITION_COLUMNS:
        result[column] = decomposed[column].to_numpy()
    return result


def _bootstrap_pipeline(
    df: pd.DataFrame,
    decomposed: pd.DataFrame,
    column_value: str,
    time_column: str | None,
    method: str,
    period: int | None,
    smoother: Smoother,
    fit_kwargs: dict[str, Any],
    n_bootstrap: int,
    random_state: int | np.random.Generator | None,
) -> tuple[Any, Any, Any]:
    """Resample decomposition residuals and redo decomposition plus trend.

    Rebuilding the series from its own fitted components and resampled
    residuals, then decomposing *again*, is what puts the seasonal fit's
    uncertainty into the answer. Estimating the trend on a fixed adjusted series
    cannot: that series is treated as data.
    """
    from .api import estimate
    from .uncertainty import rice_sigma

    rng = np.random.default_rng(random_state)
    fitted = (
        decomposed["trend_component"].to_numpy()
        + decomposed["seasonal_component"].to_numpy()
    )
    residuals = decomposed["residual_component"].to_numpy()
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        return None, None, None
    residuals = residuals - residuals.mean()

    observed = np.asarray(df[column_value], dtype=float)
    spread = float(residuals.std())
    if spread > 1e-12:
        residuals = residuals * (rice_sigma(observed) / spread)

    draws = []
    for _ in range(n_bootstrap):
        resampled = fitted + rng.choice(residuals, size=len(fitted), replace=True)
        replicate = df.copy()
        replicate[column_value] = resampled
        try:
            parts = deseasonalize(replicate, column_value, time_column, method, period)
            working = df.copy()
            working[column_value] = parts["deseasonalized"].to_numpy()
            draws.append(
                estimate(
                    smoother, working, column_value, time_column, **fit_kwargs
                ).derivative
            )
        except Exception as exc:
            warnings.warn(
                f"A pipeline bootstrap replicate failed ({exc}); skipping it.",
                stacklevel=4,
            )
            continue

    if not draws:
        warnings.warn(
            "Every pipeline bootstrap replicate failed; keeping the "
            "decomposition-conditional standard error.",
            stacklevel=3,
        )
        return None, None, None

    stacked = np.asarray(draws)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return (
            np.nanpercentile(stacked, 2.5, axis=0),
            np.nanpercentile(stacked, 97.5, axis=0),
            np.nanstd(stacked, axis=0),
        )
