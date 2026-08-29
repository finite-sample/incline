"""Removing a seasonal cycle before estimating a trend.

Seasonality is a preprocessing concern. :func:`deseasonalize` takes a frame and
returns a frame, so its result composes with every smoother and uncertainty
option in the package.

What a deseasonalized interval means
------------------------------------
The seasonal component is estimated from the same data as the trend, so an
interval that treats it as known omits decomposition uncertainty.
:func:`trend_with_deseasonalization` therefore bootstraps the whole pipeline
whenever uncertainty is requested. Callers can explicitly compose
:func:`deseasonalize` with an estimator when they want a conditional interval
that treats the adjusted series as fixed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

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

    def __post_init__(self) -> None:
        """Validate the detector result."""
        if not isinstance(self.seasonal, (bool, np.bool_)):
            raise ValueError("seasonal must be boolean")
        if self.period is not None and (
            isinstance(self.period, (bool, np.bool_))
            or not isinstance(self.period, (int, np.integer))
            or self.period < 2
        ):
            raise ValueError("period must be an integer of at least 2")
        if (
            isinstance(self.strength, (bool, np.bool_))
            or not isinstance(self.strength, (int, float, np.integer, np.floating))
            or not np.isfinite(self.strength)
            or not 0.0 <= self.strength <= 1.0
        ):
            raise ValueError("strength must be finite and between 0 and 1")
        if self.seasonal != (self.period is not None):
            raise ValueError("seasonal and period must agree")
        if self.method not in {
            "autocorrelation",
            "spectral",
            "variance",
            "none",
        }:
            raise ValueError("method is not a recognized seasonality detector")


def detect_seasonality(
    df: pd.DataFrame,
    value_column: str = "value",
    max_period: int | None = None,
) -> Seasonality:
    """Look for a repeating cycle, by three methods in decreasing reliability.

    Args:
        df: Time series data.
        value_column: Column holding the values.
        max_period: Longest cycle to consider.

    Returns:
        What was found.

    Raises:
        ValueError: If ``max_period`` is supplied but is not an integer of at
            least two.
    """
    y = np.asarray(df[value_column], dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 8:
        return Seasonality(False, None, 0.0, "none")

    # A flat series has no cycle to find, and every detector divides by its
    # variance: acf returns all-nan and warns rather than raising, which the
    # broad except below would then report as a failed check.
    if np.ptp(y) == 0:
        return Seasonality(False, None, 0.0, "none")

    if max_period is None:
        max_period = min(n // 3, 365)
    elif (
        isinstance(max_period, (bool, np.bool_))
        or not isinstance(max_period, (int, np.integer))
        or max_period < 2
    ):
        raise ValueError("max_period must be an integer of at least 2")

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
        # asarray: scipy.fft's uarray-dispatch annotations lose the array type.
        spectrum = np.abs(np.asarray(fft(signal.detrend(y)))[1 : n // 2])
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
        within_variances = np.asarray(
            [np.var(group) if len(group) > 1 else total_variance for group in groups],
            dtype=float,
        )
        within = float(np.mean(within_variances))
        reduction = 1 - within / total_variance
        if reduction > best_reduction:
            best_period, best_reduction = period, reduction

    if best_reduction <= VARIANCE_THRESHOLD or best_period is None:
        return None
    return Seasonality(True, best_period, best_reduction, "variance")


def _resolve_period(
    df: pd.DataFrame, value_column: str, period: int | None, n: int
) -> int:
    """Settle on a period, detecting one when not told."""
    if period is not None:
        if (
            isinstance(period, (bool, np.bool_))
            or not isinstance(period, (int, np.integer))
            or period < 2
        ):
            raise ValueError("period must be an integer of at least 2")
        return period
    found = detect_seasonality(df, value_column)
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
    value_column: str = "value",
    period: int | None = None,
    seasonal_window_length: int = 7,
    trend_window_length: int | None = None,
    robust: bool = True,
) -> pd.DataFrame:
    """Seasonal-trend decomposition by LOESS.

    An explicit STL request either returns an STL decomposition or raises.
    Automatic method selection may fall back to a moving average.

    Args:
        df: Time series data.
        value_column: Column holding the values.
        period: Cycle length. Detected when None.
        seasonal_window_length: Length of the seasonal smoother; must be odd.
        trend_window_length: Length of the trend smoother; derived when None.
        robust: Downweight outliers.

    Returns:
        The frame plus :data:`DECOMPOSITION_COLUMNS`.

    Raises:
        ValueError: If STL cannot be fit with the requested configuration.
    """
    y = np.asarray(df[value_column], dtype=float)
    n = len(y)
    period = _resolve_period(df, value_column, period, n)

    if period < 2 or period >= n // 2:
        raise ValueError(f"Period {period} is unusable for a series of {n} points")
    if (
        isinstance(seasonal_window_length, (bool, np.bool_))
        or not isinstance(seasonal_window_length, (int, np.integer))
        or seasonal_window_length < 7
        or seasonal_window_length % 2 == 0
    ):
        raise ValueError("seasonal_window_length must be an odd integer of at least 7")
    if trend_window_length is not None and (
        isinstance(trend_window_length, (bool, np.bool_))
        or not isinstance(trend_window_length, (int, np.integer))
        or trend_window_length < 3
        or trend_window_length % 2 == 0
    ):
        raise ValueError("trend_window_length must be an odd integer of at least 3")
    if not isinstance(robust, (bool, np.bool_)):
        raise ValueError("robust must be boolean")
    if trend_window_length is None:
        trend_window_length = int(1.5 * period / (1 - 1.5 / seasonal_window_length))
        if trend_window_length % 2 == 0:
            trend_window_length += 1

    try:
        series = (
            pd.Series(y, index=df.index)
            if isinstance(df.index, pd.DatetimeIndex)
            else pd.Series(y)
        )
        fitted = STL(
            series,
            seasonal=seasonal_window_length,
            trend=trend_window_length,
            period=period,
            robust=robust,
        ).fit()
    except Exception as exc:
        raise ValueError(f"STL decomposition failed: {exc}") from exc

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
    value_column: str = "value",
    period: int | None = None,
) -> pd.DataFrame:
    """Classical decomposition by a centered moving average.

    Args:
        df: Time series data.
        value_column: Column holding the values.
        period: Cycle length. Detected when None.

    Returns:
        The frame plus :data:`DECOMPOSITION_COLUMNS`.

    Raises:
        ValueError: If the detected or supplied period cannot support the
            decomposition.
    """
    y = np.asarray(df[value_column], dtype=float)
    n = len(y)
    period = _resolve_period(df, value_column, period, n)

    if period < 2 or period >= n // 2:
        raise ValueError(f"Period {period} is unusable for a series of {n} points")

    trend = _centered_average(y, period, n)
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


def _centered_average(
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
            pd.Series(y).rolling(window=period, center=True).mean(),
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
    value_column: str = "value",
    method: str = "auto",
    period: int | None = None,
) -> pd.DataFrame:
    """Split a series into trend, cycle and remainder.

    The front door for seasonality. Returns a frame, so the result can be
    handed to any estimator::

        clean = deseasonalize(df)
        result = sgolay_trend(
            clean, value_column="deseasonalized", with_uncertainty=True
        )

    Args:
        df: Time series data.
        value_column: Column holding the values.
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
            return stl_decompose(df, value_column, period)
        case "simple":
            return moving_average_decompose(df, value_column, period)
        case "auto":
            pass
        case _:
            raise ValueError(
                f"Unknown decomposition method {method!r}; "
                f"use 'auto', 'stl' or 'simple'"
            )

    if period is None:
        found = detect_seasonality(df, value_column)
        if not found.seasonal:
            # Nothing to remove. Still emit the full schema so that callers
            # never branch on whether a cycle happened to be found.
            y = np.asarray(df[value_column], dtype=float)
            zeros = np.zeros(len(y))
            return _assemble(df, y, zeros, zeros, y, 0, "none")
        period = cast("int", found.period)

    try:
        return stl_decompose(df, value_column, period)
    except ValueError as exc:
        warnings.warn(
            f"{exc}; falling back to a moving-average decomposition.",
            stacklevel=2,
        )
        try:
            return moving_average_decompose(df, value_column, period)
        except ValueError:
            y = np.asarray(df[value_column], dtype=float)
            zeros = np.zeros(len(y))
            return _assemble(df, y, zeros, zeros, y, int(period), "none")


def trend_with_deseasonalization(
    df: pd.DataFrame,
    smoother: Smoother | None = None,
    value_column: str = "value",
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

    With ``with_uncertainty=True`` the uncertainty accounts for the seasonal
    fit as well as the trend fit. The standard error comes from bootstrapping
    the **whole pipeline**, resampling the decomposition residuals and refitting
    both the decomposition and the trend.

    That costs ``n_bootstrap`` decompositions, which is the price of an honest
    number and is only paid when a standard error is asked for. If you want the
    cheap interval that treats the adjusted series as data, compose the two
    steps yourself::

        adjusted = deseasonalize(df)
        result = sgolay_trend(
            adjusted, value_column="deseasonalized", with_uncertainty=True
        )

    which says plainly what it assumes.

    Args:
        df: Time series data.
        smoother: Estimator to run. Penalized spline by default.
        value_column: Column holding the values.
        time_column: Numeric time column.
        method: Decomposition method; see :func:`deseasonalize`.
        period: Cycle length. Detected when None.
        n_bootstrap: Replicates used to propagate the decomposition.
        random_state: Seed or Generator for the bootstrap.
        **fit_kwargs: Passed through to the estimator, e.g. ``with_uncertainty=True``.

    Returns:
        The estimator's usual columns plus :data:`DECOMPOSITION_COLUMNS`. One
        schema, whether or not a cycle was found.
    """
    from .api import estimate
    from .smoothers import SmoothingSpline

    decomposed = deseasonalize(df, value_column, method, period)
    chosen = smoother if smoother is not None else SmoothingSpline()

    def fit_to(values: npt.NDArray[np.float64]) -> Any:
        working = df.copy()
        working[value_column] = values
        return estimate(chosen, working, value_column, time_column, **fit_kwargs)

    point = fit_to(decomposed["deseasonalized"].to_numpy())
    result = point.to_frame(df)

    if fit_kwargs.get("with_uncertainty"):
        lower, upper, spread = _bootstrap_pipeline(
            df,
            decomposed,
            value_column,
            time_column,
            method,
            period,
            chosen,
            fit_kwargs,
            n_bootstrap,
            random_state,
            float(fit_kwargs.get("confidence_level", 0.95)),
        )
        if spread is not None:
            result["derivative_standard_error"] = spread
            result["derivative_ci_lower"] = lower
            result["derivative_ci_upper"] = upper
            result["uncertainty_method"] = "pipeline_bootstrap"
            # Same rule as TrendEstimate.significant, including the positive-se
            # requirement. Writing the comparison out again here dropped that
            # guard, and on a series with no detected cycle -- where the
            # decomposition residual is identically zero -- every replicate is
            # the same, the spread is ~1e-17, and 100% of points came back
            # flagged as trending.
            usable = (
                np.isfinite(lower)
                & np.isfinite(upper)
                & np.isfinite(spread)
                & (spread > 0)
            )
            result["significant_trend"] = usable & ((lower > 0) | (upper < 0))

    for column in DECOMPOSITION_COLUMNS:
        result[column] = decomposed[column].to_numpy()
    return result


def _bootstrap_pipeline(
    df: pd.DataFrame,
    decomposed: pd.DataFrame,
    value_column: str,
    time_column: str | None,
    method: str,
    period: int | None,
    smoother: Smoother,
    fit_kwargs: dict[str, Any],
    n_bootstrap: int,
    random_state: int | np.random.Generator | None,
    confidence_level: float,
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
    if residuals.size == 0 or float(np.std(residuals)) <= 0.0:
        # A decomposition that left no residual -- the "no cycle found" route
        # returns exactly zeros -- cannot be resampled into anything.
        return None, None, None
    residuals = residuals - residuals.mean()

    observed = np.asarray(df[value_column], dtype=float)
    spread = float(residuals.std())
    if spread > 1e-12:
        residuals = residuals * (rice_sigma(observed) / spread)

    # The inner fits only contribute a point estimate, so asking each of them
    # for its own uncertainty nests a bootstrap inside a bootstrap. With the
    # The default cross-validated SmoothingSpline previously made this 100 outer x
    # 200 inner spline fits -- about 4.7 minutes for one default call.
    inner_kwargs = {
        k: v
        for k, v in fit_kwargs.items()
        if k
        not in {
            "with_uncertainty",
            "n_bootstrap",
            "simultaneous",
            "random_state",
        }
    }

    draws = []
    for _ in range(n_bootstrap):
        resampled = fitted + rng.choice(residuals, size=len(fitted), replace=True)
        replicate = df.copy()
        replicate[value_column] = resampled
        try:
            parts = deseasonalize(replicate, value_column, method, period)
            working = df.copy()
            working[value_column] = parts["deseasonalized"].to_numpy()
            draws.append(
                estimate(
                    smoother, working, value_column, time_column, **inner_kwargs
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
    # The percentiles have to follow the requested level; hard-coding 2.5/97.5
    # returned a 95% interval whatever the caller asked for, which silently
    # changed significance decisions.
    alpha = 1.0 - confidence_level
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return (
            np.nanpercentile(stacked, 100 * alpha / 2, axis=0),
            np.nanpercentile(stacked, 100 * (1 - alpha / 2), axis=0),
            np.nanstd(stacked, axis=0),
        )
