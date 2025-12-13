import warnings
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter


def compute_time_deltas(time_index: pd.Index) -> tuple[npt.NDArray[np.float64], float]:
    """Compute time deltas from a pandas time index.

    Returns:
        x: Numeric time values (days from start)
        delta: Median time step for scaling
    """
    if isinstance(time_index, pd.DatetimeIndex):
        # Convert to numeric days from start
        x = (time_index - time_index[0]).total_seconds() / 86400.0
    else:
        x = np.asarray(time_index, dtype=float)

    if len(x) > 1:
        deltas = np.diff(x)
        delta = float(np.median(deltas))

        # Check for irregular sampling (only warn if truly irregular)
        cv_deltas = np.std(deltas) / np.mean(deltas) if np.mean(deltas) > 0 else 0
        if cv_deltas > 0.1:  # Coefficient of variation > 10%
            warnings.warn(
                "Time series appears irregularly sampled. "
                "Consider interpolation or using spline methods.",
                stacklevel=2,
            )
    else:
        delta = 1.0

    return x, delta


def naive_trend(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None
) -> pd.DataFrame:
    """naive_trend.

    Gives the naive slope: look to the right, look to the left,
    travel one unit each, and get the average change. At the ends,
    we merely use the left or the right value.

    Args:
        df: pandas dataFrame time series object
        column_value: column name containing the values
        time_column: column name for time values (optional)
    """
    y = df[column_value]

    # Handle time scaling
    if time_column:
        x = df[time_column].values
        delta = float(np.median(np.diff(x)))
    elif isinstance(df.index, pd.DatetimeIndex):
        x, delta = compute_time_deltas(df.index)
    else:
        delta = 1.0

    y_1 = y.shift(1)
    y_2 = y.shift(-1)

    y1_diff = (y - y_1) / delta  # backward difference
    yneg1_diff = (y_2 - y) / delta  # forward difference

    yy = pd.concat(
        [
            y.rename("orig"),
            y_1.rename("plus_1"),
            y_2.rename("min_1"),
            y1_diff.rename("plus_1_diff"),
            yneg1_diff.rename("min_1_diff"),
        ],
        axis=1,
    )
    odf = df.copy()
    odf["derivative_value"] = yy[["plus_1_diff", "min_1_diff"]].mean(axis=1)
    odf["derivative_method"] = "naive"
    odf["function_order"] = None
    odf["derivative_order"] = 1

    return odf


def spline_trend(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    function_order: int = 3,
    derivative_order: int = 1,
    s: float | None = None,
    use_gcv: bool = False,
) -> pd.DataFrame:
    """spline_trend.

    Interpolates time series with splines of 'function_order'. And then
    calculates the derivative_order using the smoothed function.

    Args:
        df: pandas dataFrame time series object
        column_value: column name containing the values
        time_column: column name for time values (optional)
        function_order: spline order (default is 3)
        derivative_order: (0, 1, 2, ... with default as 1)
        s: smoothing factor (if None, auto-estimated)
        use_gcv: use generalized cross-validation (requires scipy>=1.10)

    Returns:
        DataFrame: dataframe with 6 columns:- datetime,
            function_order (value of the polynomial order), smoothed_value,
            derivative_method, derivative_order, derivative_value.

        A row can be 2012-01-01, "spline", 2, 1, 0
    """
    y = df[column_value].values

    # Get time values
    if time_column:
        x = df[time_column].values
    elif isinstance(df.index, pd.DatetimeIndex):
        x, _delta = compute_time_deltas(df.index)
    else:
        x = np.arange(len(df), dtype=float)

    # Handle smoothing parameter
    s_used: float | str
    if use_gcv:
        try:
            from scipy.interpolate import make_smoothing_spline

            spl = make_smoothing_spline(x, y, lam=None)  # GCV selection
            smoothed = spl(x)
            deriv = spl(x, nu=derivative_order)
            s_used = "GCV"
        except ImportError:
            warnings.warn(
                "GCV requires scipy >= 1.10, falling back to UnivariateSpline",
                stacklevel=2,
            )
            use_gcv = False

    if not use_gcv:
        if s is None:
            # Residual-based smoothing factor estimate
            residual_var = np.var(np.diff(y))
            s = float(len(y) * residual_var * 0.1)  # Rule of thumb

        spl = UnivariateSpline(x, y, k=function_order, s=s)
        smoothed = spl(x)
        deriv = spl(x, nu=derivative_order)
        s_used = float(s) if s is not None else 0.0

    odf = df.copy()
    odf["smoothed_value"] = smoothed
    odf["derivative_value"] = deriv
    odf["function_order"] = function_order
    odf["derivative_method"] = "spline"
    odf["derivative_order"] = derivative_order
    odf["smoothing_parameter"] = s_used
    return odf


def sgolay_trend(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    function_order: int = 3,
    derivative_order: int = 1,
    window_length: int = 15,
    delta: float | None = None,
) -> pd.DataFrame:
    """sgolay_trend.

    Interpolates time series with savitzky-golay using polynomials of
    'function_order'. And then calculates the derivative_order using
    the smoothed function.

    Args:
        df: pandas dataFrame time series object
        column_value: column name containing the values
        time_column: column name for time values (optional)
        window_length: window size (default is 15, must be odd)
        function_order: polynomial order (default is 3)
        derivative_order: (0, 1, 2, ... with default as 1)
        delta: time step for derivative scaling (auto-computed if None)

    Returns:
        DataFrame: dataframe with 6 columns:- datetime,
            function_order (value of the polynomial order), smoothed_value,
            derivative_method, derivative_order, derivative_value.

        Sample row: 2012-01-01, "sgolay", 2, 1, 0
    """
    y = df[column_value].values

    # Get time delta for proper scaling
    if delta is None:
        if time_column:
            x = df[time_column].values
            delta = float(np.median(np.diff(x)))
        elif isinstance(df.index, pd.DatetimeIndex):
            x, delta = compute_time_deltas(df.index)
        else:
            delta = 1.0

    # Check for irregular sampling
    if time_column or isinstance(df.index, pd.DatetimeIndex):
        if time_column:
            x = df[time_column].values
        else:
            x, _ = compute_time_deltas(df.index)

        if len(x) > 1:
            deltas = np.diff(x)
            if np.std(deltas) / np.mean(deltas) > 0.05:
                warnings.warn(
                    "Savitzky-Golay assumes uniform sampling. "
                    "Consider spline methods for irregular time series.",
                    stacklevel=2,
                )

    # Ensure odd window length
    if window_length % 2 == 0:
        window_length += 1
        warnings.warn(f"Window length must be odd, using {window_length}", stacklevel=2)

    odf = df.copy()
    odf["smoothed_value"] = savgol_filter(
        y, window_length=window_length, polyorder=function_order, delta=delta
    )
    odf["derivative_value"] = savgol_filter(
        y,
        window_length=window_length,
        polyorder=function_order,
        deriv=derivative_order,
        delta=delta,
    )
    odf["function_order"] = function_order
    odf["derivative_method"] = "sgolay"
    odf["derivative_order"] = derivative_order

    # Mark edge regions with higher uncertainty
    half_window = window_length // 2
    odf["edge_region"] = False
    odf.iloc[:half_window, odf.columns.get_loc("edge_region")] = True
    odf.iloc[-half_window:, odf.columns.get_loc("edge_region")] = True

    return odf


def trending(
    df_list: list[pd.DataFrame],
    column_id: str = "id",
    derivative_order: int = 1,
    max_or_avg: str = "max",
    k: int = 5,
    robust: bool = False,
    trim_fraction: float = 0.1,
    weighting: str = "uniform",
    confidence_level: float = 0.95,
    return_confidence: bool = False,
) -> pd.DataFrame:
    """Enhanced trending analysis with robust statistics.

    For each item in the list, calculate either the max, average, or robust
    statistic (depending on max_or_avg and robust parameters) of the Yth
    derivative over the last k time periods. Orders by trend strength.

    Parameters
    ----------
    df_list : list
        List of dataframes from trend estimation functions with 'id' column
    column_id : str
        Column name for identifying time series
    derivative_order : int
        Order of derivative (1 or 2)
    max_or_avg : str
        Aggregation method ('max', 'avg', 'median', 'trimmed_mean', 'huber')
    k : int
        Number of latest time periods to consider
    robust : bool
        Use robust statistics to handle outliers
    trim_fraction : float
        Fraction to trim for trimmed_mean (0-0.5)
    weighting : str
        Weighting scheme ('uniform', 'linear', 'exponential')
    confidence_level : float
        Confidence level for bootstrap confidence intervals
    return_confidence : bool
        Whether to return confidence intervals and significance tests

    Returns:
    -------
    pd.DataFrame
        DataFrame with ranking results and optional confidence metrics
    """
    try:
        from scipy.stats import rankdata, trim_mean
        from scipy.stats.mstats import winsorize

        has_scipy_stats = True
    except ImportError:
        has_scipy_stats = False
        if robust or max_or_avg in ["trimmed_mean", "huber"]:
            warnings.warn(
                "scipy.stats not available, falling back to basic statistics",
                stacklevel=2,
            )

    # Check for empty input
    if not df_list:
        raise ValueError("No valid data found for trending analysis")

    # Collect data from all time series
    cdf = []
    for df in df_list:
        series_data = df[df["derivative_order"] == derivative_order][-k:]
        if len(series_data) > 0:
            cdf.append(series_data)

    if not cdf:
        # Return empty DataFrame when no matching derivative_order found
        return pd.DataFrame(
            columns=[column_id, "rank", "derivative_value", "trend_strength"]
        )

    tdf = pd.concat(cdf, sort=False)

    # Group by ID and compute statistics
    results = []

    for id_val, group in tdf.groupby(column_id):
        derivatives = group["derivative_value"].values

        if len(derivatives) == 0:
            continue

        # Apply weighting if specified
        weights = _compute_weights(len(derivatives), weighting)

        # Compute primary statistic
        if robust and max_or_avg == "avg":
            max_or_avg = "trimmed_mean"

        if max_or_avg == "max":
            trend_value = np.max(derivatives)
        elif max_or_avg in ["avg", "mean"]:
            if robust and has_scipy_stats:
                # Use winsorized mean for robustness
                winsorized = winsorize(
                    derivatives, limits=[trim_fraction, trim_fraction]
                )
                trend_value = np.average(winsorized, weights=weights)
            else:
                trend_value = np.average(derivatives, weights=weights)
        elif max_or_avg == "median":
            trend_value = np.median(derivatives)
        elif max_or_avg == "trimmed_mean" and has_scipy_stats:
            trend_value = trim_mean(derivatives, trim_fraction)
        elif max_or_avg == "huber" and has_scipy_stats:
            trend_value = _huber_mean(derivatives)
        else:
            # Fallback to simple mean
            trend_value = np.mean(derivatives)

        result = {
            "id": id_val,
            "trend_value": trend_value,
        }

        # Add robust statistics if requested
        if robust or return_confidence:
            result.update(_compute_robust_statistics(derivatives, weights))

        # Add confidence intervals if requested
        if return_confidence:
            try:
                ci_lower, ci_upper = _bootstrap_trend_ci(
                    derivatives, weights, max_or_avg, confidence_level
                )
                result["ci_lower"] = ci_lower
                result["ci_upper"] = ci_upper
                result["significant"] = ci_lower > 0 or ci_upper < 0
            except Exception:
                result["ci_lower"] = np.nan
                result["ci_upper"] = np.nan
                result["significant"] = False

        results.append(result)

    # Create output dataframe
    odf = pd.DataFrame(results)

    if len(odf) == 0:
        return pd.DataFrame(columns=["id", "trend_value"])

    # Add ranking
    odf["rank"] = (
        rankdata(-odf["trend_value"], method="ordinal")
        if has_scipy_stats
        else range(1, len(odf) + 1)
    )

    # Sort by rank
    odf = odf.sort_values("rank").reset_index(drop=True)

    # Rename for backward compatibility
    odf = odf.rename(columns={"trend_value": "max_or_avg"})

    return odf


def _compute_weights(n: int, weighting: str) -> np.ndarray:
    """Compute weights for time series aggregation."""
    if weighting == "uniform":
        return np.ones(n)
    elif weighting == "linear":
        return np.linspace(0.5, 1.0, n)
    elif weighting == "exponential":
        alpha = 0.3
        return alpha ** np.arange(n - 1, -1, -1)
    else:
        return np.ones(n)


def _huber_mean(x: np.ndarray, c: float = 1.345) -> float:
    """Compute Huber M-estimator of location."""
    x = np.asarray(x)
    n = len(x)

    if n == 0:
        return float("nan")
    if n == 1:
        return float(x[0])

    # Initial estimate - use mean for better convergence properties
    mu = float(np.mean(x))
    sigma = float(np.median(np.abs(x - np.median(x))) / 0.6745)  # MAD scale estimate

    if sigma == 0:
        return mu

    # Iterative reweighting
    for _ in range(20):  # More iterations for better convergence
        residuals = (x - mu) / sigma
        weights = np.where(np.abs(residuals) <= c, 1.0, c / (np.abs(residuals) + 1e-10))

        mu_new = float(np.sum(weights * x) / np.sum(weights))

        if abs(mu_new - mu) < 1e-8:  # Tighter convergence
            break
        mu = mu_new

    return mu


def _compute_robust_statistics(x: np.ndarray, weights: np.ndarray) -> dict:
    """Compute robust statistics for a time series."""
    try:
        from scipy.stats import iqr

        has_iqr = True
    except ImportError:
        has_iqr = False

    x = np.asarray(x)

    stats = {
        "median": np.median(x),
        "mad": np.median(np.abs(x - np.median(x))),  # Median Absolute Deviation
        "std": np.std(x),
        "iqr": np.percentile(x, 75) - np.percentile(x, 25) if not has_iqr else iqr(x),
        "n_obs": len(x),
    }

    # Robustness indicators
    stats["outlier_fraction"] = _outlier_fraction(x)
    stats["skewness"] = _robust_skewness(x)

    return stats


def _outlier_fraction(x: np.ndarray) -> float:
    """Estimate fraction of outliers using IQR method."""
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25

    if iqr == 0:
        return 0.0

    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr

    outliers = (x < lower_bound) | (x > upper_bound)
    return float(np.mean(outliers))


def _robust_skewness(x: np.ndarray) -> float:
    """Compute robust skewness using quartiles."""
    q25, q50, q75 = np.percentile(x, [25, 50, 75])

    if q75 - q25 == 0:
        return 0.0

    return float((q75 + q25 - 2 * q50) / (q75 - q25))


def _bootstrap_trend_ci(
    values: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    statistic: str,
    confidence_level: float,
    n_bootstrap: int = 100,
) -> tuple[float, float]:
    """Bootstrap confidence intervals for trend statistics."""
    try:
        from scipy.stats import trim_mean

        has_trim_mean = True
    except ImportError:
        has_trim_mean = False

    bootstrap_stats = []
    n = len(values)

    for _ in range(n_bootstrap):
        # Bootstrap sample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        boot_values = values[indices]
        boot_weights = weights[indices] if len(weights) == n else np.ones(n)

        # Compute statistic
        if statistic == "max":
            boot_stat = float(np.max(boot_values))
        elif statistic in ["avg", "mean"]:
            boot_stat = float(np.average(boot_values, weights=boot_weights))
        elif statistic == "median":
            boot_stat = float(np.median(boot_values))
        elif statistic == "trimmed_mean" and has_trim_mean:
            boot_stat = float(trim_mean(boot_values, 0.1))
        else:
            boot_stat = float(np.mean(boot_values))

        bootstrap_stats.append(boot_stat)

    # Compute confidence interval
    alpha = 1 - confidence_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)

    ci_lower = float(np.percentile(bootstrap_stats, lower_percentile))
    ci_upper = float(np.percentile(bootstrap_stats, upper_percentile))

    return ci_lower, ci_upper


def bootstrap_derivative_ci(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    method: Literal["spline", "sgolay"] = "spline",
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
    block_size: int | None = None,
    **method_kwargs,
) -> pd.DataFrame:
    """Estimate confidence intervals for derivatives using block bootstrap.

    This handles autocorrelation in time series by using block bootstrap.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data
    column_value : str
        Value column
    time_column : str, optional
        Time column name
    method : str
        'spline' or 'sgolay'
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    block_size : int, optional
        Block size for bootstrap. If None, uses sqrt(n)
    **method_kwargs
        Additional arguments for the trend method

    Returns:
    -------
    pd.DataFrame
        Original results plus CI columns
    """
    n = len(df)

    if block_size is None:
        block_size = int(np.sqrt(n))

    # Select method
    trend_func: Any
    if method == "spline":
        trend_func = spline_trend
    elif method == "sgolay":
        trend_func = sgolay_trend
    else:
        raise ValueError(f"Unknown method: {method}")

    # Get point estimate
    result = trend_func(
        df, column_value=column_value, time_column=time_column, **method_kwargs
    )

    # Bootstrap samples
    bootstrap_derivatives: list[npt.NDArray[np.float64]] = []

    for _ in range(n_bootstrap):
        # Create block bootstrap sample
        n_blocks = n // block_size + 1
        blocks = []

        for _ in range(n_blocks):
            start_idx = np.random.randint(0, n - block_size + 1)
            blocks.append(df.iloc[start_idx : start_idx + block_size])

        bootstrap_df = pd.concat(blocks, ignore_index=True).iloc[:n]
        # Preserve time structure if using time column
        if time_column:
            bootstrap_df[time_column] = df[time_column].values
        elif isinstance(df.index, pd.DatetimeIndex):
            bootstrap_df.index = df.index

        # Compute derivative for bootstrap sample
        try:
            boot_result = trend_func(
                bootstrap_df,
                column_value=column_value,
                time_column=time_column,
                **method_kwargs,
            )
            bootstrap_derivatives.append(boot_result["derivative_value"].values)
        except Exception as e:
            # Log bootstrap failure and continue
            warnings.warn(f"Bootstrap sample failed: {e}", stacklevel=3)
            continue

    if len(bootstrap_derivatives) > 0:
        bootstrap_array = np.array(bootstrap_derivatives)

        # Compute confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        result["derivative_ci_lower"] = np.percentile(
            bootstrap_array, lower_percentile, axis=0
        )
        result["derivative_ci_upper"] = np.percentile(
            bootstrap_array, upper_percentile, axis=0
        )
        result["derivative_std"] = np.std(bootstrap_array, axis=0)

        # Flag significant trends (CI doesn't include 0)
        result["significant_trend"] = (result["derivative_ci_lower"] > 0) | (
            result["derivative_ci_upper"] < 0
        )
    else:
        warnings.warn(
            "Bootstrap failed, no confidence intervals computed", stacklevel=2
        )

    return result


def select_smoothing_parameter_cv(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    method: Literal["spline", "sgolay"] = "spline",
    param_name: str = "s",
    param_range: np.ndarray | None = None,
    cv_folds: int = 5,
    **method_kwargs,
) -> tuple[float, pd.DataFrame]:
    """Select smoothing parameter using cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data
    column_value : str
        Value column
    time_column : str, optional
        Time column name
    method : str
        'spline' or 'sgolay'
    param_name : str
        Parameter to optimize ('s' for spline, 'window_length' for sgolay)
    param_range : array-like, optional
        Range of parameters to test
    cv_folds : int
        Number of CV folds

    Returns:
    -------
    best_param : float
        Optimal parameter value
    cv_results : pd.DataFrame
        CV scores for each parameter
    """
    n = len(df)

    # Default parameter ranges
    if param_range is None:
        if param_name == "s":
            # Smoothing factor range for splines
            var_y = np.var(df[column_value])
            param_range = np.logspace(
                np.log10(n * var_y * 0.001), np.log10(n * var_y * 10), 20
            )
        elif param_name == "window_length":
            # Window length range for SG (odd numbers only)
            max_window = min(51, n // 2)
            param_range = np.arange(5, max_window, 2)
        else:
            # Default fallback
            param_range = np.array([1.0])

    # Cross-validation
    fold_size = n // cv_folds
    cv_scores = []

    for param in param_range:
        fold_errors = []

        for fold in range(cv_folds):
            # Split data
            test_start = fold * fold_size
            test_end = min(test_start + fold_size, n)
            test_idx = np.arange(test_start, test_end)
            train_idx = np.concatenate(
                [np.arange(0, test_start), np.arange(test_end, n)]
            )

            if len(train_idx) < 10:
                continue

            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            # Fit on train
            kwargs = method_kwargs.copy()
            kwargs[param_name] = param

            try:
                if method == "spline":
                    result = spline_trend(
                        train_df,
                        column_value=column_value,
                        time_column=time_column,
                        **kwargs,
                    )
                else:
                    result = sgolay_trend(
                        train_df,
                        column_value=column_value,
                        time_column=time_column,
                        **kwargs,
                    )

                # Predict on test (simple interpolation)
                test_pred = np.interp(
                    test_idx, train_idx, result["smoothed_value"].values
                )

                # Compute error
                test_true = test_df[column_value].values
                mse = np.mean((test_true - test_pred) ** 2)
                fold_errors.append(mse)
            except Exception:
                fold_errors.append(np.inf)

        if fold_errors:
            cv_scores.append(
                {
                    "parameter": param,
                    "mean_mse": np.mean(fold_errors),
                    "std_mse": np.std(fold_errors),
                }
            )

    cv_results = pd.DataFrame(cv_scores)
    if len(cv_results) > 0:
        best_idx = cv_results["mean_mse"].idxmin()
        best_param = cv_results.loc[best_idx, "parameter"]
    else:
        # Fallback
        best_param = param_range[len(param_range) // 2]
        warnings.warn("CV failed, using middle parameter value", stacklevel=2)

    return best_param, cv_results


if __name__ == "__main__":
    pass
