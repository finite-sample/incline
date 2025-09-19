"""
State-space models for trend estimation using Kalman filtering.

This module provides principled trend estimation with natural uncertainty
quantification through state-space models and Kalman filtering.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, Dict, Any
import warnings

# Check for optional dependencies
try:
    from statsmodels.tsa.statespace import mlemodel
    from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
    from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from scipy.linalg import block_diag
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .trend import compute_time_deltas


class LocalLinearTrend:
    """
    Local linear trend model using Kalman filter.

    This model represents the time series as:
    y_t = μ_t + ε_t
    μ_t = μ_{t-1} + β_{t-1} + η_t
    β_t = β_{t-1} + ζ_t

    Where μ_t is the level, β_t is the slope (trend), and ε_t, η_t, ζ_t
    are independent Gaussian noise terms.
    """

    def __init__(self,
                 obs_variance: Optional[float] = None,
                 level_variance: Optional[float] = None,
                 slope_variance: Optional[float] = None):
        """
        Parameters
        ----------
        obs_variance : float, optional
            Observation noise variance (estimated if None)
        level_variance : float, optional
            Level innovation variance (estimated if None)
        slope_variance : float, optional
            Slope innovation variance (estimated if None)
        """
        self.obs_variance = obs_variance
        self.level_variance = level_variance
        self.slope_variance = slope_variance

        self.fitted_params = None
        self.filter_results = None
        self.smoother_results = None

    def _setup_kalman_filter(self, y: np.ndarray):
        """Set up the Kalman filter for local linear trend."""
        n = len(y)

        # State vector: [level, slope]
        # Observation equation: y_t = [1, 0] * [level_t, slope_t]' + ε_t
        design = np.array([[1., 0.]])

        # Transition equation: [level_t, slope_t]' = [[1, 1], [0, 1]] * [level_{t-1}, slope_{t-1}]' + [η_t, ζ_t]'
        transition = np.array([[1., 1.],
                              [0., 1.]])

        # Selection matrix (which states get innovations)
        selection = np.eye(2)

        # Initialize Kalman filter
        kf = KalmanFilter(
            k_endog=1,  # Number of observed variables
            k_states=2,  # Number of state variables
            design=design,
            transition=transition,
            selection=selection
        )

        return kf

    def _log_likelihood(self, params: np.ndarray, y: np.ndarray) -> float:
        """Compute log-likelihood for parameter estimation."""
        obs_var, level_var, slope_var = np.exp(params)  # Ensure positive

        kf = self._setup_kalman_filter(y)

        # Set up covariance matrices
        kf['obs_cov', 0, 0] = obs_var
        kf['state_cov', 0, 0] = level_var
        kf['state_cov', 1, 1] = slope_var

        # Initial state
        kf.initialize_approximate_diffuse(variance=1e6)

        try:
            # Run filter
            results = kf.filter(y)
            return -results.loglikelihood_burn  # Return negative for minimization
        except np.linalg.LinAlgError:
            return np.inf

    def fit(self, y: np.ndarray) -> 'LocalLinearTrend':
        """
        Fit the local linear trend model to data.

        Parameters
        ----------
        y : np.ndarray
            Observed time series

        Returns
        -------
        self : LocalLinearTrend
            Fitted model
        """
        if not HAS_STATSMODELS or not HAS_SCIPY:
            raise ImportError(
                "statsmodels and scipy are required for state-space models. "
                "Install with: pip install statsmodels scipy"
            )

        y = np.asarray(y)
        n = len(y)

        if n < 4:
            raise ValueError("Need at least 4 observations for local linear trend")

        # Handle missing values
        if np.any(np.isnan(y)):
            warnings.warn("Missing values detected, using available data")

        # If variances are provided, use them
        if (self.obs_variance is not None and
            self.level_variance is not None and
                self.slope_variance is not None):

            params = np.log(
                [self.obs_variance, self.level_variance, self.slope_variance])
        else:
            # Estimate parameters via maximum likelihood
            # Initial parameter guess
            y_var = np.nanvar(y)
            initial_params = np.log([y_var * 0.1,  # obs variance
                                     y_var * 0.01,   # level variance
                                     y_var * 0.001])  # slope variance

            # Optimize
            try:
                result = minimize(
                    self._log_likelihood,
                    initial_params,
                    args=(y,),
                    method='BFGS',
                    options={'disp': False}
                )

                if result.success:
                    params = result.x
                else:
                    warnings.warn("Optimization failed, using initial guess")
                    params = initial_params
            except Exception as e:
                warnings.warn(f"Parameter estimation failed: {e}, using initial guess")
                params = initial_params

        # Store fitted parameters
        self.fitted_params = {
            'obs_variance': np.exp(params[0]),
            'level_variance': np.exp(params[1]),
            'slope_variance': np.exp(params[2])
        }

        # Set up Kalman filter with fitted parameters
        kf = self._setup_kalman_filter(y)
        kf['obs_cov', 0, 0] = self.fitted_params['obs_variance']
        kf['state_cov', 0, 0] = self.fitted_params['level_variance']
        kf['state_cov', 1, 1] = self.fitted_params['slope_variance']
        kf.initialize_approximate_diffuse(variance=1e6)

        # Run filter and smoother
        self.filter_results = kf.filter(y)
        self.smoother_results = kf.smooth(y)

        return self

    def get_level(self, confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get estimated level (smoothed values) with confidence intervals.

        Parameters
        ----------
        confidence_level : float
            Confidence level for intervals

        Returns
        -------
        level : np.ndarray
            Estimated level
        lower : np.ndarray
            Lower confidence bound
        upper : np.ndarray
            Upper confidence bound
        """
        if self.smoother_results is None:
            raise ValueError("Model must be fitted first")

        # Level is the first state
        level = self.smoother_results.smoothed_state[0, :]
        level_var = self.smoother_results.smoothed_state_cov[0, 0, :]
        level_se = np.sqrt(level_var)

        # Confidence intervals
        from scipy.stats import norm
        alpha = 1 - confidence_level
        z_score = norm.ppf(1 - alpha / 2)

        lower = level - z_score * level_se
        upper = level + z_score * level_se

        return level, lower, upper

    def get_slope(self, confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get estimated slope (trend) with confidence intervals.

        Parameters
        ----------
        confidence_level : float
            Confidence level for intervals

        Returns
        -------
        slope : np.ndarray
            Estimated slope
        lower : np.ndarray
            Lower confidence bound
        upper : np.ndarray
            Upper confidence bound
        """
        if self.smoother_results is None:
            raise ValueError("Model must be fitted first")

        # Slope is the second state
        slope = self.smoother_results.smoothed_state[1, :]
        slope_var = self.smoother_results.smoothed_state_cov[1, 1, :]
        slope_se = np.sqrt(slope_var)

        # Confidence intervals
        from scipy.stats import norm
        alpha = 1 - confidence_level
        z_score = norm.ppf(1 - alpha / 2)

        lower = slope - z_score * slope_se
        upper = slope + z_score * slope_se

        return slope, lower, upper


class StructuralTrendModel:
    """
    More general structural time series model with optional seasonal component.

    This extends the local linear trend to include:
    - Seasonal component (optional)
    - Irregular component
    - Multiple frequency components
    """

    def __init__(self,
                 seasonal_periods: Optional[list] = None,
                 damped_trend: bool = False,
                 **kwargs):
        """
        Parameters
        ----------
        seasonal_periods : list, optional
            List of seasonal periods to include
        damped_trend : bool
            Whether to use damped trend component
        **kwargs
            Additional parameters for variance components
        """
        self.seasonal_periods = seasonal_periods or []
        self.damped_trend = damped_trend
        self.kwargs = kwargs

        self.fitted_model = None

    def fit(self, y: np.ndarray):
        """Fit structural model using statsmodels."""
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels required for structural time series models")

        from statsmodels.tsa.statespace.structural import UnobservedComponents

        # Configure model components
        trend = 'local linear trend'
        if self.damped_trend:
            trend = 'damped trend'

        seasonal = None
        if self.seasonal_periods:
            # Use first seasonal period (statsmodels limitation)
            seasonal = self.seasonal_periods[0]

        # Fit model
        try:
            model = UnobservedComponents(
                y,
                level=True,
                trend=True,
                seasonal=seasonal,
                **self.kwargs
            )

            self.fitted_model = model.fit(disp=False)
            return self

        except Exception as e:
            warnings.warn(f"Structural model fitting failed: {e}")
            raise

    def get_components(self) -> Dict[str, np.ndarray]:
        """Get all estimated components."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")

        return {
            'level': self.fitted_model.level.smoothed,
            'trend': self.fitted_model.trend.smoothed,
            'seasonal': getattr(self.fitted_model, 'seasonal', None),
            'irregular': self.fitted_model.irregular.smoothed
        }


def kalman_trend(
    df: pd.DataFrame,
    column_value: str = 'value',
    time_column: Optional[str] = None,
    model_type: str = 'local_linear',
    confidence_level: float = 0.95,
    **model_kwargs
) -> pd.DataFrame:
    """
    Estimate trend using Kalman filtering and state-space models.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data
    column_value : str
        Value column name
    time_column : str, optional
        Time column name
    model_type : str
        Type of state-space model ('local_linear', 'structural')
    confidence_level : float
        Confidence level for intervals
    **model_kwargs
        Additional model parameters

    Returns
    -------
    pd.DataFrame
        Results with trend estimates and confidence intervals
    """
    y = df[column_value].values

    # Handle time scaling if needed
    if time_column:
        x = df[time_column].values
        delta = np.median(np.diff(x))
    elif isinstance(df.index, pd.DatetimeIndex):
        x, delta = compute_time_deltas(df.index)
    else:
        delta = 1.0

    if model_type == 'local_linear':
        # Fit local linear trend model
        model = LocalLinearTrend(**model_kwargs)
        model.fit(y)

        # Get results
        level, level_lower, level_upper = model.get_level(confidence_level)
        slope, slope_lower, slope_upper = model.get_slope(confidence_level)

        # Scale slope by time step
        slope = slope / delta
        slope_lower = slope_lower / delta
        slope_upper = slope_upper / delta

        # Create output dataframe
        odf = df.copy()
        odf['smoothed_value'] = level
        odf['smoothed_value_lower'] = level_lower
        odf['smoothed_value_upper'] = level_upper
        odf['derivative_value'] = slope
        odf['derivative_ci_lower'] = slope_lower
        odf['derivative_ci_upper'] = slope_upper
        odf['derivative_method'] = 'kalman'
        odf['derivative_order'] = 1
        odf['model_type'] = 'local_linear_trend'

        # Add significance test
        odf['significant_trend'] = (slope_lower > 0) | (slope_upper < 0)

        # Add fitted parameters
        for param, value in model.fitted_params.items():
            odf[f'fitted_{param}'] = value

    elif model_type == 'structural':
        # Fit structural model
        model = StructuralTrendModel(**model_kwargs)
        model.fit(y)

        components = model.get_components()

        # Estimate derivative from trend component
        trend_component = components['trend']
        if trend_component is not None:
            # Finite difference approximation
            slope = np.gradient(trend_component) / delta
        else:
            slope = np.zeros(len(y))

        odf = df.copy()
        odf['smoothed_value'] = components['level']
        odf['derivative_value'] = slope
        odf['derivative_method'] = 'kalman'
        odf['derivative_order'] = 1
        odf['model_type'] = 'structural'

        # Add components
        for comp_name, comp_values in components.items():
            if comp_values is not None:
                odf[f'{comp_name}_component'] = comp_values

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return odf


def adaptive_kalman_trend(
    df: pd.DataFrame,
    column_value: str = 'value',
    time_column: Optional[str] = None,
    adaptation_window: int = 20,
    **kwargs
) -> pd.DataFrame:
    """
    Adaptive Kalman trend estimation with time-varying parameters.

    This method estimates trend using a sliding window approach where
    Kalman filter parameters are re-estimated periodically.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data
    column_value : str
        Value column name  
    time_column : str, optional
        Time column name
    adaptation_window : int
        Window size for parameter adaptation
    **kwargs
        Additional Kalman filter parameters

    Returns
    -------
    pd.DataFrame
        Adaptive trend estimates
    """
    y = df[column_value].values
    n = len(y)

    if n < adaptation_window:
        # Fall back to standard Kalman filtering
        return kalman_trend(df, column_value, time_column, **kwargs)

    # Initialize results arrays
    smoothed_values = np.full(n, np.nan)
    derivatives = np.full(n, np.nan)
    ci_lower = np.full(n, np.nan)
    ci_upper = np.full(n, np.nan)

    # Rolling estimation
    for i in range(adaptation_window, n + 1):
        # Define window
        start_idx = max(0, i - adaptation_window)
        end_idx = i

        # Extract window data
        y_window = y[start_idx:end_idx]
        df_window = df.iloc[start_idx:end_idx].copy()

        # Fit model on window
        try:
            result_window = kalman_trend(
                df_window,
                column_value=column_value,
                time_column=time_column,
                **kwargs
            )

            # Store result for last point in window
            if len(result_window) > 0:
                last_idx = i - 1
                smoothed_values[last_idx] = result_window['smoothed_value'].iloc[-1]
                derivatives[last_idx] = result_window['derivative_value'].iloc[-1]

                if 'derivative_ci_lower' in result_window.columns:
                    ci_lower[last_idx] = result_window['derivative_ci_lower'].iloc[-1]
                    ci_upper[last_idx] = result_window['derivative_ci_upper'].iloc[-1]

        except Exception as e:
            warnings.warn(f"Adaptive estimation failed at position {i}: {e}")
            continue

    # Create output dataframe
    odf = df.copy()
    odf['smoothed_value'] = smoothed_values
    odf['derivative_value'] = derivatives
    odf['derivative_ci_lower'] = ci_lower
    odf['derivative_ci_upper'] = ci_upper
    odf['derivative_method'] = 'adaptive_kalman'
    odf['derivative_order'] = 1
    odf['adaptation_window'] = adaptation_window

    # Add significance flags
    valid_ci = ~(np.isnan(ci_lower) | np.isnan(ci_upper))
    odf['significant_trend'] = False
    odf.loc[valid_ci, 'significant_trend'] = (
        (ci_lower[valid_ci] > 0) | (ci_upper[valid_ci] < 0)
    )

    return odf


# Convenience function for automatic model selection
def select_kalman_model(
    df: pd.DataFrame,
    column_value: str = 'value',
    time_column: Optional[str] = None
) -> str:
    """
    Automatically select appropriate Kalman model based on data characteristics.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data
    column_value : str
        Value column name
    time_column : str, optional
        Time column name

    Returns
    -------
    str
        Recommended model type
    """
    y = df[column_value].values
    n = len(y)

    # Check for seasonality
    from .seasonal import detect_seasonality
    seasonality_info = detect_seasonality(df, column_value, time_column)

    # Check for trend changes (structural breaks)
    if n > 40:
        # Simple test for trend changes using rolling variance
        window = n // 4
        rolling_var = pd.Series(y).rolling(window).var()
        var_changes = np.std(rolling_var.dropna()) / np.mean(rolling_var.dropna())

        if var_changes > 0.5:
            return 'adaptive_kalman'  # Time-varying parameters

    if seasonality_info['seasonal'] and seasonality_info['strength'] > 0.3:
        return 'structural'  # Include seasonal components
    elif n < 50:
        return 'local_linear'  # Simple for small datasets
    else:
        return 'local_linear'  # Default choice
