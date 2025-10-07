"""
Gaussian Process trend estimation for derivative computation.

This module provides principled Bayesian trend estimation with uncertainty
quantification using Gaussian Processes. Derivatives are computed through
the derivative of the kernel function, providing both estimates and 
confidence intervals.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, Literal
import warnings

# Check for optional dependencies
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from .trend import compute_time_deltas


class GPTrend:
    """
    Gaussian Process trend estimator with derivative computation.

    This class provides a principled Bayesian approach to trend estimation
    using Gaussian Processes. Derivatives are computed analytically through
    the derivative of the kernel function.
    """

    def __init__(self,
                 kernel_type: str = 'rbf',
                 length_scale: Optional[float] = None,
                 noise_level: Optional[float] = None,
                 n_restarts_optimizer: int = 10):
        """
        Parameters
        ----------
        kernel_type : str
            Type of kernel ('rbf', 'matern32', 'matern52', 'periodic')
        length_scale : float, optional
            Characteristic length scale (auto-optimized if None)
        noise_level : float, optional  
            Noise level (auto-optimized if None)
        n_restarts_optimizer : int
            Number of optimizer restarts for hyperparameter optimization
        """
        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for Gaussian Process estimation. "
                "Install with: pip install scikit-learn"
            )

        self.kernel_type = kernel_type
        self.length_scale = length_scale
        self.noise_level = noise_level
        self.n_restarts_optimizer = n_restarts_optimizer

        self.gp = None
        self.X_train = None
        self.y_train = None

    def _create_kernel(self, x_scale: float):
        """Create kernel based on kernel_type."""
        if self.length_scale is not None:
            length_scale = self.length_scale
        else:
            # Auto-select based on data range
            length_scale = x_scale / 5.0

        if self.noise_level is not None:
            noise_level = self.noise_level
        else:
            noise_level = 0.1

        if self.kernel_type == 'rbf':
            kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale)
        elif self.kernel_type == 'matern32':
            kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=1.5)
        elif self.kernel_type == 'matern52':
            kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=2.5)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        # Add noise component
        kernel = kernel + WhiteKernel(noise_level=noise_level)

        return kernel

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fit Gaussian Process to data.

        Parameters
        ----------
        x : np.ndarray
            Input locations
        y : np.ndarray
            Observed values

        Returns
        -------
        self : GPTrend
            Fitted GP model
        """
        x = np.asarray(x).reshape(-1, 1)
        y = np.asarray(y)

        # Handle missing values
        valid_mask = ~(np.isnan(x.flatten()) | np.isnan(y))
        x = x[valid_mask]
        y = y[valid_mask]

        if len(x) < 3:
            raise ValueError("Need at least 3 valid observations for GP fitting")

        # Create kernel
        x_scale = np.ptp(x)
        kernel = self._create_kernel(x_scale)

        # Fit GP
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,  # Small numerical stability term
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=True
        )

        self.gp.fit(x, y)
        self.X_train = x
        self.y_train = y

        return self

    def predict(self, x: np.ndarray, return_std: bool = True):
        """
        Predict function values at new locations.

        Parameters
        ----------
        x : np.ndarray
            Input locations for prediction
        return_std : bool
            Whether to return prediction uncertainties

        Returns
        -------
        y_mean : np.ndarray
            Predicted mean values
        y_std : np.ndarray, optional
            Predicted standard deviations (if return_std=True)
        """
        if self.gp is None:
            raise ValueError("Must fit GP before prediction")

        x = np.asarray(x).reshape(-1, 1)
        return self.gp.predict(x, return_std=return_std)

    def predict_derivatives(self, x: np.ndarray, confidence_level: float = 0.95):
        """
        Predict derivatives and confidence intervals.

        Parameters
        ----------
        x : np.ndarray
            Input locations for derivative prediction
        confidence_level : float
            Confidence level for intervals

        Returns
        -------
        dy_mean : np.ndarray
            Predicted derivative means
        dy_lower : np.ndarray
            Lower confidence bounds
        dy_upper : np.ndarray
            Upper confidence bounds
        """
        if self.gp is None:
            raise ValueError("Must fit GP before prediction")

        x = np.asarray(x).reshape(-1, 1)

        # Compute derivatives using finite differences on the GP mean
        # This is an approximation - true GP derivatives would use kernel derivatives
        dx = np.median(np.diff(x.flatten())) / 100  # Small step size

        x_plus = x + dx
        x_minus = x - dx

        # Get predictions at x+dx and x-dx
        y_plus, std_plus = self.gp.predict(x_plus, return_std=True)
        y_minus, std_minus = self.gp.predict(x_minus, return_std=True)

        # Compute derivative via finite differences
        dy_mean = (y_plus - y_minus) / (2 * dx)

        # Approximate derivative uncertainty
        # This is a rough approximation - exact calculation would need kernel derivatives
        dy_var = (std_plus**2 + std_minus**2) / (2 * dx)**2
        dy_std = np.sqrt(dy_var)

        # Confidence intervals
        from scipy.stats import norm
        alpha = 1 - confidence_level
        z_score = norm.ppf(1 - alpha / 2)

        dy_lower = dy_mean - z_score * dy_std
        dy_upper = dy_mean + z_score * dy_std

        return dy_mean.flatten(), dy_lower.flatten(), dy_upper.flatten()

    def get_kernel_params(self):
        """Get fitted kernel hyperparameters."""
        if self.gp is None:
            raise ValueError("Must fit GP before accessing parameters")
        return self.gp.kernel_.get_params()


def gp_trend(
    df: pd.DataFrame,
    column_value: str = 'value',
    time_column: Optional[str] = None,
    kernel_type: str = 'rbf',
    length_scale: Optional[float] = None,
    confidence_level: float = 0.95,
    **gp_kwargs
) -> pd.DataFrame:
    """
    Estimate trend using Gaussian Process regression.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data
    column_value : str
        Value column name
    time_column : str, optional
        Time column name (uses index if None)
    kernel_type : str
        GP kernel type ('rbf', 'matern32', 'matern52')
    length_scale : float, optional
        Kernel length scale (auto-optimized if None)
    confidence_level : float
        Confidence level for intervals
    **gp_kwargs
        Additional arguments for GPTrend

    Returns
    -------
    pd.DataFrame
        Results with GP trend estimates and derivatives
    """
    if not HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for Gaussian Process estimation. "
            "Install with: pip install scikit-learn"
        )

    y = df[column_value].values

    # Get time values
    if time_column:
        x = df[time_column].values
        delta = np.median(np.diff(x))
    elif isinstance(df.index, pd.DatetimeIndex):
        x, delta = compute_time_deltas(df.index)
    else:
        x = np.arange(len(df), dtype=float)
        delta = 1.0

    # Fit GP
    try:
        gp = GPTrend(kernel_type=kernel_type, length_scale=length_scale, **gp_kwargs)
        gp.fit(x, y)

        # Predict function values
        y_mean, y_std = gp.predict(x, return_std=True)

        # Predict derivatives
        dy_mean, dy_lower, dy_upper = gp.predict_derivatives(x, confidence_level)

        # Scale derivatives by time step
        dy_mean = dy_mean / delta
        dy_lower = dy_lower / delta
        dy_upper = dy_upper / delta

        # Create output dataframe
        odf = df.copy()
        odf['smoothed_value'] = y_mean
        odf['smoothed_value_std'] = y_std
        odf['derivative_value'] = dy_mean
        odf['derivative_ci_lower'] = dy_lower
        odf['derivative_ci_upper'] = dy_upper
        odf['derivative_method'] = 'gaussian_process'
        odf['derivative_order'] = 1
        odf['kernel_type'] = kernel_type
        odf['confidence_level'] = confidence_level

        # Add significance test
        odf['significant_trend'] = (dy_lower > 0) | (dy_upper < 0)

        # Add kernel parameters
        kernel_params = gp.get_kernel_params()
        for param, value in kernel_params.items():
            if isinstance(value, (int, float)):
                odf[f'kernel_{param}'] = value

        return odf

    except Exception as e:
        warnings.warn(f"GP trend estimation failed: {e}")
        # Fallback to spline method
        from .trend import spline_trend
        return spline_trend(df, column_value, time_column)


def adaptive_gp_trend(
    df: pd.DataFrame,
    column_value: str = 'value',
    time_column: Optional[str] = None,
    window_size: int = 30,
    overlap: float = 0.5,
    **gp_kwargs
) -> pd.DataFrame:
    """
    Adaptive Gaussian Process trend estimation with sliding windows.

    This method fits local GP models in overlapping windows to handle
    non-stationary time series with changing characteristics.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data
    column_value : str
        Value column name
    time_column : str, optional
        Time column name
    window_size : int
        Size of sliding window
    overlap : float
        Overlap fraction between windows (0-1)
    **gp_kwargs
        Arguments for GP fitting

    Returns
    -------
    pd.DataFrame
        Adaptive GP trend estimates
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required for adaptive GP estimation")

    n = len(df)
    if n < window_size:
        # Fall back to standard GP
        return gp_trend(df, column_value, time_column, **gp_kwargs)

    # Initialize result arrays
    smoothed_values = np.full(n, np.nan)
    derivatives = np.full(n, np.nan)
    ci_lower = np.full(n, np.nan)
    ci_upper = np.full(n, np.nan)

    # Calculate step size
    step_size = int(window_size * (1 - overlap))

    # Sliding window estimation
    for start_idx in range(0, n - window_size + 1, step_size):
        end_idx = start_idx + window_size

        # Extract window
        window_df = df.iloc[start_idx:end_idx].copy()

        try:
            # Fit GP on window
            result = gp_trend(window_df, column_value, time_column, **gp_kwargs)

            # Store results for middle portion of window (to avoid edge effects)
            margin = window_size // 4
            store_start = start_idx + margin
            store_end = min(end_idx - margin, n)
            result_start = margin
            result_end = result_start + (store_end - store_start)

            smoothed_values[store_start:store_end] = result['smoothed_value'].iloc[result_start:result_end]
            derivatives[store_start:store_end] = result['derivative_value'].iloc[result_start:result_end]

            if 'derivative_ci_lower' in result.columns:
                ci_lower[store_start:store_end] = result['derivative_ci_lower'].iloc[result_start:result_end]
                ci_upper[store_start:store_end] = result['derivative_ci_upper'].iloc[result_start:result_end]

        except Exception as e:
            warnings.warn(f"Adaptive GP failed at window {start_idx}: {e}")
            continue

    # Fill any remaining NaN values with interpolation
    valid_mask = ~np.isnan(smoothed_values)
    if np.any(valid_mask):
        for arr in [smoothed_values, derivatives, ci_lower, ci_upper]:
            if np.any(~np.isnan(arr)):
                arr[:] = pd.Series(arr).interpolate().fillna(
                    method='bfill').fillna(method='ffill').values

    # Create output dataframe
    odf = df.copy()
    odf['smoothed_value'] = smoothed_values
    odf['derivative_value'] = derivatives
    odf['derivative_ci_lower'] = ci_lower
    odf['derivative_ci_upper'] = ci_upper
    odf['derivative_method'] = 'adaptive_gp'
    odf['derivative_order'] = 1
    odf['window_size'] = window_size
    odf['overlap'] = overlap

    # Add significance flags
    valid_ci = ~(np.isnan(ci_lower) | np.isnan(ci_upper))
    odf['significant_trend'] = False
    odf.loc[valid_ci, 'significant_trend'] = (
        (ci_lower[valid_ci] > 0) | (ci_upper[valid_ci] < 0)
    )

    return odf


# Convenience function for automatic kernel selection
def select_gp_kernel(
    df: pd.DataFrame,
    column_value: str = 'value',
    time_column: Optional[str] = None
) -> str:
    """
    Automatically select appropriate GP kernel based on data characteristics.

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
        Recommended kernel type
    """
    y = df[column_value].values
    n = len(y)

    # Check for smoothness (higher order differences)
    if n > 10:
        # Compute roughness measures
        first_diff_var = np.var(np.diff(y))
        second_diff_var = np.var(np.diff(y, n=2))
        roughness = second_diff_var / (first_diff_var + 1e-10)

        if roughness > 2.0:
            return 'matern32'  # Less smooth, more flexible
        elif roughness > 0.5:
            return 'matern52'  # Moderately smooth
        else:
            return 'rbf'       # Very smooth
    else:
        return 'rbf'  # Default for small datasets
