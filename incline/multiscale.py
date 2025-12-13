"""SiZer (SIgnificance of ZERo crossings) multi-scale analysis for trend estimation.

This module implements SiZer maps that show statistical significance of derivatives
across multiple smoothing scales. This helps identify robust trend features and
distinguish signal from noise at different resolutions.
"""

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd


# Check for optional dependencies
try:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from scipy.stats import norm

    HAS_SCIPY_STATS = True
except ImportError:
    HAS_SCIPY_STATS = False

from .trend import compute_time_deltas


class SiZer:
    """SiZer (SIgnificance of ZERo crossings) analysis for multi-scale trend detection.

    SiZer creates maps showing where derivatives are significantly positive,
    negative, or insignificant across multiple smoothing bandwidths. This helps
    identify robust trend features that persist across scales.
    """

    def __init__(
        self,
        bandwidths: npt.NDArray[np.float64] | None = None,
        n_bandwidths: int = 20,
        bandwidth_range: tuple[float, float] = (0.01, 0.5),
        confidence_level: float = 0.95,
        method: str = "loess",
    ) -> None:
        """Initialize SiZer.

        Parameters
        ----------
        bandwidths : np.ndarray, optional
            Specific bandwidths to use (auto-generated if None)
        n_bandwidths : int
            Number of bandwidths (if bandwidths not specified)
        bandwidth_range : tuple
            Range of bandwidths as (min_frac, max_frac) of data range
        confidence_level : float
            Confidence level for significance testing
        method : str
            Smoothing method ('loess', 'gp', 'spline')
        """
        self.bandwidths = bandwidths
        self.n_bandwidths = n_bandwidths
        self.bandwidth_range = bandwidth_range
        self.confidence_level = confidence_level
        self.method = method

        self.sizer_map = None
        self.significance_map = None
        self.x_values = None
        self.derivative_estimates = None
        self.derivative_se = None

    def _generate_bandwidths(self, n_points: int) -> npt.NDArray[np.float64]:
        """Generate logarithmically spaced bandwidths."""
        if self.bandwidths is not None:
            return self.bandwidths

        min_bw, max_bw = self.bandwidth_range
        # Ensure minimum bandwidth gives at least 3 points
        min_bw = max(min_bw, 3.0 / n_points)
        max_bw = min(max_bw, 0.9)  # Don't exceed 90% of data

        return np.logspace(np.log10(min_bw), np.log10(max_bw), self.n_bandwidths)

    def _estimate_derivatives_at_bandwidth(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], bandwidth: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Estimate derivatives and standard errors at a single bandwidth."""
        n = len(x)
        derivatives = np.full(n, np.nan)
        std_errors = np.full(n, np.nan)

        if self.method == "loess":
            try:
                from .advanced import loess_trend

                df_temp = pd.DataFrame({"value": y})
                if not isinstance(x[0], (int, float)):
                    df_temp.index = x
                    result = loess_trend(df_temp, frac=bandwidth)
                else:
                    df_temp["time"] = x
                    result = loess_trend(df_temp, time_column="time", frac=bandwidth)

                derivatives = result["derivative_value"].values
                # Approximate standard errors from residuals
                residuals = y - result["smoothed_value"].values
                residual_std = np.std(residuals)
                std_errors = np.full(n, residual_std / np.sqrt(max(1, bandwidth * n)))

            except ImportError:
                # Fall back to local polynomial if LOESS not available
                derivatives, std_errors = self._local_polynomial_derivatives(
                    x, y, bandwidth
                )

        elif self.method == "gp":
            try:
                from .gaussian_process import gp_trend

                df_temp = pd.DataFrame({"value": y})
                if not isinstance(x[0], (int, float)):
                    df_temp.index = x
                    result = gp_trend(df_temp, length_scale=bandwidth * np.ptp(x))
                else:
                    df_temp["time"] = x
                    result = gp_trend(
                        df_temp, time_column="time", length_scale=bandwidth * np.ptp(x)
                    )

                derivatives = result["derivative_value"].values
                # Use GP uncertainty
                ci_width = result["derivative_ci_upper"] - result["derivative_ci_lower"]
                std_errors = ci_width / (2 * norm.ppf(0.975))  # Convert CI to SE

            except ImportError:
                derivatives, std_errors = self._local_polynomial_derivatives(
                    x, y, bandwidth
                )

        elif self.method == "spline":
            from .trend import spline_trend

            df_temp = pd.DataFrame({"value": y})
            # Map bandwidth to spline smoothing parameter
            s = bandwidth * n * np.var(y)
            if not isinstance(x[0], (int, float)):
                df_temp.index = x
                result = spline_trend(df_temp, s=s)
            else:
                df_temp["time"] = x
                result = spline_trend(df_temp, time_column="time", s=s)

            derivatives = result["derivative_value"].values
            # Approximate standard errors
            residuals = y - result["smoothed_value"].values
            residual_std = np.std(residuals)
            std_errors = np.full(n, residual_std * np.sqrt(bandwidth))

        else:
            # Fall back to local polynomial
            derivatives, std_errors = self._local_polynomial_derivatives(
                x, y, bandwidth
            )

        return derivatives, std_errors

    def _local_polynomial_derivatives(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], bandwidth: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Local polynomial derivative estimation (fallback method)."""
        n = len(x)
        derivatives = np.zeros(n)
        std_errors = np.zeros(n)

        # Calculate bandwidth in data units
        x_range = np.ptp(x)
        h = bandwidth * x_range

        for i in range(n):
            # Local neighborhood
            distances = np.abs(x - x[i])
            weights = np.exp(-0.5 * (distances / h) ** 2)  # Gaussian weights
            weights = weights / np.sum(weights)

            # Local linear regression
            try:
                # Weighted local linear fit
                x_local_matrix = np.column_stack([np.ones(n), x - x[i]])
                weight_matrix = np.diag(weights)
                beta = np.linalg.solve(
                    x_local_matrix.T @ weight_matrix @ x_local_matrix,
                    x_local_matrix.T @ weight_matrix @ y,
                )

                derivatives[i] = beta[1]  # Slope coefficient

                # Estimate standard error
                fitted = x_local_matrix @ beta
                residuals = y - fitted
                mse = np.sum(weights * residuals**2) / np.sum(weights)
                var_beta = mse * np.linalg.inv(
                    x_local_matrix.T @ weight_matrix @ x_local_matrix
                )
                std_errors[i] = np.sqrt(var_beta[1, 1])

            except np.linalg.LinAlgError:
                derivatives[i] = 0
                std_errors[i] = 1  # Large uncertainty for failed fits

        return derivatives, std_errors

    def fit(
        self,
        df: pd.DataFrame,
        column_value: str = "value",
        time_column: str | None = None,
    ) -> "SiZer":
        """Compute SiZer map for the given data.

        Parameters
        ----------
        df : pd.DataFrame
            Time series data
        column_value : str
            Value column name
        time_column : str, optional
            Time column name (uses index if None)

        Returns:
        -------
        self : SiZer
            Fitted SiZer object
        """
        y = df[column_value].values

        # Get time values
        if time_column:
            x = df[time_column].values
        elif isinstance(df.index, pd.DatetimeIndex):
            x, _ = compute_time_deltas(df.index)
        else:
            x = np.arange(len(df), dtype=float)

        # Remove missing values
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x = x[valid_mask]
        y = y[valid_mask]

        if len(x) < 5:
            raise ValueError("Need at least 5 valid observations for SiZer analysis")

        # Generate bandwidths
        bandwidths = self._generate_bandwidths(len(x))

        # Initialize result arrays
        n_points = len(x)
        n_bw = len(bandwidths)

        self.x_values = x
        self.derivative_estimates = np.zeros((n_bw, n_points))
        self.derivative_se = np.zeros((n_bw, n_points))

        # Compute derivatives for each bandwidth
        for i, bw in enumerate(bandwidths):
            try:
                derivs, ses = self._estimate_derivatives_at_bandwidth(x, y, bw)
                self.derivative_estimates[i, :] = derivs
                self.derivative_se[i, :] = ses
            except Exception as e:
                warnings.warn(
                    f"Failed to compute derivatives at bandwidth {bw:.3f}: {e}",
                    stacklevel=2,
                )
                # Fill with zeros (insignificant)
                self.derivative_estimates[i, :] = 0
                self.derivative_se[i, :] = 1

        # Compute significance map
        self._compute_significance_map(bandwidths)

        return self

    def _compute_significance_map(self, bandwidths: npt.NDArray[np.float64]) -> None:
        """Compute significance classification for each (x, bandwidth) pair."""
        if not HAS_SCIPY_STATS:
            warnings.warn(
                "scipy.stats not available, using simple thresholding", stacklevel=2
            )
            # Simple thresholding without proper statistics
            threshold = 2.0  # Rough approximation
            t_stats = self.derivative_estimates / (self.derivative_se + 1e-10)

            self.significance_map = np.zeros_like(t_stats, dtype=int)
            self.significance_map[t_stats > threshold] = 1  # Increasing
            self.significance_map[t_stats < -threshold] = -1  # Decreasing
            # Everything else remains 0 (insignificant)
        else:
            # Proper statistical testing
            alpha = 1 - self.confidence_level

            # Compute t-statistics
            t_stats = self.derivative_estimates / (self.derivative_se + 1e-10)

            # Critical value (two-tailed test)
            # Note: This is approximate - exact degrees of freedom would depend on bandwidth
            critical_value = norm.ppf(1 - alpha / 2)

            # Classify significance
            self.significance_map = np.zeros_like(t_stats, dtype=int)
            # Significantly increasing
            self.significance_map[t_stats > critical_value] = 1
            self.significance_map[
                t_stats < -critical_value
            ] = -1  # Significantly decreasing
            # Everything else remains 0 (insignificant)

        # Store bandwidths for plotting
        self.bandwidths = bandwidths

    def get_sizer_dataframe(self) -> pd.DataFrame:
        """Get SiZer results as a long-format DataFrame.

        Returns:
        -------
        pd.DataFrame
            Long-format DataFrame with columns: x, bandwidth, derivative, significance
        """
        if self.significance_map is None:
            raise ValueError("Must fit SiZer before getting results")

        results = []
        for i, bw in enumerate(self.bandwidths):
            for j, x_val in enumerate(self.x_values):
                results.append(
                    {
                        "x": x_val,
                        "bandwidth": bw,
                        "derivative": self.derivative_estimates[i, j],
                        "derivative_se": self.derivative_se[i, j],
                        "significance": self.significance_map[i, j],
                    }
                )

        return pd.DataFrame(results)

    def find_significant_features(
        self, min_persistence: int = 3
    ) -> dict[str, list[tuple[float, float]]]:
        """Find x-regions with persistent significant trends across multiple scales.

        Parameters
        ----------
        min_persistence : int
            Minimum number of consecutive bandwidths showing same significance

        Returns:
        -------
        dict
            Dictionary with 'increasing' and 'decreasing' keys, each containing
            list of (x_start, x_end) tuples for significant regions
        """
        if self.significance_map is None:
            raise ValueError("Must fit SiZer before finding features")

        features = {"increasing": [], "decreasing": []}

        # For each x location, check if there's persistent significance
        for j in range(len(self.x_values)):
            sig_column = self.significance_map[:, j]

            # Find runs of same significance
            current_sig = None
            run_length = 0

            for _i, sig in enumerate(sig_column):
                if sig == current_sig and sig != 0:
                    run_length += 1
                else:
                    # End of run - check if it was long enough
                    if run_length >= min_persistence and current_sig != 0:
                        x_val = self.x_values[j]
                        if current_sig == 1:
                            features["increasing"].append((x_val, x_val))
                        elif current_sig == -1:
                            features["decreasing"].append((x_val, x_val))

                    # Start new run
                    current_sig = sig
                    run_length = 1 if sig != 0 else 0

            # Check final run
            if run_length >= min_persistence and current_sig != 0:
                x_val = self.x_values[j]
                if current_sig == 1:
                    features["increasing"].append((x_val, x_val))
                elif current_sig == -1:
                    features["decreasing"].append((x_val, x_val))

        # Merge adjacent x-values into regions
        for key in features:
            if features[key]:
                merged = []
                current_start, current_end = features[key][0]

                for start, end in features[key][1:]:
                    if start <= current_end + np.median(np.diff(self.x_values)):
                        # Adjacent or overlapping - merge
                        current_end = max(current_end, end)
                    else:
                        # Gap - start new region
                        merged.append((current_start, current_end))
                        current_start, current_end = start, end

                merged.append((current_start, current_end))
                features[key] = merged

        return features

    def plot_sizer_map(
        self,
        figsize: tuple[float, float] = (12, 8),
        cmap: str = "RdBu_r",
        title: str | None = None,
    ) -> "plt.Figure":
        """Plot the SiZer significance map.

        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)
        cmap : str
            Colormap for significance levels
        title : str, optional
            Plot title

        Returns:
        -------
        matplotlib.figure.Figure
            The figure object
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for plotting SiZer maps")

        if self.significance_map is None:
            raise ValueError("Must fit SiZer before plotting")

        fig, ax = plt.subplots(figsize=figsize)

        # Create custom colormap: blue (decreasing), white (insignificant), red (increasing)
        colors = ["blue", "white", "red"]
        cmap_custom = mcolors.ListedColormap(colors)
        bounds = [-1.5, -0.5, 0.5, 1.5]
        norm = mcolors.BoundaryNorm(bounds, cmap_custom.N)

        # Plot significance map
        im = ax.imshow(
            self.significance_map,
            aspect="auto",
            origin="lower",
            cmap=cmap_custom,
            norm=norm,
            extent=[
                self.x_values[0],
                self.x_values[-1],
                np.log10(self.bandwidths[0]),
                np.log10(self.bandwidths[-1]),
            ],
        )

        # Customize axes
        ax.set_xlabel("x (time/location)")
        ax.set_ylabel("log10(bandwidth)")

        # Y-axis ticks at nice bandwidth values
        bw_ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        bw_ticks = [
            bw for bw in bw_ticks if self.bandwidths[0] <= bw <= self.bandwidths[-1]
        ]
        ax.set_yticks([np.log10(bw) for bw in bw_ticks])
        ax.set_yticklabels([f"{bw:.2f}" for bw in bw_ticks])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1])
        cbar.set_ticklabels(["Decreasing", "Insignificant", "Increasing"])
        cbar.set_label("Trend Significance")

        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"SiZer Map ({self.confidence_level:.0%} confidence)")

        plt.tight_layout()
        return fig


def sizer_analysis(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    n_bandwidths: int = 15,
    bandwidth_range: tuple[float, float] = (0.02, 0.3),
    method: str = "loess",
    confidence_level: float = 0.95,
) -> SiZer:
    """Perform SiZer analysis on time series data.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data
    column_value : str
        Value column name
    time_column : str, optional
        Time column name
    n_bandwidths : int
        Number of smoothing bandwidths to test
    bandwidth_range : tuple
        Range of bandwidths as (min_frac, max_frac)
    method : str
        Smoothing method ('loess', 'gp', 'spline')
    confidence_level : float
        Confidence level for significance testing

    Returns:
    -------
    SiZer
        Fitted SiZer object
    """
    sizer = SiZer(
        n_bandwidths=n_bandwidths,
        bandwidth_range=bandwidth_range,
        method=method,
        confidence_level=confidence_level,
    )

    return sizer.fit(df, column_value, time_column)


def quick_sizer_plot(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    **sizer_kwargs,
) -> "plt.Figure":
    """Quick SiZer analysis and plot.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data
    column_value : str
        Value column name
    time_column : str, optional
        Time column name
    **sizer_kwargs
        Additional arguments for SiZer analysis

    Returns:
    -------
    matplotlib.figure.Figure
        The SiZer plot
    """
    sizer = sizer_analysis(df, column_value, time_column, **sizer_kwargs)
    return sizer.plot_sizer_map()


# Integration with existing trend methods
def trend_with_sizer(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    trend_method: str = "spline",
    sizer_method: str = "loess",
    **trend_kwargs,
) -> pd.DataFrame:
    """Combine trend estimation with SiZer significance analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data
    column_value : str
        Value column name
    time_column : str, optional
        Time column name
    trend_method : str
        Primary trend estimation method
    sizer_method : str
        Method for SiZer analysis
    **trend_kwargs
        Arguments for trend estimation

    Returns:
    -------
    pd.DataFrame
        Results with trend estimates and SiZer significance flags
    """
    # Get primary trend estimate
    if trend_method == "spline":
        from .trend import spline_trend

        result = spline_trend(df, column_value, time_column, **trend_kwargs)
    elif trend_method == "loess":
        from .advanced import loess_trend

        result = loess_trend(df, column_value, time_column, **trend_kwargs)
    elif trend_method == "gp":
        from .gaussian_process import gp_trend

        result = gp_trend(df, column_value, time_column, **trend_kwargs)
    else:
        from .advanced import estimate_trend

        result = estimate_trend(
            df, column_value, time_column, trend_method, **trend_kwargs
        )

    # Add SiZer analysis
    try:
        sizer = sizer_analysis(df, column_value, time_column, method=sizer_method)

        # Get SiZer results at optimal bandwidth (middle of range)
        mid_bw_idx = len(sizer.bandwidths) // 2
        sizer_significance = sizer.significance_map[mid_bw_idx, :]

        # Add SiZer columns
        result["sizer_significance"] = sizer_significance
        result["sizer_increasing"] = sizer_significance == 1
        result["sizer_decreasing"] = sizer_significance == -1
        result["sizer_insignificant"] = sizer_significance == 0
        result["sizer_method"] = sizer_method

        # Find persistent features
        features = sizer.find_significant_features()
        result["persistent_increasing"] = False
        result["persistent_decreasing"] = False

        # Mark persistent regions
        if time_column:
            x_vals = df[time_column].values
        elif isinstance(df.index, pd.DatetimeIndex):
            x_vals, _ = compute_time_deltas(df.index)
        else:
            x_vals = np.arange(len(df))

        for start, end in features["increasing"]:
            mask = (x_vals >= start) & (x_vals <= end)
            result.loc[mask, "persistent_increasing"] = True

        for start, end in features["decreasing"]:
            mask = (x_vals >= start) & (x_vals <= end)
            result.loc[mask, "persistent_decreasing"] = True

    except Exception as e:
        warnings.warn(f"SiZer analysis failed: {e}", stacklevel=2)
        # Add empty SiZer columns
        n = len(result)
        result["sizer_significance"] = np.zeros(n)
        result["sizer_increasing"] = np.zeros(n, dtype=bool)
        result["sizer_decreasing"] = np.zeros(n, dtype=bool)
        result["sizer_insignificant"] = np.ones(n, dtype=bool)
        result["persistent_increasing"] = np.zeros(n, dtype=bool)
        result["persistent_decreasing"] = np.zeros(n, dtype=bool)

    return result
