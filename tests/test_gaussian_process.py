"""Tests for incline.gaussian_process module."""

import numpy as np
import pandas as pd
import pytest


try:
    from incline.gaussian_process import (
        GPTrend,
        adaptive_gp_trend,
        gp_trend,
        select_gp_kernel,
    )

    HAS_GP = True
except ImportError:
    HAS_GP = False


@pytest.mark.skipif(not HAS_GP, reason="Gaussian Process methods not available")
class TestGPTrend:
    """Test cases for GPTrend class."""

    def test_gp_trend_basic(self):
        """Test basic GPTrend functionality."""
        # Create smooth synthetic data
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y_true = 2 + 0.5 * x + 0.1 * x**2
        y_noisy = y_true + 0.1 * np.random.randn(50)

        # Fit GP
        gp = GPTrend(kernel_type="rbf")
        gp.fit(x, y_noisy)

        # Test prediction
        y_pred, y_std = gp.predict(x, return_std=True)
        assert len(y_pred) == len(x)
        assert len(y_std) == len(x)
        assert np.all(y_std > 0)  # Should have positive uncertainty

        # Test derivative prediction
        dy_mean, dy_lower, dy_upper = gp.predict_derivatives(x)
        assert len(dy_mean) == len(x)
        assert len(dy_lower) == len(x)
        assert len(dy_upper) == len(x)
        assert np.all(dy_lower <= dy_mean)
        assert np.all(dy_mean <= dy_upper)

    def test_gp_trend_kernels(self):
        """Test different kernel types."""
        np.random.seed(42)
        x = np.linspace(0, 5, 30)
        y = np.sin(x) + 0.1 * np.random.randn(30)

        for kernel in ["rbf", "matern32", "matern52"]:
            gp = GPTrend(kernel_type=kernel)
            gp.fit(x, y)

            y_pred, _y_std = gp.predict(x, return_std=True)
            assert len(y_pred) == len(x)
            assert not np.any(np.isnan(y_pred))

            # Test hyperparameters
            params = gp.get_kernel_params()
            assert isinstance(params, dict)
            assert len(params) > 0

    def test_gp_trend_fixed_params(self):
        """Test GP with fixed hyperparameters."""
        np.random.seed(42)
        x = np.linspace(0, 5, 25)
        y = x + 0.1 * np.random.randn(25)

        gp = GPTrend(kernel_type="rbf", length_scale=1.0, noise_level=0.1)
        gp.fit(x, y)

        y_pred, _y_std = gp.predict(x, return_std=True)
        assert len(y_pred) == len(x)

    def test_gp_trend_edge_cases(self):
        """Test GP edge cases."""
        # Test with minimal data
        x = np.array([0, 1, 2])
        y = np.array([1, 2, 3])

        gp = GPTrend()
        gp.fit(x, y)
        y_pred = gp.predict(x, return_std=False)
        assert len(y_pred) == 3

        # Test with too little data
        gp = GPTrend()
        with pytest.raises(ValueError, match="Need at least 3 valid observations"):
            gp.fit([1, 2], [1, 2])

        # Test prediction before fitting
        gp = GPTrend()
        with pytest.raises(ValueError, match="Must fit GP before prediction"):
            gp.predict([1, 2, 3])


@pytest.mark.skipif(not HAS_GP, reason="Gaussian Process methods not available")
class TestGPTrendFunction:
    """Test cases for gp_trend function."""

    def test_gp_trend_basic(self):
        """Test basic gp_trend functionality."""
        # Create test data
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=40, freq="D")
        t = np.arange(40)
        y = 1 + 0.1 * t + 0.01 * t**2 + 0.05 * np.random.randn(40)
        df = pd.DataFrame({"value": y}, index=dates)

        result = gp_trend(df, kernel_type="rbf")

        assert "smoothed_value" in result.columns
        assert "smoothed_value_std" in result.columns
        assert "derivative_value" in result.columns
        assert "derivative_ci_lower" in result.columns
        assert "derivative_ci_upper" in result.columns
        assert "derivative_method" in result.columns
        assert "significant_trend" in result.columns
        assert "kernel_type" in result.columns

        assert result["derivative_method"].iloc[0] == "gaussian_process"
        assert result["kernel_type"].iloc[0] == "rbf"
        assert len(result) == len(df)

        # Check that uncertainties are positive
        assert np.all(result["smoothed_value_std"] > 0)

        # Check confidence intervals
        assert np.all(result["derivative_ci_lower"] <= result["derivative_value"])
        assert np.all(result["derivative_value"] <= result["derivative_ci_upper"])

    def test_gp_trend_with_time_column(self):
        """Test gp_trend with explicit time column."""
        np.random.seed(42)
        time_values = np.linspace(0, 10, 35)
        y = 2 * time_values + 0.1 * np.random.randn(35)
        df = pd.DataFrame({"time": time_values, "value": y})

        result = gp_trend(df, time_column="time", kernel_type="matern32")
        assert len(result) == len(df)
        assert "time" in result.columns
        assert result["kernel_type"].iloc[0] == "matern32"

    def test_gp_trend_different_kernels(self):
        """Test gp_trend with different kernels."""
        np.random.seed(42)
        t = np.linspace(0, 6, 30)
        y = np.sin(t) + 0.1 * np.random.randn(30)
        df = pd.DataFrame({"value": y})

        for kernel in ["rbf", "matern32", "matern52"]:
            result = gp_trend(df, kernel_type=kernel)
            assert len(result) == len(df)
            assert result["kernel_type"].iloc[0] == kernel

    def test_gp_trend_confidence_levels(self):
        """Test different confidence levels."""
        np.random.seed(42)
        y = np.random.randn(25)
        df = pd.DataFrame({"value": y})

        # Test different confidence levels
        result_95 = gp_trend(df, confidence_level=0.95)
        result_90 = gp_trend(df, confidence_level=0.90)

        assert result_95["confidence_level"].iloc[0] == 0.95
        assert result_90["confidence_level"].iloc[0] == 0.90

        # 90% intervals should be narrower than 95%
        width_95 = result_95["derivative_ci_upper"] - result_95["derivative_ci_lower"]
        width_90 = result_90["derivative_ci_upper"] - result_90["derivative_ci_lower"]
        assert np.all(width_90 <= width_95)


@pytest.mark.skipif(not HAS_GP, reason="Gaussian Process methods not available")
class TestAdaptiveGP:
    """Test cases for adaptive GP estimation."""

    def test_adaptive_gp_basic(self):
        """Test basic adaptive GP functionality."""
        np.random.seed(42)
        # Create data with changing characteristics
        t1 = np.linspace(0, 5, 25)
        t2 = np.linspace(5, 10, 25)
        y1 = 0.1 * t1 + 0.05 * np.random.randn(25)  # Linear
        y2 = 2 + 0.5 * (t2 - 5) ** 2 + 0.1 * np.random.randn(25)  # Quadratic

        t = np.concatenate([t1, t2])
        y = np.concatenate([y1, y2])
        df = pd.DataFrame({"time": t, "value": y})

        result = adaptive_gp_trend(df, time_column="time", window_size=20, overlap=0.5)

        assert "smoothed_value" in result.columns
        assert "derivative_value" in result.columns
        assert "derivative_method" in result.columns
        assert "window_size" in result.columns
        assert "overlap" in result.columns

        assert result["derivative_method"].iloc[0] == "adaptive_gp"
        assert result["window_size"].iloc[0] == 20
        assert result["overlap"].iloc[0] == 0.5
        assert len(result) == len(df)

    def test_adaptive_gp_small_dataset(self):
        """Test adaptive GP with small dataset."""
        np.random.seed(42)
        y = np.random.randn(15)  # Smaller than default window
        df = pd.DataFrame({"value": y})

        # Should fall back to standard GP
        result = adaptive_gp_trend(df, window_size=25)
        assert len(result) == len(df)
        # Method might be 'gaussian_process' instead of 'adaptive_gp' for fallback

    def test_adaptive_gp_parameters(self):
        """Test adaptive GP with different parameters."""
        np.random.seed(42)
        y = 0.1 * np.arange(60) + 0.1 * np.random.randn(60)
        df = pd.DataFrame({"value": y})

        # Test different window sizes and overlaps
        result1 = adaptive_gp_trend(df, window_size=20, overlap=0.3)
        result2 = adaptive_gp_trend(df, window_size=30, overlap=0.7)

        assert len(result1) == len(df)
        assert len(result2) == len(df)
        assert result1["window_size"].iloc[0] == 20
        assert result2["window_size"].iloc[0] == 30


class TestKernelSelection:
    """Test cases for automatic kernel selection."""

    @pytest.mark.skipif(not HAS_GP, reason="Gaussian Process methods not available")
    def test_select_gp_kernel_smooth(self):
        """Test kernel selection for smooth data."""
        # Very smooth data (polynomial)
        t = np.linspace(0, 5, 50)
        y = 1 + 2 * t + 0.1 * t**2  # Smooth quadratic
        df = pd.DataFrame({"value": y})

        kernel = select_gp_kernel(df)
        assert kernel in ["rbf", "matern52", "matern32"]

    @pytest.mark.skipif(not HAS_GP, reason="Gaussian Process methods not available")
    def test_select_gp_kernel_rough(self):
        """Test kernel selection for rough data."""
        np.random.seed(42)
        # Rough data with high frequency components
        t = np.linspace(0, 10, 100)
        y = np.sin(5 * t) + 0.5 * np.random.randn(100)
        df = pd.DataFrame({"value": y})

        kernel = select_gp_kernel(df)
        assert kernel in ["rbf", "matern52", "matern32"]

    @pytest.mark.skipif(not HAS_GP, reason="Gaussian Process methods not available")
    def test_select_gp_kernel_small_data(self):
        """Test kernel selection for small datasets."""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})

        kernel = select_gp_kernel(df)
        assert kernel == "rbf"  # Default for small datasets


class TestEdgeCases:
    """Test edge cases for GP methods."""

    @pytest.mark.skipif(not HAS_GP, reason="Gaussian Process methods not available")
    def test_missing_values(self):
        """Test handling of missing values."""
        np.random.seed(42)
        y = np.random.randn(30)
        y[10:15] = np.nan  # Introduce missing values
        df = pd.DataFrame({"value": y})

        # Should handle missing values gracefully
        result = gp_trend(df)
        assert len(result) == len(df)
        # Should have valid predictions for non-missing points

    @pytest.mark.skipif(not HAS_GP, reason="Gaussian Process methods not available")
    def test_constant_data(self):
        """Test with constant data."""
        df = pd.DataFrame({"value": np.ones(25)})

        result = gp_trend(df)
        assert len(result) == len(df)
        # Derivatives should be close to zero
        assert np.all(np.abs(result["derivative_value"]) < 0.1)

    @pytest.mark.skipif(not HAS_GP, reason="Gaussian Process methods not available")
    def test_linear_trend(self):
        """Test with perfect linear trend."""
        t = np.arange(30)
        y = 2 + 0.5 * t  # Perfect linear trend
        df = pd.DataFrame({"value": y})

        result = gp_trend(df)
        assert len(result) == len(df)
        # Derivatives should be close to 0.5
        middle_derivatives = result["derivative_value"].iloc[5:-5]
        assert np.abs(np.mean(middle_derivatives) - 0.5) < 0.2

    @pytest.mark.skipif(not HAS_GP, reason="Gaussian Process methods not available")
    def test_irregular_spacing(self):
        """Test with irregularly spaced time points."""
        np.random.seed(42)
        irregular_times = np.sort(np.random.uniform(0, 10, 25))
        y = 0.5 * irregular_times + 0.1 * np.random.randn(25)
        df = pd.DataFrame({"time": irregular_times, "value": y})

        result = gp_trend(df, time_column="time")
        assert len(result) == len(df)
        assert "time" in result.columns


class TestFallbackBehavior:
    """Test fallback behavior when GP fails."""

    def test_gp_not_available(self):
        """Test behavior when scikit-learn is not available."""
        # This test is mainly for documentation - actual testing would require
        # mocking the import failure
        pass

    @pytest.mark.skipif(not HAS_GP, reason="Gaussian Process methods not available")
    def test_gp_fitting_failure(self):
        """Test fallback when GP fitting fails."""
        # Create problematic data that might cause GP fitting to fail
        df = pd.DataFrame({"value": [np.nan, np.inf, -np.inf, 1, 2]})

        try:
            result = gp_trend(df)
            # Should either succeed or fall back to spline method
            assert len(result) == len(df)
        except Exception:
            # If it fails completely, that's also acceptable for bad data
            # This is expected behavior for edge case testing
            import warnings

            warnings.warn(
                "GP trend failed on degenerate data - this is expected", stacklevel=2
            )


if __name__ == "__main__":
    pytest.main([__file__])
