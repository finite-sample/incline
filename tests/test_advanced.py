"""Tests for incline.advanced module."""

import numpy as np
import pandas as pd
import pytest


try:
    from incline.advanced import (
        estimate_trend,
        l1_trend_filter,
        local_polynomial_trend,
        loess_trend,
        select_trend_method,
        soft_threshold,
    )

    HAS_ADVANCED = True
except ImportError:
    HAS_ADVANCED = False


@pytest.mark.skipif(not HAS_ADVANCED, reason="Advanced methods not available")
class TestLoessTrend:
    """Test cases for LOESS trend estimation."""

    def test_loess_trend_basic(self):
        """Test basic LOESS functionality."""
        # Create smooth quadratic data with noise
        x = np.linspace(0, 10, 50)
        y_true = 2 + 3 * x + 0.1 * x**2
        y_noisy = y_true + 0.1 * np.random.randn(50)

        df = pd.DataFrame(
            {"value": y_noisy}, index=pd.date_range("2020-01-01", periods=50)
        )
        result = loess_trend(df, frac=0.3)

        assert "smoothed_value" in result.columns
        assert "derivative_value" in result.columns
        assert "derivative_method" in result.columns
        assert result["derivative_method"].iloc[0] == "loess"
        assert len(result) == len(df)

    def test_loess_trend_parameters(self):
        """Test LOESS with different parameters."""
        df = pd.DataFrame({"value": np.random.randn(30)})

        # Test different bandwidths
        result1 = loess_trend(df, frac=0.2)
        result2 = loess_trend(df, frac=0.5)

        assert len(result1) == len(df)
        assert len(result2) == len(df)
        assert result1["bandwidth"].iloc[0] == 0.2
        assert result2["bandwidth"].iloc[0] == 0.5

        # Test robust fitting
        result_robust = loess_trend(df, robust=True)
        result_nonrobust = loess_trend(df, robust=False)

        assert result_robust["robust"].iloc[0]
        assert not result_nonrobust["robust"].iloc[0]

    def test_loess_trend_derivative_orders(self):
        """Test LOESS with different derivative orders."""
        df = pd.DataFrame({"value": np.random.randn(40)})

        result1 = loess_trend(df, derivative_order=1)
        result2 = loess_trend(df, derivative_order=2)

        assert result1["derivative_order"].iloc[0] == 1
        assert result2["derivative_order"].iloc[0] == 2


@pytest.mark.skipif(not HAS_ADVANCED, reason="Advanced methods not available")
class TestL1TrendFilter:
    """Test cases for L1 trend filtering."""

    def test_l1_filter_basic(self):
        """Test basic L1 trend filtering."""
        # Create piecewise linear data
        np.arange(50)
        y = np.concatenate([np.ones(20), 2 * np.ones(30)]) + 0.1 * np.random.randn(50)

        df = pd.DataFrame({"value": y})
        result = l1_trend_filter(df, lambda_param=1.0)

        assert "smoothed_value" in result.columns
        assert "derivative_value" in result.columns
        assert "changepoint" in result.columns
        assert "derivative_method" in result.columns
        assert result["derivative_method"].iloc[0] == "l1_filter"

    def test_l1_filter_parameters(self):
        """Test L1 filter with different parameters."""
        df = pd.DataFrame({"value": np.random.randn(30)})

        # Test different lambda values
        result1 = l1_trend_filter(df, lambda_param=0.1)
        result2 = l1_trend_filter(df, lambda_param=10.0)

        assert result1["lambda"].iloc[0] == 0.1
        assert result2["lambda"].iloc[0] == 10.0

        # Test different derivative orders
        result_order1 = l1_trend_filter(df, derivative_order=1)
        result_order2 = l1_trend_filter(df, derivative_order=2)

        assert result_order1["derivative_order"].iloc[0] == 1
        assert result_order2["derivative_order"].iloc[0] == 2

    def test_soft_threshold(self):
        """Test soft thresholding operator."""
        x = np.array([-2, -1, 0, 1, 2])
        threshold = 0.5

        result = soft_threshold(x, threshold)
        expected = np.array([-1.5, -0.5, 0, 0.5, 1.5])

        np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.skipif(not HAS_ADVANCED, reason="Advanced methods not available")
class TestLocalPolynomial:
    """Test cases for local polynomial estimation."""

    def test_local_polynomial_basic(self):
        """Test basic local polynomial functionality."""
        df = pd.DataFrame({"value": np.random.randn(40)})
        result = local_polynomial_trend(df, bandwidth=0.3)

        assert "smoothed_value" in result.columns
        assert "derivative_value" in result.columns
        assert "derivative_method" in result.columns
        assert result["derivative_method"].iloc[0] == "local_poly"

    def test_local_polynomial_kernels(self):
        """Test different kernel functions."""
        df = pd.DataFrame({"value": np.random.randn(30)})

        for kernel in ["gaussian", "epanechnikov", "uniform"]:
            result = local_polynomial_trend(df, kernel=kernel)
            assert result["kernel"].iloc[0] == kernel
            assert len(result) == len(df)


class TestMethodSelection:
    """Test automatic method selection."""

    def test_select_trend_method(self):
        """Test automatic trend method selection."""
        # Create different types of data
        smooth_data = pd.DataFrame({"value": np.sin(np.linspace(0, 4 * np.pi, 100))})
        noisy_data = pd.DataFrame({"value": np.random.randn(100)})
        outlier_data = pd.DataFrame(
            {"value": np.concatenate([np.ones(80), [10, -10], np.ones(18)])}
        )

        # Test different selection criteria
        if HAS_ADVANCED:
            method1 = select_trend_method(smooth_data, criteria="auto")
            method2 = select_trend_method(noisy_data, criteria="robust")
            method3 = select_trend_method(outlier_data, criteria="changepoints")

            assert method1 in ["spline", "loess", "sgolay", "local_poly"]
            assert method2 in ["loess", "l1_filter"]
            assert method3 == "l1_filter"

    def test_estimate_trend_auto(self):
        """Test unified trend estimation interface."""
        df = pd.DataFrame({"value": np.random.randn(50)})

        if HAS_ADVANCED:
            result = estimate_trend(df, method="auto")
            assert "derivative_value" in result.columns
            assert "derivative_method" in result.columns

            # Test specific methods
            for method in ["spline", "sgolay"]:
                result = estimate_trend(df, method=method)
                assert len(result) == len(df)


class TestEdgeCases:
    """Test edge cases for advanced methods."""

    @pytest.mark.skipif(not HAS_ADVANCED, reason="Advanced methods not available")
    def test_small_datasets(self):
        """Test advanced methods with small datasets."""
        small_df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})

        # LOESS should handle small datasets gracefully
        try:
            result = loess_trend(small_df, frac=0.8)  # Large bandwidth for small data
            assert len(result) == len(small_df)
        except (ValueError, np.linalg.LinAlgError):
            pass  # Acceptable to fail on very small data

        # L1 filter should work
        result = l1_trend_filter(small_df, lambda_param=0.1)
        assert len(result) == len(small_df)

    @pytest.mark.skipif(not HAS_ADVANCED, reason="Advanced methods not available")
    def test_irregular_time_series(self):
        """Test with irregular time spacing."""
        irregular_times = np.sort(np.random.uniform(0, 10, 30))
        df = pd.DataFrame({"time": irregular_times, "value": np.random.randn(30)})

        result = loess_trend(df, time_column="time")
        assert len(result) == len(df)
        assert "time" in result.columns

    @pytest.mark.skipif(not HAS_ADVANCED, reason="Advanced methods not available")
    def test_missing_values(self):
        """Test handling of missing values."""
        data_with_nan = np.random.randn(30)
        data_with_nan[5:10] = np.nan

        df = pd.DataFrame({"value": data_with_nan})

        # Methods should handle NaN gracefully
        result = loess_trend(df)
        assert len(result) == len(df)


if __name__ == "__main__":
    pytest.main([__file__])
