"""Tests for incline.seasonal module."""

import numpy as np
import pandas as pd
import pytest

try:
    from incline.seasonal import (
        detect_seasonality, stl_decompose, simple_deseasonalize,
        trend_with_deseasonalization, deseasonalize_pipeline
    )
    HAS_SEASONAL = True
except ImportError:
    HAS_SEASONAL = False


@pytest.mark.skipif(not HAS_SEASONAL, reason="Seasonal methods not available")
class TestSeasonalityDetection:
    """Test cases for seasonality detection."""
    
    def test_detect_seasonality_synthetic(self):
        """Test seasonality detection with synthetic data."""
        # Create seasonal data
        t = np.arange(100)
        seasonal = 2 * np.sin(2 * np.pi * t / 12)  # Period 12
        trend = 0.1 * t
        noise = 0.1 * np.random.randn(100)
        y = trend + seasonal + noise
        
        df = pd.DataFrame({'value': y})
        result = detect_seasonality(df, max_period=50)
        
        assert isinstance(result, dict)
        assert 'seasonal' in result
        assert 'period' in result
        assert 'strength' in result
        assert 'method' in result
        
        # Should detect seasonality
        if result['seasonal']:
            assert result['period'] is not None
            assert result['strength'] > 0
            
    def test_detect_seasonality_no_pattern(self):
        """Test seasonality detection with random data."""
        # Pure random data
        y = np.random.randn(100)
        df = pd.DataFrame({'value': y})
        
        result = detect_seasonality(df)
        
        assert isinstance(result, dict)
        assert 'seasonal' in result
        
        # Should not detect strong seasonality in random data
        if not result['seasonal']:
            assert result['period'] is None
            assert result['strength'] == 0.0
            
    def test_detect_seasonality_parameters(self):
        """Test seasonality detection with different parameters."""
        # Create data with known period
        t = np.arange(60)
        y = np.sin(2 * np.pi * t / 10) + 0.1 * np.random.randn(60)
        df = pd.DataFrame({'value': y})
        
        # Test with different max_period
        result1 = detect_seasonality(df, max_period=5)
        result2 = detect_seasonality(df, max_period=20)
        
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)


@pytest.mark.skipif(not HAS_SEASONAL, reason="Seasonal methods not available")
class TestSTLDecompose:
    """Test cases for STL decomposition."""
    
    def test_stl_decompose_basic(self):
        """Test basic STL decomposition."""
        # Create seasonal time series
        dates = pd.date_range('2020-01-01', periods=120, freq='D')
        t = np.arange(120)
        seasonal = 2 * np.sin(2 * np.pi * t / 30)  # Monthly pattern
        trend = 0.05 * t
        noise = 0.2 * np.random.randn(120)
        y = trend + seasonal + noise
        
        df = pd.DataFrame({'value': y}, index=dates)
        result = stl_decompose(df, period=30)
        
        assert 'trend_component' in result.columns
        assert 'seasonal_component' in result.columns
        assert 'residual_component' in result.columns
        assert 'deseasonalized' in result.columns
        assert 'decomposition_method' in result.columns
        assert result['decomposition_method'].iloc[0] == 'stl'
        assert len(result) == len(df)
        
    def test_stl_decompose_auto_period(self):
        """Test STL with automatic period detection."""
        # Create data with clear seasonal pattern
        t = np.arange(100)
        y = np.sin(2 * np.pi * t / 12) + 0.1 * t + 0.1 * np.random.randn(100)
        df = pd.DataFrame({'value': y})
        
        # Should work without specifying period
        result = stl_decompose(df)
        assert len(result) == len(df)
        assert 'period' in result.columns
        
    def test_stl_decompose_parameters(self):
        """Test STL with different parameters."""
        t = np.arange(80)
        y = np.sin(2 * np.pi * t / 20) + 0.1 * np.random.randn(80)
        df = pd.DataFrame({'value': y})
        
        # Test with different parameters
        result1 = stl_decompose(df, period=20, seasonal=7, robust=True)
        result2 = stl_decompose(df, period=20, seasonal=11, robust=False)
        
        assert len(result1) == len(df)
        assert len(result2) == len(df)


@pytest.mark.skipif(not HAS_SEASONAL, reason="Seasonal methods not available")
class TestSimpleDeseasonalize:
    """Test cases for simple deseasonalization."""
    
    def test_simple_deseasonalize_basic(self):
        """Test basic simple deseasonalization."""
        # Create seasonal data
        t = np.arange(60)
        seasonal = np.sin(2 * np.pi * t / 12)
        trend = 0.1 * t
        y = trend + seasonal + 0.1 * np.random.randn(60)
        
        df = pd.DataFrame({'value': y})
        result = simple_deseasonalize(df, period=12)
        
        assert 'trend_component' in result.columns
        assert 'seasonal_component' in result.columns
        assert 'residual_component' in result.columns
        assert 'deseasonalized' in result.columns
        assert 'decomposition_method' in result.columns
        assert result['decomposition_method'].iloc[0] == 'simple'
        assert len(result) == len(df)
        
    def test_simple_deseasonalize_no_seasonality(self):
        """Test simple deseasonalization with no clear seasonality."""
        # Random data
        y = np.random.randn(50)
        df = pd.DataFrame({'value': y})
        
        result = simple_deseasonalize(df)
        assert len(result) == len(df)
        assert 'deseasonalized' in result.columns
        
    def test_simple_deseasonalize_edge_cases(self):
        """Test simple deseasonalization edge cases."""
        # Very short series
        short_df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        result = simple_deseasonalize(short_df, period=10)  # Period larger than data
        assert len(result) == len(short_df)
        
        # Period too large
        df = pd.DataFrame({'value': np.random.randn(20)})
        result = simple_deseasonalize(df, period=30)
        assert len(result) == len(df)


@pytest.mark.skipif(not HAS_SEASONAL, reason="Seasonal methods not available")
class TestTrendWithDeseasonalization:
    """Test cases for trend estimation with deseasonalization."""
    
    def test_trend_with_deseasonalization_basic(self):
        """Test basic trend estimation with deseasonalization."""
        # Create seasonal data
        t = np.arange(100)
        seasonal = 2 * np.sin(2 * np.pi * t / 20)
        trend = 0.05 * t + 0.001 * t**2  # Quadratic trend
        noise = 0.2 * np.random.randn(100)
        y = trend + seasonal + noise
        
        df = pd.DataFrame({'value': y})
        result = trend_with_deseasonalization(df, trend_method='spline')
        
        assert 'trend_smoothed_value' in result.columns or 'smoothed_value' in result.columns
        assert 'seasonality_detected' in result.columns
        assert 'seasonality_strength' in result.columns
        assert len(result) == len(df)
        
    def test_trend_with_deseasonalization_no_seasonality(self):
        """Test with data that has no seasonality."""
        # Non-seasonal data
        t = np.arange(50)
        y = 0.1 * t + 0.1 * np.random.randn(50)
        df = pd.DataFrame({'value': y})
        
        result = trend_with_deseasonalization(df, trend_method='spline')
        assert len(result) == len(df)
        assert 'seasonality_detected' in result.columns
        
    def test_trend_with_deseasonalization_methods(self):
        """Test different trend methods."""
        t = np.arange(60)
        y = np.sin(2 * np.pi * t / 15) + 0.05 * t + 0.1 * np.random.randn(60)
        df = pd.DataFrame({'value': y})
        
        # Test different trend methods
        for method in ['spline', 'sgolay']:
            result = trend_with_deseasonalization(df, trend_method=method)
            assert len(result) == len(df)
        
        # Test different decomposition methods
        result1 = trend_with_deseasonalization(df, decomposition_method='simple')
        result2 = trend_with_deseasonalization(df, decomposition_method='auto')
        
        assert len(result1) == len(df)
        assert len(result2) == len(df)


class TestDeseasonalizePipeline:
    """Test cases for deseasonalization pipeline."""
    
    @pytest.mark.skipif(not HAS_SEASONAL, reason="Seasonal methods not available")
    def test_deseasonalize_pipeline_basic(self):
        """Test basic pipeline functionality."""
        t = np.arange(50)
        y = np.sin(2 * np.pi * t / 12) + 0.1 * np.random.randn(50)
        df = pd.DataFrame({'value': y})
        
        # Test pipeline creation and usage
        pipeline = deseasonalize_pipeline('simple', period=12)
        result = pipeline(df)
        
        assert len(result) == len(df)
        assert 'deseasonalized' in result.columns
        
    @pytest.mark.skipif(not HAS_SEASONAL, reason="Seasonal methods not available")
    def test_deseasonalize_pipeline_auto(self):
        """Test pipeline with auto method selection."""
        df = pd.DataFrame({'value': np.random.randn(40)})
        
        pipeline = deseasonalize_pipeline('auto')
        result = pipeline(df)
        assert len(result) == len(df)


class TestEdgeCases:
    """Test edge cases for seasonal methods."""
    
    @pytest.mark.skipif(not HAS_SEASONAL, reason="Seasonal methods not available")
    def test_constant_data(self):
        """Test with constant data."""
        df = pd.DataFrame({'value': np.ones(50)})
        
        result = detect_seasonality(df)
        assert not result['seasonal']  # No seasonality in constant data
        
        decomp_result = simple_deseasonalize(df)
        assert len(decomp_result) == len(df)
        
    @pytest.mark.skipif(not HAS_SEASONAL, reason="Seasonal methods not available")
    def test_linear_trend_only(self):
        """Test with pure linear trend."""
        t = np.arange(50)
        y = 2 * t + 1  # Pure linear trend
        df = pd.DataFrame({'value': y})
        
        result = detect_seasonality(df)
        # May or may not detect seasonality in linear trend
        
        decomp_result = simple_deseasonalize(df)
        assert len(decomp_result) == len(df)
        
    @pytest.mark.skipif(not HAS_SEASONAL, reason="Seasonal methods not available")
    def test_missing_values(self):
        """Test handling of missing values."""
        t = np.arange(60)
        y = np.sin(2 * np.pi * t / 12) + 0.1 * t
        y[10:15] = np.nan  # Introduce missing values
        
        df = pd.DataFrame({'value': y})
        
        # Should handle missing values gracefully
        result = simple_deseasonalize(df, period=12)
        assert len(result) == len(df)
        
    @pytest.mark.skipif(not HAS_SEASONAL, reason="Seasonal methods not available")
    def test_very_short_series(self):
        """Test with very short time series."""
        df = pd.DataFrame({'value': [1, 2, 3]})
        
        result = detect_seasonality(df)
        assert not result['seasonal']  # Too short for seasonality
        
        decomp_result = simple_deseasonalize(df)
        assert len(decomp_result) == len(df)


if __name__ == '__main__':
    pytest.main([__file__])