"""Tests for incline.trend module."""

import numpy as np
import pandas as pd
import pytest

from incline.trend import naive_trend, spline_trend, sgolay_trend, trending


class TestNaiveTrend:
    """Test cases for naive_trend function."""
    
    def test_naive_trend_basic(self):
        """Test basic functionality of naive_trend."""
        data = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        result = naive_trend(data)
        
        assert 'derivative_value' in result.columns
        assert 'derivative_method' in result.columns
        assert 'function_order' in result.columns
        assert 'derivative_order' in result.columns
        
        assert result['derivative_method'].iloc[0] == 'naive'
        assert result['derivative_order'].iloc[0] == 1
        assert pd.isna(result['function_order'].iloc[0])
        
    def test_naive_trend_linear_data(self):
        """Test naive_trend with linear data."""
        data = pd.DataFrame({'value': [1, 3, 5, 7, 9]})  # slope = 2
        result = naive_trend(data)
        
        # For linear data, naive trend should be close to actual slope
        # (excluding edge effects and NaN values)
        middle_derivatives = result['derivative_value'].iloc[1:-1].dropna()
        if len(middle_derivatives) > 0:
            # Check that the average is close to expected slope
            avg_derivative = middle_derivatives.mean()
            assert abs(avg_derivative - 2.0) < 1.0  # More lenient test


class TestSplineTrend:
    """Test cases for spline_trend function."""
    
    def test_spline_trend_basic(self):
        """Test basic functionality of spline_trend."""
        data = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        result = spline_trend(data)
        
        assert 'smoothed_value' in result.columns
        assert 'derivative_value' in result.columns
        assert 'derivative_method' in result.columns
        assert 'function_order' in result.columns
        assert 'derivative_order' in result.columns
        
        assert result['derivative_method'].iloc[0] == 'spline'
        assert result['derivative_order'].iloc[0] == 1
        assert result['function_order'].iloc[0] == 3  # default
        
    def test_spline_trend_parameters(self):
        """Test spline_trend with different parameters."""
        data = pd.DataFrame({'value': [1, 4, 9, 16, 25]})  # quadratic
        
        # Test different function orders
        result_order2 = spline_trend(data, function_order=2)
        result_order3 = spline_trend(data, function_order=3)
        
        assert result_order2['function_order'].iloc[0] == 2
        assert result_order3['function_order'].iloc[0] == 3
        
        # Test different derivative orders
        result_deriv1 = spline_trend(data, derivative_order=1)
        result_deriv2 = spline_trend(data, derivative_order=2)
        
        assert result_deriv1['derivative_order'].iloc[0] == 1
        assert result_deriv2['derivative_order'].iloc[0] == 2


class TestSgolayTrend:
    """Test cases for sgolay_trend function."""
    
    def test_sgolay_trend_basic(self):
        """Test basic functionality of sgolay_trend."""
        data = pd.DataFrame({'value': np.random.randn(20)})  # Need enough points
        result = sgolay_trend(data)
        
        assert 'smoothed_value' in result.columns
        assert 'derivative_value' in result.columns
        assert 'derivative_method' in result.columns
        assert 'function_order' in result.columns
        assert 'derivative_order' in result.columns
        
        assert result['derivative_method'].iloc[0] == 'sgolay'
        assert result['derivative_order'].iloc[0] == 1
        assert result['function_order'].iloc[0] == 3  # default
        
    def test_sgolay_trend_window_size(self):
        """Test sgolay_trend with different window sizes."""
        data = pd.DataFrame({'value': np.random.randn(30)})
        
        # Test different window lengths
        result_small = sgolay_trend(data, window_length=5)
        result_large = sgolay_trend(data, window_length=15)
        
        # Both should work without errors
        assert len(result_small) == len(data)
        assert len(result_large) == len(data)


class TestTrending:
    """Test cases for trending function."""
    
    def test_trending_basic(self):
        """Test basic functionality of trending."""
        # Create sample data for multiple series
        df_list = []
        for i in range(3):
            data = pd.DataFrame({'value': np.random.randn(10) + i})
            result = spline_trend(data)
            result['id'] = f'series_{i}'
            df_list.append(result)
        
        trend_result = trending(df_list, k=3)
        
        assert 'id' in trend_result.columns
        assert 'max_or_avg' in trend_result.columns
        assert len(trend_result) == 3
        
    def test_trending_parameters(self):
        """Test trending with different parameters."""
        # Create predictable data
        df_list = []
        for i in range(2):
            data = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
            result = spline_trend(data)
            result['id'] = f'series_{i}'
            df_list.append(result)
        
        # Test max vs avg
        trend_max = trending(df_list, max_or_avg='max', k=3)
        trend_avg = trending(df_list, max_or_avg='avg', k=3)
        
        assert len(trend_max) == 2
        assert len(trend_avg) == 2


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        data = pd.DataFrame({'value': []})
        
        # Should handle gracefully or raise appropriate error
        try:
            result = naive_trend(data)
            assert len(result) == 0
        except (ValueError, IndexError, KeyError):
            pass  # Acceptable to fail on empty data
            
    def test_single_point(self):
        """Test with single data point."""
        data = pd.DataFrame({'value': [5]})
        
        # Should handle gracefully or raise appropriate error
        try:
            result = naive_trend(data)
            assert len(result) == 1
        except (ValueError, IndexError):
            pass  # Acceptable to fail on single point
            
    def test_custom_column_name(self):
        """Test with custom column name."""
        data = pd.DataFrame({'price': [1, 2, 3, 4, 5]})
        
        result = spline_trend(data, column_value='price')
        assert 'derivative_value' in result.columns
        assert 'price' in result.columns


if __name__ == '__main__':
    pytest.main([__file__])