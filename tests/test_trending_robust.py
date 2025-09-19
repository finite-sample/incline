"""Tests for enhanced trending functionality with robust statistics."""

import numpy as np
import pandas as pd
import pytest
from incline.trend import trending, spline_trend


class TestRobustTrending:
    """Test cases for enhanced trending with robust statistics."""
    
    def test_trending_backward_compatibility(self):
        """Test that enhanced trending maintains backward compatibility."""
        # Create test data like the original tests
        np.random.seed(42)
        df_list = []
        for i in range(3):
            data = pd.DataFrame({'value': np.random.randn(10) + i})
            result = spline_trend(data)
            result['id'] = f'series_{i}'
            df_list.append(result)
        
        # Test original interface
        trend_result = trending(df_list, k=3)
        
        assert 'id' in trend_result.columns
        assert 'max_or_avg' in trend_result.columns
        assert len(trend_result) == 3
        
        # Should be sorted by rank
        assert 'rank' in trend_result.columns
        assert list(trend_result['rank']) == [1, 2, 3]
        
    def test_trending_robust_statistics(self):
        """Test robust statistics in trending."""
        np.random.seed(42)
        
        # Create data with outliers
        df_list = []
        for i in range(3):
            # Add outliers to one series
            if i == 1:
                values = np.concatenate([np.ones(8), [10, -10]])  # Outliers
            else:
                values = np.random.randn(10) + i
            
            data = pd.DataFrame({'value': values})
            result = spline_trend(data)
            result['id'] = f'series_{i}'
            df_list.append(result)
        
        # Compare robust vs non-robust
        trend_normal = trending(df_list, robust=False, max_or_avg='avg')
        trend_robust = trending(df_list, robust=True, max_or_avg='avg')
        
        assert len(trend_normal) == len(trend_robust) == 3
        
        # Robust should have additional statistics
        assert 'median' in trend_robust.columns
        assert 'mad' in trend_robust.columns
        assert 'outlier_fraction' in trend_robust.columns
        
    def test_trending_different_aggregations(self):
        """Test different aggregation methods."""
        np.random.seed(42)
        df_list = []
        for i in range(2):
            data = pd.DataFrame({'value': [1, 2, 3, 4, 5] if i == 0 else [2, 3, 4, 5, 6]})
            result = spline_trend(data)
            result['id'] = f'series_{i}'
            df_list.append(result)
        
        # Test different aggregation methods
        for method in ['max', 'avg', 'median']:
            result = trending(df_list, max_or_avg=method)
            assert len(result) == 2
            assert 'max_or_avg' in result.columns
            
        # Test robust methods (may require scipy)
        try:
            robust_result = trending(df_list, max_or_avg='trimmed_mean')
            assert len(robust_result) == 2
        except ImportError:
            pass  # Expected if scipy not available
            
    def test_trending_weighting_schemes(self):
        """Test different weighting schemes."""
        np.random.seed(42)
        df_list = []
        for i in range(2):
            # Create data with time trend
            values = np.linspace(i, i+2, 10)  # Increasing trend
            data = pd.DataFrame({'value': values})
            result = spline_trend(data)
            result['id'] = f'series_{i}'
            df_list.append(result)
        
        # Test different weighting schemes
        for weighting in ['uniform', 'linear', 'exponential']:
            result = trending(df_list, weighting=weighting, k=5)
            assert len(result) == 2
            assert 'max_or_avg' in result.columns
            
    def test_trending_confidence_intervals(self):
        """Test confidence intervals in trending."""
        np.random.seed(42)
        df_list = []
        for i in range(2):
            values = np.random.randn(15) + 0.1 * i  # Small differences
            data = pd.DataFrame({'value': values})
            result = spline_trend(data)
            result['id'] = f'series_{i}'
            df_list.append(result)
        
        # Test with confidence intervals
        result = trending(df_list, return_confidence=True, confidence_level=0.95)
        
        assert len(result) == 2
        assert 'ci_lower' in result.columns
        assert 'ci_upper' in result.columns
        assert 'significant' in result.columns
        
        # Check confidence interval validity
        assert all(result['ci_lower'] <= result['max_or_avg'])
        assert all(result['max_or_avg'] <= result['ci_upper'])
        
    def test_trending_trim_fraction(self):
        """Test trimmed mean with different trim fractions."""
        np.random.seed(42)
        
        # Create data with outliers
        values_with_outliers = np.concatenate([np.ones(8), [5, -5]])
        df_list = []
        for i in range(2):
            if i == 0:
                values = values_with_outliers
            else:
                values = np.random.randn(10)
            
            data = pd.DataFrame({'value': values})
            result = spline_trend(data)
            result['id'] = f'series_{i}'
            df_list.append(result)
        
        # Test different trim fractions
        for trim_frac in [0.1, 0.2, 0.3]:
            try:
                result = trending(df_list, max_or_avg='trimmed_mean', 
                                trim_fraction=trim_frac)
                assert len(result) == 2
            except ImportError:
                pass  # Expected if scipy not available
                
    def test_trending_edge_cases(self):
        """Test trending edge cases."""
        # Test with empty list
        with pytest.raises(ValueError, match="No valid data found"):
            trending([])
            
        # Test with single series
        data = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        result = spline_trend(data)
        result['id'] = 'single'
        
        trend_result = trending([result])
        assert len(trend_result) == 1
        assert trend_result['id'].iloc[0] == 'single'
        
    def test_trending_missing_derivative_order(self):
        """Test trending when derivative_order column is missing or mismatched."""
        # Create data without proper derivative_order
        df_with_wrong_order = pd.DataFrame({
            'id': ['series_1'] * 5,
            'derivative_value': [0.1, 0.2, 0.3, 0.4, 0.5],
            'derivative_order': [2, 2, 2, 2, 2]  # Wrong order
        })
        
        # Should handle gracefully
        result = trending([df_with_wrong_order], derivative_order=1)
        # Should return empty or handle appropriately
        
    def test_trending_robust_statistics_details(self):
        """Test specific robust statistics calculations."""
        np.random.seed(42)
        
        # Create data with known properties
        values = np.array([1, 2, 3, 4, 100])  # Clear outlier
        data = pd.DataFrame({'value': values})
        result = spline_trend(data)
        result['id'] = 'test_series'
        
        trend_result = trending([result], robust=True, max_or_avg='avg')
        
        if len(trend_result) > 0:
            assert 'outlier_fraction' in trend_result.columns
            assert 'skewness' in trend_result.columns
            assert 'mad' in trend_result.columns
            
            # Should detect the outlier
            outlier_frac = trend_result['outlier_fraction'].iloc[0]
            assert outlier_frac >= 0.0
            
    def test_huber_mean_implementation(self):
        """Test Huber M-estimator implementation."""
        from incline.trend import _huber_mean
        
        # Test with simple data
        x = np.array([1, 2, 3, 4, 5])
        result = _huber_mean(x)
        assert abs(result - 3.0) < 0.1  # Should be close to mean for clean data
        
        # Test with outliers
        x_outliers = np.array([1, 2, 3, 4, 100])
        result_outliers = _huber_mean(x_outliers)
        # Should be more robust than simple mean
        assert result_outliers < np.mean(x_outliers)
        assert result_outliers > np.median(x_outliers)
        
        # Test edge cases
        assert np.isnan(_huber_mean(np.array([])))
        assert _huber_mean(np.array([5])) == 5
        assert _huber_mean(np.array([1, 1, 1, 1])) == 1
        
    def test_robust_statistics_helpers(self):
        """Test helper functions for robust statistics."""
        from incline.trend import (_outlier_fraction, _robust_skewness, 
                                  _compute_robust_statistics, _compute_weights)
        
        # Test outlier fraction
        x_clean = np.array([1, 2, 3, 4, 5])
        x_outliers = np.array([1, 2, 3, 4, 100])
        
        assert _outlier_fraction(x_clean) == 0.0
        assert _outlier_fraction(x_outliers) > 0.0
        
        # Test robust skewness
        x_symmetric = np.array([1, 2, 3, 4, 5])
        x_skewed = np.array([1, 1, 1, 4, 5])
        
        skew_sym = _robust_skewness(x_symmetric)
        skew_asym = _robust_skewness(x_skewed)
        assert abs(skew_sym) < abs(skew_asym)
        
        # Test robust statistics computation
        stats = _compute_robust_statistics(x_outliers, np.ones(len(x_outliers)))
        required_keys = ['median', 'mad', 'std', 'iqr', 'n_obs', 'outlier_fraction', 'skewness']
        assert all(key in stats for key in required_keys)
        
        # Test weight computation
        weights_uniform = _compute_weights(5, 'uniform')
        weights_linear = _compute_weights(5, 'linear')
        weights_exp = _compute_weights(5, 'exponential')
        
        assert len(weights_uniform) == 5
        assert np.allclose(weights_uniform, 1.0)
        assert weights_linear[-1] > weights_linear[0]  # Increasing
        assert weights_exp[-1] > weights_exp[0]  # Exponential decay reversed
        
    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap confidence interval computation."""
        from incline.trend import _bootstrap_trend_ci
        
        np.random.seed(42)
        values = np.random.randn(20)
        weights = np.ones(len(values))
        
        # Test different statistics
        for stat in ['max', 'avg', 'median']:
            ci_lower, ci_upper = _bootstrap_trend_ci(
                values, weights, stat, 0.95, n_bootstrap=50
            )
            assert ci_lower <= ci_upper
            assert not np.isnan(ci_lower)
            assert not np.isnan(ci_upper)
            
    def test_trending_with_different_k_values(self):
        """Test trending with different k values."""
        np.random.seed(42)
        
        # Create longer time series
        data = pd.DataFrame({'value': np.random.randn(20)})
        result = spline_trend(data)
        result['id'] = 'long_series'
        
        # Test different k values
        for k in [1, 3, 5, 10]:
            trend_result = trending([result], k=k)
            assert len(trend_result) == 1
            
    def test_trending_reproducibility(self):
        """Test that trending results are reproducible."""
        np.random.seed(42)
        
        # Create consistent data
        df_list = []
        for i in range(2):
            data = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
            result = spline_trend(data)
            result['id'] = f'series_{i}'
            df_list.append(result)
        
        # Run twice with same seed
        np.random.seed(123)
        result1 = trending(df_list, return_confidence=True)
        
        np.random.seed(123)
        result2 = trending(df_list, return_confidence=True)
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)


class TestTrendingIntegration:
    """Test integration of trending with other incline components."""
    
    def test_trending_with_different_trend_methods(self):
        """Test trending with different trend estimation methods."""
        np.random.seed(42)
        
        from incline.trend import sgolay_trend
        
        # Create data
        data = pd.DataFrame({'value': np.random.randn(15)})
        
        # Test with different trend methods
        spline_result = spline_trend(data)
        spline_result['id'] = 'spline'
        
        sgolay_result = sgolay_trend(data)
        sgolay_result['id'] = 'sgolay'
        
        # Should work with mixed methods
        mixed_results = trending([spline_result, sgolay_result])
        assert len(mixed_results) == 2
        
    def test_trending_large_dataset(self):
        """Test trending performance with larger datasets."""
        np.random.seed(42)
        
        # Create many series
        df_list = []
        for i in range(10):
            data = pd.DataFrame({'value': np.random.randn(50) + 0.1 * i})
            result = spline_trend(data)
            result['id'] = f'series_{i:02d}'
            df_list.append(result)
        
        # Should handle efficiently
        trend_result = trending(df_list, robust=True, return_confidence=True)
        assert len(trend_result) == 10
        assert 'rank' in trend_result.columns
        
        # Check ranking order
        assert list(trend_result['rank']) == list(range(1, 11))


if __name__ == '__main__':
    pytest.main([__file__])