"""Tests for incline.testing module."""

import numpy as np
import pandas as pd
import pytest

try:
    from incline.testing import (
        PolynomialTrend, SinusoidalTrend, ExponentialTrend, StepTrend,
        NoiseGenerator, generate_time_series, benchmark_method,
        run_comprehensive_benchmark, get_standard_test_functions,
        SimulationResult
    )
    HAS_TESTING = True
except ImportError:
    HAS_TESTING = False


@pytest.mark.skipif(not HAS_TESTING, reason="Testing framework not available")
class TestTrendFunctions:
    """Test cases for trend function classes."""
    
    def test_polynomial_trend_basic(self):
        """Test basic PolynomialTrend functionality."""
        # Test linear function: 2 + 3x
        poly = PolynomialTrend([2, 3])
        x = np.array([0, 1, 2, 3])
        
        # Test function evaluation
        y = poly(x)
        expected_y = 2 + 3 * x
        np.testing.assert_array_almost_equal(y, expected_y)
        
        # Test first derivative
        dy = poly.derivative(x, order=1)
        expected_dy = np.full_like(x, 3.0)  # Derivative of 2 + 3x is 3
        np.testing.assert_array_almost_equal(dy, expected_dy)
        
        # Test second derivative
        d2y = poly.derivative(x, order=2)
        expected_d2y = np.zeros_like(x)  # Second derivative of linear is 0
        np.testing.assert_array_almost_equal(d2y, expected_d2y)
        
    def test_polynomial_trend_quadratic(self):
        """Test PolynomialTrend with quadratic function."""
        # Test quadratic: 1 + 2x + 3x^2
        poly = PolynomialTrend([1, 2, 3])
        x = np.array([0, 1, 2])
        
        # Test function evaluation
        y = poly(x)
        expected_y = 1 + 2 * x + 3 * x**2
        np.testing.assert_array_almost_equal(y, expected_y)
        
        # Test first derivative: 2 + 6x
        dy = poly.derivative(x, order=1)
        expected_dy = 2 + 6 * x
        np.testing.assert_array_almost_equal(dy, expected_dy)
        
        # Test second derivative: 6
        d2y = poly.derivative(x, order=2)
        expected_d2y = np.full_like(x, 6.0)
        np.testing.assert_array_almost_equal(d2y, expected_d2y)
        
    def test_polynomial_trend_cubic(self):
        """Test PolynomialTrend with cubic function."""
        # Test cubic: 1 + x + x^2 + x^3
        poly = PolynomialTrend([1, 1, 1, 1])
        x = np.array([0, 1, 2])
        
        # Test function evaluation
        y = poly(x)
        expected_y = 1 + x + x**2 + x**3
        np.testing.assert_array_almost_equal(y, expected_y)
        
        # Test first derivative: 1 + 2x + 3x^2
        dy = poly.derivative(x, order=1)
        expected_dy = 1 + 2 * x + 3 * x**2
        np.testing.assert_array_almost_equal(dy, expected_dy)
        
        # Test second derivative: 2 + 6x
        d2y = poly.derivative(x, order=2)
        expected_d2y = 2 + 6 * x
        np.testing.assert_array_almost_equal(d2y, expected_d2y)
        
    def test_polynomial_trend_name(self):
        """Test PolynomialTrend name property."""
        poly1 = PolynomialTrend([1, 2])  # Linear
        poly2 = PolynomialTrend([1, 2, 3])  # Quadratic
        
        assert poly1.name == "Polynomial(deg=1)"
        assert poly2.name == "Polynomial(deg=2)"
        
    def test_sinusoidal_trend_basic(self):
        """Test SinusoidalTrend functionality."""
        # Test basic sine wave
        sin_trend = SinusoidalTrend(amplitude=2.0, frequency=0.5, phase=0.0)
        x = np.array([0, 0.5, 1.0, 1.5, 2.0])
        
        # Test function evaluation
        y = sin_trend(x)
        expected_y = 2.0 * np.sin(2 * np.pi * 0.5 * x)
        np.testing.assert_array_almost_equal(y, expected_y)
        
        # Test first derivative
        dy = sin_trend.derivative(x, order=1)
        expected_dy = 2.0 * 2 * np.pi * 0.5 * np.cos(2 * np.pi * 0.5 * x)
        np.testing.assert_array_almost_equal(dy, expected_dy)
        
        # Test second derivative
        d2y = sin_trend.derivative(x, order=2)
        expected_d2y = -2.0 * (2 * np.pi * 0.5)**2 * np.sin(2 * np.pi * 0.5 * x)
        np.testing.assert_array_almost_equal(d2y, expected_d2y)
        
    def test_exponential_trend_basic(self):
        """Test ExponentialTrend functionality."""
        exp_trend = ExponentialTrend(scale=2.0, rate=0.1)
        x = np.array([0, 1, 2, 3])
        
        # Test function evaluation
        y = exp_trend(x)
        expected_y = 2.0 * np.exp(0.1 * x)
        np.testing.assert_array_almost_equal(y, expected_y)
        
        # Test first derivative
        dy = exp_trend.derivative(x, order=1)
        expected_dy = 2.0 * 0.1 * np.exp(0.1 * x)
        np.testing.assert_array_almost_equal(dy, expected_dy)
        
        # Test second derivative
        d2y = exp_trend.derivative(x, order=2)
        expected_d2y = 2.0 * (0.1**2) * np.exp(0.1 * x)
        np.testing.assert_array_almost_equal(d2y, expected_d2y)
        
    def test_step_trend_basic(self):
        """Test StepTrend functionality."""
        step_trend = StepTrend([3, 6, 9], [1, 2, 3, 4])
        x = np.array([0, 2, 4, 7, 10])
        
        # Test function evaluation
        y = step_trend(x)
        expected_y = np.array([1, 1, 2, 3, 4])  # Based on breakpoints
        np.testing.assert_array_almost_equal(y, expected_y)
        
        # Test derivative (should be zero except at breakpoints)
        dy = step_trend.derivative(x, order=1)
        expected_dy = np.zeros_like(x)
        np.testing.assert_array_almost_equal(dy, expected_dy)


@pytest.mark.skipif(not HAS_TESTING, reason="Testing framework not available")
class TestNoiseGenerator:
    """Test cases for noise generation."""
    
    def test_white_noise(self):
        """Test white noise generation."""
        np.random.seed(42)
        noise = NoiseGenerator.white_noise(100, std=1.0, random_state=42)
        
        assert len(noise) == 100
        assert abs(np.std(noise) - 1.0) < 0.2  # Should be close to specified std
        assert abs(np.mean(noise)) < 0.2  # Should be close to zero mean
        
    def test_ar1_noise(self):
        """Test AR(1) noise generation."""
        np.random.seed(42)
        noise = NoiseGenerator.ar1_noise(100, phi=0.7, std=1.0, random_state=42)
        
        assert len(noise) == 100
        # AR(1) should have some autocorrelation
        autocorr = np.corrcoef(noise[:-1], noise[1:])[0, 1]
        assert autocorr > 0.3  # Should be positively correlated
        
    def test_seasonal_noise(self):
        """Test seasonal noise generation."""
        np.random.seed(42)
        noise = NoiseGenerator.seasonal_noise(60, period=12, amplitude=1.0, random_state=42)
        
        assert len(noise) == 60
        # Should have some periodicity
        # Test by checking if there's power at the seasonal frequency
        fft_vals = np.fft.fft(noise)
        freqs = np.fft.fftfreq(60)
        seasonal_freq_idx = np.argmin(np.abs(freqs - 1/12))
        seasonal_power = np.abs(fft_vals[seasonal_freq_idx])
        total_power = np.sum(np.abs(fft_vals))
        
        # Seasonal frequency should have significant power
        assert seasonal_power / total_power > 0.1


@pytest.mark.skipif(not HAS_TESTING, reason="Testing framework not available")
class TestTimeSeriesGeneration:
    """Test cases for time series generation."""
    
    def test_generate_time_series_basic(self):
        """Test basic time series generation."""
        trend_func = PolynomialTrend([1, 0.5])  # 1 + 0.5x
        df, true_deriv = generate_time_series(
            trend_func, n_points=50, noise_std=0.1, random_state=42
        )
        
        assert len(df) == 50
        assert len(true_deriv) == 50
        assert 'value' in df.columns
        assert 'true_value' in df.columns
        assert 'noise' in df.columns
        
        # True derivative should be close to 0.5 for linear trend
        np.testing.assert_array_almost_equal(true_deriv, 0.5, decimal=6)
        
    def test_generate_time_series_irregular(self):
        """Test time series generation with irregular spacing."""
        trend_func = PolynomialTrend([0, 1])  # x
        df, true_deriv = generate_time_series(
            trend_func, n_points=30, irregular_spacing=True, random_state=42
        )
        
        assert len(df) == 30
        assert 'time' in df.columns  # Should have time column for irregular spacing
        
        # Time values should be irregular (not evenly spaced)
        time_diffs = np.diff(df['time'])
        assert np.std(time_diffs) > 0  # Should have varying time differences
        
    def test_generate_time_series_missing_data(self):
        """Test time series generation with missing data."""
        trend_func = PolynomialTrend([2, 1])
        df, true_deriv = generate_time_series(
            trend_func, n_points=50, missing_data_prob=0.2, random_state=42
        )
        
        assert len(df) == 50
        # Should have some missing values
        missing_count = df['value'].isna().sum()
        assert missing_count > 0
        
    def test_generate_time_series_noise_types(self):
        """Test different noise types."""
        trend_func = PolynomialTrend([1, 0.1])
        
        # Test different noise types
        for noise_type in ['white', 'ar1', 'seasonal']:
            df, true_deriv = generate_time_series(
                trend_func, n_points=40, noise_type=noise_type, random_state=42
            )
            assert len(df) == 40
            assert not np.any(np.isnan(true_deriv))


@pytest.mark.skipif(not HAS_TESTING, reason="Testing framework not available")
class TestBenchmarking:
    """Test cases for benchmarking functionality."""
    
    def test_simulation_result_creation(self):
        """Test SimulationResult dataclass."""
        result = SimulationResult(
            method='test_method',
            mse_derivative=0.1,
            bias_derivative=0.05,
            coverage_95=0.95,
            coverage_90=0.90,
            mean_ci_width=0.2,
            computation_time=0.01,
            parameters={'param1': 1.0}
        )
        
        assert result.method == 'test_method'
        assert result.mse_derivative == 0.1
        assert result.parameters['param1'] == 1.0
        
    def test_benchmark_method_basic(self):
        """Test basic method benchmarking."""
        # Create synthetic data with known derivative
        trend_func = PolynomialTrend([0, 1])  # Linear with slope 1
        df, true_deriv = generate_time_series(
            trend_func, n_points=30, noise_std=0.1, random_state=42
        )
        
        # Define a simple test method
        def test_method(df_input, **kwargs):
            # Simple finite difference
            values = df_input['value'].values
            derivatives = np.gradient(values)
            result_df = df_input.copy()
            result_df['derivative_value'] = derivatives
            return result_df
        
        # Benchmark the method
        result = benchmark_method(
            'test_method', test_method, df, true_deriv
        )
        
        assert isinstance(result, SimulationResult)
        assert result.method == 'test_method'
        assert result.mse_derivative >= 0
        assert result.computation_time >= 0
        
    def test_get_standard_test_functions(self):
        """Test standard test function collection."""
        test_functions = get_standard_test_functions()
        
        assert len(test_functions) > 0
        assert all(hasattr(func, '__call__') for func in test_functions)
        assert all(hasattr(func, 'derivative') for func in test_functions)
        assert all(hasattr(func, 'name') for func in test_functions)
        
        # Test that all functions work
        x = np.array([0, 1, 2, 3, 4])
        for func in test_functions:
            y = func(x)
            dy = func.derivative(x, order=1)
            assert len(y) == len(x)
            assert len(dy) == len(x)


class TestEdgeCases:
    """Test edge cases for testing framework."""
    
    @pytest.mark.skipif(not HAS_TESTING, reason="Testing framework not available")
    def test_polynomial_derivative_edge_cases(self):
        """Test PolynomialTrend derivative edge cases."""
        # Constant function
        const_poly = PolynomialTrend([5])  # Just constant
        x = np.array([0, 1, 2])
        
        y = const_poly(x)
        expected_y = np.full_like(x, 5.0)
        np.testing.assert_array_almost_equal(y, expected_y)
        
        # Derivative of constant should be zero
        dy = const_poly.derivative(x, order=1)
        expected_dy = np.zeros_like(x)
        np.testing.assert_array_almost_equal(dy, expected_dy)
        
        # Higher order derivatives should also be zero
        d2y = const_poly.derivative(x, order=2)
        np.testing.assert_array_almost_equal(d2y, np.zeros_like(x))
        
    @pytest.mark.skipif(not HAS_TESTING, reason="Testing framework not available")
    def test_sinusoidal_higher_derivatives(self):
        """Test higher order derivatives of sinusoidal function."""
        sin_trend = SinusoidalTrend(amplitude=1.0, frequency=1.0, phase=0.0)
        x = np.array([0, 0.25, 0.5, 0.75, 1.0])
        
        # Test third derivative
        d3y = sin_trend.derivative(x, order=3)
        # Third derivative of sin should be -cos (with frequency scaling)
        expected_d3y = -(2 * np.pi)**3 * np.cos(2 * np.pi * x)
        np.testing.assert_array_almost_equal(d3y, expected_d3y)
        
        # Test fourth derivative
        d4y = sin_trend.derivative(x, order=4)
        # Fourth derivative of sin should be sin (with frequency scaling)
        expected_d4y = (2 * np.pi)**4 * np.sin(2 * np.pi * x)
        np.testing.assert_array_almost_equal(d4y, expected_d4y)
        
    @pytest.mark.skipif(not HAS_TESTING, reason="Testing framework not available")
    def test_step_function_edge_cases(self):
        """Test StepTrend edge cases."""
        # Test with points exactly at breakpoints
        step_trend = StepTrend([2, 5], [1, 2, 3])
        x = np.array([0, 2, 3, 5, 7])
        
        y = step_trend(x)
        # Values at breakpoints should follow the step function logic
        assert len(y) == len(x)
        assert not np.any(np.isnan(y))
        
    @pytest.mark.skipif(not HAS_TESTING, reason="Testing framework not available")
    def test_very_short_time_series(self):
        """Test with very short time series."""
        trend_func = PolynomialTrend([1, 1])
        df, true_deriv = generate_time_series(
            trend_func, n_points=3, random_state=42
        )
        
        assert len(df) == 3
        assert len(true_deriv) == 3
        
    @pytest.mark.skipif(not HAS_TESTING, reason="Testing framework not available")
    def test_zero_noise(self):
        """Test with zero noise."""
        trend_func = PolynomialTrend([2, 0.5])
        df, true_deriv = generate_time_series(
            trend_func, n_points=20, noise_std=0.0, random_state=42
        )
        
        # With zero noise, observed should equal true values
        np.testing.assert_array_almost_equal(
            df['value'], df['true_value'], decimal=10
        )
        np.testing.assert_array_almost_equal(df['noise'], 0.0, decimal=10)


if __name__ == '__main__':
    pytest.main([__file__])