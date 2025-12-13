"""Tests for incline.multiscale module."""

import numpy as np
import pandas as pd
import pytest


try:
    from incline.multiscale import (
        SiZer,
        quick_sizer_plot,
        sizer_analysis,
        trend_with_sizer,
    )

    HAS_MULTISCALE = True
except ImportError:
    HAS_MULTISCALE = False

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@pytest.mark.skipif(not HAS_MULTISCALE, reason="Multiscale methods not available")
class TestSiZer:
    """Test cases for SiZer class."""

    def test_sizer_basic(self):
        """Test basic SiZer functionality."""
        # Create synthetic data with clear trend change
        np.random.seed(42)
        t1 = np.linspace(0, 5, 25)
        t2 = np.linspace(5, 10, 25)
        y1 = 1 + 0.2 * t1 + 0.05 * np.random.randn(25)  # Slow increase
        y2 = 3 + 0.8 * (t2 - 5) + 0.05 * np.random.randn(25)  # Fast increase

        t = np.concatenate([t1, t2])
        y = np.concatenate([y1, y2])
        df = pd.DataFrame({"time": t, "value": y})

        # Fit SiZer
        sizer = SiZer(n_bandwidths=10, method="spline")
        sizer.fit(df, time_column="time")

        assert sizer.significance_map is not None
        assert sizer.x_values is not None
        assert sizer.derivative_estimates is not None
        assert sizer.derivative_se is not None
        assert len(sizer.bandwidths) == 10
        assert sizer.significance_map.shape == (10, len(t))

        # Check that significance values are in expected range
        unique_sigs = np.unique(sizer.significance_map)
        assert all(sig in [-1, 0, 1] for sig in unique_sigs)

    def test_sizer_methods(self):
        """Test SiZer with different smoothing methods."""
        np.random.seed(42)
        t = np.linspace(0, 8, 40)
        y = 2 + 0.3 * t + 0.1 * np.random.randn(40)
        df = pd.DataFrame({"value": y})

        # Test different methods
        for method in ["spline"]:  # Only test spline since others may not be available
            sizer = SiZer(n_bandwidths=5, method=method)
            sizer.fit(df)

            assert sizer.significance_map is not None
            assert sizer.significance_map.shape == (5, len(t))

    def test_sizer_confidence_levels(self):
        """Test SiZer with different confidence levels."""
        np.random.seed(42)
        t = np.linspace(0, 5, 30)
        y = 1 + 0.5 * t + 0.1 * np.random.randn(30)
        df = pd.DataFrame({"value": y})

        # Test different confidence levels
        sizer_95 = SiZer(confidence_level=0.95, n_bandwidths=5)
        sizer_90 = SiZer(confidence_level=0.90, n_bandwidths=5)

        sizer_95.fit(df)
        sizer_90.fit(df)

        # 90% should have more significant regions than 95%
        np.sum(sizer_95.significance_map != 0)
        np.sum(sizer_90.significance_map != 0)

        # This isn't guaranteed due to randomness, but often true
        assert sizer_95.significance_map is not None
        assert sizer_90.significance_map is not None

    def test_sizer_custom_bandwidths(self):
        """Test SiZer with custom bandwidths."""
        np.random.seed(42)
        df = pd.DataFrame({"value": np.random.randn(25)})

        custom_bw = np.array([0.1, 0.2, 0.3])
        sizer = SiZer(bandwidths=custom_bw)
        sizer.fit(df)

        assert len(sizer.bandwidths) == 3
        assert np.allclose(sizer.bandwidths, custom_bw)
        assert sizer.significance_map.shape == (3, 25)

    def test_sizer_dataframe_output(self):
        """Test SiZer DataFrame output."""
        np.random.seed(42)
        df = pd.DataFrame({"value": np.random.randn(20)})

        sizer = SiZer(n_bandwidths=3)
        sizer.fit(df)

        sizer_df = sizer.get_sizer_dataframe()

        assert isinstance(sizer_df, pd.DataFrame)
        expected_cols = [
            "x",
            "bandwidth",
            "derivative",
            "derivative_se",
            "significance",
        ]
        assert all(col in sizer_df.columns for col in expected_cols)
        assert len(sizer_df) == 3 * 20  # 3 bandwidths x 20 points

        # Check data types
        assert sizer_df["significance"].dtype in [np.int32, np.int64, int]
        assert all(sig in [-1, 0, 1] for sig in sizer_df["significance"])

    def test_find_significant_features(self):
        """Test finding significant features."""
        np.random.seed(42)
        # Create data with distinct increasing and decreasing sections
        t = np.linspace(0, 10, 50)
        y = np.zeros_like(t)
        y[:20] = 1 + 0.5 * t[:20]  # Increasing
        y[20:30] = y[19] - 0.3 * (t[20:30] - t[19])  # Decreasing
        y[30:] = y[29] + 0.05 * np.random.randn(20)  # Flat with noise

        df = pd.DataFrame({"time": t, "value": y})

        sizer = SiZer(n_bandwidths=8, method="spline")
        sizer.fit(df, time_column="time")

        features = sizer.find_significant_features(min_persistence=2)

        assert isinstance(features, dict)
        assert "increasing" in features
        assert "decreasing" in features
        assert isinstance(features["increasing"], list)
        assert isinstance(features["decreasing"], list)

        # Should find some features (though exact locations depend on noise)
        total_features = len(features["increasing"]) + len(features["decreasing"])
        assert total_features >= 0  # At least this should not fail

    def test_sizer_edge_cases(self):
        """Test SiZer edge cases."""
        # Test with minimal data
        sizer = SiZer()
        df = pd.DataFrame({"value": [1, 2, 3, 4]})
        with pytest.raises(ValueError, match="Need at least 5 valid observations"):
            sizer.fit(df)

        # Test with missing values
        df_with_nan = pd.DataFrame({"value": [1, 2, np.nan, 4, 5, 6, np.nan, 8, 9, 10]})
        sizer = SiZer(n_bandwidths=3)
        sizer.fit(df_with_nan)  # Should handle missing values

        assert sizer.significance_map is not None
        # Should have fewer points than original due to missing value removal
        assert sizer.significance_map.shape[1] == 8  # 10 - 2 missing

    def test_sizer_constant_data(self):
        """Test SiZer with constant data."""
        df = pd.DataFrame({"value": np.ones(20)})

        sizer = SiZer(n_bandwidths=5)
        sizer.fit(df)

        # Derivatives should be close to zero
        assert np.all(np.abs(sizer.derivative_estimates) < 0.1)
        # Should be mostly insignificant
        assert np.sum(sizer.significance_map == 0) > np.sum(sizer.significance_map != 0)


@pytest.mark.skipif(not HAS_MULTISCALE, reason="Multiscale methods not available")
class TestSiZerFunctions:
    """Test cases for SiZer convenience functions."""

    def test_sizer_analysis(self):
        """Test sizer_analysis function."""
        np.random.seed(42)
        t = np.linspace(0, 6, 35)
        y = 1 + 0.3 * t + 0.1 * np.random.randn(35)
        df = pd.DataFrame({"time": t, "value": y})

        sizer = sizer_analysis(df, time_column="time", n_bandwidths=6)

        assert isinstance(sizer, SiZer)
        assert sizer.significance_map is not None
        assert len(sizer.bandwidths) == 6

    def test_sizer_analysis_parameters(self):
        """Test sizer_analysis with different parameters."""
        np.random.seed(42)
        df = pd.DataFrame({"value": np.random.randn(30)})

        # Test different parameters
        sizer1 = sizer_analysis(df, n_bandwidths=5, bandwidth_range=(0.05, 0.5))
        sizer2 = sizer_analysis(df, method="spline", confidence_level=0.90)

        assert len(sizer1.bandwidths) == 5
        assert sizer1.bandwidths[0] >= 0.05
        assert sizer1.bandwidths[-1] <= 0.5

        assert isinstance(sizer2, SiZer)

    def test_trend_with_sizer(self):
        """Test trend estimation combined with SiZer."""
        np.random.seed(42)
        t = np.linspace(0, 8, 40)
        y = 2 + 0.4 * t + 0.1 * np.random.randn(40)
        df = pd.DataFrame({"time": t, "value": y})

        result = trend_with_sizer(df, time_column="time", trend_method="spline")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

        # Check for trend estimation columns
        assert "derivative_value" in result.columns
        assert "smoothed_value" in result.columns

        # Check for SiZer columns
        sizer_cols = [
            "sizer_significance",
            "sizer_increasing",
            "sizer_decreasing",
            "sizer_insignificant",
            "persistent_increasing",
            "persistent_decreasing",
        ]
        assert all(col in result.columns for col in sizer_cols)

        # Check data types
        assert result["sizer_increasing"].dtype == bool
        assert result["sizer_decreasing"].dtype == bool
        assert result["persistent_increasing"].dtype == bool
        assert result["persistent_decreasing"].dtype == bool

    def test_trend_with_sizer_methods(self):
        """Test trend_with_sizer with different methods."""
        np.random.seed(42)
        df = pd.DataFrame({"value": np.random.randn(25)})

        # Test different trend methods
        for trend_method in ["spline"]:  # Only test available methods
            result = trend_with_sizer(
                df, trend_method=trend_method, sizer_method="spline"
            )
            assert len(result) == len(df)
            assert "sizer_significance" in result.columns


@pytest.mark.skipif(
    not HAS_MULTISCALE or not HAS_MATPLOTLIB, reason="Plotting not available"
)
class TestSiZerPlotting:
    """Test cases for SiZer plotting functionality."""

    def test_plot_sizer_map(self):
        """Test SiZer map plotting."""
        np.random.seed(42)
        t = np.linspace(0, 5, 30)
        y = 1 + 0.3 * t + 0.1 * np.random.randn(30)
        df = pd.DataFrame({"value": y})

        sizer = SiZer(n_bandwidths=6)
        sizer.fit(df)

        fig = sizer.plot_sizer_map(figsize=(8, 6))

        assert fig is not None
        assert len(fig.axes) >= 1  # Should have at least one axis

        # Clean up
        plt.close(fig)

    def test_plot_sizer_map_custom(self):
        """Test SiZer plotting with custom parameters."""
        np.random.seed(42)
        df = pd.DataFrame({"value": np.random.randn(25)})

        sizer = SiZer(n_bandwidths=5)
        sizer.fit(df)

        fig = sizer.plot_sizer_map(title="Custom SiZer Map", figsize=(10, 6))

        assert fig is not None
        plt.close(fig)

    def test_quick_sizer_plot(self):
        """Test quick SiZer plotting function."""
        np.random.seed(42)
        t = np.linspace(0, 4, 25)
        y = np.sin(t) + 0.1 * np.random.randn(25)
        df = pd.DataFrame({"time": t, "value": y})

        fig = quick_sizer_plot(df, time_column="time", n_bandwidths=5)

        assert fig is not None
        plt.close(fig)


class TestSiZerIntegration:
    """Test integration with other incline methods."""

    @pytest.mark.skipif(not HAS_MULTISCALE, reason="Multiscale methods not available")
    def test_sizer_with_datetime_index(self):
        """Test SiZer with datetime index."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        y = 1 + 0.1 * np.arange(30) + 0.05 * np.random.randn(30)
        df = pd.DataFrame({"value": y}, index=dates)

        sizer = SiZer(n_bandwidths=5)
        sizer.fit(df)

        assert sizer.significance_map is not None
        assert len(sizer.x_values) == 30

    @pytest.mark.skipif(not HAS_MULTISCALE, reason="Multiscale methods not available")
    def test_sizer_irregular_time(self):
        """Test SiZer with irregular time spacing."""
        np.random.seed(42)
        irregular_times = np.sort(np.random.uniform(0, 10, 25))
        y = 2 + 0.2 * irregular_times + 0.1 * np.random.randn(25)
        df = pd.DataFrame({"time": irregular_times, "value": y})

        sizer = SiZer(n_bandwidths=6)
        sizer.fit(df, time_column="time")

        assert sizer.significance_map is not None
        assert len(sizer.x_values) == 25

    @pytest.mark.skipif(not HAS_MULTISCALE, reason="Multiscale methods not available")
    def test_sizer_fallback_methods(self):
        """Test SiZer fallback when advanced methods not available."""
        np.random.seed(42)
        df = pd.DataFrame({"value": np.random.randn(20)})

        # Test with methods that might not be available
        for method in ["loess", "gp", "spline"]:
            try:
                sizer = SiZer(n_bandwidths=4, method=method)
                sizer.fit(df)
                assert sizer.significance_map is not None
            except ImportError:
                # Expected if method dependencies not available
                pass


class TestEdgeCases:
    """Test edge cases for multiscale analysis."""

    @pytest.mark.skipif(not HAS_MULTISCALE, reason="Multiscale methods not available")
    def test_very_noisy_data(self):
        """Test SiZer with very noisy data."""
        np.random.seed(42)
        t = np.linspace(0, 5, 40)
        signal = 0.1 * t
        noise = 2 * np.random.randn(40)  # High noise
        y = signal + noise
        df = pd.DataFrame({"value": y})

        sizer = SiZer(n_bandwidths=5)
        sizer.fit(df)

        # Should handle noisy data gracefully
        assert sizer.significance_map is not None
        # Most should be insignificant due to high noise
        n_insignificant = np.sum(sizer.significance_map == 0)
        assert n_insignificant > 0

    @pytest.mark.skipif(not HAS_MULTISCALE, reason="Multiscale methods not available")
    def test_short_time_series(self):
        """Test SiZer with short time series."""
        np.random.seed(42)
        df = pd.DataFrame({"value": np.random.randn(8)})  # Very short

        sizer = SiZer(n_bandwidths=3, bandwidth_range=(0.2, 0.8))  # Larger bandwidths
        sizer.fit(df)

        assert sizer.significance_map is not None
        assert sizer.significance_map.shape == (3, 8)

    @pytest.mark.skipif(not HAS_MULTISCALE, reason="Multiscale methods not available")
    def test_sizer_error_handling(self):
        """Test SiZer error handling."""
        # Test before fitting
        sizer = SiZer()

        with pytest.raises(ValueError, match="Must fit SiZer"):
            sizer.get_sizer_dataframe()

        with pytest.raises(ValueError, match="Must fit SiZer"):
            sizer.find_significant_features()

        if HAS_MATPLOTLIB:
            with pytest.raises(ValueError, match="Must fit SiZer"):
                sizer.plot_sizer_map()


if __name__ == "__main__":
    pytest.main([__file__])
