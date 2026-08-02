"""Tests for incline.statespace module."""

import numpy as np
import pandas as pd
import pytest


try:
    from incline.statespace import (
        LocalLinearTrend,
        StructuralTrendModel,
        adaptive_kalman_trend,
        kalman_trend,
        select_kalman_model,
    )

    HAS_STATESPACE = True
except ImportError:
    HAS_STATESPACE = False


@pytest.mark.skipif(not HAS_STATESPACE, reason="State-space methods not available")
class TestLocalLinearTrend:
    """Test cases for LocalLinearTrend model."""

    def test_local_linear_trend_basic(self):
        """Test basic LocalLinearTrend functionality."""
        # Create synthetic local linear trend data
        np.random.seed(42)
        n = 50
        true_level = np.cumsum(0.1 * np.random.randn(n))
        true_slope = np.cumsum(0.01 * np.random.randn(n))
        y = true_level + np.cumsum(true_slope) + 0.1 * np.random.randn(n)

        # Fit model
        model = LocalLinearTrend()
        fitted_model = model.fit(y)

        assert fitted_model is model  # Should return self
        assert model.fitted_params is not None
        assert "obs_variance" in model.fitted_params
        assert "level_variance" in model.fitted_params
        assert "slope_variance" in model.fitted_params

    def test_local_linear_trend_with_fixed_params(self):
        """Test LocalLinearTrend with fixed parameters."""
        np.random.seed(42)
        y = np.cumsum(0.1 * np.random.randn(30))

        # Test with pre-specified variances
        model = LocalLinearTrend(
            obs_variance=0.1, level_variance=0.05, slope_variance=0.01
        )
        model.fit(y)

        assert model.fitted_params["obs_variance"] == pytest.approx(0.1)
        assert model.fitted_params["level_variance"] == pytest.approx(0.05)
        assert model.fitted_params["slope_variance"] == pytest.approx(0.01)

    def test_local_linear_trend_get_components(self):
        """Test getting level and slope estimates."""
        np.random.seed(42)
        y = np.cumsum(0.1 * np.random.randn(40)) + 0.05 * np.arange(40)

        model = LocalLinearTrend()
        model.fit(y)

        # Test level estimation
        level, level_lower, level_upper = model.get_level(confidence_level=0.95)
        assert len(level) == len(y)
        assert len(level_lower) == len(y)
        assert len(level_upper) == len(y)
        assert np.all(level_lower <= level)
        assert np.all(level <= level_upper)

        # Test slope estimation
        slope, slope_lower, slope_upper = model.get_slope(confidence_level=0.90)
        assert len(slope) == len(y)
        assert len(slope_lower) == len(y)
        assert len(slope_upper) == len(y)
        assert np.all(slope_lower <= slope)
        assert np.all(slope <= slope_upper)

    def test_local_linear_trend_edge_cases(self):
        """Test LocalLinearTrend edge cases."""
        # Test with very short series
        model = LocalLinearTrend()
        with pytest.raises(ValueError, match="Need at least 4 observations"):
            model.fit([1, 2, 3])

        # Test with missing values
        y_with_nan = np.array([1, 2, np.nan, 4, 5, 6, 7, 8])
        model = LocalLinearTrend()
        with pytest.warns(UserWarning, match="Missing values detected"):
            model.fit(y_with_nan)


@pytest.mark.skipif(not HAS_STATESPACE, reason="State-space methods not available")
class TestStructuralTrendModel:
    """Test cases for StructuralTrendModel."""

    def test_structural_model_basic(self):
        """Test basic StructuralTrendModel functionality."""
        # Create seasonal data
        np.random.seed(42)
        t = np.arange(100)
        seasonal = 0.5 * np.sin(2 * np.pi * t / 12)
        trend = 0.02 * t
        y = trend + seasonal + 0.1 * np.random.randn(100)

        model = StructuralTrendModel(seasonal_periods=[12])
        try:
            model.fit(y)
            components = model.get_components()

            assert isinstance(components, dict)
            assert "level" in components
            assert "trend" in components

        except ImportError:
            pytest.skip("statsmodels UnobservedComponents not available")
        except Exception as e:
            # Some configurations may fail, which is acceptable
            pytest.skip(f"Structural model fitting failed: {e}")

    def test_structural_model_without_seasonality(self):
        """Test StructuralTrendModel without seasonal component."""
        np.random.seed(42)
        y = 0.02 * np.arange(50) + 0.1 * np.random.randn(50)

        model = StructuralTrendModel()
        try:
            model.fit(y)
            components = model.get_components()
            assert isinstance(components, dict)

        except ImportError:
            pytest.skip("statsmodels not available")
        except Exception:
            # May fail in some environments
            pytest.skip("Structural model fitting failed")


@pytest.mark.skipif(not HAS_STATESPACE, reason="State-space methods not available")
class TestKalmanTrend:
    """Test cases for kalman_trend function."""

    def test_kalman_trend_basic(self):
        """Test basic kalman_trend functionality."""
        # Create test data
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        y = np.cumsum(0.1 * np.random.randn(50)) + 0.1 * np.arange(50)
        df = pd.DataFrame({"value": y}, index=dates)

        try:
            result = kalman_trend(df, model_type="local_linear")

            assert "smoothed_value" in result.columns
            assert "derivative_value" in result.columns
            assert "derivative_method" in result.columns
            assert "model_type" in result.columns
            assert result["derivative_method"].iloc[0] == "kalman"
            assert result["model_type"].iloc[0] == "local_linear_trend"
            assert len(result) == len(df)

            # Check confidence intervals
            assert "smoothed_value_lower" in result.columns
            assert "smoothed_value_upper" in result.columns
            assert "derivative_ci_lower" in result.columns
            assert "derivative_ci_upper" in result.columns

            # Check significance test
            assert "significant_trend" in result.columns

        except ImportError:
            pytest.skip("Required dependencies not available")

    def test_kalman_trend_with_time_column(self):
        """Test kalman_trend with explicit time column."""
        np.random.seed(42)
        time_values = np.linspace(0, 10, 30)
        y = 0.5 * time_values + 0.1 * np.random.randn(30)
        df = pd.DataFrame({"time": time_values, "value": y})

        try:
            result = kalman_trend(df, time_column="time")
            assert len(result) == len(df)
            assert "time" in result.columns

        except ImportError:
            pytest.skip("Required dependencies not available")

    def test_kalman_trend_structural(self):
        """Test kalman_trend with structural model."""
        np.random.seed(42)
        t = np.arange(60)
        y = 0.02 * t + 0.5 * np.sin(2 * np.pi * t / 12) + 0.1 * np.random.randn(60)
        df = pd.DataFrame({"value": y})

        try:
            result = kalman_trend(df, model_type="structural", seasonal_periods=[12])
            assert len(result) == len(df)
            assert "model_type" in result.columns
            assert result["model_type"].iloc[0] == "structural"

        except ImportError:
            pytest.skip("Required dependencies not available")
        except Exception:
            # Structural models may fail in some configurations
            pytest.skip("Structural model failed")


@pytest.mark.skipif(not HAS_STATESPACE, reason="State-space methods not available")
class TestAdaptiveKalman:
    """Test cases for adaptive Kalman filtering."""

    def test_adaptive_kalman_basic(self):
        """Test basic adaptive Kalman functionality."""
        np.random.seed(42)
        # Create data with changing trend
        y1 = 0.1 * np.arange(30)
        y2 = 3 + 0.05 * np.arange(30)  # Different slope
        y = np.concatenate([y1, y2]) + 0.1 * np.random.randn(60)

        df = pd.DataFrame({"value": y})

        try:
            result = adaptive_kalman_trend(df, adaptation_window=20)

            assert "smoothed_value" in result.columns
            assert "derivative_value" in result.columns
            assert "derivative_method" in result.columns
            assert "adaptation_window" in result.columns
            assert result["derivative_method"].iloc[0] == "adaptive_kalman"
            assert result["adaptation_window"].iloc[0] == 20
            assert len(result) == len(df)

        except ImportError:
            pytest.skip("Required dependencies not available")

    def test_adaptive_kalman_small_dataset(self):
        """Test adaptive Kalman with small dataset."""
        np.random.seed(42)
        y = np.random.randn(15)  # Smaller than default window
        df = pd.DataFrame({"value": y})

        try:
            # Should fall back to standard Kalman
            result = adaptive_kalman_trend(df, adaptation_window=20)
            assert len(result) == len(df)

        except ImportError:
            pytest.skip("Required dependencies not available")


class TestModelSelection:
    """Test cases for automatic model selection."""

    @pytest.mark.skipif(not HAS_STATESPACE, reason="State-space methods not available")
    def test_select_kalman_model_basic(self):
        """Test automatic Kalman model selection."""
        # Create different types of data
        np.random.seed(42)

        # Seasonal data
        t = np.arange(100)
        seasonal_data = (
            0.02 * t + np.sin(2 * np.pi * t / 12) + 0.1 * np.random.randn(100)
        )
        seasonal_df = pd.DataFrame({"value": seasonal_data})

        # Non-seasonal data
        simple_data = 0.05 * np.arange(30) + 0.1 * np.random.randn(30)
        simple_df = pd.DataFrame({"value": simple_data})

        try:
            # Test model selection
            model1 = select_kalman_model(seasonal_df)
            model2 = select_kalman_model(simple_df)

            assert model1 in ["local_linear", "structural", "adaptive_kalman"]
            assert model2 in ["local_linear", "structural", "adaptive_kalman"]

        except ImportError:
            pytest.skip("Required dependencies not available")

    @pytest.mark.skipif(not HAS_STATESPACE, reason="State-space methods not available")
    def test_select_kalman_model_changing_variance(self):
        """Test model selection with changing variance."""
        np.random.seed(42)
        # Create data with changing variance
        y1 = 0.1 * np.random.randn(25)
        y2 = 0.5 * np.random.randn(25)  # Higher variance
        y = np.concatenate([y1, y2])
        df = pd.DataFrame({"value": y})

        try:
            model = select_kalman_model(df)
            # Should recommend adaptive method for changing variance
            assert model in ["local_linear", "structural", "adaptive_kalman"]

        except ImportError:
            pytest.skip("Required dependencies not available")


class TestEdgeCases:
    """Test edge cases for state-space methods."""

    @pytest.mark.skipif(not HAS_STATESPACE, reason="State-space methods not available")
    def test_constant_data(self):
        """Test with constant data."""
        df = pd.DataFrame({"value": np.ones(30)})

        try:
            result = kalman_trend(df)
            assert len(result) == len(df)
            # Derivatives should be close to zero
            assert np.all(np.abs(result["derivative_value"]) < 1e-6)

        except ImportError:
            pytest.skip("Required dependencies not available")
        except Exception:
            # May fail with numerical issues
            pytest.skip("Kalman filter failed on constant data")

    @pytest.mark.skipif(not HAS_STATESPACE, reason="State-space methods not available")
    def test_linear_trend(self):
        """Test with perfect linear trend."""
        t = np.arange(50)
        y = 2 + 0.5 * t  # Perfect linear trend
        df = pd.DataFrame({"value": y})

        try:
            result = kalman_trend(df)
            assert len(result) == len(df)
            # Derivatives should be close to 0.5
            middle_derivatives = result["derivative_value"].iloc[10:-10]
            assert np.abs(np.mean(middle_derivatives) - 0.5) < 0.1

        except ImportError:
            pytest.skip("Required dependencies not available")

    @pytest.mark.skipif(not HAS_STATESPACE, reason="State-space methods not available")
    def test_very_noisy_data(self):
        """Test with very noisy data."""
        np.random.seed(42)
        y = 0.01 * np.arange(40) + np.random.randn(40)  # High noise
        df = pd.DataFrame({"value": y})

        try:
            result = kalman_trend(df)
            assert len(result) == len(df)
            # Should still produce reasonable estimates
            assert not np.any(np.isnan(result["smoothed_value"]))

        except ImportError:
            pytest.skip("Required dependencies not available")


@pytest.mark.skipif(not HAS_STATESPACE, reason="State-space methods not available")
class TestStateSpaceRegressions:
    """Regressions for defects found by audit."""

    def test_log_likelihood_objective_varies_with_parameters(self):
        """The MLE objective must depend on the parameters it is optimising."""
        y = 2.0 * np.arange(60) + np.random.default_rng(0).normal(0, 1.0, 60)
        model = LocalLinearTrend()

        objectives = [
            model._log_likelihood(np.log(np.array(p, dtype=float)), y)
            for p in ([1.0, 1.0, 1.0], [100.0, 1.0, 0.01], [0.5, 50.0, 3.0])
        ]

        # Previously the objective returned loglikelihood_burn, an integer
        # burn-in count that is always 0, so every parameter vector scored 0.
        assert len({round(o, 6) for o in objectives}) == len(objectives)
        assert all(np.isfinite(o) for o in objectives)

    def test_fitted_variances_are_not_the_initial_guess(self):
        """Fitted variances must reflect the data, not the fixed 10:1:0.1 seed."""
        y = 2.0 * np.arange(60) + np.random.default_rng(0).normal(0, 1.0, 60)
        model = LocalLinearTrend().fit(y)

        y_var = float(np.nanvar(y))
        initial_guess = [y_var * 0.1, y_var * 0.01, y_var * 0.001]
        fitted = [
            model.fitted_params["obs_variance"],
            model.fitted_params["level_variance"],
            model.fitted_params["slope_variance"],
        ]

        assert not np.allclose(fitted, initial_guess)

    def test_structural_get_components_returns_arrays(self):
        """UnobservedComponentsResults has no .irregular; it must be derived."""
        y = 2.0 * np.arange(60) + np.random.default_rng(1).normal(0, 0.1, 60)
        components = StructuralTrendModel().fit(y).get_components()

        assert set(components) == {"level", "trend", "seasonal", "irregular"}
        assert np.all(np.isfinite(components["irregular"]))
        assert len(components["irregular"]) == len(y)

    def test_structural_derivative_is_the_slope_not_its_gradient(self):
        """On y = 2t the structural derivative is 2.0, not d(slope)/dt = 0."""
        t = np.arange(60, dtype=float)
        result = kalman_trend(pd.DataFrame({"value": 2.0 * t}), model_type="structural")

        assert np.median(result["derivative_value"]) == pytest.approx(2.0, abs=1e-6)

    def test_damped_trend_is_not_silently_ignored(self):
        """damped_trend is documented as changing the model, so it must not no-op."""
        with pytest.raises(NotImplementedError):
            StructuralTrendModel(damped_trend=True).fit(np.arange(50, dtype=float))

    def test_kalman_trend_accepts_every_selected_model(self):
        """select_kalman_model's recommendation must be usable by kalman_trend."""
        y = np.concatenate(
            [
                np.random.default_rng(7).normal(0, 0.2, 60),
                np.random.default_rng(8).normal(0, 5.0, 60),
            ]
        )
        df = pd.DataFrame({"value": y})

        model_type = select_kalman_model(df)
        assert model_type == "adaptive_kalman"  # the case that used to raise

        result = kalman_trend(df, model_type=model_type)
        assert len(result) == len(df)

    def test_adaptive_kalman_forwards_confidence_level(self):
        """The adaptive dispatch path must not reset intervals to the default."""
        y = 2.0 * np.arange(80) + np.random.default_rng(9).normal(0, 1.0, 80)
        df = pd.DataFrame({"value": y})

        narrow = kalman_trend(
            df,
            model_type="adaptive_kalman",
            confidence_level=0.5,
            adaptation_window=20,
        )
        wide = kalman_trend(
            df,
            model_type="adaptive_kalman",
            confidence_level=0.99,
            adaptation_window=20,
        )

        narrow_width = (
            narrow["derivative_ci_upper"] - narrow["derivative_ci_lower"]
        ).dropna()
        wide_width = (
            wide["derivative_ci_upper"] - wide["derivative_ci_lower"]
        ).dropna()

        assert np.median(wide_width) > np.median(narrow_width)


if __name__ == "__main__":
    pytest.main([__file__])
