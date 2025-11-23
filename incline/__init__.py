from .trend import (
    bootstrap_derivative_ci,
    compute_time_deltas,
    naive_trend,
    select_smoothing_parameter_cv,
    sgolay_trend,
    spline_trend,
    trending,
)


# Advanced methods (with graceful import handling)
try:
    from .advanced import (
        estimate_trend,
        l1_trend_filter,
        local_polynomial_trend,
        loess_trend,
    )

    _has_advanced = True
except ImportError:
    _has_advanced = False

# Seasonal decomposition
try:
    from .seasonal import (
        simple_deseasonalize,
        stl_decompose,
        trend_with_deseasonalization,
    )

    _has_seasonal = True
except ImportError:
    _has_seasonal = False

# State-space models
try:
    from .statespace import adaptive_kalman_trend, kalman_trend

    _has_statespace = True
except ImportError:
    _has_statespace = False

# Testing framework
try:
    from .testing import generate_time_series, run_comprehensive_benchmark

    _has_testing = True
except ImportError:
    _has_testing = False

# Gaussian Process methods
try:
    from .gaussian_process import adaptive_gp_trend, gp_trend

    _has_gp = True
except ImportError:
    _has_gp = False

# Multiscale analysis methods
try:
    from .multiscale import sizer_analysis, trend_with_sizer

    _has_multiscale = True
except ImportError:
    _has_multiscale = False

__version__ = "0.2.0"
# Build __all__ dynamically based on what's available
__all__ = [
    "bootstrap_derivative_ci",
    "compute_time_deltas",
    # Core functions (always available)
    "naive_trend",
    "select_smoothing_parameter_cv",
    "sgolay_trend",
    "spline_trend",
    "trending",
]

# Add advanced methods if available
if _has_advanced:
    __all__.extend(
        [
            "estimate_trend",
            "l1_trend_filter",
            "local_polynomial_trend",
            "loess_trend",
        ]
    )

# Add seasonal methods if available
if _has_seasonal:
    __all__.extend(
        [
            "simple_deseasonalize",
            "stl_decompose",
            "trend_with_deseasonalization",
        ]
    )

# Add state-space methods if available
if _has_statespace:
    __all__.extend(
        [
            "adaptive_kalman_trend",
            "kalman_trend",
        ]
    )

# Add testing framework if available
if _has_testing:
    __all__.extend(["generate_time_series", "run_comprehensive_benchmark"])

# Add Gaussian Process methods if available
if _has_gp:
    __all__.extend(["adaptive_gp_trend", "gp_trend"])

# Add multiscale methods if available
if _has_multiscale:
    __all__.extend(["sizer_analysis", "trend_with_sizer"])
