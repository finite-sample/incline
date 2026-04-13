from importlib.metadata import version

from .advanced import (
    estimate_trend,
    l1_trend_filter,
    local_polynomial_trend,
    loess_trend,
)
from .gaussian_process import adaptive_gp_trend, gp_trend, select_gp_kernel
from .multiscale import quick_sizer_plot, sizer_analysis, trend_with_sizer
from .seasonal import (
    deseasonalize_pipeline,
    simple_deseasonalize,
    stl_decompose,
    trend_with_deseasonalization,
)
from .statespace import adaptive_kalman_trend, kalman_trend, select_kalman_model
from .testing import (
    generate_time_series,
    get_standard_test_functions,
    run_comprehensive_benchmark,
)
from .trend import (
    bootstrap_derivative_ci,
    compute_time_deltas,
    naive_trend,
    select_smoothing_parameter_cv,
    sgolay_trend,
    spline_trend,
    trending,
)


try:
    __version__ = version("incline")
except Exception:
    __version__ = "0.4.0-dev"

__all__ = [
    "adaptive_gp_trend",
    "adaptive_kalman_trend",
    "bootstrap_derivative_ci",
    "compute_time_deltas",
    "deseasonalize_pipeline",
    "estimate_trend",
    "generate_time_series",
    "get_standard_test_functions",
    "gp_trend",
    "kalman_trend",
    "l1_trend_filter",
    "local_polynomial_trend",
    "loess_trend",
    "naive_trend",
    "quick_sizer_plot",
    "run_comprehensive_benchmark",
    "select_gp_kernel",
    "select_kalman_model",
    "select_smoothing_parameter_cv",
    "sgolay_trend",
    "simple_deseasonalize",
    "sizer_analysis",
    "spline_trend",
    "stl_decompose",
    "trend_with_deseasonalization",
    "trend_with_sizer",
    "trending",
]
