from importlib.metadata import version

from .advanced import (
    estimate_trend,
    l1_trend_filter,
    local_polynomial_trend,
    loess_trend,
)
from .gaussian_process import adaptive_gp_trend, gp_trend
from .multiscale import sizer_analysis, trend_with_sizer
from .seasonal import (
    simple_deseasonalize,
    stl_decompose,
    trend_with_deseasonalization,
)
from .statespace import adaptive_kalman_trend, kalman_trend
from .testing import generate_time_series, run_comprehensive_benchmark
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
    "estimate_trend",
    "generate_time_series",
    "gp_trend",
    "kalman_trend",
    "l1_trend_filter",
    "local_polynomial_trend",
    "loess_trend",
    "naive_trend",
    "run_comprehensive_benchmark",
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
