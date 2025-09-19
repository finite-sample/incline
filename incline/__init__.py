from .trend import naive_trend
from .trend import spline_trend
from .trend import sgolay_trend
from .trend import trending
from .trend import bootstrap_derivative_ci
from .trend import select_smoothing_parameter_cv
from .trend import compute_time_deltas

# Advanced methods (with graceful import handling)
try:
    from .advanced import (  # noqa: F401
        loess_trend, l1_trend_filter, local_polynomial_trend, estimate_trend
    )
    _has_advanced = True
except ImportError:
    _has_advanced = False

# Seasonal decomposition
try:
    from .seasonal import (  # noqa: F401
        stl_decompose, simple_deseasonalize, trend_with_deseasonalization
    )
    _has_seasonal = True
except ImportError:
    _has_seasonal = False

# State-space models
try:
    from .statespace import kalman_trend, adaptive_kalman_trend  # noqa: F401
    _has_statespace = True
except ImportError:
    _has_statespace = False

# Testing framework
try:
    from .testing import (  # noqa: F401
        generate_time_series, run_comprehensive_benchmark
    )
    _has_testing = True
except ImportError:
    _has_testing = False

# Gaussian Process methods
try:
    from .gaussian_process import gp_trend, adaptive_gp_trend  # noqa: F401
    _has_gp = True
except ImportError:
    _has_gp = False

# Multiscale analysis methods
try:
    from .multiscale import sizer_analysis, trend_with_sizer  # noqa: F401
    _has_multiscale = True
except ImportError:
    _has_multiscale = False

__version__ = "0.2.0"
# Build __all__ dynamically based on what's available
__all__ = [
    # Core functions (always available)
    'naive_trend',
    'spline_trend',
    'sgolay_trend',
    'trending',
    'bootstrap_derivative_ci',
    'select_smoothing_parameter_cv',
    'compute_time_deltas',
]

# Add advanced methods if available
if _has_advanced:
    __all__.extend([
        'loess_trend',
        'l1_trend_filter',
        'local_polynomial_trend',
        'estimate_trend',
    ])

# Add seasonal methods if available
if _has_seasonal:
    __all__.extend([
        'stl_decompose',
        'simple_deseasonalize',
        'trend_with_deseasonalization',
    ])

# Add state-space methods if available
if _has_statespace:
    __all__.extend([
        'kalman_trend',
        'adaptive_kalman_trend',
    ])

# Add testing framework if available
if _has_testing:
    __all__.extend([
        'generate_time_series',
        'run_comprehensive_benchmark'
    ])

# Add Gaussian Process methods if available
if _has_gp:
    __all__.extend([
        'gp_trend',
        'adaptive_gp_trend'
    ])

# Add multiscale methods if available
if _has_multiscale:
    __all__.extend([
        'sizer_analysis',
        'trend_with_sizer'
    ])
