"""Estimate the trend at a point in a noisy time series, and say how sure you are.

Naive differencing amplifies noise, so incline smooths first and differentiates
the smooth. The interesting part is the second half: every estimator can report
uncertainty, and which machinery produces it is decided by what the smoother
*is* rather than by what it is called.

    >>> import numpy as np, pandas as pd
    >>> from incline import sgolay_trend
    >>> df = pd.DataFrame({"value": np.arange(50.0)},
    ...                   index=pd.date_range("2020-01-01", periods=50))
    >>> out = sgolay_trend(df, with_uncertainty=True)
    >>> round(out["derivative_value"].iloc[0], 6), out["uncertainty_method"].iloc[0]
    (1.0, 'operator')

Every estimator returns the same columns -- ``derivative_value``,
``derivative_standard_error``, ``derivative_ci_lower``, ``derivative_ci_upper``,
``uncertainty_method`` and ``significant_trend``. A NaN standard error paired with
``uncertainty_method=None`` is a deliberate, documented state, never a missing column.

Standard errors are opt-in via ``with_uncertainty=True`` because the exact
route costs one smoother evaluation per observation. Noise is assumed
independent unless you say otherwise with ``noise='ar1'``; under real
autocorrelation the independent assumption understates the uncertainty
substantially.
"""

from importlib.metadata import PackageNotFoundError, version

from .api import (
    estimate,
    estimate_trend,
    gp_trend,
    kalman_trend,
    l1_trend_filter,
    local_polynomial_trend,
    loess_trend,
    naive_trend,
    sgolay_trend,
    smoothing_spline_trend,
)
from .axis import TimeAxis
from .noise import AR1, IID, Given, Heteroskedastic, NoiseModel, local_sigma
from .process import GaussianProcess, StateSpace
from .ranking import trending
from .result import TrendEstimate
from .seasonal import (
    Seasonality,
    deseasonalize,
    detect_seasonality,
    moving_average_decompose,
    stl_decompose,
    trend_with_deseasonalization,
)
from .simulate import generate_time_series, standard_test_functions
from .sizer import SiZer, SiZerMap, sizer_analysis, trend_with_sizer
from .smoothers import (
    SMOOTHERS,
    L1TrendFilter,
    LocalPolynomial,
    Loess,
    NaiveDifference,
    SavitzkyGolay,
    Smoother,
    SmoothingSpline,
    build,
)

try:
    __version__ = version("incline")
except PackageNotFoundError:  # pragma: no cover - only when running from source
    __version__ = "2.0.0.dev0"

__all__ = [
    "AR1",
    "IID",
    "SMOOTHERS",
    "GaussianProcess",
    "Given",
    "Heteroskedastic",
    "L1TrendFilter",
    "LocalPolynomial",
    "Loess",
    "NaiveDifference",
    "NoiseModel",
    "SavitzkyGolay",
    "Seasonality",
    "SiZer",
    "SiZerMap",
    "Smoother",
    "SmoothingSpline",
    "StateSpace",
    "TimeAxis",
    "TrendEstimate",
    "build",
    "deseasonalize",
    "detect_seasonality",
    "estimate",
    "estimate_trend",
    "generate_time_series",
    "gp_trend",
    "kalman_trend",
    "l1_trend_filter",
    "local_polynomial_trend",
    "local_sigma",
    "loess_trend",
    "moving_average_decompose",
    "naive_trend",
    "sgolay_trend",
    "sizer_analysis",
    "smoothing_spline_trend",
    "standard_test_functions",
    "stl_decompose",
    "trend_with_deseasonalization",
    "trend_with_sizer",
    "trending",
]
