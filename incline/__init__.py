"""Estimate the trend at a point in a noisy time series, and say how sure you are.

Naive differencing amplifies noise, so incline smooths first and differentiates
the smooth. The interesting part is the second half: every estimator reports a
standard error, and which machinery produces it is decided by what the smoother
*is* rather than by what it is called.

    >>> import numpy as np, pandas as pd
    >>> from incline import sgolay_trend
    >>> df = pd.DataFrame({"value": np.arange(50.0)},
    ...                   index=pd.date_range("2020-01-01", periods=50))
    >>> out = sgolay_trend(df, se=True)
    >>> round(out["derivative_value"].iloc[0], 6), out["se_method"].iloc[0]
    (1.0, 'operator')

Every estimator returns the same columns -- ``derivative_value``,
``derivative_se``, ``derivative_ci_lower``, ``derivative_ci_upper``,
``se_method`` and ``significant_trend``. A NaN standard error paired with
``se_method=None`` is a deliberate, documented state, never a missing column.

Standard errors are opt-in via ``se=True`` because the exact route costs one
smoother evaluation per observation. Noise is assumed independent unless you say
otherwise with ``noise='ar1'``; under real autocorrelation the independent
assumption understates the uncertainty substantially.
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
    pspline_trend,
    select_trend_method,
    sgolay_trend,
    spline_trend,
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
    InterpolatingSpline,
    L1TrendFilter,
    LocalPolynomial,
    Loess,
    NaiveDifference,
    PenalizedSpline,
    SavitzkyGolay,
    Smoother,
    build,
)


try:
    __version__ = version("incline")
except PackageNotFoundError:  # pragma: no cover - only when running from source
    __version__ = "1.0.0.dev0"

__all__ = [
    "AR1",
    "IID",
    "SMOOTHERS",
    "GaussianProcess",
    "Given",
    "Heteroskedastic",
    "InterpolatingSpline",
    "L1TrendFilter",
    "LocalPolynomial",
    "Loess",
    "NaiveDifference",
    "NoiseModel",
    "PenalizedSpline",
    "SavitzkyGolay",
    "Seasonality",
    "SiZer",
    "SiZerMap",
    "Smoother",
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
    "pspline_trend",
    "select_trend_method",
    "sgolay_trend",
    "sizer_analysis",
    "spline_trend",
    "standard_test_functions",
    "stl_decompose",
    "trend_with_deseasonalization",
    "trend_with_sizer",
    "trending",
]
