"""The functional surface of the package.

Users want ``sgolay_trend(df, se=True)``, not a constructor and a fit call, so
these functions stay the documented API. They are deliberately thin: build a
:class:`~incline.axis.TimeAxis`, build a :class:`~incline.smoothers.Smoother`,
fit, render. No estimation logic lives here, which is the point -- when it did,
every method needed its own copy of the time handling and its own answer to the
question of uncertainty.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np

from .axis import TimeAxis
from .process import GaussianProcess, StateSpace
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

if TYPE_CHECKING:
    import pandas as pd

    from .noise import NoiseModel
    from .result import TrendEstimate

__all__ = [
    "estimate",
    "estimate_trend",
    "gp_trend",
    "kalman_trend",
    "l1_trend_filter",
    "local_polynomial_trend",
    "loess_trend",
    "naive_trend",
    "pspline_trend",
    "select_trend_method",
    "sgolay_trend",
    "spline_trend",
]


def estimate(
    smoother: Smoother,
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    derivative_order: int = 1,
    se: bool = False,
    noise: NoiseModel | str | None = None,
    bias_correct: bool = False,
    simultaneous: bool = False,
    confidence_level: float = 0.95,
    n_bootstrap: int = 200,
    random_state: int | np.random.Generator | None = None,
) -> TrendEstimate:
    """Run a smoother over a DataFrame and return the structured estimate.

    Args:
        smoother: The smoother to fit.
        df: Time series data.
        column_value: Column holding the values.
        time_column: Numeric time column. The index is used when None.
        derivative_order: Which derivative to estimate.
        se: Whether to compute standard errors.
        noise: Noise model, or ``'iid'`` / ``'ar1'``.
        bias_correct: Subtract estimated smoothing bias. Linear smoothers only.
        simultaneous: Return a whole-curve band instead of pointwise intervals.
        confidence_level: Confidence level for intervals.
        n_bootstrap: Replicates for nonlinear smoothers.
        random_state: Seed or Generator.

    Returns:
        The estimate, which ``.to_frame(df)`` renders as a DataFrame.
    """
    axis = TimeAxis.from_frame(df, time_column)
    y = np.asarray(df[column_value], dtype=np.float64)
    return smoother.fit(
        axis,
        y,
        order=derivative_order,
        se=se,
        noise=noise,
        bias_correct=bias_correct,
        simultaneous=simultaneous,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )


def _frame(
    smoother: Smoother,
    df: pd.DataFrame,
    column_value: str,
    time_column: str | None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Fit and render in one step."""
    return estimate(smoother, df, column_value, time_column, **kwargs).to_frame(df)


def naive_trend(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Estimate the trend by central finite differences.

    Does no smoothing, so it inherits the noise directly. Present mostly as
    the baseline the smoothing methods are meant to beat.

    Args:
        df: Time series data.
        column_value: Column holding the values.
        time_column: Numeric time column.
        **kwargs: Uncertainty options; see :func:`estimate`.

    Returns:
        The input frame plus the estimate columns.
    """
    return _frame(NaiveDifference(), df, column_value, time_column, **kwargs)


def sgolay_trend(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    function_order: int = 3,
    window_length: int = 15,
    **kwargs: Any,
) -> pd.DataFrame:
    """Estimate the trend with a Savitzky-Golay filter.

    A fixed linear filter, so standard errors are exact.

    Args:
        df: Time series data.
        column_value: Column holding the values.
        time_column: Numeric time column.
        function_order: Degree of the local polynomial.
        window_length: Filter window in observations; forced odd.
        **kwargs: Uncertainty options; see :func:`estimate`.

    Returns:
        The input frame plus the estimate columns.
    """
    smoother = SavitzkyGolay(window_length=window_length, polyorder=function_order)
    return _frame(smoother, df, column_value, time_column, **kwargs)


def spline_trend(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    function_order: int = 3,
    s: float | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Estimate the trend with a knot-selecting smoothing spline.

    ``UnivariateSpline`` places knots to meet a residual budget, so the fit
    depends on the data and no exact operator exists; standard errors come
    from the bootstrap. Use :func:`pspline_trend` for the exact route.

    Args:
        df: Time series data.
        column_value: Column holding the values.
        time_column: Numeric time column.
        function_order: Spline degree.
        s: Residual budget. Derived from the noise level when None.
        **kwargs: Uncertainty options; see :func:`estimate`.

    Returns:
        The input frame plus the estimate columns.
    """
    smoother = InterpolatingSpline(function_order=function_order, s=s)
    return _frame(smoother, df, column_value, time_column, **kwargs)


def pspline_trend(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    lam: float | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Estimate the trend with a penalized smoothing spline.

    With ``lam`` fixed this is a linear smoother and standard errors are
    exact. Leaving ``lam`` as None selects it by generalized cross-validation,
    which makes the fit data-dependent and routes uncertainty to the bootstrap.

    Args:
        df: Time series data.
        column_value: Column holding the values.
        time_column: Numeric time column.
        lam: Roughness penalty.
        **kwargs: Uncertainty options; see :func:`estimate`.

    Returns:
        The input frame plus the estimate columns.
    """
    return _frame(PenalizedSpline(lam=lam), df, column_value, time_column, **kwargs)


def loess_trend(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    frac: float = 0.3,
    degree: int = 1,
    robust: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Estimate the trend with LOESS.

    ``robust=True`` reweights according to the residuals, which makes the fit
    data-dependent; standard errors then come from the bootstrap. With
    ``robust=False`` the smoother is linear and they are exact.

    Args:
        df: Time series data.
        column_value: Column holding the values.
        time_column: Numeric time column.
        frac: Fraction of the sample in each local regression.
        degree: Degree of the local polynomial.
        robust: Whether to run robustifying iterations.
        **kwargs: Uncertainty options; see :func:`estimate`.

    Returns:
        The input frame plus the estimate columns.
    """
    smoother = Loess(frac=frac, degree=degree, robust=robust)
    return _frame(smoother, df, column_value, time_column, **kwargs)


def local_polynomial_trend(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    bandwidth: float = 0.2,
    degree: int = 2,
    kernel: str = "gaussian",
    **kwargs: Any,
) -> pd.DataFrame:
    """Estimate the trend by local polynomial regression.

    Linear given the bandwidth, so standard errors are exact and come from the
    same weighted least squares solve that produces the estimate.

    Args:
        df: Time series data.
        column_value: Column holding the values.
        time_column: Numeric time column.
        bandwidth: Kernel width as a fraction of the series span.
        degree: Degree of the local polynomial.
        kernel: ``'gaussian'``, ``'epanechnikov'`` or ``'uniform'``.
        **kwargs: Uncertainty options; see :func:`estimate`.

    Returns:
        The input frame plus the estimate columns.
    """
    smoother = LocalPolynomial(bandwidth=bandwidth, degree=degree, kernel=kernel)
    return _frame(smoother, df, column_value, time_column, **kwargs)


def l1_trend_filter(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    lambda_param: float = 1.0,
    difference_order: int = 2,
    **kwargs: Any,
) -> pd.DataFrame:
    """Estimate a piecewise-polynomial trend with sparse changes in slope.

    Args:
        df: Time series data.
        column_value: Column holding the values.
        time_column: Numeric time column.
        lambda_param: Penalty on the differences; larger means fewer kinks.
        difference_order: Order of the penalized difference. Two gives a
            piecewise-linear trend.
        **kwargs: Uncertainty options; see :func:`estimate`.

    Returns:
        The input frame plus the estimate columns.
    """
    smoother = L1TrendFilter(
        lambda_param=lambda_param, difference_order=difference_order
    )
    return _frame(smoother, df, column_value, time_column, **kwargs)


def gp_trend(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    kernel: str = "rbf",
    length_scale: float | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Estimate the trend with Gaussian process regression.

    The derivative of a Gaussian process is itself a Gaussian process, so both
    the estimate and its standard error are exact posterior quantities.

    Args:
        df: Time series data.
        column_value: Column holding the values.
        time_column: Numeric time column.
        kernel: ``'rbf'``, ``'matern32'`` or ``'matern52'``. Matern smoothness
            caps the derivative order.
        length_scale: Initial length scale; optimized when None.
        **kwargs: Uncertainty options; see :func:`estimate`.

    Returns:
        The input frame plus the estimate columns.
    """
    smoother = GaussianProcess(kernel=kernel, length_scale=length_scale)
    return _frame(smoother, df, column_value, time_column, **kwargs)


def kalman_trend(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    seasonal_periods: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Estimate the trend with a local linear trend state-space model.

    The slope is a state, so its standard error is a diagonal entry of the
    smoother covariance.

    Args:
        df: Time series data.
        column_value: Column holding the values.
        time_column: Numeric time column.
        seasonal_periods: Length of a seasonal cycle, if any.
        **kwargs: Uncertainty options; see :func:`estimate`.

    Returns:
        The input frame plus the estimate columns.
    """
    smoother = StateSpace(seasonal_periods=seasonal_periods)
    return _frame(smoother, df, column_value, time_column, **kwargs)


def select_trend_method(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    criteria: str = "auto",
) -> str:
    """Recommend a smoother for a series.

    Args:
        df: Time series data.
        column_value: Column holding the values.
        time_column: Numeric time column.
        criteria: ``'auto'``, ``'robust'``, ``'smooth'``, ``'changepoints'``,
            or ``'exact'`` to require a method with exact standard errors.

    Returns:
        A registered smoother name.

    Raises:
        ValueError: If the criteria is unknown.
    """
    match criteria:
        case "robust":
            return "loess"
        case "smooth":
            return "pspline"
        case "changepoints":
            return "l1_filter"
        case "exact":
            return "local_poly"
        case "auto":
            pass
        case _:
            raise ValueError(f"Unknown criteria: {criteria}")

    axis = TimeAxis.from_frame(df, time_column)
    y = np.asarray(df[column_value], dtype=np.float64)

    quartile_low, quartile_high = np.percentile(y, [25, 75])
    spread = quartile_high - quartile_low
    outliers = float(
        np.mean((y < quartile_low - 1.5 * spread) | (y > quartile_high + 1.5 * spread))
    )
    curvature = float(np.std(np.diff(y, n=2)))
    signal = float(np.std(np.diff(y)))
    signal_to_noise = signal / (curvature + 1e-10)

    if outliers > 0.1:
        return "loess"
    if signal_to_noise < 2:
        return "pspline"
    if not axis.is_regular:
        return "local_poly"
    if axis.n < 50:
        return "sgolay"
    return "loess"


def estimate_trend(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    method: str = "auto",
    derivative_order: int = 1,
    se: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Estimate a trend, choosing the method automatically if asked.

    Dispatches through the smoother registry rather than a hand-maintained
    branch per method, so a newly registered smoother is reachable here without
    editing this function.

    Args:
        df: Time series data.
        column_value: Column holding the values.
        time_column: Numeric time column.
        method: A registered smoother name, or ``'auto'``.
        derivative_order: Which derivative to estimate.
        se: Whether to compute standard errors.
        **kwargs: Split between the smoother's constructor and the uncertainty
            options of :func:`estimate`.

    Returns:
        The input frame plus the estimate columns.

    Raises:
        ValueError: If ``method`` names no registered smoother.
        TypeError: If a keyword matches neither the smoother's constructor
            nor the uncertainty options.
    """
    if method == "auto":
        method = select_trend_method(df, column_value, time_column)

    smoother_class = SMOOTHERS.get(method)
    if smoother_class is None:
        raise ValueError(f"Unknown method {method!r}; available: {sorted(SMOOTHERS)}")

    # dataclasses.fields() excludes ClassVar pseudo-fields. __dataclass_fields__
    # includes them, so `name`, `linear`, `supported_orders` and
    # `requires_regular_grid` were being routed into the constructor.
    # Every registered smoother is a dataclass, but the ABC cannot promise it.
    constructor_fields = {
        f.name
        for f in dataclasses.fields(smoother_class)  # pyright: ignore[reportArgumentType]
    }
    fit_options = {
        "derivative_order",
        "se",
        "noise",
        "bias_correct",
        "simultaneous",
        "confidence_level",
        "n_bootstrap",
        "random_state",
    }
    unknown = set(kwargs) - constructor_fields - fit_options
    if unknown:
        raise TypeError(
            f"{method} does not accept {sorted(unknown)}. Smoother settings: "
            f"{sorted(constructor_fields)}. Uncertainty options: "
            f"{sorted(fit_options)}."
        )
    construction = {k: v for k, v in kwargs.items() if k in constructor_fields}
    fitting = {k: v for k, v in kwargs.items() if k not in constructor_fields}

    smoother = build(method, **construction)
    return _frame(
        smoother,
        df,
        column_value,
        time_column,
        derivative_order=derivative_order,
        se=se,
        **fitting,
    )
