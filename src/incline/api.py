"""The functional surface of the package.

Users want ``sgolay_trend(df, with_uncertainty=True)``, not a constructor
and a fit call, so these functions stay the documented API. They are
deliberately thin: build a
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
    L1TrendFilter,
    LocalPolynomial,
    Loess,
    NaiveDifference,
    SavitzkyGolay,
    Smoother,
    SmoothingSpline,
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
    "sgolay_trend",
    "smoothing_spline_trend",
]


def estimate(
    smoother: Smoother,
    df: pd.DataFrame,
    value_column: str = "value",
    time_column: str | None = None,
    derivative_order: int = 1,
    with_uncertainty: bool = False,
    noise: NoiseModel | str | None = None,
    bias_correct: bool = False,
    simultaneous: bool = False,
    confidence_level: float = 0.95,
    pilot_scale: float | None = None,
    n_bootstrap: int = 200,
    random_state: int | np.random.Generator | None = None,
) -> TrendEstimate:
    """Run a smoother over a DataFrame and return the structured estimate.

    Args:
        smoother: The smoother to fit.
        df: Time series data.
        value_column: Column holding the values.
        time_column: Numeric time column. The index is used when None.
        derivative_order: Which derivative to estimate.
        with_uncertainty: Whether to compute standard errors.
        noise: Noise model, or ``'iid'`` / ``'ar1'``.
        bias_correct: Subtract estimated smoothing bias. Linear smoothers only.
        simultaneous: Return a whole-curve band for a fixed linear smoother.
        confidence_level: Confidence level for intervals.
        pilot_scale: Scale of the pilot fit used for bias correction.
        n_bootstrap: Replicates for nonlinear smoothers.
        random_state: Seed or Generator.

    Returns:
        The estimate, which ``.to_frame(df)`` renders as a DataFrame.
    """
    axis = TimeAxis.from_frame(df, time_column)
    y = np.asarray(df[value_column], dtype=np.float64)
    result = smoother.fit(
        axis,
        y,
        derivative_order=derivative_order,
        with_uncertainty=with_uncertainty,
        noise=noise,
        bias_correct=bias_correct,
        simultaneous=simultaneous,
        confidence_level=confidence_level,
        pilot_scale=pilot_scale,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    return dataclasses.replace(result, index=df.index.copy())


def _frame(
    smoother: Smoother,
    df: pd.DataFrame,
    value_column: str,
    time_column: str | None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Fit and render in one step."""
    return estimate(smoother, df, value_column, time_column, **kwargs).to_frame(df)


def naive_trend(
    df: pd.DataFrame,
    value_column: str = "value",
    time_column: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Estimate the trend by central finite differences.

    Does no smoothing, so it inherits the noise directly. Present mostly as
    the baseline the smoothing methods are meant to beat.

    Args:
        df: Time series data.
        value_column: Column holding the values.
        time_column: Numeric time column.
        **kwargs: Uncertainty options; see :func:`estimate`.

    Returns:
        The input frame plus the estimate columns.
    """
    return _frame(NaiveDifference(), df, value_column, time_column, **kwargs)


def sgolay_trend(
    df: pd.DataFrame,
    value_column: str = "value",
    time_column: str | None = None,
    degree: int = 3,
    window_length: int = 15,
    **kwargs: Any,
) -> pd.DataFrame:
    """Estimate the trend with a Savitzky-Golay filter.

    A fixed linear filter, so standard errors are exact.

    Args:
        df: Time series data.
        value_column: Column holding the values.
        time_column: Numeric time column.
        degree: Degree of the local polynomial.
        window_length: Filter window in observations; forced odd.
        **kwargs: Uncertainty options; see :func:`estimate`.

    Returns:
        The input frame plus the estimate columns.
    """
    smoother = SavitzkyGolay(window_length=window_length, degree=degree)
    return _frame(smoother, df, value_column, time_column, **kwargs)


def smoothing_spline_trend(
    df: pd.DataFrame,
    value_column: str = "value",
    time_column: str | None = None,
    penalty: float | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Estimate the trend with a cubic smoothing spline.

    With ``penalty`` fixed this is a linear smoother and standard errors are
    exact. Leaving ``penalty`` as None selects it by generalized cross-validation
    under independent noise or covariance-aware generalized maximum likelihood
    under a non-constant covariance. Adaptive fits route uncertainty to the
    bootstrap.

    Args:
        df: Time series data.
        value_column: Column holding the values.
        time_column: Numeric time column.
        penalty: Roughness penalty.
        **kwargs: Uncertainty options; see :func:`estimate`.

    Returns:
        The input frame plus the estimate columns.
    """
    return _frame(
        SmoothingSpline(penalty=penalty), df, value_column, time_column, **kwargs
    )


def loess_trend(
    df: pd.DataFrame,
    value_column: str = "value",
    time_column: str | None = None,
    span: float = 0.3,
    robust: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Estimate the trend with LOESS.

    ``robust=True`` reweights according to the residuals, which makes the fit
    data-dependent; standard errors then come from the bootstrap. With
    ``robust=False`` the smoother is linear and they are exact.

    Args:
        df: Time series data.
        value_column: Column holding the values.
        time_column: Numeric time column.
        span: Fraction of the sample in each local regression.
        robust: Whether to run robustifying iterations.
        **kwargs: Uncertainty options; see :func:`estimate`.

    Returns:
        The input frame plus the estimate columns.
    """
    smoother = Loess(span=span, robust=robust)
    return _frame(smoother, df, value_column, time_column, **kwargs)


def local_polynomial_trend(
    df: pd.DataFrame,
    value_column: str = "value",
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
        value_column: Column holding the values.
        time_column: Numeric time column.
        bandwidth: Kernel width as a fraction of the series span.
        degree: Degree of the local polynomial.
        kernel: ``'gaussian'``, ``'epanechnikov'`` or ``'uniform'``.
        **kwargs: Uncertainty options; see :func:`estimate`.

    Returns:
        The input frame plus the estimate columns.
    """
    smoother = LocalPolynomial(bandwidth=bandwidth, degree=degree, kernel=kernel)
    return _frame(smoother, df, value_column, time_column, **kwargs)


def l1_trend_filter(
    df: pd.DataFrame,
    value_column: str = "value",
    time_column: str | None = None,
    *,
    penalty: float | None = None,
    penalty_fraction: float | None = None,
    difference_order: int = 2,
    max_iter: int = 1000,
    tolerance: float = 1e-8,
    **kwargs: Any,
) -> pd.DataFrame:
    """Estimate a piecewise-polynomial trend with sparse changes in slope.

    Args:
        df: Time series data.
        value_column: Column holding the values.
        time_column: Numeric time column.
        penalty: Absolute penalty on the differences; larger means fewer kinks.
            Exactly one of this and ``penalty_fraction`` is required.
        penalty_fraction: Fraction of the smallest penalty that reduces the fit
            to a polynomial of degree ``difference_order - 1``. Exactly one of
            this and ``penalty`` is required.
        difference_order: Order of the penalized difference. Two gives a
            piecewise-linear trend.
        max_iter: Bounded least-squares iteration cap.
        tolerance: Optimizer convergence tolerance.
        **kwargs: Uncertainty options; see :func:`estimate`.

    Returns:
        The input frame plus the estimate columns.
    """
    smoother = L1TrendFilter(
        penalty=penalty,
        penalty_fraction=penalty_fraction,
        difference_order=difference_order,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    return _frame(smoother, df, value_column, time_column, **kwargs)


def gp_trend(
    df: pd.DataFrame,
    value_column: str = "value",
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
        value_column: Column holding the values.
        time_column: Numeric time column.
        kernel: ``'rbf'``, ``'matern32'`` or ``'matern52'``. Matern smoothness
            caps the derivative order.
        length_scale: Initial length scale; optimized when None.
        **kwargs: Uncertainty options; see :func:`estimate`.

    Returns:
        The input frame plus the estimate columns.
    """
    smoother = GaussianProcess(kernel=kernel, length_scale=length_scale)
    return _frame(smoother, df, value_column, time_column, **kwargs)


def kalman_trend(
    df: pd.DataFrame,
    value_column: str = "value",
    time_column: str | None = None,
    seasonal_period: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Estimate the trend with a local linear trend state-space model.

    The slope is a state, so its standard error is a diagonal entry of the
    smoother covariance. Sampling must be regular because the state transition
    advances once per observation.

    Args:
        df: Time series data.
        value_column: Column holding the values.
        time_column: Numeric time column.
        seasonal_period: Length of a seasonal cycle, if any.
        **kwargs: Uncertainty options; see :func:`estimate`.

    Returns:
        The input frame plus the estimate columns.
    """
    smoother = StateSpace(seasonal_period=seasonal_period)
    return _frame(smoother, df, value_column, time_column, **kwargs)


def estimate_trend(
    df: pd.DataFrame,
    value_column: str = "value",
    time_column: str | None = None,
    method: str = "smoothing_spline",
    derivative_order: int = 1,
    with_uncertainty: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Estimate a trend with a registered smoother.

    Dispatches through the smoother registry rather than a hand-maintained
    branch per method, so a newly registered smoother is reachable here without
    editing this function.

    Args:
        df: Time series data.
        value_column: Column holding the values.
        time_column: Numeric time column.
        method: A registered smoother name.
        derivative_order: Which derivative to estimate.
        with_uncertainty: Whether to compute standard errors.
        **kwargs: Split between the smoother's constructor and the uncertainty
            options of :func:`estimate`.

    Returns:
        The input frame plus the estimate columns.

    Raises:
        ValueError: If ``method`` names no registered smoother.
        TypeError: If a keyword matches neither the smoother's constructor
            nor the uncertainty options.
    """
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
        "with_uncertainty",
        "noise",
        "bias_correct",
        "simultaneous",
        "confidence_level",
        "pilot_scale",
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
        value_column,
        time_column,
        derivative_order=derivative_order,
        with_uncertainty=with_uncertainty,
        **fitting,
    )
