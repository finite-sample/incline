"""Tests for the functional surface.

These run over every public estimator at once. The facade is thin by design, so
what is worth testing here is that it is thin *uniformly*: the same schema, the
same option handling, the same behavior on awkward input, whichever estimator
you reach for.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import pytest

from incline import api
from incline.result import CORE_COLUMNS
from incline.smoothers import SMOOTHERS

N = 80
L1_ESTIMATOR = partial(api.l1_trend_filter, penalty_fraction=0.2)

ESTIMATORS = [
    pytest.param(api.naive_trend, id="naive"),
    pytest.param(api.sgolay_trend, id="sgolay"),
    pytest.param(api.smoothing_spline_trend, id="smoothing_spline"),
    pytest.param(api.loess_trend, id="loess"),
    pytest.param(api.local_polynomial_trend, id="local_poly"),
    pytest.param(L1_ESTIMATOR, id="l1_filter"),
    pytest.param(api.gp_trend, id="gp"),
    pytest.param(api.kalman_trend, id="kalman"),
]


def frame(seed: int = 0, slope: float = 0.05, n: int = N) -> pd.DataFrame:
    """A noisy trending series on a daily index."""
    x = np.arange(n, dtype=float)
    values = slope * x + np.sin(x / 12) + np.random.default_rng(seed).normal(0, 0.3, n)
    return pd.DataFrame({"value": values}, index=pd.date_range("2020-01-01", periods=n))


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_every_estimator_accepts_a_datetime_index(estimator):
    """Regression: local_polynomial_trend used to crash on any DatetimeIndex.

    The time derivation returned a pandas Index rather than an ndarray, and the
    entire old test suite used a RangeIndex, so the package's most common input
    went untested.
    """
    result = estimator(frame())
    assert len(result) == N
    assert result["derivative_value"].notna().sum() > N // 2


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_every_estimator_returns_the_core_schema(estimator):
    """One contract, whichever method produced the numbers."""
    result = estimator(frame())
    for column in CORE_COLUMNS:
        assert column in result.columns, f"missing {column}"
    assert "value" in result.columns


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_uncertainty_is_opt_in(estimator):
    """Without with_uncertainty=True the columns exist and are honestly empty."""
    result = estimator(frame())
    assert result["derivative_standard_error"].isna().all()
    assert result["uncertainty_method"].isna().all() | (
        result["uncertainty_method"].iloc[0] is None
    )
    assert not result["significant_trend"].any()


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_se_true_fills_in_the_uncertainty(estimator):
    """And with it, every uncertainty column is populated."""
    result = estimator(frame(), with_uncertainty=True, n_bootstrap=30)
    assert result["derivative_standard_error"].notna().any()
    assert result["derivative_ci_lower"].notna().any()
    assert result["uncertainty_method"].iloc[0] in {"operator", "bootstrap", "native"}


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("step", [1.0, 0.5, 2.0])
def test_reported_slope_is_invariant_to_sampling_step(estimator, step):
    """Regression: on y = 2t the slope is 2.0 per unit t at any spacing.

    Dividing by the time step twice, or forgetting to, both show up here. The
    original was a Gaussian process bug, but it is a property every estimator
    has to satisfy, so it is checked against all of them.
    """
    x = np.arange(N, dtype=float) * step
    data = pd.DataFrame({"value": 2.0 * x, "t": x})
    result = estimator(data, time_column="t")
    interior = result["derivative_value"].to_numpy()[15:-15]
    interior = interior[np.isfinite(interior)]
    assert np.median(interior) == pytest.approx(2.0, rel=0.15)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_integer_time_column_is_not_read_as_positional(estimator):
    """Regression: np.int64 is not a subclass of int.

    With a column stepping by 10, treating the index as positional turns a
    slope of 2.0 into 20.0 -- an order-of-magnitude error that still looks
    plausible in isolation.
    """
    times = (np.arange(N) * 10).astype(np.int64)
    data = pd.DataFrame({"value": 2.0 * times.astype(float), "t": times})
    result = estimator(data, time_column="t")
    interior = result["derivative_value"].to_numpy()[15:-15]
    interior = interior[np.isfinite(interior)]
    assert np.median(interior) == pytest.approx(2.0, rel=0.15)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_wider_confidence_level_widens_the_interval(estimator):
    """Regression: a dispatch layer must forward confidence_level.

    An adaptive branch once reset intervals to the default level, silently
    ignoring what the caller asked for.
    """
    data = frame(3)
    narrow = estimator(
        data,
        with_uncertainty=True,
        confidence_level=0.5,
        n_bootstrap=30,
        random_state=1,
    )
    wide = estimator(
        data,
        with_uncertainty=True,
        confidence_level=0.99,
        n_bootstrap=30,
        random_state=1,
    )
    narrow_width = (
        narrow["derivative_ci_upper"] - narrow["derivative_ci_lower"]
    ).mean()
    wide_width = (wide["derivative_ci_upper"] - wide["derivative_ci_lower"]).mean()
    assert wide_width > narrow_width


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_custom_value_column_is_honoured(estimator):
    """Nobody's column is called 'value'."""
    data = frame().rename(columns={"value": "price"})
    result = estimator(data, value_column="price")
    assert "price" in result.columns
    assert result["derivative_value"].notna().sum() > N // 2


@pytest.mark.parametrize(
    "estimator",
    [
        pytest.param(api.smoothing_spline_trend, id="smoothing_spline"),
        pytest.param(api.loess_trend, id="loess"),
        pytest.param(api.local_polynomial_trend, id="local_poly"),
        pytest.param(L1_ESTIMATOR, id="l1_filter"),
        pytest.param(api.gp_trend, id="gp"),
    ],
)
def test_irregular_sampling_is_handled_by_irregular_grid_methods(estimator):
    """Methods whose arithmetic uses the actual axis accept uneven spacing."""
    rng = np.random.default_rng(5)
    x = np.sort(rng.uniform(0, 100, N))
    data = pd.DataFrame({"value": 0.05 * x + rng.normal(0, 0.2, N), "t": x})
    result = estimator(data, time_column="t")
    assert result["derivative_value"].notna().sum() > N // 2


@pytest.mark.parametrize(
    "estimator", [api.naive_trend, api.sgolay_trend, api.kalman_trend]
)
def test_irregular_sampling_is_refused_by_grid_methods(estimator):
    """Grid stencils must not return a known-wrong derivative off-grid."""
    rng = np.random.default_rng(5)
    x = np.sort(rng.uniform(0, 100, N))
    data = pd.DataFrame({"value": 2.0 * x, "t": x})
    with pytest.raises(ValueError, match="uniform sampling"):
        estimator(data, time_column="t")


def test_estimate_trend_defaults_to_smoothing_spline():
    result = api.estimate_trend(frame())
    assert result["derivative_method"].iloc[0] == "smoothing_spline"


def test_removed_auto_method_is_refused():
    with pytest.raises(ValueError, match="Unknown method"):
        api.estimate_trend(frame(), method="auto")


def test_unknown_method_is_refused():
    """And likewise at the dispatcher."""
    with pytest.raises(ValueError, match="Unknown method"):
        api.estimate_trend(frame(), method="magic")


@pytest.mark.parametrize("name", sorted(SMOOTHERS))
def test_estimate_trend_reaches_every_registered_smoother(name):
    """A newly registered smoother is reachable without editing the dispatcher."""
    if name == "l1_filter":
        result = api.estimate_trend(frame(), method=name, penalty_fraction=0.2)
    else:
        result = api.estimate_trend(frame(), method=name)
    assert result["derivative_method"].iloc[0] == name


def test_estimate_trend_splits_constructor_and_fit_arguments():
    """Smoother settings and uncertainty options arrive through one **kwargs."""
    result = api.estimate_trend(
        frame(),
        method="sgolay",
        window_length=21,
        with_uncertainty=True,
        confidence_level=0.9,
    )
    assert result["window_length"].iloc[0] == 21
    assert result["derivative_standard_error"].notna().any()


def test_pilot_scale_reaches_bias_correction_through_both_public_routes():
    """The public facades must expose every supported fit option."""
    data = frame()
    smoother = SMOOTHERS["sgolay"](window_length=21)
    structured = api.estimate(
        smoother,
        data,
        with_uncertainty=True,
        bias_correct=True,
        pilot_scale=0.1,
    )
    tabular = api.estimate_trend(
        data,
        method="sgolay",
        window_length=21,
        with_uncertainty=True,
        bias_correct=True,
        pilot_scale=0.1,
    )
    np.testing.assert_allclose(
        tabular["derivative_value"], structured.derivative, equal_nan=True
    )


def test_estimate_returns_the_structured_object():
    """The object form is available for callers who want more than a frame."""
    from incline.smoothers import SavitzkyGolay

    estimate = api.estimate(SavitzkyGolay(), frame(), with_uncertainty=True)
    assert estimate.standard_error is not None
    assert estimate.provenance.method == "sgolay"
    assert next(iter(estimate.to_frame(frame()).columns)) == "value"


def test_estimate_preserves_the_source_index_without_the_source_frame():
    """The structured result must not discard a meaningful input index."""
    from incline.smoothers import SavitzkyGolay

    data = frame()
    result = api.estimate(SavitzkyGolay(), data).to_frame()
    pd.testing.assert_index_equal(result.index, data.index)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_missing_values_do_not_crash(estimator):
    """Real series have gaps; an estimator may refuse but must not explode."""
    data = frame(9)
    data.iloc[10:15, 0] = np.nan
    try:
        result = estimator(data)
    except (ValueError, np.linalg.LinAlgError):
        pytest.skip("this estimator declines NaN input, which is a valid answer")
    assert len(result) == N


def test_second_derivative_is_available_where_supported():
    """Curvature is a first-class request, not an afterthought."""
    x = np.arange(N, dtype=float)
    data = pd.DataFrame({"value": 0.5 * x**2, "t": x})
    result = api.sgolay_trend(
        data, time_column="t", derivative_order=2, window_length=21
    )
    interior = result["derivative_value"].to_numpy()[15:-15]
    assert np.allclose(interior, 1.0, atol=0.05)
    assert result["derivative_order"].iloc[0] == 2


def test_a_class_attribute_is_not_a_constructor_argument():
    """Regression: routing read ``__dataclass_fields__``, which holds ClassVars.

    ``linear``, ``name`` and ``supported_orders`` describe the class rather than
    configure an instance, so ``estimate_trend(df, method="sgolay", linear=False)``
    was routed to the constructor and raised from inside dataclasses instead of
    being reported as an unknown option.
    """
    frame = pd.DataFrame({"value": np.arange(40.0)})
    for attribute in ("linear", "name", "supported_orders"):
        with pytest.raises(TypeError, match="does not accept"):
            api.estimate_trend(frame, method="sgolay", **{attribute: False})


@pytest.mark.parametrize(
    ("function", "old_name", "value"),
    [
        (api.sgolay_trend, "column_value", "value"),
        (api.sgolay_trend, "function_order", 3),
        (api.sgolay_trend, "se", True),
        (api.smoothing_spline_trend, "s", 5.0),
        (api.smoothing_spline_trend, "lam", 1.0),
        (partial(api.l1_trend_filter, penalty_fraction=0.2), "lambda_param", 1.0),
        (api.loess_trend, "frac", 0.3),
        (api.kalman_trend, "seasonal_periods", 12),
        (api.loess_trend, "degree", 2),
    ],
)
def test_removed_public_names_are_not_accepted(function, old_name, value):
    """The breaking vocabulary migration leaves no hidden compatibility layer."""
    with pytest.raises(TypeError):
        function(frame(), **{old_name: value})


def test_l1_function_requires_exactly_one_penalty_parameter():
    """The functional API must not choose an undocumented smoothing strength."""
    with pytest.raises(ValueError, match="exactly one"):
        api.l1_trend_filter(frame())
    with pytest.raises(ValueError, match="exactly one"):
        api.l1_trend_filter(frame(), penalty=1.0, penalty_fraction=0.2)


def test_removed_spline_apis_are_not_exported():
    """The removed spline implementations leave no compatibility aliases."""
    import incline

    for name in (
        "InterpolatingSpline",
        "PenalizedSpline",
        "pspline_trend",
        "spline_trend",
    ):
        assert not hasattr(incline, name)
        assert not hasattr(api, name)


def test_uncertainty_options_reach_the_fit_and_settings_reach_the_constructor():
    """The two destinations stay distinguishable after the allowlist."""
    frame = pd.DataFrame({"value": np.arange(60.0) * 0.1})
    result = api.estimate_trend(
        frame, method="sgolay", window_length=21, with_uncertainty=True
    )
    assert result["window_length"].iloc[0] == 21
    assert result["derivative_standard_error"].notna().all()
