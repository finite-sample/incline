"""Rank many series by how strongly they are trending.

The ranking consumes :class:`~incline.result.TrendEstimate` objects so it can
carry estimator uncertainty into the summary instead of treating adjacent
points on a smoothed curve as independent observations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import norm, rankdata

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .result import TrendEstimate

Aggregation = Literal["max", "mean", "median", "last"]
Weighting = Literal["uniform", "linear", "exponential"]


def _weights(
    window_length: int, scheme: Weighting, half_life: float | None
) -> npt.NDArray[np.float64]:
    """Weights over the last ``window_length`` observations, most recent last."""
    match scheme:
        case "uniform":
            w = np.ones(window_length)
        case "linear":
            w = np.arange(1, window_length + 1, dtype=float)
        case "exponential":
            if half_life is None:  # validated by the public boundary
                raise ValueError("half_life is required for exponential weighting")
            age = np.arange(window_length - 1, -1, -1, dtype=float)
            w = 2.0 ** (-age / half_life)
        case _:
            raise ValueError(f"Unknown weighting {scheme!r}")
    return w / w.sum()


@dataclass(frozen=True)
class _Summary:
    """One series' aggregated trend and its uncertainty."""

    value: float
    standard_error: float
    exact: bool


def _aggregate(
    derivative: npt.NDArray[np.float64],
    standard_errors: npt.NDArray[np.float64] | None,
    covariance: npt.NDArray[np.float64] | None,
    aggregation: Aggregation,
    weights: npt.NDArray[np.float64],
) -> _Summary:
    """Reduce a window of derivative estimates to one number, with its error.

    A weighted mean is a linear functional, so when the full covariance of the
    window is available its variance is exactly ``w' C w``. Without the
    covariance, the standard errors alone still bound it: adjacent derivative
    estimates from one smooth are positively correlated, so treating them as
    perfectly correlated gives ``(sum_i w_i se_i)**2``, an upper bound that
    cannot understate the uncertainty. That is the right direction to err.

    ``max`` and ``median`` are not linear, so no interval is reported for
    either summary.
    """
    finite = np.isfinite(derivative)
    if not np.any(finite):
        return _Summary(float("nan"), float("nan"), False)

    match aggregation:
        case "max":
            position = int(np.nanargmax(derivative))
            value = float(derivative[position])
            return _Summary(value, float("nan"), False)
        case "last":
            position = int(np.max(np.flatnonzero(finite)))
            value = float(derivative[position])
            error = (
                float(standard_errors[position])
                if standard_errors is not None
                else float("nan")
            )
            return _Summary(value, error, standard_errors is not None)
        case "median":
            value = float(np.nanmedian(derivative))
            return _Summary(value, float("nan"), False)
        case "mean":
            w = np.where(finite, weights, 0.0)
            total = w.sum()
            if total <= 0:
                return _Summary(float("nan"), float("nan"), False)
            w = w / total
            value = float(np.nansum(w * derivative))
            if covariance is not None:
                variance = float(w @ covariance @ w)
                return _Summary(value, float(np.sqrt(max(variance, 0.0))), True)
            if standard_errors is not None:
                bound = float(np.nansum(w * standard_errors))
                return _Summary(value, bound, False)
            return _Summary(value, float("nan"), False)
        case _:
            raise ValueError(f"Unknown aggregation {aggregation!r}")


def trending(
    estimates: Mapping[str, TrendEstimate] | Sequence[TrendEstimate],
    window_length: int = 5,
    aggregation: Aggregation = "mean",
    weighting: Weighting = "uniform",
    half_life: float | None = None,
    confidence_level: float = 0.95,
    ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Rank series by recent trend strength.

    Args:
        estimates: Estimates to rank, either as a mapping from identifier to
            estimate or as a sequence alongside ``ids``.
        window_length: How many of the most recent observations to summarize.
        aggregation: ``'mean'``, ``'max'``, ``'median'`` or ``'last'``. The
            last value preserves its pointwise uncertainty exactly; the mean
            uses a conservative bound because the covariance is unavailable.
        weighting: Weighting across the window for ``'mean'``.
        half_life: Observations required for exponential weights to halve.
            Required when ``weighting='exponential'`` and unavailable otherwise.
        confidence_level: Confidence level for the reported interval.
        ids: Identifiers, when ``estimates`` is a sequence.

    Returns:
        One row per series with columns ``id``, ``trend``, ``trend_standard_error``,
        ``ci_lower``, ``ci_upper``, ``significant``, ``uncertainty_exact``, ``rank``,
        sorted strongest first. The schema does not change when the input is
        empty.

    Raises:
        TypeError: If ``ids`` is supplied with mapping input.
        ValueError: If an argument is outside its domain or identifiers cannot
            be determined.
    """
    if (
        isinstance(window_length, (bool, np.bool_))
        or not isinstance(window_length, (int, np.integer))
        or window_length < 1
    ):
        raise ValueError("window_length must be a positive integer")
    if aggregation not in {"mean", "max", "median", "last"}:
        raise ValueError(f"Unknown aggregation {aggregation!r}")
    if weighting not in {"uniform", "linear", "exponential"}:
        raise ValueError(f"Unknown weighting {weighting!r}")
    if aggregation != "mean" and weighting != "uniform":
        raise ValueError("weighting is only available when aggregation='mean'")
    if weighting == "exponential":
        if (
            isinstance(half_life, (bool, np.bool_))
            or not isinstance(half_life, (int, float, np.integer, np.floating))
            or not np.isfinite(half_life)
            or half_life <= 0
        ):
            raise ValueError(
                "half_life must be finite and positive for exponential weighting"
            )
    elif half_life is not None:
        raise ValueError("half_life is only available for exponential weighting")
    if (
        isinstance(confidence_level, (bool, np.bool_))
        or not isinstance(confidence_level, (int, float, np.integer, np.floating))
        or not np.isfinite(confidence_level)
        or not 0.0 < confidence_level < 1.0
    ):
        raise ValueError("confidence_level must be finite and strictly between 0 and 1")

    # Mapping, not dict: a non-dict Mapping used to fall through to the
    # sequence branch, which iterates keys and would have ranked strings.
    if isinstance(estimates, Mapping):
        if ids is not None:
            raise TypeError("ids is unavailable when estimates is a mapping")
        pairs = list(estimates.items())
    else:
        sequence = list(estimates)
        if ids is None:
            ids = [str(i) for i in range(len(sequence))]
        if len(ids) != len(sequence):
            raise ValueError(f"got {len(ids)} ids for {len(sequence)} estimates")
        pairs = list(zip(ids, sequence, strict=True))

    columns = [
        "id",
        "trend",
        "trend_standard_error",
        "ci_lower",
        "ci_upper",
        "significant",
        "uncertainty_exact",
        "rank",
    ]
    if not pairs:
        return pd.DataFrame({name: [] for name in columns})

    z = float(norm.ppf(1 - (1 - confidence_level) / 2))
    rows = []
    for identifier, estimate in pairs:
        window = min(window_length, estimate.axis.n)
        derivative = estimate.derivative[-window:]
        standard_errors = (
            estimate.standard_error[-window:]
            if estimate.standard_error is not None
            else None
        )
        summary = _aggregate(
            derivative,
            standard_errors,
            None,
            aggregation,
            _weights(window, weighting, half_life),
        )
        has_error = np.isfinite(summary.standard_error) and summary.standard_error > 0
        lower = summary.value - z * summary.standard_error if has_error else np.nan
        upper = summary.value + z * summary.standard_error if has_error else np.nan
        rows.append(
            {
                "id": identifier,
                "trend": summary.value,
                "trend_standard_error": summary.standard_error,
                "ci_lower": lower,
                "ci_upper": upper,
                "significant": bool(has_error and (lower > 0 or upper < 0)),
                "uncertainty_exact": summary.exact,
            }
        )

    frame = pd.DataFrame(rows)
    # rankdata propagates NaN by default, so a single series whose window is
    # entirely non-finite -- easy with local-polynomial edge NaNs -- returned
    # NaN for *every* rank and left the sort in arbitrary order. Unrankable
    # series go last instead.
    order = np.nan_to_num(-frame["trend"].to_numpy(dtype=float), nan=np.inf)
    frame["rank"] = rankdata(order, method="ordinal")
    # cast: pandas annotates list-key __getitem__ as a Series/DataFrame union.
    return cast(
        "pd.DataFrame", frame.sort_values("rank").reset_index(drop=True)[columns]
    )
