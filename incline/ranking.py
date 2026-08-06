"""Ranking many series by how strongly they are trending.

The point of this layer is to answer "which of these thousands of series is
moving fastest right now", and the honest version of that answer needs to know
which apparent movements are real.

Previously it could not. Each series arrived as a DataFrame of point estimates,
standard errors were discarded even when the smoother had produced them, and
significance came from resampling the last k derivative values as though they
were independent draws. They are not: they are adjacent points on one smoothed
curve, so they are strongly positively correlated and nearly identical by
construction. That understates the true uncertainty badly.

Now a :class:`~incline.result.TrendEstimate` carries its own standard errors and
this module propagates them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import norm, rankdata

from .result import TrendEstimate


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

Aggregation = Literal["max", "mean", "median", "last"]
Weighting = Literal["uniform", "linear", "exponential"]


def _weights(k: int, scheme: Weighting) -> npt.NDArray[np.float64]:
    """Weights over the last ``k`` observations, most recent last."""
    match scheme:
        case "uniform":
            w = np.ones(k)
        case "linear":
            w = np.arange(1, k + 1, dtype=float)
        case "exponential":
            w = np.exp(np.linspace(-2.0, 0.0, k))
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
    how: Aggregation,
    weights: npt.NDArray[np.float64],
) -> _Summary:
    """Reduce a window of derivative estimates to one number, with its error.

    A weighted mean is a linear functional, so when the full covariance of the
    window is available its variance is exactly ``w' C w``. Without the
    covariance, the standard errors alone still bound it: adjacent derivative
    estimates from one smooth are positively correlated, so treating them as
    perfectly correlated gives ``(sum_i w_i se_i)**2``, an upper bound that
    cannot understate the uncertainty. That is the right direction to err.

    ``max`` is not linear; its uncertainty is reported as that of the selected
    point, which ignores the selection itself and is therefore optimistic.
    """
    finite = np.isfinite(derivative)
    if not np.any(finite):
        return _Summary(float("nan"), float("nan"), False)

    match how:
        case "max":
            position = int(np.nanargmax(derivative))
            value = float(derivative[position])
            error = (
                float(standard_errors[position])
                if standard_errors is not None
                else float("nan")
            )
            return _Summary(value, error, False)
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
            error = (
                float(np.nanmedian(standard_errors))
                if standard_errors is not None
                else float("nan")
            )
            return _Summary(value, error, False)
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
            raise ValueError(f"Unknown aggregation {how!r}")


def trending(
    estimates: Mapping[str, TrendEstimate] | Sequence[TrendEstimate],
    k: int = 5,
    how: Aggregation = "mean",
    weighting: Weighting = "uniform",
    confidence_level: float = 0.95,
    ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Rank series by recent trend strength.

    Args:
        estimates: Estimates to rank, either as a mapping from identifier to
            estimate or as a sequence alongside ``ids``.
        k: How many of the most recent observations to summarize.
        how: ``'mean'``, ``'max'``, ``'median'`` or ``'last'``. Only ``'mean'``
            and ``'last'`` propagate uncertainty exactly.
        weighting: Weighting across the window for ``'mean'``.
        confidence_level: Confidence level for the reported interval.
        ids: Identifiers, when ``estimates`` is a sequence.

    Returns:
        One row per series with columns ``id``, ``trend``, ``trend_se``,
        ``ci_lower``, ``ci_upper``, ``significant``, ``se_exact``, ``rank``,
        sorted strongest first. The schema does not change when the input is
        empty.

    Raises:
        ValueError: If identifiers cannot be determined.
    """
    if isinstance(estimates, dict):
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
        "trend_se",
        "ci_lower",
        "ci_upper",
        "significant",
        "se_exact",
        "rank",
    ]
    if not pairs:
        return pd.DataFrame({name: [] for name in columns})

    z = float(norm.ppf(1 - (1 - confidence_level) / 2))
    rows = []
    for identifier, estimate in pairs:
        window = min(k, estimate.axis.n)
        derivative = estimate.derivative[-window:]
        standard_errors = estimate.se[-window:] if estimate.se is not None else None
        summary = _aggregate(
            derivative, standard_errors, None, how, _weights(window, weighting)
        )
        has_error = np.isfinite(summary.standard_error)
        lower = summary.value - z * summary.standard_error if has_error else np.nan
        upper = summary.value + z * summary.standard_error if has_error else np.nan
        rows.append(
            {
                "id": identifier,
                "trend": summary.value,
                "trend_se": summary.standard_error,
                "ci_lower": lower,
                "ci_upper": upper,
                "significant": bool(has_error and (lower > 0 or upper < 0)),
                "se_exact": summary.exact,
            }
        )

    frame = pd.DataFrame(rows)
    frame["rank"] = rankdata(-frame["trend"], method="ordinal")
    return frame.sort_values("rank").reset_index(drop=True)[columns]
