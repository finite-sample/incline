"""Tests for ranking many series by trend strength.

This layer used to discard standard errors even where the estimator had
produced them, and derived significance by resampling the last k derivative
values as though they were independent draws -- they are adjacent points on one
smoothed curve. Most of what follows pins the replacement.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from incline.axis import TimeAxis
from incline.ranking import trending
from incline.smoothers import SavitzkyGolay

N = 80
AXIS = TimeAxis.positional(N)

EXPECTED_COLUMNS = [
    "id",
    "trend",
    "trend_se",
    "ci_lower",
    "ci_upper",
    "significant",
    "se_exact",
    "rank",
]


def estimate_for(slope: float, seed: int = 0, se: bool = True):
    """A fitted estimate for a series with the given slope."""
    y = slope * AXIS.x + np.random.default_rng(seed).normal(0, 0.3, N)
    return SavitzkyGolay(window_length=15).fit(AXIS, y, order=1, se=se)


def test_ranks_strongest_trend_first():
    """The whole point: the fastest riser comes out on top."""
    result = trending(
        {
            "slow": estimate_for(0.01, 1),
            "fast": estimate_for(0.50, 2),
            "middle": estimate_for(0.10, 3),
        }
    )
    assert list(result["id"]) == ["fast", "middle", "slow"]
    assert list(result["rank"]) == [1, 2, 3]


def test_output_carries_the_documented_schema():
    """Column names are the contract this layer is consumed through."""
    result = trending({"a": estimate_for(0.1)})
    assert list(result.columns) == EXPECTED_COLUMNS


def test_empty_and_populated_paths_agree_on_the_schema():
    """Regression: the two return paths once emitted different columns.

    A caller reading ``result['id']`` worked or raised depending on whether
    anything matched.
    """
    empty = trending({})
    populated = trending({"a": estimate_for(0.1)})
    assert list(empty.columns) == list(populated.columns) == EXPECTED_COLUMNS
    assert len(empty) == 0


def test_standard_errors_are_propagated_not_discarded():
    """Uncertainty computed upstream must survive the aggregation."""
    result = trending({"a": estimate_for(0.2)}, how="mean")
    assert np.isfinite(result["trend_se"].iloc[0])
    assert result["ci_lower"].iloc[0] < result["trend"].iloc[0]
    assert result["trend"].iloc[0] < result["ci_upper"].iloc[0]


def test_missing_standard_errors_are_reported_as_missing():
    """Without se=True upstream there is nothing to propagate, and no pretending."""
    result = trending({"a": estimate_for(0.2, se=False)})
    assert np.isnan(result["trend_se"].iloc[0])
    assert not result["significant"].iloc[0]


def test_a_flat_series_is_not_called_significant():
    """Noise around zero slope must not clear its own error bar."""
    y = np.random.default_rng(7).normal(0, 0.3, N)
    flat = SavitzkyGolay(window_length=21).fit(AXIS, y, order=1, se=True)
    assert not trending({"flat": flat})["significant"].iloc[0]


def test_a_strong_trend_is_called_significant():
    """Calibration must not have been bought by never flagging anything."""
    assert trending({"steep": estimate_for(1.0)})["significant"].iloc[0]


@pytest.mark.parametrize("how", ["mean", "max", "median", "last"])
def test_every_aggregation_produces_the_same_schema(how):
    """Which summary you pick must not change what you have to index."""
    result = trending({"a": estimate_for(0.2), "b": estimate_for(0.4)}, how=how)
    assert list(result.columns) == EXPECTED_COLUMNS
    assert len(result) == 2


def test_unknown_aggregation_is_refused():
    """A typo must not silently pick a default summary."""
    with pytest.raises(ValueError, match="Unknown aggregation"):
        trending({"a": estimate_for(0.1)}, how="average")


def test_unknown_weighting_is_refused():
    """Likewise for the weighting scheme."""
    with pytest.raises(ValueError, match="Unknown weighting"):
        trending({"a": estimate_for(0.1)}, weighting="gaussian")


def test_weighting_actually_changes_the_answer():
    """Regression: a weighting argument that no branch honoured.

    ``avg`` was rewritten to a trimmed mean before the dispatch, which cannot
    weight, so the scheme was silently ignored on that path.
    """
    y = np.linspace(0, 5, N) ** 2
    estimate = SavitzkyGolay(window_length=15).fit(AXIS, y, order=1, se=True)
    values = {
        scheme: float(
            trending({"a": estimate}, k=10, how="mean", weighting=scheme)["trend"].iloc[
                0
            ]
        )
        for scheme in ("uniform", "linear", "exponential")
    }
    assert len(set(values.values())) == 3, values


def test_mean_uncertainty_is_conservative_rather_than_naive():
    """Adjacent derivative estimates are correlated, so independence understates.

    The bound used treats them as perfectly correlated, which cannot be too
    small. It must therefore exceed what an independence assumption would give.
    """
    estimate = estimate_for(0.2)
    result = trending({"a": estimate}, k=10, how="mean")
    tail = estimate.se[-10:]
    naive_independent = float(np.sqrt(np.sum((tail / 10) ** 2)))
    assert result["trend_se"].iloc[0] > naive_independent


def test_se_exact_flags_which_summaries_propagate_exactly():
    """max is not a linear functional, so its error bar is only indicative."""
    estimate = estimate_for(0.2)
    assert not trending({"a": estimate}, how="max")["se_exact"].iloc[0]
    assert trending({"a": estimate}, how="last")["se_exact"].iloc[0]


def test_sequence_input_gets_positional_ids():
    """A bare list is allowed; ids default to positions."""
    result = trending([estimate_for(0.5, 1), estimate_for(0.1, 2)])
    assert set(result["id"]) == {"0", "1"}


def test_supplied_ids_are_used():
    """Regression: the identifier you pass must be the identifier you get back."""
    result = trending([estimate_for(0.5), estimate_for(0.1)], ids=["alpha", "beta"])
    assert set(result["id"]) == {"alpha", "beta"}
    assert result["id"].iloc[0] == "alpha"


def test_mismatched_id_count_is_refused():
    """Silently zipping to the shorter list would mislabel every row."""
    with pytest.raises(ValueError, match="ids for"):
        trending([estimate_for(0.1)], ids=["a", "b"])


def test_window_longer_than_the_series_is_clamped():
    """Asking for more history than exists must not raise."""
    result = trending({"a": estimate_for(0.2)}, k=10_000)
    assert np.isfinite(result["trend"].iloc[0])


def test_confidence_level_reaches_the_interval():
    """A dispatch layer must forward confidence_level, not default it."""
    estimate = estimate_for(0.2)
    narrow = trending({"a": estimate}, confidence_level=0.5)
    wide = trending({"a": estimate}, confidence_level=0.99)
    narrow_width = narrow["ci_upper"].iloc[0] - narrow["ci_lower"].iloc[0]
    wide_width = wide["ci_upper"].iloc[0] - wide["ci_lower"].iloc[0]
    assert wide_width > narrow_width


def test_result_is_a_frame_sorted_by_rank():
    """Downstream code reads this top-down; the order is part of the contract."""
    result = trending({f"s{i}": estimate_for(0.05 * i, seed=i) for i in range(5)})
    assert isinstance(result, pd.DataFrame)
    assert list(result["rank"]) == sorted(result["rank"])
    assert result["trend"].is_monotonic_decreasing


def test_a_series_with_no_usable_derivative_ranks_last():
    """Regression: one NaN trend turned every rank into NaN.

    ``rankdata`` propagates NaN, so a single series whose smoother returned
    nothing usable erased the ranking for all of them -- the failure showed up
    as an unusable result rather than as one bad row.
    """
    good = {f"s{i}": estimate_for(0.05 * (i + 1), seed=i) for i in range(3)}
    broken = estimate_for(0.05, seed=9)
    broken = dataclasses.replace(broken, derivative=np.full(N, np.nan))
    result = trending({**good, "broken": broken})

    assert result["rank"].notna().all(), "a single NaN trend erased every rank"
    assert list(result["rank"]) == [1, 2, 3, 4]
    assert result["id"].iloc[-1] == "broken"
