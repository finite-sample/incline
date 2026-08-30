"""Monte Carlo validation of whole-pipeline seasonal uncertainty.

A bootstrap can return finite, plausible-looking intervals while using the
wrong dependence structure.  The positive control below generates known AR(1)
errors and asks for that same covariance model.  The negative control fits the
identical datasets while pretending the errors are independent.  Comparing the
mean reported standard error with the estimator's sampling standard deviation
separates variance calibration from smoothing bias and interval centering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from incline import AR1, IID, SavitzkyGolay
from incline.seasonal import trend_with_deseasonalization
from tests._statistics import (
    DEEP_REPS,
    FAST_REPS,
    MonteCarloResult,
    assert_se_calibrated,
)

N = 84
POINT = N // 2
PERIOD = 12
PHI = 0.7
STANDARD_DEVIATION = 0.5
SLOPE = 0.03
BOOTSTRAP_REPLICATES = 40

TIERS = [
    pytest.param(FAST_REPS, id="fast"),
    pytest.param(DEEP_REPS, id="deep", marks=pytest.mark.slow),
]


def _ar1_errors(rng: np.random.Generator) -> np.ndarray:
    """Draw stationary AR(1) errors at the calibration parameters."""
    errors = np.empty(N)
    errors[0] = rng.normal(scale=STANDARD_DEVIATION)
    innovation_scale = STANDARD_DEVIATION * np.sqrt(1 - PHI**2)
    for index in range(1, N):
        errors[index] = PHI * errors[index - 1] + rng.normal(scale=innovation_scale)
    return errors


@pytest.mark.parametrize("reps", TIERS)
def test_seasonal_ar1_standard_errors_are_calibrated(reps, capsys):
    """Correct covariance recovers sampling spread; IID is a negative control."""
    time = np.arange(N, dtype=float)
    mean = SLOPE * time + 2 * np.sin(2 * np.pi * time / PERIOD)
    estimates = np.empty(reps)
    ar1_errors = np.empty(reps)
    iid_errors = np.empty(reps)

    for replicate in range(reps):
        frame = pd.DataFrame(
            {
                "time": time,
                "value": mean + _ar1_errors(np.random.default_rng(5000 + replicate)),
            }
        )
        common = {
            "time_column": "time",
            "method": "stl",
            "period": PERIOD,
            "with_uncertainty": True,
            "n_bootstrap": BOOTSTRAP_REPLICATES,
            "random_state": 9000 + replicate,
        }
        ar1 = trend_with_deseasonalization(
            frame,
            SavitzkyGolay(window_length=21),
            noise=AR1(phi=PHI, standard_deviation=STANDARD_DEVIATION),
            **common,
        )
        iid = trend_with_deseasonalization(
            frame,
            SavitzkyGolay(window_length=21),
            noise=IID(standard_deviation=STANDARD_DEVIATION),
            **common,
        )

        estimates[replicate] = ar1["derivative_value"].iloc[POINT]
        ar1_errors[replicate] = ar1["derivative_standard_error"].iloc[POINT]
        iid_errors[replicate] = iid["derivative_standard_error"].iloc[POINT]
        assert ar1["derivative_value"].iloc[POINT] == pytest.approx(
            iid["derivative_value"].iloc[POINT]
        )

    sampling_spread = float(estimates.std(ddof=1))
    ar1_study = MonteCarloResult(
        estimates=estimates,
        standard_errors=ar1_errors,
        covered=None,
        rejected=None,
        truth=SLOPE,
    )
    iid_study = MonteCarloResult(
        estimates=estimates,
        standard_errors=iid_errors,
        covered=None,
        rejected=None,
        truth=SLOPE,
    )
    with capsys.disabled():
        print(
            f"  seasonal/ar1 se/sd={ar1_study.se_ratio:.3f} "
            f"misspecified-iid se/sd={iid_study.se_ratio:.3f} "
            f"sampling_sd={sampling_spread:.4f} reps={reps}"
        )

    assert_se_calibrated(ar1_study, "seasonal pipeline with stated AR(1) noise")
    with pytest.raises(AssertionError, match="reported standard error"):
        assert_se_calibrated(iid_study, "misspecified IID negative control")
