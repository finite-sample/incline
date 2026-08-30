"""Structured trend estimates and their pandas representation.

:class:`TrendEstimate` keeps point estimates, uncertainty, provenance, and the
original index together. :meth:`TrendEstimate.to_frame` derives the stable
tabular schema used by the convenience API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    from .axis import TimeAxis

# Columns every estimate produces, in the derivative_order they appear in the frame.
CORE_COLUMNS = (
    "smoothed_value",
    "derivative_value",
    "derivative_method",
    "derivative_order",
    "derivative_standard_error",
    "derivative_ci_lower",
    "derivative_ci_upper",
    "uncertainty_method",
    "confidence_level",
    "simultaneous",
    "bias_corrected",
    "noise_model",
    "significant_trend",
)


@dataclass(frozen=True)
class Provenance:
    """How an estimate was produced.

    Recorded so that a result can be interpreted without reconstructing the
    call that made it -- particularly which uncertainty strategy ran, since
    that determines what the interval means.

    Attributes:
        method: Smoother name, e.g. ``'sgolay'``.
        uncertainty_method: How uncertainty was obtained: ``'operator'``,
            ``'bootstrap'``, ``'native'``, or None when no standard error
            was computed.
        noise: Description of the noise model.
        bias_corrected: Whether a pilot-fit bias correction was applied.
        simultaneous: Whether the interval is a simultaneous band rather than
            a pointwise interval.
        params: Smoother-specific settings, emitted as extra frame columns.
    """

    method: str
    uncertainty_method: str | None = None
    noise: str | None = None
    bias_corrected: bool = False
    simultaneous: bool = False
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrendEstimate:
    """A smoothed series and the derivative of that smooth, with uncertainty.

    Attributes:
        axis: The time axis the estimate lives on.
        values: The smoothed series.
        derivative: The derivative of the smooth, per unit of ``axis.x``.
        derivative_order: Which derivative ``derivative`` holds.
        provenance: How the estimate was produced.
        standard_error: Standard error of ``derivative``, or None when unavailable.
        ci_lower: Lower interval bound, or None.
        ci_upper: Upper interval bound, or None.
        confidence_level: Confidence level of the interval, or None when no
            interval exists.
        index: Original pandas index, preserved for ``to_frame``.
    """

    axis: TimeAxis
    values: npt.NDArray[np.float64]
    derivative: npt.NDArray[np.float64]
    derivative_order: int
    provenance: Provenance
    standard_error: npt.NDArray[np.float64] | None = None
    ci_lower: npt.NDArray[np.float64] | None = None
    ci_upper: npt.NDArray[np.float64] | None = None
    confidence_level: float | None = None
    index: pd.Index | None = None

    def __post_init__(self) -> None:
        """Validate result shapes, intervals, and uncertainty metadata."""
        n = self.axis.n
        for name in (
            "values",
            "derivative",
            "standard_error",
            "ci_lower",
            "ci_upper",
        ):
            arr = getattr(self, name)
            if arr is None:
                continue
            array = np.asarray(arr, dtype=np.float64)
            if array.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            if len(array) != n:
                raise ValueError(
                    f"{name} has length {len(array)} but the axis has {n} points"
                )
            object.__setattr__(self, name, array)
        if (
            isinstance(self.derivative_order, (bool, np.bool_))
            or not isinstance(self.derivative_order, (int, np.integer))
            or self.derivative_order < 0
        ):
            raise ValueError("derivative_order must be a nonnegative integer")
        if self.standard_error is not None and (
            np.any(np.isinf(self.standard_error))
            or np.any(self.standard_error[np.isfinite(self.standard_error)] < 0)
        ):
            raise ValueError("standard_error must be nonnegative and not infinite")
        if (self.ci_lower is None) != (self.ci_upper is None):
            raise ValueError("ci_lower and ci_upper must be supplied together")
        if (
            self.ci_lower is not None
            and self.ci_upper is not None
            and np.any(self.ci_lower > self.ci_upper)
        ):
            raise ValueError("ci_lower cannot exceed ci_upper")
        has_interval = self.ci_lower is not None
        if self.confidence_level is None:
            if has_interval:
                raise ValueError("an interval requires confidence_level")
        elif (
            isinstance(self.confidence_level, (bool, np.bool_))
            or not isinstance(
                self.confidence_level, (int, float, np.integer, np.floating)
            )
            or not np.isfinite(self.confidence_level)
            or not 0.0 < self.confidence_level < 1.0
        ):
            raise ValueError(
                "confidence_level must be finite and strictly between 0 and 1"
            )
        elif not has_interval:
            raise ValueError("confidence_level requires interval bounds")
        if self.index is not None and len(self.index) != n:
            raise ValueError(
                f"index has length {len(self.index)} but the axis has {n} points"
            )

    @property
    def has_uncertainty(self) -> bool:
        """Whether this estimate carries a standard error."""
        return self.standard_error is not None

    @property
    def significant(self) -> npt.NDArray[np.bool_]:
        """Where the interval excludes zero.

        All False when no interval was computed -- absence of evidence is not
        reported as a significant trend.

        A zero standard error also yields False. That case arises when the
        noise level is estimated as exactly zero, which means there is no
        evidence about the noise rather than evidence of no noise. The interval
        then collapses onto the point estimate, and without this guard a
        derivative of 3e-17 -- floating-point dust on a flat line -- excludes
        zero and gets reported as a significant trend at every point.
        """
        if self.ci_lower is None or self.ci_upper is None:
            return np.zeros(self.axis.n, dtype=bool)
        usable = np.isfinite(self.ci_lower) & np.isfinite(self.ci_upper)
        if self.standard_error is not None:
            usable &= np.isfinite(self.standard_error) & (self.standard_error > 0)
        return np.asarray(
            usable & ((self.ci_lower > 0) | (self.ci_upper < 0)), dtype=bool
        )

    def to_frame(self, source: pd.DataFrame | None = None) -> pd.DataFrame:
        """Render the estimate as a DataFrame.

        Args:
            source: The original input frame. When given, its columns are
                preserved and the estimate's columns appended.

        Returns:
            A frame carrying every column in :data:`CORE_COLUMNS`, plus the
            smoother's own parameters.

        Raises:
            ValueError: If a smoother parameter uses the name of a core result
                column.
        """
        nan = np.full(self.axis.n, np.nan)
        columns: dict[str, Any] = {
            "smoothed_value": self.values,
            "derivative_value": self.derivative,
            "derivative_method": self.provenance.method,
            "derivative_order": self.derivative_order,
            "derivative_standard_error": self.standard_error
            if self.standard_error is not None
            else nan,
            "derivative_ci_lower": (
                self.ci_lower if self.ci_lower is not None else nan
            ),
            "derivative_ci_upper": (
                self.ci_upper if self.ci_upper is not None else nan
            ),
            "uncertainty_method": self.provenance.uncertainty_method,
            "confidence_level": (
                self.confidence_level if self.confidence_level is not None else np.nan
            ),
            "simultaneous": self.provenance.simultaneous,
            "bias_corrected": self.provenance.bias_corrected,
            "noise_model": self.provenance.noise,
            "significant_trend": self.significant,
        }
        collisions = set(columns).intersection(self.provenance.params)
        if collisions:
            names = ", ".join(sorted(collisions))
            raise ValueError(
                f"provenance params cannot overwrite core columns: {names}"
            )
        columns.update(self.provenance.params)

        if source is not None:
            odf = source.copy()
            for name, value in columns.items():
                odf[name] = value
            return odf

        return pd.DataFrame(
            columns,
            index=self.index if self.index is not None else pd.RangeIndex(self.axis.n),
        )

    def with_uncertainty(
        self,
        standard_error: npt.NDArray[np.float64] | None,
        uncertainty_method: str | None,
        ci_lower: npt.NDArray[np.float64] | None = None,
        ci_upper: npt.NDArray[np.float64] | None = None,
        confidence_level: float = 0.95,
        noise: str | None = None,
        simultaneous: bool = False,
    ) -> TrendEstimate:
        """Return a copy carrying uncertainty.

        When ``ci_lower``/``ci_upper`` are omitted a normal-theory interval is
        built from ``standard_error``. Bootstrap percentile intervals are not symmetric
        about the point estimate, so those are passed explicitly.

        Args:
            standard_error: Standard errors, or None.
            uncertainty_method: Label recorded in provenance.
            ci_lower: Explicit lower bounds.
            ci_upper: Explicit upper bounds.
            confidence_level: Confidence level for a normal-theory interval.
            noise: Description of the noise model used.
            simultaneous: Whether the band is simultaneous.

        Returns:
            A new TrendEstimate.

        Raises:
            ValueError: If exactly one explicit confidence bound is supplied.
        """
        from dataclasses import replace

        from scipy.stats import norm

        if (ci_lower is None) != (ci_upper is None):
            raise ValueError("ci_lower and ci_upper must be supplied together")
        if standard_error is not None and ci_lower is None:
            z = float(norm.ppf(1 - (1 - confidence_level) / 2))
            ci_lower = self.derivative - z * standard_error
            ci_upper = self.derivative + z * standard_error

        return replace(
            self,
            standard_error=standard_error,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
            provenance=replace(
                self.provenance,
                uncertainty_method=uncertainty_method,
                noise=noise,
                simultaneous=simultaneous,
            ),
        )
