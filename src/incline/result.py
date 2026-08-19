"""The result of a trend estimation.

Previously this was a naming convention: estimators appended columns named
``derivative_value``, ``smoothed_value`` and so on to a copy of the input frame,
and every consumer agreed to look for them. Nothing enforced the agreement, so
``naive_trend`` never produced ``smoothed_value`` at all, ``trending`` emitted a
different schema when it matched nothing, and -- most costly -- the ranking layer
could not use standard errors because there was no type in which to carry them.

:class:`TrendEstimate` is that type. ``.to_frame()`` still produces the pandas
view, because that is what users want to work with; the difference is that the
view is derived from a structure rather than being the structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    from .axis import TimeAxis

# Columns every estimate produces, in the order they appear in the frame.
CORE_COLUMNS = (
    "smoothed_value",
    "derivative_value",
    "derivative_method",
    "derivative_order",
    "derivative_se",
    "derivative_ci_lower",
    "derivative_ci_upper",
    "se_method",
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
        se_method: How uncertainty was obtained: ``'operator'`` (exact, from
            the smoother's linear operator), ``'bootstrap'``, ``'gp_analytic'``,
            ``'kalman'``, or None when no standard error was computed.
        noise: Description of the noise model, e.g. ``'iid(sigma=0.31)'``.
        bias_corrected: Whether a pilot-fit bias correction was applied.
        simultaneous: Whether the interval is a simultaneous band rather than
            a pointwise interval.
        params: Smoother-specific settings, emitted as extra frame columns.
    """

    method: str
    se_method: str | None = None
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
        order: Which derivative ``derivative`` holds.
        provenance: How the estimate was produced.
        se: Standard error of ``derivative``, or None when unavailable.
        ci_lower: Lower interval bound, or None.
        ci_upper: Upper interval bound, or None.
        confidence_level: Confidence level of the interval.
        index: Original pandas index, preserved for ``to_frame``.
    """

    axis: TimeAxis
    values: npt.NDArray[np.float64]
    derivative: npt.NDArray[np.float64]
    order: int
    provenance: Provenance
    se: npt.NDArray[np.float64] | None = None
    ci_lower: npt.NDArray[np.float64] | None = None
    ci_upper: npt.NDArray[np.float64] | None = None
    confidence_level: float = 0.95
    index: pd.Index | None = None

    def __post_init__(self) -> None:
        """Check that every array matches the axis length."""
        n = self.axis.n
        for name in ("values", "derivative", "se", "ci_lower", "ci_upper"):
            arr = getattr(self, name)
            if arr is not None and len(arr) != n:
                raise ValueError(
                    f"{name} has length {len(arr)} but the axis has {n} points"
                )

    @property
    def has_uncertainty(self) -> bool:
        """Whether this estimate carries a standard error."""
        return self.se is not None

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
        if self.se is not None:
            usable &= np.isfinite(self.se) & (self.se > 0)
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
        """
        nan = np.full(self.axis.n, np.nan)
        columns: dict[str, Any] = {
            "smoothed_value": self.values,
            "derivative_value": self.derivative,
            "derivative_method": self.provenance.method,
            "derivative_order": self.order,
            "derivative_se": self.se if self.se is not None else nan,
            "derivative_ci_lower": (
                self.ci_lower if self.ci_lower is not None else nan
            ),
            "derivative_ci_upper": (
                self.ci_upper if self.ci_upper is not None else nan
            ),
            "se_method": self.provenance.se_method,
            "significant_trend": self.significant,
        }
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
        se: npt.NDArray[np.float64] | None,
        se_method: str | None,
        ci_lower: npt.NDArray[np.float64] | None = None,
        ci_upper: npt.NDArray[np.float64] | None = None,
        confidence_level: float = 0.95,
        noise: str | None = None,
        simultaneous: bool = False,
    ) -> TrendEstimate:
        """Return a copy carrying uncertainty.

        When ``ci_lower``/``ci_upper`` are omitted a normal-theory interval is
        built from ``se``. Bootstrap percentile intervals are not symmetric
        about the point estimate, so those are passed explicitly.

        Args:
            se: Standard errors, or None.
            se_method: Label recorded in provenance.
            ci_lower: Explicit lower bounds.
            ci_upper: Explicit upper bounds.
            confidence_level: Confidence level for a normal-theory interval.
            noise: Description of the noise model used.
            simultaneous: Whether the band is simultaneous.

        Returns:
            A new TrendEstimate.
        """
        from dataclasses import replace

        from scipy.stats import norm

        if se is not None and (ci_lower is None or ci_upper is None):
            z = float(norm.ppf(1 - (1 - confidence_level) / 2))
            ci_lower = self.derivative - z * se
            ci_upper = self.derivative + z * se

        return replace(
            self,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
            provenance=replace(
                self.provenance,
                se_method=se_method,
                noise=noise,
                simultaneous=simultaneous,
            ),
        )
