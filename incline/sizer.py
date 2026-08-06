"""SiZer: which trend features survive being looked at from every scale.

A single bandwidth is a single opinion about what counts as signal. SiZer sweeps
the bandwidth and asks, at each scale and each point, whether the slope is
distinguishable from zero. Features that persist across many scales are real;
features that appear at one scale and vanish at the next are artifacts of that
scale.

This module does not compute standard errors, and that is the point of it.

The previous implementation carried four separate variance formulas, one per
smoothing method, and its own module docstring documented three of them as
broken: LOESS standard errors 2-23x too large (so it flagged nothing), spline
standard errors roughly 10x too small *and increasing with bandwidth*, which is
backwards (so it flagged about 90% of pure noise as trending), and a Gaussian
process branch that back-solved a standard error out of an interval width. Only
the local-polynomial sandwich was calibrated.

None of them are fixed here; all of them are gone. SiZer now sweeps
:meth:`~incline.smoothers.Smoother.with_scale` over any smoother and asks that
smoother for its uncertainty, so the significance map is exactly as trustworthy
as the estimator underneath it -- and every one of those has been measured
against Monte Carlo in ``tests/test_econometrics.py``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from .axis import TimeAxis
from .noise import NoiseModel
from .smoothers import LocalPolynomial, Smoother


if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Significance codes stored in the map.
DECREASING = -1
INSIGNIFICANT = 0
INCREASING = 1


@dataclass(frozen=True)
class SiZerMap:
    """The result of a scale sweep.

    Attributes:
        axis: The time axis every scale shares.
        scales: Smoothing scales, ascending, as fractions of the series span.
        derivative: Slope estimates, shape (n_scales, n_points).
        se: Standard errors, same shape.
        significance: -1, 0 or +1 per cell, same shape.
        confidence_level: Confidence level behind the flags.
        simultaneous: Whether each row's flags carry a whole-curve correction.
        smoother_name: Which estimator produced the sweep.
    """

    axis: TimeAxis
    scales: npt.NDArray[np.float64]
    derivative: npt.NDArray[np.float64]
    se: npt.NDArray[np.float64]
    significance: npt.NDArray[np.int_]
    confidence_level: float
    simultaneous: bool
    smoother_name: str

    def to_frame(self) -> pd.DataFrame:
        """Render the sweep as one row per (scale, point).

        Returns:
            Long-format frame with ``x``, ``scale``, ``derivative``,
            ``derivative_se`` and ``significance``.
        """
        n_scales, n_points = self.derivative.shape
        return pd.DataFrame(
            {
                "x": np.tile(self.axis.x, n_scales),
                "scale": np.repeat(self.scales, n_points),
                "derivative": self.derivative.ravel(),
                "derivative_se": self.se.ravel(),
                "significance": self.significance.ravel(),
            }
        )

    def significant_regions(
        self, min_persistence: int = 3
    ) -> dict[str, list[tuple[float, float]]]:
        """Find x-intervals whose sign holds across several consecutive scales.

        Persistence across scales is the whole idea: a feature visible only at
        one bandwidth is a property of that bandwidth.

        Args:
            min_persistence: How many consecutive scales must agree.

        Returns:
            ``{'increasing': [(start, end), ...], 'decreasing': [...]}``.
        """
        regions: dict[str, list[tuple[float, float]]] = {
            "increasing": [],
            "decreasing": [],
        }
        for label, code in (("increasing", INCREASING), ("decreasing", DECREASING)):
            # A column persists if any run of matching cells down the scale
            # axis is at least min_persistence long.
            matches = self.significance == code
            persistent = np.array(
                [
                    _longest_run(matches[:, j]) >= min_persistence
                    for j in range(matches.shape[1])
                ]
            )
            regions[label] = _contiguous_spans(self.axis.x, persistent)
        return regions

    def plot(
        self,
        figsize: tuple[float, float] = (12, 8),
        title: str | None = None,
    ) -> Figure:
        """Draw the significance map, scale against position.

        Args:
            figsize: Figure size in inches.
            title: Overrides the default title.

        Returns:
            The matplotlib figure.

        Raises:
            ImportError: If matplotlib is unavailable.
        """
        try:
            import matplotlib.colors as mcolors
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - matplotlib is a hard dep
            raise ImportError("matplotlib is required to plot a SiZer map") from exc

        figure, axes = plt.subplots(figsize=figsize)
        colormap = mcolors.ListedColormap(["#2a78d6", "#f0efec", "#d03b3b"])
        norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], colormap.N)

        image = axes.imshow(
            self.significance,
            aspect="auto",
            origin="lower",
            cmap=colormap,
            norm=norm,
            extent=(
                float(self.axis.x[0]),
                float(self.axis.x[-1]),
                float(np.log10(self.scales[0])),
                float(np.log10(self.scales[-1])),
            ),
        )
        axes.set_xlabel("x")
        axes.set_ylabel("smoothing scale (fraction of span)")

        ticks = [
            t
            for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
            if self.scales[0] <= t <= self.scales[-1]
        ]
        if ticks:
            axes.set_yticks([np.log10(t) for t in ticks])
            axes.set_yticklabels([f"{t:.2f}" for t in ticks])

        bar = plt.colorbar(image, ax=axes, ticks=[-1, 0, 1])
        bar.set_ticklabels(["Decreasing", "Not distinguishable", "Increasing"])

        band = "whole-curve" if self.simultaneous else "pointwise"
        axes.set_title(
            title or f"SiZer, {self.smoother_name} ({self.confidence_level:.0%} {band})"
        )
        figure.tight_layout()
        return figure


def _longest_run(flags: npt.NDArray[np.bool_]) -> int:
    """Length of the longest run of True in a boolean vector."""
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def _contiguous_spans(
    x: npt.NDArray[np.float64], flags: npt.NDArray[np.bool_]
) -> list[tuple[float, float]]:
    """Merge flagged positions into (start, end) intervals."""
    spans: list[tuple[float, float]] = []
    start: int | None = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append((float(x[start]), float(x[i - 1])))
            start = None
    if start is not None:
        spans.append((float(x[start]), float(x[-1])))
    return spans


@dataclass(frozen=True)
class SiZer:
    """A scale sweep over any smoother.

    Attributes:
        smoother: The estimator to sweep. Its uncertainty machinery is what the
            map rests on, so prefer one with an exact operator.
        scales: Explicit scales. Generated log-spaced when None.
        n_scales: How many scales to generate.
        scale_range: Smallest and largest scale, as fractions of the span.
        confidence_level: Confidence level for the flags.
        noise: Noise model passed to the smoother, or ``'iid'`` / ``'ar1'``.
        simultaneous: Ask for a whole-curve band at each scale rather than
            pointwise intervals. See the note on multiplicity below.

    Note:
        With ``simultaneous=True`` each *row* of the map is corrected for
        testing every x at that scale. It is not corrected jointly across
        scales as well, because neighboring scales are so strongly dependent
        that treating them as separate tests would be far too conservative.
        Reading a single cell in isolation therefore still overstates
        confidence; reading persistence across scales, which is what
        :meth:`SiZerMap.significant_regions` does, is the intended use.

        Only smoothers with an exact operator support the whole-curve band. For
        a bootstrapped smoother the request is dropped and the map records
        ``simultaneous=False`` rather than claiming a correction it did not make.
    """

    smoother: Smoother = field(default_factory=lambda: LocalPolynomial(degree=2))
    scales: npt.NDArray[np.float64] | None = None
    n_scales: int = 20
    scale_range: tuple[float, float] = (0.02, 0.5)
    confidence_level: float = 0.95
    noise: NoiseModel | str | None = None
    simultaneous: bool = True

    def _scale_grid(self, n_points: int) -> npt.NDArray[np.float64]:
        """Log-spaced scales, clipped to what the sample can support."""
        if self.scales is not None:
            return np.asarray(self.scales, dtype=np.float64)
        low, high = self.scale_range
        # Below three points a local fit has nothing to fit.
        low = max(low, 3.0 / n_points)
        high = min(high, 0.9)
        if low >= high:
            low, high = high / 2, high
        return np.logspace(np.log10(low), np.log10(high), self.n_scales)

    def fit(
        self,
        df: pd.DataFrame,
        column_value: str = "value",
        time_column: str | None = None,
    ) -> SiZerMap:
        """Sweep the smoother across scales.

        Args:
            df: Time series data.
            column_value: Column holding the values.
            time_column: Numeric time column; the index is used when None.

        Returns:
            The assembled map.

        Raises:
            ValueError: If fewer than five usable observations remain.
        """
        y = np.asarray(df[column_value], dtype=np.float64)
        axis = TimeAxis.from_frame(df, time_column)

        usable = np.isfinite(y)
        if int(np.sum(usable)) < 5:
            raise ValueError(
                f"SiZer needs at least 5 finite observations, got {int(np.sum(usable))}"
            )

        scales = self._scale_grid(axis.n)
        derivative = np.full((len(scales), axis.n), np.nan)
        standard_error = np.full((len(scales), axis.n), np.nan)
        significance = np.zeros((len(scales), axis.n), dtype=int)
        achieved_simultaneous = self.simultaneous

        for row, scale in enumerate(scales):
            smoother = self.smoother.with_scale(float(scale), axis)
            try:
                estimate = smoother.fit(
                    axis,
                    y,
                    order=1,
                    se=True,
                    noise=self.noise,
                    simultaneous=self.simultaneous,
                    confidence_level=self.confidence_level,
                )
            except Exception as exc:  # a bad scale must not lose the sweep
                warnings.warn(
                    f"scale {scale:.3g} failed ({exc}); leaving that row blank.",
                    stacklevel=2,
                )
                continue

            derivative[row] = estimate.derivative
            if estimate.se is not None:
                standard_error[row] = estimate.se
            # The smoother already decided what excludes zero, at whichever
            # band width was asked for. Sign it and store.
            flagged = estimate.significant
            significance[row] = np.where(
                flagged, np.sign(estimate.derivative).astype(int), INSIGNIFICANT
            )
            if self.simultaneous and not estimate.provenance.simultaneous:
                achieved_simultaneous = False

        if self.simultaneous and not achieved_simultaneous:
            warnings.warn(
                f"{self.smoother.name} has no exact operator, so a whole-curve "
                f"band is unavailable; the map is pointwise.",
                stacklevel=2,
            )

        return SiZerMap(
            axis=axis,
            scales=scales,
            derivative=derivative,
            se=standard_error,
            significance=significance,
            confidence_level=self.confidence_level,
            simultaneous=achieved_simultaneous,
            smoother_name=self.smoother.name,
        )


def sizer_analysis(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    smoother: Smoother | None = None,
    **kwargs: Any,
) -> SiZerMap:
    """Run a scale sweep in one call.

    Args:
        df: Time series data.
        column_value: Column holding the values.
        time_column: Numeric time column.
        smoother: Estimator to sweep; local polynomial by default.
        **kwargs: Passed to :class:`SiZer`.

    Returns:
        The assembled map.
    """
    config = SiZer(smoother=smoother or LocalPolynomial(degree=2), **kwargs)
    return config.fit(df, column_value, time_column)


def trend_with_sizer(
    df: pd.DataFrame,
    column_value: str = "value",
    time_column: str | None = None,
    smoother: Smoother | None = None,
    min_persistence: int = 3,
    **kwargs: Any,
) -> pd.DataFrame:
    """Estimate a trend and mark where it survives a scale sweep.

    Args:
        df: Time series data.
        column_value: Column holding the values.
        time_column: Numeric time column.
        smoother: Estimator used for both the trend and the sweep.
        min_persistence: Consecutive scales required to call a feature real.
        **kwargs: Passed to :class:`SiZer`.

    Returns:
        The estimator's usual columns plus ``sizer_significance``,
        ``persistent_increasing`` and ``persistent_decreasing``.
    """
    from .api import estimate

    chosen = smoother or LocalPolynomial(degree=2)
    result = estimate(chosen, df, column_value, time_column, se=True).to_frame(df)

    sizer_map = SiZer(smoother=chosen, **kwargs).fit(df, column_value, time_column)
    middle = len(sizer_map.scales) // 2
    result["sizer_significance"] = sizer_map.significance[middle]

    regions = sizer_map.significant_regions(min_persistence)
    x = sizer_map.axis.x
    for label in ("increasing", "decreasing"):
        flags = np.zeros(len(x), dtype=bool)
        for start, end in regions[label]:
            flags |= (x >= start) & (x <= end)
        result[f"persistent_{label}"] = flags

    return result
