"""SiZer: which trend features survive being looked at from every scale.

A single bandwidth is a single opinion about what counts as signal. SiZer sweeps
the bandwidth and asks, at each scale and each point, whether the slope is
distinguishable from zero. Features that persist across many scales are real;
features that appear at one scale and vanish at the next are artifacts of that
scale.

SiZer sweeps :meth:`~incline.smoothers.Smoother.with_scale` and asks each fitted
smoother for its own uncertainty. The map therefore uses the same uncertainty
model as the underlying estimator rather than maintaining separate variance
formulas for each smoothing method.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from .axis import TimeAxis
from .smoothers import LocalPolynomial, Smoother

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .noise import NoiseModel

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
        standard_error: Standard errors, same shape.
        significance: -1, 0 or +1 per cell, same shape.
        confidence_level: Confidence level behind the flags.
        simultaneous: Whether each row's flags carry a whole-curve correction.
        smoother_name: Which estimator produced the sweep.
    """

    axis: TimeAxis
    scales: npt.NDArray[np.float64]
    derivative: npt.NDArray[np.float64]
    standard_error: npt.NDArray[np.float64]
    significance: npt.NDArray[np.int_]
    confidence_level: float
    simultaneous: bool
    smoother_name: str

    def __post_init__(self) -> None:
        """Validate the scale-by-position result arrays and metadata."""
        scales = np.asarray(self.scales, dtype=np.float64)
        if (
            scales.ndim != 1
            or scales.size == 0
            or not np.all(np.isfinite(scales))
            or np.any(scales <= 0)
            or np.any(scales > 1)
            or np.any(np.diff(scales) <= 0)
        ):
            raise ValueError(
                "scales must be a nonempty, strictly increasing vector in (0, 1]"
            )
        expected_shape = (len(scales), self.axis.n)
        derivative = np.asarray(self.derivative, dtype=np.float64)
        standard_error = np.asarray(self.standard_error, dtype=np.float64)
        significance = np.asarray(self.significance)
        for name, array in (
            ("derivative", derivative),
            ("standard_error", standard_error),
            ("significance", significance),
        ):
            if array.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {array.shape}"
                )
        if np.any(np.isinf(derivative)):
            raise ValueError("derivative cannot contain infinite values")
        if np.any(np.isinf(standard_error)) or np.any(
            standard_error[np.isfinite(standard_error)] < 0
        ):
            raise ValueError("standard_error must be nonnegative and not infinite")
        if not np.all(np.isin(significance, (DECREASING, INSIGNIFICANT, INCREASING))):
            raise ValueError("significance values must be -1, 0 or 1")
        if (
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
        if not isinstance(self.simultaneous, (bool, np.bool_)):
            raise ValueError("simultaneous must be boolean")
        if not isinstance(self.smoother_name, str) or not self.smoother_name:
            raise ValueError("smoother_name must be a nonempty string")
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "derivative", derivative)
        object.__setattr__(self, "standard_error", standard_error)
        object.__setattr__(self, "significance", significance.astype(int))

    def to_frame(self) -> pd.DataFrame:
        """Render the sweep as one row per (scale, point).

        Returns:
            Long-format frame with ``x``, ``scale``, ``derivative``,
            ``derivative_standard_error`` and ``significance``.
        """
        n_scales, n_points = self.derivative.shape
        return pd.DataFrame(
            {
                "x": np.tile(self.axis.x, n_scales),
                "scale": np.repeat(self.scales, n_points),
                "derivative": self.derivative.ravel(),
                "derivative_standard_error": self.standard_error.ravel(),
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

        Raises:
            ValueError: If ``min_persistence`` is not a positive integer.
        """
        if (
            isinstance(min_persistence, (bool, np.bool_))
            or not isinstance(min_persistence, (int, np.integer))
            or min_persistence < 1
        ):
            raise ValueError("min_persistence must be a positive integer")
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
            ValueError: If ``figsize`` is not a positive finite pair or
                ``title`` is not a string or None.
        """
        try:
            size = np.asarray(figsize, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("figsize must be a positive finite pair") from exc
        if size.shape != (2,) or not np.all(np.isfinite(size)) or np.any(size <= 0):
            raise ValueError("figsize must be a positive finite pair")
        if title is not None and not isinstance(title, str):
            raise ValueError("title must be a string or None")
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
    """A scale sweep over a smoother with an explicit scale knob.

    Attributes:
        smoother: The estimator to sweep. It must implement a real smoothing
            scale; naive differencing and state-space models do not. Its
            uncertainty machinery is what the map rests on, so prefer one with
            an exact operator.
        scales: Explicit scales. Generated log-spaced when None.
        n_scales: How many scales to generate.
        scale_range: Smallest and largest scale, as fractions of the span.
        confidence_level: Confidence level for the flags.
        noise: Noise model instance passed to the smoother, or ``'iid'``,
            ``'ar1'``, or ``'heteroskedastic'``.
        simultaneous: Ask for a whole-curve band at each scale rather than
            pointwise intervals. See the note on multiplicity below.
        n_bootstrap: Bootstrap replicates for nonlinear smoothers.
        random_state: Seed or Generator controlling every bootstrap in the sweep.

    Note:
        With ``simultaneous=True`` each *row* of the map is corrected for
        testing every x at that scale. It is not corrected jointly across
        scales as well, because neighboring scales are so strongly dependent
        that treating them as separate tests would be far too conservative.
        Reading a single cell in isolation therefore still overstates
        confidence; reading persistence across scales, which is what
        :meth:`SiZerMap.significant_regions` does, is the intended use.

        Only smoothers with an exact operator support the whole-curve band.
        Requesting one from another smoother raises rather than silently
        changing the inferential target.
    """

    smoother: Smoother = field(default_factory=lambda: LocalPolynomial(degree=2))
    scales: npt.NDArray[np.float64] | None = None
    n_scales: int = 20
    scale_range: tuple[float, float] = (0.02, 0.5)
    confidence_level: float = 0.95
    noise: NoiseModel | str | None = None
    simultaneous: bool = True
    n_bootstrap: int = 200
    random_state: int | np.random.Generator | None = None

    def __post_init__(self) -> None:
        """Validate the scale grid and uncertainty configuration."""
        if (
            isinstance(self.n_scales, (bool, np.bool_))
            or not isinstance(self.n_scales, (int, np.integer))
            or self.n_scales < 1
        ):
            raise ValueError("n_scales must be a positive integer")
        if (
            not isinstance(self.scale_range, tuple)
            or len(self.scale_range) != 2
            or not np.all(np.isfinite(self.scale_range))
            or not 0.0 < self.scale_range[0] < self.scale_range[1] <= 1.0
        ):
            raise ValueError("scale_range must be an increasing pair inside (0, 1]")
        if self.scales is not None:
            scales = np.asarray(self.scales)
            if (
                scales.ndim != 1
                or scales.size == 0
                or not np.issubdtype(scales.dtype, np.number)
                or not np.all(np.isfinite(scales))
                or np.any(scales <= 0)
                or np.any(scales > 1)
                or np.any(np.diff(scales) <= 0)
            ):
                raise ValueError(
                    "scales must be a nonempty, strictly increasing "
                    "one-dimensional sequence inside (0, 1]"
                )
        if (
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
        if (
            isinstance(self.n_bootstrap, (bool, np.bool_))
            or not isinstance(self.n_bootstrap, (int, np.integer))
            or self.n_bootstrap < 2
        ):
            raise ValueError("n_bootstrap must be an integer of at least 2")
        if not isinstance(self.simultaneous, (bool, np.bool_)):
            raise ValueError("simultaneous must be boolean")

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
        value_column: str = "value",
        time_column: str | None = None,
    ) -> SiZerMap:
        """Sweep the smoother across scales.

        Args:
            df: Time series data.
            value_column: Column holding the values.
            time_column: Numeric time column; the index is used when None.

        Returns:
            The assembled map.

        Raises:
            ValueError: If fewer than five usable observations remain or the
                smoother has no scale knob.
        """
        y = np.asarray(df[value_column], dtype=np.float64)
        axis = TimeAxis.from_frame(df, time_column)

        usable = np.isfinite(y)
        if int(np.sum(usable)) < 5:
            raise ValueError(
                f"SiZer needs at least 5 finite observations, got {int(np.sum(usable))}"
            )
        if not usable.all():
            # The count was checked and then the raw series handed on anyway,
            # so a few NaNs produced an all-NaN map that reads as "nothing is
            # trending" rather than "nothing could be computed".
            gaps = int((~usable).sum())
            raise ValueError(
                f"SiZer cannot sweep a series with {gaps} "
                f"missing value{'' if gaps == 1 else 's'}; "
                f"interpolate or drop {'it' if gaps == 1 else 'them'} first."
            )

        if self.smoother.has_native_posterior:
            if self.noise is not None:
                raise ValueError(
                    f"{self.smoother.name} carries its own posterior; "
                    "noise is unavailable"
                )
            if self.simultaneous:
                raise ValueError(
                    f"{self.smoother.name} has no simultaneous whole-curve "
                    "posterior band"
                )
        elif self.simultaneous and not self.smoother.is_linear:
            raise ValueError(
                f"{self.smoother.name} has no simultaneous whole-curve band; "
                "set simultaneous=False"
            )

        scales = self._scale_grid(axis.n)
        derivative = np.full((len(scales), axis.n), np.nan)
        standard_error = np.full((len(scales), axis.n), np.nan)
        significance = np.zeros((len(scales), axis.n), dtype=int)
        rng = np.random.default_rng(self.random_state)

        for row, scale in enumerate(scales):
            smoother = self.smoother.with_scale(float(scale), axis)
            try:
                estimate = smoother.fit(
                    axis,
                    y,
                    derivative_order=1,
                    with_uncertainty=True,
                    noise=self.noise,
                    simultaneous=self.simultaneous,
                    confidence_level=self.confidence_level,
                    n_bootstrap=self.n_bootstrap,
                    random_state=rng,
                )
            except Exception as exc:  # a bad scale must not lose the sweep
                warnings.warn(
                    f"scale {scale:.3g} failed ({exc}); leaving that row blank.",
                    stacklevel=2,
                )
                continue

            derivative[row] = estimate.derivative
            if estimate.standard_error is not None:
                standard_error[row] = estimate.standard_error
            # The smoother already decided what excludes zero, at whichever
            # band width was asked for. Sign it and store.
            flagged = estimate.significant
            significance[row] = np.where(
                flagged, np.sign(estimate.derivative).astype(int), INSIGNIFICANT
            )
        return SiZerMap(
            axis=axis,
            scales=scales,
            derivative=derivative,
            standard_error=standard_error,
            significance=significance,
            confidence_level=self.confidence_level,
            simultaneous=self.simultaneous,
            smoother_name=self.smoother.name,
        )


def sizer_analysis(
    df: pd.DataFrame,
    value_column: str = "value",
    time_column: str | None = None,
    smoother: Smoother | None = None,
    **kwargs: Any,
) -> SiZerMap:
    """Run a scale sweep in one call.

    Args:
        df: Time series data.
        value_column: Column holding the values.
        time_column: Numeric time column.
        smoother: Estimator to sweep; local polynomial by default.
        **kwargs: Passed to :class:`SiZer`.

    Returns:
        The assembled map.
    """
    config = SiZer(smoother=smoother or LocalPolynomial(degree=2), **kwargs)
    return config.fit(df, value_column, time_column)


def trend_with_sizer(
    df: pd.DataFrame,
    value_column: str = "value",
    time_column: str | None = None,
    smoother: Smoother | None = None,
    min_persistence: int = 3,
    *,
    scales: npt.NDArray[np.float64] | None = None,
    n_scales: int = 20,
    scale_range: tuple[float, float] = (0.02, 0.5),
    confidence_level: float = 0.95,
    noise: NoiseModel | str | None = None,
    simultaneous: bool = True,
    n_bootstrap: int = 200,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Estimate a trend and mark where it survives a scale sweep.

    Args:
        df: Time series data.
        value_column: Column holding the values.
        time_column: Numeric time column.
        smoother: Estimator used for both the trend and the sweep.
        min_persistence: Consecutive scales required to call a feature real.
        scales: Explicit smoothing scales, or None to generate them.
        n_scales: Number of generated scales.
        scale_range: Lower and upper generated scales.
        confidence_level: Confidence level for both the trend and the map.
        noise: Noise model used by both the trend and the map.
        simultaneous: Whether both requests use whole-curve bands.
        n_bootstrap: Bootstrap replicates for nonlinear smoothers.
        random_state: Seed or Generator controlling all bootstrap draws.

    Returns:
        The estimator's usual columns plus ``sizer_significance``,
        ``persistent_increasing`` and ``persistent_decreasing``.
    """
    from .api import estimate

    chosen = smoother or LocalPolynomial(degree=2)
    result = estimate(
        chosen,
        df,
        value_column,
        time_column,
        with_uncertainty=True,
        noise=noise,
        simultaneous=simultaneous,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    ).to_frame(df)

    sizer_map = SiZer(
        smoother=chosen,
        scales=scales,
        n_scales=n_scales,
        scale_range=scale_range,
        confidence_level=confidence_level,
        noise=noise,
        simultaneous=simultaneous,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    ).fit(df, value_column, time_column)
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
