"""The time axis of a series.

Every estimator needs the same three things from a time index: numeric
positions, a representative step size, and whether the sampling is regular.
Before this module each estimator derived them itself, at fifteen call sites,
with enough drift between them that ``local_polynomial_trend`` crashed on any
``DatetimeIndex`` -- the derivation returned a pandas ``Index`` rather than an
ndarray and nothing downstream expected that.

:class:`TimeAxis` is that derivation, done once and validated once.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    from typing import Self

# Coefficient of variation of the spacing above which sampling is "irregular".
# Methods built on a uniform grid (Savitzky-Golay) are wrong beyond this.
REGULARITY_TOLERANCE = 0.05


@dataclass(frozen=True)
class TimeAxis:
    """Numeric time positions for a series.

    Attributes:
        x: Strictly increasing positions. Days from the start for a
            ``DatetimeIndex``, otherwise the index values themselves.
        delta: Median spacing between consecutive positions. Derivatives are
            reported per unit of ``x``, so this is the scale factor between
            "per observation" and "per unit time".
        unit: ``'days'`` when derived from datetimes, else ``'index'``.
    """

    x: npt.NDArray[np.float64]
    delta: float
    unit: str

    @classmethod
    def from_index(cls, index: pd.Index) -> Self:
        """Build an axis from a pandas index.

        Args:
            index: A ``DatetimeIndex`` or ``PeriodIndex`` (converted to days
                from the start) or any numeric index.

        Returns:
            The corresponding TimeAxis.

        Raises:
            ValueError: If the index carries no time information.
        """
        if isinstance(index, pd.PeriodIndex):
            # Monthly and quarterly series are ordinarily indexed this way; the
            # period's start is the point the observation belongs to.
            index = index.to_timestamp()

        if isinstance(index, pd.DatetimeIndex):
            values = np.asarray(
                (index - index[0]).total_seconds() / 86400.0, dtype=np.float64
            )
            unit = "days"
        else:
            try:
                values = np.asarray(index, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                # Say what to do. Reaching numpy with a string index otherwise
                # surfaces "could not convert string to float: 'r0'", which
                # names neither the cause nor the remedy.
                raise ValueError(
                    f"Cannot read a time axis from a {type(index).__name__} of "
                    f"dtype {index.dtype}. Use a DatetimeIndex, a PeriodIndex "
                    f"or a numeric index, or pass time_column= naming a "
                    f"numeric column."
                ) from exc
            unit = "index"
        return cls._build(values, unit)

    @classmethod
    def from_frame(cls, df: pd.DataFrame, time_column: str | None = None) -> Self:
        """Build an axis from a DataFrame, preferring an explicit time column.

        Args:
            df: The series being estimated.
            time_column: Numeric time column. When None the index is used.

        Returns:
            The corresponding TimeAxis.
        """
        if time_column is not None:
            values = np.asarray(df[time_column], dtype=np.float64)
            return cls._build(values, "index")
        return cls.from_index(df.index)

    @classmethod
    def positional(cls, n: int) -> Self:
        """Build a unit-spaced axis of length ``n``.

        Args:
            n: Number of observations.

        Returns:
            A TimeAxis over ``0, 1, ..., n-1``.
        """
        return cls._build(np.arange(n, dtype=np.float64), "index")

    @classmethod
    def _build(cls, values: npt.NDArray[np.float64], unit: str) -> Self:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(f"Time axis must be 1-D, got shape {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("Time axis contains non-finite values")
        if len(values) > 1:
            steps = np.diff(values)
            if np.any(steps <= 0):
                raise ValueError(
                    "Time axis must be strictly increasing; sort the series first"
                )
            delta = float(np.median(steps))
        else:
            delta = 1.0
        return cls(x=values, delta=delta, unit=unit)

    @property
    def n(self) -> int:
        """Number of observations."""
        return len(self.x)

    def key(self) -> tuple[str, int, bytes]:
        """A hashable identity for this axis.

        Used to cache linear operators, which depend on the axis and the
        smoother's settings but never on the observed values.
        """
        return (self.unit, self.n, self.x.tobytes())

    @property
    def span(self) -> float:
        """Distance from the first to the last observation."""
        return float(self.x[-1] - self.x[0]) if self.n > 1 else 0.0

    @property
    def spacing_cv(self) -> float:
        """Coefficient of variation of the spacing. Zero for a uniform grid."""
        if self.n < 3:
            return 0.0
        steps = np.diff(self.x)
        mean_step = float(np.mean(steps))
        if mean_step <= 0:
            return 0.0
        return float(np.std(steps) / mean_step)

    @property
    def is_regular(self) -> bool:
        """Whether the sampling is uniform enough for grid-based methods."""
        return self.spacing_cv <= REGULARITY_TOLERANCE

    def require_regular(self, method: str) -> None:
        """Warn when a grid-based method is used on irregular sampling.

        Args:
            method: Name of the method, used in the warning message.
        """
        if not self.is_regular:
            warnings.warn(
                f"{method} assumes uniform sampling but the spacing varies "
                f"(cv={self.spacing_cv:.2f}). Prefer spline or local "
                f"polynomial methods for irregular series.",
                stacklevel=3,
            )
