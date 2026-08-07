"""Precompute every panel the uncertainty explorer can show.

The explorer is a static page: no server, no Python in the browser. So every
combination of series, method, smoothing scale and noise model is fitted here
and written out as JSON, which the page then only has to draw.

Run from the repository root::

    uv run --frozen python docs/_static/build_explorer_data.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from incline import smoothers as sm
from incline.axis import TimeAxis
from incline.noise import AR1, IID
from incline.process import GaussianProcess


warnings.filterwarnings("ignore")

N = 130
SCALES = [0.06, 0.10, 0.16, 0.25, 0.40]
OUTPUT = Path(__file__).parent / "explorer_data.json"

METHODS = {
    "sgolay": ("Savitzky-Golay", sm.SavitzkyGolay(polyorder=3), True),
    "local_poly": ("Local polynomial", sm.LocalPolynomial(degree=2), True),
    "loess": ("LOESS", sm.Loess(degree=1, robust=False), True),
    "pspline": ("Penalized spline", sm.PenalizedSpline(lam=1.0), True),
    "spline": ("Spline (knot-selecting)", sm.InterpolatingSpline(), False),
    "gp": ("Gaussian process", GaussianProcess(n_restarts=0), False),
}


def _ar1_noise(rng: np.random.Generator, n: int, phi: float, sigma: float):
    """Draw AR(1) noise with the given marginal standard deviation."""
    innovation = sigma * np.sqrt(1 - phi**2)
    noise = np.empty(n)
    noise[0] = rng.normal(0, sigma)
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + rng.normal(0, innovation)
    return noise


def build_series() -> dict[str, dict]:
    """The four example series, each with its true derivative where known."""
    x = np.arange(N, dtype=float)
    rng = np.random.default_rng(11)
    series: dict[str, dict] = {}

    trend = 0.04 * x + 2.5 * np.sin(x / 18)
    series["smooth"] = {
        "label": "Trend + cycle",
        "note": "A smooth trend with independent noise. The textbook case.",
        "y": (trend + rng.normal(0, 0.45, N)).tolist(),
        "truth": (0.04 + 2.5 * np.cos(x / 18) / 18).tolist(),
    }

    step = np.where(x < N / 2, 1.0, 4.0) + 0.01 * x
    series["step"] = {
        "label": "Step change",
        "note": (
            "A discontinuity. No smoother can represent it, so every method "
            "trades a spike in bias for its smoothness."
        ),
        "y": (step + rng.normal(0, 0.35, N)).tolist(),
        "truth": None,
    }

    correlated = 0.03 * x + 1.5 * np.sin(x / 25)
    series["ar1"] = {
        "label": "Correlated noise",
        "note": (
            "AR(1) errors with phi=0.75. Wandering noise mimics trend, which "
            "is exactly what the independent-noise assumption cannot see."
        ),
        "y": (correlated + _ar1_noise(rng, N, 0.75, 0.5)).tolist(),
        "truth": (0.03 + 1.5 * np.cos(x / 25) / 25).tolist(),
    }

    prices = pd.read_csv(Path("examples/data/AAPL.csv"), parse_dates=["Date"])
    close = prices["Adj Close"].to_numpy(dtype=float)[-N:]
    series["stock"] = {
        "label": "AAPL close",
        "note": (
            "Real prices, where the truth is unknown and the honest question "
            "is whether an apparent move clears its own error bar."
        ),
        "y": close.tolist(),
        "truth": None,
    }
    return series


def _rounded(values, n):
    """Round for the payload, tolerating a bootstrap that returned nothing."""
    if values is None:
        return [None] * n
    return np.round(values, 6).tolist()


def panel(smoother, axis, y, supports_bias):
    """Fit one configuration under both noise models."""
    out: dict[str, object] = {}
    base = smoother.fit(axis, y, order=1, se=True, noise=IID(), n_bootstrap=60)
    out["smoothed"] = _rounded(base.values, axis.n)
    out["derivative"] = _rounded(base.derivative, axis.n)
    out["se_iid"] = _rounded(base.se, axis.n)
    out["se_method"] = base.provenance.se_method

    correlated = smoother.fit(axis, y, order=1, se=True, noise=AR1(), n_bootstrap=60)
    out["se_ar1"] = _rounded(correlated.se, axis.n)

    if smoother.is_linear:
        simultaneous = smoother.fit(
            axis,
            y,
            order=1,
            se=True,
            noise=IID(),
            simultaneous=True,
            random_state=0,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (simultaneous.ci_upper - simultaneous.derivative) / np.where(
                simultaneous.se > 0, simultaneous.se, np.nan
            )
        out["simultaneous_multiplier"] = float(np.nanmedian(ratio))
    else:
        out["simultaneous_multiplier"] = None

    if supports_bias and smoother.is_linear:
        corrected = smoother.fit(
            axis, y, order=1, se=True, noise=IID(), bias_correct=True
        )
        out["bc_derivative"] = _rounded(corrected.derivative, axis.n)
        out["bc_se"] = _rounded(corrected.se, axis.n)
    else:
        out["bc_derivative"] = None
        out["bc_se"] = None
    return out


def main() -> None:
    """Fit every combination and write the JSON payload."""
    axis = TimeAxis.positional(N)
    series = build_series()
    payload: dict[str, object] = {
        "n": N,
        "x": axis.x.tolist(),
        "scales": SCALES,
        "series": {k: v for k, v in series.items()},
        "methods": {k: {"label": v[0], "linear": None} for k, v in METHODS.items()},
        "panels": {},
    }

    for series_key, spec in series.items():
        y = np.asarray(spec["y"], dtype=float)
        for method_key, (_, prototype, supports_bias) in METHODS.items():
            for index, scale in enumerate(SCALES):
                smoother = prototype.with_scale(scale, axis)
                key = f"{series_key}|{method_key}|{index}"
                payload["panels"][key] = panel(smoother, axis, y, supports_bias)
                payload["methods"][method_key]["linear"] = smoother.is_linear
            print(f"  {series_key:8s} {method_key}")  # noqa: T201

    blob = json.dumps(payload, separators=(",", ":"))
    OUTPUT.write_text(blob)

    # Inline the data so the page is genuinely self-contained: no fetch, so it
    # works from the filesystem and under a strict content security policy.
    template = (Path(__file__).parent / "explorer_template.html").read_text()
    page = Path(__file__).parent / "explorer.html"
    page.write_text(template.replace("__DATA__", blob))

    print(  # noqa: T201
        f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB) and "
        f"{page} ({page.stat().st_size / 1024:.0f} KB)"
    )


if __name__ == "__main__":
    main()
