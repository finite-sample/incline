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
RANDOM_STATE = 11
SCALES = [0.06, 0.10, 0.16, 0.25, 0.40]
OUTPUT = Path(__file__).parent / "explorer_data.json"

METHODS = {
    "sgolay": ("Savitzky-Golay", sm.SavitzkyGolay(degree=3), True),
    "local_poly": ("Local polynomial", sm.LocalPolynomial(degree=2), True),
    "loess": ("LOESS", sm.Loess(robust=False), True),
    "smoothing_spline": (
        "Smoothing spline",
        sm.SmoothingSpline(penalty=1.0),
        True,
    ),
    "gp": ("Gaussian process", GaussianProcess(n_restarts=0), False),
}


def _ar1_noise(rng: np.random.Generator, n: int, phi: float, standard_deviation: float):
    """Draw AR(1) noise with the given marginal standard deviation."""
    innovation = standard_deviation * np.sqrt(1 - phi**2)
    noise = np.empty(n)
    noise[0] = rng.normal(0, standard_deviation)
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + rng.normal(0, innovation)
    return noise


def build_series() -> dict[str, dict]:
    """The four example series, each with its true derivative where known."""
    x = np.arange(N, dtype=float)
    rng = np.random.default_rng(RANDOM_STATE)
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
    fit_options = {
        "derivative_order": 1,
        "with_uncertainty": True,
        "n_bootstrap": 60,
    }
    if smoother.has_native_posterior:
        base = smoother.fit(axis, y, **fit_options)
        correlated = base
    else:
        base = smoother.fit(axis, y, noise=IID(), **fit_options)
        correlated = smoother.fit(axis, y, noise=AR1(), **fit_options)
    out["smoothed"] = _rounded(base.values, axis.n)
    out["derivative"] = _rounded(base.derivative, axis.n)
    out["standard_error_iid"] = _rounded(base.standard_error, axis.n)
    out["uncertainty_method"] = base.provenance.uncertainty_method

    out["standard_error_ar1"] = _rounded(correlated.standard_error, axis.n)

    if smoother.is_linear:
        simultaneous = smoother.fit(
            axis,
            y,
            derivative_order=1,
            with_uncertainty=True,
            noise=IID(),
            simultaneous=True,
            random_state=RANDOM_STATE,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (simultaneous.ci_upper - simultaneous.derivative) / np.where(
                simultaneous.standard_error > 0, simultaneous.standard_error, np.nan
            )
        out["simultaneous_multiplier"] = float(np.nanmedian(ratio))
    else:
        out["simultaneous_multiplier"] = None

    if supports_bias and smoother.is_linear:
        corrected = smoother.fit(
            axis,
            y,
            derivative_order=1,
            with_uncertainty=True,
            noise=IID(),
            bias_correct=True,
        )
        out["bias_corrected_derivative"] = _rounded(corrected.derivative, axis.n)
        out["bias_corrected_standard_error"] = _rounded(
            corrected.standard_error, axis.n
        )
    else:
        out["bias_corrected_derivative"] = None
        out["bias_corrected_standard_error"] = None
    return out


def main() -> None:
    """Fit every combination and write the JSON payload."""
    axis = TimeAxis.positional(N)
    series = build_series()
    payload: dict[str, object] = {
        "n": N,
        "x": axis.x.tolist(),
        "scales": SCALES,
        "series": dict(series.items()),
        "methods": {
            key: {
                "label": value[0],
                "linear": None,
                "supports_noise_model": not value[1].has_native_posterior,
            }
            for key, value in METHODS.items()
        },
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
    OUTPUT.write_text(f"{blob}\n")

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
