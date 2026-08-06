# Quickstart

## Estimate a trend

Smooth the series, then differentiate the smooth:

```python
import numpy as np
import pandas as pd
from incline import sgolay_trend

df = pd.DataFrame(
    {"value": np.linspace(0, 10, 100) + np.random.normal(0, 0.5, 100)},
    index=pd.date_range("2020-01-01", periods=100),
)

result = sgolay_trend(df, window_length=15, function_order=3)
result[["smoothed_value", "derivative_value"]].head()
```

The derivative is reported **per unit of the time axis** — per day for a daily
`DatetimeIndex`, per unit of your time column if you pass one.

## Ask for uncertainty

An estimate without a standard error is hard to act on. Pass `se=True`:

```python
result = sgolay_trend(df, se=True)

result[[
    "derivative_value",
    "derivative_se",
    "derivative_ci_lower",
    "derivative_ci_upper",
    "significant_trend",
    "se_method",
]].head()
```

It is opt-in because the exact route costs one smoother evaluation per
observation, and that should be a decision rather than a surprise.

`significant_trend` is True where the interval excludes zero — where the data
support saying the series is moving at all.

## The output columns

Every estimator returns the same columns, whichever method produced them:

| Column | Meaning |
|---|---|
| `smoothed_value` | The fitted curve |
| `derivative_value` | Its derivative, per unit time |
| `derivative_method` | Which smoother ran |
| `derivative_order` | Which derivative this is |
| `derivative_se` | Standard error, or NaN |
| `derivative_ci_lower` / `_upper` | Interval bounds, or NaN |
| `se_method` | `operator`, `native`, `bootstrap`, or None |
| `significant_trend` | Whether the interval excludes zero |

A NaN standard error paired with `se_method` of None is a deliberate, documented
state — never a missing column — so downstream code can always index it.

Smoothers add their own settings as extra columns: `window_length` for
Savitzky-Golay, `bandwidth` for LOESS and local polynomials, and so on.

## Choosing a method

```python
from incline import (
    naive_trend,            # central differences; the baseline to beat
    sgolay_trend,           # local polynomial on a fixed window
    local_polynomial_trend, # kernel-weighted local regression
    loess_trend,            # LOESS
    pspline_trend,          # penalized smoothing spline
    spline_trend,           # knot-selecting smoothing spline
    l1_trend_filter,        # piecewise-polynomial with sparse kinks
    gp_trend,               # Gaussian process
    kalman_trend,           # local linear trend state-space model
)
```

If you have no strong preference, let the package pick:

```python
from incline import estimate_trend, select_trend_method

print(select_trend_method(df))          # e.g. 'loess'
result = estimate_trend(df, method="auto", se=True)
```

Pass `criteria="exact"` to `select_trend_method` to require a method whose
standard errors are exact rather than bootstrapped.

## Second derivatives

```python
acceleration = sgolay_trend(df, derivative_order=2, window_length=21)
```

Not every method supports every order — a Matérn-3/2 Gaussian process is only
once differentiable, and asking for more raises rather than returning a number.

## An explicit time column

When the index is not the time axis:

```python
df = pd.DataFrame({"t": [0.0, 1.5, 2.0, 4.5], "value": [1.0, 2.0, 2.5, 5.0]})
result = sgolay_trend(df, column_value="value", time_column="t")
```

## Correlated errors

The default assumes independent noise. When errors are autocorrelated — which
is common in real series — that assumption reports standard errors substantially
smaller than they should be:

```python
result = sgolay_trend(df, se=True, noise="ar1")
```

See [Uncertainty](uncertainty.md) for what that costs and what it buys.

## Ranking many series

```python
from incline import trending, estimate, SavitzkyGolay

estimates = {
    name: estimate(SavitzkyGolay(), series, se=True)
    for name, series in your_series.items()
}

ranked = trending(estimates, k=5, how="mean")
ranked[["id", "trend", "trend_se", "significant", "rank"]]
```

Standard errors computed upstream are carried through, so the ranking can tell
you which of the leaders are actually distinguishable from flat.
