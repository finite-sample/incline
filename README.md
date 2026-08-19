# incline: Estimate Local Trend in a Noisy Time Series

[![PyPI version](https://img.shields.io/pypi/v/incline.svg)](https://pypi.python.org/pypi/incline)
[![Downloads](https://static.pepy.tech/badge/incline)](https://pepy.tech/project/incline)
[![CI](https://github.com/finite-sample/incline/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/incline/actions?query=workflow%3Aci)
[![Docs](https://img.shields.io/badge/docs-github.io-blue)](https://finite-sample.github.io/incline/)

How fast is this series moving *right now*? Differencing consecutive
observations amplifies noise rather than revealing signal, so incline smooths
the series first and differentiates the smooth.

The second half is the part worth having: **every estimator reports a standard
error**, and which machinery produces it is decided by what the smoother *is*
rather than by what it is called.

```python
import numpy as np
import pandas as pd
from incline import sgolay_trend

df = pd.DataFrame(
    {"value": np.linspace(0, 10, 100) + np.random.normal(0, 0.5, 100)},
    index=pd.date_range("2020-01-01", periods=100),
)

result = sgolay_trend(df, se=True)
result[["derivative_value", "derivative_se", "significant_trend", "se_method"]].head()
```

`significant_trend` tells you where the data support saying the series is moving
at all. `se_method` tells you how that was established.

## How uncertainty is computed

| Route | When it applies | What you get |
|---|---|---|
| `operator` | The derivative is a fixed linear map of the data | The **exact** sampling variance — no asymptotics, no resampling |
| `native` | The smoother is a probability model (Gaussian process, state space) | Its own posterior variance |
| `bootstrap` | Everything else | A simulated sampling distribution |

Which case a smoother falls into is settled by probing it, not by assumption:

| Linear — exact variance | Nonlinear — bootstrapped |
|---|---|
| Savitzky-Golay | `UnivariateSpline` (chooses knots from the data) |
| Local polynomial | Penalized spline with GCV |
| Penalized spline at fixed `λ` | LOESS with `robust=True` *(the default)* |
| LOESS with `robust=False` | L1 trend filter |
| Naive differencing | |

A smoother that claims to be linear has its operator checked against its own
output before any exact standard error is issued, so a wrong claim raises rather
than quietly producing wrong inference.

## Do the standard errors work?

Measured, not asserted. `tests/test_econometrics.py` simulates from known truth
and checks the classic properties — unbiasedness, coverage, size and power. Over
400 replicates, with the truth inside each estimator's approximation space so
that smoothing bias is exactly zero:

| estimator | bias (t) | reported SE ÷ actual spread | coverage of a nominal 95% interval |
|---|---|---|---|
| local polynomial, degree 2 | −0.27 | 1.039 | 0.955 |
| Savitzky-Golay, order 3 | +1.29 | 1.074 | 0.963 |
| naive differencing | +0.53 | 1.004 | 0.963 |
| LOESS | +1.16 | 1.026 | 0.938 |
| penalized spline | +1.24 | 0.994 | 0.948 |

Every bias is indistinguishable from zero and every coverage sits inside the
binomial band for 0.95. Under the null the significance flag fires 3.7–7.0% of
the time against a nominal 5%; under a real trend, power rises monotonically to
1.0.

## What the intervals do not tell you

They describe the derivative of the **fitted curve**, not of reality. The gap is
smoothing bias, and it is set by your bandwidth. At an oversmoothed bandwidth a
perfectly calibrated interval still misses the truth — `bias_correct=True`
re-centers it, at roughly five times the width.

The default also assumes **independent** noise. Under AR(1) errors with φ=0.7
that reports standard errors about a quarter of their true size and covers 39%
of the time; `noise="ar1"` recovers most of it. See
[Limitations](https://finite-sample.github.io/incline/limitations.html) for both
in full, with numbers.

## Methods

```python
from incline import (
    naive_trend,  # central differences; the baseline to beat
    sgolay_trend,  # local polynomial on a fixed window
    local_polynomial_trend,  # kernel-weighted local regression
    loess_trend,  # LOESS
    pspline_trend,  # penalized smoothing spline
    spline_trend,  # knot-selecting smoothing spline
    l1_trend_filter,  # piecewise-polynomial with sparse kinks
    gp_trend,  # Gaussian process, exact derivative posterior
    kalman_trend,  # local linear trend state-space model
)
```

Plus `SiZer` for multi-scale analysis, `deseasonalize` for seasonal adjustment,
and `trending` for ranking thousands of series by how fast they are moving —
propagating the standard errors, so the ranking can say which leaders are
actually distinguishable from flat.

## Installation

```bash
pip install incline
```

## Documentation

[finite-sample.github.io/incline](https://finite-sample.github.io/incline/) —
including an [interactive
explorer](https://finite-sample.github.io/incline/uncertainty.html) where you
can move the smoothing slider and watch the interval trade width for bias.

For background on what "the trend over a window" even means, see
[this note](http://gbytes.gsood.com/2018/06/22/talking-on-a-tangent/).

## Authors

Gaurav Sood and contributors.

## License

MIT
