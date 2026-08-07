# Uncertainty

A trend estimate without a standard error is a number you cannot act on. The
interactive explorer below is the fastest way to see what incline's intervals do
and where they stop being trustworthy.

```{raw} html
:file: _static/explorer.html
```

## How a standard error gets computed

Which machinery applies is decided by what the smoother *is*, never by its name.

| Route | When | What you get |
|---|---|---|
| `operator` | The derivative is a fixed linear map of the data | The **exact** sampling variance, `diag(L Σ Lᵀ)` — no asymptotics, no resampling |
| `native` | The smoother is a probability model (Gaussian process, state space) | Its own posterior variance, which it already knows |
| `bootstrap` | Everything else | A simulated sampling distribution |

Whether a smoother is linear is settled by probing it, not by assumption:

| Linear — exact variance | Nonlinear — bootstrapped |
|---|---|
| Savitzky-Golay | `UnivariateSpline` (picks knots from the data) |
| Local polynomial | Penalized spline with GCV (penalty chosen from the data) |
| Penalized spline at fixed `lam` | LOESS with `robust=True` *(the default)* |
| LOESS with `robust=False` | L1 trend filter |
| Naive differencing | |

The declaration is enforced. A smoother that claims to be linear has its operator
checked against its own output before any exact standard error is issued, so a
wrong claim raises rather than quietly producing wrong inference.

## Two things a standard error does not tell you

**It is about the smooth, not the truth.** Every smoother estimates the derivative
of its own smoothed curve. The gap between that and the true derivative is
smoothing bias, and it is governed by the bandwidth you chose. Measured over 120
replicates on a known trend:

| Method | reported SE ÷ actual spread | coverage of its own estimand | coverage of the *true* derivative |
|---|---|---|---|
| Savitzky-Golay, window 21 | 1.010 | 0.950 | 0.950 |
| Naive differencing | 1.001 | 0.949 | 0.949 |
| Local polynomial, bw 0.15 | 1.008 | 0.947 | **0.043** |
| Penalized spline, λ=5·10⁴ | 1.018 | 0.952 | **0.056** |
| LOESS, frac 0.3 | 1.026 | 0.952 | **0.089** |

The variance is right in every row. The last column collapses only where the
bandwidth oversmooths — the interval is correctly sized and centered in the wrong
place. `bias_correct=True` re-centers it, at roughly five times the width; on the
LOESS row that moves coverage from 0.089 to 0.941.

**It assumes independent noise unless told otherwise.** Under AR(1) errors with
φ=0.7 the independence assumption reports standard errors **29% of their true
size**. Pass `noise='ar1'`:

```python
from incline import sgolay_trend

result = sgolay_trend(df, se=True, noise="ar1")
```

The autocorrelation is estimated from second differences of the raw series, never
from the smoother's residuals — smoothing strips the low-frequency noise along
with the trend, and residual-based estimates of φ come out around 0.21 when the
truth is 0.7.

## Pointwise versus whole-curve

A 95% pointwise interval fails somewhere along a 130-point curve far more often
than 5% of the time. `simultaneous=True` widens to a band that covers the whole
curve at once — for the explorer's default series that multiplier is 3.46 rather
than 1.96.

## The columns

Every estimator returns the same schema, whether or not it can support a standard
error:

```
derivative_value      the point estimate
derivative_se         NaN when unavailable
derivative_ci_lower   NaN when derivative_se is NaN
derivative_ci_upper
se_method             'operator' | 'native' | 'bootstrap' | None
significant_trend     False when no interval exists
```

`derivative_se` of NaN with `se_method` of None is a deliberate, documented state.
It is never a missing column, so downstream code can always index it.

Standard errors are opt-in via `se=True`: the exact route costs one smoother
evaluation per observation, and that should be a choice rather than a surprise.
