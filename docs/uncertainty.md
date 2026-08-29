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
| `operator` | The derivative is a fixed linear map of the data | The sampling variance `diag(L Σ Lᵀ)`, conditional on the fitted or supplied noise covariance — no asymptotics, no resampling |
| `native` | The smoother is a probability model (Gaussian process, state space) | Its own posterior variance, which it already knows |
| `bootstrap` | Everything else | A simulated sampling distribution |

Whether a smoother is linear is settled by probing it, not by assumption:

| Linear — exact variance | Nonlinear — bootstrapped |
|---|---|
| Savitzky-Golay | Smoothing spline with GCV (penalty chosen from the data) |
| Local polynomial | LOESS with `robust=True` *(the default)* |
| Smoothing spline at fixed `penalty` | L1 trend filter |
| LOESS with `robust=False` | |
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
| Smoothing spline, λ=5·10⁴ | 1.018 | 0.952 | **0.056** |
| LOESS, span 0.3 | 1.026 | 0.952 | **0.089** |

The variance is right in every row. The last column collapses only where the
bandwidth oversmooths — the interval is correctly sized and centered in the wrong
place. `bias_correct=True` re-centers it, at roughly five times the width; on the
LOESS row that moves coverage from 0.089 to 0.941.

**It assumes independent noise unless told otherwise.** Under AR(1) errors with
φ=0.7 the independence assumption reports standard errors **29% of their true
size**. Pass `noise='ar1'`:

```python
from incline import sgolay_trend

result = sgolay_trend(df, with_uncertainty=True, noise="ar1")
```

`noise` is an estimation option, not a label. Without
`with_uncertainty=True`, it is accepted only by an adaptive smoothing spline,
where covariance changes penalty selection and the fitted curve. Fixed
smoothers reject it because it cannot affect their point estimate. Gaussian
process and state-space smoothers model noise internally and reject the
external option in both modes.

The autocorrelation is estimated from second differences of the raw series, never
from the smoother's residuals — smoothing strips the low-frequency noise along
with the trend, and residual-based estimates of φ come out around 0.21 when the
truth is 0.7.

For a nonlinear smoother, `Given(covariance)` uses a Gaussian parametric
bootstrap: each replicate draws an error vector from the complete supplied
covariance and refits the smoother. This preserves arbitrary off-diagonal
dependence that residual or block resampling cannot reconstruct from one
series. Supplying a covariance does not specify higher moments, so Gaussian
errors are the explicit distributional assumption on this route.

For an adaptive smoothing spline, the fitted covariance also enters the point
fit: the roughness penalty is selected by covariance-aware generalized maximum
likelihood and the curve is fit by penalized generalized least squares. Its
bootstrap draws Gaussian errors from that covariance and repeats covariance and
penalty estimation in every replicate. This follows the correlated-spline
framework of [Diggle and Hutchinson (1989)](https://doi.org/10.1111/j.1467-842X.1989.tb00510.x)
and [Wang (1998)](https://doi.org/10.1080/01621459.1998.10474115).

The GML fit reports `generalized_penalty` in its provenance and output frame.
It is the coefficient on roughness in the covariance-weighted objective
$ (y-f)^T \Sigma^{-1}(y-f) + \lambda f^T Kf $. Its scale therefore depends on
the fitted covariance. It is a diagnostic, not a value to pass back through the
public `penalty` argument, which configures SciPy's independent-error spline.

## Pointwise versus whole-curve

A 95% pointwise interval fails somewhere along a 130-point curve far more often
than 5% of the time. For fixed linear smoothers, `simultaneous=True` widens to
a band that covers the whole curve at once — for the explorer's default series
that multiplier is 3.46 rather than 1.96. Bootstrap and native-posterior
smoothers reject this option because they do not provide a validated
whole-curve band.

## The columns

Every estimator returns the same schema, whether or not it can support a standard
error:

```
derivative_value      the point estimate
derivative_standard_error         NaN when unavailable
derivative_ci_lower   NaN when derivative_standard_error is NaN
derivative_ci_upper
uncertainty_method             'operator' | 'native' | 'bootstrap' | None
confidence_level                interval level, or NaN
simultaneous                    whether this is a whole-curve band
bias_corrected                  whether pilot-fit correction was applied
significant_trend     False when no interval exists
```

`derivative_standard_error` of NaN with `uncertainty_method` of None is a deliberate, documented state.
It is never a missing column, so downstream code can always index it.

Standard errors are opt-in via `with_uncertainty=True`: the exact route costs one smoother
evaluation per observation, and that should be a choice rather than a surprise.
