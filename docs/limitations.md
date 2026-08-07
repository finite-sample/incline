# Limitations

What the package does not do, and what its numbers do not mean. Everything
quantified below is measured by `tests/test_econometrics.py`, which runs against
the shipped code rather than against a description of it.

## A standard error is about the smooth, not about the truth

Every estimator here smooths first and differentiates the smooth. What it
estimates is therefore the derivative of *its own fitted curve*, and the
standard error describes how that quantity varies across repeated samples.

The gap between the fitted curve and reality is **smoothing bias**, and it is
governed by the bandwidth you chose — not by the uncertainty machinery. The two
are separable, and worth separating, because they fail differently:

| Method (bandwidth) | reported SE ÷ actual spread | covers its own estimand | covers the **true** derivative |
|---|---|---|---|
| Savitzky-Golay, window 21 | 1.010 | 0.950 | 0.950 |
| Naive differencing | 1.001 | 0.949 | 0.949 |
| Local polynomial, bw 0.15 | 1.008 | 0.947 | **0.043** |
| Penalized spline, λ=5·10⁴ | 1.018 | 0.952 | **0.056** |
| LOESS, frac 0.3 | 1.026 | 0.952 | **0.089** |

The variance is right in every row. Where the last column collapses, the
interval is correctly sized and centered in the wrong place, because that
bandwidth oversmooths the series it was given.

Two ways to deal with it:

- **Undersmooth.** A narrower bandwidth trades precision for a smaller bias.
- **Correct it.** `bias_correct=True` estimates the bias with a less-smoothed
  pilot fit and subtracts it. On the LOESS row above that moves coverage from
  0.089 to 0.941, at roughly five times the interval width.

When the truth happens to lie inside a method's approximation space — a
quadratic for a degree-2 local polynomial, say — the bias is exactly zero and
coverage of the true derivative is nominal. That is the regime the calibration
tests use, precisely so that the standard error is measured on its own terms.

## Independent noise is assumed unless you say otherwise

`noise='iid'` is the default and it is often wrong. Under AR(1) errors with
φ=0.7, measured over 400 replicates:

| | reported SE ÷ actual spread | coverage of a nominal 95% interval |
|---|---|---|
| assuming independence | **0.243** | **0.393** |
| `noise='ar1'` | 1.219 | 0.810 |

Ignoring dependence reports standard errors about a quarter of their true size,
and a nominal 95% interval covers under 40% of the time. Modeling it recovers
most of that but not all: the autocorrelation is itself estimated from a finite
series, so the interval inherits that uncertainty and lands around 0.81 rather
than 0.95.

The autocorrelation is estimated from **second differences of the raw series**,
never from the smoother's residuals. Smoothing strips the low-frequency part of
the noise along with the trend, and a residual-based estimate of φ comes out
around 0.21 when the truth is 0.7.

Only AR(1) is supported. Long-memory processes, and dependence that changes over
the series, are not.

Those figures are for a smoother with an exact operator. For a **bootstrapped**
smoother under dependence the picture is worse and the direction is the other
way: the block bootstrap is over-dispersed, reporting standard errors about
1.7x the estimator's true spread even when handed the exact noise level. An
estimated scalar scale is therefore left to the bootstrap's own
difference-based estimate rather than replaced by the AR(1) one, which lands it
near 0.77 instead — under rather than over, and much closer to right. A scale
you state explicitly is always used as given.

## Gaussian processes and state-space models are gated differently

Both are shrinkage estimators: biased by construction, because that is what
shrinkage is. The unbiasedness and frequentist-coverage gates above therefore do
not apply to them.

The guarantee that does apply is Bayesian — when the data come from the model's
prior, a 95% credible interval should contain the truth 95% of the time. That is
measured in `tests/test_bayesian_calibration.py`:

| | coverage of a nominal 95% interval | reported SE ÷ actual error |
|---|---|---|
| Gaussian process, kernel held fixed | 0.900 | — |
| Local linear trend, variances re-estimated | **0.800** | 0.772 |

Neither reaches nominal, for different and identified reasons.

The Gaussian process falls slightly short because the fit standardizes the
response: `noise_level` and the signal amplitude end up in units of the series'
own standard deviation rather than its original units, and the amplitude is
fixed at 1 internally and cannot be set. A caller therefore cannot specify a
prior exactly, so the prior being conditioned on is not quite the prior the data
came from.

The state-space model falls further short, and its
intervals are **conditional on the fitted variances**, and that estimation error
is not propagated, so they come out about a quarter too narrow.

A correction for this was implemented and then removed. It scaled the interval
by each variance parameter's relative standard error, `bse / |param|` — which is
undefined at the boundary, and variances land on exactly zero routinely whenever
a component is not needed. Over 40 fits the median inflation factor was 10⁵ and
the largest 3·10⁸, turning a standard error of 0.017 into 4.6·10⁶. A correction
that can be eight orders of magnitude wrong is worse than the bias it removes.
Propagating hyperparameter uncertainty properly is not implemented.

## Seasonal adjustment costs uncertainty, and that is now counted

The seasonal component is estimated from the same data as the trend, so an
interval that treats it as known is too narrow. Measured on a linear trend with
a twelve-period cycle:

| | coverage of a nominal 95% interval | reported SE ÷ actual spread |
|---|---|---|
| treating the seasonal fit as exact | 0.917 | 0.861 |
| **what `trend_with_deseasonalization` does** | 0.983 | 1.052 |
| oracle — true seasonal component known | 0.933 | — |

Asking `trend_with_deseasonalization` for a standard error bootstraps the
**whole pipeline**: it resamples the decomposition's residuals, redoes the
decomposition *and* the trend fit together, and takes the interval from the
spread. It errs slightly conservative, which is the right direction.

That costs `n_bootstrap` decompositions, and is only paid when `se=True`. To
skip it, compose the two steps yourself — which states the assumption instead of
hiding it:

```python
adjusted = deseasonalize(df)
result = sgolay_trend(adjusted, column_value="deseasonalized", se=True)
```

## Non-Gaussian noise is fine; non-constant variance is not

These two are worth separating, because one is a non-issue and the other hides.

**Heavy tails and skew barely matter.** The derivative is a weighted sum of many
observations, so its own distribution is close to Gaussian regardless, and the
variance formula only ever needed the second moment. Measured coverage for a
nominal 95% interval: Gaussian 0.950, Student-t with 3 degrees of freedom 0.960,
Laplace 0.930, centered exponential 0.930.

**Non-constant variance is a real failure, and looks fine in the middle.** With
one noise level fitted to a series whose scale rises fivefold:

| position | local noise sd | coverage | reported SE ÷ actual spread |
|---|---|---|---|
| early | 0.33 | 1.000 | **1.97** |
| middle | 0.60 | 0.980 | 1.06 |
| late | 0.88 | **0.830** | **0.66** |

The midpoint looks perfectly calibrated, which is exactly why this is easy to
miss. Pass `noise=Heteroskedastic()` — or `noise="heteroskedastic"` — and the
per-point noise level is estimated from a rolling second-difference window,
restoring 0.90–0.92 coverage and SE ratios of 0.91–1.00 at every position.

## Pointwise intervals are not whole-curve statements

A 95% pointwise interval fails somewhere along a 130-point curve far more often
than 5% of the time. `simultaneous=True` widens to a band that covers the whole
curve at once — for a typical series that multiplier is around 3.5 rather than
1.96 — and is available only for smoothers with an exact operator.

SiZer's map applies this per scale. It is **not** corrected jointly across
scales as well, because neighboring scales are so dependent that treating them
as separate tests would be far too conservative. Reading a single cell in
isolation still overstates confidence; reading persistence across scales is the
intended use.

## Missing values are refused, not filled in

A series with `NaN` values is rejected rather than estimated. Interpolate or
drop the gaps first:

```python
trend = sgolay_trend(df.assign(value=df["value"].interpolate()), se=True)
```

This is a deliberate refusal, and it is stricter than the package used to be.
Every route to uncertainty gives a *wrong* answer on a gapped series rather
than an unavailable one, which is the worse failure because the caller cannot
see it. The noise estimators drop the gap and take second differences across
it, reading the resulting jump as noise: on a trending series with a 20-point
gap that moved the estimated noise level from 0.286 to 1.782 and the AR(1)
scale to 8.11, with a spurious autocorrelation of 0.93. Standard errors six
times too wide report a genuine trend as insignificant.

Interpolating first is not free either — it invents data, and the interval will
not know that. But it is a choice you make with your eyes open, at a place in
your own code where you can see it, which the silent version was not.

Estimating on the observed subset against its true irregular axis would be the
principled alternative, since a gapped regular series *is* an irregular series
and the package already handles those. That is not implemented.

## Boundaries

Every smoother has data on one side only at the ends of a series, so intervals
flare there. That is real, not an artifact, and it is why the calibration tests
measure interior points.

## Cost

The exact route recovers the estimator's linear operator by evaluating it once
per observation, which is O(n²) work. Operators are cached per (smoother, axis,
derivative order), so a multi-scale sweep or a simulation study over one grid
pays it once. For very long series prefer a smoother with a closed-form
operator, or use the bootstrap.

## Still not covered

Long-memory noise, dependence whose strength changes across the series, and
propagating hyperparameter uncertainty in the state-space model.
