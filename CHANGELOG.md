# Changelog

## 1.0.0 — 2026-08-06

A rewrite. Every estimator now reports a standard error, and how it gets one is
decided by what the smoother *is* rather than by what it is called. The claim
that this package did "comprehensive uncertainty quantification" was previously
false; it is now measured.

**This release breaks everything written against 0.5.0.** See the migration
table below.

### Why the rewrite

Auditing 0.5.0 found sound point estimates and largely broken uncertainty:

- `GPTrend.predict_derivatives` differenced the posterior mean at 1/100 of the
  sample spacing and summed the endpoint variances as if independent. The
  reported standard error was **3,614× too large**, intervals swallowed every
  plausible value, and `significant_trend` fired on **0%** of a series with an
  obvious trend.
- `bootstrap_derivative_ci` resampled blocks of rows and then reattached the
  original timestamps, destroying the trend being estimated. On a clean linear
  trend of slope 0.101 it returned the interval `[-0.298, 0.577]`. It had no
  tests.
- SiZer's `spline` branch flagged about **90% of pure noise** as trending; its
  `loess` branch flagged almost nothing. The module's own docstring said so.
- `local_polynomial_trend` **crashed on any `DatetimeIndex`** — the package's
  most common input. Every test used a `RangeIndex`, so nothing caught it.
- Six of eleven estimators reported no uncertainty at all.

Adding standard errors to eleven independent functions was not the fix; the
structure was. `trend.py` and `advanced.py` were mutually dependent, dispatch
happened by `match` on strings with imports inside function bodies, and the
result was a naming convention rather than a type — so the ranking layer
discarded standard errors because nothing carried them.

### How uncertainty is computed now

| Route | When | What you get |
|---|---|---|
| `operator` | The derivative is a fixed linear map of the data | The **exact** sampling variance, `diag(L Σ Lᵀ)` |
| `native` | The smoother is a probability model | Its own posterior variance |
| `bootstrap` | Everything else | A simulated sampling distribution |

Which case applies is settled by probing the estimator, not by assumption.
Savitzky-Golay, local polynomials, naive differencing, LOESS with `robust=False`
and penalized splines at fixed `lam` are linear. `UnivariateSpline` (which picks
knots from the data), GCV splines, LOESS with `robust=True` — the default — and
the L1 trend filter are not. A smoother that claims to be linear has its
operator checked against its own output before any exact standard error is
issued.

### Migration

| 0.5.0 | 1.0.0 |
|---|---|
| `bootstrap_derivative_ci(df, method="spline")` | `spline_trend(df, se=True)` |
| `adaptive_gp_trend`, `adaptive_kalman_trend` | removed |
| `select_gp_kernel`, `select_kalman_model` | removed |
| `select_smoothing_parameter_cv` | removed |
| `trending([frames], derivative_order=1, max_or_avg="max", k=5)` | `trending({id: estimate}, how="max", k=5)` — takes `TrendEstimate` objects and propagates their standard errors |
| `stl_decompose`, `simple_deseasonalize`, `deseasonalize_pipeline` | `deseasonalize(df)` — one function, one schema |
| `trend_with_deseasonalization(df, trend_method="spline")` | `trend_with_deseasonalization(df, SavitzkyGolay())` — takes a smoother, not a name |
| `sizer_analysis(...) -> SiZer` | `sizer_analysis(...) -> SiZerMap` |
| `SiZer(method="loess")` | `SiZer(smoother=LocalPolynomial())` |
| `compute_time_deltas(index)` | `TimeAxis.from_index(index)` |
| `incline.testing` | `incline.simulate` |
| `gp_trend(df, kernel_type="rbf")` | `gp_trend(df, kernel="rbf")` |
| `kalman_trend(df, model_type="local_linear")` | `kalman_trend(df)` |
| `sgolay_trend(df, window_size=15)` | `sgolay_trend(df, window_length=15)` |

Output columns are now the same for every estimator:

```
smoothed_value  derivative_value  derivative_method  derivative_order
derivative_se   derivative_ci_lower  derivative_ci_upper
se_method       significant_trend
```

A NaN standard error paired with `se_method` of `None` is a deliberate,
documented state — never a missing column. Gone: `edge_region`, `changepoint`,
`smoothed_value_std`, `fitted_*_variance`, `trend_derivative_value`,
`max_or_avg`.

### Added

- `se=True` on every estimator. Off by default: the exact route costs one
  smoother evaluation per observation, and that should be a choice.
- `noise=` — `IID` (default), `AR1`, `Heteroskedastic`, or an explicit
  covariance via `Given`.
- `bias_correct=True`, which subtracts smoothing bias estimated from a
  less-smoothed pilot fit. The corrected estimator is itself linear, so exact
  variances still apply.
- `simultaneous=True` for bands that cover the whole curve rather than each
  point separately.
- `Smoother`, `TrendEstimate`, `TimeAxis` and the `SMOOTHERS` registry as
  public API, so a new smoother works everywhere without editing a dispatcher.
- `PenalizedSpline` (linear at fixed `lam`, so exact standard errors) alongside
  the existing knot-selecting `spline_trend`.
- `GaussianProcess` gains `amplitude`, `optimize` and `standardize`, which
  together let a prior be stated exactly.
- An interactive uncertainty explorer in the documentation.

### Fixed

- The Gaussian process derivative posterior is now analytic — the kernel is
  differentiated rather than the posterior mean finite-differenced. Median
  standard error went from 256.2 to 0.071 on the audit series.
- `GaussianProcess.with_scale` was a no-op: sklearn re-optimizes hyperparameters
  regardless of `n_restarts_optimizer`, so a multi-scale sweep produced one
  scale repeated.
- `local_polynomial_trend` crashed on `DatetimeIndex`; the time-axis derivation
  returned a pandas `Index` rather than an ndarray.
- A zero standard error let a derivative of 3e-17 be reported as a significant
  trend at every point of a flat line.
- `L1TrendFilter.with_scale` mapped to an absolute penalty whose whole range was
  saturated, so every scale returned byte-identical output.
- `naive_trend` never emitted `smoothed_value`; `trending` returned a different
  schema when it matched nothing; `l1_trend_filter` conflated `function_order`
  with `derivative_order`.
- The state-space model's `damped_trend` option was forwarded to a statsmodels
  parameter that does not exist, so it silently did nothing.
- `docs/conf.py` set two unregistered Sphinx keys, so the documentation built
  with `allow_errors=True` and broken examples rendered as tracebacks.
- CI installed the extras `.[test,advanced]`, neither of which existed.
- `noise=` was ignored by every bootstrapped smoother: the fitted noise model
  was computed and discarded, and the bootstrap re-derived its own scale, so an
  explicit `IID(sigma=...)`, `Heteroskedastic` or `Given` had no effect on the
  reported uncertainty.
- `TimeAxis.require_regular` existed but was never called, so Savitzky-Golay and
  naive differencing applied a uniform-grid method to unevenly spaced series
  without warning. On an irregular axis with a true slope of 2.0, Savitzky-Golay
  returned a median of 2.446 silently.
- `GaussianProcess(kernel="matern52", derivative_order=2)` raised, though a
  nu=5/2 process is twice mean-square differentiable and the capability was
  advertised. The second-derivative kernel and its prior variance are now
  implemented.
- `trend_with_deseasonalization(..., confidence_level=x)` returned a 95%
  interval whatever level was requested, because the pipeline bootstrap
  hard-coded its percentiles.

- The deseasonalizing pipeline reported every point of a pure-noise series as
  significantly trending. When the decomposition left residuals with no spread
  each resample reproduced the same curve, so the standard error came out at
  7e-18 and the significance flag was a comparison against zero.
- `AR1(phi=...)` ignored the stated autocorrelation when scaling the noise
  level, returning sigma 1.417 for phi of 0.0 and of 0.9 alike. Sigma is
  recovered by dividing the second difference's observed variance by its
  theoretical value at a given phi, and that divisor falls from 6 to 0.9 across
  the grid, so one sigma cannot describe the process at another phi.
- `generate_time_series` carried a daily `DatetimeIndex` whatever `x_range`
  said, so an estimator read a spacing of one day while the returned true
  derivative was per unit of x. Over (0, 10) with 100 points those differ by a
  factor of ten, which silently rescaled anything measured against it.
- A single series whose smoother returned nothing usable turned *every* rank
  into NaN, because `rankdata` propagates it. Unusable series now sort last.
- Savitzky-Golay raised from inside scipy on a series shorter than its window,
  reporting "polyorder must be less than window_length" and naming neither the
  length nor the remedy. The window is clamped where one exists and the error
  says so where none does.
- `estimate_trend` routed keyword arguments through `__dataclass_fields__`,
  which includes `ClassVar` pseudo-fields, so `linear=False` was passed to the
  constructor and raised from inside `dataclasses` rather than being reported
  as an unknown option.
- The Gaussian process and state-space models accepted `noise=` and
  `simultaneous=` and silently ignored both — neither exposes a linear operator,
  so the interval comes from its own posterior. They now say so.
- SiZer checked that enough observations were finite and then swept the raw
  series anyway, so a single missing value produced an entirely blank map, which
  reads as "nothing is trending" rather than "nothing could be computed".
- The operator cache was bounded at 64 entries, each holding two n-by-n
  matrices. At n=2000 that is 64 MB apiece, so a multi-scale sweep on a long
  series could reach roughly 4 GB before evicting anything. The bound is now in
  bytes.

- A `PeriodIndex` was rejected, though monthly and quarterly series are
  ordinarily indexed that way. It is now read as a time axis.
- An index carrying no time information — strings, categories — surfaced a raw
  `could not convert string to float: 'r0'` from numpy, naming neither the cause
  nor the remedy. It now says to use a datetime, period or numeric index, or to
  pass `time_column=`.

A scale you state explicitly is always used. A scale the package *estimates* as
a plain scalar is left to the bootstrap's own difference-based estimate:
`estimate_ar1`'s sigma exists to pair with a full AR(1) covariance and runs
about 1.4x high on its own, which on top of an already over-dispersed block
bootstrap gave standard errors 2.8x the estimator's real spread. The block
bootstrap over-covers under dependence regardless — about 1.7x even with the
exact noise level — and that is documented rather than papered over.

### Verified

Measured over 400 replicates with the truth inside each estimator's
approximation space, so smoothing bias is exactly zero:

| estimator | bias (t) | reported SE ÷ actual spread | coverage of a nominal 95% interval |
|---|---|---|---|
| local polynomial, degree 2 | −0.27 | 1.039 | 0.955 |
| Savitzky-Golay, order 3 | +1.29 | 1.074 | 0.963 |
| naive differencing | +0.53 | 1.004 | 0.963 |
| LOESS | +1.16 | 1.026 | 0.938 |
| penalized spline | +1.24 | 0.994 | 0.948 |

Under the null the significance flag fires 3.7–7.0% of the time against a
nominal 5%; under a real trend, power rises monotonically to 1.0. Gaussian
process credible intervals cover 0.945 when the data come from its prior. SiZer
flags 3.1% of pure noise pointwise and 0.1% with a whole-curve band, against
~90% before.

Known shortfalls, all measured and documented in `docs/limitations.md`: the
state-space model covers 0.80 because its intervals are conditional on the
fitted variances; `noise="ar1"` reaches 0.81 rather than 0.95 because the
autocorrelation is itself estimated; and a standard error describes the
derivative of the *fitted curve*, not of the truth — the gap is smoothing bias
and it is set by your bandwidth.

### Removed

`incline.trend`, `incline.advanced`, `incline.gaussian_process`,
`incline.statespace`, `incline.multiscale` and `incline.testing`. The robust
statistics options on `trending` (`robust`, `trim_fraction`, winsorized and
Huber means) went with it: they hardened the point estimate while significance
came from resampling autocorrelated estimates as though independent, which was
the larger error.

---

## 0.5.0 and earlier

See the git history.
