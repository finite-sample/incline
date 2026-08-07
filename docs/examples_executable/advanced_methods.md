# Advanced methods

Every example here runs when the documentation is built, so nothing on this page
can drift from the code.

The thread running through all of it: an estimate of a trend is not worth much
without a statement of how sure you are. Each method below reports one, and
which machinery produces it depends on what the smoother *is*.

```{jupyter-execute}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from incline import (
    SavitzkyGolay, LocalPolynomial, GaussianProcess, StateSpace,
    gp_trend, kalman_trend, sgolay_trend, local_polynomial_trend,
    deseasonalize, trend_with_deseasonalization,
    SiZer, sizer_analysis, estimate,
)

rng = np.random.default_rng(7)
n = 200
t = np.arange(n, dtype=float)
index = pd.date_range("2020-01-01", periods=n, freq="D")
```

## Gaussian process regression

The derivative of a Gaussian process is itself a Gaussian process. Its posterior
mean and variance both follow from differentiating the covariance function, so
the standard error is exact rather than approximated.

```{jupyter-execute}
truth = 0.02 * t + 2 * np.sin(t / 25)
true_slope = 0.02 + 2 * np.cos(t / 25) / 25
df = pd.DataFrame({"value": truth + rng.normal(0, 0.5, n)}, index=index)

gp = gp_trend(df, kernel="rbf", se=True)
gp[["derivative_value", "derivative_se", "se_method"]].head()
```

```{jupyter-execute}
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axes[0].plot(t, df["value"], ".", color="0.6", ms=3, label="observed")
axes[0].plot(t, gp["smoothed_value"], lw=2, label="GP posterior mean")
axes[0].set_ylabel("value")
axes[0].legend(frameon=False)

axes[1].fill_between(
    t, gp["derivative_ci_lower"], gp["derivative_ci_upper"],
    alpha=0.25, label="95% interval",
)
axes[1].plot(t, gp["derivative_value"], lw=2, label="estimated slope")
axes[1].plot(t, true_slope, "--", color="0.3", lw=1.5, label="true slope")
axes[1].axhline(0, color="0.4", lw=1)
axes[1].set_ylabel("slope per day")
axes[1].legend(frameon=False)
plt.tight_layout()
```

The Matérn kernels are less smooth than the squared exponential, and that is not
a detail you can ignore: a process with smoothness ν has only ⌊ν⌋ derivatives.
Asking for one it does not have raises rather than returning a number.

```{jupyter-execute}
for kernel in ("rbf", "matern32", "matern52"):
    result = gp_trend(df, kernel=kernel, se=True)
    print(f"{kernel:9s} median se = {result['derivative_se'].median():.4f}")

try:
    gp_trend(df, kernel="matern32", derivative_order=2, se=True)
except ValueError as exc:
    print(f"\nmatern32, order 2 -> {exc}")
```

## State-space models

In a local linear trend model the slope is a state, so its uncertainty is a
diagonal entry of the smoother covariance — no extra machinery needed.

```{jupyter-execute}
regime = np.concatenate([
    np.zeros(50), np.linspace(0, 3, 50), np.full(50, 3.0),
    np.linspace(3, 1, 50),
])
regime_df = pd.DataFrame({"value": regime + rng.normal(0, 0.25, n)}, index=index)

kalman = kalman_trend(regime_df, se=True)

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(t, regime_df["value"], ".", color="0.6", ms=3, label="observed")
axes[0].plot(t, kalman["smoothed_value"], lw=2, label="smoothed level")
axes[0].plot(t, regime, "--", color="0.3", lw=1.5, label="true level")
axes[0].legend(frameon=False)

axes[1].fill_between(
    t, kalman["derivative_ci_lower"], kalman["derivative_ci_upper"], alpha=0.25
)
axes[1].plot(t, kalman["derivative_value"], lw=2)
axes[1].axhline(0, color="0.4", lw=1)
axes[1].set_ylabel("slope per day")
plt.tight_layout()
```

## Seasonality is preprocessing

`deseasonalize` returns a DataFrame, so it composes with every estimator rather
than needing one of its own.

```{jupyter-execute}
seasonal_df = pd.DataFrame(
    {"value": 0.03 * t + 4 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.4, n)},
    index=index,
)

parts = deseasonalize(seasonal_df)
print("detected period:", parts["period"].iloc[0])
print("method:", parts["decomposition_method"].iloc[0])

fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
axes[0].plot(t, parts["value"], color="0.6", lw=1)
axes[0].set_ylabel("observed")
axes[1].plot(t, parts["seasonal_component"], lw=1)
axes[1].set_ylabel("seasonal")
axes[2].plot(t, parts["deseasonalized"], lw=1)
axes[2].set_ylabel("adjusted")
plt.tight_layout()
```

Any smoother can then be applied to the adjusted series:

```{jupyter-execute}
adjusted = trend_with_deseasonalization(
    seasonal_df, SavitzkyGolay(window_length=21), se=True
)
print(adjusted[["derivative_value", "derivative_se", "se_method"]].iloc[100])
```

```{admonition} What that interval covers
:class: note

The seasonal component was estimated from the same data as the trend, so
treating it as known would make the interval about 10% too narrow. Asking for a
standard error here therefore bootstraps the whole pipeline -- decomposition and
trend fit together -- which is why `se_method` reads `pipeline_bootstrap`.
```

## Multi-scale analysis

A single bandwidth is a single opinion about what counts as signal. SiZer sweeps
the bandwidth and reports, at each scale and position, whether the slope is
distinguishable from zero. Features that persist across scales are real.

```{jupyter-execute}
multiscale_df = pd.DataFrame(
    {"value": np.sin(t / 30) + 0.3 * np.sin(t / 5) + rng.normal(0, 0.3, n)},
    index=index,
)

sizer_map = sizer_analysis(multiscale_df, n_scales=14)
figure = sizer_map.plot(figsize=(10, 5))
```

Red is significantly increasing, blue significantly decreasing, pale neither.
Because SiZer asks the smoother for its uncertainty rather than computing its
own, the map is exactly as calibrated as the estimator underneath.

```{jupyter-execute}
regions = sizer_map.significant_regions(min_persistence=4)
for direction, spans in regions.items():
    print(f"{direction}: {[(round(a), round(b)) for a, b in spans][:4]}")
```

## Comparing methods on one series

```{jupyter-execute}
methods = {
    "Savitzky-Golay": SavitzkyGolay(window_length=21, polyorder=3),
    "local polynomial": LocalPolynomial(bandwidth=0.15, degree=2),
    "Gaussian process": GaussianProcess(n_restarts=0),
    "state space": StateSpace(),
}

fig, ax = plt.subplots(figsize=(10, 5))
rows = []
for label, smoother in methods.items():
    result = estimate(smoother, df, se=True)
    ax.plot(t, result.derivative, lw=1.6, label=label)
    rows.append({
        "method": label,
        "se_method": result.provenance.se_method,
        "median se": float(np.nanmedian(result.se)),
        "share significant": float(result.significant.mean()),
    })

ax.plot(t, true_slope, "--", color="0.3", lw=2, label="truth")
ax.axhline(0, color="0.4", lw=1)
ax.set_ylabel("slope per day")
ax.legend(frameon=False, ncol=2)
plt.tight_layout()

pd.DataFrame(rows)
```

The `se_method` column is the point. `operator` means the estimator is a fixed
linear map of the data and its variance is exact; `native` means it is a
probability model that already knew its own posterior; `bootstrap` means neither
applied and the sampling distribution had to be simulated.

## When the noise is correlated

The default assumes independent errors. Under real autocorrelation that
understates the uncertainty badly — measured against Monte Carlo truth at
φ=0.7, the reported standard errors come out around a quarter of their true
size, and nominal 95% intervals cover about 40% of the time.

```{jupyter-execute}
phi, sigma = 0.75, 0.5
noise = np.empty(n)
noise[0] = rng.normal(0, sigma)
for i in range(1, n):
    noise[i] = phi * noise[i - 1] + rng.normal(0, sigma * np.sqrt(1 - phi**2))

correlated = pd.DataFrame({"value": 0.01 * t + noise}, index=index)

independent = sgolay_trend(correlated, window_length=21, se=True, noise="iid")
modeled = sgolay_trend(correlated, window_length=21, se=True, noise="ar1")

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.fill_between(
    t, modeled["derivative_ci_lower"], modeled["derivative_ci_upper"],
    alpha=0.25, label="95% interval, AR(1) noise",
)
ax.plot(t, independent["derivative_ci_upper"], lw=1.2, color="C1",
        label="same interval assuming independence")
ax.plot(t, independent["derivative_ci_lower"], lw=1.2, color="C1")
ax.plot(t, modeled["derivative_value"], lw=1.8, color="C0", label="slope")
ax.axhline(0, color="0.4", lw=1)
ax.legend(frameon=False)
plt.tight_layout()

print("points called significant, assuming independence:",
      int(independent["significant_trend"].sum()))
print("points called significant, modeling AR(1):      ",
      int(modeled["significant_trend"].sum()))
```

The series has no trend beyond a slope of 0.01 per day. The gap between those
two counts is how many apparent discoveries the independence assumption
manufactures out of wandering noise.
