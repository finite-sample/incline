# API reference

Generated from the source, so it cannot drift from what the package actually
exports.

## Estimating a trend

The functional surface. Each builds a smoother, fits it, and returns a
DataFrame carrying the columns described in [Uncertainty](uncertainty.md).

```{eval-rst}
.. currentmodule:: incline

.. autosummary::
   :toctree: generated
   :nosignatures:

   naive_trend
   sgolay_trend
   spline_trend
   pspline_trend
   loess_trend
   local_polynomial_trend
   l1_trend_filter
   gp_trend
   kalman_trend
   estimate_trend
   select_trend_method
   estimate
```

## Smoothers

The objects behind those functions. Use these directly when you want to hand
one to `SiZer`, to `trend_with_deseasonalization`, or to `estimate`.

Whether a smoother is *linear* decides how its uncertainty is computed: linear
smoothers get an exact operator variance, the rest are bootstrapped.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Smoother
   SavitzkyGolay
   NaiveDifference
   LocalPolynomial
   Loess
   PenalizedSpline
   InterpolatingSpline
   L1TrendFilter
   GaussianProcess
   StateSpace
   build
```

`incline.SMOOTHERS` is the registry those names are looked up in: a mapping from
name to class. `build(name, **kwargs)` constructs from it, and
`estimate_trend(method=...)` dispatches through it, so a newly registered
smoother is reachable everywhere without editing a dispatch table.

## Noise models

What `noise=` accepts. The default assumes independence; under real
autocorrelation that understates the uncertainty substantially.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   NoiseModel
   IID
   AR1
   Heteroskedastic
   Given
   local_sigma
```

## Results

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   TrendEstimate
   TimeAxis
```

## Multi-scale analysis

Which features survive being looked at from every smoothing scale.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   SiZer
   SiZerMap
   sizer_analysis
   trend_with_sizer
```

## Seasonality

Decomposition is preprocessing: `deseasonalize` returns a frame, which any
estimator then accepts.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   deseasonalize
   detect_seasonality
   stl_decompose
   moving_average_decompose
   trend_with_deseasonalization
   Seasonality
```

## Ranking many series

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   trending
```

## Simulation

Synthetic series with known derivatives, used by the package's own calibration
tests and available for yours.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   generate_time_series
   standard_test_functions
```
