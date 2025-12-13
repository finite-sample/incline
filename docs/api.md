# API Reference

This page contains the API reference for all public functions in incline.

## Core Functions

The incline package provides the following main functions for trend estimation:

- `naive_trend`: Simple forward/backward difference method
- `spline_trend`: Spline interpolation based trend estimation  
- `sgolay_trend`: Savitzky-Golay filter based trend estimation
- `trending`: Aggregate trends across multiple time series

## Advanced Functions

Additional functions for advanced analysis:

- `bootstrap_derivative_ci`: Calculate confidence intervals using bootstrap
- `select_smoothing_parameter_cv`: Select optimal smoothing parameters via cross-validation

## Utility Functions

Helper functions:

- `compute_time_deltas`: Compute time differences in a time series

For detailed documentation of each function, please refer to the source code or use Python's built-in help system:

```python
import incline
help(incline.spline_trend)
```