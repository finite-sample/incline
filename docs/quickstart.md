# Quick Start Guide

This guide will help you get started with incline quickly.

## Basic Usage

incline provides four main functions for trend estimation:

1. **naive_trend**: Simple forward/backward difference
2. **spline_trend**: Spline interpolation based trend
3. **sgolay_trend**: Savitzky-Golay filter based trend  
4. **trending**: Aggregate trends across multiple time series

## Simple Example

```python
import pandas as pd
from incline import naive_trend, spline_trend, sgolay_trend

# Create sample time series data with proper datetime index
data = {
    'timestamp': pd.date_range('2020-01-01', periods=10, freq='D'),
    'value': [1, 3, 2, 5, 8, 7, 12, 15, 14, 18]
}
df = pd.DataFrame(data)
df = df.set_index('timestamp')

# Method 1: Naive trend estimation (with automatic time scaling)
naive_result = naive_trend(df)
print("Naive trend:")
print(naive_result[['value', 'derivative_value']].head())

# Method 2: Spline-based trend estimation (auto-selects smoothing parameter)
spline_result = spline_trend(df, function_order=3, derivative_order=1)
print("\nSpline trend:")
print(spline_result[['value', 'smoothed_value', 'derivative_value']].head())

# Method 3: Savitzky-Golay trend estimation (with edge effect marking)
sgolay_result = sgolay_trend(df, window_length=5, function_order=2)
print("\nSavitzky-Golay trend:")
print(sgolay_result[['value', 'smoothed_value', 'derivative_value', 'edge_region']].head())
```

## Understanding the Output

All trend functions return a DataFrame with these columns:

- **Original data columns**: Your input data is preserved
- **smoothed_value**: The smoothed version of your time series (None for naive method)
- **derivative_value**: The estimated derivative/trend at each point (properly scaled by time units)
- **derivative_method**: Which method was used ('naive', 'spline', or 'sgolay')
- **function_order**: The polynomial order used for smoothing
- **derivative_order**: The order of derivative calculated (1 for slope, 2 for acceleration)
- **smoothing_parameter**: The smoothing parameter used (for splines)
- **edge_region**: Boolean flag marking less reliable estimates near boundaries (Savitzky-Golay only)

## Working with Multiple Time Series

Use the `trending` function to analyze multiple time series:

```python
from incline import trending

# Process multiple time series
results = []
for i, ts_data in enumerate([df1, df2, df3]):  # Your time series list
    result = spline_trend(ts_data)
    result['id'] = f'series_{i}'  # Add identifier
    results.append(result)

# Find which series are trending most strongly
trend_summary = trending(
    results, 
    derivative_order=1,  # First derivative (slope)
    max_or_avg='max',    # Maximum trend in time window
    k=3                  # Look at last 3 time points
)

print("Series ranked by maximum trend:")
print(trend_summary.sort_values('max_or_avg', ascending=False))
```

## Parameter Tuning

**For spline_trend**:

- `function_order`: Higher values = smoother curves (default: 3)
- `s`: Smoothing factor, higher = more smoothing (default: 3)
- `derivative_order`: 1 for slope, 2 for acceleration (default: 1)

**For sgolay_trend**:

- `window_length`: Size of smoothing window, must be odd (default: 15)
- `function_order`: Polynomial order for fitting (default: 3)
- `derivative_order`: 1 for slope, 2 for acceleration (default: 1)

## Advanced Features

### Uncertainty Quantification

Get confidence intervals for derivative estimates:

```python
from incline import bootstrap_derivative_ci

# Get 95% confidence intervals using block bootstrap
result_with_ci = bootstrap_derivative_ci(
    df, 
    method='spline',
    n_bootstrap=100,
    confidence_level=0.95
)

# Check which trends are statistically significant
significant_trends = result_with_ci[result_with_ci['significant_trend']]
print(f"Found {len(significant_trends)} significant trend points")
```

### Automatic Parameter Selection

Use cross-validation to select optimal smoothing parameters:

```python
from incline import select_smoothing_parameter_cv

# Find optimal smoothing parameter for spline
best_s, cv_results = select_smoothing_parameter_cv(
    df, 
    method='spline',
    param_name='s',
    cv_folds=5
)

print(f"Optimal smoothing parameter: {best_s}")

# Use the optimal parameter
optimal_result = spline_trend(df, s=best_s)
```

### Time Vector Support

Work with irregular time series or explicit time columns:

```python
# With explicit time column
df_with_time = pd.DataFrame({
    'time_hours': [0, 1.5, 3.2, 5.1, 7.8],  # Irregular spacing
    'temperature': [20.1, 21.3, 19.8, 22.5, 24.1]
})

# Specify time column for proper scaling
result = spline_trend(df_with_time, 
                     column_value='temperature',
                     time_column='time_hours')
```

### Edge Effect Handling

Be aware of less reliable estimates near boundaries:

```python
result = sgolay_trend(df, window_length=15)

# Filter out edge regions for more reliable analysis
reliable_points = result[~result['edge_region']]
print(f"Reliable estimates: {len(reliable_points)}/{len(result)}")
```

## Next Steps

- Check out the [examples](examples.md) for real-world use cases
- See the [API reference](api.md) for detailed function documentation  
- Read about [limitations](limitations.md) to understand when and how to use each method
- Look at the example notebook in the repository for stock market analysis