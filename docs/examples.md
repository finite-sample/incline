# Examples

This page contains detailed examples of using incline for various time series analysis tasks.

:::{note}
**New: Executable Examples Available!**  
For comprehensive, executable examples with live plots and analysis, see our [Executable Examples](examples_executable/index.md) section. These examples run automatically during documentation build and showcase the full capabilities of incline with real outputs.
:::

## Static Code Examples

The examples below provide code snippets for common use cases:

## Stock Market Trend Analysis

This example shows how to analyze trends in stock price data:

```python
import pandas as pd
import matplotlib.pyplot as plt
from incline import spline_trend, sgolay_trend, naive_trend

# Load stock data from examples/data directory
df = pd.read_csv('examples/data/AAPL.csv', parse_dates=['Date'], index_col='Date')

# Focus on closing prices
price_data = df[['Close']].rename(columns={'Close': 'value'})

# Calculate trends using different methods
naive_result = naive_trend(price_data)
spline_result = spline_trend(price_data, function_order=3, s=10)
sgolay_result = sgolay_trend(price_data, window_length=21, function_order=3)

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Original data and smoothed versions
ax1.plot(price_data.index, price_data['value'], 'k-', alpha=0.7, label='Original')
ax1.plot(spline_result.index, spline_result['smoothed_value'], 'r-', label='Spline')
ax1.plot(sgolay_result.index, sgolay_result['smoothed_value'], 'b-', label='Savitzky-Golay')
ax1.set_ylabel('Price ($)')
ax1.set_title('Stock Price Smoothing')
ax1.legend()

# Derivatives (trends)
ax2.plot(naive_result.index, naive_result['derivative_value'], 'g-', label='Naive')
ax2.plot(spline_result.index, spline_result['derivative_value'], 'r-', label='Spline')
ax2.plot(sgolay_result.index, sgolay_result['derivative_value'], 'b-', label='Savitzky-Golay')
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax2.set_ylabel('Trend ($/day)')
ax2.set_xlabel('Date')
ax2.set_title('Price Trends')
ax2.legend()

plt.tight_layout()
plt.show()
```

## Comparing Multiple Stocks

Analyze and rank multiple stocks by their recent trends:

```python
from incline import trending

# List of stock symbols and their data
stocks = ['AAPL', 'GOOG', 'MSFT']
results = []

for symbol in stocks:
    # Load and process each stock
    df = pd.read_csv(f'examples/data/{symbol}.csv', parse_dates=['Date'], index_col='Date')
    price_data = df[['Close']].rename(columns={'Close': 'value'})
    
    # Calculate spline trend
    trend_result = spline_trend(price_data, function_order=3, s=10)
    trend_result['id'] = symbol
    results.append(trend_result)

# Rank stocks by maximum trend in last 5 days
trend_ranking = trending(
    results,
    derivative_order=1,
    max_or_avg='max',
    k=5
)

print("Stocks ranked by strongest upward trend (last 5 days):")
print(trend_ranking.sort_values('max_or_avg', ascending=False))
```

## Seasonal Data Analysis

Analyze seasonal patterns with trend estimation:

```python
import numpy as np

# Generate seasonal data with trend
dates = pd.date_range('2020-01-01', periods=365, freq='D')
seasonal_component = 10 * np.sin(2 * np.pi * np.arange(365) / 365)
trend_component = 0.02 * np.arange(365)  # Linear trend
noise = np.random.normal(0, 2, 365)

seasonal_data = pd.DataFrame({
    'value': seasonal_component + trend_component + noise
}, index=dates)

# Extract trend using different window sizes
short_window = sgolay_trend(seasonal_data, window_length=15, function_order=2)
long_window = sgolay_trend(seasonal_data, window_length=91, function_order=2)

# Plot comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(seasonal_data.index, seasonal_data['value'], 'k-', alpha=0.5, label='Original')
ax1.plot(short_window.index, short_window['smoothed_value'], 'r-', label='Short window (15 days)')
ax1.plot(long_window.index, long_window['smoothed_value'], 'b-', label='Long window (91 days)')
ax1.set_ylabel('Value')
ax1.set_title('Seasonal Data with Different Smoothing Windows')
ax1.legend()

ax2.plot(short_window.index, short_window['derivative_value'], 'r-', label='Short window trend')
ax2.plot(long_window.index, long_window['derivative_value'], 'b-', label='Long window trend')
ax2.axhline(y=0.02, color='g', linestyle='--', label='True trend')
ax2.set_ylabel('Trend')
ax2.set_xlabel('Date')
ax2.set_title('Estimated Trends')
ax2.legend()

plt.tight_layout()
plt.show()
```

## Acceleration Analysis

Analyze acceleration (second derivative) to detect trend changes:

```python
# Use stock data or any time series
df = pd.read_csv('examples/data/AAPL.csv', parse_dates=['Date'], index_col='Date')
price_data = df[['Close']].rename(columns={'Close': 'value'})

# Calculate first and second derivatives
first_deriv = spline_trend(price_data, derivative_order=1, s=5)
second_deriv = spline_trend(price_data, derivative_order=2, s=5)

# Find points of high acceleration (trend changes)
acceleration_threshold = np.std(second_deriv['derivative_value']) * 2
high_accel_points = second_deriv[
    np.abs(second_deriv['derivative_value']) > acceleration_threshold
]

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

# Price
ax1.plot(price_data.index, price_data['value'], 'k-')
ax1.scatter(high_accel_points.index, 
           [price_data.loc[idx, 'value'] for idx in high_accel_points.index],
           color='red', s=50, label='High acceleration points')
ax1.set_ylabel('Price')
ax1.set_title('Stock Price with Acceleration Events')
ax1.legend()

# First derivative (velocity/trend)
ax2.plot(first_deriv.index, first_deriv['derivative_value'], 'b-')
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax2.set_ylabel('Trend (1st derivative)')
ax2.set_title('Price Trend')

# Second derivative (acceleration)
ax3.plot(second_deriv.index, second_deriv['derivative_value'], 'r-')
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax3.axhline(y=acceleration_threshold, color='r', linestyle='--', alpha=0.5)
ax3.axhline(y=-acceleration_threshold, color='r', linestyle='--', alpha=0.5)
ax3.set_ylabel('Acceleration (2nd derivative)')
ax3.set_xlabel('Date')
ax3.set_title('Price Acceleration')

plt.tight_layout()
plt.show()

print(f"Found {len(high_accel_points)} high acceleration events")
print("Dates of significant trend changes:")
for date in high_accel_points.index:
    print(f"  {date.strftime('%Y-%m-%d')}")
```

## Parameter Sensitivity Analysis

Understand how different parameters affect your results:

```python
# Test different smoothing parameters
smoothing_factors = [1, 3, 10, 30, 100]

fig, axes = plt.subplots(len(smoothing_factors), 1, figsize=(12, 15))

for i, s_factor in enumerate(smoothing_factors):
    result = spline_trend(price_data, function_order=3, s=s_factor)
    
    axes[i].plot(price_data.index, price_data['value'], 'k-', alpha=0.3, label='Original')
    axes[i].plot(result.index, result['smoothed_value'], 'r-', linewidth=2, label='Smoothed')
    axes[i].set_title(f'Smoothing factor s = {s_factor}')
    axes[i].set_ylabel('Price')
    axes[i].legend()

axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.show()
```

## Advanced Methods

### Gaussian Process Trend Analysis

Gaussian Processes provide principled Bayesian trend estimation with full uncertainty quantification:

```python
from incline import gp_trend

# Load your data
df = pd.read_csv('examples/data/AAPL.csv', parse_dates=['Date'], index_col='Date')
price_data = df[['Close']].rename(columns={'Close': 'value'})

# Gaussian Process with different kernels
rbf_result = gp_trend(price_data, kernel_type='rbf', confidence_level=0.95)
matern_result = gp_trend(price_data, kernel_type='matern52', confidence_level=0.95)

# Plot with uncertainty bands
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# RBF kernel results
ax1.plot(price_data.index, price_data['value'], 'k-', alpha=0.3, label='Original')
ax1.plot(rbf_result.index, rbf_result['smoothed_value'], 'r-', label='GP Mean (RBF)')
ax1.fill_between(rbf_result.index, 
                 rbf_result['derivative_ci_lower'], 
                 rbf_result['derivative_ci_upper'],
                 alpha=0.2, color='red', label='95% CI')
ax1.set_title('Gaussian Process with RBF Kernel')
ax1.legend()

# Show trend significance
significant_trends = rbf_result[rbf_result['significant_trend']]
ax2.plot(rbf_result.index, rbf_result['derivative_value'], 'b-', label='Trend')
ax2.fill_between(rbf_result.index, 
                 rbf_result['derivative_ci_lower'], 
                 rbf_result['derivative_ci_upper'],
                 alpha=0.2, color='blue')
ax2.scatter(significant_trends.index, 
           significant_trends['derivative_value'],
           color='red', s=20, label='Significant trends')
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax2.set_title('Trend with 95% Confidence Intervals')
ax2.set_ylabel('Trend ($/day)')
ax2.legend()

plt.tight_layout()
plt.show()

print(f"Kernel hyperparameters: {rbf_result['kernel_k1__k1__constant_value'].iloc[0]:.3f}")
print(f"Significant trend points: {len(significant_trends)}/{len(rbf_result)}")
```

### Kalman Filter State-Space Models

Use Kalman filtering for principled trend estimation with natural uncertainty quantification:

```python
from incline import kalman_trend, adaptive_kalman_trend

# Basic local linear trend model
kalman_result = kalman_trend(price_data, model_type='local_linear', confidence_level=0.95)

# Adaptive Kalman with time-varying parameters
adaptive_result = adaptive_kalman_trend(price_data, adaptation_window=30)

# Compare results
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

# Smoothed values
ax1.plot(price_data.index, price_data['value'], 'k-', alpha=0.5, label='Original')
ax1.plot(kalman_result.index, kalman_result['smoothed_value'], 'b-', label='Kalman')
ax1.plot(adaptive_result.index, adaptive_result['smoothed_value'], 'r-', label='Adaptive Kalman')
ax1.set_title('State-Space Trend Estimation')
ax1.legend()

# Trends with uncertainty
ax2.plot(kalman_result.index, kalman_result['derivative_value'], 'b-', label='Kalman trend')
ax2.fill_between(kalman_result.index,
                 kalman_result['derivative_ci_lower'],
                 kalman_result['derivative_ci_upper'],
                 alpha=0.2, color='blue')
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax2.set_title('Trend Estimates with Kalman Filter Uncertainty')
ax2.set_ylabel('Trend')
ax2.legend()

# Model parameters over time
ax3.plot(kalman_result.index, kalman_result['fitted_obs_variance'], label='Observation Variance')
ax3.plot(kalman_result.index, kalman_result['fitted_level_variance'], label='Level Variance')
ax3.plot(kalman_result.index, kalman_result['fitted_slope_variance'], label='Slope Variance')
ax3.set_title('Estimated Model Parameters')
ax3.set_ylabel('Variance')
ax3.set_yscale('log')
ax3.legend()

plt.tight_layout()
plt.show()

print(f"Significant trends (Kalman): {kalman_result['significant_trend'].sum()}")
print(f"Model log-likelihood: {kalman_result['fitted_obs_variance'].iloc[0]:.3f}")
```

### Seasonal Decomposition and Trend Analysis

Handle seasonal data with STL decomposition before trend analysis:

```python
from incline import stl_decompose, trend_with_deseasonalization, detect_seasonality

# Generate seasonal data for demonstration
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=365*2, freq='D')
trend = 0.05 * np.arange(len(dates))
seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
noise = np.random.normal(0, 2, len(dates))

seasonal_data = pd.DataFrame({
    'value': 100 + trend + seasonal + noise
}, index=dates)

# Detect seasonality first
seasonality_info = detect_seasonality(seasonal_data)
print(f"Seasonality detected: {seasonality_info['seasonal']}")
print(f"Estimated period: {seasonality_info['period']} days")
print(f"Seasonality strength: {seasonality_info['strength']:.3f}")

# STL decomposition
stl_result = stl_decompose(seasonal_data, period=365)

# Trend analysis on deseasonalized data
detrended_result = trend_with_deseasonalization(
    seasonal_data, 
    trend_method='spline',
    decomposition_method='stl'
)

# Plot decomposition
fig, axes = plt.subplots(4, 1, figsize=(12, 12))

axes[0].plot(seasonal_data.index, seasonal_data['value'])
axes[0].set_title('Original Data')
axes[0].set_ylabel('Value')

axes[1].plot(stl_result.index, stl_result['trend_component'])
axes[1].set_title('Trend Component')
axes[1].set_ylabel('Trend')

axes[2].plot(stl_result.index, stl_result['seasonal_component'])
axes[2].set_title('Seasonal Component')
axes[2].set_ylabel('Seasonal')

axes[3].plot(detrended_result.index, detrended_result['trend_derivative_value'])
axes[3].axhline(y=0.05, color='r', linestyle='--', label='True trend (0.05)')
axes[3].axhline(y=0, color='k', linestyle='--', alpha=0.5)
axes[3].set_title('Estimated Trend After Deseasonalization')
axes[3].set_ylabel('Trend Rate')
axes[3].set_xlabel('Date')
axes[3].legend()

plt.tight_layout()
plt.show()
```

### Multiscale Analysis with SiZer

SiZer (SIgnificance of ZERo crossings) shows trend significance across multiple smoothing scales:

```python
from incline import sizer_analysis, trend_with_sizer

# Perform SiZer analysis
sizer_result = sizer_analysis(
    price_data, 
    n_bandwidths=20, 
    bandwidth_range=(0.02, 0.3),
    method='loess'
)

# Plot SiZer map
sizer_fig = sizer_result.plot_sizer_map(
    figsize=(12, 8),
    title='SiZer Significance Map for Stock Prices'
)

# Find persistent significant features
features = sizer_result.find_significant_features(min_persistence=3)
print(f"Persistent increasing regions: {len(features['increasing'])}")
print(f"Persistent decreasing regions: {len(features['decreasing'])}")

# Combined trend analysis with SiZer
combined_result = trend_with_sizer(
    price_data,
    trend_method='loess',
    sizer_method='loess'
)

# Plot trends with SiZer significance
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Main trend
ax1.plot(price_data.index, price_data['value'], 'k-', alpha=0.3, label='Original')
ax1.plot(combined_result.index, combined_result['smoothed_value'], 'b-', label='LOESS Trend')
ax1.set_title('Trend Estimation')
ax1.legend()

# Trends colored by SiZer significance
increasing = combined_result[combined_result['sizer_increasing']]
decreasing = combined_result[combined_result['sizer_decreasing']]
insignificant = combined_result[combined_result['sizer_insignificant']]

ax2.scatter(increasing.index, increasing['derivative_value'], 
           color='red', s=10, label='Significantly Increasing', alpha=0.7)
ax2.scatter(decreasing.index, decreasing['derivative_value'],
           color='blue', s=10, label='Significantly Decreasing', alpha=0.7)
ax2.scatter(insignificant.index, insignificant['derivative_value'],
           color='gray', s=5, label='Insignificant', alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax2.set_title('Trend Significance from SiZer Analysis')
ax2.set_ylabel('Trend')
ax2.set_xlabel('Date')
ax2.legend()

plt.tight_layout()
plt.show()
```

### Robust Methods for Noisy Data

Use LOESS and L1 trend filtering for outlier-resistant analysis:

```python
from incline import loess_trend, l1_trend_filter

# Add some outliers to demonstrate robustness
outlier_data = price_data.copy()
outlier_indices = np.random.choice(len(outlier_data), size=10, replace=False)
outlier_data.iloc[outlier_indices] *= 1.5  # 50% price spikes

# Compare robust vs non-robust methods
spline_result = spline_trend(outlier_data, s=10)
loess_result = loess_trend(outlier_data, frac=0.2, robust=True)
l1_result = l1_trend_filter(outlier_data, lambda_param=1.0)

# Plot comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Smoothed values
ax1.plot(outlier_data.index, outlier_data['value'], 'k-', alpha=0.5, label='Data with outliers')
ax1.plot(spline_result.index, spline_result['smoothed_value'], 
         'g-', label='Spline (non-robust)', alpha=0.8)
ax1.plot(loess_result.index, loess_result['smoothed_value'], 
         'r-', label='Robust LOESS', linewidth=2)
ax1.plot(l1_result.index, l1_result['smoothed_value'], 
         'b-', label='L1 Trend Filter', linewidth=2)
ax1.set_title('Robust vs Non-Robust Smoothing with Outliers')
ax1.legend()

# Derivatives
ax2.plot(spline_result.index, spline_result['derivative_value'], 
         'g-', label='Spline trends', alpha=0.8)
ax2.plot(loess_result.index, loess_result['derivative_value'], 
         'r-', label='LOESS trends', linewidth=2)
ax2.plot(l1_result.index, l1_result['derivative_value'], 
         'b-', label='L1 trends', linewidth=2)

# Mark changepoints from L1 filter
changepoints = l1_result[l1_result['changepoint']]
ax2.scatter(changepoints.index, changepoints['derivative_value'],
           color='blue', s=50, marker='x', label='Detected changepoints')

ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax2.set_title('Trend Estimates - Robust Methods Handle Outliers Better')
ax2.set_ylabel('Trend')
ax2.set_xlabel('Date')
ax2.legend()

plt.tight_layout()
plt.show()

print(f"Detected changepoints: {changepoints.index.tolist()}")
print(f"LOESS bandwidth used: {loess_result['bandwidth'].iloc[0]:.3f}")
```

### Method Comparison and Selection

Choose the right method for your data characteristics:

```python
from incline import select_smoothing_parameter_cv

# Generate different types of test data
np.random.seed(42)
n_points = 100
x = np.linspace(0, 10, n_points)

# Different signal types
linear_trend = 2 * x + np.random.normal(0, 1, n_points)
nonlinear_smooth = np.sin(x) * x + np.random.normal(0, 0.5, n_points)
noisy_with_outliers = x + np.random.normal(0, 2, n_points)
noisy_with_outliers[20:25] += 10  # Add outliers
piecewise = np.where(x < 5, 2*x, 2*x + 5) + np.random.normal(0, 0.5, n_points)

datasets = {
    'Linear Trend': linear_trend,
    'Nonlinear Smooth': nonlinear_smooth,
    'Noisy with Outliers': noisy_with_outliers,
    'Piecewise Linear': piecewise
}

# Test all methods on each dataset
methods = ['naive', 'spline', 'sgolay']
if loess_trend:  # Check if advanced methods available
    methods.extend(['loess'])

fig, axes = plt.subplots(len(datasets), 3, figsize=(15, 12))

for i, (dataset_name, data) in enumerate(datasets.items()):
    test_df = pd.DataFrame({'value': data}, index=pd.date_range('2020-01-01', periods=n_points))
    
    # Apply each method
    naive_res = naive_trend(test_df)
    spline_res = spline_trend(test_df, s=5)
    sgolay_res = sgolay_trend(test_df, window_length=11, function_order=3)
    
    # Original data
    axes[i, 0].plot(test_df.index, test_df['value'], 'k-', alpha=0.5, label='Original')
    axes[i, 0].plot(spline_res.index, spline_res['smoothed_value'], 'r-', label='Spline')
    axes[i, 0].plot(sgolay_res.index, sgolay_res['smoothed_value'], 'b-', label='S-G')
    axes[i, 0].set_title(f'{dataset_name} - Smoothing')
    axes[i, 0].legend()
    
    # Derivatives
    axes[i, 1].plot(naive_res.index, naive_res['derivative_value'], 'g-', label='Naive')
    axes[i, 1].plot(spline_res.index, spline_res['derivative_value'], 'r-', label='Spline')
    axes[i, 1].plot(sgolay_res.index, sgolay_res['derivative_value'], 'b-', label='S-G')
    axes[i, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[i, 1].set_title('Derivatives')
    axes[i, 1].legend()
    
    # Method performance (MSE from true derivative where known)
    if dataset_name == 'Linear Trend':
        true_derivative = 2.0
        naive_mse = np.mean((naive_res['derivative_value'] - true_derivative)**2)
        spline_mse = np.mean((spline_res['derivative_value'] - true_derivative)**2)
        sgolay_mse = np.mean((sgolay_res['derivative_value'] - true_derivative)**2)
        
        axes[i, 2].bar(['Naive', 'Spline', 'S-G'], [naive_mse, spline_mse, sgolay_mse])
        axes[i, 2].set_title('MSE from True Derivative')
        axes[i, 2].set_ylabel('MSE')
    else:
        axes[i, 2].text(0.5, 0.5, 'No ground truth\navailable', 
                       ha='center', va='center', transform=axes[i, 2].transAxes)
        axes[i, 2].set_title('Performance')

plt.tight_layout()
plt.show()

# Automatic parameter selection example
print("Cross-validation for optimal smoothing parameter:")
try:
    best_s, cv_results = select_smoothing_parameter_cv(
        test_df, 
        method='spline', 
        param_name='s',
        cv_folds=5
    )
    print(f"Optimal smoothing parameter: {best_s:.2f}")
except:
    print("Cross-validation requires more data points")
```

For more examples, check out the Jupyter notebook in the `examples/` directory that demonstrates real stock market analysis using incline.