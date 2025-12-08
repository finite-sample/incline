# Examples

This page contains detailed examples of using incline for various time series analysis tasks.

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

For more examples, check out the Jupyter notebook in the `examples/` directory that demonstrates real stock market analysis using incline.