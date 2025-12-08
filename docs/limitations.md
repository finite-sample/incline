# Mathematical Foundations and Methods

This document provides comprehensive information about the mathematical foundations, methods, and best practices for the incline package.

## Core Mathematical Problem

The package addresses the problem of estimating the instantaneous rate of change (derivative) at a point $t_0$ in a noisy time series:

$$y_i = f(t_i) + \epsilon_i$$

where $f$ is the underlying smooth function and $\epsilon_i$ is noise. We want to estimate $f'(t_0)$ along with confidence intervals.

## Available Methods and Capabilities

The incline package provides a comprehensive suite of trend estimation methods:

### 1. Basic Methods
   
- **Naive (Central Differences):** Simple finite differences for clean data
- **Savitzky-Golay Filtering:** Local polynomial fitting with uniform sampling
- **Spline Interpolation:** Smooth curve fitting with automatic time scaling
   
### 2. Advanced Nonparametric Methods
   
- **LOESS/LOWESS:** Locally weighted scatterplot smoothing with robust options
- **Local Polynomial Regression:** Kernel-weighted polynomial fits
- **L1 Trend Filtering:** Piecewise linear trends with changepoint detection
   
### 3. Bayesian and State-Space Methods
   
- **Gaussian Process Regression:** Full posterior distribution with principled uncertainty
- **Kalman Filtering:** Local linear trend models with adaptive parameters
- **Structural Time Series:** Seasonal decomposition with state-space modeling
   
### 4. Multiscale Analysis
   
- **SiZer Maps:** Significance of trends across multiple smoothing scales
- **Adaptive Methods:** Time-varying parameters for non-stationary series
   
### 5. Seasonal and Robust Methods
   
- **STL Decomposition:** Seasonal-trend decomposition using Loess
- **Robust Statistics:** Outlier-resistant trend ranking and aggregation
- **Bootstrap Confidence Intervals:** Non-parametric uncertainty quantification

## Key Features and Improvements

### 1. Automatic Time Scaling ✅
   
All methods now handle:
   
- DateTime indices with proper scaling
- Irregular sampling (where applicable)
- Custom time columns
- Automatic detection of time units
   
```python
# Automatic time handling
result = spline_trend(df)  # Uses datetime index
result = gp_trend(df, time_column='timestamp')  # Custom time column
```

### 2. Parameter Selection ✅
   
- **Cross-validation:** `select_smoothing_parameter_cv()`
- **Automatic selection:** Built into GP and Kalman methods  
- **Adaptive methods:** `adaptive_gp_trend()`, `adaptive_kalman_trend()`
   
```python
# Automatic parameter selection
best_s, cv_scores = select_smoothing_parameter_cv(df, method='spline')
result = gp_trend(df)  # Auto-optimizes hyperparameters
```

### 3. Comprehensive Uncertainty Quantification ✅
   
- **Bootstrap confidence intervals:** All basic methods
- **Bayesian posteriors:** Gaussian Process methods
- **Kalman uncertainty:** State-space models
- **Significance testing:** SiZer analysis
   
```python
# Multiple uncertainty quantification approaches
result = bootstrap_derivative_ci(df, n_bootstrap=200)
result = gp_trend(df, confidence_level=0.95)
result = kalman_trend(df)  # Natural uncertainty from Kalman filter
```

### 4. Robust to Irregular Sampling ✅
   
Most methods handle irregular sampling:
   
- Splines, LOESS, GP, Kalman: Native support
- Savitzky-Golay: Requires regular sampling (automatically detected)
   
### 5. Multiscale Significance Analysis ✅
   
```python
# SiZer analysis across scales
sizer = sizer_analysis(df, n_bandwidths=20)
features = sizer.find_significant_features()

# Combined trend + significance
result = trend_with_sizer(df, trend_method='loess')
```

## Mathematical Details

### Savitzky-Golay Derivatives

The filter fits a polynomial of degree $p$ using least squares over a window of size $2m+1$:

$$\hat{f}(t) = \sum_{j=0}^{p} a_j t^j$$

The derivative is:

$$\hat{f}'(t_0) = \sum_{j=1}^{p} j \cdot a_j t_0^{j-1}$$

**Key requirement:** For $k$-th derivative, need $p \geq k$ and window size $\geq p + 1$.

**Scaling:** Derivative must be divided by $\Delta t^k$ where $\Delta t$ is the time step.

### Spline Derivatives

Cubic splines minimize:

$$\sum_{i=1}^{n} w_i (y_i - f(x_i))^2 + \lambda \int [f''(x)]^2 dx$$

subject to $\sum w_i (y_i - f(x_i))^2 \leq s$.

The derivative is obtained analytically from the spline coefficients.

**Advantage:** Handles irregular sampling naturally.

**Challenge:** Choosing $s$ requires domain knowledge or cross-validation.

### Naive Method (Central Differences)

$$f'(t_i) \approx \frac{y_{i+1} - y_{i-1}}{2\Delta t}$$

**Pros:** Simple, unbiased for linear trends

**Cons:** High variance, sensitive to noise, poor at boundaries

## Autocorrelation and Serial Dependence

Time series typically have autocorrelated errors:

$$\epsilon_t = \rho \epsilon_{t-1} + \eta_t$$

This violates independence assumptions and means:

1. Standard errors are underestimated
2. Confidence intervals are too narrow
3. Significance tests are invalid

**Solution:** Use block bootstrap or model the autocorrelation explicitly.

## Best Practices

### 1. Always specify time units
   
```python
# BAD: Assumes unit time
result = spline_trend(df)

# GOOD: Explicit time handling
result = improved_spline_trend(df, time_column='date')
```

### 2. Check sampling regularity
   
```python
time_diffs = df.index.to_series().diff()
if time_diffs.std() / time_diffs.mean() > 0.1:
    print("Warning: Irregular sampling detected")
    # Use splines, not Savitzky-Golay
```

### 3. Validate smoothing parameters
   
```python
# Use cross-validation
best_s, cv_results = select_smoothing_parameter_cv(
    df, param_name='s', method='spline'
)
```

### 4. Quantify uncertainty
   
```python
# Get confidence intervals
result = bootstrap_derivative_ci(
    df, method='spline', n_bootstrap=100
)

# Check if trend is significant
significant = result['significant_trend']
```

### 5. Handle seasonality
   
For seasonal data, consider:
   
- Pre-deseasonalizing with STL decomposition
- Using longer smoothing windows (> seasonal period)
- Fitting seasonal models explicitly

### 6. Be cautious at boundaries
   
```python
# Mark unreliable edge estimates
window = 15
result['reliable'] = True
result.iloc[:window//2, 'reliable'] = False
result.iloc[-window//2:, 'reliable'] = False
```

## Alternative Approaches

For more robust trend estimation, consider:

1. **Local polynomial regression (LOESS)**
   - More flexible than Savitzky-Golay
   - Better edge handling
   - Available in statsmodels

2. **State-space models**
   - Explicit modeling of trend component
   - Natural uncertainty quantification
   - Handles missing data

3. **Gaussian processes**
   - Full posterior distribution for derivatives
   - Principled uncertainty quantification
   - Heavy computationally

4. **L1 trend filtering**
   - Piecewise linear trends
   - Automatic changepoint detection
   - Robust to outliers

## References

1. Savitzky, A., & Golay, M. J. (1964). Smoothing and differentiation of data by simplified least squares procedures. Analytical chemistry, 36(8), 1627-1639.

2. De Boor, C. (1978). A practical guide to splines. Springer-Verlag.

3. Fan, J., & Gijbels, I. (1996). Local polynomial modelling and its applications. Chapman and Hall.

4. Kim, S. J., Koh, K., Boyd, S., & Gorinevsky, D. (2009). ℓ1 trend filtering. SIAM review, 51(2), 339-360.