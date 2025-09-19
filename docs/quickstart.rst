Quick Start Guide
=================

This guide will help you get started with incline quickly.

Basic Usage
-----------

incline provides four main functions for trend estimation:

1. **naive_trend**: Simple forward/backward difference
2. **spline_trend**: Spline interpolation based trend
3. **sgolay_trend**: Savitzky-Golay filter based trend  
4. **trending**: Aggregate trends across multiple time series

Simple Example
--------------

.. code-block:: python

    import pandas as pd
    from incline import naive_trend, spline_trend, sgolay_trend

    # Create sample time series data
    data = {
        'timestamp': pd.date_range('2020-01-01', periods=10, freq='D'),
        'value': [1, 3, 2, 5, 8, 7, 12, 15, 14, 18]
    }
    df = pd.DataFrame(data)
    df = df.set_index('timestamp')

    # Method 1: Naive trend estimation
    naive_result = naive_trend(df)
    print("Naive trend:")
    print(naive_result[['value', 'derivative_value']].head())

    # Method 2: Spline-based trend estimation
    spline_result = spline_trend(df, function_order=3, derivative_order=1)
    print("\nSpline trend:")
    print(spline_result[['value', 'smoothed_value', 'derivative_value']].head())

    # Method 3: Savitzky-Golay trend estimation
    sgolay_result = sgolay_trend(df, window_length=5, function_order=2)
    print("\nSavitzky-Golay trend:")
    print(sgolay_result[['value', 'smoothed_value', 'derivative_value']].head())

Understanding the Output
------------------------

All trend functions return a DataFrame with these columns:

- **Original data columns**: Your input data is preserved
- **smoothed_value**: The smoothed version of your time series (None for naive method)
- **derivative_value**: The estimated derivative/trend at each point
- **derivative_method**: Which method was used ('naive', 'spline', or 'sgolay')
- **function_order**: The polynomial order used for smoothing
- **derivative_order**: The order of derivative calculated (1 for slope, 2 for acceleration)

Working with Multiple Time Series
----------------------------------

Use the ``trending`` function to analyze multiple time series:

.. code-block:: python

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

Parameter Tuning
----------------

**For spline_trend**:

- ``function_order``: Higher values = smoother curves (default: 3)
- ``s``: Smoothing factor, higher = more smoothing (default: 3)
- ``derivative_order``: 1 for slope, 2 for acceleration (default: 1)

**For sgolay_trend**:

- ``window_length``: Size of smoothing window, must be odd (default: 15)
- ``function_order``: Polynomial order for fitting (default: 3)
- ``derivative_order``: 1 for slope, 2 for acceleration (default: 1)

Next Steps
----------

- Check out the :doc:`examples` for real-world use cases
- See the :doc:`api` for detailed function documentation
- Look at the example notebook in the repository for stock market analysis