incline: Estimate Trend at a Particular Point in a Noisy Time Series
====================================================================

.. image:: https://img.shields.io/pypi/v/incline.svg
    :target: https://pypi.python.org/pypi/incline
.. image:: https://static.pepy.tech/badge/incline
    :target: https://pepy.tech/project/incline

Trends in time series are valuable. If the cost of a product rises suddenly, it likely indicates a sudden shortfall in supply or a sudden rise in demand. If the cost of claims filed by a patient rises sharply, it may suggest rapidly worsening health. But how do we estimate the trend at a particular time in a noisy time series? Smooth the time series using any one of the many methods, local polynomials or via GAMs or similar such methods, and then estimate the derivative(s) of the function at the chosen point in time.

The package provides a couple of ways of approximating the underlying function for the time series:

- fitting a local higher order polynomial via Savitzky-Golay over a window of choice
- fitting a smoothing spline

The package provides a way to estimate the first and second derivative at any given time using either of those methods. Beyond these smarter methods, the package also provides a way a naive estimator of slope---average change when you move one-step forward (step = observed time units) and one-step backward. The users can also calculate average or max. slope over a time window (over observed time steps).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples
   limitations

Installation
============

.. code-block:: bash

    pip install incline

Quick Start
===========

.. code-block:: python

    from incline import spline_trend, sgolay_trend, naive_trend
    import pandas as pd

    # Your time series data
    df = pd.DataFrame({'value': [1, 2, 3, 5, 8, 13, 21]})
    
    # Estimate trend using spline interpolation
    result = spline_trend(df)
    
    # Or using Savitzky-Golay filter
    result = sgolay_trend(df)
    
    # Or using naive method
    result = naive_trend(df)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`