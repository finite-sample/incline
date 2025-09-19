incline: Estimate Trend at a Particular Point in a Noisy Time Series
-----------------------------------------------------------------------

.. image:: https://img.shields.io/pypi/v/incline.svg
    :target: https://pypi.python.org/pypi/incline
.. image:: https://static.pepy.tech/badge/incline
    :target: https://pepy.tech/project/incline
.. image:: https://github.com/finite-sample/incline/workflows/ci/badge.svg
    :target: https://github.com/finite-sample/incline/actions?query=workflow%3Aci
.. image:: https://img.shields.io/badge/docs-github.io-blue
    :target: https://finite-sample.github.io/incline/

Trends in time series are valuable. If the cost of a product rises suddenly, it likely indicates a sudden shortfall in supply or a sudden rise in demand. If the cost of claims filed by a patient rises sharply, it may suggest rapidly worsening health. But how do we estimate the trend at a particular time in a noisy time series? Smooth the time series using any one of the many methods, local polynomials or via GAMs or similar such methods, and then estimate the derivative(s) of the function at the chosen point in time.

The package provides a couple of ways of approximating the underlying function for the time series:

- fitting a local higher order polynomial via Savitzky-Golay over a window of choice

- fitting a smoothing spline

The package provides a way to estimate the first and second derivative at any given time using either of those methods. Beyond these smarter methods, the package also provides a way a naive estimator of slope---average change when you move one-step forward (step = observed time units) and one-step backward. The users can also calculate average or max. slope over a time window (over observed time steps).

The difference between naive estimates and estimates based on smoothed time series can be substantial. In the `example <incline/examples/incline_example.ipynb>`__ we provide, the correlation is -.47.

.. figure:: trend.png
   :scale: 100 %
   :alt: Example of derivatives


Clarification
~~~~~~~~~~~~~

Sometimes we want to know what the "trend" was over a particular time
window. But what that means is not 100% clear. For a synopsis of the
issues, see
`here <http://gbytes.gsood.com/2018/06/22/talking-on-a-tangent/>`__.

Underlying Machinery
~~~~~~~~~~~~~~~~~~~~

Savitzky-Golay
^^^^^^^^^^^^^^

Filter the time series using local polynomials and get an estimate of
the derivative in one shot. For more information, see the `Python
dcoumentation <https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.signal.savgol_filter.html>`__
and
`Wikipedia <https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter>`__

Univariate Splines
^^^^^^^^^^^^^^^^^^

Find more details `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html>`__


Assumption
~~~~~~~~~~~~~~~~~~~

Silly as it is, for now, we assume that the time series is a) complete, and b) increases with unit time intervals.

API
~~~

The package wraps the functions for doing local smoothing and derivative
estimation for a standardized interface. We use this standard interface
to estimate the trend at a particular set of points in parallel for
thousands of time series.

The package ``incline`` exposes 4 functions:

1. ``naive_trend``:
    
    **Input:**
    
    -  df: pandas dataFrame `time series
       object <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>`__
    
    **Functionality:**
    
    -  estimates the derivative at a location by taking the average of
       change when you move one unit to the right and change when you move
       one unit to the left.
    
    **Output:**
    
    dataframe with 6 columns (smoothed value column just has ``None``):
    ``datetime, function_order (value of the polynomial order), smoothed_value, derivative_method, derivative_order, derivative_value``.

2. ``spline_trend``:

    **Input:**
    
    -  df: pandas dataFrame `time series
       object <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>`__
    -  function\_order: spline order (default is 3)---fitting with cubic
       splines. The knot placement is determined by the smoothing factor s.
    -  derivative\_order: (0, 1, 2, ... with default as 1)
    -  s: smoothing factor. the total unnormalized global cost that we are willing to bear. larger values give more smoothed estimates. See the 
       `documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html>`__ for details. 
    
    **Functionality:**
    
    Interpolates time series with splines of 'function\_order'. And then
    calculates the derivative\_order using the smoothed function.
    
    **Output:**
    
    dataframe with 6 columns:
    ``datetime, function_order (value of the polynomial order), smoothed_value, derivative_method, derivative_order, derivative_value``.
    
    A row can be 2012-01-01, "spline", 2, 1, 0

3. ``sgolay_trend``:

    **Input:**
    
    -  df pandas dataFrame `time series
       object <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>`__
    -  window\_size: default is 15
    -  function\_order: polynomial order (default is 3)
    -  derivative\_order: (0, 1, 2, ... with default as 1)
    
    **Functionality:**
    
    Interpolates time series with savitzky-golay using polynomials of
    'function\_order'. And then calculates the derivative\_order using the
    smoothed function.
    
    **Output:**
    
    dataframe with 6 columns:
    ``datetime, function_order (value of the polynomial order), smoothed_value, derivative_method, derivative_order, derivative_value``.
    
    Sample row: 2012-01-01, "savitzky-golay", 2, 1, 0

4. ``trending``:

    **Input:**
    
    -  df\_list: list of outputs (dataframes) from ``savitzky_golay_trend``
       or ``spline_trend`` with a new column called 'id' that identifies the
       time series
    -  derivative\_order: (1 or 2)
    -  k: number of latest time periods to consider.
    -  max\_or\_avg: "max" or "avg"
    
    **Functionality:**
    
    for each item in the list, calculate either the max or the average
    (depending on max\_or\_avg) of the Yth derivative (based on the
    derivative\_order) over the last k time\_periods (based on the input).
    It then orders the list based on max to min.
    
    For instance, for derivative\_order = 1, max\_or\_avg = "max",
    time\_periods = 3, for each item in the list, the function will take the
    max of the last 3 rows of the dataframe entries identifying the 1st
    derivative.
    
    So each item in the list produces one number (max or avg.). We then
    produce a new dataframe with 2 columns: ``id, max_or_avg``
    
    **Output:**
    
    Dataframe with 2 columns: ``id, max_or_avg``

Installation
~~~~~~~~~~~~

::

    pip install incline

Usage
~~~~~

::

    from incline import spline_trend

    locpol = spline_trend(time_series, , ...)

Examples
~~~~~~~~

Please look at this `notebook <https://github.com/soodoku/incline/blob/master/incline/examples/incline_example.ipynb>`_. for how to use incline using data from the stock market.

License
~~~~~~~

The package is released under the `MIT
License <https://opensource.org/licenses/MIT>`__.

Authors
~~~~~~~

Suriyan Laohaprapanon and Gaurav Sood

Additional Reading
~~~~~~~~~~~~~~~~~~

While we don't provide this in the package but you could approximate the function using:

1. Penalized cubic splines using GAMS via `pyGAM <https://github.com/dswah/pyGAM>`__. For more information, see
these `lecture notes <https://web.stanford.edu/class/stats202/content/lec17.pdf>`__  

2. Or, `nonparametrically <https://pythonhosted.org/PyQt-Fit/NonParam_tut.html>`__

And here's a paper on `Derivative Estimation with Local Polynomial Fitting 
<https://dl.acm.org/citation.cfm?id=2502590>`__
