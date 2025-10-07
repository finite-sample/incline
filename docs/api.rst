API Reference
=============

This page contains the API reference for all public functions in incline.

Main Trend Estimation Functions
--------------------------------

.. automodule:: incline.trend
   :members: naive_trend, spline_trend, sgolay_trend, trending
   :show-inheritance:

Advanced Functions
------------------

.. automodule:: incline.trend
   :members: bootstrap_derivative_ci, select_smoothing_parameter_cv
   :show-inheritance:

Utility Functions
-----------------

.. automodule:: incline.trend
   :members: compute_time_deltas
   :show-inheritance:

Function Details
----------------

Core Trend Functions
~~~~~~~~~~~~~~~~~~~~

naive_trend
^^^^^^^^^^^

.. autofunction:: incline.trend.naive_trend

spline_trend
^^^^^^^^^^^^

.. autofunction:: incline.trend.spline_trend

sgolay_trend
^^^^^^^^^^^^

.. autofunction:: incline.trend.sgolay_trend

trending
^^^^^^^^

.. autofunction:: incline.trend.trending

Statistical Functions
~~~~~~~~~~~~~~~~~~~~~

bootstrap_derivative_ci
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: incline.trend.bootstrap_derivative_ci

select_smoothing_parameter_cv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: incline.trend.select_smoothing_parameter_cv

Utility Functions
~~~~~~~~~~~~~~~~~

compute_time_deltas
^^^^^^^^^^^^^^^^^^^

.. autofunction:: incline.trend.compute_time_deltas