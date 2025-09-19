# -*- coding: utf-8 -*-

import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline


def naive_trend(df, column_value='value'):
    """
    naive_trend

    Gives the naive slope: look to the right, look to the left, 
    travel one unit each, and get the average change. At the ends,
    we merely use the left or the right value.

    Args:
        df: pandas dataFrame time series object
    """
    y = df[column_value]

    y_1 = y.shift(1)
    y_2 = y.shift(-1)

    y1_diff = y - y_1  # backward difference
    yneg1_diff = y_2 - y  # forward difference

    yy = pd.concat([y.rename('orig'),
                    y_1.rename('plus_1'),
                    y_2.rename('min_1'),
                    y1_diff.rename('plus_1_diff'),
                    yneg1_diff.rename('min_1_diff')], axis = 1)
    odf = df.copy()
    odf['derivative_value'] = yy[['plus_1_diff', 'min_1_diff']].mean(axis = 1)
    odf['derivative_method'] = 'naive'
    odf['function_order'] = None
    odf['derivative_order'] = 1

    return odf


def spline_trend(df, column_value='value', function_order=3,
                 derivative_order=1, s=3):
    """
    spline_trend

    Interpolates time series with splines of 'function_order'. And then
    calculates the derivative_order using the smoothed function.

    Args:
        df: pandas dataFrame time series object
        function_order: spline order (default is 3)
        derivative_order: (0, 1, 2, ... with default as 1)

    Returns:
        DataFrame: dataframe with 6 columns:- datetime,
            function_order (value of the polynomial order), smoothed_value,
            derivative_method, derivative_order, derivative_value.

        A row can be 2012-01-01, "spline", 2, 1, 0
    """
    x = df.reset_index().index.values.astype(float)
    y = df[column_value]
    spl = UnivariateSpline(x, y, k=function_order, s=s)
    odf = df.copy()
    odf['smoothed_value'] = spl(x)
    odf['derivative_value'] = spl(x, nu=derivative_order)
    odf['function_order'] = function_order
    odf['derivative_method'] = 'spline'
    odf['derivative_order'] = derivative_order
    return odf


def sgolay_trend(df, column_value='value', function_order=3,
                 derivative_order=1, window_length=15):
    """
    sgolay_trend

    Interpolates time series with savitzky-golay using polynomials of
    'function_order'. And then calculates the derivative_order using
    the smoothed function.

    Args:
        df: pandas dataFrame time series object
        window_size: default is 15
        function_order: polynomial order (default is 3)
        derivative_order: (0, 1, 2, ... with default as 1)

    Returns:
        DataFrame: dataframe with 6 columns:- datetime,
            function_order (value of the polynomial order), smoothed_value,
            derivative_method, derivative_order, derivative_value.

        Sample row: 2012-01-01, "sgolay", 2, 1, 0
    """
    y = df[column_value]
    odf = df.copy()
    odf['smoothed_value'] = savgol_filter(y, window_length=window_length,
                                            polyorder=function_order)
    odf['derivative_value'] = savgol_filter(y, window_length=window_length,
                                            polyorder=function_order,
                                            deriv=derivative_order)
    odf['function_order'] = function_order
    odf['derivative_method'] = 'sgolay'
    odf['derivative_order'] = derivative_order
    return odf


def trending(df_list, column_id='id', derivative_order=1, max_or_avg='max',
             k=5):
    """
    trending

    For each item in the list, calculate either the max or the average
    (depending on max_or_avg) of the Yth derivative (based on the
    derivative_order) over the last k time_periods (based on the input).
    It then orders the list based on max to min.
    
    For instance, for derivative_order = 1, max_or_avg = "max",
    time_periods = 3, for each item in the list, the function will take
    the max of the last 3 rows of the dataframe entries identifying the
    1st derivative.

    So each item in the list produces one number (max or avg.). We then
    produce a new dataframe with 2 columns: id, max_or_avg

    Args:
        df_list: list of outputs (dataframes) from sgolay_trend or
            spline_trend with a new column called 'id' that identifies
            the time series
        derivative_order: (1 or 2)
        k: number of latest time periods to consider.
        max_or_avg: "max" or "avg"


    Returns:
        DataFrame: dataframe with 2 columns: id, max_or_avg
    """

    cdf = []
    for df in df_list:
        cdf.append(df[df.derivative_order == derivative_order][-k:])
    tdf = pd.concat(cdf, sort=False)
    if max_or_avg == 'avg':
        max_or_avg = 'mean'
    odf = tdf.groupby('id').agg({'derivative_value': max_or_avg})
    odf.reset_index(inplace=True)
    odf.columns = ['id', 'max_or_avg'] 
    return odf


if __name__ == "__main__":
    pass
