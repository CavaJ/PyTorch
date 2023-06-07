from typing import Iterable, Hashable

import numpy as np
import pandas as pd
from scipy import stats

# functions to find outliers: iqr, z_score, modified_z_score
def iqr(df: pd.DataFrame, column_name: str, discard_negatives: bool = False, discard_zeros: bool = False):
    """
    Function to find outliers in a specific column of a dataframe.
    This approach uses the Interquartile Range (IQR) to detect outliers.
    Outliers are defined as 1.5*IQR above the third quartile or below the first quartile.
    :param df: DataFrame in which to find outliers.
    :param column_name: Name of the column to search for outliers.
    :param discard_negatives: whether to discard negative values during outlier finding
    :param discard_zeros: whether to discard zeros during outlier finding
    :return: DataFrame of outliers in the specified column.
    """
    # drop nan values beforehand
    df = df.dropna()

    if discard_negatives:
        df = df[df[column_name] >= 0]
    if discard_zeros:
        df = df[df[column_name] != 0]

    # Define Q1, Q3, and IQR
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr_value = q3 - q1
    # Define lower and upper bounds for outliers
    lower_bound = q1 - 1.5 * iqr_value
    upper_bound = q3 + 1.5 * iqr_value
    # Identify the outliers
    outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
    return outliers


def z_score(df: pd.DataFrame, column_name: str, discard_negatives: bool = False, discard_zeros: bool = False):
    """
    This function takes a pandas DataFrame df and a column_name as input.
    It returns the outliers of the DataFrame's column based on the z-score.
    The intuition behind Z-score is to describe any data point by finding their relationship with the Standard Deviation
    and Mean of the group of data points. Z-score is finding the distribution of data where mean is 0 and standard
    deviation is 1 (i.e., normal distribution). Outliers will have a Z-score above a certain threshold (commonly 3). It
    will automatically convert copy of the data points to a normal distribution and find outliers on that.

    So even if your data is not already standardized, this function will calculate the appropriate z-scores.
    In other words, it will do the standardization for you as part of the process.
    That's why this function can be useful for finding outliers even in non-standardized data. Each z-score tells you
    how many standard deviations away from the mean each point is. In normally distributed data, we would expect 99.7%
    of values to be within 3 standard deviations from the mean, so z-scores with an absolute value of 3 or more are
    often considered to be outliers.

    :param df: DataFrame in which to find outliers.
    :param column_name: Name of the column to search for outliers.
    :param discard_negatives: whether to discard negative values during outlier finding
    :param discard_zeros: whether to discard zeros during outlier finding
    :return: DataFrame of outliers in the specified column.
    """
    # drop nan values beforehand
    df = df.dropna()

    if discard_negatives:
        df = df[df[column_name] >= 0]
    if discard_zeros:
        df = df[df[column_name] != 0]

    z_scores = np.abs(stats.zscore(df[column_name]))
    outliers = df[(z_scores > 3)]

    return outliers


def modified_z_score(df: pd.DataFrame, column_name: str, discard_negatives: bool = False, discard_zeros: bool = False):
    """
    This function takes a DataFrame and a column name as input, and identifies outliers in the column
    using the modified Z-score method with a Median Absolute Deviation (MAD). Outliers are defined as values
    whose modified Z-scores are above a threshold of 3.5.

    The number 0.6745 in the calculation of the modified Z-score is a scaling constant that is used to convert the
    absolute deviation from the median into a Z-score in the standard normal distribution.

    The threshold of 3.5 is used because it's a common choice for identifying outliers using the modified Z-score
    method. In a standard normal distribution, about 99.7% of the data will fall within 3 standard deviations of
    the mean. Since the modified Z-score is designed to mimic the standard Z-score for normally-distributed data,
    the same threshold is often used. However, because the modified Z-score is less affected by outliers than the
    standard Z-score, a slightly larger threshold (like 3.5 instead of 3) is sometimes used to be more conservative
    about identifying outliers.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        column_name (str): The name of the column in which to find outliers.
        discard_negatives (bool): whether to discard negative values during outlier finding
        discard_zeros (bool): whether to discard zeros during outlier finding

    Returns:
        pandas.DataFrame: A DataFrame containing the outliers in the specified column.

    Usage:
        outliers = modified_z_score(df, 'column_name')
    """
    # drop nan values beforehand
    df = df.dropna()

    if discard_negatives:
        df = df[df[column_name] >= 0]
    if discard_zeros:
        df = df[df[column_name] != 0]

    mad_value = stats.median_abs_deviation(df[column_name])
    median = np.median(df[column_name])
    modified_z_scores = 0.6745 * (df[column_name] - median) / mad_value
    outliers = df[(np.abs(modified_z_scores) > 3.5)]
    return outliers



def get_unique_insertion_order_list(original_iterable: Iterable[Hashable], drop_nan=False) -> list:
    """
    Function to retrieve list of unique elements from any Iterable[Hashable]
    while preserving insertion order, with time complexity of O(n).
    If drop_nan is True, np.nan and None values will be dropped.
    """
    seen = set()
    seen_add = seen.add
    if drop_nan:
        return [x for x in original_iterable if not pd.isnull(x) and not (x in seen or seen_add(x))]
    # handle nan carefully, since it is not equal to itself
    seen_nan = False
    result = []
    for x in original_iterable:
        if pd.isnull(x):  # if the element is np.nan
            if not seen_nan:
                result.append(np.nan)  # if there is a None in the given iterable, pd.isnull() is true; add nan instead
                seen_nan = True  # set the flag to true
        elif not (x in seen or seen_add(x)):  # membership test and .add() function for the set is O(1)
            result.append(x)
    return result


def get_set_of_unique_elements(original_iterable: Iterable[Hashable], drop_nan=False) -> set:
    """
    Function to retrieve set of unique elements from any Iterable[Hashable] with time complexity of O(n).
    The resulting set will have O(1) average time complexity for membership tests as compared to e.g. pandas's unique().
    If drop_nan is True, np.nan and None values will be dropped.
    """
    seen = set()
    seen_nan = False
    for x in original_iterable:
        if pd.isnull(x):  # if the element is np.nan
            if not seen_nan and not drop_nan:
                seen.add(np.nan)  # if there is a None in the given iterable, pd.isnull() is true; add nan instead
                seen_nan = True  # set the flag to true
        else:
            seen.add(x)
    return seen