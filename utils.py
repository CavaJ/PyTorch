from typing import Iterable, Hashable

import numpy as np
import pandas as pd
from scipy import stats

# functions to find outliers: iqr, z_score, modified_z_score
def iqr(df: pd.DataFrame, column_name: str, drop_nan: bool = False, discard_negatives: bool = False,
        discard_zeros: bool = False, closeness_tolerance_to_zero: float = 1e-4) -> pd.DataFrame:
    """
    Function to find outliers in a specific column of a dataframe.
    This approach uses the Interquartile Range (IQR) to detect outliers.
    Outliers are defined as 1.5*IQR above the third quartile or below the first quartile.
    :param df: DataFrame in which to find outliers.
    :param column_name: Name of the column to search for outliers.
    :param drop_nan: whether to drop nan values beforehand, by default False
    :param discard_negatives: whether to discard negative values from the data distribution during outlier finding
    :param discard_zeros: whether to discard zeros from the data distribution during outlier finding
    :param closeness_tolerance_to_zero: closeness tolerance to zero
    :return: DataFrame of outliers in the specified column.
    """
    df = _filter_values(df, column_name, drop_nan, discard_negatives, discard_zeros, closeness_tolerance_to_zero)

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


def z_score(df: pd.DataFrame, column_name: str, drop_nan: bool = False, discard_negatives: bool = False,
            discard_zeros: bool = False, closeness_tolerance_to_zero: float = 1e-4) -> pd.DataFrame:
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
    :param discard_negatives: whether to discard negative values from the data distribution during outlier finding
    :param drop_nan: whether to drop nan values beforehand, by default False
    :param discard_zeros: whether to discard zeros from the data distribution during outlier finding
    :param closeness_tolerance_to_zero: closeness tolerance to zero
    :return: DataFrame of outliers in the specified column.
    """
    df = _filter_values(df, column_name, drop_nan, discard_negatives, discard_zeros, closeness_tolerance_to_zero)

    z_scores = np.abs(stats.zscore(df[column_name]))
    outliers = df[(z_scores > 3)]

    return outliers


def modified_z_score(df: pd.DataFrame, column_name: str, drop_nan: bool = False, discard_negatives: bool = False,
                     discard_zeros: bool = False, closeness_tolerance_to_zero: float = 1e-4) -> pd.DataFrame:
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
        drop_nan (bool): whether to drop nan values beforehand, by default False
        discard_negatives (bool): whether to discard negative values from the data distribution during outlier finding
        discard_zeros (bool): whether to discard zeros from the data distribution during outlier finding
        closeness_tolerance_to_zero (float): closeness tolerance to zero

    Returns:
        pandas.DataFrame: A DataFrame containing the outliers in the specified column.

    Usage:
        outliers = modified_z_score(df, 'column_name')
    """
    df = _filter_values(df, column_name, drop_nan, discard_negatives, discard_zeros, closeness_tolerance_to_zero)

    mad_value = stats.median_abs_deviation(df[column_name])
    median = np.median(df[column_name])
    modified_z_scores = 0.6745 * (df[column_name] - median) / mad_value
    outliers = df[(np.abs(modified_z_scores) > 3.5)]
    return outliers


def _filter_values(df: pd.DataFrame, column_name: str, drop_nan: bool, discard_negatives: bool,
                   discard_zeros: bool, closeness_tolerance_to_zero: float) -> pd.DataFrame:
    if drop_nan:
        # drop nan values
        df = df[df[column_name].notna()]

    if discard_negatives:
        df = df[df[column_name] >= 0]
    if discard_zeros:
        # Mask for values close to zero
        mask = np.isclose(df[column_name], 0, atol=closeness_tolerance_to_zero)
        # Filter out rows where values in the column are close to zero.
        # if e.g. closeness_tolerance_to_zero is 1e-4, then 0.001 will not be close to 0, 0.0001 will be close to 0.
        df = df[~mask]

    return df


def _intersection_of_dfs_on_column(df1, df2, df3, column_name):
    """
    Function to get the intersection of values of three DataFrames based on a certain column.

    :param df1: First DataFrame
    :param df2: Second DataFrame
    :param df3: Third DataFrame
    :param column_name: The name of the column to perform the intersection on
    :return: A DataFrame with the intersection of values on the given column
    """
    return df1[[column_name]].merge(df2[[column_name]], on=column_name).merge(df3[[column_name]], on=column_name)


def intersection_of_outliers_found_by_iqr_z_score_and_modified_z_score(df: pd.DataFrame, column_name: str,
                                                                       drop_nan: bool = False,
                                                                       discard_negatives: bool = False,
                                                                       discard_zeros: bool = False,
                                                                       closeness_tolerance_to_zero: float = 1e-4) \
                                                                       -> pd.DataFrame:
    """
    Find the intersection of outliers in a DataFrame's column as detected by three methods: IQR, Z-score, and
    Modified Z-score.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    column_name : str
        The name of the column in the DataFrame for which to find the outliers.
    drop_nan: bool
        Whether to drop nan values beforehand, by default False.
    discard_negatives : bool, optional
        Whether to discard negative values from the analysis, by default False.
    discard_zeros : bool, optional
        Whether to discard zero values from the analysis, by default False.
    closeness_tolerance_to_zero : float, optional
        The tolerance within which a value is considered to be close to zero, by default 1e-4.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the rows that are considered as outliers by all three methods (IQR, Z-score, and
        Modified Z-score).

    Example
    -------
    df = pd.DataFrame({'measurement': [1, 2, 3, 400, 5, 6, 700, 8, 9]})
    intersection_of_outliers_found_by_iqr_z_score_and_modified_z_score(df, 'measurement')
    """
    df = _filter_values(df, column_name, drop_nan, discard_negatives, discard_zeros, closeness_tolerance_to_zero)

    iqr_df = iqr(df, column_name)
    z_score_df = z_score(df, column_name)
    modified_z_score_df = modified_z_score(df, column_name)
    return _intersection_of_dfs_on_column(iqr_df, z_score_df, modified_z_score_df, column_name)



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