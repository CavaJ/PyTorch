from typing import Union, Iterable, Hashable

import numpy as np
import pandas as pd

import Levenshtein # python-Levenshtein==0.21.1

# functions to find outliers: iqr, z_score, modified_z_score
def iqr(df: pd.DataFrame, column_name: str, drop_nan: bool = False, discard_negatives: bool = False,
        discard_zeros: bool = False, closeness_tolerance_to_zero: float = 1e-4) -> pd.DataFrame:
    """
    Function to find outliers in a specific column of a dataframe.
    This approach uses the Interquartile Range (IQR) to detect outliers.
    Outliers are defined as 1.5*IQR above the third quartile or below the first quartile.
    :param df: A dataframe in which to find outliers.
    :param column_name: A name of the column to search for outliers.
    :param drop_nan: Whether to drop nan values beforehand, by default, False.
    :param discard_negatives: Whether to discard negative values from the data distribution during outlier finding.
    :param discard_zeros: Whether to discard zeros from the data distribution during outlier finding.
    :param closeness_tolerance_to_zero: A closeness tolerance to zero.
    :return: A dataframe of outliers in the specified column.
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
    and Mean for the group of data points.
    Z-score is finding the distribution of data where mean is 0,
    and standard deviation is 1 (i.e., normal distribution).
    Outliers will have a Z-score above a certain threshold (commonly 3).
    It will automatically convert copy of the data points to a normal distribution and find outliers on that.

    So even if your data is not already standardized, this function will calculate the appropriate z-scores.
    In other words, it will do the standardization for you as part of the process.
    That's why this function can be useful to find outliers even in non-standardized data.
    Each z-score tells you how many standard deviations away from the mean each point is.
    In normally distributed data, we would expect 99.7% of values to be within 3 standard deviations from the mean,
    so z-scores with an absolute value of 3 or more are often considered to be outliers.

    :param df: A dataframe in which to find outliers.
    :param column_name: A name of the column to search for outliers.
    :param discard_negatives: Whether to discard negative values from the data distribution during outlier finding.
    :param drop_nan: Whether to drop nan values beforehand, by default, False.
    :param discard_zeros: Whether to discard zeros from the data distribution during outlier finding.
    :param closeness_tolerance_to_zero: A closeness tolerance to zero.
    :return: A dataframe of outliers in the specified column.
    """
    df = _filter_values(df, column_name, drop_nan, discard_negatives, discard_zeros, closeness_tolerance_to_zero)

    # Calculating z-scores with numpy
    mean = np.mean(df[column_name])
    std_dev = np.std(df[column_name])
    z_scores = np.abs((df[column_name] - mean) / std_dev)

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


    :param df: The input DataFrame.
    :param column_name: The name of the column in which to find outliers.
    :param drop_nan: Whether to drop nan values beforehand, by default, False.
    :param discard_negatives: Whether to discard negative values from the data distribution during outlier finding.
    :param discard_zeros: Whether to discard zeros from the data distribution during outlier finding.
    :param closeness_tolerance_to_zero: A closeness tolerance to zero.
    :return: A dataframe containing the outliers in the specified column.

    Usage:
        outliers = modified_z_score(df, 'column_name')
    """
    df = _filter_values(df, column_name, drop_nan, discard_negatives, discard_zeros, closeness_tolerance_to_zero)

    mad_value = _mad(df[column_name])
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
        # If e.g. closeness_tolerance_to_zero is 1e-4, then 0.001 will not be close to 0, 0.0001 will be close to 0.
        df = df[~mask]

    return df


# Function to calculate Median Absolute Deviation
def _mad(data: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """
    Compute the Median Absolute Deviation of the given data.

    Parameters
    ----------
    data : Union[pd.Series, np.ndarray]
        The data you want to compute the MAD for.
        It can be a pandas' Series or a numpy array.

    Returns
    -------
    np.ndarray
        The computed MAD of the data.
    """
    return np.median(np.abs(data - np.median(data)))


def _intersection_of_dfs_on_column(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame, column_name: str):
    """
    Function to get the intersection of values for three DataFrames based on a certain column.

    :param df1: The first dataframe.
    :param df2: The second dataframe.
    :param df3: The third dataframe.
    :param column_name: The name of the column to perform the intersection on.
    :return: A dataframe with the intersection of values on the given column.
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

    :param df: The dataframe which contains the data.
    :param column_name:  The name of the column in the DataFrame for which to find the outliers.
    :param drop_nan: Whether to drop nan values beforehand, by default, False.
    :param discard_negatives: Whether to discard negative values from the analysis, by default, False.
    :param discard_zeros: Whether to discard zero values from the analysis, by default, False.
    :param closeness_tolerance_to_zero: The tolerance within which a value is considered to be close to zero,
                                        by default 1e-4.
    :return: A dataframe containing the rows that are considered as outliers by all three methods (IQR, Z-score, and
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
    
    
    
def sanitize_phone_or_fax_number(number: str) -> str:
    """
    Sanitizes phone/fax number by keeping only numbers, '+' (only if at the beginning), '-' (hyphen) and ' ' (space).

    :param number: Input phone/fax number as a string.
    :return: Sanitized phone/fax number.
    """
    # if number string does not contain any digit, return empty string
    # r'\D' is equivalent to r'[^0-9]'
    if len(re.sub(r'\D', '', number)) == 0:
        return ""

    # remove special characters except a digit, plus, hyphen and whitespace
    number = re.sub(r'[^0-9+\-\s]', '', number)

    # replace function to determine if it should be a hyphen or a space
    def replacement(match):
        # If the matched string contains a hyphen, replace with a hyphen, otherwise with a space
        return '-' if '-' in match.group(0) else ' '

    # remove all '+' characters
    sanitized = number.replace('+', '')
    # replace any combination of consecutive hyphens and/or whitespaces after dropping '+' and
    # remove leading and trailing whitespaces and hyphens
    sanitized = re.sub(r'[-\s]+', replacement, sanitized).strip(" -")

    # have the version of phone/fax number with digits and plus only
    # (without any special character, hyphen and whitespace)
    number_with_digits_and_plus_only = re.sub(r'[^0-9+]', '', number)

    # If the phone/fax number with digits and plus only starts with "+" add it back to the beginning
    if number_with_digits_and_plus_only.startswith('+'):
        sanitized = '+' + sanitized

    return sanitized
    
    
def sanitize_string_for_levenshtein_check(string: str) -> str:
    """
    This method sanitizes strings for the Levenshtein check.
    :param string: The string to sanitize.
    :return: The sanitized string.
    """
    # convert string to a lower case
    string = string.lower()

    # remove special characters (everything except letters, numbers, spaces, hyphens and underscores)
    string = re.sub(r'[^A-Za-z0-9\s\-_]', '', string)

    # replace consecutive combinations of hyphens, spaces and underscores with a single space
    string = re.sub(r'[\-\s_]{2,}', ' ', string)

    # remove leading and trailing hyphens, spaces and underscores
    string = string.strip('- _')

    return string


def get_levenshtein_string_similarity_match(string_1: str, string_2: str, similarity_threshold: str = 'high') -> bool:
    """
    This method returns whether two strings (both will be sanitized first) are similar, according to Levenshtein's
    similarity check and a given similarity threshold.
    :param string_1: The first string for comparison.
    :param string_2: The second string for comparison.
    :param similarity_threshold: The threshold to set whether high, moderate or low similarity should be checked
      (possible values: 'high', 'moderate', 'low').
    """
    # sanitize strings
    string_1 = sanitize_string_for_levenshtein_check(string_1)
    string_2 = sanitize_string_for_levenshtein_check(string_2)

    # get the string similarity (returns between 0 and 1, 0 meaning not similar at all and 1 meaning exactly similar)
    similarity_ratio = Levenshtein.ratio(string_1, string_2)

    # return whether the strings are similar, according to the given threshold
    if similarity_threshold == 'high':
        return similarity_ratio >= 0.9
    elif similarity_threshold == 'medium':
        return similarity_ratio >= 0.7
    elif similarity_threshold == 'low':
        return similarity_ratio > 0.5
    else:
        raise Exception(f"Similarity threshold '{similarity_threshold}' not supported")