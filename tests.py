import numpy as np
import pandas as pd
import pytest

from utils import modified_z_score, z_score, \
    intersection_of_outliers_found_by_iqr_z_score_and_modified_z_score, iqr, sanitize_phone_or_fax_number, get_unique_insertion_order_list, get_set_of_unique_elements


@pytest.mark.parametrize("input_df, column_name, expected_output", [
    (pd.DataFrame({'data': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}), 'data', pd.DataFrame(columns=['data'])),
    (pd.DataFrame({'data': [1, np.nan, 2, 3, 100, 5, 6, 7, 8, 9, 10]}), 'data', pd.DataFrame({'data': [100]})),
    (pd.DataFrame({'data': [-5, 0, 1, 2, 3, 100, 5, 6, 7, 8, 9, 10]}), 'data', pd.DataFrame({'data': [100]}))
])
def test_iqr(input_df, column_name, expected_output):
    assert iqr(input_df, column_name, drop_nan=True, discard_negatives=True,
               discard_zeros=True)[column_name].tolist() == expected_output[column_name].tolist()


@pytest.mark.parametrize("input_df, column_name, expected_output", [
    (pd.DataFrame({'data': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}), 'data', pd.DataFrame({'data': []})),
    (pd.DataFrame({'data': [1, np.nan, 2, 3, 100, 5, 6, 7, 8, 9, 10]}), 'data', pd.DataFrame({'data': []})),
    (pd.DataFrame({'data': [-5, 0, 1, 2, 3, 100, 5, 6, 7, 8, 9, 10]}), 'data', pd.DataFrame({'data': []}))
])
def test_z_score(input_df, column_name, expected_output):
    assert z_score(input_df, column_name, drop_nan=True, discard_negatives=True,
                   discard_zeros=True)[column_name].tolist() == expected_output[column_name].tolist()


@pytest.mark.parametrize("input_df, column_name, expected_output", [
    (pd.DataFrame({'data': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}), 'data', pd.DataFrame({'data': []})),
    (pd.DataFrame({'data': [1, np.nan, 2, 3, 100, 5, 6, 7, 8, 9, 10]}), 'data', pd.DataFrame({'data': [100]})),
    (pd.DataFrame({'data': [-5, 0, 1, 2, 3, 100, 5, 6, 7, 8, 9, 10]}), 'data', pd.DataFrame({'data': [100]}))
])
def test_modified_z_score(input_df, column_name, expected_output):
    assert modified_z_score(input_df, column_name, drop_nan=True, discard_negatives=True,
                            discard_zeros=True)[column_name].tolist() == expected_output[column_name].tolist()


@pytest.mark.parametrize("input_df, column_name, expected_output", [
    (pd.DataFrame({'data': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}), 'data', pd.DataFrame(columns=['data'])),
    (pd.DataFrame({'data': [1, 2, np.nan, 3, 100, 5, 6, 7, 8, 9, 10]}), 'data', pd.DataFrame(columns=['data'])),
    (pd.DataFrame({'data': [-5, 0, 1, 2, 3, 100, 5, 6, 7, 8, 9, 10]}), 'data', pd.DataFrame(columns=['data']))
])
def test_intersection_of_outliers_found_by_iqr_z_score_and_modified_z_score(input_df, column_name, expected_output):
    assert intersection_of_outliers_found_by_iqr_z_score_and_modified_z_score(input_df, column_name,
                                                                              drop_nan=True, discard_negatives=True,
                                                                              discard_zeros=True)[column_name]\
               .tolist() == expected_output[column_name].tolist()



@pytest.mark.parametrize("input_number, expected_output", [
    # Normal cases
    ("+1-123-456 7890", "+1-123-456 7890"),
    ("(123) 456-789", "123 456-789"),
    (" +123 456 789", "+123 456 789"),
    ('+49 123 456789', '+49 123 456789'),
    ('+49-987-654321', '+49-987-654321'),

    # Edge cases
    ("+1-abc-123", "+1-123"),  # Corrected expected output
    ("(123) 456-78a90", "123 456-7890"),
    ("", ""),  # Empty string
    ("abcdefgh", ""),  # Only characters
    ("!@#$$%&*()", ""),  # Only special characters
    ("!@#$$%&*()1234567890", "1234567890"),  # Special characters with numbers
    ("+1 123 456 7890", "+1 123 456 7890"),  # Already sanitized
    ("+1 (123) 456-7890", "+1 123 456-7890"),
    ("+44 20 7123 4567", "+44 20 7123 4567"),
    ("123.456.7890", "1234567890"),
    ("123-456-7890", "123-456-7890"),
    ("+91 98765-43210", "+91 98765-43210"),
    ("+   1  2 3", "+1 2 3"),  # Multiple spaces

    # Edge cases with plus at the beginning, in the middle or at the end
    ("+1-abc-123++", "+1-123"),
    ("(123) +456-78a90", "123 456-7890"),
    ("+1 12+34 56+7890", "+1 1234 567890"),
    ("123/456+7890", "1234567890"),
    ("123\\456+\\7890", "1234567890"),
    ("+1 (123)/456\\-78a90+++", "+1 123456-7890"),
    ("123+456", "123456"),
    ("+123+456", "+123456"),
    (" +123 + 456", "+123 456"),

    # Cases with / and \
    ("123/456/7890", "1234567890"),
    ("123\\456\\7890", "1234567890"),
    ("+1 (123)/456\\-78a90", "+1 123456-7890"),  # Corrected expected output

    # Other cases
    ("a+123b 456-789", "+123 456-789"),
    (" -- 123- 34-- 56 --45-- - -", "123-34-56-45"),
    (" --123  34- -34 -34 - -1  2--", "123 34-34-34-1 2"),
    (' +49  123-456/789+34', '+49 123-45678934'),
    ('+123--456----789', '+123-456-789'),
    ('a-+123-456--78', '+123-456-78'),
    ("-123456789", "123456789"),
    ("--12345678--", "12345678"),
    ("abc +123- 456 def", "+123-456"),
    ("-d - a- + -123 + 456 a 789-c-0 e--21", "+123 456 789-0-21"),

    # Cases where number string contains no digit, a single digit, or a combination of digits, pluses and hyphens
    ("+++++++", ""),
    ("- - - --  --", ""),
    ("- - + - -- +  --", ""),
    ("++++++5", "+5"),
    ("-- --  1 +++", "1"),
    ("++- - +++3+++ - - ++ - - ", "+3"),
    ("+-1+-2++3--", "+1-23")
])
def test_sanitize_phone_or_fax_number(input_number, expected_output):
    # NOTE: here, the numbers have variable lengths, whereas in reality,
    # the lengths of the numbers are limited by the factory's model declarations
    assert sanitize_phone_or_fax_number(input_number) == expected_output
    
    
    

@pytest.mark.parametrize("iterable, expected_result", [
    (['a', 'b', 'c', 'd', 'b', 'a'], ['a', 'b', 'c', 'd']),
    ((45, 92, 23, 12, 56, 11, 78, 92, 11, 24, 55, 67, 56, 23, 78), [45, 92, 23, 12, 56, 11, 78, 24, 55, 67]),
    (pd.Series((np.NaN, np.NaN, np.NaN, 5, 6, 7, 1)), [np.nan, 5.0, 6.0, 7.0, 1.0]),
    (np.array([None, 1, 6, np.NaN, np.NaN, np.NaN, 5, 6, 7, 1, None]), [np.nan, 1, 6, 5, 7])
])
def test_get_unique_insertion_order_list(iterable, expected_result):
    result = get_unique_insertion_order_list(iterable)
    if pd.isnull(expected_result[0]):
        assert pd.isnull(result[0]) and expected_result[1:] == result[1:]
    else:
        assert expected_result == result


@pytest.mark.parametrize("input_data, expected_output", [
    (['a', 'b', 'c', 'd', 'b', 'a'], {'a', 'b', 'c', 'd'}),
    ((45, 92, 23, 12, 56, 11, 78, 92, 11, 24, 55, 67, 56, 23, 78), {67, 11, 12, 45, 78, 55, 23, 56, 24, 92}),
    (pd.Series((np.NaN, np.NaN, np.NaN, 5, 6, 7, 1)), {np.nan, 1.0, 5.0, 6.0, 7.0}),
    (np.array([None, 1, 6, np.NaN, np.NaN, np.NaN, 5, 6, 7, 1, None]), {np.nan, 1, 5, 6, 7}),
])
def test_get_set_of_unique_elements(input_data, expected_output):
    actual_output = get_set_of_unique_elements(input_data)
    assert actual_output == expected_output, f"For {input_data}, expected {expected_output} but got {actual_output}"