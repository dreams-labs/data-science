"""
tests used to audit the files in the data-science/src folder
"""
# pylint: disable=C0302 # over 1000 lines
# pylint: disable=C0413 # import not at top of doc (due to local import)
# pylint: disable=W0612 # unused variables (due to test reusing functions with 2 outputs)
# pylint: disable=W0621 # redefining from outer scope triggering on pytest fixtures
# pylint: disable=W1203 # fstrings in logs
# pylint: disable=E0401 # can't find import (due to local import)
# pyright: reportMissingModuleSource=false

import sys
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pytest
from dreams_core import core as dc

# pyright: reportMissingImports=false
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import feature_engineering.time_windows_orchestration as tw

load_dotenv()
logger = dc.setup_logger()

# TODO: update tests to accommodate new time window sequencing

# ===================================================== #
#                                                       #
#                 U N I T   T E S T S                   #
#                                                       #
# ===================================================== #

# ------------------------------------------ #
# join_dataset_all_windows_dfs() unit tests
# ------------------------------------------ #
@pytest.mark.xfail
@pytest.mark.unit
def test_merge_and_fill_training_data_same_coin_ids():
    """
    Test the merge_and_fill_training_data function with two DataFrames
    that both have coin_id values 1, 2, 3 and the 'fill_zeros' strategy.
    """
    # Create mock DataFrames with the same coin_ids
    df1 = pd.DataFrame({
        'coin_id': [1, 2, 3],
        'metric_1': [100, 200, 300]
    })
    df2 = pd.DataFrame({
        'coin_id': [1, 2, 3],
        'metric_2': [400, 500, 600]
    })

    # fill_zeros happy path
    # ---------------------
    # List of tuples (df, fill_strategy, filename), where 'filename' is a placeholder for logging
    input_dfs = [
        (df1, 'fill_zeros', 'file1'),
        (df2, 'fill_zeros', 'file2')
    ]

    # Run the function
    training_data_df, merge_logs_df = tw.join_dataset_all_windows_dfs(input_dfs)

    # Assert that the merged DataFrame matches the expected DataFrame
    expected_df = pd.DataFrame({
        'coin_id': [1, 2, 3],
        'metric_1': [100, 200, 300],
        'metric_2': [400, 500, 600]
    })
    np.array_equal(training_data_df.values,expected_df.values)

    # Assert that the logs match the expected logs
    expected_logs = pd.DataFrame({
        'file': ['file1', 'file2'],
        'original_count': [3, 3],
        'filled_count': [0, 0],
    })
    np.array_equal(merge_logs_df.values,expected_logs.values)

    # drop_records happy path
    # ---------------------
    # Rerun the same function with drop_records and confirm that the output is identical
    input_dfs = [
        (df1, 'drop_records', 'file1'),
        (df2, 'drop_records', 'file2')
    ]

    # Run the function
    training_data_df, merge_logs_df = tw.join_dataset_all_windows_dfs(input_dfs)

    # Assert that the merged DataFrame matches the expected DataFrame
    np.array_equal(training_data_df.values,expected_df.values)

    # Assert that the logs match the expected logs
    np.array_equal(merge_logs_df.values,expected_logs.values)



@pytest.mark.xfail
@pytest.mark.unit
def test_merge_and_fill_training_data_fill_zeros():
    """
    Test that merge_and_fill_training_data correctly applies the 'fill_zeros' strategy for missing coin_ids.
    """
    # Define mock DataFrames
    df1 = pd.DataFrame({'coin_id': [1, 2, 3], 'metric_1': [10, 20, 30]})
    df2 = pd.DataFrame({'coin_id': [2, 3], 'metric_2': [50, 60]})

    # Expected DataFrame
    expected_df = pd.DataFrame({
        'coin_id': [1, 2, 3],
        'metric_1': [10, 20, 30],
        'metric_2': [0, 50, 60]
    })

    # Call the function
    merged_df, merge_logs_df = tw.join_dataset_all_windows_dfs([
        (df1, 'fill_zeros', 'df1'),
        (df2, 'fill_zeros', 'df2')
    ])

    # Assert that the merged DataFrame matches the expected DataFrame
    assert np.array_equal(merged_df.values, expected_df.values), "merged_df has unexpected values."

    # Check logs
    df1_log = merge_logs_df[merge_logs_df['file'] == 'df1']
    df2_log = merge_logs_df[merge_logs_df['file'] == 'df2']

    # df1 has no filling, but 1 dropped coin_id (coin_id 1 missing from df2)
    assert df1_log['filled_count'].iloc[0] == 0, "df1 should have no filled entries."

    # df2 has 1 filled entry for coin_id 1
    assert df2_log['filled_count'].iloc[0] == 1, "df2 should have 1 filled entry."


@pytest.mark.xfail
@pytest.mark.unit
def test_merge_and_fill_training_data_drop_records():
    """
    Test the merge_and_fill_training_data function when the 'drop_records' strategy is used.
    Ensures that the merge works correctly and that filled_count is logged appropriately.
    """
    # Mock DataFrames
    df1 = pd.DataFrame({
        'coin_id': [1, 2, 3],
        'metric_1': [10, 20, 30]
    })

    df2 = pd.DataFrame({
        'coin_id': [2, 3],
        'metric_2': [200, 300]
    })

    # Expected output when drop_records is applied: rows for coin 1 should be dropped
    expected_df = pd.DataFrame({
        'coin_id': [2, 3],
        'metric_1': [20, 30],
        'metric_2': [200, 300]
    })

    # Run the function
    merged_df, logs_df = tw.join_dataset_all_windows_dfs([
        (df1, 'drop_records', 'df1'),
        (df2, 'drop_records', 'df2')
    ])

    # Assert the merged DataFrame is correct
    assert np.array_equal(merged_df.values, expected_df.values), "merged_df has unexpected values."

    # Assert the logs are correct
    # df1 should have no filled rows, and df2 should also have no filled rows (since we used drop_records)
    expected_logs = pd.DataFrame({
        'file': ['df1', 'df2'],
        'original_count': [3, 2],
        'filled_count': [0, 0]
    })

    assert np.array_equal(logs_df.values, expected_logs.values), "merged_df has unexpected values."
