"""
tests used to audit the files in the data-science/src folder
"""
# pylint: disable=W1203 # fstrings in logs
# pylint: disable=C0302 # file over 1000 lines
# pylint: disable=E0401 # can't find import (due to local import)
# pylint: disable=C0413 # import not at top of doc (due to local import)
# pylint: disable=W0612 # unused variables (due to test reusing functions with 2 outputs)
# pylint: disable=W0621 # redefining from outer scope triggering on pytest fixtures

import sys
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pytest
from dreams_core import core as dc

# pyright: reportMissingImports=false
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import training_data.data_retrieval as dr

load_dotenv()
logger = dc.setup_logger()


# ===================================================== #
#                                                       #
#                 U N I T   T E S T S                   #
#                                                       #
# ===================================================== #




# ---------------------------------------- #
# impute_market_cap() unit tests
# ---------------------------------------- #

@pytest.mark.unit
def test_all_market_cap_present(mocker):
    """
    Test that when all 'market_cap' values are present (coverage = 1),
    the 'market_cap_imputed' column matches the original 'market_cap' values
    and no warnings are logged.
    """
    # Mock the root logger's warning method to track if any warnings are emitted
    mock_logger = mocker.patch('logging.warning')

    # Define the input DataFrame with full market_cap coverage
    input_data = {
        'coin_id': ['BTC', 'BTC', 'ETH', 'ETH'],
        'date': ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02'],
        'price': [30000, 31000, 2000, 2100],
        'market_cap': [600000000, 620000000, 400000000, 420000000]
    }
    input_df = pd.DataFrame(input_data)

    # Define the expected DataFrame with 'market_cap_imputed' identical to 'market_cap'
    expected_market_cap_imputed = input_df['market_cap'].astype('Int64')
    expected_df = input_df.copy()
    expected_df['market_cap_imputed'] = expected_market_cap_imputed

    # Invoke the impute_market_cap function
    result_df = dr.impute_market_cap(input_df, min_coverage=0.7)

    # Assert that 'market_cap_imputed' matches the original 'market_cap' values
    assert np.allclose(
        result_df['market_cap_imputed'],
        expected_df['market_cap_imputed'],
        equal_nan=True
    ), "The 'market_cap_imputed' does not match the original 'market_cap' values."

    # Assert that the 'market_cap_imputed' column is of type Int64
    assert result_df['market_cap_imputed'].dtype == 'Int64', (
        "'market_cap_imputed' column is not of type Int64."
    )

    # Assert that no warnings were logged
    mock_logger.assert_not_called()


@pytest.mark.unit
def test_market_cap_below_min_coverage(mocker):
    """
    Test that when a coin's 'market_cap' coverage is below 'min_coverage',
    it is excluded from imputation, and 'market_cap_imputed' matches the original
    'market_cap' values without any warnings being logged.
    """
    # Mock the root logger's warning method to track if any warnings are emitted
    mock_logger = mocker.patch('logging.warning')

    # Define the input DataFrame with partial market_cap coverage below min_coverage
    input_data = {
        'coin_id': ['BTC', 'BTC', 'ETH', 'ETH'],
        'date': ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02'],
        'price': [30000, 31000, 2000, 2100],
        'market_cap': [np.nan, np.nan, 400000000, np.nan]
    }
    input_df = pd.DataFrame(input_data)

    # Define the expected 'market_cap_imputed' column
    # Since coverage < min_coverage, no imputation occurs
    # 'market_cap_imputed' should match 'market_cap' where present and remain NaN otherwise
    expected_market_cap_imputed = input_df['market_cap'].astype('Int64')
    expected_df = input_df.copy()
    expected_df['market_cap_imputed'] = expected_market_cap_imputed

    # Invoke the impute_market_cap function with min_coverage=0.7
    result_df = dr.impute_market_cap(input_df, min_coverage=0.7)

    # Assert that 'market_cap_imputed' matches the expected values
    assert np.allclose(
        result_df['market_cap_imputed'],
        expected_df['market_cap_imputed'],
        equal_nan=True
    ), "The 'market_cap_imputed' does not match the expected values when coverage is below min_coverage."

    # Assert that the 'market_cap_imputed' column is of type Int64
    assert result_df['market_cap_imputed'].dtype == 'Int64', (
        "'market_cap_imputed' column is not of type Int64."
    )

    # Assert that no warnings were logged since no imputation should occur
    mock_logger.assert_not_called()


@pytest.mark.unit
def test_imputed_market_cap_exceeds_historical_max():
    """
    Test that when an imputed 'market_cap' exceeds max_multiple times the historical maximum,
    those values are set to np.nan while valid imputed values remain.
    """
    # Define input DataFrame where imputation will cause some values to exceed the threshold
    input_data = {
        'coin_id': ['BTC'] * 5,
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'price': [30000, 30000, 60000, 90000, 30000],  # Price doubles then triples
        'market_cap': [600000000, np.nan, np.nan, np.nan, 600000000]
    }
    input_df = pd.DataFrame(input_data)

    # Define expected values based on max_multiple=2.0:
    # - First and last rows are original values
    # - Second row will be imputed normally (same price = same market cap)
    # - Third row will be imputed but capped (price doubled)
    # - Fourth row will be set to np.nan (price tripled)
    expected_market_cap_imputed = pd.Series(
        [600000000, 600000000, 1200000000, pd.NA, 600000000],
        dtype='Int64'
    )
    expected_df = input_df.copy()
    expected_df['market_cap_imputed'] = expected_market_cap_imputed

    # Invoke function with max_multiple=2.0
    result_df = dr.impute_market_cap(input_df, min_coverage=0.2, max_multiple=2.0)

    # Assert that market_cap_imputed matches expected values
    assert np.allclose(
        result_df['market_cap_imputed'],
        expected_df['market_cap_imputed'],
        equal_nan=True
    ), "The 'market_cap_imputed' values don't correctly handle the max_multiple threshold"

    # Assert that the market_cap_imputed column is of type Int64
    assert result_df['market_cap_imputed'].dtype == 'Int64', (
        "'market_cap_imputed' column is not of type Int64."
    )


@pytest.mark.unit
def test_multiple_coins_varying_coverage():
    """
    Test that impute_market_cap correctly handles multiple coins with varying coverage:
    - Imputes only for coins meeting min_coverage threshold
    - Correctly applies max_multiple threshold to imputed values
    - Preserves original values
    - Excludes coins below coverage threshold
    """
    # Define input DataFrame with multiple coins and varying scenarios
    input_data = {
        'coin_id': ['BTC', 'BTC', 'BTC',  # 2/3 values present = 0.67 coverage
                    'ETH', 'ETH', 'ETH',    # 2/3 values present = 0.67 coverage
                    'DOGE', 'DOGE'],        # 0/2 values present = 0 coverage
        'date': ['2023-01-01', '2023-01-02', '2023-01-03',
                '2023-01-01', '2023-01-02', '2023-01-03',
                '2023-01-01', '2023-01-02'],
        'price': [30000, 30000, 90000,    # BTC price triples on day 3
                    2000, 2100, 2200,        # ETH steady increase
                    0.05, 0.06],             # DOGE excluded due to no coverage
        'market_cap': [600000000, np.nan, np.nan,      # BTC
                        400000000, np.nan, 440000000,       # ETH
                        np.nan, np.nan]                     # DOGE
    }
    input_df = pd.DataFrame(input_data)

    # Define expected values:
    # BTC: Day 2 imputed normally, Day 3's tripled value exceeds max_multiple=2.0
    # ETH: Day 2 imputed normally (within threshold)
    # DOGE: All values remain np.nan due to insufficient coverage
    expected_market_cap_imputed = pd.Series(
        [600000000, 600000000, pd.NA,     # BTC
            400000000, 420000000, 440000000, # ETH
            np.nan, np.nan],                 # DOGE
        dtype='Int64'
    )
    expected_df = input_df.copy()
    expected_df['market_cap_imputed'] = expected_market_cap_imputed

    # Invoke with min_coverage=0.5 and max_multiple=2.0
    result_df = dr.impute_market_cap(input_df, min_coverage=0.2, max_multiple=2.0)

    # Assert that market_cap_imputed matches expected values
    assert np.allclose(
        result_df.sort_values(['coin_id','date'])['market_cap_imputed'],
        expected_df.sort_values(['coin_id','date'])['market_cap_imputed'],
        equal_nan=True
    ), "The market_cap_imputed values don't correctly handle multiple coins"

    # Assert that the market_cap_imputed column is of type Int64
    assert result_df['market_cap_imputed'].dtype == 'Int64', (
        "market_cap_imputed column is not of type Int64."
    )


@pytest.mark.unit
def test_market_cap_missing_at_start():
    """
    Test that when a coin's 'market_cap' is missing at the start of its time series
    but meets the 'min_coverage' threshold, the function imputes the missing values correctly.
    """
    # Define the input DataFrame with 'market_cap' missing at the start for BTC and DOGE
    input_data = {
        'coin_id': ['BTC', 'BTC', 'BTC', 'ETH', 'ETH', 'DOGE', 'DOGE'],
        'date': ['2023-01-01', '2023-01-02', '2023-01-03',
                 '2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02'],
        'price': [30000, 31000, 32000, 2000, 2100, 0.05, 0.06],
        'market_cap': [np.nan, 620000000, 640000000, 400000000, 420000000, np.nan, 6000000]
    }
    input_df = pd.DataFrame(input_data)

    # Define the expected 'market_cap_imputed' values
    expected_market_cap_imputed = pd.Series(
        [np.nan, 620000000, 640000000, 400000000, 420000000, np.nan, 6000000],
        dtype='Int64'
    )
    expected_df = input_df.copy()
    expected_df['market_cap_imputed'] = expected_market_cap_imputed

    # Invoke the impute_market_cap function with min_coverage=0.7
    result_df = dr.impute_market_cap(input_df, min_coverage=0.7)

    # Assert that 'market_cap_imputed' matches the expected values
    assert np.allclose(
        result_df.sort_values(['coin_id','date'])['market_cap_imputed'],
        expected_df.sort_values(['coin_id','date'])['market_cap_imputed'],
        equal_nan=True
    ), "The 'market_cap_imputed' does not correctly handle missing values at the start of the time series."

    # Assert that the 'market_cap_imputed' column is of type Int64 where applicable
    assert result_df['market_cap_imputed'].dtype == 'Int64', (
        "'market_cap_imputed' column is not of type Int64."
    )


@pytest.mark.unit
def test_all_market_cap_missing_coverage_zero():
    """
    Test that when all 'market_cap' values are missing for a coin (coverage = 0),
    the function excludes the coin from imputation, leaving all 'market_cap_imputed' values as NaN.
    """
    # Define the input DataFrame with all 'market_cap' values missing for DOGE
    input_data = {
        'coin_id': ['BTC', 'BTC', 'ETH', 'ETH', 'DOGE', 'DOGE'],
        'date': ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02'],
        'price': [30000, 31000, 2000, 2100, 0.05, 0.06],
        'market_cap': [600000000, 620000000, 400000000, 420000000, np.nan, np.nan]
    }
    input_df = pd.DataFrame(input_data)

    # Define the expected 'market_cap_imputed' values
    expected_market_cap_imputed = pd.Series(
        [600000000, 620000000, 400000000, 420000000, np.nan, np.nan],
        dtype='Int64'
    )
    expected_df = input_df.copy()
    expected_df['market_cap_imputed'] = expected_market_cap_imputed

    # Invoke the impute_market_cap function with min_coverage=0.7
    result_df = dr.impute_market_cap(input_df, min_coverage=0.7)

    # Assert that 'market_cap_imputed' matches the expected values
    assert np.allclose(
        result_df.sort_values(['coin_id','date'])['market_cap_imputed'],
        expected_df.sort_values(['coin_id','date'])['market_cap_imputed'],
        equal_nan=True
    ), "The 'market_cap_imputed' does not correctly handle cases where all 'market_cap' values are missing."

    # Assert that the 'market_cap_imputed' column is of type Int64 where applicable
    assert result_df['market_cap_imputed'].dtype == 'Int64', (
        "'market_cap_imputed' column is not of type Int64."
    )


@pytest.mark.unit
def test_market_cap_missing_intermittently():
    """
    Test that when market_cap values are missing intermittently:
    - Valid imputations within max_multiple threshold are calculated correctly
    - Imputations exceeding max_multiple * historical_max become np.nan
    - Original values are preserved
    - Imputation uses correct price ratios despite gaps
    """
    # Define input DataFrame with intermittent missing values
    # Coverage: 5/8 records = 0.625 (above min_coverage=0.5)
    input_data = {
        'coin_id': ['BTC'] * 8,
        'date': [
            '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04',
            '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08'
        ],
        'price': [
            1000, 1000, 1200, 1300,     # Normal variation
            3500, 1500, 1600, 3000      # Includes spike that will exceed max_multiple
        ],
        'market_cap': [
            20000000, np.nan, 24000000, 26000000,
            np.nan, 30000000, 32000000, np.nan
        ]
    }
    input_df = pd.DataFrame(input_data)

    # Define expected values:
    # - Day 2: Normal imputation (same price as Day 1)
    # - Day 5: Will be np.nan (price spike exceeds max_multiple=2.0 of historical max)
    # - Day 8: Normal imputation
    expected_market_cap_imputed = pd.Series(
        [20000000, 20000000, 24000000, 26000000,
            pd.NA, 30000000, 32000000, 60000000],
        dtype='Int64'
    )
    expected_df = input_df.copy()
    expected_df['market_cap_imputed'] = expected_market_cap_imputed

    # Invoke with min_coverage=0.5 and max_multiple=2.0
    result_df = dr.impute_market_cap(input_df, min_coverage=0.5, max_multiple=2.0)

    # Assert that market_cap_imputed matches expected values
    assert np.allclose(
        result_df['market_cap_imputed'],
        expected_df['market_cap_imputed'],
        equal_nan=True
    ), "The market_cap_imputed values don't correctly handle intermittent missing values with max_multiple threshold"

    # Assert that the market_cap_imputed column is of type Int64
    assert result_df['market_cap_imputed'].dtype == 'Int64', (
        "market_cap_imputed column is not of type Int64."
    )


# ---------------------------------------- #
# clean_profits_df() unit tests
# ---------------------------------------- #
