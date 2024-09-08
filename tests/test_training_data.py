"""
tests used to audit the files in the data-science/src folder
"""
# pylint: disable=W1203 # fstrings in logs
# pylint: disable=C0301 # line over 100 chars
# pylint: disable=W0621 # redefining from outer scope triggering on pytest fixtures

import sys
import os
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pytest
from dreams_core import core as dc

# pylint: disable=E0401 # can't find import
# pylint: disable=C0413 # import not at top of doc
# import training_data python functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import training_data as td

load_dotenv()
logger = dc.setup_logger()


# -------------------------- #
# Tests for retrieve_transfers_data()
# -------------------------- #

@pytest.mark.slow
def test_transfers_data_quality():
    """
    Retrieves transfers_df and performs comprehensive data quality checks.
    """
    logger.info("Testing transfers_df from retrieve_transfers_data()...")

    # Example modeling period start date
    modeling_period_start = '2024-03-01'

    # Retrieve transfers_df
    transfers_df = td.retrieve_transfers_data(modeling_period_start)

    # Test 1: No duplicate records
    # ----------------------------
    deduped_df = transfers_df[['coin_id', 'wallet_address', 'date']].drop_duplicates()
    logger.info(f"Original transfers_df length: {len(transfers_df)}, Deduplicated: {len(deduped_df)}")
    assert len(transfers_df) == len(deduped_df), "There are duplicate rows based on coin_id, wallet_address, and date"

    # Test 2: All coin-wallet pairs have a record at the end of the training period
    # ----------------------------------------------------------------------------
    transfers_df_filtered = transfers_df[transfers_df['date'] < modeling_period_start]
    pairs_in_training_period = transfers_df_filtered[['coin_id', 'wallet_address']].drop_duplicates()
    training_period_end = pd.to_datetime(modeling_period_start) - pd.Timedelta(1, 'day')
    period_end_df = transfers_df[transfers_df['date'] == training_period_end]

    logger.info(f"Found {len(pairs_in_training_period)} total pairs in training period with {len(period_end_df)} having data at period end.")
    assert len(pairs_in_training_period) == len(period_end_df), "Not all training data coin-wallet pairs have a record at the end of the training period"

    # Test 3: No negative balances
    # ----------------------------
    # the threshold is set to -0.1 to account for rounding errors from the dune ingestion pipeline
    negative_balances = transfers_df[transfers_df['balance'] < -0.1]
    logger.info(f"Found {len(negative_balances)} records with negative balances.")
    assert len(negative_balances) == 0, "There are negative balances in the dataset"

    # Test 4: Date range check
    # ------------------------
    min_date = transfers_df['date'].min()
    max_date = transfers_df['date'].max()
    expected_max_date = pd.to_datetime(modeling_period_start) - pd.Timedelta(1, 'day')
    logger.info(f"Date range: {min_date} to {max_date}")
    assert max_date == expected_max_date, f"The last date in the dataset should be {expected_max_date}"

    # Test 5: No missing values
    # -------------------------
    missing_values = transfers_df.isnull().sum()
    logger.info(f"Missing values:\n{missing_values}")
    assert missing_values.sum() == 0, "There are missing values in the dataset"

    # Test 6: Balance consistency
    # ---------------------------
    transfers_df['balance_change'] = transfers_df.groupby(['coin_id', 'wallet_address'])['balance'].diff()
    transfers_df['expected_change'] = transfers_df['net_transfers']

    # Calculate the difference between balance_change and expected_change
    transfers_df['diff'] = transfers_df['balance_change'] - transfers_df['expected_change']

    # Define a threshold for acceptable discrepancies (e.g., 1e-8)
    # currently set to 0.1 for coins with values e+13 and e+14 that are showing rounding issues
    threshold = 0.1

    # Find inconsistent balances, ignoring the first record for each coin-wallet pair
    # and allowing for small discrepancies
    inconsistent_balances = transfers_df[
        (~transfers_df['balance_change'].isna()) &  # Ignore first records
        (transfers_df['diff'].abs() > threshold)    # Allow small discrepancies
    ]

    if len(inconsistent_balances) > 0:
        logger.warning(f"Found {len(inconsistent_balances)} records with potentially inconsistent balance changes.")
        logger.warning("Sample of inconsistent balances:")
        logger.warning(inconsistent_balances.head().to_string())
    else:
        logger.info("No significant balance inconsistencies found.")

    # Instead of a hard assertion, we'll log a warning if inconsistencies are found
    # This allows the test to pass while still alerting us to potential issues
    assert len(inconsistent_balances) == 0, f"Found {len(inconsistent_balances)} records with potentially inconsistent balance changes. Check logs for details."

    logger.info("All data quality checks passed successfully.")



# ------------------------------- #
# Tests for calculate_wallet_profitability()
# ------------------------------- #

# the follow are tests for function calculate_wallet_profitability()
@pytest.fixture
def sample_transfers_df():
    """
    Fixture that provides a sample DataFrame for transfers data.

    Returns:
    pd.DataFrame: A DataFrame containing sample transfer data with columns:
    'coin_id', 'wallet_address', 'date', 'net_transfers', and 'balance'.
    """
    return pd.DataFrame({
        'coin_id': ['BTC', 'BTC', 'ETH', 'ETH'],
        'wallet_address': ['wallet1', 'wallet1', 'wallet2', 'wallet2'],
        'date': [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 1), datetime(2023, 1, 2)],
        'net_transfers': [10, -5, 20, 10],
        'balance': [10, 5, 20, 30]
    })

@pytest.fixture
def sample_prices_df():
    """
    Fixture that provides a sample DataFrame for price data.

    Returns:
    pd.DataFrame: A DataFrame containing sample price data with columns:
    'coin_id', 'date', and 'price'.
    """
    return pd.DataFrame({
        'coin_id': ['BTC', 'BTC', 'ETH', 'ETH'],
        'date': [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 1), datetime(2023, 1, 2)],
        'price': [100, 120, 50, 55]
    })

def test_happy_path(sample_transfers_df, sample_prices_df):
    """
    Test the happy path scenario with sample data.
    This test also covers the case where all necessary prices are present.
    """
    result = td.calculate_wallet_profitability(sample_transfers_df, sample_prices_df)

    assert len(result) == 4
    assert 'profitability_change' in result.columns
    assert 'profitability_cumulative' in result.columns
    assert not result['price'].isnull().any(), "There should be no missing prices in the result"
    assert len(result) == len(sample_transfers_df), "The result should have the same number of rows as the input transfers DataFrame"

    # Check profitability calculations for BTC wallet1
    btc_wallet1 = result[(result['coin_id'] == 'BTC') & (result['wallet_address'] == 'wallet1')]
    assert btc_wallet1['profitability_change'].tolist() == [0, 200]  # 0 for day 1, (120-100)*10 for day 2
    assert btc_wallet1['profitability_cumulative'].tolist() == [0, 200]

def test_empty_dataframes_raise_exception():
    """
    Test that the function raises a ValueError when given empty DataFrames.
    """
    empty_df = pd.DataFrame(columns=['coin_id', 'wallet_address', 'date', 'net_transfers', 'balance'])
    empty_prices = pd.DataFrame(columns=['coin_id', 'date', 'price'])

    with pytest.raises(ValueError, match="Input DataFrames cannot be empty"):
        td.calculate_wallet_profitability(empty_df, empty_prices)

def test_missing_price_data(sample_transfers_df):
    """
    Test that the function raises a ValueError when there are missing prices.
    """
    incomplete_prices_df = pd.DataFrame({
        'coin_id': ['BTC', 'ETH'],
        'date': [datetime(2023, 1, 1), datetime(2023, 1, 1)],
        'price': [100, 50]
    })

    with pytest.raises(ValueError, match="Missing prices found for some transfer dates"):
        td.calculate_wallet_profitability(sample_transfers_df, incomplete_prices_df)

def test_price_decline(sample_transfers_df):
    """
    Test profitability calculation when prices decline.
    """
    declining_prices_df = pd.DataFrame({
        'coin_id': ['BTC', 'BTC', 'ETH', 'ETH'],
        'date': [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 1), datetime(2023, 1, 2)],
        'price': [100, 80, 50, 40]
    })

    result = td.calculate_wallet_profitability(sample_transfers_df, declining_prices_df)
    btc_wallet1 = result[(result['coin_id'] == 'BTC') & (result['wallet_address'] == 'wallet1')]
    assert btc_wallet1['profitability_change'].tolist() == [0, -200]  # 0 for day 1, (80-100)*10 for day 2
    assert btc_wallet1['profitability_cumulative'].tolist() == [0, -200]

def test_large_numbers():
    """
    Test with very large numbers to check for potential overflow issues.
    """
    large_transfers_df = pd.DataFrame({
        'coin_id': ['BTC'],
        'wallet_address': ['wallet1'],
        'date': [datetime(2023, 1, 1)],
        'net_transfers': [1e9],
        'balance': [1e9]
    })
    large_prices_df = pd.DataFrame({
        'coin_id': ['BTC'],
        'date': [datetime(2023, 1, 1)],
        'price': [1e6]
    })

    result = td.calculate_wallet_profitability(large_transfers_df, large_prices_df)
    assert not result['profitability_cumulative'].isnull().any()
    assert not np.isinf(result['profitability_cumulative']).any()


# ------------------------------- #
# Tests for clean_profits_df()
# ------------------------------- #

@pytest.fixture
def sample_profits_df():
    """
    dummy version of profits_df output by calculate_wallet_profitability()
    """
    data = {
        'coin_id': ['BTC', 'BTC', 'ETH', 'ETH', 'BTC', 'ETH'],
        'wallet_address': ['addr1', 'addr1', 'addr2', 'addr2', 'addr3', 'addr3'],
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02', '2023-01-01', '2023-01-01']),
        'profitability_cumulative': [5000000.0, 8000000.0, 15000000.0, -12000000.0, 3000000.0, 9000000.0]
    }
    return pd.DataFrame(data)

def test_clean_profits_df_custom_filter(sample_profits_df):
    """
    Test the clean_profits_df function with a custom profitability filter.

    This test case verifies that:
    1. The function correctly filters out coin_id-wallet_address pairs where any day's
       profitability exceeds the custom threshold of 5,000,000 (positive or negative).
    2. The cleaned DataFrame contains only the expected records.
    3. The exclusions DataFrame contains the correct pairs that were filtered out.

    Expected behavior:
    - BTC-addr1 pair should be excluded (profitability > 5,000,000)
    - ETH-addr2 pair should be excluded (profitability > 5,000,000 and < -5,000,000)
    - ETH-addr3 pair should be excluded (profitability > 5,000,000)
    - Only BTC-addr3 pair should remain in the cleaned DataFrame

    Args:
        sample_profits_df (pd.DataFrame): Fixture providing a sample DataFrame for testing.
    """
    # Suppress the specific DeprecationWarning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, message="np.find_common_type is deprecated")
        cleaned_df, exclusions_df = td.clean_profits_df(sample_profits_df, profitability_filter=5000000)

    assert len(cleaned_df) == 1  # Only BTC-addr3 pair should remain
    assert len(exclusions_df) == 3  # All pairs except BTC-addr3 should be excluded
    assert set(cleaned_df['wallet_address']) == {'addr3'}
    assert set(exclusions_df['wallet_address']) == {'addr1', 'addr2', 'addr3'}
