"""
tests used to audit the files in the data-science/src folder
"""
# pylint: disable=C0301 # line over 100 chars
# pylint: disable=C0303 # trailing whitespace
# pylint: disable=C0413 # import not at top of doc (due to local import)
# pylint: disable=C0116 # missing docstring
# pylint: disable=W1203 # fstrings in logs
# pylint: disable=W0612 # unused variables (due to test reusing functions with 2 outputs)
# pylint: disable=W0621 # redefining from outer scope triggering on pytest fixtures
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
import coin_wallet_metrics as cwm
from utils import load_config

load_dotenv()
logger = dc.setup_logger()





# ===================================================== #
#                                                       #
#                 U N I T   T E S T S                   #
#                                                       #
# ===================================================== #




# ---------------------------------------- #
# classify_wallet_cohort() unit tests
# ---------------------------------------- #

# Sample profits data
def sample_wallet_cohort_profits_df():
    """
    Sample DataFrame for testing classify_wallet_cohort function
    """
    data = {
        'coin_id': ['coin_1', 'coin_1', 'coin_1', 'coin_2', 'coin_3', 'coin_1'],
        'wallet_address': ['wallet_1', 'wallet_2', 'wallet_2', 'wallet_1', 'wallet_3', 'wallet_4'],
        'date': ['2024-02-15', '2024-02-18', '2024-02-20', '2024-02-18', '2024-02-25', '2024-02-20'],
        'usd_inflows': [5000, 7500, 7500, 8000, 20000, 1000],
        'profits_cumulative': [3000, 2000, 8000, 6000, 9000, 500]
    }

    # Recompute total return: total_return = profits_cumulative / usd_inflows
    df = pd.DataFrame(data)
    df['total_return'] = df['profits_cumulative'] / df['usd_inflows']
    return df

# Sample config for wallet cohort
def sample_wallet_cohort_config():
    """
    Sample configuration for testing classify_wallet_cohort function
    """
    return {
        'wallet_minimum_inflows': 10000,
        'wallet_maximum_inflows': 50000,
        'coin_profits_win_threshold': 5000,  # Coin must have profits of at least 5000 USD to be a win
        'coin_return_win_threshold': 0.5,  # Coin must have at least a 50% return to be a win
        'wallet_min_coin_wins': 1  # Minimum of 1 coin must meet the "win" threshold for the wallet to join the cohort
    }

# Test case for classify_wallet_cohort
def test_wallet_cohort_classification():
    """
    Unit test for classify_wallet_cohort() function with assertions for specific items.
    """
    sample_profits_df = sample_wallet_cohort_profits_df()
    sample_config = sample_wallet_cohort_config()

    # Run classification
    cohort_wallets_df = cwm.classify_wallet_cohort(sample_profits_df, sample_config)

    # Test 1: Ensure wallets are filtered correctly based on inflows eligibility criteria.
    eligible_wallets = cohort_wallets_df['wallet_address'].unique()
    assert 'wallet_4' not in eligible_wallets, "Wallet_4 should be excluded due to insufficient inflows."
    assert 'wallet_2' in eligible_wallets, "Wallet_2 should be included."
    assert 'wallet_3' in eligible_wallets, "Wallet_3 should be included."

    # Test 2: Verify that wallets are classified based on profits win threshold.
    wallet_1_wins = cohort_wallets_df[cohort_wallets_df['wallet_address'] == 'wallet_1']['winning_coins'].values[0]
    wallet_2_wins = cohort_wallets_df[cohort_wallets_df['wallet_address'] == 'wallet_2']['winning_coins'].values[0]
    assert wallet_1_wins == 1, f"Expected 1 winning coin for Wallet_1, got {wallet_1_wins}"
    assert wallet_2_wins == 1, f"Expected 1 winning coin for Wallet_2, got {wallet_2_wins}"

    # Test 3: Ensure wallets are classified based on combined profits and return thresholds.
    wallet_2_is_cohort = cohort_wallets_df[cohort_wallets_df['wallet_address'] == 'wallet_2']['in_cohort'].values[0]
    assert wallet_2_is_cohort, "Wallet_2 should be classified as a cohort member."

    # Test 5: Check summary metrics for wallet_1
    wallet_1_metrics = cohort_wallets_df[cohort_wallets_df['wallet_address'] == 'wallet_1']
    assert wallet_1_metrics['usd_inflows'].values[0] == 13000, f"Expected total inflows for Wallet_1 to be 13000, got {wallet_1_metrics['usd_inflows'].values[0]}"
    assert wallet_1_metrics['total_coins'].values[0] == 2, f"Expected total coins for Wallet_1 to be 2, got {wallet_1_metrics['total_coins'].values[0]}"
    assert wallet_1_metrics['total_profits'].values[0] == 9000, f"Expected total profits for Wallet_1 to be 9000, got {wallet_1_metrics['total_profits'].values[0]}"




# ------------------------------------------ #
# test_generate_buysell_metrics_df() unit tests
# ------------------------------------------ #

# Updated mock data fixture for profits_df with wallet5 included
@pytest.fixture
def mock_profits_df():
    data = {
        'wallet_address': [
            'wallet1', 'wallet1', 'wallet1', 'wallet1', 'wallet1',  # wallet1 transactions (coin1)
            'wallet2', 'wallet2', 'wallet2', 'wallet2', 'wallet2',  # wallet2 transactions (coin2)
            'wallet3', 'wallet3', 'wallet3', 'wallet3', 'wallet3',  # wallet3 transactions (coin3)
            'wallet1', 'wallet1',  # wallet1 transactions (coin4 - outside cohort)
            'wallet4', 'wallet4',  # wallet4 transactions (coin1 - outside cohort)
            'wallet5', 'wallet5', 'wallet5'  # wallet5 transactions (coin1 and coin2)
        ],
        'coin_id': [
            'coin1', 'coin1', 'coin1', 'coin1', 'coin1',  # coin1 (wallet1)
            'coin2', 'coin2', 'coin2', 'coin2', 'coin2',  # coin2 (wallet2)
            'coin3', 'coin3', 'coin3', 'coin3', 'coin3',  # coin3 (wallet3)
            'coin4', 'coin4',  # coin4 (wallet1 - outside cohort)
            'coin1', 'coin1',  # wallet4 coin1 (outside cohort)
            'coin1', 'coin2', 'coin2'  # wallet5 purchases: coin1 (1/1/24), coin2 (1/2/24 and 1/3/24)
        ],
        'date': [
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',  # wallet1 coin1
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',  # wallet2 coin2
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',  # wallet3 coin3
            '2024-01-01', '2024-01-02',  # wallet1 coin4
            '2024-01-01', '2024-01-02',  # wallet4 coin1 (outside cohort)
            '2024-01-01', '2024-01-02', '2024-01-03'  # wallet5 transactions
        ],
        'balance': [
            100, 130, 230, 230, 220,  # wallet1 (coin1)
            200, 180, 230, 190, 190,  # wallet2 (coin2)
            50, 60, 40, 60, 60,  # wallet3 (coin3)
            400, 425,  # wallet1 (coin4 - outside cohort)
            600, 620,  # wallet4 coin1 (outside cohort)
            100, 200, 200  # wallet5 coin1 and coin2 purchases
        ],
        'net_transfers': [
            100, +30, +100, 0, -10,  # wallet1 (coin1)
            200, -20, +50, -40, 0,  # wallet2 (coin2)
            50, +10, -20, +20, 0,  # wallet3 (coin3)
            400, +25,  # wallet1 (coin4 - outside cohort)
            600, +20,  # wallet4 coin1 (outside cohort)
            100, 200, 200  # wallet5 transactions
        ]
    }

    return pd.DataFrame(data)

@pytest.mark.unit
def test_unit_generate_buysell_metrics_df(mock_profits_df):
    """
    tests the generation of buysell metrics for a wallet-coin cohort
    """
    cohort_wallets = ['wallet1', 'wallet2', 'wallet3', 'wallet5']  # Include wallet5
    training_period_end = '2024-01-05'  # Set a training period end date

    # Call the function
    result_df = cwm.generate_buysell_metrics_df(mock_profits_df, training_period_end, cohort_wallets)

    # Test the output structure
    expected_columns = [
        'date', 'buyers_new', 'buyers_repeat', 'total_buyers', 'sellers_new', 'sellers_repeat',
        'total_sellers', 'total_bought', 'total_sold', 'total_net_transfers', 'total_volume',
        'total_holders', 'total_balance', 'coin_id'
    ]

    for col in expected_columns:
        assert col in result_df.columns, f"Missing column: {col}"

    # Assertions for wallet5:
    # buyers_new for coin1 on 1/1/24 should be 2 (wallet1 and wallet5)
    assert result_df[(result_df['coin_id'] == 'coin1') & (result_df['date'] == '2024-01-01')]['buyers_new'].iloc[0] == 2

    # buyers_new for coin2 on 1/2/24 should be 1 (wallet5)
    assert result_df[(result_df['coin_id'] == 'coin2') & (result_df['date'] == '2024-01-02')]['buyers_new'].iloc[0] == 1

    # buyers_repeat for coin2 on 1/3/24 should be 2 (wallet2 and wallet5)
    assert result_df[(result_df['coin_id'] == 'coin2') & (result_df['date'] == '2024-01-03')]['buyers_repeat'].iloc[0] == 2

    # Filter mock_profits_df to only include cohort wallets
    cohort_profits_df = mock_profits_df[mock_profits_df['wallet_address'].isin(cohort_wallets)]

    # total_bought should match the sum of positive net_transfers in cohort_profits_df
    total_bought_mock = cohort_profits_df[cohort_profits_df['net_transfers'] > 0]['net_transfers'].sum()
    total_bought_result = result_df['total_bought'].sum()
    assert total_bought_mock == total_bought_result, f"Total bought does not match: {total_bought_mock} != {total_bought_result}"

    # total_sold should match the sum of absolute values of negative net_transfers in cohort_profits_df
    total_sold_mock = abs(cohort_profits_df[cohort_profits_df['net_transfers'] < 0]['net_transfers'].sum())
    total_sold_result = result_df['total_sold'].sum()
    assert total_sold_mock == total_sold_result, f"Total sold does not match: {total_sold_mock} != {total_sold_result}"

    # Assertions for total_balance of coin2 on all 5 days
    coin2_balances = mock_profits_df[mock_profits_df['coin_id'] == 'coin2'].groupby('date')['balance'].sum()
    for date, expected_balance in coin2_balances.items():
        result_balance = result_df[(result_df['coin_id'] == 'coin2') & (result_df['date'] == date)]['total_balance'].iloc[0]
        assert expected_balance == result_balance, f"Balance mismatch for coin2 on {date}: {expected_balance} != {result_balance}"


# ------------------------------------------ #
# fill_buysell_metrics_df() unit tests
# ------------------------------------------ #

@pytest.mark.unit
def test_fill_buysell_metrics_df():
    """
    Unit test for the fill_buysell_metrics_df function.

    This test checks the following:

    1. Missing dates within the input DataFrame's date range are correctly identified and filled.
    2. Missing rows beyond the latest date (up to the training_period_end) are added with NaN values initially, then filled appropriately.
    3. Forward-filling for 'total_balance' and 'total_holders' works as expected, ensuring that:
       - Values are forward-filled correctly for each coin_id.
       - Any missing dates earlier than the first record for 'total_balance' or 'total_holders' are filled with 0.
    4. 'buyers_new' and other transaction-related columns (e.g., 'total_bought', 'total_sold') are correctly filled with 0 for missing dates.

    Test Case Summary:
    - The test includes three coins: 'coin1', 'coin2', and 'coin3'.
    - 'coin1' has two records: one on 2024-01-01 and one on 2024-01-04, and is missing values for 2024-01-02, 2024-01-03, and 2024-01-05.
    - 'coin2' has two records: one on 2024-01-01 and one on 2024-01-03, and is missing values for 2024-01-02 and 2024-01-05.
    - 'coin3' has a single record on 2024-01-03 and is missing values for 2024-01-04 and 2024-01-05.

    Expected Assertions:
    - The 'total_balance' and 'buyers_new' columns are correctly forward-filled for each coin, ensuring no values leak across coin_ids.
    - For each coin_id, records with missing 'total_balance' or 'total_holders' before the first non-null date are filled with 0.
    - Transaction-related columns (e.g., 'buyers_new') are filled with 0 for any missing dates.
    """
    buysell_metrics_df = pd.DataFrame({
        'coin_id': ['coin1', 'coin1', 'coin2', 'coin2', 'coin3'],
        'date': [pd.Timestamp('2024-01-01'),
                 pd.Timestamp('2024-01-04'),
                 pd.Timestamp('2024-01-01'),
                 pd.Timestamp('2024-01-03'),
                 pd.Timestamp('2024-01-03')
                ],
        'total_balance': [100, 110, 200, None, 300],  # Added coin3 with balance 300 on 2024-01-03
        'total_bought': [50, 20, 75, None, 60],
        'total_sold': [10, 5, 15, None, 20],
        'total_net_transfers': [40, 15, 60, None, 40],
        'total_volume': [100, 35, 150, None, 80],
        'total_holders': [10, 11, 20, None, 30],  # Added coin3 with holders 30 on 2024-01-03
        'buyers_new': [1, 0, 2, None, 3],  # Added coin3 with 3 new buyers on 2024-01-03
        'buyers_repeat': [0, 1, 0, None, 1],
        'total_buyers': [1, 1, 2, None, 4],
        'sellers_new': [0, 1, 1, None, 2],
        'sellers_repeat': [1, 0, 1, None, 1],
        'total_sellers': [1, 1, 2, None, 3]
    })

    training_period_end = pd.Timestamp('2024-01-05')

    # Call the function to fill missing dates and values
    result = cwm.fill_buysell_metrics_df(buysell_metrics_df, training_period_end)

    # Assert total_balance for coin1 is filled correctly
    expected_total_balance_coin1 = [100, 100, 100, 110, 110]
    result_total_balance_coin1 = result[result['coin_id'] == 'coin1']['total_balance'].tolist()
    assert result_total_balance_coin1 == expected_total_balance_coin1, f"Expected {expected_total_balance_coin1}, but got {result_total_balance_coin1}"

    # Assert buyers_new for coin1 is filled correctly
    expected_buyers_new_coin1 = [1, 0, 0, 0, 0]
    result_buyers_new_coin1 = result[result['coin_id'] == 'coin1']['buyers_new'].tolist()
    assert result_buyers_new_coin1 == expected_buyers_new_coin1, f"Expected {expected_buyers_new_coin1}, but got {result_buyers_new_coin1}"

    # Assert total_balance for coin3 is filled correctly
    expected_total_balance_coin3 = [0, 0, 300, 300, 300]
    result_total_balance_coin3 = result[result['coin_id'] == 'coin3']['total_balance'].tolist()
    assert result_total_balance_coin3 == expected_total_balance_coin3, f"Expected {expected_total_balance_coin3}, but got {result_total_balance_coin3}"

    # Assert buyers_new for coin3 is filled correctly
    expected_buyers_new_coin3 = [0, 0, 3, 0, 0]
    result_buyers_new_coin3 = result[result['coin_id'] == 'coin3']['buyers_new'].tolist()
    assert result_buyers_new_coin3 == expected_buyers_new_coin3, f"Expected {expected_buyers_new_coin3}, but got {result_buyers_new_coin3}"



# ------------------------------------------ #
# generate_time_series_indicators() unit tests
# ------------------------------------------ #

@pytest.fixture
def sample_time_series_df():
    """Fixture that provides a sample DataFrame for the time series with multiple coin_ids."""
    data = {
        'coin_id': [1, 1, 1, 2, 2, 2],
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01', '2023-01-02', '2023-01-03'],
        'price': [100, 110, 120, 200, 210, 220]
    }
    df = pd.DataFrame(data)
    return df

@pytest.fixture
def sample_metrics_config():
    """Fixture that provides a sample metrics configuration for time series analysis."""
    return {
        'time_series': {
            'prices': {
                'price': {
                    'metrics': {
                        'sma': {
                            'parameters': {
                                'period': 2
                            }
                        },
                        'ema': {
                            'parameters': {
                                'period': 2
                            }
                        }
                    }
                }
            }
        }
    }

@pytest.fixture
def sample_config():
    """Fixture that provides a sample general config for the date range."""
    return {
        'training_data': {
            'training_period_start': '2023-01-01',
            'training_period_end': '2023-01-03'
        }
    }

@pytest.mark.unit
def test_generate_time_series_indicators_basic_functionality(sample_time_series_df, sample_metrics_config, sample_config):
    """
    Test the basic functionality of generate_time_series_indicators to ensure that SMA and EMA
    are calculated correctly for a simple DataFrame with multiple coin_ids, and that the date
    range filtering works.
    """
    # Convert the date to datetime in the sample data
    sample_time_series_df['date'] = pd.to_datetime(sample_time_series_df['date'])

    # Run the generate_time_series_indicators function
    result_df, _ = cwm.generate_time_series_indicators(
        sample_time_series_df,
        sample_config,
        sample_metrics_config['time_series']['prices']['price']['metrics'],
        value_column='price'
    )

    # Expected columns in the result
    expected_columns = ['coin_id', 'date', 'price', 'price_sma', 'price_ema']

    # Assert that the columns exist in the result
    assert all(col in result_df.columns for col in expected_columns), "Missing expected columns in the result."

    # Assert that SMA and EMA are calculated correctly
    expected_sma_1 = [100.0, 105.0, 115.0]  # SMA for coin_id=1 with period=2
    expected_ema_1 = [100.0, 106.666667, 115.555556]  # EMA for coin_id=1 with period=2

    # Confirm that the SMA result matches the expected, with special logic to handle NaNs
    for i, (expected, actual) in enumerate(zip(
        expected_sma_1,
        result_df[result_df['coin_id'] == 1]['price_sma'].tolist()
    )):
        if np.isnan(expected) and np.isnan(actual):
            continue  # Both values are NaN, so this is considered equal
        assert expected == actual, f"Mismatch at index {i}: expected {expected}, got {actual}"

    # Confirm that the EMA result matches the expected
    assert result_df[result_df['coin_id'] == 1]['price_ema'].tolist() == pytest.approx(
        expected_ema_1,
        abs=1e-2
    ), "EMA calculation incorrect for coin_id=1"

    # Check for another coin_id
    expected_sma_2 = [200.0, 205.0, 215.0]  # SMA for coin_id=2 with period=2
    expected_ema_2 = [200.0, 206.666667, 215.555556]  # EMA for coin_id=2 with period=2

    # Confirm that the SMA result matches the expected, with special logic to handle NaNs
    for i, (expected, actual) in enumerate(zip(
        expected_sma_2,
        result_df[result_df['coin_id'] == 2]['price_sma'].tolist()
    )):
        if np.isnan(expected) and np.isnan(actual):
            continue  # Both values are NaN, so this is considered equal
        assert expected == actual, f"Mismatch at index {i}: expected {expected}, got {actual}"

    # Confirm that the EMA result matches the expected
    assert result_df[result_df['coin_id'] == 2]['price_ema'].tolist() == pytest.approx(
        expected_ema_2,
        abs=1e-2
    ), "EMA calculation incorrect for coin_id=2"

    # Confirm that the output df has the same number of rows as the filtered input df
    assert len(result_df) == len(sample_time_series_df), "Output row count does not match input row count"

@pytest.mark.unit
def test_generate_time_series_indicators_different_periods(sample_time_series_df, sample_config):
    """
    Test the functionality of generate_time_series_indicators with different periods for SMA and EMA.
    """
    # Adjust the sample_metrics_config for different periods
    sample_metrics_config = {
        'time_series': {
            'prices': {
                'price': {
                    'metrics': {
                        'sma': {
                            'parameters': {
                                'period': 3  # Different period for SMA
                            }
                        },
                        'ema': {
                            'parameters': {
                                'period': 2  # Different period for EMA
                            }
                        }
                    }
                }
            }
        }
    }

    # Convert the date to datetime in the sample data
    sample_time_series_df['date'] = pd.to_datetime(sample_time_series_df['date'])

    # Run the generate_time_series_metrics function
    result_df, _ = cwm.generate_time_series_indicators(
        sample_time_series_df,
        sample_config,
        sample_metrics_config['time_series']['prices']['price']['metrics'],
        value_column='price'
    )

    # Expected columns in the result
    expected_columns = ['coin_id', 'date', 'price', 'price_sma', 'price_ema']

    # Assert that the columns exist in the result
    assert all(col in result_df.columns for col in expected_columns), "Missing expected columns in the result."

    # Expected SMA and EMA values for coin_id=1
    expected_sma_1 = [100.0, 105.0, 110.0]  # SMA for coin_id=1 with period=3
    expected_ema_1 = [100.0, 106.666667, 115.555556]  # EMA for coin_id=1 with period=2

    # Confirm that the SMA result matches the expected, with special logic to handle NaNs
    for i, (expected, actual) in enumerate(zip(
        expected_sma_1,
        result_df[result_df['coin_id'] == 1]['price_sma'].tolist()
    )):
        if np.isnan(expected) and np.isnan(actual):
            continue  # Both values are NaN, so this is considered equal
        assert expected == actual, f"Mismatch at index {i}: expected {expected}, got {actual}"

    # Confirm that the EMA result matches the expected
    assert result_df[result_df['coin_id'] == 1]['price_ema'].tolist() == pytest.approx(
        expected_ema_1,
        abs=1e-2
    ), "EMA calculation incorrect for coin_id=1"

    # Expected SMA and EMA values for coin_id=2
    expected_sma_2 = [200.0, 205.0, 210.0]  # SMA for coin_id=2 with period=3
    expected_ema_2 = [200.0, 206.666667, 215.555556]  # EMA for coin_id=2 with period=2

    # Confirm that the SMA result matches the expected, with special logic to handle NaNs
    for i, (expected, actual) in enumerate(zip(
        expected_sma_2,
        result_df[result_df['coin_id'] == 2]['price_sma'].tolist()
    )):
        if np.isnan(expected) and np.isnan(actual):
            continue  # Both values are NaN, so this is considered equal
        assert expected == actual, f"Mismatch at index {i}: expected {expected}, got {actual}"

    # Confirm that the EMA result matches the expected
    assert result_df[result_df['coin_id'] == 2]['price_ema'].tolist() == pytest.approx(
        expected_ema_2,
        abs=1e-2
    ), "EMA calculation incorrect for coin_id=2"

@pytest.mark.unit
def test_generate_time_series_indicators_value_column_does_not_exist(sample_time_series_df, sample_metrics_config, sample_config):
    """
    Test that generate_time_series_indicators raises a KeyError if the value_column does not exist in the time_series_df.
    """
    # Use a value_column that doesn't exist in the DataFrame
    invalid_value_column = 'non_existent_column'

    with pytest.raises(KeyError, match=f"Input DataFrame does not include column '{invalid_value_column}'"):
        cwm.generate_time_series_indicators(
            time_series_df=sample_time_series_df,
            config=sample_config,
            value_column_indicators_config=sample_metrics_config['time_series']['prices']['price']['metrics'],
            value_column=invalid_value_column
        )


@pytest.mark.unit
def test_generate_time_series_indicators_value_column_contains_nan(sample_time_series_df, sample_metrics_config, sample_config):
    """
    Test that generate_time_series_indicators raises a ValueError if the value_column contains null values.
    """
    # Introduce null values into the 'price' column
    sample_time_series_df.loc[0, 'price'] = None

    with pytest.raises(ValueError, match="contains null values"):
        cwm.generate_time_series_indicators(
            time_series_df=sample_time_series_df,
            config=sample_config,
            value_column_indicators_config=sample_metrics_config['time_series']['prices']['price']['metrics'],
            value_column='price'
        )




# ======================================================== #
#                                                          #
#            I N T E G R A T I O N   T E S T S             #
#                                                          #
# ======================================================== #


# ---------------------------------- #
# set up config and module-level fixtures
# ---------------------------------- #

@pytest.fixture(scope="session")
def config():
    """
    Fixture to load the configuration from the YAML file.
    """
    return load_config('tests/test_config/test_config.yaml')

@pytest.fixture(scope="session")
def metrics_config():
    """
    Fixture to load the configuration from the YAML file.
    """
    return load_config('tests/test_config/test_metrics_config.yaml')

@pytest.fixture(scope="session")
def prices_df():
    """
    Fixture to load the prices_df from the fixtures folder.
    """
    return pd.read_csv('tests/fixtures/prices_df.csv')

@pytest.fixture(scope="session")
def cleaned_profits_df():
    """
    Fixture to load the cleaned_profits_df from the fixtures folder.
    """
    return pd.read_csv('tests/fixtures/cleaned_profits_df.csv')


# ---------------------------------------- #
# classify_wallet_cohort() integration tests
# ---------------------------------------- #

@pytest.fixture(scope='session')
def wallet_cohort_df(cleaned_profits_df, config):
    """
    Builds wallet_cohort_df for data quality checks.
    """
    profits_df = cleaned_profits_df
    first_cohort = next(iter(config['datasets']['wallet_cohorts']))
    wallet_cohort_df = cwm.classify_wallet_cohort(profits_df, config['datasets']['wallet_cohorts'][first_cohort])
    return wallet_cohort_df

# Save cohort_summary_df.csv in fixtures/
# ----------------------------------------
def test_save_cohort_summary_df(wallet_cohort_df):
    """
    This is not a test! This function saves a wallet_cohort_df.csv in the fixtures folder
    so it can be used for integration tests in other modules.
    """
    # Save the cleaned DataFrame to the fixtures folder
    wallet_cohort_df.to_csv('tests/fixtures/wallet_cohort_df.csv', index=False)
    logger.info("Saved tests/fixtures/wallet_cohort_df.csv from production data...")


    # Add some basic assertions to ensure the data was saved correctly
    assert wallet_cohort_df is not None
    assert len(wallet_cohort_df) > 0

@pytest.mark.integration
def test_no_duplicate_wallets(wallet_cohort_df):
    """
    Test to assert there are no duplicate wallet addresses in the wallet_cohort_df.
    """

    # Group by coin_id and wallet_address and check for duplicates
    duplicates = wallet_cohort_df.duplicated(subset=['wallet_address'], keep=False)

    # Assert that there are no duplicates in wallet_cohort_df
    assert not duplicates.any(), "Duplicate wallet addresses found in wallet_cohort_df"



# ------------------------------------------- #
# generate_buysell_metrics_df() integration tests
# ------------------------------------------- #

@pytest.fixture(scope="session")
def buysell_metrics_df(cleaned_profits_df, wallet_cohort_df, config):
    """
    Fixture to generate the buysell_metrics_df.
    """
    # Generate inputs for generate_buysell_metrics_df
    cohort_wallets = wallet_cohort_df[wallet_cohort_df['in_cohort']]['wallet_address'].unique()

    # Generate the buysell_metrics_df
    return cwm.generate_buysell_metrics_df(
        cleaned_profits_df,
        config['training_data']['training_period_end'],
        cohort_wallets
    )

# Save buysell_metrics_df.csv in fixtures/
# ----------------------------------------
@pytest.mark.integration
def test_save_buysell_metrics_df(buysell_metrics_df):
    """
    This is not a test! This function saves a buysell_metrics_df.csv in the fixtures folder
    so it can be used for integration tests in other modules.
    """
    # Save the cleaned DataFrame to the fixtures folder
    buysell_metrics_df.to_csv('tests/fixtures/buysell_metrics_df.csv', index=False)

    # Add some basic assertions to ensure the data was saved correctly
    assert buysell_metrics_df is not None
    assert len(buysell_metrics_df) > 0


@pytest.mark.integration
def test_integration_buysell_metrics_df(buysell_metrics_df, cleaned_profits_df, wallet_cohort_df, config):
    """
    Integration test for the buysell_metrics_df fixture.
    Validates the structure and key calculations in the final DataFrame.
    """

    # 1. Validate Structure: Check for expected columns in buysell_metrics_df
    expected_columns = [
        'date', 'buyers_new', 'buyers_repeat', 'total_buyers', 'sellers_new', 'sellers_repeat',
        'total_sellers', 'total_bought', 'total_sold', 'total_net_transfers', 'total_volume',
        'total_holders', 'total_balance', 'coin_id'
    ]
    assert set(expected_columns).issubset(buysell_metrics_df.columns), "Missing expected columns in buysell_metrics_df"

    # 2. Validate Key Feature Calculations
    # Filter the cleaned_profits_df to only include cohort wallets and coins
    cohort_wallets = wallet_cohort_df[wallet_cohort_df['in_cohort']]['wallet_address']

    cohort_profits_df = cleaned_profits_df[
        (cleaned_profits_df['wallet_address'].isin(cohort_wallets)) &
        (cleaned_profits_df['date'] <= config['training_data']['training_period_end'])  # Add date filtering
    ]

    # Check that total_bought matches the sum of positive net_transfers in cohort_profits_df
    total_bought_mock = cohort_profits_df[cohort_profits_df['net_transfers'] > 0]['net_transfers'].sum()
    total_bought_result = buysell_metrics_df['total_bought'].sum()
    assert total_bought_mock == pytest.approx(total_bought_result, rel=1e-9), f"Total bought mismatch: {total_bought_mock} != {total_bought_result}"

    # Check that total_sold matches the sum of negative net_transfers in cohort_profits_df
    total_sold_mock = abs(cohort_profits_df[cohort_profits_df['net_transfers'] < 0]['net_transfers'].sum())
    total_sold_result = buysell_metrics_df['total_sold'].sum()
    assert total_sold_mock == pytest.approx(total_sold_result, rel=1e-9), f"Total sold mismatch: {total_sold_mock} != {total_sold_result}"

    # Check that total_net_transfers matches the net of all net_transfers in cohort_profits_df
    total_net_transfers_mock = cohort_profits_df['net_transfers'].sum()
    total_net_transfers_result = buysell_metrics_df['total_net_transfers'].sum()
    assert total_net_transfers_mock == pytest.approx(total_net_transfers_result, rel=1e-9), f"Total net transfers mismatch: {total_net_transfers_mock} != {total_net_transfers_result}"

    # 3. Data Quality Checks
    # Ensure there are no NaN values in critical columns
    critical_columns = buysell_metrics_df.columns
    for col in critical_columns:
        assert buysell_metrics_df[col].isnull().sum() == 0, f"Found NaN values in {col}"

    # Check that all dates in buysell_metrics_df fall within the expected range
    assert buysell_metrics_df['date'].max() <= pd.to_datetime(config['training_data']['training_period_end']), \
        "Found data beyond the training period end date"
    assert buysell_metrics_df['date'].min() >= pd.to_datetime(config['training_data']['training_period_start']), \
        "Found data before the training period start date"

    # Check for missing dates in each coin-wallet pair up to the training_period_end
    missing_dates = buysell_metrics_df.groupby('coin_id').apply(
        lambda x: pd.date_range(start=x['date'].min(), end=pd.to_datetime(config['training_data']['training_period_end'])).difference(x['date'])
    ,include_groups=False)
    if any(len(missing) > 0 for missing in missing_dates):
        raise ValueError("Timeseries contains missing dates. Ensure all dates are filled up to the training_period_end before calling flatten_coin_date_df().")


# ------------------------------------------- #
# generate_buysell_metrics_df() integration tests
# ------------------------------------------- #

@pytest.mark.integration
def test_generate_time_series_metrics_no_nulls_and_row_count(prices_df, config, metrics_config):
    """
    Integration test for cwm.generate_time_series_metrics to confirm that:
    1. The returned DataFrame has no null values.
    2. The number of rows in the output matches the input prices_df.
    """
    # Define dataset key and column name
    value_column = 'price'

    # Identify coins that have complete data for the period
    coin_data_range = prices_df.groupby('coin_id')['date'].agg(['min', 'max'])

    # Full duration coins: Data spans the entire training to modeling period
    training_period_start = pd.to_datetime(config['training_data']['training_period_start'])
    training_period_end = pd.to_datetime(config['training_data']['training_period_end'])

    full_duration_days = (training_period_end - training_period_start).days + 1
    full_duration_coins = coin_data_range[
        (coin_data_range['min'] <= config['training_data']['training_period_start']) &
        (coin_data_range['max'] >= config['training_data']['training_period_end'])
    ].index

    expected_rows = len(full_duration_coins) * full_duration_days

    # Run the generate_time_series_indicators function
    full_metrics_df, partial_time_series_metrics_df = cwm.generate_time_series_indicators(
        time_series_df=prices_df,
        config=config,
        value_column_indicators_config=metrics_config['time_series']['market_data']['price']['indicators'],
        value_column=value_column
    )

    # Check that the number of rows in the result matches the expected number of rows
    assert len(full_metrics_df) == expected_rows, "The number of rows in the output does not match the expected number of rows."

    # Check that there are no null values in the result
    assert not full_metrics_df.isnull().values.any(), "The output DataFrame contains null values."
