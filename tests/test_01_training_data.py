"""
tests used to audit the files in the data-science/src folder
"""
# pylint: disable=W1203 # fstrings in logs
# pylint: disable=C0301 # line over 100 chars
# pylint: disable=E0401 # can't find import (due to local import)
# pylint: disable=C0413 # import not at top of doc (due to local import)
# pylint: disable=W0612 # unused variables (due to test reusing functions with 2 outputs)
# pylint: disable=W0621 # redefining from outer scope triggering on pytest fixtures

import sys
import os
import pandas as pd
from dotenv import load_dotenv
import pytest
from dreams_core import core as dc

# pyright: reportMissingImports=false
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import training_data as td
from utils import load_config

load_dotenv()
logger = dc.setup_logger()


# ===================================================== #
#                                                       #
#                 U N I T   T E S T S                   #
#                                                       #
# ===================================================== #

# -------------------------------- #
# fill_prices_gaps() unit tests
# -------------------------------- #

@pytest.fixture
def max_gap_days():
    """
    Fixture for max_gap_days parameter, reused across multiple tests.
    """
    return 2

@pytest.mark.unit
def test_no_gaps(max_gap_days):
    """
    Test case for no gaps. Ensure that no changes are made to the input data.
    """
    # Input data
    prices_df_no_gaps = pd.DataFrame({
        'coin_id': ['coin1', 'coin1', 'coin1', 'coin2', 'coin2', 'coin2'],
        'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03',
                                '2024-01-01', '2024-01-02', '2024-01-03']),
        'price': [100, 101, 102, 200, 201, 202]
    })

    # Expected output: Should remain the same as input
    expected_df = prices_df_no_gaps[['date', 'coin_id', 'price']].copy()

    # Run the function
    prices_filled_df, outcomes_df = td.fill_prices_gaps(prices_df_no_gaps, max_gap_days)

    # Reorder columns for comparison
    prices_filled_df = prices_filled_df[['date', 'coin_id', 'price']]

    # Assertions
    pd.testing.assert_frame_equal(prices_filled_df, expected_df)
    assert all(outcomes_df['outcome'] == 'no gaps')

@pytest.mark.unit
def test_gaps_below_max(max_gap_days):
    """
    Test case for a DataFrame with gaps below max_gap_days. Ensure gaps are forward-filled.
    """
    # Input data
    prices_df_gaps_below_max = pd.DataFrame({
        'coin_id': ['coin1', 'coin1', 'coin1', 'coin2', 'coin2'],
        'date': pd.to_datetime(['2024-01-01', '2024-01-03', '2024-01-04',
                                '2024-01-01', '2024-01-03']),
        'price': [100, 102, 103, 200, 202]
    })

    # Expected data after filling
    expected_data = {
        'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04',  # coin1
                                '2024-01-01', '2024-01-02', '2024-01-03']),  # coin2
        'coin_id': ['coin1', 'coin1', 'coin1', 'coin1',
                    'coin2', 'coin2', 'coin2'],
        'price': [100.0, 100, 102, 103,
                  200, 200, 202]
    }
    expected_df = pd.DataFrame(expected_data)[['date', 'coin_id', 'price']]

    # Run the function
    prices_filled_df, outcomes_df = td.fill_prices_gaps(prices_df_gaps_below_max, max_gap_days)

    # Reorder columns for comparison
    prices_filled_df = prices_filled_df[['date', 'coin_id', 'price']]

    # Assertions
    pd.testing.assert_frame_equal(prices_filled_df, expected_df)
    assert all(outcomes_df['outcome'] == 'gaps below threshold')

@pytest.mark.unit
def test_gaps_at_max(max_gap_days):
    """
    Test case for a DataFrame with gaps exactly equal to max_gap_days. Ensure gaps are forward-filled.
    """
    # Input data
    prices_df_gaps_at_max = pd.DataFrame({
        'coin_id': ['coin1', 'coin1', 'coin1', 'coin2', 'coin2'],
        'date': pd.to_datetime(['2024-01-01', '2024-01-03', '2024-01-05',
                                '2024-01-01', '2024-01-03']),
        'price': [100, 102, 103, 200, 202]
    })

    # Expected data after filling
    expected_data = {
        'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',  # coin1
                                '2024-01-01', '2024-01-02', '2024-01-03']),  # coin2
        'coin_id': ['coin1', 'coin1', 'coin1', 'coin1', 'coin1',
                    'coin2', 'coin2', 'coin2'],
        'price': [100.0, 100, 102, 102, 103,
                  200, 200, 202]
    }
    expected_df = pd.DataFrame(expected_data)[['date', 'coin_id', 'price']]

    # Run the function
    prices_filled_df, outcomes_df = td.fill_prices_gaps(prices_df_gaps_at_max, max_gap_days)

    # Reorder columns for comparison
    prices_filled_df = prices_filled_df[['date', 'coin_id', 'price']]

    # Assertions
    pd.testing.assert_frame_equal(prices_filled_df, expected_df)
    assert all(outcomes_df['outcome'] == 'gaps below threshold')

@pytest.mark.unit
def test_gaps_above_max(max_gap_days):
    """
    Test case for a DataFrame with gaps exceeding max_gap_days. Ensure these coins are excluded.
    """
    # Input data
    prices_df_gaps_above_max = pd.DataFrame({
        'coin_id': ['coin1', 'coin1', 'coin1', 'coin2', 'coin2','coin3','coin3','coin3'],
        'date': pd.to_datetime(['2024-01-01', '2024-01-05', '2024-01-06',
                                '2024-01-01', '2024-01-07',
                                '2024-01-01', '2024-01-02','2024-01-04']),
        'price': [100.0, 105, 106, 200, 207, 15, 18, 29]
    })

    # Expected data: No data should be returned since gaps are too large
    expected_data = {
        'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']),
        'coin_id': ['coin3', 'coin3', 'coin3', 'coin3'],
        'price': [15.0, 18, 18, 29]
    }
    expected_df = pd.DataFrame(expected_data)[['date', 'coin_id', 'price']]

    # Run the function
    prices_filled_df, outcomes_df = td.fill_prices_gaps(prices_df_gaps_above_max, max_gap_days)

    # Assertions
    pd.testing.assert_frame_equal(prices_filled_df, expected_df)

@pytest.mark.unit
def test_mixed_gaps(max_gap_days):
    """
    Test case for a DataFrame with a mix of no gaps, gaps below, and gaps above max_gap_days.
    """
    # Input data
    prices_df_mixed_gaps = pd.DataFrame({
        'coin_id': ['coin1', 'coin1', 'coin1', 'coin2', 'coin2', 'coin2', 'coin3', 'coin3'],
        'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03',
                                '2024-01-01', '2024-01-04', '2024-01-05',
                                '2024-01-01', '2024-01-10']),
        'price': [100, 101, 102, 200, 204, 205, 300, 310]
    })

    # Expected data:
    # - coin1 has no gaps.
    # - coin2 has gaps below the threshold (filled for 2024-01-02, 2024-01-03).
    # - coin3 has a gap too large (should be excluded).
    expected_data = {
        'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03',  # coin1
                                '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']),  # coin2
        'coin_id': ['coin1', 'coin1', 'coin1',
                    'coin2', 'coin2', 'coin2', 'coin2', 'coin2'],
        'price': [100.0, 101, 102,
                  200, 200, 200, 204, 205]
    }
    expected_df = pd.DataFrame(expected_data)[['date', 'coin_id', 'price']]

    # Run the function
    prices_filled_df, outcomes_df = td.fill_prices_gaps(prices_df_mixed_gaps, max_gap_days)

    # Reorder columns for comparison
    prices_filled_df = prices_filled_df[['date', 'coin_id', 'price']]

    # Assertions
    pd.testing.assert_frame_equal(prices_filled_df, expected_df)
    assert 'no gaps' in outcomes_df['outcome'].values
    assert 'gaps below threshold' in outcomes_df['outcome'].values
    assert 'gaps above threshold' in outcomes_df['outcome'].values


# ---------------------------------------- #
# calculate_wallet_profitability() unit tests
# ---------------------------------------- #

@pytest.fixture
def sample_transfers_df():
    """
    Create a sample transfers DataFrame for testing.
    """
    data = {
        'coin_id': ['BTC', 'BTC', 'ETH', 'ETH', 'BTC', 'ETH', 'MYRO', 'MYRO', 'MYRO',
                    'BTC', 'ETH', 'BTC', 'ETH', 'MYRO'],
        'wallet_address': ['wallet1', 'wallet1', 'wallet1', 'wallet2', 'wallet2', 'wallet2', 'wallet3', 'wallet3', 'wallet3',
                           'wallet1', 'wallet1', 'wallet2', 'wallet2', 'wallet3'],
        'date': [
            '2023-01-01', '2023-02-01', '2023-01-01', '2023-01-01', '2023-01-01', '2023-02-01',
            '2023-01-01', '2023-02-01', '2023-03-01',
            '2023-04-01', '2023-04-01', '2023-04-01', '2023-04-01', '2023-04-01'
        ],
        'net_transfers': [10.0, 5, 100, 50, 20, 25, 1000, 500, -750,
                          0, 0, 0, -10, 0],
        'balance': [10.0, 15, 100, 50, 20, 75, 1000, 1500, 750,
                    15, 100, 20, 65, 750]
    }
    df = pd.DataFrame(data)

    # Convert coin_id to categorical and date to datetime
    df['coin_id'] = df['coin_id'].astype('category')
    df['date'] = pd.to_datetime(df['date'])

    return df

@pytest.fixture
def sample_prices_df():
    """
    Create a sample prices DataFrame for testing.
    """
    data = {
        'date': [
            '2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01',
            '2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01',
            '2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01'
        ],
        'coin_id': ['BTC', 'BTC', 'BTC', 'BTC', 'ETH', 'ETH', 'ETH', 'ETH', 'MYRO', 'MYRO', 'MYRO', 'MYRO'],
        'price': [20000.0, 21000, 22000, 23000, 1500, 1600, 1700, 1800, 10, 15, 12, 8]
    }
    df = pd.DataFrame(data)

    # Convert coin_id to categorical and date to datetime
    df['coin_id'] = df['coin_id'].astype('category')
    df['date'] = pd.to_datetime(df['date'])

    return df


@pytest.mark.unit
def test_calculate_wallet_profitability_profitability(sample_transfers_df, sample_prices_df):
    """
    Test profitability calculations for multiple wallets and coins.
    """
    profits_df = td.prepare_profits_data(sample_transfers_df, sample_prices_df)
    result = td.calculate_wallet_profitability(profits_df)

    # Check profitability for wallet1, BTC
    wallet1_btc = result[(result['wallet_address'] == 'wallet1') & (result['coin_id'] == 'BTC')]
    assert wallet1_btc.loc[wallet1_btc['date'] == '2023-02-01', 'profits_change'].values[0] == pytest.approx(10000)  # (21000 - 20000) * 10
    assert wallet1_btc.loc[wallet1_btc['date'] == '2023-02-01', 'profits_cumulative'].values[0] == pytest.approx(10000)
    assert wallet1_btc.loc[wallet1_btc['date'] == '2023-04-01', 'profits_change'].values[0] == pytest.approx(30000)  # (23000 - 21000) * 15
    assert wallet1_btc.loc[wallet1_btc['date'] == '2023-04-01', 'profits_cumulative'].values[0] == pytest.approx(40000)  # 10000 + 15000 + 15000

    # Check profitability for wallet2, ETH
    wallet2_eth = result[(result['wallet_address'] == 'wallet2') & (result['coin_id'] == 'ETH')]
    assert wallet2_eth.loc[wallet2_eth['date'] == '2023-02-01', 'profits_change'].values[0] == pytest.approx(5000)  # (1600 - 1500) * 50
    assert wallet2_eth.loc[wallet2_eth['date'] == '2023-02-01', 'profits_cumulative'].values[0] == pytest.approx(5000)
    assert wallet2_eth.loc[wallet2_eth['date'] == '2023-04-01', 'profits_change'].values[0] == pytest.approx(15000)  # (1800 - 1600) * 75
    assert wallet2_eth.loc[wallet2_eth['date'] == '2023-04-01', 'profits_cumulative'].values[0] == pytest.approx(20000)  # 5000 + 15000

    # Check profitability for wallet3, MYRO
    wallet3_myro = result[(result['wallet_address'] == 'wallet3') & (result['coin_id'] == 'MYRO')]
    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-02-01', 'profits_change'].values[0] == pytest.approx(5000)  # (15 - 10) * 1000
    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-03-01', 'profits_change'].values[0] == pytest.approx(-4500)  # (12 - 15) * 1500
    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-03-01', 'profits_cumulative'].values[0] == pytest.approx(500)
    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-04-01', 'profits_change'].values[0] == pytest.approx(-3000)  # (8 - 12) * 750
    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-04-01', 'profits_cumulative'].values[0] == pytest.approx(-2500)  # 500 - 3000

# pylint: disable=R0914 # too many local variables
@pytest.mark.unit
def test_calculate_wallet_profitability_usd_calculations(sample_transfers_df, sample_prices_df):
    """
    Test USD-related calculations (inflows, balances, total return).
    """
    profits_df = td.prepare_profits_data(sample_transfers_df, sample_prices_df)
    result = td.calculate_wallet_profitability(profits_df)

    # Check USD calculations for wallet1, BTC
    wallet1_btc = result[(result['wallet_address'] == 'wallet1') & (result['coin_id'] == 'BTC')]
    initial_investment_wallet1 = 10 * 20000  # 10 BTC * $20,000
    second_investment_wallet1 = 5 * 21000    # 5 BTC * $21,000
    total_investment_wallet1 = initial_investment_wallet1 + second_investment_wallet1
    expected_balance_wallet1 = 15 * 23000    # 15 BTC * $23,000
    expected_total_return_wallet1 = 40000 / total_investment_wallet1

    assert wallet1_btc.loc[wallet1_btc['date'] == '2023-01-01', 'usd_inflows'].values[0] == pytest.approx(initial_investment_wallet1)
    assert wallet1_btc.loc[wallet1_btc['date'] == '2023-02-01', 'usd_inflows'].values[0] == pytest.approx(second_investment_wallet1)
    assert wallet1_btc.loc[wallet1_btc['date'] == '2023-02-01', 'usd_inflows_cumulative'].values[0] == pytest.approx(total_investment_wallet1)
    assert wallet1_btc.loc[wallet1_btc['date'] == '2023-04-01', 'usd_balance'].values[0] == pytest.approx(expected_balance_wallet1)
    assert wallet1_btc.loc[wallet1_btc['date'] == '2023-04-01', 'total_return'].values[0] == pytest.approx(expected_total_return_wallet1)

    # Check USD calculations for wallet2, ETH
    wallet2_eth = result[(result['wallet_address'] == 'wallet2') & (result['coin_id'] == 'ETH')]
    initial_investment_wallet2 = 50 * 1500  # 50 ETH * $1,500
    second_investment_wallet2 = 25 * 1600  # 25 ETH * $1,600
    total_investment_wallet2 = initial_investment_wallet2 + second_investment_wallet2
    expected_balance_wallet2 = 65 * 1800  # 65 ETH * $1,800
    expected_total_return_wallet2 = 20000 / total_investment_wallet2

    assert wallet2_eth.loc[wallet2_eth['date'] == '2023-01-01', 'usd_inflows'].values[0] == pytest.approx(initial_investment_wallet2)
    assert wallet2_eth.loc[wallet2_eth['date'] == '2023-02-01', 'usd_inflows'].values[0] == pytest.approx(second_investment_wallet2)
    assert wallet2_eth.loc[wallet2_eth['date'] == '2023-04-01', 'usd_inflows_cumulative'].values[0] == pytest.approx(total_investment_wallet2)
    assert wallet2_eth.loc[wallet2_eth['date'] == '2023-04-01', 'usd_balance'].values[0] == pytest.approx(expected_balance_wallet2)
    assert wallet2_eth.loc[wallet2_eth['date'] == '2023-04-01', 'total_return'].values[0] == pytest.approx(expected_total_return_wallet2)

    # Check USD calculations for wallet3, MYRO
    wallet3_myro = result[(result['wallet_address'] == 'wallet3') & (result['coin_id'] == 'MYRO')]
    initial_investment_wallet3 = 1000 * 10  # 1000 MYRO * $10
    second_investment_wallet3 = 500 * 15   # 500 MYRO * $15
    total_investment_wallet3 = initial_investment_wallet3 + second_investment_wallet3
    expected_balance_wallet3 = 750 * 8  # 750 MYRO * $8
    current_value_wallet3 = 750 * 8  # Current holdings: 750 MYRO * $8
    sold_value_wallet3 = 750 * 12     # Sold tokens: 750 MYRO * $12
    profit_wallet3 = current_value_wallet3 + sold_value_wallet3 - total_investment_wallet3
    expected_total_return_wallet3 = profit_wallet3 / total_investment_wallet3

    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-01-01', 'usd_inflows'].values[0] == pytest.approx(initial_investment_wallet3)
    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-02-01', 'usd_inflows'].values[0] == pytest.approx(second_investment_wallet3)
    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-03-01', 'usd_inflows_cumulative'].values[0] == pytest.approx(total_investment_wallet3)
    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-04-01', 'usd_balance'].values[0] == pytest.approx(expected_balance_wallet3)
    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-04-01', 'total_return'].values[0] == pytest.approx(expected_total_return_wallet3)


@pytest.fixture
def price_data_transfers_df():
    """
    Create a sample transfers DataFrame for testing interactions with price data.
    """
    data = {
        'coin_id': ['BTC', 'BTC', 'MYRO', 'MYRO'],
        'wallet_address': ['wallet1', 'wallet1', 'wallet2', 'wallet2'],
        'date': [
            '2023-03-01', '2023-04-01',  # BTC wallet1 buys during training period
            '2023-02-20', '2023-03-10'   # MYRO wallet2 buys during training period (before price data)
        ],
        'net_transfers': [10.0, -10.0, 1000.0, -1000.0],  # Buys and sells
        'balance': [10.0, 0.0, 1000.0, 0.0]  # Balance adjustments after buy and sell
    }
    df = pd.DataFrame(data)
    df['coin_id'] = df['coin_id'].astype('category')
    df['date'] = pd.to_datetime(df['date'])
    return df

@pytest.fixture
def price_data_prices_df():
    """
    Create a sample prices DataFrame for testing interactions with price data.
    """
    data = {
        'date': ['2023-03-15', '2023-04-01', '2023-03-15', '2023-04-01'],
        'coin_id': ['BTC', 'BTC', 'MYRO', 'MYRO'],
        'price': [22000.0, 23000, 12, 15]  # Price data available starting from 2023-03-15
    }
    df = pd.DataFrame(data)
    df['coin_id'] = df['coin_id'].astype('category')
    df['date'] = pd.to_datetime(df['date'])
    return df

@pytest.mark.unit
def test_price_data_interactions(price_data_transfers_df, price_data_prices_df):
    """
    Test interactions between wallet transfers and available price data.
    """
    profits_df = td.prepare_profits_data(price_data_transfers_df, price_data_prices_df)
    result = td.calculate_wallet_profitability(profits_df)

    # Test scenario: Buy during training period before price data, sell after price data
    wallet1_btc = result[(result['wallet_address'] == 'wallet1') & (result['coin_id'] == 'BTC')]
    wallet1_btc_profits = (23000-22000) * 10
    assert wallet1_btc.iloc[0]['date'] == pd.Timestamp('2023-03-15')  # First row should reflect earliest price data
    assert wallet1_btc.iloc[0]['profits_cumulative'] == 0  # No profit on initial transfer in
    assert wallet1_btc.iloc[1]['profits_cumulative'] == wallet1_btc_profits  # Profitability calculation should be valid

    # Test scenario: Buy and sell before price data is available
    wallet2_myro = result[(result['wallet_address'] == 'wallet2') & (result['coin_id'] == 'MYRO')]
    assert wallet2_myro.empty  # No rows should exist, as no price data was available for the transaction



# ---------------------------------------- #
# clean_profits_df() unit tests
# ---------------------------------------- #

@pytest.fixture
def sample_clean_profits_profits_df():
    """
    Fixture to create a sample DataFrame for testing td.clean_profits_df function.
    """
    data = {
        'coin_id': ['coin_1', 'coin_2', 'coin_3', 'coin_1', 'coin_2', 'coin_3', 'coin_4', 'coin_1', 'coin_2', 'coin_3', 'coin_1', 'coin_2','coin_1'],
        'wallet_address': ['wallet_1', 'wallet_1', 'wallet_2', 'wallet_2', 'wallet_3', 'wallet_3', 'wallet_4', 'wallet_4', 'wallet_5', 'wallet_5', 'wallet_6', 'wallet_6','wallet_7'],
        'profits_change': [2000, 500, 18000, -7000, 5000, 12000, 16000, 1000, 18000, -7000, -10000, -8000, 15000],  # Updated profits for Wallet 4, 5, and added Wallet 6
        'usd_inflows': [5000, 4000, 7000, 6000, 8000, 9000, 4000, 3000, 5000, 3000, 6000, 3000, 10000]  # Updated inflows and Wallet 6 with 10,000 inflows
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_clean_profits_data_cleaning_config():
    """
    Fixture for the data cleaning configuration.
    """
    return {
        'profitability_filter': 15000,  # Updated profitability filter
        'inflows_filter': 10000  # Updated inflows filter
    }

# Test Case 1: Basic functionality test
@pytest.mark.unit
def test_clean_profits_basic_functionality(sample_clean_profits_profits_df, sample_clean_profits_data_cleaning_config):
    """
    Test basic functionality where some wallets exceed the thresholds and others do not.
    """
    cleaned_df, exclusions_df = td.clean_profits_df(sample_clean_profits_profits_df, sample_clean_profits_data_cleaning_config)
    assert cleaned_df['wallet_address'].nunique() == 2  # 2 wallets should be excluded: one for profits, one for inflows
    assert 'wallet_1' in cleaned_df['wallet_address'].values  # wallet_1 stays within limits
    assert 'wallet_3' not in cleaned_df['wallet_address'].values  # wallet_3 exceeds both thresholds

# Test Case 2: Wallet with exactly the threshold profitability and inflows
@pytest.mark.unit
def test_clean_profits_exact_threshold(sample_clean_profits_profits_df, sample_clean_profits_data_cleaning_config):
    """
    Test a wallet with exactly the threshold values for profitability and inflows.
    """
    cleaned_df, exclusions_df = td.clean_profits_df(sample_clean_profits_profits_df, sample_clean_profits_data_cleaning_config)
    assert not 'wallet_7' in cleaned_df['wallet_address'].values  # wallet_5 should not be excluded as it's exactly at the threshold


# Test Case 3: Wallet with negative profitability exceeding the threshold
@pytest.mark.unit
def test_clean_profits_negative_profitability(sample_clean_profits_profits_df, sample_clean_profits_data_cleaning_config):
    """
    Test exclusion of wallets with negative profitability exceeding the threshold.
    """
    cleaned_df, exclusions_df = td.clean_profits_df(sample_clean_profits_profits_df, sample_clean_profits_data_cleaning_config)
    assert 'wallet_6' not in cleaned_df['wallet_address'].values  # wallet_6 exceeds the negative profitability threshold

# Test Case 4: Multiple wallets with profits and inflows exceeding thresholds
@pytest.mark.unit
def test_clean_profits_multiple_exceeding_thresholds(sample_clean_profits_profits_df, sample_clean_profits_data_cleaning_config):
    """
    Test exclusion of wallets where either profits or inflows exceed thresholds.
    """
    cleaned_df, exclusions_df = td.clean_profits_df(sample_clean_profits_profits_df, sample_clean_profits_data_cleaning_config)
    assert len(exclusions_df) == 5  # 5 wallets should be excluded based on either profits or inflows

# Test Case 5: Wallet with multiple transactions but total profits below thresholds
@pytest.mark.unit
def test_clean_profits_multiple_transactions_below_threshold(sample_clean_profits_profits_df, sample_clean_profits_data_cleaning_config):
    """
    Test wallet with multiple transactions where total profits remain below thresholds.
    """
    cleaned_df, exclusions_df = td.clean_profits_df(sample_clean_profits_profits_df, sample_clean_profits_data_cleaning_config)
    assert 'wallet_1' in cleaned_df['wallet_address'].values  # wallet_1 remains below threshold and should not be excluded

# Test Case 8: Wallet with extreme profits but inflows below the threshold
@pytest.mark.unit
def test_clean_profits_extreme_profits_but_low_inflows(sample_clean_profits_profits_df, sample_clean_profits_data_cleaning_config):
    """
    Test exclusion of wallet with extreme profits but inflows below the threshold.
    """
    sample_clean_profits_profits_df.loc[sample_clean_profits_profits_df['wallet_address'] == 'wallet_4', 'profits_change'] = 16000
    cleaned_df, exclusions_df = td.clean_profits_df(sample_clean_profits_profits_df, sample_clean_profits_data_cleaning_config)
    assert 'wallet_4' not in cleaned_df['wallet_address'].values  # wallet_4 should be excluded based on profits

# Test Case 9: Wallet with extreme inflows but zero profits
@pytest.mark.unit
def test_clean_profits_extreme_inflows_but_zero_profits(sample_clean_profits_profits_df, sample_clean_profits_data_cleaning_config):
    """
    Test exclusion of wallet with extreme inflows but no significant profits.
    """
    sample_clean_profits_profits_df.loc[sample_clean_profits_profits_df['wallet_address'] == 'wallet_3', 'profits_change'] = 0
    sample_clean_profits_profits_df.loc[sample_clean_profits_profits_df['wallet_address'] == 'wallet_3', 'usd_inflows_cumulative'] = 17000
    cleaned_df, exclusions_df = td.clean_profits_df(sample_clean_profits_profits_df, sample_clean_profits_data_cleaning_config)
    assert 'wallet_3' not in cleaned_df['wallet_address'].values  # wallet_3 should be excluded based on inflows

# Test Case 11: Wallet with inflows/profits across multiple coins but aggregate exceeds threshold
@pytest.mark.unit
def test_clean_profits_aggregate_inflows_profits_across_coins(sample_clean_profits_profits_df, sample_clean_profits_data_cleaning_config):
    """
    Test exclusion of wallet where aggregated inflows/profits across multiple coins exceed thresholds.
    """
    sample_clean_profits_profits_df.loc[sample_clean_profits_profits_df['wallet_address'] == 'wallet_5', 'profits_change'] = 18000
    sample_clean_profits_profits_df.loc[sample_clean_profits_profits_df['wallet_address'] == 'wallet_5', 'usd_inflows_cumulative'] = 8000
    cleaned_df, exclusions_df = td.clean_profits_df(sample_clean_profits_profits_df, sample_clean_profits_data_cleaning_config)
    assert 'wallet_5' not in cleaned_df['wallet_address'].values  # wallet_5 should be excluded based on aggregated profits



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
    cohort_wallets_df = td.classify_wallet_cohort(sample_profits_df, sample_config)

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






# ======================================================== #
#                                                          #
#            I N T E G R A T I O N   T E S T S             #
#                                                          #
# ======================================================== #


# ---------------------------------- #
# set up config and module-level variables
# ---------------------------------- #

config = load_config('tests/test_config/test_config.yaml')

# Module-level variables
TRAINING_PERIOD_START = config['training_data']['training_period_start']
TRAINING_PERIOD_END = config['training_data']['training_period_end']
MODELING_PERIOD_START = config['training_data']['modeling_period_start']
MODELING_PERIOD_END = config['training_data']['modeling_period_end']


# ---------------------------------------- #
# retrieve_transfers_data() integration tests
# ---------------------------------------- #

@pytest.fixture(scope='session')
def transfers_df():
    """
    retrieves transfers_df for data quality checks
    """
    logger.info("Beginning integration testing...")
    logger.info("Generating transfers_df fixture from production data...")
    return td.retrieve_transfers_data(TRAINING_PERIOD_START, MODELING_PERIOD_START, MODELING_PERIOD_END)

@pytest.mark.integration
def test_transfers_data_quality(transfers_df):
    """
    Retrieves transfers_df and performs comprehensive data quality checks.
    """
    logger.info("Testing transfers_df from retrieve_transfers_data()...")
    transfers_df = transfers_df.copy(deep=False)  # Create a copy to avoid affecting subsequent tests

    # Test 1: No duplicate records
    # ----------------------------
    deduped_df = transfers_df[['coin_id', 'wallet_address', 'date']].drop_duplicates()
    logger.info(f"Original transfers_df length: {len(transfers_df)}, Deduplicated: {len(deduped_df)}")
    assert len(transfers_df) == len(deduped_df), "There are duplicate rows based on coin_id, wallet_address, and date"

    # Test 2: All coin-wallet pairs have a record at the end of the training period
    # ----------------------------------------------------------------------------
    transfers_df_filtered = transfers_df[transfers_df['date'] < MODELING_PERIOD_START]
    pairs_in_training_period = transfers_df_filtered[['coin_id', 'wallet_address']].drop_duplicates()
    period_end_df = transfers_df[transfers_df['date'] == TRAINING_PERIOD_END]

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
    expected_max_date = pd.to_datetime(MODELING_PERIOD_END)
    logger.info(f"transfers_df date range: {min_date} to {max_date}")
    assert max_date == expected_max_date, f"The last date in the dataset should be {expected_max_date}"

    # Test 5: No missing values
    # -------------------------
    missing_values = transfers_df.isna().sum()
    assert missing_values.sum() == 0, "There are missing values in the dataset"

    # Test 6: Balance consistency
    # ---------------------------
    transfers_df['balance_change'] = transfers_df.groupby(['coin_id', 'wallet_address'],observed=True)['balance'].diff()
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

    # Test 8: Ensure all applicable wallets have records as of the training_period_end
    # ------------------------------------------------------------------------------------------
    # get a list of all coin-wallet pairs as of the training_period_end
    training_transfers_df = transfers_df[transfers_df['date'] <= TRAINING_PERIOD_END]
    training_wallets_df = training_transfers_df[['coin_id', 'wallet_address']].drop_duplicates()

    # get a list of all coin-wallet pairs on the training_transfers_end date
    training_end_df = transfers_df[transfers_df['date'] == TRAINING_PERIOD_END]
    training_end_df = training_end_df[['coin_id', 'wallet_address']].drop_duplicates()

    # confirm that they are the same length
    assert len(training_wallets_df) == len(training_end_df), "Some wallets are missing a record as of the training_period_end"

    # Test 9: Ensure all wallets have records as of the modeling_period_start
    # ------------------------------------------------------------------------------------------
    # get a list of all coin-wallet pairs
    modeling_transfers_df = transfers_df[transfers_df['date'] <= MODELING_PERIOD_START]
    modeling_wallets_df = modeling_transfers_df[['coin_id', 'wallet_address']].drop_duplicates()

    # get a list of all coin-wallet pairs on the modeling_period_start
    modeling_start_df = transfers_df[transfers_df['date'] == MODELING_PERIOD_START]
    modeling_start_df = modeling_start_df[['coin_id', 'wallet_address']].drop_duplicates()

    # confirm that they are the same length
    assert len(modeling_wallets_df) == len(modeling_start_df), "Some wallets are missing a record as of the modeling_period_start"

    # Test 9: Ensure all wallets have records as of the modeling_period_end
    # ------------------------------------------------------------------------------------------
    # get a list of all coin-wallet pairs
    modeling_transfers_df = transfers_df[transfers_df['date'] <= MODELING_PERIOD_END]
    modeling_wallets_df = modeling_transfers_df[['coin_id', 'wallet_address']].drop_duplicates()

    # get a list of all coin-wallet pairs on the modeling_period_end
    modeling_end_df = transfers_df[transfers_df['date'] == MODELING_PERIOD_END]
    modeling_end_df = modeling_end_df[['coin_id', 'wallet_address']].drop_duplicates()

    # confirm that they are the same length
    assert len(modeling_wallets_df) == len(modeling_end_df), "Some wallets are missing a record as of the modeling_period_end"

    # Test 10: Confirm no records exist prior to the training period start
    # ------------------------------------------------------------------------------------------
    assert len(transfers_df[transfers_df['date']<TRAINING_PERIOD_START]) == 0, "Records prior to training_period_start exist"

    logger.info("All transfers_df data quality checks passed successfully.")


# ---------------------------------------- #
# retrieve_prices_data() integration tests
# ---------------------------------------- #

@pytest.fixture(scope='session')
def prices_df():
    """
    Retrieve and preprocess the prices_df, filling gaps as needed.
    """
    logger.info("Generating prices_df from production data...")
    prices_df = td.retrieve_prices_data()
    prices_df, _ = td.fill_prices_gaps(prices_df, max_gap_days=2)
    return prices_df

# Save prices_df.csv in fixtures/
# ----------------------------------------
def test_save_prices_df(prices_df):
    """
    This is not a test! This function saves a prices_df.csv in the fixtures folder so it can be
    used for integration tests in other modules.
    """
    # Save the prices DataFrame to the fixtures folder
    prices_df.to_csv('tests/fixtures/prices_df.csv', index=False)

    # Add some basic assertions to ensure the data was saved correctly
    assert prices_df is not None
    assert len(prices_df) > 0


# ---------------------------------------- #
# calculate_wallet_profitability() integration tests
# ---------------------------------------- #
# tests the data quality of the production data as calculated from the transfers_df() fixture

@pytest.fixture(scope='session')
def profits_df(transfers_df, prices_df):
    """
    Builds profits_df from production data for data quality checks.
    """
    logger.info("Generating profits_df from production data...")
    profits_df = td.prepare_profits_data(transfers_df, prices_df)
    profits_df = td.calculate_wallet_profitability(profits_df)
    return profits_df


@pytest.mark.integration
def test_profits_df_completeness(profits_df):
    """
    Checks if there are any NaN values in profits_df.
    """
    missing_values = profits_df.isna().sum()
    assert missing_values.sum() == 0, "There are missing values in the dataset"


@pytest.mark.integration
def test_modeling_period_end_wallet_completeness(profits_df):
    """
    Checks if all of the coin-wallet pairs at the end of the training period
    have data at the end of the modeling period to esnure they can be analyzed
    for profitability.
    """
    training_period_end = pd.to_datetime(MODELING_PERIOD_START) - pd.Timedelta(1, 'day')
    modeling_period_end = MODELING_PERIOD_END

    # Get all coin-wallet pairs at the end of the training period
    training_profits_df = profits_df[profits_df['date'] == training_period_end]
    training_profits_df = training_profits_df[['coin_id', 'wallet_address']].drop_duplicates()

    # Get all coin-wallet pairs at the end of the modeling period
    modeling_profits_df = profits_df[profits_df['date']==modeling_period_end]
    modeling_profits_df = modeling_profits_df[['coin_id', 'wallet_address']].drop_duplicates()

    # Check if there are any pairs at the end of the training period without records at the end of the modeling period
    missing_pairs = training_profits_df.merge(modeling_profits_df, on=['coin_id', 'wallet_address'], how='left', indicator=True)
    missing_pairs = missing_pairs[missing_pairs['_merge'] == 'left_only']

    # Assert that no pairs are missing
    assert missing_pairs.empty, "Some coin-wallet pairs in training_profits_df are missing from modeling_profits_df"



# ---------------------------------------- #
# clean_profits_df() integration tests
# ---------------------------------------- #

@pytest.fixture(scope='session')
def cleaned_profits_df(profits_df):
    """
    Fixture to run clean_profits_df() and return both the cleaned DataFrame and exclusions DataFrame.
    Uses thresholds from the config file.
    """
    logger.info("Generating cleaned_profits_df from clean_profits_df()...")
    cleaned_df, exclusions_df = td.clean_profits_df(profits_df, config['data_cleaning'])
    return cleaned_df, exclusions_df

# Save cleaned_profits_df.csv in fixtures/
# ----------------------------------------
def test_save_cleaned_profits_df(cleaned_profits_df):
    """
    This is not a test! This function saves a cleaned_profits_df.csv in the fixtures folder so it can be \
    used for integration tests in other modules.
    """
    cleaned_df,_ = cleaned_profits_df

    # Save the cleaned DataFrame to the fixtures folder
    cleaned_df.to_csv('tests/fixtures/cleaned_profits_df.csv', index=False)

    # Add some basic assertions to ensure the data was saved correctly
    assert cleaned_df is not None
    assert len(cleaned_df) > 0

@pytest.mark.integration
def test_clean_profits_exclusions(cleaned_profits_df, profits_df):
    """
    Test that all excluded wallets breach either the profitability or inflows threshold.
    Uses thresholds from the config file.
    """
    cleaned_df, exclusions_df = cleaned_profits_df

    # Check that every excluded wallet breached at least one threshold
    exclusions_with_breaches = exclusions_df.merge(profits_df, on='wallet_address', how='inner')

    # Calculate the total profits and inflows per wallet
    wallet_agg_df = exclusions_with_breaches.groupby('wallet_address').agg({
        'profits_change': 'sum',
        'usd_inflows': 'sum'
    }).reset_index()

    # Apply threshold check from the config
    profitability_filter = config['data_cleaning']['profitability_filter']
    inflows_filter = config['data_cleaning']['inflows_filter']

    breaches_df = wallet_agg_df[
        (wallet_agg_df['profits_change'] >= profitability_filter) |
        (wallet_agg_df['profits_change'] <= -profitability_filter) |
        (wallet_agg_df['usd_inflows'] >= inflows_filter)
    ]

    # Assert that all excluded wallets breached a threshold
    assert len(exclusions_df) == len(breaches_df), "Some excluded wallets do not breach a threshold."

@pytest.mark.integration
def test_clean_profits_remaining_count(cleaned_profits_df, profits_df):
    """
    Test that the count of remaining records in the cleaned DataFrame matches the expected count.
    """
    cleaned_df, exclusions_df = cleaned_profits_df

    # Get the total number of unique wallets before and after cleaning
    input_wallet_count = profits_df['wallet_address'].nunique()
    cleaned_wallet_count = cleaned_df['wallet_address'].nunique()
    excluded_wallet_count = exclusions_df['wallet_address'].nunique()

    # Assert that the remaining records equal the difference between the input and excluded records
    assert input_wallet_count == cleaned_wallet_count + excluded_wallet_count, \
        "The count of remaining wallets does not match the input minus excluded records."

@pytest.mark.integration
def test_clean_profits_aggregate_sums(cleaned_profits_df):
    """
    Test that the aggregation of profits and inflows for the remaining wallets stays within the configured thresholds.
    Uses thresholds from the config file.
    """
    cleaned_df, _ = cleaned_profits_df

    # Aggregate the profits and inflows for the remaining wallets
    remaining_wallets_agg_df = cleaned_df.groupby('wallet_address').agg({
        'profits_change': 'sum',
        'usd_inflows': 'sum'
    }).reset_index()

    # Apply the thresholds from the config
    profitability_filter = config['data_cleaning']['profitability_filter']
    inflows_filter = config['data_cleaning']['inflows_filter']

    # Ensure no remaining wallets exceed the thresholds
    over_threshold_wallets = remaining_wallets_agg_df[
        (remaining_wallets_agg_df['profits_change'] >= profitability_filter) |
        (remaining_wallets_agg_df['profits_change'] <= -profitability_filter) |
        (remaining_wallets_agg_df['usd_inflows'] >= inflows_filter)
    ]

    # Assert that no wallets in the cleaned DataFrame breach the thresholds
    assert over_threshold_wallets.empty, "Some remaining wallets exceed the thresholds."


# ---------------------------------------- #
# classify_wallet_cohort() tests
# ---------------------------------------- #

@pytest.fixture(scope='session')
def wallet_cohort_df(cleaned_profits_df):
    """
    Builds shark_wallets_df from shark_coins_df for data quality checks.
    """
    profits_df, _ = cleaned_profits_df  # Use the cleaned profits DataFrame
    wallet_cohort_df = td.classify_wallet_cohort(profits_df, config['wallet_cohorts']['sharks'])
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

    # Assert that there are no duplicates in sharks_df
    assert not duplicates.any(), "Duplicate wallet addresses found in sharks_df"
