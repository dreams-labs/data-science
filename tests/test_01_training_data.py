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
import contextlib
import threading
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pytest
from dreams_core import core as dc

# pyright: reportMissingImports=false
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import training_data.data_retrieval as dr
import training_data.profits_row_imputation as pri
import coin_wallet_metrics.coin_wallet_metrics as cwm
from utils import load_config

load_dotenv()
logger = dc.setup_logger()


# ===================================================== #
#                                                       #
#                 U N I T   T E S T S                   #
#                                                       #
# ===================================================== #

# ---------------------------------------- #
# clean_profits_df() unit tests
# ---------------------------------------- #
@pytest.fixture
def sample_profits_df_for_cleaning():
    """
    Fixture to create a sample profits DataFrame with multiple coins per wallet.
    """
    return pd.DataFrame({
        'coin_id': ['BTC', 'ETH', 'BTC', 'ETH', 'LTC', 'BTC', 'ETH'],
        'wallet_address': ['wallet1', 'wallet1', 'wallet2', 'wallet2','wallet2',
                           'wallet3', 'wallet3'],
        'date': pd.date_range(start='2023-01-01', periods=7),
        'profits_cumulative': [5000, 3000, 1000, 500, 500, 100, 50],
        'usd_inflows_cumulative': [10000, 8000, 2000, 1500, 1500, 500, 250]
    })

@pytest.fixture
def sample_data_cleaning_config():
    """
    Fixture to create a sample data cleaning configuration.
    """
    return {
        'profitability_filter': 7500,
        'inflows_filter': 15000
    }

@pytest.mark.unit
def test_multiple_coins_per_wallet(sample_profits_df_for_cleaning, sample_data_cleaning_config):
    """
    Test scenario where wallets own multiple coins, some exceeding thresholds when aggregated.
    """
    # Call the function
    cleaned_df, exclusions_logs_df = dr.clean_profits_df(sample_profits_df_for_cleaning,
                                                         sample_data_cleaning_config)

    # Expected results
    expected_cleaned_df = sample_profits_df_for_cleaning[
        sample_profits_df_for_cleaning['wallet_address'].isin(['wallet2', 'wallet3'])
    ].reset_index(drop=True)
    expected_exclusions = pd.DataFrame({
        'wallet_address': ['wallet1'],
        'profits_exclusion': [True],
        'inflows_exclusion': [True]
    })

    # Assertions
    assert len(cleaned_df) == 5  # wallet2 (3 records) and wallet3 (2 records) should remain
    assert np.array_equal(cleaned_df.values, expected_cleaned_df.values)
    assert np.array_equal(exclusions_logs_df.values, expected_exclusions.values)

    # Check if profits and inflows are approximately correct for the remaining wallets
    # 1000 + 500 + 500 + 100 + 50
    assert pytest.approx(cleaned_df['profits_cumulative'].sum(), abs=1e-4) == 2150
    # 2000 + 1500 + 1500 + 500 + 250
    assert pytest.approx(cleaned_df['usd_inflows_cumulative'].sum(), abs=1e-4) == 5750

@pytest.fixture
def profits_at_threshold_df():
    """
    Fixture to create a sample profits DataFrame with profits at, above, and below the threshold.
    """
    return pd.DataFrame({
        'coin_id': ['BTC', 'ETH', 'LTC', 'XRP', 'DOGE'],
        'wallet_address': ['wallet1', 'wallet2', 'wallet3', 'wallet4', 'wallet5'],
        'date': pd.date_range(start='2023-01-01', periods=5),
        'profits_cumulative': [5000, 7500, 7501, 7499, 3000],
        'usd_inflows_cumulative': [10000, 12000, 13000, 11000, 8000]
    })

@pytest.fixture
def profits_at_threshold_config():
    """
    Fixture to create a data cleaning configuration with a specific profitability threshold.
    """
    return {
        'profitability_filter': 7500,
        'inflows_filter': 15000
    }

@pytest.mark.unit
def test_profits_exactly_at_threshold(profits_at_threshold_df, profits_at_threshold_config):
    """
    Test scenario where some wallets have profits exactly at the threshold value.
    """
    # Call the function
    cleaned_df, exclusions_logs_df = dr.clean_profits_df(profits_at_threshold_df,
                                                         profits_at_threshold_config)

    # Expected results
    expected_cleaned_df = profits_at_threshold_df[
        profits_at_threshold_df['wallet_address'].isin(['wallet1', 'wallet4', 'wallet5'])
    ].reset_index(drop=True)

    expected_exclusions = pd.DataFrame({
        'wallet_address': ['wallet2', 'wallet3'],
        'profits_exclusion': [True, True],
        'inflows_exclusion': [False, False]
    })

    # Assertions
    assert len(cleaned_df) == 3  # wallet1, wallet4, and wallet5 should remain
    assert np.array_equal(cleaned_df.values, expected_cleaned_df.values)
    assert np.array_equal(exclusions_logs_df.values, expected_exclusions.values)

    # Check if the correct wallets are present in the cleaned DataFrame
    assert set(cleaned_df['wallet_address']) == {'wallet1', 'wallet4', 'wallet5'}

    # Verify that wallets at or above the threshold are excluded
    assert 'wallet2' not in cleaned_df['wallet_address'].values
    assert 'wallet3' not in cleaned_df['wallet_address'].values

    # Check if profits and inflows are approximately correct for the remaining wallets
    # 5000 + 7499 + 3000
    assert pytest.approx(cleaned_df['profits_cumulative'].sum(), abs=1e-4) == 15499
     # 10000 + 11000 + 8000
    assert pytest.approx(cleaned_df['usd_inflows_cumulative'].sum(), abs=1e-4) == 29000

@pytest.fixture
def negative_profits_df():
    """
    Fixture to create a sample profits DataFrame with various levels of negative profits (losses).
    """
    return pd.DataFrame({
        'coin_id': ['BTC', 'ETH', 'LTC', 'XRP', 'DOGE', 'ADA'],
        'wallet_address': ['wallet1', 'wallet2', 'wallet3', 'wallet4', 'wallet5', 'wallet6'],
        'date': pd.date_range(start='2023-01-01', periods=6),
        'profits_cumulative': [-5000, -7500, -7501, -7499, 3000, 0],
        'usd_inflows_cumulative': [10000, 12000, 13000, 11000, 8000, 5000]
    })

@pytest.fixture
def negative_profits_config():
    """
    Fixture to create a data cleaning configuration with a specific profitability threshold.
    """
    return {
        'profitability_filter': 7500,
        'inflows_filter': 15000
    }

@pytest.mark.unit
def test_negative_profits_losses(negative_profits_df, negative_profits_config):
    """
    Test scenario where some wallets have significant negative profits (losses).
    """
    # Call the function
    cleaned_df, exclusions_logs_df = dr.clean_profits_df(negative_profits_df,
                                                         negative_profits_config)

    # Expected results
    expected_cleaned_df = negative_profits_df[
        negative_profits_df['wallet_address'].isin(['wallet1', 'wallet4', 'wallet5', 'wallet6'])
    ].reset_index(drop=True)

    expected_exclusions = pd.DataFrame({
        'wallet_address': ['wallet2', 'wallet3'],
        'profits_exclusion': [True, True],
        'inflows_exclusion': [False, False]
    })

    # Assertions
    assert len(cleaned_df) == 4  # wallet1, wallet4, wallet5, and wallet6 should remain
    assert np.array_equal(cleaned_df.values, expected_cleaned_df.values)
    assert np.array_equal(exclusions_logs_df.values, expected_exclusions.values)

    # Check if the correct wallets are present in the cleaned DataFrame
    assert set(cleaned_df['wallet_address']) == {'wallet1', 'wallet4', 'wallet5', 'wallet6'}

    # Verify that wallets with losses at or beyond the threshold are excluded
    assert 'wallet2' not in cleaned_df['wallet_address'].values
    assert 'wallet3' not in cleaned_df['wallet_address'].values

    # Check if profits and inflows are approximately correct for the remaining wallets
    # -5000 + -7499 + 3000 + 0
    assert pytest.approx(cleaned_df['profits_cumulative'].sum(), abs=1e-4) == -9499
    # 10000 + 11000 + 8000 + 5000
    assert pytest.approx(cleaned_df['usd_inflows_cumulative'].sum(), abs=1e-4) == 34000

    # Verify that wallets with losses are present in the cleaned DataFrame
    assert (cleaned_df['profits_cumulative'] < 0).any()

    # Verify that the wallet with zero profit is included
    assert (cleaned_df['profits_cumulative'] == 0).any()



# -------------------------------------------- #
# calculate_new_profits_values() unit tests
# -------------------------------------------- #

@pytest.fixture
def sample_profits_df():
    """
    Fixture to create a sample profits DataFrame for testing.

    Returns:
        pd.DataFrame: A df with sample data for testing calculate_new_profits_values function.
    """
    return pd.DataFrame({
        'coin_id': ['btc', 'eth', 'btc', 'eth'],
        'wallet_address': ['addr1', 'addr1', 'addr2', 'addr2'],
        'date': pd.date_range(start='2023-01-01', periods=4),
        'price_previous': [9000, 150, 9100, 155],
        'price': [9500, 160, 9300, 158],
        'usd_balance': [1000, 2000, 1500, 2500],
        'profits_cumulative': [100, 200, 150, 250],
        'usd_inflows_cumulative': [900, 1800, 1350, 2250]
    }).set_index(['coin_id', 'wallet_address', 'date'])

@pytest.fixture
def target_date():
    """
    Fixture to provide a target date for testing.

    Returns:
        datetime: A sample target date.
    """
    return pd.Timestamp('2023-01-05')

@pytest.mark.unit
def test_calculate_new_profits_values_normal_data(sample_profits_df, target_date):
    """
    Test calculate_new_profits_values function with normal data.

    This test checks if the function correctly calculates new financial metrics
    for imputed rows using a sample DataFrame with realistic data.

    Args:
        sample_profits_df (pd.DataFrame): Fixture providing sample input data.
        target_date (datetime): Fixture providing a target date for imputation.
    """
    result = pri.calculate_new_profits_values(sample_profits_df, target_date)
    result = result.set_index(['coin_id', 'wallet_address', 'date'])

    # Check if the result has the correct structure
    assert isinstance(result, pd.DataFrame)
    assert result.index.names == ['coin_id', 'wallet_address', 'date']
    assert set(result.columns) == {
        'profits_cumulative', 'usd_balance',
        'usd_net_transfers', 'usd_inflows', 'usd_inflows_cumulative', 'total_return'
    }

    # Check if calculations are correct for the first row (btc, addr1)
    first_row = result.loc[('btc', 'addr1', target_date)]
    assert first_row['profits_cumulative'] == pytest.approx(155.55555556)  # 100 + 55.55555556
    assert first_row['usd_balance'] == pytest.approx(1055.55555556)  # (9500/9000) * 1000
    assert first_row['usd_net_transfers'] == 0
    assert first_row['usd_inflows'] == 0
    assert first_row['usd_inflows_cumulative'] == 900
    assert first_row['total_return'] == pytest.approx(0.1728, abs=1e-4)  # 155.55555556 / 900

    # Check if all rows have the correct target_date
    assert (result.index.get_level_values('date') == target_date).all()

    # Check if the number of rows matches the input DataFrame
    assert len(result) == len(sample_profits_df)


@pytest.fixture
def zero_price_change_df():
    """
    Fixture to create a DataFrame with zero price change for testing.

    Returns:
        pd.DataFrame: A DataFrame with sample data where price equals price_previous.
    """
    return pd.DataFrame({
        'coin_id': ['btc', 'eth', 'btc', 'eth'],
        'wallet_address': ['addr1', 'addr1', 'addr2', 'addr2'],
        'date': pd.date_range(start='2023-01-01', periods=4),
        'price_previous': [9000, 150, 9000, 150],
        'price': [9000, 150, 9000, 150],
        'usd_balance': [1000, 2000, 1500, 2500],
        'profits_cumulative': [100, 200, 150, 250],
        'usd_inflows_cumulative': [900, 1800, 1350, 2250]
    }).set_index(['coin_id', 'wallet_address', 'date'])

@pytest.mark.unit
def test_calculate_new_profits_values_zero_price_change(zero_price_change_df, target_date):
    """
    Test calculate_new_profits_values function with zero price change.

    This test checks if the function correctly handles cases where there is no change in price,
    ensuring that profits and balances remain unchanged.

    Args:
        zero_price_change_df (pd.DataFrame): Fixture providing sample input data with no price change.
        target_date (datetime): Fixture providing a target date for imputation.
    """
    result = pri.calculate_new_profits_values(zero_price_change_df, target_date)
    result = result.set_index(['coin_id', 'wallet_address', 'date'])

    # Check if the result has the correct structure
    assert isinstance(result, pd.DataFrame)

    # Check if calculations are correct for all rows
    for (coin_id, wallet_address), group in zero_price_change_df.groupby(['coin_id', 'wallet_address']):
        result_row = result.loc[(coin_id, wallet_address, target_date)]
        original_row = group.iloc[0]  # Take the first row of the group

        assert result_row['profits_cumulative'] == pytest.approx(original_row['profits_cumulative'], abs=1e-4)
        assert result_row['usd_balance'] == pytest.approx(original_row['usd_balance'], abs=1e-4)
        assert result_row['usd_net_transfers'] == 0
        assert result_row['usd_inflows'] == 0
        assert result_row['usd_inflows_cumulative'] == original_row['usd_inflows_cumulative']
        assert result_row['total_return'] == pytest.approx(
            original_row['profits_cumulative'] / original_row['usd_inflows_cumulative'], abs=1e-4)

    # Check if all rows have the correct target_date
    assert (result.index.get_level_values('date') == target_date).all()

    # Check if the number of rows matches the input DataFrame
    assert len(result) == len(zero_price_change_df.groupby(['coin_id', 'wallet_address']))


@pytest.fixture
def negative_price_change_df():
    """
    Fixture to create a DataFrame with negative price changes for testing.

    Returns:
        pd.DataFrame: A DataFrame with sample data where price is less than price_previous.
    """
    return pd.DataFrame({
        'coin_id': ['btc', 'eth', 'btc', 'eth'],
        'wallet_address': ['addr1', 'addr1', 'addr2', 'addr2'],
        'date': pd.date_range(start='2023-01-01', periods=4),
        'price_previous': [10000, 200, 10000, 200],
        'price': [9000, 180, 9500, 190],
        'usd_balance': [1000, 2000, 1500, 2500],
        'profits_cumulative': [100, 200, 150, 250],
        'usd_inflows_cumulative': [900, 1800, 1350, 2250]
    }).set_index(['coin_id', 'wallet_address', 'date'])

@pytest.mark.unit
def test_calculate_new_profits_values_negative_price_change(negative_price_change_df, target_date):
    """
    Test calculate_new_profits_values function with negative price changes.

    This test checks if the function correctly handles cases where there is a decrease in price,
    ensuring that profits decrease and balances are adjusted accordingly.

    Args:
        negative_price_change_df (pd.DataFrame): Fixture providing sample input data with
            price decreases.
        target_date (datetime): Fixture providing a target date for imputation.
    """
    result = pri.calculate_new_profits_values(negative_price_change_df, target_date)
    result = result.set_index(['coin_id', 'wallet_address', 'date'])

    # Check if the result has the correct structure
    assert isinstance(result, pd.DataFrame)

    # Check if calculations are correct for all rows
    for (coin_id, wallet_address), group in negative_price_change_df.groupby(['coin_id', 'wallet_address']):
        result_row = result.loc[(coin_id, wallet_address, target_date)]
        original_row = group.iloc[0]  # Take the first row of the group

        price_ratio = original_row['price'] / original_row['price_previous']
        expected_profits_change = (price_ratio - 1) * original_row['usd_balance']
        expected_profits_cumulative = original_row['profits_cumulative'] + expected_profits_change
        expected_usd_balance = price_ratio * original_row['usd_balance']

        assert result_row['profits_cumulative'] == pytest.approx(expected_profits_cumulative, abs=1e-4)
        assert result_row['usd_balance'] == pytest.approx(expected_usd_balance, abs=1e-4)
        assert result_row['usd_net_transfers'] == 0
        assert result_row['usd_inflows'] == 0
        assert result_row['usd_inflows_cumulative'] == original_row['usd_inflows_cumulative']
        assert result_row['total_return'] == pytest.approx(
            expected_profits_cumulative / original_row['usd_inflows_cumulative'], abs=1e-4)

        # Additional checks specific to negative price change
        assert result_row['profits_cumulative'] < original_row['profits_cumulative']
        assert result_row['usd_balance'] < original_row['usd_balance']

    # Check if all rows have the correct target_date
    assert (result.index.get_level_values('date') == target_date).all()

    # Check if the number of rows matches the input DataFrame
    assert len(result) == len(negative_price_change_df.groupby(['coin_id', 'wallet_address']))


@pytest.fixture
def zero_usd_balance_df():
    """
    Fixture to create a DataFrame with zero USD balance for testing.

    Returns:
        pd.DataFrame: A DataFrame with sample data where usd_balance is zero for all rows.
    """
    return pd.DataFrame({
        'coin_id': ['btc', 'eth', 'btc', 'eth'],
        'wallet_address': ['addr1', 'addr1', 'addr2', 'addr2'],
        'date': pd.date_range(start='2023-01-01', periods=4),
        'price_previous': [9000, 150, 9000, 150],
        'price': [9500, 160, 9300, 158],
        'usd_balance': [0, 0, 0, 0],
        'profits_cumulative': [100, 200, 150, 250],
        'usd_inflows_cumulative': [900, 1800, 1350, 2250]
    }).set_index(['coin_id', 'wallet_address', 'date'])

@pytest.mark.unit
def test_calculate_new_profits_values_zero_usd_balance(zero_usd_balance_df, target_date):
    """
    Test calculate_new_profits_values function with zero USD balance.

    This test checks if the function correctly handles cases where the USD balance is zero,
    ensuring that profits and balances remain zero regardless of price changes.

    Args:
        zero_usd_balance_df (pd.DataFrame): Fixture providing sample input data with zero USD balances.
        target_date (datetime): Fixture providing a target date for imputation.
    """
    result = pri.calculate_new_profits_values(zero_usd_balance_df, target_date)
    result = result.set_index(['coin_id', 'wallet_address', 'date'])

    # Check if calculations are correct for all rows
    for (coin_id, wallet_address), group in zero_usd_balance_df.groupby(['coin_id', 'wallet_address']):
        result_row = result.loc[(coin_id, wallet_address, target_date)]
        original_row = group.iloc[0]  # Take the first row of the group

        assert result_row['profits_cumulative'] == original_row['profits_cumulative']
        assert result_row['usd_balance'] == 0
        assert result_row['usd_net_transfers'] == 0
        assert result_row['usd_inflows'] == 0
        assert result_row['usd_inflows_cumulative'] == original_row['usd_inflows_cumulative']
        assert result_row['total_return'] == pytest.approx(
            original_row['profits_cumulative'] / original_row['usd_inflows_cumulative'], abs=1e-4)

    # Check if all rows have the correct target_date
    assert (result.index.get_level_values('date') == target_date).all()

    # Check if the number of rows matches the input DataFrame
    assert len(result) == len(zero_usd_balance_df.groupby(['coin_id', 'wallet_address']))


@pytest.fixture
def multiple_coins_wallets_df():
    """
    Fixture to create a DataFrame with multiple coin_ids and wallet_addresses for testing.

    Returns:
        pd.DataFrame: A DataFrame with sample data for various coins and wallet addresses.
    """
    return pd.DataFrame({
        'coin_id': ['btc', 'eth', 'ltc', 'xrp', 'btc', 'eth', 'ltc', 'xrp'],
        'wallet_address': ['addr1', 'addr1', 'addr2', 'addr2', 'addr3', 'addr3', 'addr4', 'addr4'],
        'date': pd.date_range(start='2023-01-01', periods=8),
        'price_previous': [9000, 150, 50, 0.3, 9100, 155, 52, 0.31],
        'price': [9500, 160, 48, 0.32, 9300, 158, 51, 0.30],
        'usd_balance': [1000, 2000, 1500, 2500, 3000, 4000, 3500, 4500],
        'profits_cumulative': [100, 200, 150, 250, 300, 400, 350, 450],
        'usd_inflows_cumulative': [900, 1800, 1350, 2250, 2700, 3600, 3150, 4050]
    }).set_index(['coin_id', 'wallet_address', 'date'])

@pytest.mark.unit
def test_calculate_new_profits_values_multiple_coins_wallets(
        multiple_coins_wallets_df,
        target_date
        ):
    """
    Test calculate_new_profits_values function with multiple coin_ids and wallet_addresses.

    This test checks if the function correctly handles a diverse set of coins and wallet addresses,
    ensuring that calculations are correct and independent for each unique combination.

    Args:
        multiple_coins_wallets_df (pd.DataFrame): Fixture providing sample input data with various
            coins and wallets.
        target_date (datetime): Fixture providing a target date for imputation.
    """
    result = pri.calculate_new_profits_values(multiple_coins_wallets_df, target_date)
    result = result.set_index(['coin_id', 'wallet_address', 'date'])

    # Check if calculations are correct for all rows
    for (coin_id, wallet_address), group in multiple_coins_wallets_df.groupby(['coin_id', 'wallet_address']):
        result_row = result.loc[(coin_id, wallet_address, target_date)]
        original_row = group.iloc[0]  # Take the first row of the group

        price_ratio = original_row['price'] / original_row['price_previous']
        expected_profits_change = (price_ratio - 1) * original_row['usd_balance']
        expected_profits_cumulative = original_row['profits_cumulative'] + expected_profits_change
        expected_usd_balance = price_ratio * original_row['usd_balance']

        assert result_row['profits_cumulative'] == pytest.approx(expected_profits_cumulative, abs=1e-4)
        assert result_row['usd_balance'] == pytest.approx(expected_usd_balance, abs=1e-4)
        assert result_row['usd_net_transfers'] == 0
        assert result_row['usd_inflows'] == 0
        assert result_row['usd_inflows_cumulative'] == original_row['usd_inflows_cumulative']
        assert result_row['total_return'] == pytest.approx(
            expected_profits_cumulative / original_row['usd_inflows_cumulative'], abs=1e-4)

    # Check if all rows have the correct target_date
    assert (result.index.get_level_values('date') == target_date).all()

    # Check if the number of rows matches the input DataFrame
    assert len(result) == len(multiple_coins_wallets_df.groupby(['coin_id', 'wallet_address']))

    # Check if all unique coin_ids and wallet_addresses are preserved
    assert set(result.index.get_level_values('coin_id')) == set(
        multiple_coins_wallets_df.index.get_level_values('coin_id'))
    assert set(result.index.get_level_values('wallet_address')) == set(
        multiple_coins_wallets_df.index.get_level_values('wallet_address'))



# -------------------------------------------- #
# impute_profits_df_rows() unit tests
# -------------------------------------------- #

@pytest.fixture
def sample_profits_df_missing_dates():
    """
    Fixture to create a sample profits DataFrame with missing intermediate dates for testing.
    """
    data = {
        'coin_id': ['BTC', 'ETH', 'BTC', 'ETH', 'BTC', 'ETH', 'BTC', 'ETH'],
        'wallet_address': ['wallet1', 'wallet1', 'wallet2', 'wallet2',
                           'wallet1', 'wallet1', 'wallet2', 'wallet2'],
        'date': [
            '2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01',
            '2023-01-05', '2023-01-05', '2023-01-07', '2023-01-07'
        ],
        'profits_change': [100, 50, 200, 75, 50, 25, 150, 100],
        'usd_net_transfers': [0, 0, 0, 0, 50, 25, -100, -50],
        'usd_inflows': [1000, 500, 2000, 1000, 0, 0, 0, 0]
    }

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])

    # Set the index before calculations
    df = df.set_index(['coin_id', 'wallet_address', 'date']).sort_index()

    # Calculate cumulative and derived columns
    df['profits_cumulative'] = df.groupby(['coin_id', 'wallet_address'])['profits_change'].cumsum()
    df = df.drop(columns='profits_change')
    df['usd_inflows_cumulative'] = df.groupby(['coin_id', 'wallet_address'])['usd_inflows'].cumsum()

    # Calculate usd_balance
    df['usd_balance'] = (df['usd_inflows_cumulative'] +
                         df['profits_cumulative'] +
                         df.groupby(['coin_id', 'wallet_address'])['usd_net_transfers'].cumsum())

    # Calculate total_return
    df['total_return'] = df['profits_cumulative'] / df['usd_inflows_cumulative']
    df = df.reset_index()

    return df


@pytest.fixture
def sample_prices_df_missing_dates():
    """
    Fixture to create a sample prices DataFrame with continuous dates for testing.
    """
    dates = pd.date_range(start='2023-01-01', end='2023-01-07')
    data = {
        'coin_id': ['BTC', 'ETH'] * 7,
        'date': [date for date in dates for _ in range(2)],
        'price': [
            10000, 200,  # 2023-01-01
            10100, 205,  # 2023-01-02
            10200, 210,  # 2023-01-03
            10300, 215,  # 2023-01-04
            10400, 220,  # 2023-01-05
            10500, 225,  # 2023-01-06
            10600, 230   # 2023-01-07
        ]
    }

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    return df

@pytest.mark.unit
def test_impute_profits_df_rows_base_case(sample_profits_df_missing_dates,
                                          sample_prices_df_missing_dates):
    """
    Test the base case for impute_profits_df_rows function.

    This test checks the basic functionality of the function, including:
    - Correct output structure
    - Imputation of missing rows
    - Correct calculations for imputed rows
    - Handling of multiple coins and wallets
    """
    target_date = pd.Timestamp('2023-01-06')

    result = pri.impute_profits_df_rows(sample_profits_df_missing_dates,
                                       sample_prices_df_missing_dates,
                                       target_date)

    # Check output structure
    assert isinstance(result, pd.DataFrame)
    expected_columns = ['coin_id', 'wallet_address', 'date', 'profits_cumulative',
                        'usd_balance', 'usd_net_transfers', 'usd_inflows',
                        'usd_inflows_cumulative', 'total_return']
    assert set(result.columns) == set(expected_columns)

    # Check that only rows for the target date are returned
    assert (result['date'] == target_date).all()

    # Check that the correct number of rows are imputed
    assert len(result) == 4  # 2 coins * 2 wallets

    # Check calculations for imputed rows
    for _, row in result.iterrows():
        coin_id = row['coin_id']
        wallet_address = row['wallet_address']

        # Get the last known values for this coin-wallet pair
        last_known = sample_profits_df_missing_dates[
            (sample_profits_df_missing_dates['coin_id'] == coin_id) &
            (sample_profits_df_missing_dates['wallet_address'] == wallet_address) &
            (sample_profits_df_missing_dates['date'] < target_date)
        ].iloc[-1]

        # Get the prices
        price_previous = sample_prices_df_missing_dates[
            (sample_prices_df_missing_dates['coin_id'] == coin_id) &
            (sample_prices_df_missing_dates['date'] == last_known['date'])
        ]['price'].values[0]
        price_current = sample_prices_df_missing_dates[
            (sample_prices_df_missing_dates['coin_id'] == coin_id) &
            (sample_prices_df_missing_dates['date'] == target_date)
        ]['price'].values[0]

        # Check calculations
        price_ratio = price_current / price_previous
        expected_profits_change = (price_ratio - 1) * last_known['usd_balance']
        expected_profits_cumulative = last_known['profits_cumulative'] + expected_profits_change
        expected_usd_balance = last_known['usd_balance'] * price_ratio

        assert row['profits_cumulative'] == pytest.approx(expected_profits_cumulative, rel=1e-6)
        assert row['usd_balance'] == pytest.approx(expected_usd_balance, rel=1e-6)
        assert row['usd_net_transfers'] == 0
        assert row['usd_inflows'] == 0
        assert row['usd_inflows_cumulative'] == last_known['usd_inflows_cumulative']
        assert row['total_return'] == pytest.approx(
            row['profits_cumulative'] / row['usd_inflows_cumulative'], rel=1e-6)

    # Check handling of multiple coins and wallets
    assert set(result['coin_id']) == {'BTC', 'ETH'}
    assert set(result['wallet_address']) == {'wallet1', 'wallet2'}

@pytest.mark.unit
def test_impute_profits_df_rows_early_target_date(
        sample_profits_df_missing_dates,sample_prices_df_missing_dates):
    """
    Test that impute_profits_df_rows raises a ValueError when the target date is earlier than all
    dates in profits_df.
    """
    # Earlier than any date in sample_profits_df_missing_dates
    early_target_date = pd.Timestamp('2022-12-31')

    with pytest.raises(ValueError) as excinfo:
        pri.impute_profits_df_rows(
            sample_profits_df_missing_dates,
            sample_prices_df_missing_dates,
            early_target_date)

    assert "Target date is earlier than all dates in profits_df" in str(excinfo.value)

@pytest.mark.unit
def test_impute_profits_df_rows_late_target_date(
        sample_profits_df_missing_dates,
        sample_prices_df_missing_dates
        ):
    """
    Test that impute_profits_df_rows raises a ValueError when the target date is later than all
    dates in prices_df.
    """
    # Later than any date in sample_prices_df_missing_dates
    late_target_date = pd.Timestamp('2023-01-08')

    with pytest.raises(ValueError) as excinfo:
        pri.impute_profits_df_rows(
            sample_profits_df_missing_dates,
            sample_prices_df_missing_dates,
            late_target_date)

    assert "Target date is later than all dates in prices_df" in str(excinfo.value)

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
def profits_df_base():
    """
    retrieves transfers_df for data quality checks
    """
    logger.info("Beginning integration testing...")
    logger.info("Generating profits_df fixture from production data...")
    # retrieve profits data
    profits_df = dr.retrieve_profits_data(TRAINING_PERIOD_START,
                                          MODELING_PERIOD_END,
                                          config['data_cleaning']['minimum_wallet_inflows'])

    # filter data to only 5% of coin_ids
    np.random.seed(42)
    unique_coin_ids = profits_df['coin_id'].unique()
    sample_coin_ids = np.random.choice(unique_coin_ids, size=int(0.05 * len(unique_coin_ids)), replace=False)
    profits_df = profits_df[profits_df['coin_id'].isin(sample_coin_ids)]

    # perform the rest of the data cleaning steps
    profits_df, _ = cwm.split_dataframe_by_coverage(
        profits_df,
        TRAINING_PERIOD_START,
        MODELING_PERIOD_END,
        id_column='coin_id'
    )
    return profits_df

@pytest.fixture(scope='session')
def cleaned_profits_df(profits_df_base):
    """
    Fixture to run clean_profits_df() and return both the cleaned DataFrame and exclusions DataFrame.
    Uses thresholds from the config file.
    """
    logger.info("Generating cleaned_profits_df from clean_profits_df()...")
    cleaned_df, exclusions_df = dr.clean_profits_df(profits_df_base, config['data_cleaning'])
    return cleaned_df, exclusions_df


@contextlib.contextmanager
def single_threaded():
    """
    helper function to avoid multithreading which breaks in pytest
    """
    _original_thread_count = threading.active_count()
    yield
    current_thread_count = threading.active_count()
    if current_thread_count > _original_thread_count:
        raise AssertionError(f"Test created new threads: {current_thread_count - _original_thread_count}")


@pytest.fixture(scope='session')
def profits_df(prices_df,cleaned_profits_df):
    """
    Returns the final profits_df after the full processing sequence.
    """
    profits_df, _ = cleaned_profits_df

    # impute period boundary dates
    dates_to_impute = [
        TRAINING_PERIOD_END,
        MODELING_PERIOD_START,
        MODELING_PERIOD_END
    ]
    # this must use only 1 thread to work in a testing environment
    with single_threaded():
        profits_df = pri.impute_profits_for_multiple_dates(profits_df, prices_df, dates_to_impute, n_threads=1)

    return profits_df

# Save profits_df.csv in fixtures/
# ----------------------------------------
def test_save_profits_df(profits_df):
    """
    This is not a test! This function saves a cleaned_profits_df.csv in the fixtures folder so it can be \
    used for integration tests in other modules.
    """

    # Save the cleaned DataFrame to the fixtures folder
    profits_df.to_csv('tests/fixtures/cleaned_profits_df.csv', index=False)

    # Add some basic assertions to ensure the data was saved correctly
    assert profits_df is not None
    assert len(profits_df) > 0



@pytest.mark.integration
class TestProfitsDataQuality:
    """
    Various tests on the data quality of the final profits_df version
    """
    def test_no_duplicate_records(self, profits_df):
        """Test 1: No duplicate records"""
        deduped_df = profits_df[['coin_id', 'wallet_address', 'date']].drop_duplicates()
        logger.info(f"Original profits_df length: {len(profits_df)}, Deduplicated: {len(deduped_df)}")
        assert (len(profits_df) == len(deduped_df)
                ), "There are duplicate rows based on coin_id, wallet_address, and date"

    def test_records_at_training_period_end(self, profits_df):
        """Test 2: All coin-wallet pairs have a record at the end of the training period"""
        profits_df_filtered = profits_df[profits_df['date'] < MODELING_PERIOD_START]
        pairs_in_training_period = profits_df_filtered[['coin_id', 'wallet_address']].drop_duplicates()
        period_end_df = profits_df[profits_df['date'] == TRAINING_PERIOD_END]

        logger.info(f"Found {len(pairs_in_training_period)} total pairs in training period with \
                    {len(period_end_df)} having data at period end.")
        assert (len(pairs_in_training_period) == len(period_end_df)
                ), "Not all coin-wallet pairs have a record at the end of the training period"

    def test_no_negative_usd_balances(self, profits_df):
        """Test 3: No negative USD balances"""
        negative_balances = profits_df[profits_df['usd_balance'] < -0.1]
        logger.info(f"Found {len(negative_balances)} records with negative USD balances.")
        assert len(negative_balances) == 0, "There are negative USD balances in the dataset"

    def test_date_range(self, profits_df):
        """Test 4: Date range check"""
        min_date = profits_df['date'].min()
        max_date = profits_df['date'].max()
        expected_max_date = pd.to_datetime(MODELING_PERIOD_END)
        logger.info(f"profits_df date range: {min_date} to {max_date}")
        assert (max_date == expected_max_date
                ), f"The last date in the dataset should be {expected_max_date}"

    def test_no_missing_values(self, profits_df):
        """Test 5: No missing values"""
        missing_values = profits_df.isna().sum()
        assert (missing_values.sum() == 0
                ), f"There are missing values in the dataset: {missing_values[missing_values > 0]}"

    def test_records_at_training_period_end_all_wallets(self, profits_df):
        """Test 7: Ensure all applicable wallets have records as of the training_period_end"""
        training_profits_df = profits_df[profits_df['date'] <= TRAINING_PERIOD_END]
        training_wallets_df = training_profits_df[['coin_id', 'wallet_address']].drop_duplicates()

        training_end_df = profits_df[profits_df['date'] == TRAINING_PERIOD_END]
        training_end_df = training_end_df[['coin_id', 'wallet_address']].drop_duplicates()

        assert (len(training_wallets_df) == len(training_end_df)
                ), "Some wallets are missing a record as of the training_period_end"

    def test_records_at_modeling_period_start(self, profits_df):
        """Test 8: Ensure all wallets have records as of the modeling_period_start"""
        modeling_profits_df = profits_df[profits_df['date'] <= MODELING_PERIOD_START]
        modeling_wallets_df = modeling_profits_df[['coin_id', 'wallet_address']].drop_duplicates()

        modeling_start_df = profits_df[profits_df['date'] == MODELING_PERIOD_START]
        modeling_start_df = modeling_start_df[['coin_id', 'wallet_address']].drop_duplicates()

        assert (len(modeling_wallets_df) == len(modeling_start_df)
                ), "Some wallets are missing a record as of the modeling_period_start"

    def test_records_at_modeling_period_end(self, profits_df):
        """Test 9: Ensure all wallets have records as of the modeling_period_end"""
        modeling_profits_df = profits_df[profits_df['date'] <= MODELING_PERIOD_END]
        modeling_wallets_df = modeling_profits_df[['coin_id', 'wallet_address']].drop_duplicates()

        modeling_end_df = profits_df[profits_df['date'] == MODELING_PERIOD_END]
        modeling_end_df = modeling_end_df[['coin_id', 'wallet_address']].drop_duplicates()

        assert (len(modeling_wallets_df) == len(modeling_end_df)
                ), "Some wallets are missing a record as of the modeling_period_end"

    def test_no_records_before_training_period_start(self, profits_df):
        """Test 10: Confirm no records exist prior to the training period start"""
        early_records = profits_df[profits_df['date'] < TRAINING_PERIOD_START]
        assert (len(early_records) == 0
                ), f"Found {len(early_records)} records prior to training_period_start"


# ---------------------------------------- #
# retrieve_market_data() integration tests
# ---------------------------------------- #

@pytest.fixture(scope='session')
def market_data_df():
    """
    Retrieve and preprocess the market_data_df, filling gaps as needed.
    """
    logger.info("Generating market_data_df from production data...")
    market_data_df = dr.retrieve_market_data()
    market_data_df = dr.clean_market_data(market_data_df, config)
    market_data_df, _ = cwm.split_dataframe_by_coverage(
        market_data_df,
        start_date=config['training_data']['training_period_start'],
        end_date=config['training_data']['modeling_period_end'],
        id_column='coin_id'
    )
    return market_data_df

@pytest.fixture(scope='session')
def prices_df(market_data_df):
    """
    Retrieve and preprocess the market_data_df, filling gaps as needed.
    """
    prices_df = market_data_df[['coin_id','date','price']].copy()
    return prices_df


# Save market_data_df.csv in fixtures/
# ----------------------------------------
def test_save_market_data_df(market_data_df, prices_df):
    """
    This is not a test! This function saves a market_data_df.csv in the fixtures folder so it
    can be used for integration tests in other modules.
    """
    # Save the prices DataFrame to the fixtures folder
    market_data_df.to_csv('tests/fixtures/market_data_df.csv', index=False)
    prices_df.to_csv('tests/fixtures/prices_df.csv', index=False)


    # Add some basic assertions to ensure the data was saved correctly
    assert market_data_df is not None
    assert len(market_data_df) > 0

    # Assert that there are no null values
    assert market_data_df.isna().sum().sum() == 0



# ---------------------------------------- #
# retrieve_metadata_data() integration tests
# ---------------------------------------- #

@pytest.fixture(scope='session')
def metadata_df():
    """
    Retrieve and preprocess the metadata_df.
    """
    logger.info("Generating metadata_df from production data...")
    metadata_df = dr.retrieve_metadata_data()
    return metadata_df

# Save metadata_df.csv in fixtures/
# ----------------------------------------
def test_save_metadata_df(metadata_df):
    """
    This is not a test! This function saves a metadata_df.csv in the fixtures folder so it
    can be used for integration tests in other modules.
    """
    # Save the metadata DataFrame to the fixtures folder
    metadata_df.to_csv('tests/fixtures/metadata_df.csv', index=False)

    # Add some basic assertions to ensure the data was saved correctly
    assert metadata_df is not None
    assert len(metadata_df) > 0

    # Assert that coin_id is unique
    assert metadata_df['coin_id'].is_unique, "coin_id is not unique in metadata_df"


# ---------------------------------------- #
# clean_profits_df() integration tests
# ---------------------------------------- #

@pytest.mark.integration
def test_clean_profits_exclusions(cleaned_profits_df, profits_df_base):
    """
    Test that all excluded wallets breach either the profitability or inflows threshold.
    Uses thresholds from the config file.
    """
    cleaned_df, exclusions_df = cleaned_profits_df

    # Check that every excluded wallet breached at least one threshold
    exclusions_with_breaches = exclusions_df.merge(profits_df_base, on='wallet_address', how='inner')

    # Calculate the total profits and inflows per wallet
    wallet_coin_agg_df = (exclusions_with_breaches.sort_values('date')
                                                  .groupby(['wallet_address','coin_id'], observed=True)
                                                  .agg({
                                                     'profits_cumulative': 'last',
                                                     'usd_inflows_cumulative': 'last'
                                                 }).reset_index())

    wallet_agg_df = (wallet_coin_agg_df.groupby('wallet_address')
                                       .agg({
                                           'profits_cumulative': 'sum',
                                           'usd_inflows_cumulative': 'sum'
                                       })
                                       .reset_index())

    # Apply threshold check from the config
    profitability_filter = config['data_cleaning']['profitability_filter']
    inflows_filter = config['data_cleaning']['inflows_filter']

    breaches_df = wallet_agg_df[
        (wallet_agg_df['profits_cumulative'] >= profitability_filter) |
        (wallet_agg_df['profits_cumulative'] <= -profitability_filter) |
        (wallet_agg_df['usd_inflows_cumulative'] >= inflows_filter)
    ]
    # Assert that all excluded wallets breached a threshold
    assert len(exclusions_df) == len(breaches_df), "Some excluded wallets do not breach a threshold."

@pytest.mark.integration
def test_clean_profits_remaining_count(cleaned_profits_df, profits_df_base):
    """
    Test that the count of remaining records in the cleaned DataFrame matches the expected count.
    """
    cleaned_df, exclusions_df = cleaned_profits_df

    # Get the total number of unique wallets before and after cleaning
    input_wallet_count = profits_df_base['wallet_address'].nunique()
    cleaned_wallet_count = cleaned_df['wallet_address'].nunique()
    excluded_wallet_count = exclusions_df['wallet_address'].nunique()

    # Assert that the remaining records equal the difference between the input and excluded records
    assert input_wallet_count == cleaned_wallet_count + excluded_wallet_count, \
        "The count of remaining wallets does not match the input minus excluded records."

@pytest.mark.integration
def test_clean_profits_aggregate_sums(cleaned_profits_df):
    """
    Test that the aggregation of profits and inflows for the remaining wallets stays within the
    configured thresholds. Uses thresholds from the config file.
    """
    cleaned_df, _ = cleaned_profits_df

    # Aggregate the profits and inflows for the remaining wallets
    remaining_wallet_coins_agg_df = (cleaned_df.sort_values('date')
                                            .groupby(['coin_id','wallet_address'], observed=True)
                                            .agg({
                                                'profits_cumulative': 'last',
                                                'usd_inflows_cumulative': 'last'
                                            })
                                            .reset_index())

    remaining_wallets_agg_df = (remaining_wallet_coins_agg_df.groupby('wallet_address')
                                            .agg({
                                                'profits_cumulative': 'sum',
                                                'usd_inflows_cumulative': 'sum'
                                            })
                                            .reset_index())


    # Apply the thresholds from the config
    profitability_filter = config['data_cleaning']['profitability_filter']
    inflows_filter = config['data_cleaning']['inflows_filter']
    # Ensure no remaining wallets exceed the thresholds
    over_threshold_wallets = remaining_wallets_agg_df[
        (remaining_wallets_agg_df['profits_cumulative'] >= profitability_filter) |
        (remaining_wallets_agg_df['profits_cumulative'] <= -profitability_filter) |
        (remaining_wallets_agg_df['usd_inflows_cumulative'] >= inflows_filter)
    ]

    # Assert that no wallets in the cleaned DataFrame breach the thresholds
    assert over_threshold_wallets.empty, "Some remaining wallets exceed the thresholds."
