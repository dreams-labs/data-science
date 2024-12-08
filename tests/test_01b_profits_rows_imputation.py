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
