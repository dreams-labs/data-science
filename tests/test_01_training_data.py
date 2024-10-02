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




# -------------------------------------------- #
# generate_coin_metadata_features() unit tests
# -------------------------------------------- #

@pytest.fixture
def mock_metadata_df():
    """
    Fixture to create a mock metadata DataFrame for testing.
    Includes edge cases for chain threshold and category unpacking.
    """
    data = {
        'coin_id': ['coin_1', 'coin_2', 'coin_3', 'coin_4', 'coin_5'],
        'categories': [['meme'], [], ['defi', 'dex'], ['nft'], []],
        'chain': ['ethereum', 'solana', 'solana', 'binance', 'base']
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_config():
    """
    Fixture to provide a mock configuration with a chain threshold value.
    """
    return {
        'datasets': {
            'coin_facts': {
                'coin_metadata': {
                    'chain_threshold': 2  # Threshold set to 2 for testing
                }
            }
        }
    }

# Chain Threshold Tests
@pytest.mark.unit
def test_chain_threshold(mock_metadata_df, mock_config):
    """
    Test to ensure that chains are correctly categorized based on the threshold.
    - Chain 'solana' has 2 coins (at the threshold).
    - Chain 'ethereum' has 1 coin (below the threshold).
    - Chain 'binance' and 'base' have 1 coin (below the threshold).
    """
    result_df = td.generate_coin_metadata_features(mock_metadata_df, mock_config)

    # Assert solana is included as a boolean column
    assert 'chain_solana' in result_df.columns
    assert result_df['chain_solana'].sum() == 2  # 2 solana coins

    # Assert ethereum, binance, and base are categorized as 'chain_other'
    assert 'chain_other' in result_df.columns
    assert result_df['chain_other'].sum() == 3  # 3 coins below the threshold

    # Assert chains like ethereum, binance, and base don't have their own columns
    assert 'chain_ethereum' not in result_df.columns
    assert 'chain_binance' not in result_df.columns
    assert 'chain_base' not in result_df.columns

# Category Unpacking Tests
@pytest.mark.unit
def test_category_unpacking(mock_metadata_df, mock_config):
    """
    Test to ensure that categories are correctly unpacked, including:
    - Coins with 0 categories.
    - Coins with 1 category.
    - Coins with 2+ categories.
    """
    result_df = td.generate_coin_metadata_features(mock_metadata_df, mock_config)

    # Assert correct category columns exist
    assert 'category_meme' in result_df.columns
    assert 'category_defi' in result_df.columns
    assert 'category_dex' in result_df.columns
    assert 'category_nft' in result_df.columns

    # Test coins with 0 categories
    assert not result_df.loc[result_df['coin_id'] == 'coin_2', 'category_meme'].values[0]
    assert not result_df.loc[result_df['coin_id'] == 'coin_5', 'category_meme'].values[0]

    # Test coins with 1 category
    assert result_df.loc[result_df['coin_id'] == 'coin_1', 'category_meme'].values[0]
    assert result_df.loc[result_df['coin_id'] == 'coin_4', 'category_nft'].values[0]

    # Test coins with 2+ categories
    assert result_df.loc[result_df['coin_id'] == 'coin_3', 'category_defi'].values[0]
    assert result_df.loc[result_df['coin_id'] == 'coin_3', 'category_dex'].values[0]




# -------------------------------------------- #
# calculate_new_profits_values() unit tests
# -------------------------------------------- #

@pytest.fixture
def sample_profits_df():
    """
    Fixture to create a sample profits DataFrame for testing.

    Returns:
        pd.DataFrame: A DataFrame with sample data for testing calculate_new_profits_values function.
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
    result = td.calculate_new_profits_values(sample_profits_df, target_date)

    # Check if the result has the correct structure
    assert isinstance(result, pd.DataFrame)
    assert result.index.names == ['coin_id', 'wallet_address', 'date']
    assert set(result.columns) == {
        'profits_change', 'profits_cumulative', 'usd_balance',
        'usd_net_transfers', 'usd_inflows', 'usd_inflows_cumulative', 'total_return'
    }

    # Check if calculations are correct for the first row (btc, addr1)
    first_row = result.loc[('btc', 'addr1', target_date)]
    assert first_row['profits_change'] == pytest.approx(55.55555556)  # (9500/9000 - 1) * 1000
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
    result = td.calculate_new_profits_values(zero_price_change_df, target_date)

    # Check if the result has the correct structure
    assert isinstance(result, pd.DataFrame)
    assert result.index.names == ['coin_id', 'wallet_address', 'date']
    assert set(result.columns) == {
        'profits_change', 'profits_cumulative', 'usd_balance',
        'usd_net_transfers', 'usd_inflows', 'usd_inflows_cumulative', 'total_return'
    }

    # Check if calculations are correct for all rows
    for (coin_id, wallet_address), group in zero_price_change_df.groupby(['coin_id', 'wallet_address']):
        result_row = result.loc[(coin_id, wallet_address, target_date)]
        original_row = group.iloc[0]  # Take the first row of the group

        assert result_row['profits_change'] == pytest.approx(0, abs=1e-4)
        assert result_row['profits_cumulative'] == pytest.approx(original_row['profits_cumulative'], abs=1e-4)
        assert result_row['usd_balance'] == pytest.approx(original_row['usd_balance'], abs=1e-4)
        assert result_row['usd_net_transfers'] == 0
        assert result_row['usd_inflows'] == 0
        assert result_row['usd_inflows_cumulative'] == original_row['usd_inflows_cumulative']
        assert result_row['total_return'] == pytest.approx(original_row['profits_cumulative'] / original_row['usd_inflows_cumulative'], abs=1e-4)

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
        negative_price_change_df (pd.DataFrame): Fixture providing sample input data with price decreases.
        target_date (datetime): Fixture providing a target date for imputation.
    """
    result = td.calculate_new_profits_values(negative_price_change_df, target_date)

    # Check if the result has the correct structure
    assert isinstance(result, pd.DataFrame)
    assert result.index.names == ['coin_id', 'wallet_address', 'date']
    assert set(result.columns) == {
        'profits_change', 'profits_cumulative', 'usd_balance',
        'usd_net_transfers', 'usd_inflows', 'usd_inflows_cumulative', 'total_return'
    }

    # Check if calculations are correct for all rows
    for (coin_id, wallet_address), group in negative_price_change_df.groupby(['coin_id', 'wallet_address']):
        result_row = result.loc[(coin_id, wallet_address, target_date)]
        original_row = group.iloc[0]  # Take the first row of the group

        price_ratio = original_row['price'] / original_row['price_previous']
        expected_profits_change = (price_ratio - 1) * original_row['usd_balance']
        expected_profits_cumulative = original_row['profits_cumulative'] + expected_profits_change
        expected_usd_balance = price_ratio * original_row['usd_balance']

        assert result_row['profits_change'] == pytest.approx(expected_profits_change, abs=1e-4)
        assert result_row['profits_cumulative'] == pytest.approx(expected_profits_cumulative, abs=1e-4)
        assert result_row['usd_balance'] == pytest.approx(expected_usd_balance, abs=1e-4)
        assert result_row['usd_net_transfers'] == 0
        assert result_row['usd_inflows'] == 0
        assert result_row['usd_inflows_cumulative'] == original_row['usd_inflows_cumulative']
        assert result_row['total_return'] == pytest.approx(expected_profits_cumulative / original_row['usd_inflows_cumulative'], abs=1e-4)

        # Additional checks specific to negative price change
        assert result_row['profits_change'] < 0
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
    result = td.calculate_new_profits_values(zero_usd_balance_df, target_date)

    # Check if the result has the correct structure
    assert isinstance(result, pd.DataFrame)
    assert result.index.names == ['coin_id', 'wallet_address', 'date']
    assert set(result.columns) == {
        'profits_change', 'profits_cumulative', 'usd_balance',
        'usd_net_transfers', 'usd_inflows', 'usd_inflows_cumulative', 'total_return'
    }

    # Check if calculations are correct for all rows
    for (coin_id, wallet_address), group in zero_usd_balance_df.groupby(['coin_id', 'wallet_address']):
        result_row = result.loc[(coin_id, wallet_address, target_date)]
        original_row = group.iloc[0]  # Take the first row of the group

        assert result_row['profits_change'] == 0
        assert result_row['profits_cumulative'] == original_row['profits_cumulative']
        assert result_row['usd_balance'] == 0
        assert result_row['usd_net_transfers'] == 0
        assert result_row['usd_inflows'] == 0
        assert result_row['usd_inflows_cumulative'] == original_row['usd_inflows_cumulative']
        assert result_row['total_return'] == pytest.approx(original_row['profits_cumulative'] / original_row['usd_inflows_cumulative'], abs=1e-4)

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
def test_calculate_new_profits_values_multiple_coins_wallets(multiple_coins_wallets_df, target_date):
    """
    Test calculate_new_profits_values function with multiple coin_ids and wallet_addresses.

    This test checks if the function correctly handles a diverse set of coins and wallet addresses,
    ensuring that calculations are correct and independent for each unique combination.

    Args:
        multiple_coins_wallets_df (pd.DataFrame): Fixture providing sample input data with various coins and wallets.
        target_date (datetime): Fixture providing a target date for imputation.
    """
    result = td.calculate_new_profits_values(multiple_coins_wallets_df, target_date)

    # Check if the result has the correct structure
    assert isinstance(result, pd.DataFrame)
    assert result.index.names == ['coin_id', 'wallet_address', 'date']
    assert set(result.columns) == {
        'profits_change', 'profits_cumulative', 'usd_balance',
        'usd_net_transfers', 'usd_inflows', 'usd_inflows_cumulative', 'total_return'
    }

    # Check if calculations are correct for all rows
    for (coin_id, wallet_address), group in multiple_coins_wallets_df.groupby(['coin_id', 'wallet_address']):
        result_row = result.loc[(coin_id, wallet_address, target_date)]
        original_row = group.iloc[0]  # Take the first row of the group

        price_ratio = original_row['price'] / original_row['price_previous']
        expected_profits_change = (price_ratio - 1) * original_row['usd_balance']
        expected_profits_cumulative = original_row['profits_cumulative'] + expected_profits_change
        expected_usd_balance = price_ratio * original_row['usd_balance']

        assert result_row['profits_change'] == pytest.approx(expected_profits_change, abs=1e-4)
        assert result_row['profits_cumulative'] == pytest.approx(expected_profits_cumulative, abs=1e-4)
        assert result_row['usd_balance'] == pytest.approx(expected_usd_balance, abs=1e-4)
        assert result_row['usd_net_transfers'] == 0
        assert result_row['usd_inflows'] == 0
        assert result_row['usd_inflows_cumulative'] == original_row['usd_inflows_cumulative']
        assert result_row['total_return'] == pytest.approx(expected_profits_cumulative / original_row['usd_inflows_cumulative'], abs=1e-4)

    # Check if all rows have the correct target_date
    assert (result.index.get_level_values('date') == target_date).all()

    # Check if the number of rows matches the input DataFrame
    assert len(result) == len(multiple_coins_wallets_df.groupby(['coin_id', 'wallet_address']))

    # Check if all unique coin_ids and wallet_addresses are preserved
    assert set(result.index.get_level_values('coin_id')) == set(multiple_coins_wallets_df.index.get_level_values('coin_id'))
    assert set(result.index.get_level_values('wallet_address')) == set(multiple_coins_wallets_df.index.get_level_values('wallet_address'))



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
def profits_df():
    """
    retrieves transfers_df for data quality checks
    """
    logger.info("Beginning integration testing...")
    logger.info("Generating profits_df fixture from production data...")
    # retrieve profits data
    profits_df = td.retrieve_profits_data(TRAINING_PERIOD_START, MODELING_PERIOD_END)
    profits_df, _ = cwm.split_dataframe_by_coverage(
        profits_df,
        TRAINING_PERIOD_START,
        MODELING_PERIOD_END,
        id_column='coin_id'
    )
    profits_df, _ = td.clean_profits_df(profits_df, config['data_cleaning'])

    # impute period boundary dates
    dates_to_impute = [
        TRAINING_PERIOD_END,
        MODELING_PERIOD_START,
        MODELING_PERIOD_END
    ]
    profits_df = td.impute_profits_for_multiple_dates(profits_df, prices_df, dates_to_impute, n_threads=24)

    return profits_df

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
# retrieve_market_data() integration tests
# ---------------------------------------- #

@pytest.fixture(scope='session')
def market_data_df():
    """
    Retrieve and preprocess the market_data_df, filling gaps as needed.
    """
    logger.info("Generating market_data_df from production data...")
    market_data_df = td.retrieve_market_data()
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
    This is not a test! This function saves a market_data_df.csv in the fixtures folder so it can be
    used for integration tests in other modules.
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
    metadata_df = td.retrieve_metadata_data()
    return metadata_df

# Save metadata_df.csv in fixtures/
# ----------------------------------------
def test_save_metadata_df(metadata_df):
    """
    This is not a test! This function saves a metadata_df.csv in the fixtures folder so it can be
    used for integration tests in other modules.
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
    wallet_agg_df = exclusions_with_breaches.groupby('wallet_address', observed=True).agg({
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
    remaining_wallets_agg_df = cleaned_df.groupby('wallet_address', observed=True).agg({
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
