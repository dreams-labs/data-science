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
# retrieve_transfers_data()
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
# calculate_wallet_profitability()
# ------------------------------- #

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
        'net_transfers': [10, 5, 100, 50, 20, 25, 1000, 500, -750,
                          0, 0, 0, -10, 0],
        'balance': [10, 15, 100, 50, 20, 75, 1000, 1500, 750,
                    15, 100, 20, 65, 750]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_prices_df():
    """
    Create a sample prices DataFrame for testing.
    """
    data = {
        'coin_id': ['BTC', 'BTC', 'BTC', 'BTC', 'ETH', 'ETH', 'ETH', 'ETH', 'MYRO', 'MYRO', 'MYRO', 'MYRO'],
        'date': [
            '2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01',
            '2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01',
            '2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01'
        ],
        'price': [20000, 21000, 22000, 23000, 1500, 1600, 1700, 1800, 10, 15, 12, 8]
    }
    return pd.DataFrame(data)

def test_calculate_wallet_profitability_profitability(sample_transfers_df, sample_prices_df):
    """
    Test profitability calculations for multiple wallets and coins.
    """
    result = td.calculate_wallet_profitability(sample_transfers_df, sample_prices_df)
    
    # Check profitability for wallet1, BTC
    wallet1_btc = result[(result['wallet_address'] == 'wallet1') & (result['coin_id'] == 'BTC')]
    assert wallet1_btc.loc[wallet1_btc['date'] == '2023-02-01', 'profits_change'].values[0] == pytest.approx(10000)  # (21000 - 20000) * 10
    assert wallet1_btc.loc[wallet1_btc['date'] == '2023-02-01', 'profits_total'].values[0] == pytest.approx(10000)
    assert wallet1_btc.loc[wallet1_btc['date'] == '2023-04-01', 'profits_change'].values[0] == pytest.approx(30000)  # (23000 - 21000) * 15
    assert wallet1_btc.loc[wallet1_btc['date'] == '2023-04-01', 'profits_total'].values[0] == pytest.approx(40000)  # 10000 + 15000 + 15000

    # Check profitability for wallet2, ETH
    wallet2_eth = result[(result['wallet_address'] == 'wallet2') & (result['coin_id'] == 'ETH')]
    assert wallet2_eth.loc[wallet2_eth['date'] == '2023-02-01', 'profits_change'].values[0] == pytest.approx(5000)  # (1600 - 1500) * 50
    assert wallet2_eth.loc[wallet2_eth['date'] == '2023-02-01', 'profits_total'].values[0] == pytest.approx(5000)
    assert wallet2_eth.loc[wallet2_eth['date'] == '2023-04-01', 'profits_change'].values[0] == pytest.approx(15000)  # (1800 - 1600) * 75
    assert wallet2_eth.loc[wallet2_eth['date'] == '2023-04-01', 'profits_total'].values[0] == pytest.approx(20000)  # 5000 + 15000

    # Check profitability for wallet3, MYRO
    wallet3_myro = result[(result['wallet_address'] == 'wallet3') & (result['coin_id'] == 'MYRO')]
    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-02-01', 'profits_change'].values[0] == pytest.approx(5000)  # (15 - 10) * 1000
    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-03-01', 'profits_change'].values[0] == pytest.approx(-4500)  # (12 - 15) * 1500
    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-03-01', 'profits_total'].values[0] == pytest.approx(500)
    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-04-01', 'profits_change'].values[0] == pytest.approx(-3000)  # (8 - 12) * 750
    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-04-01', 'profits_total'].values[0] == pytest.approx(-2500)  # 500 - 3000

def test_calculate_wallet_profitability_usd_calculations(sample_transfers_df, sample_prices_df):
    """
    Test USD-related calculations (inflows, balances, total return).
    """
    result = td.calculate_wallet_profitability(sample_transfers_df, sample_prices_df)

    # Check USD calculations for wallet1, BTC
    wallet1_btc = result[(result['wallet_address'] == 'wallet1') & (result['coin_id'] == 'BTC')]
    initial_investment_wallet1 = 10 * 20000  # 10 BTC * $20,000
    second_investment_wallet1 = 5 * 21000    # 5 BTC * $21,000
    total_investment_wallet1 = initial_investment_wallet1 + second_investment_wallet1
    expected_balance_wallet1 = 15 * 23000    # 15 BTC * $23,000
    expected_total_return_wallet1 = 40000 / total_investment_wallet1

    assert wallet1_btc.loc[wallet1_btc['date'] == '2023-01-01', 'usd_inflows'].values[0] == pytest.approx(initial_investment_wallet1)
    assert wallet1_btc.loc[wallet1_btc['date'] == '2023-02-01', 'usd_inflows'].values[0] == pytest.approx(second_investment_wallet1)
    assert wallet1_btc.loc[wallet1_btc['date'] == '2023-02-01', 'usd_total_inflows'].values[0] == pytest.approx(total_investment_wallet1)
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
    assert wallet2_eth.loc[wallet2_eth['date'] == '2023-04-01', 'usd_total_inflows'].values[0] == pytest.approx(total_investment_wallet2)
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
    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-03-01', 'usd_total_inflows'].values[0] == pytest.approx(total_investment_wallet3)
    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-04-01', 'usd_balance'].values[0] == pytest.approx(expected_balance_wallet3)
    assert wallet3_myro.loc[wallet3_myro['date'] == '2023-04-01', 'total_return'].values[0] == pytest.approx(expected_total_return_wallet3)
