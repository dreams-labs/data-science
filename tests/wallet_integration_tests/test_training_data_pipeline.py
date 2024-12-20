"""
tests used to audit the files in the data-science/src folder
"""
# pylint: disable=wrong-import-position  # import not at top of doc (due to local import)
# pylint: disable=redefined-outer-name  # redefining from outer scope triggering on pytest fixtures
# pylint: disable=unused-argument
# pyright: reportMissingModuleSource=false

import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
import pytest
import pandas as pd
from dreams_core import core as dc

# pyright: reportMissingImports=false
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
from wallet_modeling.wallets_config_manager import WalletsConfig
import wallet_modeling.wallet_modeling_orchestrator as wmo

load_dotenv()
logger = dc.setup_logger()

config_path = Path(__file__).parent.parent / 'test_config' / 'test_wallets_config.yaml'
wallets_config = WalletsConfig.load_from_yaml(config_path)


# ===================================================== #
#                                                       #
#           I N T E G R A T I O N   T E S T S           #
#                                                       #
# ===================================================== #

# ------------------------------------------ #
# retrieve_period_datasets() unit tests
# ------------------------------------------ #


@pytest.fixture(scope='session')
def period_datasets():
    """
    Integration fixture retrieving actual wallet data for all periods.

    Returns:
    - tuple: (training_profits_df, training_market_df, modeling_profits_df,
             modeling_market_df, combined_profits_df, combined_market_df)
    """
    # Temporarily adjust log level
    logger.info("Generating training, modeling, and combined datasets...")
    logger.setLevel(logging.WARNING)

    # Target the dev schema to avoid a very long runtime
    wallets_config['training_data']['dataset'] = 'dev'

    # Get initial training data and coin cohort
    training_profits_df, training_market_df, coin_cohort = wmo.retrieve_period_datasets(
        wallets_config['training_data']['training_period_start'],
        wallets_config['training_data']['training_period_end']
    )

    # Get modeling period data
    modeling_profits_df, modeling_market_df, _ = wmo.retrieve_period_datasets(
        wallets_config['training_data']['modeling_period_start'],
        wallets_config['training_data']['modeling_period_end'],
        coin_cohort=coin_cohort
    )

    # Get combined period data
    combined_profits_df, combined_market_df, _ = wmo.retrieve_period_datasets(
        wallets_config['training_data']['training_period_start'],
        wallets_config['training_data']['modeling_period_end'],
        coin_cohort=coin_cohort
    )

    datasets = (
        training_profits_df, training_market_df,
        modeling_profits_df, modeling_market_df,
        combined_profits_df, combined_market_df,
        coin_cohort
    )
    logger.setLevel(logging.INFO)
    logger.info("All dev data retrieved.")

    return datasets


def test_time_period_boundaries(period_datasets):
    """Test that period boundaries align correctly"""
    training_df, _, modeling_df, _, _, _, _ = period_datasets
    training_last = training_df['date'].max()
    modeling_first = modeling_df['date'].min()
    assert training_last == modeling_first

def test_coin_set_consistency(period_datasets):
    """Test that coin sets match between periods"""
    training_df, _, modeling_df, _, combined_df, _, _ = period_datasets
    training_coins = set(training_df['coin_id'])
    modeling_coins = set(modeling_df['coin_id'])
    combined_coins = set(combined_df['coin_id'])
    assert training_coins == combined_coins
    assert len(training_coins - modeling_coins) == 0

def test_transfer_amount_consistency(period_datasets):
    """Test that transfer amounts sum correctly"""
    training_df, _, modeling_df, _, combined_df, _, _ = period_datasets
    training_transfers = abs(training_df['usd_net_transfers']).astype('float64').sum()
    modeling_transfers = abs(modeling_df['usd_net_transfers']).astype('float64').sum()
    combined_transfers = abs(combined_df['usd_net_transfers']).astype('float64').sum()

    balance_diff = abs(combined_transfers - (training_transfers + modeling_transfers))
    balance_pct = balance_diff / combined_transfers if combined_transfers != 0 else 0
    assert (balance_diff < 0.01) or (balance_pct < 0.0001), f"Balance difference: ${balance_diff:,.2f} ({balance_pct:.2%})"

def test_time_period_boundaries_and_balances(period_datasets):
    """
    Test that:
    1. Period boundaries align correctly
    2. Total USD balances match exactly at the boundary
    """
    training_df, _, modeling_df, _, _, _, _ = period_datasets

    # Check date boundaries align
    training_last = training_df['date'].max()
    modeling_first = modeling_df['date'].min()
    assert training_last == modeling_first

    # Check balances match at boundary
    training_end_df = training_df[training_df['date']==training_last]
    modeling_start_df = modeling_df[modeling_df['date']==modeling_first]

    training_end_balance = training_end_df['usd_balance'].astype('float64').sum()
    modeling_start_balance = modeling_start_df['usd_balance'].astype('float64').sum()

    # Balance difference must be within 0.0001%
    assert abs(training_end_balance / modeling_start_balance - 1) < 0.000001

def test_wallet_coin_balance_continuity(period_datasets):
    """
    Test that all wallet-coin pair balances match at the training/modeling boundary
    using vectorized operations. Allows for 0.0001% difference due to floating point math.
    """
    training_df, _, modeling_df, _, _, _, _ = period_datasets

    # Get boundary data
    training_last = training_df['date'].max()
    training_end_df = training_df[training_df['date']==training_last]

    modeling_first = modeling_df['date'].min()
    modeling_start_df = modeling_df[modeling_df['date']==modeling_first]

    # Create merged df on composite key
    balance_compare_df = pd.merge(
        training_end_df[['wallet_address', 'coin_id', 'usd_balance']],
        modeling_start_df[['wallet_address', 'coin_id', 'usd_balance']],
        on=['wallet_address', 'coin_id'],
        suffixes=('_train', '_model')
    )

    # Convert to float64 for consistency
    balance_compare_df['usd_balance_train'] = balance_compare_df['usd_balance_train'].astype('float64')
    balance_compare_df['usd_balance_model'] = balance_compare_df['usd_balance_model'].astype('float64')

    # Filter out zero balance pairs to avoid div by zero
    nonzero_mask = ~((balance_compare_df['usd_balance_train'] == 0) &
                        (balance_compare_df['usd_balance_model'] == 0))
    balance_compare_df = balance_compare_df[nonzero_mask]

    # Calculate both absolute and percentage differences
    balance_compare_df['abs_diff'] = abs(
        balance_compare_df['usd_balance_train'] -
        balance_compare_df['usd_balance_model']
    )

    balance_compare_df['pct_diff'] = abs(
        balance_compare_df['usd_balance_train'] /
        balance_compare_df['usd_balance_model'] - 1
    )

    # Flag significant mismatches (both conditions must be true)
    significant_diffs = balance_compare_df[
        (balance_compare_df['abs_diff'] > 0.1) &
        (balance_compare_df['pct_diff'] > 0.00001)
    ]

    assert len(significant_diffs) == 0, \
        "Found wallet-coin pairs with significant balance mismatches (>$0.01 and >0.0001%)"


@pytest.fixture(scope='session')
def validation_datasets(period_datasets):
    """
    Integration fixture retrieving validation period and full-range data.

    Params:
    - period_datasets: Tuple from previous fixture including coin_cohort

    Returns:
    - tuple: (validation_profits_df, validation_market_df,
             full_range_profits_df, full_range_market_df)
    """
    logger.info("Generating validation and full-range datasets...")
    logger.setLevel(logging.WARNING)

    # Extract coin_cohort from period_datasets
    coin_cohort = period_datasets[-1]

    # Get validation period data
    validation_profits_df, validation_market_df, _ = wmo.retrieve_period_datasets(
        wallets_config['training_data']['validation_period_start'],
        wallets_config['training_data']['validation_period_end'],
        coin_cohort=coin_cohort
    )

    # Get full range data
    full_range_profits_df, full_range_market_df, _ = wmo.retrieve_period_datasets(
        wallets_config['training_data']['training_period_start'],
        wallets_config['training_data']['validation_period_end'],
        coin_cohort=coin_cohort
    )

    logger.setLevel(logging.INFO)
    return (validation_profits_df, validation_market_df,
            full_range_profits_df, full_range_market_df)


def test_full_range_transfers_consistency(period_datasets, validation_datasets):
    """Test that combined metrics from individual periods match full-range data.

    Params:
    - period_datasets (tuple): Contains training_df and modeling_df
    - validation_datasets (tuple): Contains validation_df and full_range_df

    Raises:
    - AssertionError: If difference exceeds 0.01 USD and 0.01% threshold
    """
    # Unpack datasets
    training_df, _, modeling_df, _, _, _, _ = period_datasets
    validation_df, _, full_range_df, _ = validation_datasets

    # Sum transfers across individual periods
    period_transfers = (
        abs(training_df['usd_net_transfers']).astype('float64').sum() +
        abs(modeling_df['usd_net_transfers']).astype('float64').sum() +
        abs(validation_df['usd_net_transfers']).astype('float64').sum()
    )

    # Compare to full range transfers
    full_range_transfers = abs(full_range_df['usd_net_transfers']).astype('float64').sum()

    # Calculate absolute and percentage differences
    transfer_diff = abs(full_range_transfers - period_transfers)
    transfer_pct = transfer_diff / full_range_transfers if full_range_transfers != 0 else 0

    # Assert both thresholds must be exceeded to fail
    assert (transfer_diff < 0.01) or (transfer_pct < 0.0001), \
        f"Transfer difference: ${transfer_diff:,.2f} ({transfer_pct:.2%})"
