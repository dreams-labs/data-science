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
        combined_profits_df, combined_market_df
    )
    logger.setLevel(logging.INFO)
    logger.info("All dev data retrieved.")

    return datasets


def test_time_period_boundaries(period_datasets):
    """Test that period boundaries align correctly"""
    training_df, _, modeling_df, _, _, _ = period_datasets
    training_last = training_df['date'].max()
    modeling_first = modeling_df['date'].min()
    assert training_last == modeling_first

def test_coin_set_consistency(period_datasets):
    """Test that coin sets match between periods"""
    training_df, _, modeling_df, _, combined_df, _ = period_datasets
    training_coins = set(training_df['coin_id'])
    modeling_coins = set(modeling_df['coin_id'])
    combined_coins = set(combined_df['coin_id'])
    assert training_coins == combined_coins
    assert len(training_coins - modeling_coins) == 0

def test_transfer_amount_consistency(period_datasets):
    """Test that transfer amounts sum correctly"""
    training_df, _, modeling_df, _, combined_df, _ = period_datasets
    training_transfers = abs(training_df['usd_net_transfers']).astype('float64').sum()
    modeling_transfers = abs(modeling_df['usd_net_transfers']).astype('float64').sum()
    combined_transfers = abs(combined_df['usd_net_transfers']).astype('float64').sum()
    assert abs(combined_transfers - (training_transfers + modeling_transfers)) < 0.01

def test_time_period_boundaries_and_balances(period_datasets):
    """
    Test that:
    1. Period boundaries align correctly
    2. Total USD balances match exactly at the boundary
    """
    training_df, _, modeling_df, _, _, _ = period_datasets

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
