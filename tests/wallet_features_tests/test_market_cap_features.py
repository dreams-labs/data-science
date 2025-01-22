"""
tests used to audit the files in the data-science/src folder
"""
# pylint: disable=C0302 # over 1000 can you deflines
# pylint: disable=C0413 # import not at top of doc (due to local import)
# pylint: disable=W0612 # unused variables (due to test reusing functions with 2 outputs)
# pylint: disable=W0621 # redefining from outer scope triggering on pytest fixtures
# pylint: disable=E0401 # can't find import (due to local import)
# pyright: reportMissingModuleSource=false

import sys
from pathlib import Path
from datetime import timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pytest
from dreams_core import core as dc

# pyright: reportMissingImports=false
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
import wallet_features.market_cap_features as wmc
from wallet_modeling.wallets_config_manager import WalletsConfig

load_dotenv()
logger = dc.setup_logger()

config_path = Path(__file__).parent.parent / 'test_config' / 'test_wallets_config.yaml'
wallets_config = WalletsConfig.load_from_yaml(config_path)

# ===================================================== #
#                                                       #
#                 U N I T   T E S T S                   #
#                                                       #
# ===================================================== #

# ------------------------------------------------- #
# calculate_average_holding_period() unit tests
# ------------------------------------------------- #

@pytest.mark.unit
def test_force_fill_market_cap_mixed_scenarios():
    """
    Tests force_fill_market_cap with different market cap patterns across coins:
    - Coin A: Complete data (control)
    - Coin B: Partial data with gaps
    - Coin C: No data
    - Coin D: Edge missing data
    """
    # Create test data with different market cap patterns
    dates = pd.date_range('2024-01-01', '2024-01-05')
    input_df = pd.DataFrame({
        'date': dates.repeat(4),
        'coin_id': ['A', 'B', 'C', 'D'] * 5,
        'market_cap_imputed': [
            # Coin A: complete data
            100, None, None, 400,  # Jan 1
            100, 200,  None, 400,  # Jan 2
            100, None, None, 400,  # Jan 3
            100, 200,  None, None, # Jan 4
            100, None, None, 400   # Jan 5
        ]
    })

    result = wmc.force_fill_market_cap(input_df)

    # Get default fill value from module config
    default_fill = wallets_config['data_cleaning']['market_cap_default_fill']

    # Calculate expected values for each coin
    expected_values = {
        'A': [100] * 5,  # Unchanged values
        'B': [200] * 5,  # Gaps filled with 200
        'C': [default_fill] * 5, # All default value
        'D': [400] * 5   # Gaps filled with 400
    }

    for coin_id, expected in expected_values.items():
        coin_data = result[result['coin_id'] == coin_id]['market_cap_filled'].values
        assert np.allclose(coin_data, expected, equal_nan=True), \
            f"Mismatch in filled values for coin {coin_id}"

    # Verify original data wasn't modified
    assert 'market_cap_imputed' in result.columns, "Original column should be preserved"


@pytest.mark.unit
def test_calculate_volume_weighted_market_cap_mixed_scenarios():
    """
    Tests volume weighted market cap calculations with different wallet scenarios:
    - Wallet 1: Normal case with varied volumes
    - Wallet 2: Zero volume (should use simple average)
    - Wallet 3: Single high volume transaction
    - Wallet 4: Equal volumes (weighted avg should match simple avg)
    """
    input_df = pd.DataFrame([
        # Wallet 1: Normal varied volumes
        {'wallet_address': 1, 'usd_net_transfers': 100, 'market_cap_filled': 1000, 'is_imputed': False},
        {'wallet_address': 1, 'usd_net_transfers': 200, 'market_cap_filled': 2000, 'is_imputed': False},
        {'wallet_address': 1, 'usd_net_transfers': -300, 'market_cap_filled': 3000, 'is_imputed': False},

        # Wallet 2: Zero volume case
        {'wallet_address': 2, 'usd_net_transfers': 0, 'market_cap_filled': 1500, 'is_imputed': True},
        {'wallet_address': 2, 'usd_net_transfers': 0, 'market_cap_filled': 2500, 'is_imputed': True},

        # Wallet 3: Single high volume
        {'wallet_address': 3, 'usd_net_transfers': -1000, 'market_cap_filled': 5000, 'is_imputed': False},

        # Wallet 4: Equal volumes
        {'wallet_address': 4, 'usd_net_transfers': 50, 'market_cap_filled': 1000, 'is_imputed': False},
        {'wallet_address': 4, 'usd_net_transfers': 50, 'market_cap_filled': 3000, 'is_imputed': False},
    ])

    result = wmc.calculate_volume_weighted_market_cap(input_df,'market_cap_filled')

    # Calculate expected values:
    # Wallet 1: (1000*100 + 2000*200 + 3000*300)/(100 + 200 + 300) = 2333.33
    # Wallet 2: Simple average of (1500 + 2500)/2 = 2000
    # Wallet 3: Single value = 5000
    # Wallet 4: Equal weights (1000 + 3000)/2 = 2000
    expected_values = {
        1: 2333.33,
        2: np.nan,
        3: 5000.00,
        4: 2000.00
    }

    for wallet, expected in expected_values.items():
        assert np.isclose(
            result.loc[wallet, 'volume_wtd_market_cap'],
            expected,
            rtol=1e-2, equal_nan=True
        ), f"Incorrect weighted market cap for wallet {wallet}"


@dataclass
class ProfitsValidator:
    """
    Validates profits DataFrame follows expected format and constraints.
    Only validates training period data.
    """
    def validate_all(self, profits_df, training_period_start, training_period_end):
        """Run all validation checks and return dict of results"""

        dates = {
            'training_starting_balance_date': pd.to_datetime(training_period_start) - timedelta(days=1),
            'training_period_start': pd.to_datetime(training_period_start),
            'training_period_end': pd.to_datetime(training_period_end),
        }

        return {
            'no_duplicates': self.check_no_duplicates(profits_df),
            'period_boundaries': self.check_period_boundaries(profits_df, dates),
            'no_negatives': self.check_no_negative_balances(profits_df),
            'date_range': self.check_date_range(profits_df, dates),
            'no_missing': self.check_no_missing_values(profits_df)
        }

    def check_no_duplicates(self, profits_df):
        """Check for duplicate records"""
        deduped_df = profits_df[['coin_id', 'wallet_address', 'date']].drop_duplicates()
        return len(profits_df) == len(deduped_df)

    def check_period_boundaries(self, profits_df, dates):
        """Check records exist at period boundaries"""
        profits_df['date'] = pd.to_datetime(profits_df['date'])
        pairs = profits_df[['coin_id', 'wallet_address']].drop_duplicates()
        n_pairs = len(pairs)

        period_df = profits_df[profits_df['date'] == dates['training_period_end']]
        period_pairs = period_df[['coin_id', 'wallet_address']].drop_duplicates()
        return len(period_pairs) == n_pairs

    def check_no_negative_balances(self, profits_df):
        """Check for negative USD balances"""
        return (profits_df['usd_balance'] >= -0.1).all()

    def check_date_range(self, profits_df, dates):
        """Verify date coverage"""
        profits_df['date'] = pd.to_datetime(profits_df['date'])
        return (profits_df['date'].min() >= dates['training_starting_balance_date'] and
                profits_df['date'].max() == dates['training_period_end'])

    def check_no_missing_values(self, profits_df):
        """Check for missing values"""
        return not profits_df.isna().any().any()



# pylint:disable=line-too-long
@pytest.fixture
def test_profits_data():
    """
    Returns raw profits data that can be remapped for many-to-many testing.
    """
    training_period_start = '2024-01-01'
    training_period_end = '2024-10-01'

    profits_data = [
        # w01_multiple_coins - btc & eth (multiple transactions, multiple coins)
        {'coin_id': 'btc', 'wallet_address': 'w01_multiple_coins', 'date': '2024-01-01', 'usd_balance': 100, 'usd_net_transfers': 100, 'is_imputed': False},
        {'coin_id': 'btc', 'wallet_address': 'w01_multiple_coins', 'date': '2024-05-01', 'usd_balance': 120, 'usd_net_transfers': 50, 'is_imputed': False},
        {'coin_id': 'btc', 'wallet_address': 'w01_multiple_coins', 'date': '2024-10-01', 'usd_balance': 180, 'usd_net_transfers': 0, 'is_imputed': True},

        {'coin_id': 'eth', 'wallet_address': 'w01_multiple_coins', 'date': '2024-01-01', 'usd_balance': 200, 'usd_net_transfers': 200, 'is_imputed': False},
        {'coin_id': 'eth', 'wallet_address': 'w01_multiple_coins', 'date': '2024-05-01', 'usd_balance': 300, 'usd_net_transfers': 50, 'is_imputed': False},
        {'coin_id': 'eth', 'wallet_address': 'w01_multiple_coins', 'date': '2024-10-01', 'usd_balance': 280, 'usd_net_transfers': 0, 'is_imputed': True},

        # w02_net_loss - btc (net loss)
        {'coin_id': 'btc', 'wallet_address': 'w02_net_loss', 'date': '2024-01-01', 'usd_balance': 300, 'usd_net_transfers': 300, 'is_imputed': False},
        {'coin_id': 'btc', 'wallet_address': 'w02_net_loss', 'date': '2024-05-01', 'usd_balance': 250, 'usd_net_transfers': -100, 'is_imputed': False},
        {'coin_id': 'btc', 'wallet_address': 'w02_net_loss', 'date': '2024-10-01', 'usd_balance': 100, 'usd_net_transfers': 0, 'is_imputed': True},

        # w03_sell_all_and_rebuy
        {'coin_id': 'eth', 'wallet_address': 'w03_sell_all_and_rebuy', 'date': '2024-01-01', 'usd_balance': 50, 'usd_net_transfers': 50, 'is_imputed': False},
        {'coin_id': 'eth', 'wallet_address': 'w03_sell_all_and_rebuy', 'date': '2024-03-01', 'usd_balance': 0,  'usd_net_transfers': -50, 'is_imputed': False},
        {'coin_id': 'eth', 'wallet_address': 'w03_sell_all_and_rebuy', 'date': '2024-08-01', 'usd_balance': 40, 'usd_net_transfers': 40, 'is_imputed': False},
        {'coin_id': 'eth', 'wallet_address': 'w03_sell_all_and_rebuy', 'date': '2024-10-01', 'usd_balance': 42, 'usd_net_transfers': 0, 'is_imputed': True},

        # w04_only_period_end - btc (only final row)
        {'coin_id': 'sol', 'wallet_address': 'w04_only_period_end', 'date': '2024-10-01', 'usd_balance': 70, 'usd_net_transfers': 70, 'is_imputed': False},

        # w04a_only_period_end_w_balance - btc
        {'coin_id': 'eth', 'wallet_address': 'w04a_only_period_end_w_balance', 'date': '2023-12-31', 'usd_balance': 30, 'usd_net_transfers': 0, 'is_imputed': True},
        {'coin_id': 'eth', 'wallet_address': 'w04a_only_period_end_w_balance', 'date': '2024-10-01', 'usd_balance': 90, 'usd_net_transfers': 50, 'is_imputed': False},

        # w04b_only_period_start_buy
        {'coin_id': 'sol', 'wallet_address': 'w04b_only_period_start_buy', 'date': '2024-01-01', 'usd_balance': 300, 'usd_net_transfers': 300, 'is_imputed': False},
        {'coin_id': 'sol', 'wallet_address': 'w04b_only_period_start_buy', 'date': '2024-10-01', 'usd_balance': 900, 'usd_net_transfers': 0, 'is_imputed': True},

        # w04c_only_period_start_buy_w_existing_balance
        {'coin_id': 'btc', 'wallet_address': 'w04c_only_period_start_buy_w_existing_balance', 'date': '2023-12-31', 'usd_balance': 40, 'usd_net_transfers': 0, 'is_imputed': True},
        {'coin_id': 'btc', 'wallet_address': 'w04c_only_period_start_buy_w_existing_balance', 'date': '2024-01-01', 'usd_balance': 350, 'usd_net_transfers': 300, 'is_imputed': False},
        {'coin_id': 'btc', 'wallet_address': 'w04c_only_period_start_buy_w_existing_balance', 'date': '2024-10-01', 'usd_balance': 1050, 'usd_net_transfers': 0, 'is_imputed': True},

        # w04d_only_period_start_sell
        {'coin_id': 'sol', 'wallet_address': 'w04d_only_period_start_sell', 'date': '2023-12-31', 'usd_balance': 200, 'usd_net_transfers': 0, 'is_imputed': True},
        {'coin_id': 'sol', 'wallet_address': 'w04d_only_period_start_sell', 'date': '2024-01-01', 'usd_balance': 0, 'usd_net_transfers': -200, 'is_imputed': False},
        {'coin_id': 'sol', 'wallet_address': 'w04d_only_period_start_sell', 'date': '2024-10-01', 'usd_balance': 0, 'usd_net_transfers': 0, 'is_imputed': True},

        # w04e_only_period_start_sell_partial
        {'coin_id': 'btc', 'wallet_address': 'w04e_only_period_start_sell_partial', 'date': '2023-12-31', 'usd_balance': 510, 'usd_net_transfers': 0, 'is_imputed': True},
        {'coin_id': 'btc', 'wallet_address': 'w04e_only_period_start_sell_partial', 'date': '2024-01-01', 'usd_balance': 500, 'usd_net_transfers': -10, 'is_imputed': False},
        {'coin_id': 'btc', 'wallet_address': 'w04e_only_period_start_sell_partial', 'date': '2024-10-01', 'usd_balance': 600, 'usd_net_transfers': 0, 'is_imputed': True},

        # w05_only_imputed - btc (only imputed rows at start and end)
        {'coin_id': 'sol', 'wallet_address': 'w05_only_imputed', 'date': '2023-12-31', 'usd_balance': 50, 'usd_net_transfers': 0, 'is_imputed': True},
        {'coin_id': 'sol', 'wallet_address': 'w05_only_imputed', 'date': '2024-10-01', 'usd_balance': 70, 'usd_net_transfers': 0, 'is_imputed': True},

        # w06_tiny_transactions - very small transactions relative to portfolio size
        {'coin_id': 'myro', 'wallet_address': 'w06_tiny_transactions', 'date': '2023-12-31', 'usd_balance': 1250, 'usd_net_transfers': 0, 'is_imputed': True},
        {'coin_id': 'myro', 'wallet_address': 'w06_tiny_transactions', 'date': '2024-02-01', 'usd_balance': 1220, 'usd_net_transfers': 1, 'is_imputed': False},
        {'coin_id': 'myro', 'wallet_address': 'w06_tiny_transactions', 'date': '2024-08-01', 'usd_balance': 0, 'usd_net_transfers': -350, 'is_imputed': False},
        {'coin_id': 'myro', 'wallet_address': 'w06_tiny_transactions', 'date': '2024-10-01', 'usd_balance': 0, 'usd_net_transfers': 0, 'is_imputed': True},

        # w07_tiny_transactions2 - very small transactions relative to portfolio size
        {'coin_id': 'floki', 'wallet_address': 'w07_tiny_transactions2', 'date': '2023-12-31', 'usd_balance': 400, 'usd_net_transfers': 0, 'is_imputed': True},
        {'coin_id': 'floki', 'wallet_address': 'w07_tiny_transactions2', 'date': '2024-02-01', 'usd_balance': 1220, 'usd_net_transfers': -20, 'is_imputed': False},
        {'coin_id': 'floki', 'wallet_address': 'w07_tiny_transactions2', 'date': '2024-08-01', 'usd_balance': 0, 'usd_net_transfers': -150, 'is_imputed': False},
        {'coin_id': 'floki', 'wallet_address': 'w07_tiny_transactions2', 'date': '2024-10-01', 'usd_balance': 0, 'usd_net_transfers': 0, 'is_imputed': True},

        # w08_offsetting_transactions - large offsetting transactions in the middle of the period
        {'coin_id': 'sol', 'wallet_address': 'w08_offsetting_transactions', 'date': '2023-12-31', 'usd_balance': 500, 'usd_net_transfers': 0, 'is_imputed': True},
        {'coin_id': 'sol', 'wallet_address': 'w08_offsetting_transactions', 'date': '2024-02-01', 'usd_balance': 10400, 'usd_net_transfers': 10000, 'is_imputed': False},
        {'coin_id': 'sol', 'wallet_address': 'w08_offsetting_transactions', 'date': '2024-02-02', 'usd_balance': 400, 'usd_net_transfers': -10000, 'is_imputed': False},
        {'coin_id': 'sol', 'wallet_address': 'w08_offsetting_transactions', 'date': '2024-10-01', 'usd_balance': 750, 'usd_net_transfers': 0, 'is_imputed': True},

        # w09_memecoin_winner - Large swings in portfolio value
        {'coin_id': 'floki', 'wallet_address': 'w09_memecoin_winner', 'date': '2024-01-01', 'usd_balance': 100, 'usd_net_transfers': 100, 'is_imputed': False},
        {'coin_id': 'floki', 'wallet_address': 'w09_memecoin_winner', 'date': '2024-03-01', 'usd_balance': 250, 'usd_net_transfers': -500, 'is_imputed': False},
        {'coin_id': 'floki', 'wallet_address': 'w09_memecoin_winner', 'date': '2024-05-01', 'usd_balance': 50, 'usd_net_transfers': -100, 'is_imputed': False},
        {'coin_id': 'floki', 'wallet_address': 'w09_memecoin_winner', 'date': '2024-10-01', 'usd_balance': 10, 'usd_net_transfers': 0, 'is_imputed': True},

        # w10_memecoin_loser - Large swings in portfolio value
        {'coin_id': 'myro', 'wallet_address': 'w10_memecoin_loser', 'date': '2024-03-01', 'usd_balance': 250, 'usd_net_transfers': 250, 'is_imputed': False},
        {'coin_id': 'myro', 'wallet_address': 'w10_memecoin_loser', 'date': '2024-10-01', 'usd_balance': 0, 'usd_net_transfers': -20, 'is_imputed': False},

        # w11_sells_early
        {'coin_id': 'btc', 'wallet_address': 'w11_sells_early', 'date': '2024-03-01', 'usd_balance': 0, 'usd_net_transfers': 0, 'is_imputed': True},
        {'coin_id': 'btc', 'wallet_address': 'w11_sells_early', 'date': '2024-04-01', 'usd_balance': 250, 'usd_net_transfers': 250, 'is_imputed': False},
        {'coin_id': 'btc', 'wallet_address': 'w11_sells_early', 'date': '2024-5-01', 'usd_balance': 0, 'usd_net_transfers': -300, 'is_imputed': False},
        {'coin_id': 'btc', 'wallet_address': 'w11_sells_early', 'date': '2024-10-01', 'usd_balance': 0, 'usd_net_transfers': 0, 'is_imputed': True},

        # w12_buys_late
        {'coin_id': 'sol', 'wallet_address': 'w12_buys_late', 'date': '2024-03-01', 'usd_balance': 0, 'usd_net_transfers': 0, 'is_imputed': True},
        {'coin_id': 'sol', 'wallet_address': 'w12_buys_late', 'date': '2024-09-01', 'usd_balance': 500, 'usd_net_transfers': 250, 'is_imputed': False},
        {'coin_id': 'sol', 'wallet_address': 'w12_buys_late', 'date': '2024-10-01', 'usd_balance': 550, 'usd_net_transfers': 0, 'is_imputed': True},
    ]

    return pd.DataFrame(profits_data), training_period_start, training_period_end



@pytest.fixture
def test_profits_df(test_profits_data):
    """
    Returns test profits DataFrame with cash flow transfers added.
    """
    profits_df, training_period_start, training_period_end = test_profits_data
    profits_df = profits_df.copy()

    # Validate test data format before proceeding
    validator = ProfitsValidator()
    validation_results = validator.validate_all(
        profits_df.copy(),
        training_period_start,
        training_period_end
    )
    assert all(validation_results.values()), "Test data failed validation checks."

    # Remove rows with a rounded 0 balance and 0 transfers which happens in wmo.retrieve_datasets() once validation checks are passed
    profits_df = profits_df[
        ~((profits_df['usd_balance'] == 0) &
        (profits_df['usd_net_transfers'] == 0))
    ]
    logger.info(profits_df.columns)

    return profits_df


@pytest.fixture
def test_market_cap_data():
    """
    Returns synthetic market cap data for all coin-date pairs.
    """
    market_data = [
        {'coin_id': 'btc', 'date': '2023-12-31', 'market_cap_imputed': 200000000},
        {'coin_id': 'btc', 'date': '2024-01-01', 'market_cap_imputed': 210000000},
        {'coin_id': 'btc', 'date': '2024-03-01', 'market_cap_imputed': 220000000},
        {'coin_id': 'btc', 'date': '2024-04-01', 'market_cap_imputed': 230000000},
        {'coin_id': 'btc', 'date': '2024-05-01', 'market_cap_imputed': 240000000},
        {'coin_id': 'btc', 'date': '2024-10-01', 'market_cap_imputed': 250000000},
        {'coin_id': 'btc', 'date': '2024-5-01', 'market_cap_imputed': 245000000},

        {'coin_id': 'eth', 'date': '2023-12-31', 'market_cap_imputed': 100000000},
        {'coin_id': 'eth', 'date': '2024-01-01', 'market_cap_imputed': 105000000},
        {'coin_id': 'eth', 'date': '2024-03-01', 'market_cap_imputed': 110000000},
        {'coin_id': 'eth', 'date': '2024-05-01', 'market_cap_imputed': 115000000},
        {'coin_id': 'eth', 'date': '2024-08-01', 'market_cap_imputed': 120000000},
        {'coin_id': 'eth', 'date': '2024-10-01', 'market_cap_imputed': 125000000},

        {'coin_id': 'floki', 'date': '2023-12-31', 'market_cap_imputed': 500000},
        {'coin_id': 'floki', 'date': '2024-01-01', 'market_cap_imputed': 600000},
        {'coin_id': 'floki', 'date': '2024-02-01', 'market_cap_imputed': 800000},
        {'coin_id': 'floki', 'date': '2024-03-01', 'market_cap_imputed': 700000},
        {'coin_id': 'floki', 'date': '2024-05-01', 'market_cap_imputed': 1200000},
        {'coin_id': 'floki', 'date': '2024-08-01', 'market_cap_imputed': 2700000},
        {'coin_id': 'floki', 'date': '2024-10-01', 'market_cap_imputed': 1500000},

        {'coin_id': 'myro', 'date': '2023-12-31', 'market_cap_imputed': 1500000},
        {'coin_id': 'myro', 'date': '2024-02-01', 'market_cap_imputed': 1550000},
        {'coin_id': 'myro', 'date': '2024-03-01', 'market_cap_imputed': 1600000},
        {'coin_id': 'myro', 'date': '2024-08-01', 'market_cap_imputed': 900000},
        {'coin_id': 'myro', 'date': '2024-10-01', 'market_cap_imputed': 250000},

        {'coin_id': 'sol', 'date': '2023-12-31', 'market_cap_imputed': 50000000},
        {'coin_id': 'sol', 'date': '2024-01-01', 'market_cap_imputed': 52000000},
        {'coin_id': 'sol', 'date': '2024-02-01', 'market_cap_imputed': 54000000},
        {'coin_id': 'sol', 'date': '2024-02-02', 'market_cap_imputed': 60000000},
        {'coin_id': 'sol', 'date': '2024-03-01', 'market_cap_imputed': 50000000},
        {'coin_id': 'sol', 'date': '2024-09-01', 'market_cap_imputed': 70000000},
        {'coin_id': 'sol', 'date': '2024-10-01', 'market_cap_imputed': 65000000},
    ]

    market_cap_df = pd.DataFrame(market_data)

    return market_cap_df


@pytest.fixture
def test_market_cap_features_df(test_profits_df, test_market_cap_data):
    """
    Returns the full market_cap_features_df for testing
    """
    market_cap_features_df = wmc.calculate_market_cap_features(test_profits_df,
                                                               test_market_cap_data)

    return market_cap_features_df



def test_calculate_volume_weighted_market_cap(test_market_cap_features_df):
    """
    Test the calculate_volume_weighted_market_cap function to ensure
    volume-weighted averages are calculated correctly across all coins for a wallet.
    """
    market_cap_features_df = test_market_cap_features_df.copy()
    expected_results = {
        'w01_multiple_coins': {
            # (210M*100 + 240M*50 + 105M*200 + 115M*50) / (100+50+200+50) = 143.75M
            'volume_wtd_market_cap': (210*100 + 240*50 + 105*200 + 115*50) / (100+50+200+50) * 1e6,
        },
        'w02_net_loss': {
            # btc: (210M*300 + 240M*100) / (300+100) = 217.5M
            'volume_wtd_market_cap': 217500000,
        },
        'w03_sell_all_and_rebuy': {
            # eth: (105M*50 + 110M*50 + 120M*40) / (50+50+40) = 111M
            'volume_wtd_market_cap': 111000000,
        },
        'w04_only_period_end': {
            # sol: No volume => use simple average = 65M
            'volume_wtd_market_cap': 65000000,
        },
        'w08_offsetting_transactions': {
            # sol: (54M*10,000 + 60M*10,000) / (10,000 + 10,000) = 57M
            'volume_wtd_market_cap': 57000000,
        },
        'w09_memecoin_winner': {
            # floki: (600k*100 + 700k*500 + 1.2M*100) / (100+500+100) = 757.143k
            'volume_wtd_market_cap': 757143,
        },
        'w10_memecoin_loser': {
            # myro: (1.6M*250) / 250 = 1.6M
            'volume_wtd_market_cap': 1600000,
        },
    }

    # Validate results
    for wallet, expected in expected_results.items():
        wallet_result = market_cap_features_df.loc[wallet, 'volume_wtd_market_cap']
        assert np.isclose(wallet_result, expected['volume_wtd_market_cap'], atol=1e5), \
            f"Wallet {wallet} has incorrect volume-weighted market cap: {wallet_result}, " \
            "expected: {expected['volume_wtd_market_cap']}."



def test_calculate_portfolio_weighted_market_cap(test_profits_df,test_market_cap_data,
                                                 test_market_cap_features_df):
    """
    Test the calculate_portfolio_weighted_market_cap function to ensure
    portfolio-weighted averages are calculated correctly across all coins for a wallet.
    """
    market_cap_features_df = test_market_cap_features_df.copy()

    # Merge profits data with market cap data
    profits_df = test_profits_df
    market_cap_df = test_market_cap_data
    merged_df = profits_df.merge(
        market_cap_df,
        on=['coin_id', 'date'],
        how='left'
    )

    # Add portfolio weights (assuming usd_balance as a proxy for portfolio size)
    merged_df['portfolio_weight'] = merged_df['usd_balance']

    # Updated expected values for validation
    expected_results = {
        'w01_multiple_coins': {
            # Portfolio-weighted
            'end_portfolio_wtd_market_cap': (180*250000000 + 280*125000000) / (180 + 280)
        },
        'w02_net_loss': {
            'end_portfolio_wtd_market_cap': 250000000,
        },
        'w03_sell_all_and_rebuy': {
            'end_portfolio_wtd_market_cap': 125000000
        },
        'w04_only_period_end': {
            'end_portfolio_wtd_market_cap': 65000000,
        },
        'w07_tiny_transactions2': {
            'end_portfolio_wtd_market_cap': np.nan
        },
    }

    # Validate results
    for wallet, expected in expected_results.items():
        wallet_result = market_cap_features_df.loc[wallet, 'end_portfolio_wtd_market_cap']
        assert np.isclose(wallet_result, expected['end_portfolio_wtd_market_cap'], atol=1e5, equal_nan=True), \
            f"Wallet {wallet} has incorrect portfolio-weighted market cap: {wallet_result}, " \
            f"expected: {expected['end_portfolio_wtd_market_cap']}."
