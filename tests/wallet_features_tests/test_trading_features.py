# pylint: disable=wrong-import-position  # import not at top of doc (due to local import)
# pylint: disable=redefined-outer-name  # redefining from outer scope triggering on pytest fixtures
# pylint: disable=unused-argument
# pyright: reportMissingModuleSource=false

import sys
from datetime import timedelta
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pytest
from dreams_core import core as dc

# pyright: reportMissingImports=false
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
import wallet_features.trading_features as wtf
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


# ------------------------------------------------ #
# add_cash_flow_transfers_logic() unit tests +
# calculate_wallet_trading_features() unit tests
# ------------------------------------------------ #

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
        profits_df,
        training_period_start,
        training_period_end
    )
    assert all(validation_results.values()), "Test data failed validation checks."

    # Remove rows with a rounded 0 balance and 0 transfers which happens in wmo.retrieve_datasets() once validation checks are passed
    profits_df = profits_df[
        ~((profits_df['usd_balance'] == 0) &
        (profits_df['usd_net_transfers'] == 0))
    ]

    return profits_df, training_period_start, training_period_end


@pytest.fixture
def test_trading_features_df(test_profits_df):
    """
    Returns trading features computed from test profits data.
    """
    # Unpack tuple
    test_profits_df, training_period_start, training_period_end = test_profits_df

    # Compute trading features
    wallet_trading_features_df = wtf.calculate_wallet_trading_features(test_profits_df,
                                                                       training_period_start,
                                                                       training_period_end)

    # Return dates for recalculation
    return wallet_trading_features_df


# Initial Tests for Individual Wallet Edge Cases
# -----------------------------------------------------

@pytest.mark.unit
def test_w01_multiple_coins(test_trading_features_df):
    """
    Tests wallet trading features for a wallet with multiple coins:
    - BTC: 100 initial buy, 50 additional buy, ending at 180
    - ETH: 200 initial buy, 50 additional buy, ending at 280
    """

    # Filter to just the multiple coins test wallet
    wallet = 'w01_multiple_coins'
    wallet_features = test_trading_features_df.loc[wallet]

    # Calculate expected values
    # total_crypto_buys: btc(100 + 50) + eth(200 + 50) = 400
    expected_buys = 400

    # total_crypto_sells: no sells = 0
    expected_sells = 0

    # net_crypto_investment: 400 - 0 = 400
    expected_net = 400

    # current_gain: (180 + 280) - 400 = 60
    expected_gain = 60

    # transaction_days: 2 days (1/1 and 5/1)
    expected_txn_days = 2

    # unique_coins_traded: BTC and ETH
    expected_coins = 2

    # total_volume: abs(100) + abs(50) + abs(200) + abs(50) = 400
    expected_volume = 400

    # average_transaction: 400 / 4 = 100
    expected_avg_txn = 100

    # time_weighted_balance:
    # Initial period (12/31-5/1): 300 * 120 days = 36000
    # Final period (5/1-10/1): 400 * 154 days = 61600
    # Total = 97600 / 274 days ≈ 356.20
    expected_twb = 356.20

    # activity_density: 2 transaction days / 274 total days ≈ 0.0073
    expected_density = 2 / 274

    # volume_vs_twb_ratio: 400 / 356.20 ≈ 1.123
    expected_ratio = 400 / 356.20

    # Assert all values match
    assert wallet_features['total_crypto_buys'] == expected_buys
    assert wallet_features['total_crypto_sells'] == expected_sells
    assert wallet_features['net_crypto_investment'] == expected_net
    assert wallet_features['current_gain'] == expected_gain
    assert wallet_features['transaction_days'] == expected_txn_days
    assert wallet_features['unique_coins_traded'] == expected_coins
    assert wallet_features['total_volume'] == expected_volume
    assert wallet_features['average_transaction'] == expected_avg_txn
    assert np.isclose(wallet_features['time_weighted_balance'], expected_twb, rtol=1e-2)
    assert np.isclose(wallet_features['activity_density'], expected_density, rtol=1e-2)
    assert np.isclose(wallet_features['volume_vs_twb_ratio'], expected_ratio, rtol=1e-2)



@pytest.mark.unit
def test_w08_offsetting_transactions(test_trading_features_df):
    """
    Tests wallet trading features for a wallet with offsetting transactions:
    - 500 starting balance on 2023-12-31
    - 10000 buy on 2024-02-01
    - 10000 sell on 2024-02-02
    - 750 ending balance on 2024-10-01
    """
    # Filter to just the test case wallet
    wallet = 'w08_offsetting_transactions'
    wallet_features = test_trading_features_df.loc[wallet]

    # Calculate expected values
    # total_crypto_buys: initial 500 + 10000 buy = 10500
    expected_buys = 10500

    # total_crypto_sells: 10000 sell = 10000
    expected_sells = 10000

    # net_crypto_investment: 10500 - 10000 = 500
    expected_net = 500

    # current_gain: 750 ending - 500 cost basis = 250
    expected_gain = 250

    # transaction_days: 2 days with non-imputed transactions (2/1 and 2/2)
    expected_txn_days = 2

    # unique_coins_traded: only traded SOL
    expected_coins = 1

    # total_volume: abs(10000) + abs(-10000) = 20000
    expected_volume = 20000

    # average_transaction: 20000 / 2 = 10000
    expected_avg_txn = 10000

    # time_weighted_balance:
    # $500 * 32 days (1/1-2/1) = 16000
    # $10500 * 1 day (2/1-2/2) = 10500
    # New cost basis: $403.85 = 10500 * (1 - (10000/10400))
    # Final period duration: 242 days (2/2-9/30, because 10/1 closing balance has no hold time)
    # $403.85 * 242 = 97730.77

    # ((500 * 32) + (10500 * 1) + (242 * (10500 * (1 - (10000/10400))))) / (32 + 1 + 242)
    expected_twb = 451.748

    # activity_density: 2 transaction days / 274 total days ≈ 0.0073
    expected_density = 2 / 274

    # volume_vs_twb_ratio: 20000 / 536.23 ≈ 44.27
    expected_ratio = 20000 / 451.74

    # Assert all values match
    assert wallet_features['total_crypto_buys'] == expected_buys
    assert wallet_features['total_crypto_sells'] == expected_sells
    assert wallet_features['net_crypto_investment'] == expected_net
    assert wallet_features['current_gain'] == expected_gain
    assert wallet_features['transaction_days'] == expected_txn_days
    assert wallet_features['unique_coins_traded'] == expected_coins
    assert wallet_features['total_volume'] == expected_volume
    assert wallet_features['average_transaction'] == expected_avg_txn
    assert np.isclose(wallet_features['time_weighted_balance'], expected_twb, rtol=1e-2)
    assert np.isclose(wallet_features['activity_density'], expected_density, rtol=1e-2)
    assert np.isclose(wallet_features['volume_vs_twb_ratio'], expected_ratio, rtol=1e-2)



@pytest.mark.unit
def test_w09_memecoin_winner(test_trading_features_df):
    """
    Tests wallet trading features for a wallet with large value swings:
    - 100 buy on 2024-01-01
    - 500 sell at 250 balance on 2024-03-01
    - 100 sell at 50 balance on 2024-05-01
    - 10 ending balance on 2024-10-01
    """
    # Filter to test case wallet
    wallet = 'w09_memecoin_winner'
    wallet_features = test_trading_features_df.loc[wallet]

    # Calculate expected values
    # total_crypto_buys: initial 100 buy
    expected_buys = 100

    # total_crypto_sells: 500 + 100 = 600 in sells
    expected_sells = 600

    # net_crypto_investment: 100 - 600 = -500
    expected_net = -500

    # current_gain: 10 ending - (-500 net investment) = 510
    expected_gain = 510

    # transaction_days: 2 days with non-imputed transactions (3/1 and 5/1)
    expected_txn_days = 3

    # unique_coins_traded: only traded floki
    expected_coins = 1

    # total_volume: abs(500) + abs(100) + 100 = 700
    expected_volume = 700  # |100| + |500| + |100| = 700

    # average_transaction: 600 / 2 = 300
    expected_avg_txn = 700/3  # 233.33


    # time_weighted_balance:
    # 100 * 60 days (1/1-3/1) = 6000
    # 33.333 * 61 days (3/1-5/1) = 2033.333
    # 11.111 * 153 days (5/1-10/1) = 1700
    # ((100 * 59) + (250 * 61) + (50 * 153)) / (59 + 61 + 153)
    expected_twb = ((100 * 60) + (33.3333 * 61) + (11.1111 * 153)) / (59 + 61 + 153) # 35.65

    expected_density = 3 / 274

    expected_ratio = 700 / 35.65322

    # Assert all values match
    assert wallet_features['total_crypto_buys'] == expected_buys
    assert wallet_features['total_crypto_sells'] == expected_sells
    assert wallet_features['net_crypto_investment'] == expected_net
    assert wallet_features['current_gain'] == expected_gain
    assert wallet_features['transaction_days'] == expected_txn_days
    assert wallet_features['unique_coins_traded'] == expected_coins
    assert wallet_features['total_volume'] == expected_volume
    assert wallet_features['average_transaction'] == expected_avg_txn
    assert np.isclose(wallet_features['time_weighted_balance'], expected_twb, rtol=1e-2)
    assert np.isclose(wallet_features['activity_density'], expected_density, rtol=1e-2)
    assert np.isclose(wallet_features['volume_vs_twb_ratio'], expected_ratio, rtol=1e-2)


# ===== Volume Tests =====
@pytest.mark.unit
def test_positive_balance_columns(test_trading_features_df):
    """Verify all volume and balance metrics are non-negative"""
    volume_cols = ['total_volume', 'average_transaction', 'time_weighted_balance']
    for col in volume_cols:
        assert (test_trading_features_df[col] >= 0).all(), f"Found negative values in {col}"

@pytest.mark.unit
def test_volume_matches_transfers(test_profits_df, test_trading_features_df):
    """Verify total_volume matches sum of absolute transfers for non-imputed transactions"""
    profits_df, _, _ = test_profits_df
    expected = (profits_df[~profits_df['is_imputed']]
               .groupby('wallet_address')['usd_net_transfers']
               .agg(lambda x: abs(x).sum()))
    actual = test_trading_features_df['total_volume'].reindex(expected.index)
    assert np.allclose(actual, expected)

@pytest.mark.unit
def test_activity_density_range(test_trading_features_df):
    """Verify activity_density is between 0 and 1 since it's a ratio of days"""
    assert (test_trading_features_df['activity_density'] >= 0).all()
    assert (test_trading_features_df['activity_density'] <= 1).all()

@pytest.mark.unit
def test_transaction_days_vs_unique_coins(test_trading_features_df):
    """Verify transaction_days >= unique_coins_traded
    Since each unique coin must have at least one transaction day"""
    assert (
        test_trading_features_df['transaction_days'] >=
        test_trading_features_df['unique_coins_traded']
    ).all()

@pytest.mark.unit
def test_volume_ratio_consistency(test_trading_features_df):
    """Verify volume_vs_twb_ratio equals total_volume / time_weighted_balance"""
    expected = np.where(
        test_trading_features_df['time_weighted_balance'] > 0,
        test_trading_features_df['total_volume'] / test_trading_features_df['time_weighted_balance'],
        0
    )
    actual = test_trading_features_df['volume_vs_twb_ratio']
    assert np.allclose(actual, expected, rtol=1e-2)

@pytest.mark.unit
def test_average_transaction_bounds(test_trading_features_df):
    """Verify average_transaction is between min and max volume for each wallet
    Since it's an average of all transactions"""
    assert (
        test_trading_features_df['average_transaction'] <=
        test_trading_features_df['total_volume']
    ).all()




# Remapping Tests for Many to Many Coin-Wallet Relationships
# ----------------------------------------------------------
@pytest.fixture
def test_remapped_profits_df(test_profits_data):
    """
    Remaps the base profits data so many wallets hold many of the same coins and adds cash flow transfers.
    """
    # Reassign wallets to create a lot of overlap
    reassign_dict = {
        'w01_multiple_coins': 'w1',
        'w02_net_loss': 'w2',
        'w03_sell_all_and_rebuy': 'w2',
        'w04_only_period_end': 'w3',
        'w04a_only_period_end_w_balance': 'w3',
        'w04b_only_period_start_buy': 'w2',
        'w04c_only_period_start_buy_w_existing_balance': 'w4',
        'w04d_only_period_start_sell': 'w4',
        'w04e_only_period_start_sell_partial': 'w5',
        'w05_only_imputed': 'w5',
        'w06_tiny_transactions': 'w5',
        'w07_tiny_transactions2': 'w2',
        'w08_offsetting_transactions': 'w1',
        'w09_memecoin_winner': 'w3',
        'w10_memecoin_loser': 'w4',
        'w11_sells_early': 'w6',
        'w12_buys_late': 'w6'
    }
    profits_df, training_period_start, training_period_end = test_profits_data
    remapped_profits_df = profits_df.copy()
    remapped_profits_df['wallet_address_original'] = remapped_profits_df['wallet_address']
    remapped_profits_df['wallet_address'] = remapped_profits_df['wallet_address'].map(reassign_dict)

    # Rest of the sequence remains unchanged
    profits_df = remapped_profits_df.copy()
    training_period_start = '2024-01-01'
    training_period_end = '2024-10-01'

    # Validate test data format before proceeding
    validator = ProfitsValidator()
    validation_results = validator.validate_all(
        profits_df,
        training_period_start,
        training_period_end
    )
    assert all(validation_results.values()), "Test data failed validation checks."

    # Remove rows with a rounded 0 balance and 0 transfers which happens in wmo.retrieve_datasets() once validation checks are passed
    profits_df = profits_df[
        ~((profits_df['usd_balance'] == 0) &
        (profits_df['usd_net_transfers'] == 0))
    ]

    # Confirm that all the addresses have been mapped
    expected_addresses = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6']
    assert sorted(list(profits_df['wallet_address'].unique())) == expected_addresses

    return profits_df, training_period_start, training_period_end

@pytest.fixture
def test_remapped_trading_features_df(test_remapped_profits_df):
    """
    Returns trading features computed from test profits data.
    """

    # Unpack tuple
    remapped_profits_df, training_period_start, training_period_end = test_remapped_profits_df

    # Compute trading features
    remapped_wallet_trading_features_df = wtf.calculate_wallet_trading_features(remapped_profits_df,
                                                                       training_period_start,
                                                                       training_period_end)

    # Confirm that all the addresses have been mapped
    expected_addresses = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6']
    assert sorted(list(remapped_wallet_trading_features_df.index.values)) == expected_addresses

    return remapped_wallet_trading_features_df


@pytest.mark.unit
def test_reassigned_cash_flows_match(test_remapped_profits_df,test_profits_df):
    """
    Checks if any cash flows outputs have changed after the remapping was applied to create a lot of
    many to many relationships between wallets and coins.
    """
    # Retrieve the original columns from the remapped profits_df
    demapped_profits_df = test_remapped_profits_df[['coin_id','wallet_address_original','date','cash_flow_transfers']].copy()
    demapped_profits_df = demapped_profits_df.rename(columns={'wallet_address_original': 'wallet_address'})

    # Merge the original version with the demapped version and
    merged_df = demapped_profits_df.merge(
        test_profits_df[['coin_id', 'wallet_address', 'date', 'cash_flow_transfers']],
        on=['coin_id', 'wallet_address', 'date'],
        suffixes=('_demapped', '_cash_flow')
    )

    assert np.allclose(merged_df['cash_flow_transfers_demapped'],merged_df['cash_flow_transfers_cash_flow'])


@pytest.mark.unit
def test_volume_aggregation_after_remapping(test_profits_df, test_trading_features_df,
                                        test_remapped_profits_df, test_remapped_trading_features_df):
    """
    Verifies that total volume metrics are preserved when wallets are consolidated through remapping.

    Tests:
    1. Total volume sums match when grouping original wallets to their remapped versions
    2. Volumes are properly consolidated when multiple original wallets map to the same new wallet

    Example:
    If w1, w2 -> new_w1 and w3, w4 -> new_w2, then:
    - sum(volume of w1, w2) should equal volume of new_w1
    - sum(volume of w3, w4) should equal volume of new_w2
    """
    # Create mapping from original to new wallets
    wallet_mapping = (test_remapped_profits_df[['wallet_address', 'wallet_address_original']]
                        .drop_duplicates()
                        .set_index('wallet_address_original')['wallet_address'])

    # Join original trading features with mapping
    original_volumes = pd.DataFrame({'total_volume': test_trading_features_df['total_volume']})
    original_volumes['new_wallet'] = original_volumes.index.map(wallet_mapping)

    # Calculate expected volumes by grouping by new wallet address
    expected_volumes = original_volumes.groupby('new_wallet')['total_volume'].sum()

    # Get actual volumes from remapped features
    actual_volumes = test_remapped_trading_features_df['total_volume']

    # Sort both Series to ensure alignment
    expected_volumes = expected_volumes.sort_index()
    actual_volumes = actual_volumes.sort_index()

    assert np.allclose(expected_volumes, actual_volumes, equal_nan=True), \
        "Total volumes don't match after remapping"


@pytest.mark.unit
def test_cash_flow_aggregation_after_remapping(test_profits_df, test_trading_features_df,
                                             test_remapped_profits_df, test_remapped_trading_features_df):
    """
    Verifies that all flow metrics are preserved when wallets are consolidated through remapping.

    Tests both observed cash flows and total flows (including imputed):
    1. Observed cash flows (real transfers only):
        - cash_buy_inflows (buys)
        - cash_sell_outflows (sells)
        - cash_net_flows (net position)
    2. Total flows (including imputed rows):
        - total_inflows (all positive flows)
        - total_outflows (all negative flows)
        - total_net_flows (net including imputed)

    Example calculation:
    If w1 and w2 are merged into new_w1:
    w1 (with imputed end balance):
    - buys=$100, sells=$30, final_balance=$90
    - cash_flows: buys=$100, sells=$30, net=$70
    - total_flows: inflows=$100, outflows=$120 ($30 sells + $90 end balance), net=-$20

    w2 (with imputed end balance):
    - buys=$50, sells=$20, final_balance=$40
    - cash_flows: buys=$50, sells=$20, net=$30
    - total_flows: inflows=$50, outflows=$60 ($20 sells + $40 end balance), net=-$10

    Then new_w1 should have:
    - cash_flows: buys=$150, sells=$50, net=$100
    - total_flows: inflows=$150, outflows=$180, net=-$30
    """
    # Create mapping from original to new wallets
    wallet_mapping = (test_remapped_profits_df[['wallet_address', 'wallet_address_original']]
                     .drop_duplicates()
                     .set_index('wallet_address_original')['wallet_address'])

    # Create DataFrame with all flow metrics for original wallets
    flow_cols = ['cash_buy_inflows', 'cash_sell_outflows', 'cash_net_flows',
                 'total_inflows', 'total_outflows', 'total_net_flows']
    original_flows = pd.DataFrame(test_trading_features_df[flow_cols])
    original_flows['new_wallet'] = original_flows.index.map(wallet_mapping)

    # Calculate expected values by grouping by new wallet address
    expected_flows = original_flows.groupby('new_wallet')[flow_cols].agg({
        'cash_buy_inflows': 'sum',  # Observed buys
        'cash_sell_outflows': 'sum',  # Observed sells
        'cash_net_flows': 'sum',  # Net of observed flows
        'total_inflows': 'sum',  # All positive flows including imputed
        'total_outflows': 'sum',  # All negative flows including imputed
        'total_net_flows': 'sum',  # Net of all flows including imputed
    }).sort_index()

    # Get actual values from remapped features
    actual_flows = test_remapped_trading_features_df[flow_cols].sort_index()

    # Compare each metric
    for col in flow_cols:
        assert np.allclose(expected_flows[col], actual_flows[col], equal_nan=True), \
            f"Remapped {col} doesn't match expected values"


@pytest.mark.unit
def test_max_investment_after_remapping(test_profits_df, test_trading_features_df,
                                      test_remapped_profits_df, test_remapped_trading_features_df):
    """
    Verifies max_investment is calculated correctly after wallet remapping by:
    1. First aggregating cash flows by date for each remapped wallet
    2. Then finding the peak cumulative investment

    Example:
    Original wallets w1 and w2 map to new_w1:
    w1: coin1 peaks at $100 on Jan 1, coin2 peaks at $50 on Feb 1
    w2: coin3 peaks at $80 on Jan 1
    new_w1 max_investment should be $180 (combined Jan 1 peak) not $230 (sum of individual peaks)
    """
    # Create mapping from original to new wallets
    wallet_mapping = (test_remapped_profits_df[['wallet_address', 'wallet_address_original']]
                     .drop_duplicates()
                     .set_index('wallet_address_original')['wallet_address'])

    # Group cash flows by date and new wallet
    orig_flows = test_profits_df.copy()
    orig_flows['new_wallet'] = orig_flows['wallet_address'].map(wallet_mapping)

    # Step 1: Sum all cash flows for each wallet-date combination
    daily_flows = orig_flows.groupby(['new_wallet', 'date'])['cash_flow_transfers'].sum()

    # Step 2: Reset index to allow cumsum operation by wallet
    daily_flows = daily_flows.reset_index()

    # Step 3: Sort by wallet and date to ensure correct cumsum
    daily_flows = daily_flows.sort_values(['new_wallet', 'date'])

    # Step 4: Calculate running total of flows for each wallet
    daily_flows['cumulative_flows'] = (daily_flows.groupby('new_wallet')['cash_flow_transfers']
                                     .cumsum())

    # Step 5: Find the maximum cumulative investment for each wallet
    expected_max_inv = (daily_flows.groupby('new_wallet')['cumulative_flows']
                       .max()
                       .sort_index())

    # Get actual max investments
    actual_max_inv = test_remapped_trading_features_df['max_investment'].sort_index()

    assert np.allclose(expected_max_inv, actual_max_inv, equal_nan=True), \
        "Max investment doesn't match when calculated from remapped cash flows"


@pytest.mark.unit
def test_activity_metrics_after_remapping(test_profits_df, test_trading_features_df,
                                        test_remapped_profits_df, test_remapped_trading_features_df):
    """
    Verifies activity metrics are calculated correctly after wallet remapping.

    Tests:
    1. transaction_days: Number of unique days with non-imputed transactions
    2. unique_coins_traded: Count of unique coins across merged wallets
    3. activity_density: transaction_days / total_period_days

    Example:
    If w1 and w2 map to new_w1:
    w1: trades btc on Jan 1, eth on Jan 1 and Mar 1
    w2: trades eth on Jan 1, sol on Feb 1
    Then new_w1 should have:
    - transaction_days = 3 (Jan 1, Feb 1, Mar 1)
    - unique_coins = 3 (btc, eth, sol)
    - activity_density = 3 / period_days
    """
    # Create mapping from original to new wallets
    wallet_mapping = (test_remapped_profits_df[['wallet_address', 'wallet_address_original']]
                     .drop_duplicates()
                     .set_index('wallet_address_original')['wallet_address'])

    # Calculate transaction days
    active_dates = (test_profits_df[~test_profits_df['is_imputed']]
                   .assign(new_wallet=lambda x: x['wallet_address'].map(wallet_mapping))
                   .groupby('new_wallet')['date']
                   .nunique())

    # Calculate unique coins traded
    unique_coins = (test_profits_df[~test_profits_df['is_imputed']]
                   .assign(new_wallet=lambda x: x['wallet_address'].map(wallet_mapping))
                   .groupby('new_wallet')['coin_id']
                   .nunique())

    # Calculate activity density
    period_days = (test_profits_df['date'].max() - test_profits_df['date'].min()).days + 1
    expected_density = active_dates / period_days

    # Get actual metrics
    actual_activity = test_remapped_trading_features_df[['transaction_days', 'unique_coins_traded', 'activity_density']]

    # Compare metrics
    assert np.allclose(active_dates, actual_activity['transaction_days'], equal_nan=True), \
        "Transaction days don't match after remapping"
    assert np.allclose(unique_coins, actual_activity['unique_coins_traded'], equal_nan=True), \
        "Unique coins traded don't match after remapping"
    assert np.allclose(expected_density, actual_activity['activity_density'], equal_nan=True), \
        "Activity density doesn't match after remapping"


@pytest.mark.unit
def test_ratio_metrics_after_remapping(test_profits_df, test_trading_features_df,
                                     test_remapped_profits_df, test_remapped_trading_features_df):
    """
    Verifies ratio-based metrics are calculated correctly after wallet remapping.

    Tests:
    1. average_transaction = total_volume / number of transactions
    2. volume_vs_investment_ratio = total_volume / max_investment

    Example:
    If w1 and w2 map to new_w1:
    w1: volume=$300 (3 trades), max_inv=$200
    w2: volume=$100 (2 trades), max_inv=$100
    Then new_w1 should have:
    - average_transaction = $400/5 = $80
    - volume_vs_investment_ratio = $400/$300 = 1.33
    """
    # Create mapping from original to new wallets
    wallet_mapping = (test_remapped_profits_df[['wallet_address', 'wallet_address_original']]
                     .drop_duplicates()
                     .set_index('wallet_address_original')['wallet_address'])

    # Count transactions (non-imputed rows with non-zero transfers)
    transaction_counts = (test_profits_df[
        (~test_profits_df['is_imputed']) &
        (test_profits_df['usd_net_transfers'] != 0)
    ]
    .assign(new_wallet=lambda x: x['wallet_address'].map(wallet_mapping))
    .groupby('new_wallet')
    .size())

    # Calculate expected average transaction using total_volume from features
    expected_avg_transaction = (test_remapped_trading_features_df['total_volume'] /
                              transaction_counts)

    # Calculate expected volume vs investment ratio using features
    expected_vol_inv_ratio = np.where(
        test_remapped_trading_features_df['max_investment'] > 0,
        test_remapped_trading_features_df['total_volume'] /
        test_remapped_trading_features_df['max_investment'],
        0
    )

    # Compare metrics
    assert np.allclose(expected_avg_transaction,
                      test_remapped_trading_features_df['average_transaction'],
                      equal_nan=True), "Average transaction doesn't match after remapping"
    assert np.allclose(expected_vol_inv_ratio,
                      test_remapped_trading_features_df['volume_vs_investment_ratio'],
                      equal_nan=True), "Volume vs investment ratio doesn't match after remapping"
