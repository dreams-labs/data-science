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

    test_profits_df = pd.DataFrame(profits_data)

    # Create usd_inflows column
    test_profits_df['usd_inflows'] = test_profits_df['usd_net_transfers'].where(
        (test_profits_df['usd_net_transfers'] > 0) &
        (~test_profits_df['is_imputed']),
        0
    )

    return test_profits_df, training_period_start, training_period_end




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

    # Remove rows with a rounded 0 balance and 0 transfers which happens in wtdo.retrieve_datasets() once validation checks are passed
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
                                                                       training_period_end,
                                                                       include_twb_metrics=True)

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
    # crypto_inflows: btc(100 + 50) + eth(200 + 50) = 400
    expected_inflows = 400

    # crypto_outflows: end balance = 180 + 280
    expected_outflows = 460

    # crypto_net_flows: 400 - 460
    expected_net = -60

    # crypto_net_gain: (180 + 280) - 400 = 60
    expected_gain = 60

    # # transaction_days: 2 days (1/1 and 5/1)
    # expected_txn_days = 2

    # # unique_coins_traded: BTC and ETH
    # expected_coins = 2

    # total_volume: abs(100) + abs(50) + abs(200) + abs(50) = 400
    expected_volume = 400

    # average_transaction: 400 / 4 = 100
    expected_avg_txn = 100

    # time_weighted_balance:
    # Initial period (12/31-5/1): 300 * 120 days = 36000
    # Final period (5/1-10/1): 400 * 154 days = 61600
    # Total = 97600 / 274 days ≈ 356.20
    expected_twb = 356.20

    # # activity_density: 2 transaction days / 274 total days ≈ 0.0073
    # expected_density = 2 / 274

    # volume_vs_twb_ratio: 400 / 356.20 ≈ 1.123
    expected_ratio = 400 / 356.20

    # Assert all values match
    assert wallet_features['crypto_inflows'] == expected_inflows
    assert wallet_features['crypto_outflows'] == expected_outflows
    assert wallet_features['crypto_net_flows'] == expected_net
    assert wallet_features['crypto_net_gain'] == expected_gain
    # assert wallet_features['transaction_days'] == expected_txn_days
    # assert wallet_features['unique_coins_traded'] == expected_coins
    assert wallet_features['total_volume'] == expected_volume
    assert wallet_features['average_transaction'] == expected_avg_txn
    assert np.isclose(wallet_features['time_weighted_balance'], expected_twb, rtol=1e-2)
    # assert np.isclose(wallet_features['activity_density'], expected_density, rtol=1e-2)
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

    print(wallet_features)
    # Calculate expected values
    # crypto_inflows: 10000 + 500 balance = 10500
    expected_inflows = 10000 + 500

    # crypto_outflows: 10000 sell + 750 ending balance
    expected_outflows = 10750

    # crypto_net_flows: 10500 - 10750
    expected_net = -250

    # crypto_net_gain: 750 ending - 500 cost basis = 250
    expected_gain = 250

    # # transaction_days: 2 days with non-imputed transactions (2/1 and 2/2)
    # expected_txn_days = 2

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

    # # activity_density: 2 transaction days / 274 total days ≈ 0.0073
    # expected_density = 2 / 274

    # volume_vs_twb_ratio: 20000 / 536.23 ≈ 44.27
    expected_ratio = 20000 / 451.74

    # Assert all values match
    assert wallet_features['crypto_inflows'] == expected_inflows
    assert wallet_features['crypto_outflows'] == expected_outflows
    assert wallet_features['crypto_net_flows'] == expected_net
    assert wallet_features['crypto_net_gain'] == expected_gain
    # assert wallet_features['transaction_days'] == expected_txn_days
    assert wallet_features['unique_coins_traded'] == expected_coins
    assert wallet_features['total_volume'] == expected_volume
    assert wallet_features['average_transaction'] == expected_avg_txn
    assert np.isclose(wallet_features['time_weighted_balance'], expected_twb, rtol=1e-2)
    # assert np.isclose(wallet_features['activity_density'], expected_density, rtol=1e-2)
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
    # crypto_inflows: initial 100 buy
    expected_inflows = 100

    # crypto_outflows: 500 + 100 + 10 ending balance
    expected_outflows = 610

    # crypto_net_flows: 100 - 610
    expected_net = -510

    # crypto_net_gain: 10 ending - (-500 net investment) = 510
    expected_gain = 510

    # # transaction_days: 2 days with non-imputed transactions (3/1 and 5/1)
    # expected_txn_days = 3

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

    # expected_density = 3 / 274

    expected_ratio = 700 / 35.65322

    # Assert all values match
    assert wallet_features['crypto_inflows'] == expected_inflows
    assert wallet_features['crypto_outflows'] == expected_outflows
    assert wallet_features['crypto_net_flows'] == expected_net
    assert wallet_features['crypto_net_gain'] == expected_gain
    # assert wallet_features['transaction_days'] == expected_txn_days
    assert wallet_features['unique_coins_traded'] == expected_coins
    assert wallet_features['total_volume'] == expected_volume
    assert wallet_features['average_transaction'] == expected_avg_txn
    assert np.isclose(wallet_features['time_weighted_balance'], expected_twb, rtol=1e-2)
    # assert np.isclose(wallet_features['activity_density'], expected_density, rtol=1e-2)
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

# FeatureRemoval due to nonpredictiveness
# @pytest.mark.unit
# def test_activity_density_range(test_trading_features_df):
#     """Verify activity_density is between 0 and 1 since it's a ratio of days"""
#     assert (test_trading_features_df['activity_density'] >= 0).all()
#     assert (test_trading_features_df['activity_density'] <= 1).all()

# @pytest.mark.unit
# def test_transaction_days_vs_unique_coins(test_trading_features_df):
#     """Verify transaction_days >= unique_coins_traded
#     Since each unique coin must have at least one transaction day"""
#     assert (
#         test_trading_features_df['transaction_days'] >=
#         test_trading_features_df['unique_coins_traded']
#     ).all()

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



# ------------------------------------------------- #
# calculate_wallet_coin_time_weighted_returns() unit tests
# ------------------------------------------------- #

@pytest.mark.unit
def test_calculate_wallet_coin_time_weighted_returns_imputed_case():
    """Tests TWR calculation for a wallet with only imputed balances."""

    # Setup test data
    test_data = pd.DataFrame([
        {'coin_id': 'btc', 'wallet_address': 'wallet_a', 'date': '2024-01-01',
         'usd_balance': 50, 'usd_net_transfers': 0, 'is_imputed': True},
        {'coin_id': 'btc', 'wallet_address': 'wallet_a', 'date': '2024-10-01',
         'usd_balance': 70, 'usd_net_transfers': 0, 'is_imputed': True},
    ])
    test_data['date'] = pd.to_datetime(test_data['date'])

    # Calculate TWR
    test_data = test_data.set_index(['coin_id', 'wallet_address', 'date']).sort_index()
    result = wtf.calculate_wallet_coin_time_weighted_returns(test_data)

    # Expected values
    expected_twr = 0.40  # (70-50)/50 = 0.4
    expected_days = 274  # Jan 1 to Oct 1
    expected_annual = ((1 + 0.40) ** (365/274)) - 1  # ≈ 0.55

    # Assertions with tolerance for floating point
    assert abs(result.loc[('wallet_a', 'btc'), 'time_weighted_return'] - expected_twr) < 0.001
    assert result.loc[('wallet_a', 'btc'), 'days_held'] == expected_days
    assert abs(result.loc[('wallet_a', 'btc'), 'annualized_twr'] - expected_annual) < 0.001


@pytest.mark.unit
def test_calculate_wallet_coin_time_weighted_returns_weighted_periods():
    """Tests TWR calculation with different holding periods, amounts, and a transfer."""

    test_data = pd.DataFrame([
        {'coin_id': 'eth', 'wallet_address': 'wallet_a', 'date': '2024-01-01',
            'usd_balance': 100, 'usd_net_transfers': 100, 'is_imputed': False},  # Initial $100
        {'coin_id': 'eth', 'wallet_address': 'wallet_a', 'date': '2024-02-01',
            'usd_balance': 250, 'usd_net_transfers': 50, 'is_imputed': False},    # Added $50, value up
        {'coin_id': 'eth', 'wallet_address': 'wallet_a', 'date': '2024-10-01',
            'usd_balance': 125, 'usd_net_transfers': 0, 'is_imputed': True}     # Value dropped
    ])

    test_data['date'] = pd.to_datetime(test_data['date'])
    test_data = test_data.set_index(['coin_id', 'wallet_address', 'date']).sort_index()
    result = wtf.calculate_wallet_coin_time_weighted_returns(test_data)

    # Manual calculation:
    # Period 1: Jan 1 - Feb 1 (31 days)
    # Pre-transfer balance = 250 - 50 = 200
    # Return = 200/100 = 100% = 1.0
    # Weighted return = 1.0 * 31 = 31

    # Period 2: Feb 1 - Oct 1 (243 days)
    # Return = 125/250 = -50% = -0.5
    # Weighted return = -0.5 * 243 = -121.5

    # Total days = 274
    # Time weighted return = (31 - 121.5) / 274 = -0.33
    expected_twr = -0.33

    # Annualized = (1 - 0.33)^(365/274) - 1 ≈ -0.41
    expected_annual = ((1 + expected_twr) ** (365/274)) - 1

    # Assertions
    assert result.loc[('wallet_a', 'eth'), 'days_held'] == 274
    assert abs(result.loc[('wallet_a', 'eth'), 'time_weighted_return'] - expected_twr) < 0.01
    assert abs(result.loc[('wallet_a', 'eth'), 'annualized_twr'] - expected_annual) < 0.01


@pytest.mark.unit
def test_calculate_wallet_coin_time_weighted_returns_w02_net_loss(test_profits_df):
    """
    Tests TWR calculation for w02_net_loss with a single coin and net losses.
    """
    profits_df, _, _ = test_profits_df

    # Filter data for w02_net_loss
    w02_data = profits_df[profits_df['wallet_address'] == 'w02_net_loss']

    # BTC Calculation
    # Period 1: Jan 1 - May 1 (120 days)
    # Pre-transfer balance = 250 - (-100) = 350
    # Period return = 350 / 300 ≈ 1.1667
    # Weighted return = 0.1667 * 120 ≈ 20

    # Period 2: May 1 - Oct 1 (153 days)
    # Period return = 100 / 250 = 0.4
    # Weighted return = -0.6 * 153 = -91.8

    # Total weighted return = 20 - 91.8 = -71.8
    # Total days = 120 + 153 = 273
    # Time-weighted return = -71.8 / 273 ≈ -0.263

    time_weighted_return = -71.8 / 274

    # Annualized return
    # Annualized TWR = ((1 + TWR) ** (365 / 273)) - 1
    annualized_twr = ((1 + time_weighted_return) ** (365 / 274)) - 1

    # Calculate TWR using the function
    w02_data = w02_data.set_index(['coin_id', 'wallet_address', 'date']).sort_index()
    result = wtf.calculate_wallet_coin_time_weighted_returns(w02_data)

    # Assertions
    assert result.loc[('w02_net_loss', 'btc'), 'days_held'] == 274
    assert abs(result.loc[('w02_net_loss', 'btc'), 'time_weighted_return'] - time_weighted_return) < 0.001
    assert abs(result.loc[('w02_net_loss', 'btc'), 'annualized_twr'] - annualized_twr) < 0.001


@pytest.mark.unit
def test_calculate_wallet_coin_time_weighted_returns_w01_btc(test_profits_df):
    """
    Tests TWR calculation for w01_multiple_coins (BTC only).
    """
    profits_df, _, _ = test_profits_df

    # Filter data for w01_multiple_coins (BTC only)
    btc_data = profits_df[
        (profits_df['wallet_address'] == 'w01_multiple_coins') &
        (profits_df['coin_id'] == 'btc')
    ]

    # Manual Calculation
    # Period 1: Jan 1 - May 1 (121 days, inclusive)
    # Pre-transfer balance = 120 - 50 = 70
    # Previous balance = 100
    # Period return = 70 / 100 = 0.7
    # Weighted return = (0.7 - 1) * 121 = -36.3

    # Period 2: May 2 - Oct 1 (153 days, inclusive)
    # Pre-transfer balance = 180 (no transfer on Oct 1)
    # Previous balance = 120
    # Period return = 180 / 120 = 1.5
    # Weighted return = (1.5 - 1) * 153 = 76.5

    # Total weighted return = -36.3 + 76.5 = 40.2
    # Total days = 121 + 153 = 274
    # Time-weighted return = 40.2 / 274 ≈ 0.1467

    total_days = 121 + 153  # 274
    period_1_return = (70 / 100) - 1  # -0.3
    period_1_weighted_return = period_1_return * 121  # -36.3

    period_2_return = (180 / 120) - 1  # 0.5
    period_2_weighted_return = period_2_return * 153  # 76.5

    total_weighted_return = period_1_weighted_return + period_2_weighted_return  # -36.3 + 76.5 = 40.2
    time_weighted_return = total_weighted_return / total_days  # 40.2 / 274 ≈ 0.1467

    # Annualized return
    annualized_twr = ((1 + time_weighted_return) ** (365 / total_days)) - 1

    btc_data = btc_data.set_index(['coin_id', 'wallet_address', 'date']).sort_index()
    result = wtf.calculate_wallet_coin_time_weighted_returns(btc_data)

    # Assertions
    assert result.loc[('w01_multiple_coins', 'btc'), 'days_held'] == total_days
    assert abs(result.loc[('w01_multiple_coins', 'btc'), 'time_weighted_return'] - time_weighted_return) < 0.001
    assert abs(result.loc[('w01_multiple_coins', 'btc'), 'annualized_twr'] - annualized_twr) < 0.001


@pytest.mark.unit
def test_calculate_wallet_coin_time_weighted_returns_w01_eth(test_profits_df):
    """
    Tests TWR calculation for w01_multiple_coins (ETH only).
    """
    profits_df, _, _ = test_profits_df

    # Filter data for w01_multiple_coins (ETH only)
    eth_data = profits_df[
        (profits_df['wallet_address'] == 'w01_multiple_coins') &
        (profits_df['coin_id'] == 'eth')
    ]

    # Manual Calculation
    # Period 1: Jan 1 - May 1 (121 days, inclusive)
    # Pre-transfer balance = 300 - 50 = 250
    # Previous balance = 200
    # Period return = 250 / 200 = 1.25
    # Weighted return = (1.25 - 1) * 121 = 30.25

    # Period 2: May 2 - Oct 1 (153 days, inclusive)
    # Pre-transfer balance = 280 (no transfer on Oct 1)
    # Previous balance = 300
    # Period return = 280 / 300 = 0.9333
    # Weighted return = (0.9333 - 1) * 153 ≈ -10.22

    # Total weighted return = 30.25 - 10.22 = 20.03
    # Total days = 121 + 153 = 274
    # Time-weighted return = 20.03 / 274 ≈ 0.0731

    total_days = 121 + 153  # 274
    period_1_return = (250 / 200) - 1  # 0.25
    period_1_weighted_return = period_1_return * 121  # 30.25

    period_2_return = (280 / 300) - 1  # -0.0667
    period_2_weighted_return = period_2_return * 153  # ≈ -10.22

    total_weighted_return = period_1_weighted_return + period_2_weighted_return  # 30.25 - 10.22 = 20.03
    time_weighted_return = total_weighted_return / total_days  # 20.03 / 274 ≈ 0.0731

    # Annualized return
    annualized_twr = ((1 + time_weighted_return) ** (365 / total_days)) - 1

    # Run the function
    eth_data = eth_data.set_index(['coin_id', 'wallet_address', 'date']).sort_index()
    result = wtf.calculate_wallet_coin_time_weighted_returns(eth_data)

    # Assertions
    assert result.loc[('w01_multiple_coins', 'eth'), 'days_held'] == total_days
    assert abs(result.loc[('w01_multiple_coins', 'eth'), 'time_weighted_return'] - time_weighted_return) < 0.001
    assert abs(result.loc[('w01_multiple_coins', 'eth'), 'annualized_twr'] - annualized_twr) < 0.001






# ----------------------------------------------------------
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

    # Remove rows with a rounded 0 balance and 0 transfers which happens in wtdo.retrieve_datasets() once validation checks are passed
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
                                                                       training_period_end,
                                                                       include_twb_metrics=True)

    # Confirm that all the addresses have been mapped
    expected_addresses = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6']
    assert sorted(list(remapped_wallet_trading_features_df.index.values)) == expected_addresses

    return remapped_wallet_trading_features_df


@pytest.mark.unit
def test_volume_aggregation_after_remapping(test_trading_features_df,
                                          test_remapped_profits_df,
                                          test_remapped_trading_features_df):
    """
    Verifies that total volume metrics are preserved when wallets are consolidated through remapping.

    The test ensures volume metrics are properly summed when multiple original wallets
    are mapped to the same new wallet address.
    """
    test_remapped_profits_df, _, _ = test_remapped_profits_df

    # Create mapping from original to new wallets
    wallet_mapping = (test_remapped_profits_df[['wallet_address', 'wallet_address_original']]
                     .drop_duplicates()
                     .set_index('wallet_address_original')['wallet_address'])

    # Map original volumes to new wallet structure
    original_volumes = pd.DataFrame({
        'total_volume': test_trading_features_df['total_volume'],
        'crypto_inflows': test_trading_features_df['crypto_inflows'],
        'crypto_outflows': test_trading_features_df['crypto_outflows']
    })
    original_volumes['new_wallet'] = original_volumes.index.map(wallet_mapping)

    # Calculate expected volumes by grouping by new wallet address
    expected_volumes = original_volumes.groupby('new_wallet').agg({
        'total_volume': 'sum',
        'crypto_inflows': 'sum',
        'crypto_outflows': 'sum'
    }).sort_index()

    # Get actual volumes from remapped features
    actual_volumes = test_remapped_trading_features_df[
        ['total_volume', 'crypto_inflows', 'crypto_outflows']
    ].sort_index()

    # Compare each volume metric
    for col in ['total_volume', 'crypto_inflows', 'crypto_outflows']:
        assert np.allclose(expected_volumes[col], actual_volumes[col], equal_nan=True), \
            f"{col} doesn't match after remapping"


# FeatureRemoval due to nonpredictiveness
# @pytest.mark.unit
# def test_activity_metrics_after_remapping(test_profits_df,
#                                         test_remapped_profits_df,
#                                         test_remapped_trading_features_df):
#     """
#     Verifies activity metrics are calculated correctly after wallet remapping.

#     Tests:
#     1. transaction_days: Number of unique dates with non-imputed transactions
#     2. unique_coins_traded: Count of distinct coins traded by each wallet
#     3. activity_density: transaction_days / total_period_days
#     """
#     # Unpack tuples
#     test_profits_df, _, _ = test_profits_df
#     test_remapped_profits_df, _, _ = test_remapped_profits_df

#     # Create mapping from original to new wallets
#     wallet_mapping = (test_remapped_profits_df[['wallet_address', 'wallet_address_original']]
#                      .drop_duplicates()
#                      .set_index('wallet_address_original')['wallet_address'])

#     # Calculate expected transaction days (non-imputed activity)
#     active_dates = (test_profits_df[~test_profits_df['is_imputed']]
#                    .assign(new_wallet=lambda x: x['wallet_address'].map(wallet_mapping))
#                    .groupby('new_wallet')['date']
#                    .nunique()
#                    .sort_index())

#     # Calculate expected unique coins
#     unique_coins = (test_profits_df[~test_profits_df['is_imputed']]
#                    .assign(new_wallet=lambda x: x['wallet_address'].map(wallet_mapping))
#                    .groupby('new_wallet')['coin_id']
#                    .nunique()
#                    .sort_index())

#     # Calculate expected activity density
#     period_days = (pd.to_datetime(test_profits_df['date']).max() -
#                   pd.to_datetime(test_profits_df['date']).min()).days
#     expected_density = active_dates / period_days

#     # Get actual metrics, sorted for comparison
#     actual = test_remapped_trading_features_df[[
#         'transaction_days',
#         'unique_coins_traded',
#         'activity_density'
#     ]].sort_index()

#     # Compare each metric with specific error messages
#     assert np.allclose(active_dates, actual['transaction_days'], equal_nan=True), \
#         "Transaction days don't match after remapping"
#     assert np.allclose(unique_coins, actual['unique_coins_traded'], equal_nan=True), \
#         "Unique coins traded don't match after remapping"
#     assert np.allclose(expected_density, actual['activity_density'], equal_nan=True), \
#         "Activity density doesn't match after remapping"


@pytest.mark.unit
def test_ratio_metrics_after_remapping(test_profits_df,
                                     test_remapped_profits_df,
                                     test_remapped_trading_features_df):
    """
    Verifies ratio-based metrics are calculated correctly after wallet remapping.

    Tests:
    1. average_transaction = total_volume / num_transactions
    2. volume_vs_twb_ratio = total_volume / time_weighted_balance
    """
    # Unpack tuples
    test_profits_df, _, _ = test_profits_df
    test_remapped_profits_df, _, _ = test_remapped_profits_df

    # Create mapping from original to new wallets
    wallet_mapping = (test_remapped_profits_df[['wallet_address', 'wallet_address_original']]
                        .drop_duplicates()
                        .set_index('wallet_address_original')['wallet_address'])

    # Count transactions and sum volumes for non-imputed rows
    metrics = (test_profits_df[~test_profits_df['is_imputed']]
                .assign(new_wallet=lambda x: x['wallet_address'].map(wallet_mapping))
                .groupby('new_wallet')
                .agg(
                    txn_count=('usd_net_transfers', 'size'),
                    total_volume=('usd_net_transfers', lambda x: abs(x).sum())
                ))

    expected_avg_transaction = (metrics['total_volume'] / metrics['txn_count']).sort_index()

    # Get actual metrics
    actual = test_remapped_trading_features_df[[
        'average_transaction',
        'volume_vs_twb_ratio'
    ]].sort_index()

    # Compare metrics
    assert np.allclose(expected_avg_transaction, actual['average_transaction'], rtol=1e-2), \
        "Average transaction doesn't match after remapping"

    # Verify volume_vs_twb_ratio calculation
    expected_ratio = np.where(
        test_remapped_trading_features_df['time_weighted_balance'] > 0,
        test_remapped_trading_features_df['total_volume'] /
        test_remapped_trading_features_df['time_weighted_balance'],
        0
    )
    assert np.allclose(expected_ratio, actual['volume_vs_twb_ratio'], rtol=1e-2), \
        "Volume vs TWB ratio doesn't match expected calculation"


@pytest.mark.unit
def test_trading_metrics_aggregation(test_trading_features_df,
                                   test_remapped_profits_df,
                                   test_remapped_trading_features_df):
    """
    Verifies trading metrics sum correctly when wallets are remapped.
    """
    test_remapped_profits_df, _, _ = test_remapped_profits_df

    # Create mapping from original to new wallets
    wallet_mapping = (test_remapped_profits_df[['wallet_address', 'wallet_address_original']]
                     .drop_duplicates()
                     .set_index('wallet_address_original')['wallet_address'])

    # Sum original wallet metrics grouped by their new mapped wallet
    original_metrics = pd.DataFrame(test_trading_features_df[[
        'crypto_inflows',
        'crypto_outflows',
        'crypto_net_flows',
        'crypto_net_gain'
    ]])
    original_metrics['new_wallet'] = original_metrics.index.map(wallet_mapping)
    expected = original_metrics.groupby('new_wallet').sum().sort_index()

    # Get actual remapped metrics
    actual = test_remapped_trading_features_df[[
        'crypto_inflows',
        'crypto_outflows',
        'crypto_net_flows',
        'crypto_net_gain'
    ]].sort_index()

    # Compare each metric
    assert np.allclose(expected, actual, rtol=1e-2), "Remapped trading metrics don't sum correctly"
