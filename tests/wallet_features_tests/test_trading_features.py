"""
tests used to audit the files in the data-science/src folder
"""
# pylint: disable=wrong-import-position  # import not at top of doc (due to local import)
# pylint: disable=redefined-outer-name  # redefining from outer scope triggering on pytest fixtures
# pylint: disable=unused-argument
# pyright: reportMissingModuleSource=false

import sys
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

@dataclass
class ProfitsValidator:
    """
    Validates profits DataFrame follows expected format and constraints.
    Only validates training period data.
    """
    def validate_all(self, profits_df, training_period_start, training_period_end):
        """Run all validation checks and return dict of results"""
        dates = {
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
        return (profits_df['date'].min() >= dates['training_period_start'] and
                profits_df['date'].max() == dates['training_period_end'])

    def check_no_missing_values(self, profits_df):
        """Check for missing values"""
        return not profits_df.isna().any().any()


# pylint:disable=line-too-long
@pytest.fixture
def test_profits_df():
    """
    Returns test profits DataFrame with cash flow transfers added.
    """
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

        # w03_sell_all_and_rebuy - ada (sell full balance mid-way and repurchase)
        {'coin_id': 'ada', 'wallet_address': 'w03_sell_all_and_rebuy', 'date': '2024-01-01', 'usd_balance': 50, 'usd_net_transfers': 50, 'is_imputed': False},
        {'coin_id': 'ada', 'wallet_address': 'w03_sell_all_and_rebuy', 'date': '2024-03-01', 'usd_balance': 0,  'usd_net_transfers': -50, 'is_imputed': False},
        {'coin_id': 'ada', 'wallet_address': 'w03_sell_all_and_rebuy', 'date': '2024-08-01', 'usd_balance': 40, 'usd_net_transfers': 40, 'is_imputed': False},
        {'coin_id': 'ada', 'wallet_address': 'w03_sell_all_and_rebuy', 'date': '2024-10-01', 'usd_balance': 42, 'usd_net_transfers': 0, 'is_imputed': True},

        # w04_only_period_end - btc (only final row)
        {'coin_id': 'btc', 'wallet_address': 'w04_only_period_end', 'date': '2024-10-01', 'usd_balance': 70, 'usd_net_transfers': 70, 'is_imputed': False},

        # w04a_only_period_end_w_balance - btc
        {'coin_id': 'btc', 'wallet_address': 'w04a_only_period_end_w_balance', 'date': '2024-01-01', 'usd_balance': 30, 'usd_net_transfers': 0, 'is_imputed': True},
        {'coin_id': 'btc', 'wallet_address': 'w04a_only_period_end_w_balance', 'date': '2024-10-01', 'usd_balance': 90, 'usd_net_transfers': 50, 'is_imputed': False},

        # w04b_only_period_start_buy
        {'coin_id': 'btc', 'wallet_address': 'w04b_only_period_start_buy', 'date': '2024-01-01', 'usd_balance': 300, 'usd_net_transfers': 300, 'is_imputed': False},
        {'coin_id': 'btc', 'wallet_address': 'w04b_only_period_start_buy', 'date': '2024-10-01', 'usd_balance': 900, 'usd_net_transfers': 0, 'is_imputed': True},

        # w04c_only_period_start_buy_w_existing_balance
        {'coin_id': 'btc', 'wallet_address': 'w04c_only_period_start_buy_w_existing_balance', 'date': '2024-01-01', 'usd_balance': 350, 'usd_net_transfers': 300, 'is_imputed': False},
        {'coin_id': 'btc', 'wallet_address': 'w04c_only_period_start_buy_w_existing_balance', 'date': '2024-10-01', 'usd_balance': 1050, 'usd_net_transfers': 0, 'is_imputed': True},

        # w04d_only_period_start_sell
        {'coin_id': 'btc', 'wallet_address': 'w04d_only_period_start_sell', 'date': '2024-01-01', 'usd_balance': 0, 'usd_net_transfers': -200, 'is_imputed': False},
        {'coin_id': 'btc', 'wallet_address': 'w04d_only_period_start_sell', 'date': '2024-10-01', 'usd_balance': 0, 'usd_net_transfers': 0, 'is_imputed': True},

        # w04e_only_period_start_sell_partial
        {'coin_id': 'btc', 'wallet_address': 'w04e_only_period_start_sell_partial', 'date': '2024-01-01', 'usd_balance': 500, 'usd_net_transfers': -10, 'is_imputed': False},
        {'coin_id': 'btc', 'wallet_address': 'w04e_only_period_start_sell_partial', 'date': '2024-10-01', 'usd_balance': 600, 'usd_net_transfers': 0, 'is_imputed': True},

        # w05_only_imputed - btc (only imputed rows at start and end)
        {'coin_id': 'btc', 'wallet_address': 'w05_only_imputed', 'date': '2024-01-01', 'usd_balance': 50, 'usd_net_transfers': 0, 'is_imputed': True},
        {'coin_id': 'btc', 'wallet_address': 'w05_only_imputed', 'date': '2024-10-01', 'usd_balance': 70, 'usd_net_transfers': 0, 'is_imputed': True},

        # w06_tiny_transactions - very small transactions relative to portfolio size
        {'coin_id': 'myro', 'wallet_address': 'w06_tiny_transactions', 'date': '2024-01-01', 'usd_balance': 1250, 'usd_net_transfers': 0, 'is_imputed': True},
        {'coin_id': 'myro', 'wallet_address': 'w06_tiny_transactions', 'date': '2024-02-01', 'usd_balance': 1220, 'usd_net_transfers': 1, 'is_imputed': False},
        {'coin_id': 'myro', 'wallet_address': 'w06_tiny_transactions', 'date': '2024-08-01', 'usd_balance': 0, 'usd_net_transfers': -350, 'is_imputed': False},
        {'coin_id': 'myro', 'wallet_address': 'w06_tiny_transactions', 'date': '2024-10-01', 'usd_balance': 0, 'usd_net_transfers': 0, 'is_imputed': True},

        # w07_tiny_transactions2 - very small transactions relative to portfolio size
        {'coin_id': 'floki', 'wallet_address': 'w07_tiny_transactions2', 'date': '2024-01-01', 'usd_balance': 400, 'usd_net_transfers': 0, 'is_imputed': True},
        {'coin_id': 'floki', 'wallet_address': 'w07_tiny_transactions2', 'date': '2024-02-01', 'usd_balance': 1220, 'usd_net_transfers': -20, 'is_imputed': False},
        {'coin_id': 'floki', 'wallet_address': 'w07_tiny_transactions2', 'date': '2024-08-01', 'usd_balance': 0, 'usd_net_transfers': -150, 'is_imputed': False},
        {'coin_id': 'floki', 'wallet_address': 'w07_tiny_transactions2', 'date': '2024-10-01', 'usd_balance': 0, 'usd_net_transfers': 0, 'is_imputed': True},

        # w08_offsetting_transactions - large offsetting transactions in the middle of the period
        {'coin_id': 'floki', 'wallet_address': 'w08_offsetting_transactions', 'date': '2024-01-01', 'usd_balance': 500, 'usd_net_transfers': 0, 'is_imputed': True},
        {'coin_id': 'floki', 'wallet_address': 'w08_offsetting_transactions', 'date': '2024-02-01', 'usd_balance': 10400, 'usd_net_transfers': 10000, 'is_imputed': False},
        {'coin_id': 'floki', 'wallet_address': 'w08_offsetting_transactions', 'date': '2024-02-02', 'usd_balance': 400, 'usd_net_transfers': -10000, 'is_imputed': False},
        {'coin_id': 'floki', 'wallet_address': 'w08_offsetting_transactions', 'date': '2024-10-01', 'usd_balance': 750, 'usd_net_transfers': 0, 'is_imputed': True},

        # w09_memecoin_winner - Large swings in portfolio value
        {'coin_id': 'pepe', 'wallet_address': 'w09_memecoin_winner', 'date': '2024-01-01', 'usd_balance': 100, 'usd_net_transfers': 100, 'is_imputed': False},
        {'coin_id': 'pepe', 'wallet_address': 'w09_memecoin_winner', 'date': '2024-03-01', 'usd_balance': 250, 'usd_net_transfers': -500, 'is_imputed': False},
        {'coin_id': 'pepe', 'wallet_address': 'w09_memecoin_winner', 'date': '2024-05-01', 'usd_balance': 50, 'usd_net_transfers': -100, 'is_imputed': False},
        {'coin_id': 'pepe', 'wallet_address': 'w09_memecoin_winner', 'date': '2024-10-01', 'usd_balance': 10, 'usd_net_transfers': 0, 'is_imputed': True},

        # w10_memecoin_loser - Large swings in portfolio value
        {'coin_id': 'bome', 'wallet_address': 'w10_memecoin_loser', 'date': '2024-03-01', 'usd_balance': 250, 'usd_net_transfers': 250, 'is_imputed': False},
        {'coin_id': 'bome', 'wallet_address': 'w10_memecoin_loser', 'date': '2024-10-01', 'usd_balance': 0, 'usd_net_transfers': -20, 'is_imputed': False},

        # w11_sells_early
        {'coin_id': 'btc', 'wallet_address': 'w11_sells_early', 'date': '2024-03-01', 'usd_balance': 0, 'usd_net_transfers': 0, 'is_imputed': True},
        {'coin_id': 'btc', 'wallet_address': 'w11_sells_early', 'date': '2024-04-01', 'usd_balance': 250, 'usd_net_transfers': 250, 'is_imputed': False},
        {'coin_id': 'btc', 'wallet_address': 'w11_sells_early', 'date': '2024-5-01', 'usd_balance': 0, 'usd_net_transfers': -300, 'is_imputed': False},
        {'coin_id': 'btc', 'wallet_address': 'w11_sells_early', 'date': '2024-10-01', 'usd_balance': 0, 'usd_net_transfers': 0, 'is_imputed': True},

        # w12_buys_late
        {'coin_id': 'btc', 'wallet_address': 'w12_buys_late', 'date': '2024-03-01', 'usd_balance': 0, 'usd_net_transfers': 0, 'is_imputed': True},
        {'coin_id': 'btc', 'wallet_address': 'w12_buys_late', 'date': '2024-09-01', 'usd_balance': 500, 'usd_net_transfers': 250, 'is_imputed': False},
        {'coin_id': 'btc', 'wallet_address': 'w12_buys_late', 'date': '2024-10-01', 'usd_balance': 550, 'usd_net_transfers': 0, 'is_imputed': True},

    ]
    profits_df = pd.DataFrame(profits_data)
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

    # Remove rows with a rounded 0 balance and 0 transfers which happens in wo.retrieve_datasets() once validation checks are passed
    profits_df = profits_df[
        ~((profits_df['usd_balance'] == 0) &
        (profits_df['usd_net_transfers'] == 0))
    ]

    # Add cash flow transfers logic
    cash_flow_profits_df = wtf.add_cash_flow_transfers_logic(profits_df)

    return cash_flow_profits_df


@pytest.fixture
def test_trading_features_df(test_profits_df):
    """
    Returns trading features computed from test profits data.
    """
    return wtf.calculate_wallet_trading_features(test_profits_df)


def test_w01_multiple_coins(test_profits_df, test_trading_features_df):
    """
    Test wallet with multiple coins and transactions.

    Scenario:
    - Two coins (BTC, ETH)
    - Multiple transactions per coin
    - Mix of real and imputed rows
    """
    # Get w01 data
    wallet = 'w01_multiple_coins'
    w01_profits = test_profits_df[test_profits_df['wallet_address'] == wallet]
    w01_features = test_trading_features_df.loc[wallet]

    # Test basic metrics
    assert w01_features['transaction_days'] == 2  # Jan 1 and May 1
    assert w01_features['unique_coins_traded'] == 2  # BTC and ETH
    assert w01_features['cash_buy_inflows'] == 400  # Initial: BTC 100 + ETH 200, Add: BTC 50 + ETH 50

    # Test volume metrics
    assert w01_features['total_volume'] == 400  # Sum of all transfers
    assert w01_features['average_transaction'] == 100  # 400 / 4 transactions

    # Test imputed metrics
    assert w01_features['total_inflows'] == 400  # Initial balances
    assert w01_features['net_gain'] > 0  # Should be profitable given ending balances > deposits

    # Test activity metrics
    total_days = (w01_profits['date'].max() - w01_profits['date'].min()).days + 1
    assert w01_features['activity_density'] == pytest.approx(2 / total_days, rel=1e-10)


# ===== Volume Tests =====
def test_volume_is_positive(test_trading_features_df):
    """Verify all volume metrics are non-negative"""
    volume_cols = ['total_volume', 'average_transaction']
    for col in volume_cols:
        assert (test_trading_features_df[col] >= 0).all(), f"Found negative values in {col}"

def test_volume_matches_transfers(test_profits_df, test_trading_features_df):
    """Verify total_volume matches sum of absolute transfers"""
    expected = test_profits_df.groupby('wallet_address')['usd_net_transfers'].agg(lambda x: abs(x).sum())
    actual = test_trading_features_df['total_volume'].reindex(expected.index)
    np.allclose(actual, expected)


# # ===== Cash Flow Tests =====
def test_cash_flows_sum_correctly(test_trading_features_df):
    """Verify cash flow components sum correctly"""
    cash_invested = test_trading_features_df['cash_buy_inflows']
    cash_withdrawn = test_trading_features_df['cash_sell_outflows']
    net_flows = test_trading_features_df['cash_net_flows']

    expected_net = cash_withdrawn - cash_invested
    assert np.allclose(net_flows, expected_net)

def test_flows_match_activity_days(test_profits_df, test_trading_features_df):
    """Verify transaction days match observed activity"""
    active_days = (test_profits_df[~test_profits_df['is_imputed']]
                    .groupby('wallet_address')['date']
                    .nunique()
                    .reindex(test_trading_features_df.index)
                    .fillna(0))

    np.allclose(test_trading_features_df['transaction_days'],active_days)


# ===== Return Tests =====
def test_returns_in_reasonable_range(test_trading_features_df):
    """Verify return metrics are within reasonable bounds"""
    return_cols = ['time_weighted_return', 'annualized_twr']
    for col in return_cols:
        assert (test_trading_features_df[col].abs() <= 10).all(), f"Found extreme values in {col}"
