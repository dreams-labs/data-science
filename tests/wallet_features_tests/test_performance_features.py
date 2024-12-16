"""
tests used to audit the files in the data-science/src folder
"""
# pylint: disable=wrong-import-position  # import not at top of doc (due to local import)
# pylint: disable=redefined-outer-name  # redefining from outer scope triggering on pytest fixtures
# pylint: disable=unused-argument
# pyright: reportMissingModuleSource=false

import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import pytest
from dreams_core import core as dc

# pyright: reportMissingImports=false
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
import wallet_features.performance_features as wpf
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
# calculate_time_weighted_returns() unit tests
# ------------------------------------------------- #

@pytest.fixture
def portfolio_test_data():
    """Test data fixture with both BTC and ETH holdings."""
    test_data = pd.DataFrame([
        # BTC wallet with imputed values
        {'coin_id': 'btc', 'wallet_address': 'wallet_a', 'date': '2024-01-01',
         'usd_balance': 50, 'usd_net_transfers': 0, 'is_imputed': True},
        {'coin_id': 'btc', 'wallet_address': 'wallet_a', 'date': '2024-10-01',
         'usd_balance': 70, 'usd_net_transfers': 0, 'is_imputed': True},
        # ETH wallet with transfers
        {'coin_id': 'eth', 'wallet_address': 'wallet_a', 'date': '2024-01-01',
         'usd_balance': 100, 'usd_net_transfers': 100, 'is_imputed': False},
        {'coin_id': 'eth', 'wallet_address': 'wallet_a', 'date': '2024-02-01',
         'usd_balance': 250, 'usd_net_transfers': 50, 'is_imputed': False},
        {'coin_id': 'eth', 'wallet_address': 'wallet_a', 'date': '2024-10-01',
         'usd_balance': 125, 'usd_net_transfers': 0, 'is_imputed': False}
    ])
    test_data['date'] = pd.to_datetime(test_data['date'])
    return test_data


@pytest.mark.unit
def test_calculate_time_weighted_returns_imputed_case():
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
    result = wpf.calculate_time_weighted_returns(test_data)

    # Expected values
    expected_twr = 0.40  # (70-50)/50 = 0.4
    expected_days = 274  # Jan 1 to Oct 1
    expected_annual = ((1 + 0.40) ** (365/274)) - 1  # ≈ 0.55

    # Assertions with tolerance for floating point
    assert abs(result.loc['wallet_a', 'time_weighted_return'] - expected_twr) < 0.001
    assert result.loc['wallet_a', 'days_held'] == expected_days
    assert abs(result.loc['wallet_a', 'annualized_twr'] - expected_annual) < 0.001


@pytest.mark.unit
def test_calculate_time_weighted_returns_weighted_periods():
    """Tests TWR calculation with different holding periods, amounts, and a transfer."""

    test_data = pd.DataFrame([
        {'coin_id': 'eth', 'wallet_address': 'wallet_a', 'date': '2024-01-01',
            'usd_balance': 100, 'usd_net_transfers': 100, 'is_imputed': False},  # Initial $100
        {'coin_id': 'eth', 'wallet_address': 'wallet_a', 'date': '2024-02-01',
            'usd_balance': 250, 'usd_net_transfers': 50, 'is_imputed': False},    # Added $50, value up
        {'coin_id': 'eth', 'wallet_address': 'wallet_a', 'date': '2024-10-01',
            'usd_balance': 125, 'usd_net_transfers': 0, 'is_imputed': False}     # Value dropped
    ])

    test_data['date'] = pd.to_datetime(test_data['date'])
    result = wpf.calculate_time_weighted_returns(test_data)

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
    assert result.loc['wallet_a', 'days_held'] == 274
    assert abs(result.loc['wallet_a', 'time_weighted_return'] - expected_twr) < 0.01
    assert abs(result.loc['wallet_a', 'annualized_twr'] - expected_annual) < 0.01
