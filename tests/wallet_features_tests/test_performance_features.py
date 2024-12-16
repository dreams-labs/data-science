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

@pytest.mark.unit
def test_calculate_time_weighted_returns_imputed_case():
    """Tests TWR calculation for a wallet with only imputed balances."""

    # Setup test data
    test_data = pd.DataFrame([
        {'coin_id': 'btc', 'wallet_address': 'w05_only_imputed', 'date': '2024-01-01',
         'usd_balance': 50, 'usd_net_transfers': 0, 'is_imputed': True},
        {'coin_id': 'btc', 'wallet_address': 'w05_only_imputed', 'date': '2024-10-01',
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
    assert abs(result.loc['w05_only_imputed', 'time_weighted_return'] - expected_twr) < 0.001
    assert result.loc['w05_only_imputed', 'days_held'] == expected_days
    assert abs(result.loc['w05_only_imputed', 'annualized_twr'] - expected_annual) < 0.001


@pytest.mark.unit
def test_calculate_time_weighted_returns_multiple_coins():
    """Tests TWR calculation for wallet with BTC and ETH positions over 2024."""

    test_data = pd.DataFrame([
        # BTC transactions
        {'coin_id': 'btc', 'wallet_address': 'w01_multiple_coins', 'date': '2024-01-01',
         'usd_balance': 100, 'usd_net_transfers': 100, 'is_imputed': False},
        {'coin_id': 'btc', 'wallet_address': 'w01_multiple_coins', 'date': '2024-05-01',
         'usd_balance': 120, 'usd_net_transfers': 50, 'is_imputed': False},
        {'coin_id': 'btc', 'wallet_address': 'w01_multiple_coins', 'date': '2024-10-01',
         'usd_balance': 180, 'usd_net_transfers': 0, 'is_imputed': True},

        # ETH transactions
        {'coin_id': 'eth', 'wallet_address': 'w01_multiple_coins', 'date': '2024-01-01',
         'usd_balance': 200, 'usd_net_transfers': 200, 'is_imputed': False},
        {'coin_id': 'eth', 'wallet_address': 'w01_multiple_coins', 'date': '2024-05-01',
         'usd_balance': 300, 'usd_net_transfers': 50, 'is_imputed': False},
        {'coin_id': 'eth', 'wallet_address': 'w01_multiple_coins', 'date': '2024-10-01',
         'usd_balance': 280, 'usd_net_transfers': 0, 'is_imputed': True},
    ])

    test_data['date'] = pd.to_datetime(test_data['date'])
    result = wpf.calculate_time_weighted_returns(test_data)

    # Let's calculate expected returns:
    # BTC:
    # Jan-May: 100 -> 120-50 = -30% over 121 days
    # May-Oct: 120 -> 180 = 50% over 153 days

    # ETH:
    # Jan-May: 200 -> 300-50 = 25% over 121 days
    # May-Oct: 300 -> 280 = -6.7% over 153 days

    # Total days = 274 (Jan 1 to Oct 1)

    assert result.loc['w01_multiple_coins', 'days_held'] == 274

    # The weighted return should reflect both coins' performance weighted by time
    # Let's add some reasonable bounds based on the performance
    time_weighted_return = result.loc['w01_multiple_coins', 'time_weighted_return']
    assert 0.1 < time_weighted_return < 0.3  # We expect positive returns but moderated by the mixed performance

    # Check annualization
    annual_return = result.loc['w01_multiple_coins', 'annualized_twr']
    assert 0.15 < annual_return < 0.45  # Annualized should be higher due to partial year scaling

@pytest.mark.unit
def test_calculate_time_weighted_returns_memecoin_winner():
    """Tests TWR calculation for wallet with large gains and withdrawals followed by losses."""
    test_data = pd.DataFrame([
        {'coin_id': 'pepe', 'wallet_address': 'w09_memecoin_winner', 'date': '2024-01-01',
            'usd_balance': 100, 'usd_net_transfers': 100, 'is_imputed': False},
        {'coin_id': 'pepe', 'wallet_address': 'w09_memecoin_winner', 'date': '2024-03-01',
            'usd_balance': 250, 'usd_net_transfers': -500, 'is_imputed': False},
        {'coin_id': 'pepe', 'wallet_address': 'w09_memecoin_winner', 'date': '2024-05-01',
            'usd_balance': 50, 'usd_net_transfers': -100, 'is_imputed': False},
        {'coin_id': 'pepe', 'wallet_address': 'w09_memecoin_winner', 'date': '2024-10-01',
            'usd_balance': 10, 'usd_net_transfers': 0, 'is_imputed': True}
    ])

    test_data['date'] = pd.to_datetime(test_data['date'])
    result = wpf.calculate_time_weighted_returns(test_data)

    # Calculate expected returns:
    # Jan-Mar: 100 -> 250+500 = 650% over 60 days
    # Mar-May: 250 -> 50+100 = -60% over 61 days
    # May-Oct: 50 -> 10 = -80% over 153 days
    # Total days = 274 (Jan 1 to Oct 1)

    assert result.loc['w09_memecoin_winner', 'days_held'] == 274

    # Despite large withdrawals, returns should reflect high initial gains
    time_weighted_return = result.loc['w09_memecoin_winner', 'time_weighted_return']
    assert 0.5 < time_weighted_return < 1.0  # Expecting significant positive return due to early gains

    # Check annualization
    annual_return = result.loc['w09_memecoin_winner', 'annualized_twr']
    assert 1.0 < annual_return < 3.0  # Higher due to annualization of partial year

@pytest.mark.unit
def test_calculate_time_weighted_returns_memecoin_loser():
    """Tests TWR calculation for wallet with complete loss scenario."""

    test_data = pd.DataFrame([
        {'coin_id': 'bome', 'wallet_address': 'w10_memecoin_loser', 'date': '2024-03-01',
            'usd_balance': 250, 'usd_net_transfers': 250, 'is_imputed': False},
        {'coin_id': 'bome', 'wallet_address': 'w10_memecoin_loser', 'date': '2024-10-01',
            'usd_balance': 0, 'usd_net_transfers': -20, 'is_imputed': False}
    ])

    test_data['date'] = pd.to_datetime(test_data['date'])
    result = wpf.calculate_time_weighted_returns(test_data)

    # Calculate expected returns:
    # Mar-Oct: 250 -> 0+20 = -92% over 214 days
    # Total days = 214 (Mar 1 to Oct 1)

    assert result.loc['w10_memecoin_loser', 'days_held'] == 214

    # Should show significant negative returns
    time_weighted_return = result.loc['w10_memecoin_loser', 'time_weighted_return']
    assert -1.0 < time_weighted_return < -0.8  # Expecting ~-90% return

    # Check annualization
    annual_return = result.loc['w10_memecoin_loser', 'annualized_twr']
    assert -1.0 < annual_return < -0.9  # Heavy losses should persist in annualization

@pytest.mark.unit
def test_calculate_time_weighted_returns_weighted_periods():
    """Tests TWR calculation with different holding periods, amounts, and a transfer."""

    test_data = pd.DataFrame([
        {'coin_id': 'btc', 'wallet_address': 'w03_weighted', 'date': '2024-01-01',
            'usd_balance': 100, 'usd_net_transfers': 100, 'is_imputed': False},  # Initial $100
        {'coin_id': 'btc', 'wallet_address': 'w03_weighted', 'date': '2024-02-01',
            'usd_balance': 250, 'usd_net_transfers': 50, 'is_imputed': False},    # Added $50, value up
        {'coin_id': 'btc', 'wallet_address': 'w03_weighted', 'date': '2024-10-01',
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
    assert result.loc['w03_weighted', 'days_held'] == 274
    assert abs(result.loc['w03_weighted', 'time_weighted_return'] - expected_twr) < 0.01
    assert abs(result.loc['w03_weighted', 'annualized_twr'] - expected_annual) < 0.01
