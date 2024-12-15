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

# ------------------------------------------ #
# add_cash_flow_transfers_logic() unit tests
# ------------------------------------------ #

@pytest.fixture
def sample_profits_df():
    """
    Creates a simple profits DataFrame with two wallet-coin pairs over three dates.

    Structure:
    - Wallet 1: Has balance and transfers on all dates
    - Wallet 2: Only has activity on first and last date
    """
    data = [
        # Wallet 1 activity
        {'coin_id': 'coin1', 'wallet_address': 1, 'date': '2024-01-01',
         'usd_balance': 1000.00, 'usd_net_transfers': 1000.00},
        {'coin_id': 'coin1', 'wallet_address': 1, 'date': '2024-01-02',
         'usd_balance': 1100.00, 'usd_net_transfers': 0.00},
        {'coin_id': 'coin1', 'wallet_address': 1, 'date': '2024-01-03',
         'usd_balance': 1200.00, 'usd_net_transfers': 0.00},

        # Wallet 2 activity
        {'coin_id': 'coin1', 'wallet_address': 2, 'date': '2024-01-01',
         'usd_balance': 500.00, 'usd_net_transfers': 500.00},
        {'coin_id': 'coin1', 'wallet_address': 2, 'date': '2024-01-03',
         'usd_balance': 600.00, 'usd_net_transfers': 0.00}
    ]
    return pd.DataFrame(data)

@pytest.mark.unit
def test_basic_cash_flow_transfers():
    """
    Tests the basic cash flow transfers calculation for a single wallet over multiple dates.

    Logical Steps:
    1. First date (2024-01-01):
        - Initial transfer: 1000.00
        - Starting balance adjustment: -1000.00
        - Net cash_flow_transfers should be 0.00

    2. Middle date (2024-01-02):
        - No transfers, no adjustments
        - cash_flow_transfers should be 0.00

    3. Final date (2024-01-03):
        - No initial transfer
        - Ending balance becomes cash_flow_transfers
        - Should be 1200.00
    """
    # Arrange
    input_data = [
        {'coin_id': 'coin1', 'wallet_address': 1, 'date': '2024-01-01',
         'usd_balance': 1000.00, 'usd_net_transfers': 1000.00},
        {'coin_id': 'coin1', 'wallet_address': 1, 'date': '2024-01-02',
         'usd_balance': 1100.00, 'usd_net_transfers': 0.00},
        {'coin_id': 'coin1', 'wallet_address': 1, 'date': '2024-01-03',
         'usd_balance': 1200.00, 'usd_net_transfers': 0.00}
    ]
    test_df = pd.DataFrame(input_data)

    # Act
    result_df = wtf.add_cash_flow_transfers_logic(test_df)

    # Assert
    expected_cash_flows = [0.00, 0.00, 1200.00]
    assert np.allclose(result_df['cash_flow_transfers'], expected_cash_flows, equal_nan=True)

@pytest.mark.unit
def test_comprehensive_cash_flow_transfers():
    """
    Tests cash_flow_transfers calculations across multiple edge case scenarios.

    Test Wallets:
    1. Single Day Activity:
       - One transaction on period start (2024-01-01)
       - Expected: cash_flow_transfers = final balance (since start = end)

    2. Zero Balance Patterns:
       - Starts with transfer but zero balance
       - Has mid-period exit and re-entry
       - Ends with zero balance
       - Expected: cash_flows match transfers except at start/end dates

    3. Exit/Re-entry Pattern:
       - Normal starting balance and transfers
       - Complete exit (zero balance period)
       - Re-entry with new balance
       - Expected: Captures full cycle of investments/returns

    4. Imputed Forward:
       - All activity before period start
       - Gets imputed to period start
       - Expected: Starting adjustment reflects imputed balance

    Logical Steps for each row's cash_flow_transfers:
    1. Initialize with usd_net_transfers
    2. For start date: Subtract usd_balance (represents initial investment)
    3. For end date: Replace with usd_balance (represents final value)
    """
    # Arrange
    test_data = [
        # Wallet 1: Single Day Activity (period start)
        {'coin_id': 'coin1', 'wallet_address': 1, 'date': '2024-01-01',
         'usd_balance': 1000.00, 'usd_net_transfers': 1000.00, 'is_imputed': False},

        # Wallet 2: Zero Balance Patterns
        {'coin_id': 'coin1', 'wallet_address': 2, 'date': '2024-01-01',
         'usd_balance': 0.00, 'usd_net_transfers': 500.00, 'is_imputed': False},
        {'coin_id': 'coin1', 'wallet_address': 2, 'date': '2024-01-15',
         'usd_balance': 550.00, 'usd_net_transfers': 0.00, 'is_imputed': False},
        {'coin_id': 'coin1', 'wallet_address': 2, 'date': '2024-02-01',
         'usd_balance': 0.00, 'usd_net_transfers': -550.00, 'is_imputed': False},
        {'coin_id': 'coin1', 'wallet_address': 2, 'date': '2024-03-01',
         'usd_balance': 0.00, 'usd_net_transfers': 0.00, 'is_imputed': False},

        # Wallet 3: Exit & Re-entry
        {'coin_id': 'coin1', 'wallet_address': 3, 'date': '2024-01-01',
         'usd_balance': 1000.00, 'usd_net_transfers': 1000.00, 'is_imputed': False},
        {'coin_id': 'coin1', 'wallet_address': 3, 'date': '2024-02-01',
         'usd_balance': 0.00, 'usd_net_transfers': -1100.00, 'is_imputed': False},  # Sold at profit
        {'coin_id': 'coin1', 'wallet_address': 3, 'date': '2024-03-01',
         'usd_balance': 2000.00, 'usd_net_transfers': 2000.00, 'is_imputed': False},

        # Wallet 4: Imputed Forward
        {'coin_id': 'coin1', 'wallet_address': 4, 'date': '2024-01-01',
         'usd_balance': 1500.00, 'usd_net_transfers': 0.00, 'is_imputed': True}
    ]
    test_df = pd.DataFrame(test_data)

    # Act
    result_df = wtf.add_cash_flow_transfers_logic(test_df)

    # Assert
    # For Wallet 1 (Single day activity)
    w1_flows = result_df[result_df['wallet_address'] == 1]['cash_flow_transfers']
    assert np.allclose(w1_flows, [1000.00], equal_nan=True)

    # For Wallet 2 (Zero balance patterns)
    w2_flows = result_df[result_df['wallet_address'] == 2]['cash_flow_transfers']
    expected_w2 = [500.00, 0.00, -550.00, 0.00]  # No balance adjustments since start/end are 0
    assert np.allclose(w2_flows, expected_w2, equal_nan=True)

    # For Wallet 3 (Exit & Re-entry)
    w3_flows = result_df[result_df['wallet_address'] == 3]['cash_flow_transfers']
    expected_w3 = [0.00, -1100.00, 2000.00]  # Start adjusted to 0, end shows final balance
    assert np.allclose(w3_flows, expected_w3, equal_nan=True)

    # For Wallet 4 (Imputed)
    w4_flows = result_df[result_df['wallet_address'] == 4]['cash_flow_transfers']
    expected_w4 = [-1500.00]  # Negative starting balance as it represents initial investment
    assert np.allclose(w4_flows, expected_w4, equal_nan=True)
