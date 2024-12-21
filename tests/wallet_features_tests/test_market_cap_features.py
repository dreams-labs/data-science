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
        {'wallet_address': 1, 'volume': 100, 'market_cap_filled': 1000},
        {'wallet_address': 1, 'volume': 200, 'market_cap_filled': 2000},
        {'wallet_address': 1, 'volume': 300, 'market_cap_filled': 3000},

        # Wallet 2: Zero volume case
        {'wallet_address': 2, 'volume': 0, 'market_cap_filled': 1500},
        {'wallet_address': 2, 'volume': 0, 'market_cap_filled': 2500},

        # Wallet 3: Single high volume
        {'wallet_address': 3, 'volume': 1000, 'market_cap_filled': 5000},

        # Wallet 4: Equal volumes
        {'wallet_address': 4, 'volume': 50, 'market_cap_filled': 1000},
        {'wallet_address': 4, 'volume': 50, 'market_cap_filled': 3000},
    ])

    result = wmc.calculate_volume_weighted_market_cap(input_df)

    # Calculate expected values:
    # Wallet 1: (1000*100 + 2000*200 + 3000*300)/(100 + 200 + 300) = 2333.33
    # Wallet 2: Simple average of (1500 + 2500)/2 = 2000
    # Wallet 3: Single value = 5000
    # Wallet 4: Equal weights (1000 + 3000)/2 = 2000
    expected_values = {
        1: 2333.33,
        2: 2000.00,
        3: 5000.00,
        4: 2000.00
    }

    for wallet, expected in expected_values.items():
        assert np.isclose(
            result.loc[wallet, 'volume_wtd_market_cap'],
            expected,
            rtol=1e-2
        ), f"Incorrect weighted market cap for wallet {wallet}"


@pytest.mark.unit
def test_calculate_ending_balance_weighted_market_cap_mixed_scenarios():
    """
    Tests ending balance weighted market cap calculations with different wallet scenarios:
    - Wallet 1: Multiple coins with different balances on final date
    - Wallet 2: Zero balance on final date (should use simple average)
    - Wallet 3: Single coin balance
    - Wallet 4: Equal balances across coins (weighted avg should match simple avg)
    """
    latest_date = pd.Timestamp('2024-01-05')
    earlier_date = pd.Timestamp('2024-01-04')

    input_df = pd.DataFrame([
        # Wallet 1: Multiple coins, varied balances
        {'wallet_address': 1, 'date': latest_date, 'usd_balance': 1000, 'market_cap_filled': 10000},
        {'wallet_address': 1, 'date': latest_date, 'usd_balance': 2000, 'market_cap_filled': 20000},
        {'wallet_address': 1, 'date': latest_date, 'usd_balance': 3000, 'market_cap_filled': 30000},

        # Wallet 2: Zero balance on final date
        {'wallet_address': 2, 'date': latest_date, 'usd_balance': 0, 'market_cap_filled': 15000},
        {'wallet_address': 2, 'date': latest_date, 'usd_balance': 0, 'market_cap_filled': 25000},
        {'wallet_address': 2, 'date': earlier_date, 'usd_balance': 5000, 'market_cap_filled': 20000},

        # Wallet 3: Single coin balance
        {'wallet_address': 3, 'date': latest_date, 'usd_balance': 5000, 'market_cap_filled': 50000},

        # Wallet 4: Equal balances
        {'wallet_address': 4, 'date': latest_date, 'usd_balance': 1000, 'market_cap_filled': 10000},
        {'wallet_address': 4, 'date': latest_date, 'usd_balance': 1000, 'market_cap_filled': 30000},
    ])

    result = wmc.calculate_ending_balance_weighted_market_cap(input_df)

    # Calculate expected values:
    # Wallet 1: (10000*1000 + 20000*2000 + 30000*3000)/(1000 + 2000 + 3000) = 23333.33
    # Wallet 2: Simple average of (15000 + 25000)/2 = 20000 (earlier date ignored)
    # Wallet 3: Single value = 50000
    # Wallet 4: Equal weights (10000 + 30000)/2 = 20000
    expected_portfolio_wtd = {
        1: 23333.33,
        2: 20000.00,
        3: 50000.00,
        4: 20000.00
    }

    expected_ending_balance = {
        1: 6000.00,  # 1000 + 2000 + 3000
        2: 0.00,     # All zero on final date
        3: 5000.00,  # Single balance
        4: 2000.00   # 1000 + 1000
    }

    for wallet, expected_wtd in expected_portfolio_wtd.items():
        assert np.isclose(
            result.loc[wallet, 'portfolio_wtd_market_cap'],
            expected_wtd,
            rtol=1e-2
        ), f"Incorrect weighted market cap for wallet {wallet}"

        assert np.isclose(
            result.loc[wallet, 'ending_portfolio_usd'],
            expected_ending_balance[wallet],
            rtol=1e-2
        ), f"Incorrect ending balance for wallet {wallet}"
