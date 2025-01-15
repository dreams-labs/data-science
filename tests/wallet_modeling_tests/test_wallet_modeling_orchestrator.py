# pylint: disable=wrong-import-position  # import not at top of doc (due to local import)
# pylint: disable=redefined-outer-name  # redefining from outer scope triggering on pytest fixtures
# pylint: disable=unused-argument
# pyright: reportMissingModuleSource=false

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# pyright: reportMissingImports=false
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
import wallet_modeling.wallet_modeling_orchestrator as wmo



# ===================================================== #
#                                                       #
#                 U N I T   T E S T S                   #
#                                                       #
# ===================================================== #


# ------------------------------------------------ #
# hybridize_wallet_address() &
# dehybridize_wallet_address() unit tests
# ------------------------------------------------ #


@pytest.mark.unit
def test_wallet_coin_hybridization():
    """
    Tests the hybridization and dehybridization of wallet-coin pairs.

    Test data structure:
    Wallet 1001: holds BTC, ETH, SOL (tests single wallet, multiple coins)
    Wallet 1002: holds BTC, ETH (tests multiple wallets holding same coins)
    Wallet 1003: holds BTC (completes many-to-many relationship test)

    Expected behavior validation:
    1. Hybridization should create unique integers for each wallet-coin pair
    2. All 6 combinations should map to different integers
    3. Dehybridization should perfectly restore original values
    4. Non-key columns should remain unchanged
    5. Data types should be preserved
    """
    # Create test data with many-to-many relationships
    test_df = pd.DataFrame({
        'wallet_address': [1001, 1001, 1001, 1002, 1002, 1003],
        'coin_id': ['btc', 'eth', 'sol', 'btc', 'eth', 'btc'],
        'value': [100, 200, 300, 400, 500, 600]
    })
    original_df = test_df.copy()

    # Test hybridization
    hybrid_df, hybrid_map = wmo.hybridize_wallet_address(test_df)

    # Validate hybrid keys
    assert len(hybrid_map) == 6, "Should create exactly 6 unique mappings"
    assert hybrid_df['wallet_address'].nunique() == 6, "Should have 6 unique hybrid keys"
    assert all(isinstance(x, (int, np.integer)) for x in hybrid_df['wallet_address']), \
        "All hybrid keys should be integers"

    # Test value preservation
    assert np.allclose(hybrid_df['value'], original_df['value'], equal_nan=True), \
        "Non-key columns should remain unchanged"

    # Test dehybridization
    restored_df = wmo.dehybridize_wallet_address(hybrid_df, hybrid_map)

    # Validate restoration
    assert np.allclose(restored_df['wallet_address'], original_df['wallet_address'], equal_nan=True), \
        "Wallet addresses should be perfectly restored"
    assert np.array_equal(restored_df['coin_id'], original_df['coin_id']), \
        "Coin IDs should be perfectly restored"
    assert np.allclose(restored_df['value'], original_df['value'], equal_nan=True), \
        "Values should remain unchanged through entire process"


@pytest.mark.unit
def test_wallet_coin_hybridization_with_temporal_data():
    """
    Tests hybridization with time series data for wallet-coin pairs.

    Test data structure:
    - Wallet 1001: trades BTC on 3 dates, ETH on 2 dates
    - Wallet 1002: trades BTC on 2 dates, SOL on 1 date
    - Wallet 1003: trades ETH on 2 dates

    This creates a three-way relationship:
    - Wallets can hold multiple coins
    - Coins can be held by multiple wallets
    - Each wallet-coin pair can have activity on multiple dates

    Expected behavior validation:
    1. Each wallet-coin pair should map to a single hybrid key regardless of date
    2. Multiple dates for same wallet-coin pair should share same hybrid key
    3. Time series data and transfer values should be preserved exactly
    4. Dehybridization should restore original wallet-coin pairs across all dates
    """
    # Create test data with temporal relationships
    test_df = pd.DataFrame({
        'wallet_address': [
            # Wallet 1001 activity
            1001, 1001, 1001,  # BTC on 3 dates
            1001, 1001,        # ETH on 2 dates
            # Wallet 1002 activity
            1002, 1002,        # BTC on 2 dates
            1002,              # SOL on 1 date
            # Wallet 1003 activity
            1003, 1003         # ETH on 2 dates
        ],
        'coin_id': [
            'btc', 'btc', 'btc',
            'eth', 'eth',
            'btc', 'btc',
            'sol',
            'eth', 'eth'
        ],
        'date': [
            '2024-01-01', '2024-01-15', '2024-01-30',  # 1001-BTC dates
            '2024-01-10', '2024-01-25',                # 1001-ETH dates
            '2024-01-05', '2024-01-20',                # 1002-BTC dates
            '2024-01-15',                              # 1002-SOL date
            '2024-01-01', '2024-01-30'                 # 1003-ETH dates
        ],
        'usd_net_transfers': [100, -50, 75, 200, -100, 300, -150, 50, 400, -200]
    })
    original_df = test_df.copy()

    # Test hybridization
    hybrid_df, hybrid_map = wmo.hybridize_wallet_address(test_df)

    # Validate unique wallet-coin pair mappings
    unique_wallet_coins = len(test_df.groupby(['wallet_address', 'coin_id']))
    assert len(hybrid_map) == unique_wallet_coins, \
        "Should create exactly one mapping per unique wallet-coin pair"

    # Verify same hybrid key used across dates
    hybrid_keys_by_wallet_coin = hybrid_df.groupby(['coin_id'])['wallet_address'].nunique()
    original_wallets_by_coin = test_df.groupby(['coin_id'])['wallet_address'].nunique()
    assert np.array_equal(hybrid_keys_by_wallet_coin, original_wallets_by_coin), \
        "Should maintain same number of unique wallets per coin"

    # Test dehybridization
    restored_df = wmo.dehybridize_wallet_address(hybrid_df, hybrid_map)

    # Validate complete restoration
    assert np.allclose(restored_df['wallet_address'], original_df['wallet_address'], equal_nan=True), \
        "Wallet addresses should be perfectly restored"
    assert np.array_equal(restored_df['coin_id'], original_df['coin_id']), \
        "Coin IDs should be perfectly restored"
    assert np.array_equal(restored_df['date'], original_df['date']), \
        "Dates should remain unchanged"
    assert np.allclose(restored_df['usd_net_transfers'], original_df['usd_net_transfers'],
                      equal_nan=True), \
        "Transfer values should remain unchanged"
