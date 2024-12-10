"""
tests used to audit the files in the data-science/src folder
"""
# pylint: disable=C0302 # over 1000 lines
# pylint: disable=C0413 # import not at top of doc (due to local import)
# pylint: disable=W0612 # unused variables (due to test reusing functions with 2 outputs)
# pylint: disable=W0621 # redefining from outer scope triggering on pytest fixtures
# pylint: disable=W1203 # fstrings in logs
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
import wallet_features.wallet_coin_features as wcf

load_dotenv()
logger = dc.setup_logger()

# ===================================================== #
#                                                       #
#                 U N I T   T E S T S                   #
#                                                       #
# ===================================================== #

# ------------------------------------------------- #
# calculate_timing_features_for_column() unit tests
# ------------------------------------------------- #

@pytest.mark.unit
def test_calculate_timing_features_basic():
    """
    Test basic functionality of calculate_timing_features_for_column with a simple dataset.

    Scenario:
    - Wallet A makes 2 buys and 1 sell
    - Buy 1: $100 when metric = 0.5
    - Buy 2: $200 when metric = 1.0
    - Sell 1: $150 when metric = -0.5

    Expected calculations:
    Buy weighted avg = (100 * 0.5 + 200 * 1.0) / (100 + 200) = 0.833333
    Buy mean = (0.5 + 1.0) / 2 = 0.75
    Sell weighted avg = (-0.5 * 150) / 150 = -0.5
    Sell mean = -0.5
    """
    # Create sample data
    test_df = pd.DataFrame({
        'wallet_address': ['wallet_a'] * 3,
        'usd_net_transfers': [100, 200, -150],
        'test_metric': [0.5, 1.0, -0.5]
    })

    # Calculate features
    result = wcf.calculate_timing_features_for_column(test_df, 'test_metric')

    # Expected values - calculating each component:
    # Buy weighted: (100 * 0.5 + 200 * 1.0) / (100 + 200) = 0.833333
    # Buy mean: (0.5 + 1.0) / 2 = 0.75
    # Sell weighted: (-0.5 * 150) / 150 = -0.5
    # Sell mean: Single value = -0.5
    expected = pd.DataFrame(
        {
            'test_metric_buy_mean': [0.75],
            'test_metric_buy_weighted': [0.833333],
            'test_metric_sell_mean': [-0.5],
            'test_metric_sell_weighted': [-0.5],
        },
        index=pd.Index(['wallet_a'], name='wallet_address')
    )

    # Verify all values match expected
    assert np.allclose(
        result,
        expected,
        equal_nan=True
    ), f"Expected {expected}\nGot {result}"


@pytest.mark.unit
def test_calculate_timing_features_empty_groups():
    """
    Test calculate_timing_features_for_column handles wallets with only buys or only sells.

    Scenario:
    - Wallet A: only has buys
        - Buy 1: $100 when metric = 0.5
        - Buy 2: $200 when metric = 1.0
    - Wallet B: only has sells
        - Sell 1: $150 when metric = -0.5
        - Sell 2: $300 when metric = -1.0

    Expected calculations:
    Wallet A:
        Buy weighted avg = (100 * 0.5 + 200 * 1.0) / (100 + 200) = 0.833333
        Buy mean = (0.5 + 1.0) / 2 = 0.75
        Sell weighted avg = NaN (no sells)
        Sell mean = NaN (no sells)

    Wallet B:
        Buy weighted avg = NaN (no buys)
        Buy mean = NaN (no buys)
        Sell weighted avg = (150 * -0.5 + 300 * -1.0) / (150 + 300) = -0.833333
        Sell mean = (-0.5 + -1.0) / 2 = -0.75
    """
    # Create sample data
    test_df = pd.DataFrame({
        'wallet_address': ['wallet_a', 'wallet_a', 'wallet_b', 'wallet_b'],
        'usd_net_transfers': [100, 200, -150, -300],
        'test_metric': [0.5, 1.0, -0.5, -1.0]
    })

    # Calculate features
    result = wcf.calculate_timing_features_for_column(test_df, 'test_metric')

    # Expected values
    expected = pd.DataFrame(
        {
            'test_metric_buy_mean': [0.75, np.nan],
            'test_metric_buy_weighted': [0.833333, np.nan],
            'test_metric_sell_mean': [np.nan, -0.75],
            'test_metric_sell_weighted': [np.nan, -0.833333],
        },
        index=pd.Index(['wallet_a', 'wallet_b'], name='wallet_address')
    )

    # Verify all values match expected
    assert np.allclose(
        result,
        expected,
        equal_nan=True
    ), f"Expected {expected}\nGot {result}"

@pytest.mark.unit
def test_calculate_timing_features_extreme_values():
    """
    Test calculate_timing_features_for_column handles extreme transaction values correctly.

    Scenario:
    - Wallet A:
        - Buy: Very large transaction ($1B) with very small metric (0.0001)
        - Buy: Very small transaction ($0.01) with large metric (100.0)
        - Sell: Very large negative transaction (-$1B) with zero metric value

    Expected calculations:
    Buy weighted avg:
        (1e9 * 0.0001 + 0.01 * 100.0) / (1e9 + 0.01) ≈ 0.0001
        Note: Small transaction has negligible impact due to size difference

    Buy mean:
        (0.0001 + 100.0) / 2 = 50.00005
        Note: Unweighted mean is heavily influenced by both values

    Sell weighted avg:
        (1e9 * 0) / 1e9 = 0
        Note: Single sell with zero metric value

    Sell mean:
        0 (single value)
    """
    # Create sample data with extreme values
    test_df = pd.DataFrame({
        'wallet_address': ['wallet_a'] * 3,
        'usd_net_transfers': [1e9, 0.01, -1e9],
        'test_metric': [0.0001, 100.0, 0.0]
    })

    # Calculate features
    result = wcf.calculate_timing_features_for_column(test_df, 'test_metric')

    # Expected values
    expected = pd.DataFrame(
        {
            'test_metric_buy_mean': [50.00005],    # Simple average of extremes
            'test_metric_buy_weighted': [0.0001],  # Dominated by large transaction
            'test_metric_sell_mean': [0.0],         # Single zero-value sell
            'test_metric_sell_weighted': [0.0],    # Single zero-value sell
        },
        index=pd.Index(['wallet_a'], name='wallet_address')
    )

    # Verify weighted buy average (requires higher rtol due to extreme value differences)
    assert np.isclose(
        result['test_metric_buy_weighted'].iloc[0],
        expected['test_metric_buy_weighted'].iloc[0],
        rtol=1e-5
    ), "Buy weighted average calculation failed with extreme values"

    # Verify buy mean
    assert np.isclose(
        result['test_metric_buy_mean'].iloc[0],
        expected['test_metric_buy_mean'].iloc[0]
    ), "Buy mean calculation failed with extreme values"

    # Verify sell calculations
    assert np.isclose(
        result['test_metric_sell_weighted'].iloc[0],
        expected['test_metric_sell_weighted'].iloc[0]
    ), "Sell weighted average calculation failed with zero value"

    assert np.isclose(
        result['test_metric_sell_mean'].iloc[0],
        expected['test_metric_sell_mean'].iloc[0]
    ), "Sell mean calculation failed with zero value"
