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
import wallet_features.market_timing_features as wmt
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
# calculate_offsets() unit tests
# ------------------------------------------ #

@pytest.fixture
def basic_market_timing_features_config():
    """
    Fixture providing a basic wallet features configuration with market timing offsets.

    Returns:
        Dict containing market timing configuration with RSI and SMA offset values
    """
    return {
        'market_timing': {
            'offsets': {
                'price_rsi_14': {
                    'offsets': [2, 3],
                    'retain_base_columns': False
                },
                'price_sma_7': {
                    'offsets': [1],
                    'retain_base_columns': True
                }
            }
        }
    }

@pytest.fixture
def mock_wallets_features_config(monkeypatch, basic_market_timing_features_config):
    """Mock the wallets_features_config at the module level"""
    monkeypatch.setattr(wmt, 'wallets_features_config', basic_market_timing_features_config)

@pytest.fixture
def basic_market_timing_metrics_config():
    """
    Fixture providing a basic wallet features configuration with market timing offsets.

    Returns:
        Dict containing market timing configuration with RSI and SMA offset values
    """
    return {
        'time_series': {
            'market_data': {
                'price': {
                    'indicators': {
                        'sma': {
                            'parameters': {
                                'window': [7]
                            }
                        },
                        'rsi': {
                            'parameters': {
                                'window': [14]
                            }
                        }
                    }
                }
            }
        }
    }

@pytest.fixture
def mock_wallets_metrics_config(monkeypatch, basic_market_timing_metrics_config):
    """Mock the wallets_metrics_config at the module level"""
    monkeypatch.setattr(wmt, 'wallets_metrics_config', basic_market_timing_metrics_config)


# Add these to your existing imports

@pytest.mark.unit
def test_successful_offset_calculation(mock_wallets_features_config,mock_wallets_metrics_config):
    """
    Test successful calculation of offsets when DataFrame has sufficient records.

    Steps:
    1. Create sample DataFrame with two coins and sufficient records
    2. Apply offset calculations
    3. Verify all expected offset columns are created
    4. Verify offset values are calculated correctly
    """
    # Create sample data with sufficient records
    df = pd.DataFrame({
        'coin_id': ['BTC', 'BTC', 'BTC', 'BTC', 'ETH', 'ETH', 'ETH', 'ETH'],
        'price_rsi_14': [10, 20, 30, 40, 15, 25, 35, 45],
        'price_sma_7': [100, 200, 300, 400, 150, 250, 350, 450]
    })

    # Calculate offsets
    result = wmt.calculate_offsets(df)

    # Verify all expected columns exist
    expected_columns = [
        'coin_id', 'price_rsi_14', 'price_sma_7',
        'price_rsi_14_lead_2', 'price_rsi_14_lead_3',
        'price_sma_7_lead_1'
    ]
    assert all(col in result.columns for col in expected_columns)

    # Verify offset calculations for BTC
    # For price_rsi_14_lead_2, each value should match the value 2 positions ahead
    assert np.allclose(
        result[result['coin_id'] == 'BTC']['price_rsi_14_lead_2'].iloc[0:2].values,
        np.array([30, 40]),
        equal_nan=True
    )

    # For price_sma_7_lead_1, each value should match the value 1 position ahead
    assert np.allclose(
        result[result['coin_id'] == 'BTC']['price_sma_7_lead_1'].iloc[0:3].values,
        np.array([200, 300, 400]),
        equal_nan=True
    )

@pytest.mark.unit
def test_insufficient_records(mock_wallets_features_config,mock_wallets_metrics_config):
    """
    Test offset calculation when one coin has insufficient records.

    Steps:
    1. Create sample DataFrame where ETH has fewer records than offset window
    2. Apply offset calculations
    3. Verify NaN values are properly placed for insufficient data
    """
    # Create sample data where ETH has insufficient records
    df = pd.DataFrame({
        'coin_id': ['BTC', 'BTC', 'BTC', 'BTC', 'ETH', 'ETH'],  # ETH has fewer records
        'price_rsi_14': [10, 20, 30, 40, 15, 25],
        'price_sma_7': [100, 200, 300, 400, 150, 250]
    })

    result = wmt.calculate_offsets(df)

    # Verify that ETH's offset values are NaN where there isn't enough data
    eth_data = result[result['coin_id'] == 'ETH']
    assert pd.isna(eth_data['price_rsi_14_lead_3']).all()  # 3-step offset should be all NaN
    assert pd.isna(eth_data['price_rsi_14_lead_2']).all()  # 2-step offset should be all NaN

@pytest.mark.unit
def test_missing_column_in_df(monkeypatch):
    """
    Test handling of configuration with column that doesn't exist in DataFrame.
    """
    df = pd.DataFrame({
        'coin_id': ['BTC', 'ETH'],
        'price_rsi_3': [10, 15]  # Only RSI-3 exists, not RSI-4
    })

    invalid_config = {
        'market_timing': {
            'offsets': {
                'price_rsi_4': {  # This column doesn't exist
                    'offsets': [1],
                    'retain_base_columns': True
                }
            }
        }
    }

    # Mock the wallets_features_config with invalid config
    monkeypatch.setattr(wmt, 'wallets_features_config', invalid_config)

    with pytest.raises(wmt.FeatureConfigError) as exc_info:
        wmt.calculate_offsets(df)

    assert "Column 'price_rsi_4' not found in DataFrame" in str(exc_info.value)


@pytest.mark.unit
def test_relative_changes_calculation(mock_wallets_features_config,mock_wallets_metrics_config):
    """
    Test calculation of relative changes between base and offset columns.

    Steps:
    1. Create sample DataFrame with offset columns
    2. Calculate relative changes
    3. Verify changes are calculated correctly
    4. Verify column retention based on configuration
    """
    # Create sample data with offset columns
    df = pd.DataFrame({
        'coin_id': ['BTC', 'BTC', 'BTC', 'ETH', 'ETH', 'ETH'],
        'price_rsi_14': [10, 20, 30, 15, 25, 35],
        'price_rsi_14_lead_2': [30, 40, np.nan, 35, 45, np.nan],
        'price_rsi_14_lead_3': [40, np.nan, np.nan, 45, np.nan, np.nan],
        'price_sma_7': [100, 200, 300, 150, 250, 350],
        'price_sma_7_lead_1': [200, 300, np.nan, 250, 350, np.nan]
    })

    result_df,relative_change_columns = wmt.calculate_relative_changes(df)

    # Verify relative change calculations
    # For price_rsi_14_vs_lead_2: ((lead_2 - base) / base) * 100
    expected_rsi_change = (30 - 10) / 10  # 200%
    assert np.isclose(
        result_df['price_rsi_14/lead_2'].iloc[0],
        expected_rsi_change,
        equal_nan=True
    )

    # Verify that the relative_change_columns list is correct
    expected_relative_change_columns = [
        'price_rsi_14/lead_2',
        'price_rsi_14/lead_3',
        'price_sma_7/lead_1'
    ]
    assert relative_change_columns == expected_relative_change_columns

    # Verify column retention
    # price_rsi_14 should be dropped (retain_base_columns: False)
    assert 'price_rsi_14' not in result_df.columns
    assert 'price_rsi_14_lead_2' not in result_df.columns

    # price_sma_7 should be retained (retain_base_columns: True)
    assert 'price_sma_7' in result_df.columns
    assert 'price_sma_7_lead_1' in result_df.columns



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
    result = wmt.calculate_timing_features_for_column(test_df, 'test_metric')

    # Expected values - calculating each component:
    # Buy weighted: (100 * 0.5 + 200 * 1.0) / (100 + 200) = 0.833333
    # Buy mean: (0.5 + 1.0) / 2 = 0.75
    # Sell weighted: (-0.5 * 150) / 150 = -0.5
    # Sell mean: Single value = -0.5
    expected = pd.DataFrame(
        {
            'test_metric/buy_mean': [0.75],
            'test_metric/sell_mean': [-0.5],
            'test_metric/buy_weighted': [0.833333],
            'test_metric/sell_weighted': [-0.5],
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
    result = wmt.calculate_timing_features_for_column(test_df, 'test_metric')

    # Expected values
    expected = pd.DataFrame(
        {
            'test_metric/buy_mean': [0.75, np.nan],
            'test_metric/sell_mean': [np.nan, -0.75],
            'test_metric/buy_weighted': [0.833333, np.nan],
            'test_metric/sell_weighted': [np.nan, -0.833333],
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
    result = wmt.calculate_timing_features_for_column(test_df, 'test_metric')

    # Expected values
    expected = pd.DataFrame(
        {
            'test_metric/buy_mean': [50.00005],    # Simple average of extremes
            'test_metric/sell_mean': [0.0],         # Single zero-value sell
            'test_metric/buy_weighted': [0.0001],  # Dominated by large transaction
            'test_metric/sell_weighted': [0.0],    # Single zero-value sell
        },
        index=pd.Index(['wallet_a'], name='wallet_address')
    )

    # Verify weighted buy average (requires higher rtol due to extreme value differences)
    assert np.isclose(
        result['test_metric/buy_weighted'].iloc[0],
        expected['test_metric/buy_weighted'].iloc[0],
        rtol=1e-5
    ), "Buy weighted average calculation failed with extreme values"

    # Verify buy mean
    assert np.isclose(
        result['test_metric/buy_mean'].iloc[0],
        expected['test_metric/buy_mean'].iloc[0]
    ), "Buy mean calculation failed with extreme values"

    # Verify sell calculations
    assert np.isclose(
        result['test_metric/sell_weighted'].iloc[0],
        expected['test_metric/sell_weighted'].iloc[0]
    ), "Sell weighted average calculation failed with zero value"

    assert np.isclose(
        result['test_metric/sell_mean'].iloc[0],
        expected['test_metric/sell_mean'].iloc[0]
    ), "Sell mean calculation failed with zero value"
