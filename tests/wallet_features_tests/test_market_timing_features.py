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

@pytest.fixture
def mock_wallets_config(monkeypatch):
    """Mock the wallets_config at the module level with simplified offsets"""
    test_config = {
        'features': {
            'market_timing_offsets': [1, -1, 3, -3],
            'market_timing_offset_winsorization': 0.03  # Adding standard winsorization
        },
        'data_cleaning': {
            'usd_materiality': 100  # Adding materiality threshold
        }
    }
    monkeypatch.setattr(wmt, 'wallets_config', test_config)

@pytest.fixture
def indicator_columns(basic_market_timing_metrics_config):
    """Indicator column list based on demo config"""
    indicators_config = basic_market_timing_metrics_config['time_series']['market_data']
    return wmt.identify_indicator_columns(indicators_config)


# Add these to your existing imports

@pytest.mark.unit
def test_successful_offset_calculation(mock_wallets_config, indicator_columns):
    """Test offset calculations with simplified offset windows"""
    df = pd.DataFrame({
        'coin_id': ['BTC', 'BTC', 'BTC', 'BTC', 'ETH', 'ETH', 'ETH', 'ETH'],
        'price_rsi_14': [10, 20, 30, 40, 15, 25, 35, 45],
        'price_sma_7': [100, 200, 300, 400, 150, 250, 350, 450]
    })

    result = wmt.calculate_offsets(df, indicator_columns)

    # Verify expected columns exist
    expected_columns = [
        'coin_id', 'price_rsi_14', 'price_sma_7',
        'price_rsi_14_lead_1', 'price_rsi_14_lag_1',
        'price_rsi_14_lead_3', 'price_rsi_14_lag_3',
        'price_sma_7_lead_1', 'price_sma_7_lag_1',
        'price_sma_7_lead_3', 'price_sma_7_lag_3'
    ]
    assert all(col in result.columns for col in expected_columns)

    # Verify lead/lag calculations for BTC
    btc_data = result[result['coin_id'] == 'BTC']
    # Check lead 1
    assert np.allclose(
        btc_data['price_rsi_14_lead_1'].iloc[0:3].values,
        np.array([20, 30, 40]),
        equal_nan=True
    )
    # Check lag 1
    assert np.allclose(
        btc_data['price_rsi_14_lag_1'].iloc[1:4].values,
        np.array([10, 20, 30]),
        equal_nan=True
    )



@pytest.mark.unit
def test_insufficient_records(mock_wallets_config, indicator_columns):
    """
    Test offset calculation when coins have insufficient records for all windows.
    BTC has 4 records, ETH has 2 records.
    """
    df = pd.DataFrame({
        'coin_id': [
            # BTC has 4 records
            'BTC', 'BTC', 'BTC', 'BTC',
            # ETH only has 2 records
            'ETH', 'ETH'
        ],
        'date': pd.date_range('2024-01-01', periods=4).tolist() +
               pd.date_range('2024-01-01', periods=2).tolist(),
        'price_rsi_14': [
            # BTC values
            10, 20, 30, 40,
            # ETH values
            15, 25
        ],
        'price_sma_7': [
            # BTC values
            100, 200, 300, 400,
            # ETH values
            150, 250
        ]
    })

    result = wmt.calculate_offsets(df, indicator_columns)

    # Test BTC results
    btc_data = result[result['coin_id'] == 'BTC']

    # Lead 1 assertions - should have values except last row
    assert pd.isna(btc_data['price_rsi_14_lead_1'].iloc[-1])
    assert not pd.isna(btc_data['price_rsi_14_lead_1'].iloc[:-1]).any()

    # Lead 3 assertions - only first row can see 3 ahead
    assert not pd.isna(btc_data['price_rsi_14_lead_3'].iloc[0])  # Can see to index 3
    assert pd.isna(btc_data['price_rsi_14_lead_3'].iloc[1:]).all()  # Rest are NaN

    # Lag 1 assertions - should have values except first row
    assert pd.isna(btc_data['price_rsi_14_lag_1'].iloc[0])
    assert not pd.isna(btc_data['price_rsi_14_lag_1'].iloc[1:]).any()

    # Test ETH results (shorter series)
    eth_data = result[result['coin_id'] == 'ETH']

    # Only 2 rows, so most things should be NaN
    assert pd.isna(eth_data['price_rsi_14_lead_3']).all()  # Too short for 3 step
    assert pd.isna(eth_data['price_rsi_14_lag_3']).all()   # Too short for 3 step
    assert pd.isna(eth_data['price_rsi_14_lead_1'].iloc[-1])  # Last row can't look ahead
    assert pd.isna(eth_data['price_rsi_14_lag_1'].iloc[0])    # First row can't look back



@pytest.mark.unit
def test_relative_changes_calculation(mock_wallets_config, indicator_columns):
    """Test relative change calculations with simplified offset windows"""
    df = pd.DataFrame({
        'coin_id': [
            # BTC data points
            'BTC', 'BTC', 'BTC', 'BTC', 'BTC', 'BTC', 'BTC', 'BTC', 'BTC',
            # ETH data points
            'ETH', 'ETH', 'ETH', 'ETH', 'ETH', 'ETH', 'ETH', 'ETH', 'ETH'
        ],
        'date': pd.date_range('2024-01-01', periods=9).tolist() * 2,
        'price_rsi_14': [
            # BTC values increasing by 10
            10, 20, 30, 40, 50, 60, 70, 80, 90,
            # ETH values increasing by 10, starting at 15
            15, 25, 35, 45, 55, 65, 75, 85, 95
        ],
        'price_sma_7': [
            # BTC values increasing by 100
            100, 200, 300, 400, 500, 600, 700, 800, 900,
            # ETH values increasing by 100, starting at 150
            150, 250, 350, 450, 550, 650, 750, 850, 950
        ]
    })

    # Now rows 4-5 should have complete lead/lag windows for both coins
    df_with_offsets = wmt.calculate_offsets(df, indicator_columns)

    # Then calculate relative changes
    result_df, relative_change_columns = wmt.calculate_relative_changes(df_with_offsets, indicator_columns)

    # Test calculations for BTC data
    btc_data = result_df[result_df['coin_id'] == 'BTC']

    # Lead 1: (next_value - current_value) / current_value
    expected_lead_1 = (20 - 10) / 10  # Should be 1.0
    assert np.isclose(
        btc_data['price_rsi_14/lead_1'].iloc[0],
        expected_lead_1,
        equal_nan=True
    )

    # Lead 3: (value_3_ahead - current_value) / current_value
    expected_lead_3 = (40 - 10) / 10  # Should be 3.0
    assert np.isclose(
        btc_data['price_rsi_14/lead_3'].iloc[0],
        expected_lead_3,
        equal_nan=True
    )

    # Verify NaN patterns at edges
    assert pd.isna(btc_data['price_rsi_14/lead_1'].iloc[-1])  # No next value available
    assert pd.isna(btc_data['price_rsi_14/lead_3'].iloc[-3:]).all()  # No 3-day ahead values
    assert pd.isna(btc_data['price_rsi_14/lag_1'].iloc[0])  # No previous value
    assert pd.isna(btc_data['price_rsi_14/lag_3'].iloc[:3]).all()  # No 3-day prior values

    # Verify expected columns exist
    expected_change_columns = [
        'price_rsi_14/lead_1', 'price_rsi_14/lag_1',
        'price_rsi_14/lead_3', 'price_rsi_14/lag_3',
        'price_sma_7/lead_1', 'price_sma_7/lag_1',
        'price_sma_7/lead_3', 'price_sma_7/lag_3'
    ]
    assert sorted(relative_change_columns) == sorted(expected_change_columns)


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
    test_df['abs_net_transfers'] = test_df['usd_net_transfers'].abs()
    test_df['transaction_side'] = np.where(test_df['usd_net_transfers'] > 0, 'buy', 'sell')


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
    test_df['abs_net_transfers'] = test_df['usd_net_transfers'].abs()
    test_df['transaction_side'] = np.where(test_df['usd_net_transfers'] > 0, 'buy', 'sell')

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
    test_df['abs_net_transfers'] = test_df['usd_net_transfers'].abs()
    test_df['transaction_side'] = np.where(test_df['usd_net_transfers'] > 0, 'buy', 'sell')

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
