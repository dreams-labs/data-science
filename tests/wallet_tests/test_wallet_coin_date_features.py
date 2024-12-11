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
import wallet_features.wallet_coin_date_features as wcdf
from wallet_modeling.wallets_config_manager import WalletsConfig

load_dotenv()
logger = dc.setup_logger()
wallets_config = WalletsConfig.load_from_yaml('../config/wallets_config.yaml')

# ===================================================== #
#                                                       #
#                 U N I T   T E S T S                   #
#                                                       #
# ===================================================== #

# ------------------------------------------ #
# calculate_offsets() unit tests
# ------------------------------------------ #

@pytest.fixture
def basic_market_timing_config():
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

@pytest.mark.unit
def test_successful_offset_calculation(basic_market_timing_config):
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
    result = wcdf.calculate_offsets(df, basic_market_timing_config)

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
def test_insufficient_records(basic_market_timing_config):
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

    result = wcdf.calculate_offsets(df, basic_market_timing_config)

    # Verify that ETH's offset values are NaN where there isn't enough data
    eth_data = result[result['coin_id'] == 'ETH']
    assert pd.isna(eth_data['price_rsi_14_lead_3']).all()  # 3-step offset should be all NaN
    assert pd.isna(eth_data['price_rsi_14_lead_2']).all()  # 2-step offset should be all NaN

@pytest.mark.unit
def test_missing_column_in_df():
    """
    Test handling of configuration with column that doesn't exist in DataFrame.

    Steps:
    1. Create config with non-existent column
    2. Verify appropriate exception is raised
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

    with pytest.raises(wcdf.FeatureConfigError) as exc_info:
        wcdf.calculate_offsets(df, invalid_config)

    assert "Column 'price_rsi_4' not found in DataFrame" in str(exc_info.value)

@pytest.mark.unit
def test_relative_changes_calculation(basic_market_timing_config):
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

    result_df,relative_change_columns = wcdf.calculate_relative_changes(df, basic_market_timing_config)

    # Verify relative change calculations
    # For price_rsi_14_vs_lead_2: ((lead_2 - base) / base) * 100
    expected_rsi_change = (30 - 10) / 10  # 200%
    assert np.isclose(
        result_df['price_rsi_14_vs_lead_2'].iloc[0],
        expected_rsi_change,
        equal_nan=True
    )

    # Verify that the relative_change_columns list is correct
    expected_relative_change_columns = [
        'price_rsi_14_vs_lead_2',
        'price_rsi_14_vs_lead_3',
        'price_sma_7_vs_lead_1'
    ]
    assert relative_change_columns == expected_relative_change_columns

    # Verify column retention
    # price_rsi_14 should be dropped (retain_base_columns: False)
    assert 'price_rsi_14' not in result_df.columns
    assert 'price_rsi_14_lead_2' not in result_df.columns

    # price_sma_7 should be retained (retain_base_columns: True)
    assert 'price_sma_7' in result_df.columns
    assert 'price_sma_7_lead_1' in result_df.columns
