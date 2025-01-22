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
import wallet_features.performance_features as wpf
import utils as u
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


def test_profits_features_calculation():
    """Verify that crypto_net_gain and crypto_net_flows are correctly
    calculated and preserved from input data"""
    mock_input = pd.DataFrame({
        'crypto_net_gain': [100, -50],
        'crypto_net_flows': [80, -30],
        'crypto_inflows': [1000, 500],
        'crypto_outflows': [920, 530]
    })
    result = wpf.calculate_profits_features(mock_input)
    assert np.allclose(result['crypto_net_gain'], mock_input['crypto_net_gain'])
    assert np.allclose(result['crypto_net_flows'], mock_input['crypto_net_flows'])


def test_balance_features_core_metrics():
    """Verify that core balance metrics are preserved and non-negative"""
    mock_input = pd.DataFrame({
        'max_investment': [1000, 2000],
        'time_weighted_balance': [800, 1500],
        'active_time_weighted_balance': [900, 1800]
    })
    result = wpf.calculate_balance_features(mock_input)
    assert (result >= 0).all().all()  # All balance metrics should be non-negative
    assert np.allclose(result['max_investment'], mock_input['max_investment'])


@pytest.fixture
def sample_performance_features_df():
    """
    Fixture to provide a sample DataFrame for testing ratio calculations.
    Includes profits and balance metrics for multiple wallets,
    including one with all 0 values and another with losses.
    """
    data = {
        'profits_total_gain': [100, 200, 300, 0, -50],
        'profits_realized_gain': [50, 150, 250, 0, -30],
        'profits_unrealized_gain': [50, 50, 50, 0, -20],
        'balance_max_investment': [1000, 2000, 3000, 0, 500],
        'balance_time_weighted_balance': [500, 1500, 2500, 0, 300],
        'balance_active_time_weighted_balance': [400, 1200, 2200, 0, 250],
    }
    index = ['wallet1', 'wallet2', 'wallet3', 'wallet4_zero', 'wallet5_loss']
    return pd.DataFrame(data, index=index)


@pytest.mark.unit
def test_ratio_calculation(sample_performance_features_df):
    """
    Test to validate the correctness of ratios calculated by wpf.calculte_performanc_ratios.
    Steps:
    1. Verify ratios for each profits column divided by each balance column.
    2. Check for division by zero handling (wallet4_zero).
    3. Check handling of losses (wallet5_loss).
    4. Assert output DataFrame matches expected results, allowing for minor precision differences.
    """
    # Generate the ratio DataFrame
    ratio_df = wpf.calculate_performance_ratios(sample_performance_features_df)

    # Expected ratios
    expected_data = {
        'performance_total_gain_v_max_investment': [0.1, 0.1, 0.1, 0, -0.1],
        'performance_total_gain_v_time_weighted_balance': [0.2, 0.1333, 0.12, 0, -0.1667],
        'performance_total_gain_v_active_time_weighted_balance': [0.25, 0.1667, 0.1364, 0, -0.2],
        'performance_realized_gain_v_max_investment': [0.05, 0.075, 0.0833, 0, -0.06],
        'performance_realized_gain_v_time_weighted_balance': [0.1, 0.1, 0.1, 0, -0.1],
        'performance_realized_gain_v_active_time_weighted_balance': [0.125, 0.125, 0.1136, 0, -0.12],
        'performance_unrealized_gain_v_max_investment': [0.05, 0.025, 0.0167, 0, -0.04],
        'performance_unrealized_gain_v_time_weighted_balance': [0.1, 0.0333, 0.02, 0, -0.0667],
        'performance_unrealized_gain_v_active_time_weighted_balance': [0.125, 0.0417, 0.0227, 0, -0.08],
    }
    expected_df = pd.DataFrame(expected_data, index=sample_performance_features_df.index)

    # Explanation of assertions:
    # 1. Ratios are calculated as profits_value / balance_value for corresponding columns.
    #    Example: For wallet1, total_gain_v_max_investment = 100 / 1000 = 0.1.
    # 2. Wallet4 with all 0 values should produce NaN for all ratios (division by zero).
    # 3. Wallet5 with losses should produce negative ratios.
    # 4. Round both calculated and expected values to 4 decimal places for consistent comparison.

    # Round both DataFrames to 4 decimal places for comparison
    rounded_ratio_df = ratio_df.round(4)
    rounded_expected_df = expected_df.round(4)

    # Assert equality after rounding
    assert np.allclose(
        rounded_ratio_df.values,
        rounded_expected_df.values,
        equal_nan=True
    ), "Calculated ratios do not match the expected values after rounding."


@pytest.mark.unit
def test_transform_performance_ratios(sample_performance_features_df, monkeypatch):
    """
    Test to validate performance ratio transformations with ntile=2 override.
    """
    # Override config setting
    monkeypatch.setitem(wallets_config['features'], 'ranking_ntiles', 2)

    # Steps 1-3 remain unchanged
    ratio_df = wpf.calculate_performance_ratios(sample_performance_features_df)
    balance_features_df = sample_performance_features_df.filter(like='balance_')
    transformed_df = wpf.transform_performance_ratios(ratio_df, balance_features_df)

    # Assertions 1-4 remain unchanged
    for col in ratio_df.columns:
        # Base ratio validation
        assert np.allclose(
            transformed_df[f"{col}/base"].values,
            ratio_df[col].values,
            equal_nan=True
        )

        # Rank validation
        expected_rank = ratio_df[col].rank(method="average", pct=True).values
        assert np.allclose(
            transformed_df[f"{col}/rank"].values,
            expected_rank,
            equal_nan=True
        )

        # Log validation
        expected_log = np.sign(ratio_df[col]) * np.log1p(ratio_df[col].abs())
        assert np.allclose(
            transformed_df[f"{col}/log"].values,
            expected_log,
            equal_nan=True
        )

        # Winsorization validation
        winsorization_threshold = wallets_config['features']['returns_winsorization']
        expected_winsorized = u.winsorize(ratio_df[col], cutoff=winsorization_threshold)
        assert np.allclose(
            transformed_df[f"{col}/winsorized"].values,
            expected_winsorized.values,
            equal_nan=True
        )

        # Modified ntile validation using overridden value
        denominator = col.split("/")[1]
        balance_col = f"balance_{denominator}"
        metric_ntiles = pd.qcut(
            balance_features_df[balance_col],
            q=2,  # Explicitly use 2 to match override
            labels=False,
            duplicates="drop"
        )
        expected_ntile_rank = (
            ratio_df[col]
            .groupby(metric_ntiles)
            .rank(method="average", pct=True)
            .fillna(0)
        )
        assert np.allclose(
            transformed_df[f"{col}/ntile_rank"].values,
            expected_ntile_rank.values,
            equal_nan=True
        )
