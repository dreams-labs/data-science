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
    """Verify that crypto_net_gain and net_crypto_investment are correctly
    calculated and preserved from input data"""
    mock_input = pd.DataFrame({
        'crypto_net_gain': [100, -50],
        'net_crypto_investment': [80, -30],
        'total_crypto_buys': [1000, 500],
        'total_crypto_sells': [920, 530]
    })
    result = wpf.calculate_profits_features(mock_input)
    assert np.allclose(result['crypto_net_gain'], mock_input['crypto_net_gain'])
    assert np.allclose(result['net_crypto_investment'], mock_input['net_crypto_investment'])


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
def test_transform_performance_ratios(sample_performance_features_df):
    """
    Test to validate the correctness of transformations applied to performance ratios by
    wpf.transform_performance_ratios.
    Steps:
    1. Generate raw ratios using wpf.calculate_performance_ratios.
    2. Apply transformations, including rank, log, winsorization, and ntile rank.
    3. Validate the correctness of each transformation step.
    """
    # Step 1: Generate raw ratios
    ratio_df = wpf.calculate_performance_ratios(sample_performance_features_df)

    # Step 2: Define balance metrics for ntile calculation
    balance_features_df = sample_performance_features_df.filter(like='balance_')

    # Step 3: Transform the performance ratios
    transformed_df = wpf.transform_performance_ratios(ratio_df, balance_features_df)

    # Explanation of assertions:
    # 1. Base ratios: Validate that the base ratios remain unchanged after transformation.
    for col in ratio_df.columns:
        assert np.allclose(
            transformed_df[f"{col}/base"].values,
            ratio_df[col].values,
            equal_nan=True
        ), f"Base ratio for {col} does not match expected values."

    # 2. Rank: Validate that the ranks are correctly calculated as percentiles.
    for col in ratio_df.columns:
        expected_rank = ratio_df[col].rank(method="average", pct=True).values
        assert np.allclose(
            transformed_df[f"{col}/rank"].values,
            expected_rank,
            equal_nan=True
        ), f"Rank for {col} does not match expected values."

    # 3. Log transformation: Validate signed log calculations.
    for col in ratio_df.columns:
        expected_log = np.sign(ratio_df[col]) * np.log1p(ratio_df[col].abs())
        assert np.allclose(
            transformed_df[f"{col}/log"].values,
            expected_log,
            equal_nan=True
        ), f"Log transformation for {col} does not match expected values."

    # 4. Winsorization: Validate winsorized ratios based on config.
    # Assume wallets_config['features']['returns_winsorization'] = 0.05
    winsorization_threshold = wallets_config['features']['returns_winsorization']
    for col in ratio_df.columns:
        series = ratio_df[col]
        expected_winsorized = u.winsorize(series, cutoff=winsorization_threshold)
        assert np.allclose(
            transformed_df[f"{col}/winsorized"].values,
            expected_winsorized.values,
            equal_nan=True
        ), f"Winsorized values for {col} do not match expected values."

    # 5. Ntile rank: Validate that ntile ranks are calculated correctly.
    ntile_count = 10  # Assume config sets this to 10
    for col in ratio_df.columns:
        denominator = col.split("/")[1]
        balance_col = f"balance_{denominator}"
        metric_ntiles = pd.qcut(
            balance_features_df[balance_col],
            q=ntile_count,
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
        ), f"Ntile rank for {col} does not match expected values."


















# # ------------------------------------------------- #
# # calculate_time_weighted_returns() unit tests
# # ------------------------------------------------- #

# @pytest.fixture
# def portfolio_test_data():
#     """Test data fixture with both BTC and ETH holdings."""
#     test_data = pd.DataFrame([
#         # BTC wallet with imputed values
#         {'coin_id': 'btc', 'wallet_address': 'wallet_a', 'date': '2024-01-01',
#          'usd_balance': 50, 'usd_net_transfers': 0, 'is_imputed': True},
#         {'coin_id': 'btc', 'wallet_address': 'wallet_a', 'date': '2024-10-01',
#          'usd_balance': 70, 'usd_net_transfers': 0, 'is_imputed': True},
#         # ETH wallet with transfers
#         {'coin_id': 'eth', 'wallet_address': 'wallet_a', 'date': '2024-01-01',
#          'usd_balance': 100, 'usd_net_transfers': 100, 'is_imputed': False},
#         {'coin_id': 'eth', 'wallet_address': 'wallet_a', 'date': '2024-02-01',
#          'usd_balance': 250, 'usd_net_transfers': 50, 'is_imputed': False},
#         {'coin_id': 'eth', 'wallet_address': 'wallet_a', 'date': '2024-10-01',
#          'usd_balance': 125, 'usd_net_transfers': 0, 'is_imputed': False}
#     ])
#     test_data['date'] = pd.to_datetime(test_data['date'])
#     return test_data


# @pytest.mark.unit
# def test_calculate_time_weighted_returns_imputed_case():
#     """Tests TWR calculation for a wallet with only imputed balances."""

#     # Setup test data
#     test_data = pd.DataFrame([
#         {'coin_id': 'btc', 'wallet_address': 'wallet_a', 'date': '2024-01-01',
#          'usd_balance': 50, 'usd_net_transfers': 0, 'is_imputed': True},
#         {'coin_id': 'btc', 'wallet_address': 'wallet_a', 'date': '2024-10-01',
#          'usd_balance': 70, 'usd_net_transfers': 0, 'is_imputed': True},
#     ])
#     test_data['date'] = pd.to_datetime(test_data['date'])

#     # Calculate TWR
#     result = wpf.calculate_time_weighted_returns(test_data)

#     # Expected values
#     expected_twr = 0.40  # (70-50)/50 = 0.4
#     expected_days = 274  # Jan 1 to Oct 1
#     expected_annual = ((1 + 0.40) ** (365/274)) - 1  # ≈ 0.55

#     # Assertions with tolerance for floating point
#     assert abs(result.loc['wallet_a', 'time_weighted_return'] - expected_twr) < 0.001
#     assert result.loc['wallet_a', 'days_held'] == expected_days
#     assert abs(result.loc['wallet_a', 'annualized_twr'] - expected_annual) < 0.001


# @pytest.mark.unit
# def test_calculate_time_weighted_returns_weighted_periods():
#     """Tests TWR calculation with different holding periods, amounts, and a transfer."""

#     test_data = pd.DataFrame([
#         {'coin_id': 'eth', 'wallet_address': 'wallet_a', 'date': '2024-01-01',
#             'usd_balance': 100, 'usd_net_transfers': 100, 'is_imputed': False},  # Initial $100
#         {'coin_id': 'eth', 'wallet_address': 'wallet_a', 'date': '2024-02-01',
#             'usd_balance': 250, 'usd_net_transfers': 50, 'is_imputed': False},    # Added $50, value up
#         {'coin_id': 'eth', 'wallet_address': 'wallet_a', 'date': '2024-10-01',
#             'usd_balance': 125, 'usd_net_transfers': 0, 'is_imputed': False}     # Value dropped
#     ])

#     test_data['date'] = pd.to_datetime(test_data['date'])
#     result = wpf.calculate_time_weighted_returns(test_data)

#     # Manual calculation:
#     # Period 1: Jan 1 - Feb 1 (31 days)
#     # Pre-transfer balance = 250 - 50 = 200
#     # Return = 200/100 = 100% = 1.0
#     # Weighted return = 1.0 * 31 = 31

#     # Period 2: Feb 1 - Oct 1 (243 days)
#     # Return = 125/250 = -50% = -0.5
#     # Weighted return = -0.5 * 243 = -121.5

#     # Total days = 274
#     # Time weighted return = (31 - 121.5) / 274 = -0.33
#     expected_twr = -0.33

#     # Annualized = (1 - 0.33)^(365/274) - 1 ≈ -0.41
#     expected_annual = ((1 + expected_twr) ** (365/274)) - 1

#     # Assertions
#     assert result.loc['wallet_a', 'days_held'] == 274
#     assert abs(result.loc['wallet_a', 'time_weighted_return'] - expected_twr) < 0.01
#     assert abs(result.loc['wallet_a', 'annualized_twr'] - expected_annual) < 0.01
