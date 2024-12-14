"""
Calculates metrics aggregated at the wallet level
"""
import logging
import pandas as pd
import numpy as np

# local module imports
import wallet_features.performance_features as wp
import wallet_features.trading_features as wtf
from wallet_modeling.wallets_config_manager import WalletsConfig

# pylint:disable=invalid-name  # "X_test" doesn't conform to snake_case naming

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()


def calculate_validation_metrics(X_test, y_pred, validation_profits_df):
    """
    Calculate validation period metrics for wallet scoring and group performance by score buckets.

    Parameters:
    -----------
    X_test : pandas.DataFrame
        Test set features with wallet addresses as index
    y_pred : array-like
        Predicted scores from the model
    validation_profits_df : pandas.DataFrame
        DataFrame containing validation period profit data

    Returns:
    --------
    wallet_performance_df (pd.DataFrame)
        Wallet-indexed dataframe showing performance metrics over the validation period
    bucketed_wallet_performance_df (pd.DataFrame)
        Grouped validation metrics by score bucket, including:
        - Number of wallets
        - Mean and median invested amounts
        - Mean and median net gains
        - Mean and median returns
    """
    # Calculate validation period wallet metrics
    validation_profits_df = wtf.add_cash_flow_transfers_logic(validation_profits_df)
    wallet_trading_features_df = wtf.calculate_wallet_trading_features(validation_profits_df)
    validation_wallets_df = wp.calculate_performance_features(wallet_trading_features_df)

    # Attach validation period performance to modeling period scores
    wallet_performance_df = pd.DataFrame()
    wallet_performance_df['wallet_address'] = X_test.index.values
    wallet_performance_df['score'] = y_pred
    wallet_performance_df['score_rounded'] = np.ceil(wallet_performance_df['score']*20)/20
    wallet_performance_df = wallet_performance_df.set_index('wallet_address')
    wallet_performance_df = wallet_performance_df.join(validation_wallets_df, how='left')

    # Group wallets by score bucket and assess performance
    bucketed_performance_df = wallet_performance_df.groupby('score_rounded').agg(
        wallets=('score', 'count'),
        mean_invested=('invested', 'mean'),
        mean_net_gain=('net_gain', 'mean'),
        median_invested=('invested', 'median'),
        median_net_gain=('net_gain', 'median'),
    )

    # Calculate return metrics
    bucketed_performance_df['mean_return'] = (bucketed_performance_df['mean_net_gain']
                                                     / bucketed_performance_df['mean_invested'])
    bucketed_performance_df['median_return'] = (bucketed_performance_df['median_net_gain']
                                                       / bucketed_performance_df['median_invested'])

    return wallet_performance_df, bucketed_performance_df
