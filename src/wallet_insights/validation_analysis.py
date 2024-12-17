"""
Calculates metrics aggregated at the wallet level
"""
import logging
import pandas as pd
import numpy as np

# local module imports
import wallet_features.performance_features as wpf
import wallet_features.trading_features as wtf
from wallet_modeling.wallets_config_manager import WalletsConfig

# pylint:disable=invalid-name  # "X_test" doesn't conform to snake_case naming

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()


def calculate_validation_metrics(X_test, y_pred, validation_profits_df):
    """
    Params:
    - X_test (DataFrame): test set features with wallet addresses as index
    - y_pred (array): predicted scores from model
    - validation_profits_df (DataFrame): validation period profit data

    Returns:
    - wallet_performance_df (DataFrame): wallet-level validation metrics
    - bucketed_wallet_performance_df (DataFrame): grouped metrics by score bucket
    """
    # Calculate validation period wallet metrics
    validation_profits_df = wtf.add_cash_flow_transfers_logic(validation_profits_df)
    wallet_trading_features_df = wtf.calculate_wallet_trading_features(validation_profits_df)
    validation_wallets_df = wpf.calculate_performance_features(wallet_trading_features_df)

    # Attach validation period performance to modeling period scores
    wallet_performance_df = pd.DataFrame()
    wallet_performance_df['wallet_address'] = X_test.index.values
    wallet_performance_df['score'] = y_pred
    # Convert to float64 first, then round to avoid float32 precision issues
    wallet_performance_df['score_rounded'] = (np.ceil(wallet_performance_df['score'].astype(np.float64)*20)/20).round(2)
    wallet_performance_df = wallet_performance_df.set_index('wallet_address')
    wallet_performance_df = wallet_performance_df.join(validation_wallets_df, how='left')

    # Rest of the function remains the same...
    bucketed_performance_df = wallet_performance_df.groupby('score_rounded').agg(
        wallets=('score', 'count'),
        mean_invested=('max_investment', 'mean'),
        mean_total_net_flows=('total_net_flows', 'mean'),
        median_invested=('max_investment', 'median'),
        median_total_net_flows=('total_net_flows', 'median'),
    )

    bucketed_performance_df['mean_return'] = (bucketed_performance_df['mean_total_net_flows']
                                                     / bucketed_performance_df['mean_invested'])
    bucketed_performance_df['median_return'] = (bucketed_performance_df['median_total_net_flows']
                                                       / bucketed_performance_df['median_invested'])

    return wallet_performance_df, bucketed_performance_df
