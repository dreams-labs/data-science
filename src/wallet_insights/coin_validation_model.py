import logging
import pandas as pd
import numpy as np

# local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig

# pylint:disable=invalid-name  # X doesn't conform to snake case

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare features and target with better transformations.

    Params:
    - df (DataFrame): input dataframe with coin metrics

    Returns:
    - X (ndarray): processed feature matrix
    - y (ndarray): target values
    """
    # Keep market_cap this time but log transform it
    df = df.copy()
    df['log_market_cap'] = np.log1p(df['market_cap_filled'])

    # Log transform highly skewed numeric columns
    skewed_cols = [
        'weighted_avg_score', 'top_wallet_balance',
        'total_balance', 'avg_wallet_balance'
    ]
    for col in skewed_cols:
        df[f'log_{col}'] = np.log1p(df[col])

    # Create ratios that might be predictive
    df['wallet_concentration'] = df['top_wallet_balance'] / (df['total_balance'] + 1)

    # Select features including new transformations
    feature_cols = [
        'log_market_cap', 'log_weighted_avg_score',
        'log_top_wallet_balance', 'log_total_balance',
        'top_wallet_count', 'total_wallets', 'mean_score',
        'score_std', 'top_wallet_balance_pct', 'wallet_concentration',
        'score_confidence'
    ]

    X = df[feature_cols].values
    y = df['coin_return'].values

    return X, y
