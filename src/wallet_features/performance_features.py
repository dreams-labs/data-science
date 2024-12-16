"""
Calculates wallet-level financial performance metrics
"""
import logging
import pandas as pd
import numpy as np

# local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()


def calculate_performance_features(wallets_df):
    """
    Generates various target variables for modeling wallet performance.

    Parameters:
    - wallets_df: pandas DataFrame with columns ['net_gain', 'max_investment']

    Returns:
    - DataFrame with additional target variables
    """
    metrics_df = wallets_df[['max_investment','net_gain']].copy().round(6)
    returns_winsorization = wallets_config['modeling']['returns_winsorization']
    epsilon = 1e-10

    # Calculate base return
    metrics_df['return'] = np.where(abs(metrics_df['max_investment']) == 0,0,
                                    metrics_df['net_gain'] / metrics_df['max_investment'])

    # Apply winsorization
    if returns_winsorization > 0:
        metrics_df['return'] = u.winsorize(metrics_df['return'],returns_winsorization)

    # Risk-Adjusted Dollar Return
    metrics_df['risk_adj_return'] = metrics_df['net_gain'] * \
        (1 + np.log10(metrics_df['max_investment'] + epsilon))

    # Normalize returns
    metrics_df['norm_return'] = (metrics_df['return'] - metrics_df['return'].min()) / \
        (metrics_df['return'].max() - metrics_df['return'].min())

    # Normalize logged investments
    log_invested = np.log10(metrics_df['max_investment'] + epsilon)
    metrics_df['norm_invested'] = (log_invested - log_invested.min()) / \
        (log_invested.max() - log_invested.min())

    # Performance score
    metrics_df['performance_score'] = (0.6 * metrics_df['norm_return'] +
                                     0.4 * metrics_df['norm_invested'])

    # Log-weighted return
    metrics_df['log_weighted_return'] = metrics_df['return'] * \
        np.log10(metrics_df['max_investment'] + epsilon)

    # Hybrid score (combining absolute and relative performance)
    max_gain = metrics_df['net_gain'].abs().max()
    metrics_df['norm_gain'] = metrics_df['net_gain'] / max_gain
    metrics_df['hybrid_score'] = (metrics_df['norm_gain'] +
                                metrics_df['norm_return']) / 2

    # Size-adjusted rank
    # Create mask for zero values
    zero_mask = metrics_df['max_investment'] == 0

    # Create quartiles series initialized with 'q0' for zero values
    quartiles = pd.Series('q0', index=metrics_df.index)

    # Calculate quartiles for non-zero values
    non_zero_quartiles = pd.qcut(metrics_df['max_investment'][~zero_mask],
                                q=4,
                                labels=['q1', 'q2', 'q3', 'q4'])

    # Assign the quartiles to non-zero values
    quartiles[~zero_mask] = non_zero_quartiles

    # Calculate size-adjusted rank within each quartile
    metrics_df['size_adjusted_rank'] = metrics_df.groupby(quartiles)['return'].rank(pct=True)


    # Clean up intermediate columns
    cols_to_drop = ['norm_return', 'norm_invested', 'norm_gain']
    metrics_df = metrics_df.drop(columns=[c for c in cols_to_drop
                                        if c in metrics_df.columns])

    return metrics_df.round(6)
