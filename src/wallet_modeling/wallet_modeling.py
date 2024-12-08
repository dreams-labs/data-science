"""
Calculates metrics aggregated at the wallet level
"""

import logging
import pandas as pd
import numpy as np

# pylint:disable=W1203  # f strings in logs

# set up logger at the module level
logger = logging.getLogger(__name__)



def generate_target_variables(wallets_df):
    """
    Generates various target variables for modeling wallet performance.

    Parameters:
    - wallets_df: pandas DataFrame with columns ['net_gain', 'invested']

    Returns:
    - DataFrame with additional target variables
    """
    metrics_df = wallets_df.copy()
    epsilon = 1e-10

    # Calculate base return
    metrics_df['return'] = metrics_df['net_gain'] / metrics_df['invested']

    # Risk-Adjusted Dollar Return
    metrics_df['risk_adj_return'] = metrics_df['net_gain'] * \
        (1 + np.log10(metrics_df['invested'] + epsilon))

    # Normalize returns
    metrics_df['norm_return'] = (metrics_df['return'] - metrics_df['return'].min()) / \
        (metrics_df['return'].max() - metrics_df['return'].min())

    # Normalize logged investments
    log_invested = np.log10(metrics_df['invested'] + epsilon)
    metrics_df['norm_invested'] = (log_invested - log_invested.min()) / \
        (log_invested.max() - log_invested.min())

    # Performance score
    metrics_df['performance_score'] = (0.6 * metrics_df['norm_return'] +
                                     0.4 * metrics_df['norm_invested'])

    # Log-weighted return
    metrics_df['log_weighted_return'] = metrics_df['return'] * \
        np.log10(metrics_df['invested'] + epsilon)

    # Hybrid score (combining absolute and relative performance)
    max_gain = metrics_df['net_gain'].abs().max()
    metrics_df['norm_gain'] = metrics_df['net_gain'] / max_gain
    metrics_df['hybrid_score'] = (metrics_df['norm_gain'] +
                                metrics_df['norm_return']) / 2

    # Size-adjusted rank
    quartiles = pd.qcut(metrics_df['invested'], q=4, labels=['q1', 'q2', 'q3', 'q4'])
    metrics_df['size_adjusted_rank'] = metrics_df.groupby(quartiles)['return'] \
        .rank(pct=True)

    # Clean up intermediate columns
    cols_to_drop = ['norm_return', 'norm_invested', 'norm_gain']
    metrics_df = metrics_df.drop(columns=[c for c in cols_to_drop
                                        if c in metrics_df.columns])

    return metrics_df.round(6)
