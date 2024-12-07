"""
Calculates metrics aggregated at the wallet level
"""

import logging
import pandas as pd
import numpy as np

# pylint:disable=W1203  # f strings in logs

# set up logger at the module level
logger = logging.getLogger(__name__)


def add_performance_metrics(wallets_df, min_investment_threshold=1000):
    """
    Adds various performance metrics to a DataFrame containing wallet-level metrics.

    Parameters:
    - wallets_df: pandas DataFrame with columns ['net_gain', 'invested'] indexed by wallet_address
    - min_investment_threshold: minimum investment amount for qualified metrics (default: 1000)

    Returns:
    - DataFrame with additional performance metrics
    """
    # Create copy to avoid modifying original
    metrics_df = wallets_df.copy()

    # 0. Basic Return
    metrics_df['return'] = metrics_df['net_gain']/metrics_df['invested']

    # 1. Risk-Adjusted Dollar Return
    # Add small epsilon to handle zero investments
    epsilon = 1e-10
    metrics_df['risk_adj_return'] = metrics_df['net_gain'] * (1 + np.log10(metrics_df['invested'] + epsilon))

    # 2. Composite Score
    # Normalize returns (handling extreme values)
    metrics_df['norm_return'] = (metrics_df['return'] - metrics_df['return'].min()) / \
        (metrics_df['return'].max() - metrics_df['return'].min())

    # Normalize logged investments
    log_invested = np.log10(metrics_df['invested'] + epsilon)
    metrics_df['norm_invested'] = (log_invested - log_invested.min()) / \
        (log_invested.max() - log_invested.min())

    # Combined score (0.6 weight on return, 0.4 on investment size)
    metrics_df['performance_score'] = (0.6 * metrics_df['norm_return'] +
                                     0.4 * metrics_df['norm_invested'])

    # 3. Thresholded Approach
    metrics_df['qualified_return'] = np.where(
        metrics_df['invested'] >= min_investment_threshold,
        metrics_df['return'],
        0
    )

    # 4. Investment-Weighted Return
    # Scale returns by investment quartile (1-4)
    metrics_df['investment_quartile'] = pd.qcut(metrics_df['invested'],
                                              q=4,
                                              labels=[1, 2, 3, 4])
    metrics_df['inv_weighted_return'] = metrics_df['return'] * \
        metrics_df['investment_quartile'].astype(float)

    # 5. Logarithmic Investment-Weighted Return
    # More gradual scaling of investment impact
    metrics_df['log_weighted_return'] = metrics_df['return'] * \
        np.log10(metrics_df['invested'] + epsilon)

    # 6. Hybrid Score
    # Combine absolute gains with return percentage
    max_gain = metrics_df['net_gain'].abs().max()
    metrics_df['norm_gain'] = metrics_df['net_gain'] / max_gain
    metrics_df['hybrid_score'] = (metrics_df['norm_gain'] + metrics_df['norm_return']) / 2

    # 7. Investment-Size Buckets
    # Create categorical performance metric based on both return and investment size
    metrics_df['investment_size'] = pd.cut(metrics_df['invested'],
                                         bins=[0, 100, 1000, 10000, float('inf')],
                                         labels=['micro', 'small', 'medium', 'large'])

    # Calculate percentile rank of return within each investment size bucket
    metrics_df['size_adjusted_rank'] = metrics_df.groupby('investment_size')['return'] \
        .rank(pct=True)

    # Clean up intermediate columns
    cols_to_drop = ['norm_return', 'norm_invested', 'norm_gain', 'investment_quartile']
    metrics_df = metrics_df.drop(columns=cols_to_drop)

    # Add metadata about metrics
    metrics_df = metrics_df.round(6)  # Round to 6 decimal places for cleaner numbers

    return metrics_df
