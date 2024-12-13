"""
Calculates metrics aggregated at the wallet-coin level
"""

import logging
import pandas as pd

# Local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig

# set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()



def calculate_timing_features_for_column(df, metric_column):
    """
    Calculate timing features for a single metric column from pre-merged DataFrame.

    Args:
        df (pd.DataFrame): Pre-merged DataFrame with columns [wallet_address, usd_net_transfers, metric_column]
        metric_column (str): Name of the column to analyze

    Returns:
        pd.DataFrame: DataFrame indexed by wallet_address with columns:
            - {metric_column}_buy_weighted
            - {metric_column}_buy_mean
            - {metric_column}_sell_weighted
            - {metric_column}_sell_mean
    """
    # Split into buys and sells
    buys = df[df['usd_net_transfers'] > 0].copy()
    sells = df[df['usd_net_transfers'] < 0].copy()

    features = pd.DataFrame(index=df['wallet_address'].unique())

    # Vectorized buy calculations
    if not buys.empty:
        # Regular mean
        features[f"{metric_column}_buy_mean"] = (
            buys.groupby('wallet_address')[metric_column].mean()
        )

        # Weighted mean: First compute the products, then group
        buys['weighted_values'] = buys[metric_column] * abs(buys['usd_net_transfers'])
        weighted_sums = buys.groupby('wallet_address')['weighted_values'].sum()
        weight_sums = buys.groupby('wallet_address')['usd_net_transfers'].apply(abs).sum()
        features[f"{metric_column}_buy_weighted"] = weighted_sums / weight_sums

    # Similar for sells
    if not sells.empty:
        features[f"{metric_column}_sell_mean"] = (
            sells.groupby('wallet_address')[metric_column].mean()
        )

        sells['weighted_values'] = sells[metric_column] * abs(sells['usd_net_transfers'])
        weighted_sums = sells.groupby('wallet_address')['weighted_values'].sum()
        weight_sums = sells.groupby('wallet_address')['usd_net_transfers'].apply(abs).sum()
        features[f"{metric_column}_sell_weighted"] = weighted_sums / weight_sums

    return features


def generate_all_timing_features(
    profits_df,
    market_timing_df,
    relative_change_columns,
    min_transaction_size=0
):
    """
    Generate timing features for multiple market metric columns.

    Args:
        profits_df (pd.DataFrame): DataFrame with columns [coin_id, date, wallet_address, usd_net_transfers]
        market_timing_df (pd.DataFrame): DataFrame with market timing metrics indexed by (coin_id, date)
        relative_change_columns (list): List of column names from market_timing_df to analyze
        min_transaction_size (float): Minimum absolute USD value of transaction to consider

    Returns:
        pd.DataFrame: DataFrame indexed by wallet_address with generated features for each input column
    """
    # Get list of wallets to include
    wallet_addresses = profits_df['wallet_address'].unique()

    # Filter by minimum transaction size
    filtered_profits = profits_df[
        abs(profits_df['usd_net_transfers']) >= min_transaction_size
    ].copy()

    # Perform the merge once
    timing_profits_df = filtered_profits.merge(
        market_timing_df[relative_change_columns + ['coin_id', 'date']],
        on=['coin_id', 'date'],
        how='left'
    )

    # Calculate features for each column
    all_features = []
    for col in relative_change_columns:
        logger.debug("Generating timing performance features for %s...", col)
        col_features = calculate_timing_features_for_column(
            timing_profits_df,
            col
        )
        all_features.append(col_features)

    # Combine all feature sets
    if all_features:
        result = pd.concat(all_features, axis=1)
    else:
        result = pd.DataFrame(
            index=pd.Index(wallet_addresses, name='wallet_address')
        )

    # Ensure all wallet addresses are included and fill NaNs
    result = pd.DataFrame(
        result,
        index=pd.Index(wallet_addresses, name='wallet_address')
    ).fillna(0)

    return result
