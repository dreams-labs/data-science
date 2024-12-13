"""
Calculates metrics aggregated at the wallet level
"""
import time
import logging
from pathlib import Path
import pandas as pd
import yaml

# Local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig
import wallet_modeling.wallet_modeling as wm
import wallet_features.wallet_coin_features as wcf
import wallet_features.wallet_coin_date_features as wcdf
import wallet_features.trading_features as wtf
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()
wallets_metrics_config = u.load_config('../config/wallets_metrics_config.yaml')
wallets_features_config = yaml.safe_load(Path('../config/wallets_features_config.yaml').read_text(encoding='utf-8'))


def calculate_wallet_features(profits_df, market_indicators_data_df, transfers_data_df, wallet_cohort):
    """
    Calculates all features for the wallets in a given profits_df

    Params:
    - profits_df (df): for the window over which the metrics should be computed
    - market_indicators_data_df (df): the full market data df with indicators added
    - transfers_data_df (df): each wallet's lifetime transfers data
    - wallet_cohort (array-like): Array of all wallet addresses that should be present

    Returns:
    - wallet_features_df (df): df indexed on wallet_address with all features
    """
    # Create a DataFrame with all wallets that should exist
    wallet_features_df = pd.DataFrame(index=wallet_cohort)
    wallet_features_df.index.name = 'wallet_address'

    # Trading features (inner join, custom fill)
    profits_df = wtf.add_cash_flow_transfers_logic(profits_df)
    trading_features = wtf.calculate_wallet_trading_features(profits_df)
    trading_features = wtf.fill_trading_features_data(trading_features, wallet_cohort)
    wallet_features_df = wallet_features_df.join(trading_features, how='inner')

    # Market timing features (fill zeros)
    timing_features = calculate_market_timing_features(profits_df, market_indicators_data_df)
    wallet_features_df = wallet_features_df.join(timing_features, how='left')\
        .fillna({col: 0 for col in timing_features.columns})

    # Market cap features (fill zeros)
    market_features = wcdf.calculate_market_cap_features(profits_df, market_indicators_data_df)
    wallet_features_df = wallet_features_df.join(market_features, how='left')\
        .fillna({col: 0 for col in market_features.columns})

    # Transfers features (fill -1)
    transfers_features = calculate_transfers_features(profits_df, transfers_data_df)
    wallet_features_df = wallet_features_df.join(transfers_features, how='left')\
        .fillna({col: -1 for col in transfers_features.columns})

    # Performance features (inner join, no fill)
    performance_features = wm.generate_target_variables(wallet_features_df)
    wallet_features_df = wallet_features_df.join(
        performance_features.drop(['invested', 'net_gain'], axis=1),
        how='inner'
    )

    return wallet_features_df





def calculate_market_timing_features(profits_df, market_indicators_data_df):
    """
    Calculate features capturing how wallet transaction timing aligns with future market movements.

    This function performs a sequence of transformations to assess wallet timing performance:
    1. Enriches market data with technical indicators (RSIs, SMAs) on price and volume
    2. Calculates future values of these indicators at different time offsets (e.g., 7, 14, 30 days ahead)
    3. Computes relative changes between current and future indicator values
    4. For each wallet's transactions, calculates value-weighted and unweighted averages of these future changes,
        treating buys and sells separately

    Args:
        profits_df (pd.DataFrame): Wallet transaction data with columns:
            - wallet_address (pd.Categorical): Unique wallet identifier
            - coin_id (str): Identifier for the traded coin
            - date (pd.Timestamp): Transaction date
            - usd_net_transfers (float): USD value of transaction (positive=buy, negative=sell)

        market_indicators_data_df (pd.DataFrame): Raw market data with columns:
            - coin_id (str): Identifier for the coin
            - date (pd.Timestamp): Date of market data
            - price (float): Asset price
            - volume (float): Trading volume
            - all of the indicators specific in wallets_metrics_config

    Returns:
        pd.DataFrame: Features indexed by wallet_address with columns for each market timing metric:
            {indicator}_vs_lead_{n}_{direction}_{type}
            where:
            - indicator: The base metric (e.g., price_rsi_14, volume_sma_7)
            - n: The forward-looking period (e.g., 7, 14, 30 days)
            - direction: 'buy' or 'sell'
            - type: 'weighted' (by USD value) or 'mean'

            All wallet_addresses from the categorical index are included with zeros for missing data.
    """
    start_time = time.time()
    logger.info("Calculating market timing features...")

    # add timing offset features
    market_timing_df = wcdf.calculate_offsets(market_indicators_data_df,wallets_features_config)
    market_timing_df,relative_change_columns = wcdf.calculate_relative_changes(market_timing_df,wallets_features_config)

    # flatten the wallet-coin-date transactions into wallet-indexed features
    wallet_timing_features_df = wcf.generate_all_timing_features(
        profits_df,
        market_timing_df,
        relative_change_columns,
        wallets_config['features']['timing_metrics_min_transaction_size'],
    )

    logger.info("Calculated market timing features after %.2f seconds.",
                time.time() - start_time)

    return wallet_timing_features_df



def calculate_transfers_features(profits_df, transfers_data_df):
    """
    Retrieves facts about the wallet's transfer activity based on blockchain data.

    Params:
        profits_df (df): the profits_df for the period that the features will reflect
        transfers_data_df (df): each wallet's lifetime transfers data

    Returns:
        transfers_features_df (df): dataframe indexed on wallet_id with transfers feature columns
    """
    # Inner join lifetime transfers with the profits_df window to filter on date
    window_transfers_data_df = pd.merge(
        profits_df,
        transfers_data_df,
        left_on=['coin_id', 'date', 'wallet_address'],
        right_on=['coin_id', 'first_transaction', 'wallet_id'],
        how='inner'
    )

    # Append buyer numbers to the merged_df
    transfers_features_df = window_transfers_data_df.groupby('wallet_id').agg({
        'buyer_number': ['count', 'mean', 'median', 'min']
    })
    transfers_features_df.columns = [
        'new_coin_buy_counts',
        'avg_buyer_number',
        'median_buyer_number',
        'min_buyer_number'
    ]

    # Rename to the wallet_id index to "wallet_address" to be consistent with the other functions
    transfers_features_df.index.name = 'wallet_address'

    return transfers_features_df
