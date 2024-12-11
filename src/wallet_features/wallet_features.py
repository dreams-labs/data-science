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

    # Add trading behavior features
    profits_df = wcf.add_cash_flow_transfers_logic(profits_df)
    wallet_trading_features_df = calculate_wallet_trading_features(profits_df)
    wallet_trading_features_df = fill_trading_features_data(wallet_trading_features_df, wallet_cohort)
    wallet_features_df = wallet_features_df.join(wallet_trading_features_df, how='inner')

    # Add transaction timing features
    wallet_timing_features_df = calculate_market_timing_features(profits_df, market_indicators_data_df)
    wallet_features_df = wallet_features_df.join(wallet_timing_features_df, how='left').fillna(0)

    # Add market cap features
    market_features_df = wcdf.calculate_market_cap_features(profits_df,market_indicators_data_df)
    wallet_features_df = wallet_features_df.join(
        market_features_df.drop('total_volume',axis=1)
        ,how='left'
        ).fillna(0)

    # Add transfers data features
    transfers_features_df = calculate_transfers_features(profits_df, transfers_data_df)
    wallet_features_df = wallet_features_df.join(
        market_features_df.drop('total_volume',axis=1)
        ,how='left'
        ).fillna(0)

    # Add financial performance features
    performance_df = wm.generate_target_variables(wallet_features_df)
    performance_df = performance_df.drop(['invested','net_gain'],axis=1)
    wallet_features_df = wallet_features_df.join(performance_df)

    return wallet_features_df



def calculate_wallet_trading_features(profits_df):
    """
    Calculates the return on investment for the wallet and additional aggregation metrics,
    ensuring proper date-based ordering for cumulative calculations.

    Profits_df must have initial balances reflected as positive cash_flow_transfers
    and ending balances reflected as negative cash_flow_transfers for calculations to
    be accurately reflected.

    - Invested: the maximum amount of cumulative net inflows for the wallet,
        properly ordered by date to ensure accurate running totals
    - Return: All net transfers summed together, showing the combined change
        in assets and balance

    Params:
    - profits_df (pd.DataFrame): df showing all usd net transfers for coin-wallet pairs,
        with columns coin_id, wallet_address, date, cash_flow_transfers, usd_net_transfers,
        and is_imputed

    Returns:
    - wallet_metrics_df (pd.DataFrame): df keyed on wallet_address with columns
        'invested', 'net_gain', 'return', and additional aggregation metrics
    """
    start_time = time.time()
    logger.info("Calculating wallet trading features...")

    # Ensure date is in datetime format
    profits_df['date'] = pd.to_datetime(profits_df['date'])

    # Sort by date and wallet_address to ensure proper cumulative calculations
    profits_df = profits_df.sort_values(['wallet_address', 'date'])

    # Precompute necessary transformations
    profits_df['abs_usd_net_transfers'] = profits_df['usd_net_transfers'].abs()

    # Calculate cumsum by wallet, respecting date order
    profits_df['cumsum_cash_flow_transfers'] = profits_df.groupby('wallet_address')['cash_flow_transfers'].cumsum()

    # Calculate per-coin cumulative flows to catch potential issues
    profits_df['cumsum_by_coin'] = profits_df.groupby(['wallet_address', 'coin_id'])['cash_flow_transfers'].cumsum()

    # Metrics that take into account imputed rows/profits
    logger.debug("Calculating wallet metrics based on imputed performance...")
    imputed_metrics_df = profits_df.groupby('wallet_address').agg(
        invested=('cumsum_cash_flow_transfers', 'max'),
        net_gain=('cash_flow_transfers', lambda x: -1 * x.sum()),  # outflows reflect profit-taking
        unique_coins_traded=('coin_id', 'nunique'),
        first_activity=('date', 'min'),
        last_activity=('date', 'max')
    )

    # Metrics only based on observed activity
    logger.debug("Calculating wallet metrics based on observed behavior...")
    observed_metrics_df = profits_df[~profits_df['is_imputed']].groupby('wallet_address').agg(
        transaction_days=('date', 'nunique'),  # Changed from count to nunique for actual trading days
        total_volume=('abs_usd_net_transfers', 'sum'),
        average_transaction=('abs_usd_net_transfers', 'mean')
    )

    # Join all metrics together
    wallet_trading_features_df = imputed_metrics_df.join(observed_metrics_df)

    # Fill 0s for wallets without observed activity
    wallet_trading_features_df = wallet_trading_features_df.fillna(0)

    # Remove any negative 0s
    wallet_trading_features_df = wallet_trading_features_df.replace(-0,0)

    # Compute additional derived metrics
    wallet_trading_features_df['activity_days'] = (wallet_trading_features_df['last_activity'] -
                                        wallet_trading_features_df['first_activity']).dt.days + 1
    wallet_trading_features_df['activity_density'] = (wallet_trading_features_df['transaction_days']
                                                      / wallet_trading_features_df['activity_days'])

    logger.info(f"Wallet trading features computed after {time.time() - start_time:.2f} seconds")

    return wallet_trading_features_df



def fill_trading_features_data(wallet_trading_features_df, wallet_cohort):
    """
    Fill missing wallet data for all wallets in wallet_cohort that are not in window_wallets_df.

    Parameters:
    wallet_trading_features_df (pd.DataFrame): DataFrame with wallet trading features
    wallet_cohort (array-like): Array of all wallet addresses that should be present

    Returns:
    pd.DataFrame: Complete DataFrame with filled values for missing wallets
    """

    # Create the fill value dictionary
    fill_values = {
        'invested': 0,
        'net_gain': 0,
        'unique_coins_traded': 0,
        'transaction_days': 0,
        'total_volume': 0,
        'average_transaction': 0,
    }

    # Create a DataFrame with all wallets that should exist
    complete_df = pd.DataFrame(index=wallet_cohort)
    complete_df.index.name = 'wallet_address'

    # Add the fill value columns
    for column, fill_value in fill_values.items():
        complete_df[column] = fill_value

    # Update with actual values where they exist
    complete_df.update(wallet_trading_features_df)

    return complete_df



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

    logger.info("Calculated market timing features after %.2f.",
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
