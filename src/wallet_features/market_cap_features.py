"""
Calculates metrics aggregated at the wallet-coin-date level
"""
import logging
import time
import pandas as pd
import numpy as np

# Local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig
import utils as u

# set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()


def force_fill_market_cap(market_data_df):
    """
    Creates a column 'market_cap_filled' with complete coverage by:
    1. Back- and forwardfilling any existing values which breached thresholds in
        of data_retrieval.impute_market_cap().
    2. Blanket filling the remaining empty values with a constant value from the config.

    Params:
    - market_data_df (df): dataframe with column date, coin_id, market_cap, market_cap_imputed
    Returns:
    - market_data_df (df): input dataframe with new column 'market_cap_filled'
    """

    # Force fill market cap to generate a complete feature set
    market_data_df['market_cap_filled'] = market_data_df['market_cap_imputed']

    # Backfill and forward fill if there are any values at all, since these missing values failed imputation checks
    market_data_df['market_cap_filled'] = market_data_df.groupby('coin_id',observed=True)['market_cap_filled'].bfill()
    market_data_df['market_cap_filled'] = market_data_df.groupby('coin_id',observed=True)['market_cap_filled'].ffill()

    # For coins with no market cap data at all, bulk fill with a constant
    default_market_cap = wallets_config['data_cleaning']['market_cap_default_fill']
    market_data_df['market_cap_filled'] = market_data_df['market_cap_filled'].fillna(default_market_cap)

    return market_data_df


def calculate_volume_weighted_market_cap(profits_market_features_df):
    """
    Calculate volume-weighted average market cap for each wallet address.
    If volume is 0 for all records of a wallet, uses simple average instead.

    Parameters:
    - profits_market_features_df (df): DataFrame containing wallet_address, volume, and market_cap_filled

    Returns:
    pandas.DataFrame: DataFrame with wallet addresses and their weighted average market caps
    """
    # Create a copy to avoid modifying original
    df_calc = profits_market_features_df.copy()

    # Group by wallet_address
    grouped = df_calc.groupby('wallet_address')

    # Calculate weighted averages
    results = []

    for wallet, group in grouped:
        total_volume = group['volume'].sum()

        if total_volume > 0:
            # If there's volume, calculate volume-weighted average
            weighted_avg = np.average(
                group['market_cap_filled'],
                weights=group['volume']
            )
        else:
            # If no volume, use simple average
            weighted_avg = group['market_cap_filled'].mean()

        results.append({
            'wallet_address': wallet,
            'volume_wtd_market_cap': weighted_avg
        })

    volume_wtd_df = pd.DataFrame(results)
    volume_wtd_df = volume_wtd_df.set_index('wallet_address')

    return volume_wtd_df


def calculate_ending_balance_weighted_market_cap(profits_market_features_df):
    """
    Calculate USD balance-weighted average market cap for each wallet address
    using only the most recent date's data.

    Parameters:
    - profits_market_features_df (df): DataFrame containing wallet_address, usd_balance,
        date, and market_cap_filled columns

    Returns:
    - balance_wtd_df (df): DataFrame with wallet addresses and their balance-weighted
        average market caps
    """
    # Create a copy to avoid modifying original
    df_calc = profits_market_features_df.copy()

    # Get the latest date
    latest_date = df_calc['date'].max()

    # Filter for only the latest date
    latest_df = df_calc[df_calc['date'] == latest_date]

    # Group by wallet_address
    results = []

    for wallet, group in latest_df.groupby('wallet_address'):
        total_balance = group['usd_balance'].sum()

        if total_balance > 0:
            # If there's balance, calculate balance-weighted average
            weighted_avg = np.average(
                group['market_cap_filled'],
                weights=group['usd_balance']
            )
        else:
            # If no balance, use simple average
            weighted_avg = group['market_cap_filled'].mean()

        results.append({
            'wallet_address': wallet,
            'ending_portfolio_usd': total_balance,
            'portfolio_wtd_market_cap': weighted_avg
        })

    balance_wtd_df = pd.DataFrame(results)
    balance_wtd_df = balance_wtd_df.set_index('wallet_address')

    return balance_wtd_df


@u.timing_decorator
def calculate_market_cap_features(profits_df,market_data_df):
    """
    Calculates metrics about each wallet's interaction with coins of different market caps:
    1. Volume-weighted average market cap (uses only real transfers)
    2. Ending balance-weighted market cap (uses final period balances)

    Params:
    - profits_df (DataFrame): Daily profits with both real transfers and imputed period boundary rows
    - market_data_df (DataFrame): Price and market cap data

    Returns:
    - market_features_df (DataFrame): Market cap features indexed on wallet_address

    """
    start_time = time.time()
    logger.debug("Calculating market cap features...")

    # Force fill market cap gaps
    filled_market_cap_df = force_fill_market_cap(market_data_df)

    # Merge market cap data
    profits_market_cap_df = profits_df.merge(
        filled_market_cap_df[['date', 'coin_id', 'market_cap_filled']],
        on=['date', 'coin_id'],
        how='inner'
    )

    # Calculate ending balance weighted metrics using period end rows
    ending_balance_wtd_df = calculate_ending_balance_weighted_market_cap(
        profits_market_cap_df[profits_market_cap_df['date'] == profits_market_cap_df['date'].max()]
    )

    # Calculate volume-weighted metrics using only real transfers
    volume_wtd_df = calculate_volume_weighted_market_cap(profits_market_cap_df[~profits_market_cap_df['is_imputed']])

    # Merge into a wallet-indexed df of features
    market_features_df = volume_wtd_df.copy()
    market_features_df = market_features_df.join(ending_balance_wtd_df)

    logger.debug("Calculated market cap features after %.2f seconds.",
                 time.time() - start_time)

    return market_features_df
