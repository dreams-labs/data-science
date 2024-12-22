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

    df_calc = df_calc[~df_calc['is_imputed']]
    df_calc['volume'] = abs(df_calc['usd_net_transfers'])

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


def calculate_ending_balance_weighted_market_cap(profits_market_cap_df):
    """
    Calculate USD balance-weighted average market cap for each wallet address
    using only the most recent date's data.

    Parameters:
    - profits_market_cap_df (df): profits_df with a 'market_cap_filled' column added

    Returns:
    - wallet_end_balance_wtd_mc_df (df): DataFrame with index wallet_address and column
        'end_portfolio_wtd_market_cap' showing avg mc weighted by ending balance, or -1
        if the ending balance is 0
    """
    # Create a copy to avoid modifying original
    ending_balances_profits_df = profits_market_cap_df.copy()

    # Retrieve the ending balances of all coin-wallet pairs
    ending_balances_profits_df = ending_balances_profits_df.drop_duplicates(subset=['coin_id', 'wallet_address'],
                                                                            keep='last')

    # Compute the ending market cap dollars
    ending_balances_profits_df['market_cap_balance'] = (ending_balances_profits_df['usd_balance']
                                                        * ending_balances_profits_df['market_cap_filled'])

    # Calculate each wallet's total ending crypto balance and market cap dollars
    wallet_end_balance_wtd_mc_df = ending_balances_profits_df.groupby('wallet_address',observed=True).agg(
        total_usd_balance=('usd_balance','sum'),
        total_market_cap_balance=('market_cap_balance','sum')
    )

    # Divide to find weighted average market cap
    wallet_end_balance_wtd_mc_df['end_portfolio_wtd_market_cap'] = np.where(
                                    # indicates to xgboost there is no value
                                    wallet_end_balance_wtd_mc_df['total_usd_balance'] == 0,-1,

                                    # finds weighted average
                                    (wallet_end_balance_wtd_mc_df['total_market_cap_balance']
                                        / wallet_end_balance_wtd_mc_df['total_usd_balance'])
                                )

    return wallet_end_balance_wtd_mc_df[['end_portfolio_wtd_market_cap']]


# @u.timing_decorator
# def calculate_market_cap_features(profits_df,market_data_df):
#     """
#     Calculates metrics about each wallet's interaction with coins of different market caps:
#     1. Volume-weighted average market cap (uses only real transfers)
#     2. Ending balance-weighted market cap (uses final period balances)

#     Params:
#     - profits_df (DataFrame): Daily profits with both real transfers and imputed period boundary rows
#     - market_data_df (DataFrame): Price and market cap data

#     Returns:
#     - market_cap_features_df (DataFrame): Market cap features indexed on wallet_address

#     """
#     logger.debug("Calculating market cap features...")

#     # Confirm that all coin_id-date pairs with profits_df records have market data available
#     profits_pairs = set(profits_df[['coin_id', 'date']].apply(tuple, axis=1))
#     market_cap_pairs = set(market_data_df[['coin_id', 'date']].apply(tuple, axis=1))
#     missing_in_market_cap = profits_pairs - market_cap_pairs
#     if missing_in_market_cap:
#         raise ValueError(f"Missing coin_id-date pairs in market_cap_df: {missing_in_market_cap}")

#     # Force fill market cap gaps
#     filled_market_cap_df = force_fill_market_cap(market_data_df)

#     # Merge market cap data and sort
#     profits_market_cap_df = profits_df.merge(
#         filled_market_cap_df[['date', 'coin_id', 'market_cap_filled']],
#         on=['date', 'coin_id'],
#         how='inner'
#     )
#     profits_market_cap_df = profits_market_cap_df.sort_values(by=['coin_id', 'wallet_address', 'date'])

#     # Calculate ending balance weighted metrics using period end rows
#     profits_ending_balances_df = profits_market_cap_df.drop_duplicates(subset=['coin_id', 'wallet_address'],
#                                                                        keep='last')
#     ending_balance_wtd_df = calculate_ending_balance_weighted_market_cap(profits_ending_balances_df)

#     # Calculate volume-weighted metrics using only real transfers
#     volume_wtd_df = calculate_volume_weighted_market_cap(profits_market_cap_df[~profits_market_cap_df['is_imputed']])

#     # Merge into a wallet-indexed df of features
#     market_cap_features_df = volume_wtd_df.copy()
#     market_cap_features_df = market_cap_features_df.join(ending_balance_wtd_df)

#     return market_cap_features_df
