"""
Calculates metrics aggregated at the wallet-coin-date level
"""
import logging
import numpy as np

# Local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig
import utils as u

# set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()



# ------------------------------
#         Core Interface
# ------------------------------

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
    - market_cap_features_df (DataFrame): Market cap features indexed on wallet_address

    """
    # Copy dfs
    profits_df = profits_df.copy()
    market_data_df = market_data_df.copy()

    # Data quality check: all profits_df coin-date records have market data
    missing_pairs = ~profits_df.set_index(['coin_id', 'date']).index.isin(
        market_data_df.set_index(['coin_id', 'date']).index
    )
    if missing_pairs.any():
        problematic_pairs = profits_df[missing_pairs][['coin_id', 'date']]
        raise ValueError(f"Missing coin_id-date pairs in market_cap_df:\n{problematic_pairs}")

    # Force fill market cap gaps
    filled_market_cap_df = force_fill_market_cap(market_data_df)

    # Merge market cap data
    profits_market_cap_df = profits_df.merge(
        filled_market_cap_df[['date', 'coin_id', 'market_cap_imputed', 'market_cap_filled']],
        on=['date', 'coin_id'],
        how='inner'
    )

    # Calculate ending balance weighted metrics
    wallet_end_balance_wtd_mc_df = calculate_ending_balance_weighted_market_cap(profits_market_cap_df)

    # Calculate volume-weighted metrics using only real transfers
    volume_wtd_df = calculate_volume_weighted_market_cap(profits_market_cap_df)

    # Merge into a wallet-indexed df of features
    market_cap_features_df = wallet_end_balance_wtd_mc_df.join(volume_wtd_df,how='inner')

    return market_cap_features_df


# ------------------------------
#         Helper Functions
# ------------------------------

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
    pandas.DataFrame: DataFrame with wallet addresses and column 'volume_wtd_market_cap'
    """
    # Create a copy to avoid modifying original
    df_calc = profits_market_features_df.copy()

    # Volume is equal to absolute value of USD transfers
    df_calc['volume'] = abs(df_calc['usd_net_transfers'])

    # Multiply volume by mc for weighted average calculations
    df_calc['market_cap_volume'] = df_calc['volume'] * df_calc['market_cap_filled']


    # Calculate each wallet's total ending crypto balance and market cap dollars
    wallet_volume_wtd_mc_df = df_calc.groupby('wallet_address',observed=True).agg(
        total_volume=('volume','sum'),
        total_market_cap_volume=('market_cap_volume','sum')
    )

    wallet_volume_wtd_mc_df['volume_wtd_market_cap'] =  np.where(
        # set value to nan if there is not volume
        wallet_volume_wtd_mc_df['total_volume'] == 0, np.nan,

        # calculate weighted average if there is volume
        (wallet_volume_wtd_mc_df['total_market_cap_volume']
        / wallet_volume_wtd_mc_df['total_volume'])
    )

    return wallet_volume_wtd_mc_df[['volume_wtd_market_cap']]


def calculate_ending_balance_weighted_market_cap(profits_market_cap_df):
    """
    Calculate USD balance-weighted average market cap for each wallet address
    using only the most recent date's data.

    Parameters:
    - profits_market_cap_df (df): profits_df with a 'market_cap_filled' column added

    Returns:
    - wallet_end_balance_wtd_mc_df (df): DataFrame with index wallet_address and column
        'end_portfolio_wtd_market_cap' showing avg mc weighted by ending balance
    """
    # Create a copy to avoid modifying original
    profits_df_end = profits_market_cap_df.copy()

    # Retrieve the ending balances of all coin-wallet pairs
    profits_df_end = profits_df_end.drop_duplicates(subset=['coin_id', 'wallet_address'],
                                                    keep='last')

    # Compute the ending market cap dollars
    profits_df_end['market_cap_balance'] = (profits_df_end['usd_balance']
                                            * profits_df_end['market_cap_filled'])

    # Calculate each wallet's total ending crypto balance and market cap dollars
    wallet_end_balance_wtd_mc_df = profits_df_end.groupby('wallet_address',observed=True).agg(
        total_usd_balance=('usd_balance','sum'),
        total_market_cap_balance=('market_cap_balance','sum')
    )

    # Divide to find weighted average market cap
    wallet_end_balance_wtd_mc_df['end_portfolio_wtd_market_cap'] = np.where(
        # indicates to xgboost there is no value
        wallet_end_balance_wtd_mc_df['total_usd_balance'] == 0, np.nan,

        # finds weighted average
        (wallet_end_balance_wtd_mc_df['total_market_cap_balance']
            / wallet_end_balance_wtd_mc_df['total_usd_balance'])
    )

    return wallet_end_balance_wtd_mc_df[['end_portfolio_wtd_market_cap']]
