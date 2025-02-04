import logging
import numpy as np
import pandas as pd

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
def calculate_market_cap_features(profits_df, market_data_df):
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
    # 1. Identify relevant columns and modify dfs as needed
    market_cap_cols = wallets_config['features']['market_cap_feature_columns']
    profits_df = profits_df.copy()
    market_data_df = market_data_df.copy()

    # Force fill market cap gaps if applicable
    if 'market_cap_filled' in market_cap_cols and 'market_cap_filled' not in market_data_df.columns:
        market_data_df = force_fill_market_cap(
            market_data_df,
            wallets_config['data_cleaning']['market_cap_default_fill']
        )

    # Alias the base column if applicable
    if 'market_cap_unadj' in market_cap_cols:
        market_data_df['market_cap_unadj'] = market_data_df['market_cap']

    # Confirm all market cap columns exist in the df
    missing_cols = [col for col in market_cap_cols if col not in market_data_df.columns]
    if missing_cols:
        raise ValueError(f"The following columns are missing from the DataFrame: {missing_cols}")


    # 2. Merge market cap data onto profits_df
    profits_market_cap_df = profits_df.merge(
        market_data_df[['date', 'coin_id'] + market_cap_cols],
        on=['date', 'coin_id'],
        how='left',
        indicator=True)

    # Confirm all pairs matched
    missing_pairs = profits_market_cap_df[profits_market_cap_df['_merge'] == 'left_only'][['coin_id', 'date']]
    if not missing_pairs.empty:
        raise ValueError(f"Missing coin_id-date pairs in market_cap_df:\n{missing_pairs}")
    profits_market_cap_df = profits_market_cap_df.drop(columns=['_merge'])


    # 3. Generate market cap features for all market cap columns
    # Precompute end balance profits_df
    profits_market_cap_df_end = (profits_market_cap_df.copy()
                                 .drop_duplicates(subset=['coin_id', 'wallet_address'],
                                                  keep='last'))

    all_features = []
    for col in market_cap_cols:
        # Calculate both feature sets
        balance_wtd = calculate_ending_balance_weighted_market_cap(profits_market_cap_df_end, col)
        volume_wtd = calculate_volume_weighted_market_cap(profits_market_cap_df, col)
        distribution = calculate_distribution_features(profits_market_cap_df_end, col)

        # Add suffix to identify source column
        suffix = f"/{col}"
        features_df = balance_wtd
        features_df = features_df.join(volume_wtd, how='inner')
        features_df = features_df.join(distribution, how='inner')
        features_df = features_df.add_suffix(suffix)

        all_features.append(features_df)

    # Join all feature sets together
    market_cap_features_df = pd.concat(all_features, axis=1)

    return market_cap_features_df


# ------------------------------
#         Helper Functions
# ------------------------------

def force_fill_market_cap(market_data_df, default_market_cap_fill):
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
    market_data_df['market_cap_filled'] = market_data_df['market_cap_filled'].fillna(default_market_cap_fill)

    return market_data_df



def calculate_volume_weighted_market_cap(profits_market_features_df, market_cap_col):
    """
    Calculate volume-weighted average market cap for each wallet address.
    If volume is 0 for all records of a wallet, uses simple average instead.

    Parameters:
    - profits_market_features_df (df): DataFrame containing wallet_address, volume, and market_cap_col
    - market_cap_col (str): the column with market cap data in it

    Returns:
    pandas.DataFrame: DataFrame with wallet addresses and column 'volume_wtd_market_cap'
    """
    # Create a copy to avoid modifying original
    df_calc = profits_market_features_df.copy()

    # Remove any records of coins that do not have market cap data available
    df_calc = df_calc.dropna(subset=[market_cap_col])

    # Volume is equal to absolute value of USD transfers
    df_calc['volume'] = abs(df_calc['usd_net_transfers'])

    # Multiply volume by mc for weighted average calculations
    df_calc['market_cap_volume'] = df_calc['volume'] * df_calc[market_cap_col]

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



def calculate_ending_balance_weighted_market_cap(profits_df_end, market_cap_col):
    """
    Calculate USD balance-weighted average market cap for each wallet address
    using only the most recent date's data.

    Parameters:
    - profits_df_end (df): profits_df with column market_cap_col on the period end date
    - market_cap_col (str): the column with market cap data in it

    Returns:
    - wallet_end_balance_wtd_mc_df (df): DataFrame with index wallet_address and column
        'end_portfolio_wtd_market_cap' showing avg mc weighted by ending balance
    """
    # Retrieve the ending balances of all coin-wallet pairs
    profits_df_end = profits_df_end.copy().dropna(subset=[market_cap_col])

    # Compute the ending market cap dollars
    profits_df_end['market_cap_balance'] = (profits_df_end['usd_balance']
                                            * profits_df_end[market_cap_col])

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



def calculate_distribution_features(profits_df_end: pd.DataFrame, market_cap_col: str) -> pd.DataFrame:
    """
    Params:
    - profits_df_end (df): profits_df with column market_cap_col on the period end date
    - market_cap_col (str): Name of the column with market cap data.

    Returns:
    - features_df (DataFrame): Wallet-level features with columns:
        - portfolio_mcap_std: Std dev of market caps for final-held coins.
        - concentration_index: Sum of squared position weights (Herfindahl-Hirschman).
        - largest_coin_frac: Fraction of total balance in the largest single coin.
        - largest_coin_usd: Absolute USD balance of the largest single coin.
    """
    # Retrieve the ending balances of all coin-wallet pairs
    profits_df_end = profits_df_end.copy().dropna(subset=[market_cap_col])

    # Compute total balance per wallet
    profits_df_end["total_usd_balance"] = profits_df_end.groupby(
        "wallet_address",
        observed=True
    )["usd_balance"].transform("sum")

    # Filter out zero-balance wallets
    profits_df_end = profits_df_end[profits_df_end["total_usd_balance"] > 0]

    # Aggregate to wallet level
    features_df = profits_df_end.groupby("wallet_address", observed=True).agg(
        portfolio_mcap_mean=(market_cap_col, "mean"),
        portfolio_mcap_median=(market_cap_col, "median"),
        portfolio_mcap_min=(market_cap_col, "min"),
        portfolio_mcap_max=(market_cap_col, "max"),
        portfolio_mcap_std=(market_cap_col, "std"),
    )

    return features_df
