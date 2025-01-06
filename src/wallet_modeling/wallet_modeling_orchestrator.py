"""Orchestrates groups of functions to generate wallet model pipeline"""
import time
import logging
from typing import Tuple,Optional,Dict
import pandas as pd
import numpy as np

# Local module imports
import training_data.profits_row_imputation as pri
import coin_wallet_metrics.indicators as ind
import wallet_modeling.wallet_training_data as wtd
import wallet_features.trading_features as wtf
from wallet_modeling.wallets_config_manager import WalletsConfig
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()



@u.timing_decorator
def retrieve_period_datasets(period_start_date, period_end_date, coin_cohort=None, parquet_prefix=None):
    """
    Retrieves and processes data for a specific period. If coin_cohort provided,
    filters to those coins. Otherwise applies full cleaning pipeline to establish cohort.

    Params:
    - period_start_date,period_end_date (str): Period boundaries
    - coin_cohort (set, optional): Coin IDs from training cohort
    - parquet_prefix (str, optional): Prefix for saved parquet files

    Returns:
    - tuple: (profits_df, market_data_df, coin_cohort) for the period
    """
    # Get raw period data
    profits_df, market_data_df = wtd.retrieve_raw_datasets(period_start_date, period_end_date)

    # Apply cleaning process including coin cohort filter if specified
    market_data_df = wtd.clean_market_dataset(market_data_df, profits_df,
                                                period_start_date, period_end_date,
                                                coin_cohort)
    profits_df = profits_df[profits_df['coin_id'].isin(market_data_df['coin_id'])]

    # Set the coin_cohort if it hadn't already been passed
    if not coin_cohort:
        coin_cohort = set(market_data_df['coin_id'].unique())
        logger.info("Defined coin cohort of %s coins after applying data cleaning filters.",
                    len(coin_cohort))

    # Impute the period end (period start is pre-imputed during profits_df generation)
    imputed_profits_df = pri.impute_profits_for_multiple_dates(profits_df, market_data_df,
                                                               [period_end_date], n_threads=1)

    # Format and optionally save the datasets
    profits_df_formatted, market_data_df_formatted = wtd.format_and_save_datasets(
        imputed_profits_df,
        market_data_df,
        period_start_date,
        parquet_prefix
    )

    return profits_df_formatted, market_data_df_formatted, coin_cohort



@u.timing_decorator
def define_training_wallet_cohort(profits_df: pd.DataFrame,
                                  market_data_df: pd.DataFrame,
                                ) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Orchestrates the definition of a wallet cohort for model training by:
    1. Imputing profits at period boundaries
    2. Calculating wallet-level trading metrics
    3. Filtering wallets based on behavior thresholds
    4. Uploading filtered cohort to BigQuery

    Params:
        profits_df (DataFrame): Historical profit and balance data for all wallets
        market_data_df (DataFrame): Market prices and metadata for relevant period

    Returns:
        training_cohort_profits_df (DataFrame): Profits data filtered to selected wallets
        training_wallet_cohort (ndarray): Array of wallet addresses that pass filters
    """
    start_time = time.time()
    training_period_start = wallets_config['training_data']['training_period_start']
    training_period_end = wallets_config['training_data']['training_period_end']

    # Impute the training period end (training period start is pre-imputed into profits_df generation)
    imputed_profits_df = pri.impute_profits_for_multiple_dates(profits_df, market_data_df,
                                                               [training_period_end], n_threads=24)

    # Create a training period only profits_df
    training_profits_df = (
        imputed_profits_df[imputed_profits_df['date'] <= training_period_end]
        .copy()
    )

    # Confirm valid dates for training period
    u.assert_period(profits_df, training_period_start, training_period_end)
    u.assert_period(market_data_df, training_period_start, training_period_end)

    # Compute wallet level metrics over duration of training period
    training_wallet_metrics_df = wtf.calculate_wallet_trading_features(training_profits_df,
                                                                       training_period_start,
                                                                       training_period_end,
                                                                       calculate_twb_metrics=False)

    # Apply filters based on wallet behavior during the training period
    filtered_training_wallet_metrics_df = wtd.apply_wallet_thresholds(training_wallet_metrics_df)
    training_wallet_cohort = filtered_training_wallet_metrics_df.index.values

    # Upload the cohort to BigQuery for additional complex feature generation
    wtd.upload_wallet_cohort(training_wallet_cohort)
    logger.info("Training wallet cohort defined as %s wallets after %.2f seconds.",
                len(training_wallet_cohort), time.time()-start_time)

    # Create a profits_df that only includes the wallet cohort
    training_cohort_profits_df = training_profits_df[training_profits_df['wallet_address'].isin(training_wallet_cohort)]

    return training_cohort_profits_df, training_wallet_cohort



@u.timing_decorator
def split_training_window_profits_dfs(training_profits_df,training_market_data_df,wallet_cohort):
    """
    Adds imputed rows at the start and end date of all windows
    """
    # Filter to only wallet cohort
    cohort_profits_df = training_profits_df[training_profits_df['wallet_address'].isin(wallet_cohort)]

    # Impute all training window dates
    training_window_boundary_dates = wtd.generate_training_window_imputation_dates()
    training_windows_profits_df = pri.impute_profits_for_multiple_dates(cohort_profits_df,
                                                                        training_market_data_df,
                                                                        training_window_boundary_dates,
                                                                        n_threads=1)

    # Split profits_df into training windows
    training_windows_profits_dfs = wtd.split_training_window_dfs(training_windows_profits_df)

    return training_windows_profits_dfs



@u.timing_decorator
def generate_training_indicators_df(training_market_data_df_full,wallets_metrics_config,
                                    parquet_filename="training_market_indicators_data_df",
                                    parquet_folder="temp/wallet_modeling_dfs"):
    """
    Adds the configured indicators to the training period market_data_df and stores it
    as a parquet file by default (or returns it).

    Default save location: temp/wallet_modeling_dfs/market_indicators_data_df.parquet

    Params:
    - training_market_data_df_full (df): market_data_df with complete historical data, because indicators can
        have long lookback periods (e.g. SMA 200)
    - wallets_metrics_config (dict): metrics_config.py compatible metrics definitions
    - parquet_file, parquet_folder (strings): if these have values, the output df will be saved to this
        location instead of being returned

    Returns:
    - market_indicators_data_df (df): market_data_df for the training period only

    """
    logger.info("Beginning indicator generation process...")

    # Validate that no records exist after the training period
    training_period_end = wallets_config['training_data']['training_period_end']
    latest_market_data_record = training_market_data_df_full['date'].max()
    if latest_market_data_record > pd.to_datetime(training_period_end):
        raise ValueError(
            f"Detected data after the end of the training period in training_market_data_df_full."
            f"Latest record found: {latest_market_data_record} vs period end of {training_period_end}"
        )

    # Adds time series ratio metrics that can have additional indicators applied to them
    market_indicators_data_df = ind.add_market_data_dualcolumn_indicators(training_market_data_df_full)

    # Adds indicators to all configured time series
    market_indicators_data_df = ind.generate_time_series_indicators(market_indicators_data_df,
                                                            wallets_metrics_config['time_series']['market_data'],
                                                            'coin_id')

    # Filters out pre-training period records now that we've computed lookback and rolling metrics
    market_indicators_data_df = market_indicators_data_df[market_indicators_data_df['date']
                                                    >=wallets_config['training_data']['training_starting_balance_date']]

    # Reset OBV to 0 at training start if it exists
    training_start = pd.to_datetime(wallets_config['training_data']['training_starting_balance_date'])
    if 'obv' in market_indicators_data_df.columns:
        # Group by coin_id since OBV is coin-specific
        for coin_id in market_indicators_data_df['coin_id'].unique():
            mask = (market_indicators_data_df['coin_id'] == coin_id) & \
                  (market_indicators_data_df['date'] >= training_start)
            coin_idx = market_indicators_data_df[mask].index
            if len(coin_idx) > 0:
                # Reset OBV to start from 0 for each coin's training period
                market_indicators_data_df.loc[coin_idx, 'obv'] -= \
                    market_indicators_data_df.loc[coin_idx[0], 'obv']

    # If a parquet file location is specified, store the files there and return nothing
    if parquet_filename:
        parquet_filepath = f"{parquet_folder}/{parquet_filename}.parquet"
        market_indicators_data_df.to_parquet(parquet_filepath,index=False)
        logger.info(f"Stored market_indicators_data_df with shape {market_indicators_data_df.shape} "
                    f"to {parquet_filepath}.")

        return None

    # If no parquet file is configured then return the df
    else:
        return market_indicators_data_df



@u.timing_decorator
def identify_modeling_cohort(modeling_period_profits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds boolean flag indicating if wallet meets modeling period activity criteria

    Params:
    - modeling_period_profits_df (DataFrame): Input profits data with index wallet_address

    Returns:
    - DataFrame: Original dataframe index wallet_address with added boolean in_wallet_cohort \
        column that indicates if the wallet met the wallet cohort thresholds
    """

    logger.info("Identifying modeling cohort...")

    # Validate date range
    u.assert_period(modeling_period_profits_df,
                    wallets_config['training_data']['modeling_period_start'],
                    wallets_config['training_data']['modeling_period_end'])

    # Calculate modeling period wallet metrics
    modeling_wallets_df = wtf.calculate_wallet_trading_features(modeling_period_profits_df,
                                            wallets_config['training_data']['modeling_period_start'],
                                            wallets_config['training_data']['modeling_period_end'],
                                            calculate_twb_metrics=True)

    # Extract thresholds
    modeling_min_investment = wallets_config['modeling']['modeling_min_investment']
    modeling_min_coins_traded = wallets_config['modeling']['modeling_min_coins_traded']

    # Create boolean mask for qualifying wallets
    meets_criteria = (
        (modeling_wallets_df['max_investment'] >= modeling_min_investment) &
        (modeling_wallets_df['unique_coins_traded'] >= modeling_min_coins_traded)
    )

    # Log stats about wallet cohort
    total_wallets = len(modeling_wallets_df)
    qualifying_wallets = meets_criteria.sum()
    logger.info(
        f"Identified {qualifying_wallets} qualifying wallets ({100*qualifying_wallets/total_wallets:.2f}% "
        f"of {total_wallets} total wallets with modeling period activity) meeting modeling cohort criteria: "
        f"min_investment=${modeling_min_investment}, min_days={modeling_min_coins_traded}"
    )

    # Add boolean flag column as 1s and 0s
    modeling_wallets_df['in_modeling_cohort'] = meets_criteria.astype(int)


    return modeling_wallets_df



# -----------------------------------
#   Hybrid Index Utility Functions
# -----------------------------------

def hybridize_wallet_address(
    df: pd.DataFrame,
    hybrid_cw_id_map: Optional[Dict[Tuple[int, str], int]] = None
) -> Tuple[pd.DataFrame, Dict[Tuple[int, str], int]]:
    """
    Maps wallet_address-coin_id pairs to unique integers for efficient indexing.

    Params:
    - df (DataFrame): dataframe with columns ['coin_id','wallet_address']
    - hybrid_cw_id_map (dict): mapping of (wallet,coin) tuples to integers

    Returns:
    - df (DataFrame): input df with hybrid integer keys
    - hybrid_cw_id_map (dict): mapping of (wallet,coin) tuples to integers
    """
    # Create unique mapping for wallet-coin pairs
    unique_pairs = list(zip(df['wallet_address'], df['coin_id']))

    # Generate new mapping if none was provided
    if hybrid_cw_id_map is None:
        hybrid_cw_id_map = {pair: idx for idx, pair in enumerate(set(unique_pairs), 1)}

    # Vectorized mapping of pairs to integers
    df['wallet_address'] = pd.Series(unique_pairs).map(hybrid_cw_id_map)

    return df, hybrid_cw_id_map



def dehybridize_wallet_address(
    df: pd.DataFrame,
    hybrid_cw_id_map: Dict[Tuple[int, str], int]
) -> pd.DataFrame:
    """
    Restores original wallet_address-coin_id pairs from hybrid integer keys.

    Params:
    - df (DataFrame): dataframe with hybrid integer keys in wallet_address
    - hybrid_cw_id_map (dict): mapping of (wallet,coin) tuples to integers

    Returns:
    - df (DataFrame): input df with original wallet_address restored
    """
    # Create reverse mapping
    reverse_map = {v: k for k, v in hybrid_cw_id_map.items()}

    # Vectorized mapping back to original tuples
    df['wallet_address'] = df['wallet_address'].map(reverse_map).map(lambda x: x[0])

    return df
