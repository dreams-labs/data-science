"""
Orchestrates groups of functions to generate wallet model pipeline
"""

import time
import logging
import pandas as pd

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

    if coin_cohort is not None:
        # Filter to existing cohort before processing
        profits_df = profits_df[profits_df['coin_id'].isin(coin_cohort)]
        market_data_df = market_data_df[market_data_df['coin_id'].isin(coin_cohort)]
    else:
        # Apply full cleaning to establish cohort
        market_data_df = wtd.clean_market_dataset(market_data_df, profits_df, period_start_date, period_end_date)
        profits_df = profits_df[profits_df['coin_id'].isin(market_data_df['coin_id'])]
        coin_cohort = set(market_data_df['coin_id'].unique())
        logger.info("Defined coin cohort of %s coins after applying data cleaning filters.",
                    len(coin_cohort))

    # Impute period end date
    

    # Format and optionally save the datasets
    profits_df_formatted, market_data_df_formatted = wtd.format_and_save_datasets(
        profits_df,
        market_data_df,
        period_start_date,
        parquet_prefix
    )

    return profits_df_formatted, market_data_df_formatted, coin_cohort


@u.timing_decorator
def define_wallet_cohort(profits_df,market_data_df):
    """
    Applies transformations and filters to identify wallets that pass data cleaning filters
    """
    start_time = time.time()
    logger.info("Defining wallet cohort based on cleaning params...")

    # Impute the training period end (training period start is pre-imputed into profits_df generation)
    training_period_end = [wallets_config['training_data']['training_period_end']]
    imputed_profits_df = pri.impute_profits_for_multiple_dates(profits_df, market_data_df,
                                                            training_period_end, n_threads=24)

    # Create a training period only profits_df
    training_profits_df = imputed_profits_df[
        imputed_profits_df['date']<=wallets_config['training_data']['training_period_end']
        ].copy()

    # Compute wallet level metrics over duration of training period
    logger.info("Generating training period trading features...")
    training_profits_df = wtf.add_cash_flow_transfers_logic(training_profits_df)
    training_wallet_metrics_df = wtf.calculate_wallet_trading_features(training_profits_df)

    # Apply filters based on wallet behavior during the training period
    logger.info("Identifying and uploading wallet cohort...")
    filtered_training_wallet_metrics_df = wtd.apply_wallet_thresholds(training_wallet_metrics_df)
    wallet_cohort = filtered_training_wallet_metrics_df.index.values

    # Upload the cohort to BigQuery for additional complex feature generation
    wtd.upload_wallet_cohort(wallet_cohort)

    logger.info("Cohort defined as %s wallets after %.2f seconds.",
                len(wallet_cohort), time.time()-start_time)

    return wallet_cohort



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

    # drop imputed total_return column
    training_windows_profits_df = training_windows_profits_df.drop('total_return', axis=1)

    # Split profits_df into training windows
    training_profits_df, training_windows_profits_dfs = wtd.split_training_window_dfs(training_windows_profits_df)

    return training_profits_df, training_windows_profits_dfs



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

    # Validate date range
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
                                                        >=wallets_config['training_data']['training_period_start']]

    # If a parquet file location is specified, store the files there and return nothing
    if parquet_filename:
        parquet_filepath = f"{parquet_folder}/{parquet_filename}.parquet"
        market_indicators_data_df.to_parquet(parquet_filepath,index=False)
        logger.info(f"Stored market_indicators_data_df with shape {market_indicators_data_df.shape} "
                    f"to {parquet_filepath}.")

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
    modeling_period_start = wallets_config['training_data']['modeling_period_start']
    modeling_period_end = wallets_config['training_data']['modeling_period_end']

    if not modeling_period_profits_df['date'].between(modeling_period_start, modeling_period_end).all():
        raise ValueError(
            f"Detected dates outside the modeling period range: "
            f"{modeling_period_start} to {modeling_period_end}"
        )

    # Calculate modeling period wallet metrics
    modeling_period_profits_df = wtf.add_cash_flow_transfers_logic(modeling_period_profits_df)
    modeling_wallets_df = wtf.calculate_wallet_trading_features(modeling_period_profits_df)

    # Extract thresholds
    min_modeling_investment = wallets_config['data_cleaning']['min_modeling_investment']
    min_modeling_transaction_days = wallets_config['data_cleaning']['min_modeling_transaction_days']

    # Create boolean mask for qualifying wallets
    meets_criteria = (
        (modeling_wallets_df['max_investment'] >= min_modeling_investment) &
        (modeling_wallets_df['transaction_days'] >= min_modeling_transaction_days)
    )

    # Log stats about wallet cohort
    total_wallets = len(modeling_wallets_df)
    qualifying_wallets = meets_criteria.sum()
    logger.info(
        f"Identified {qualifying_wallets} qualifying wallets ({100*qualifying_wallets/total_wallets:.2f}% "
        f"of {total_wallets} total wallets with modeling period activity) meeting modeling cohort criteria: "
        f"min_investment=${min_modeling_investment}, min_days={min_modeling_transaction_days}"
    )

    # Add boolean flag column as 1s and 0s
    modeling_wallets_df['in_modeling_cohort'] = meets_criteria.astype(int)


    return modeling_wallets_df
