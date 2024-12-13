"""
Orchestrates groups of functions to generate wallet model pipeline
"""

import time
import logging
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from dreams_core import core as dc

# Local module imports
import training_data.data_retrieval as dr
import training_data.profits_row_imputation as pri
import coin_wallet_metrics.indicators as ind
import wallet_modeling.wallet_training_data as wtd
import wallet_features.trading_features as wtf
import wallet_features.market_cap_features as wmc
from wallet_modeling.wallets_config_manager import WalletsConfig
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()


@u.timing_decorator
def retrieve_period_datasets(start_date,end_date,
                             parquet_prefix=None,parquet_folder="temp/wallet_modeling_dfs"):
    """
    Retrieves market and profits data
    """

    # Retrieve profits_df and market_data_df concurrently
    with ThreadPoolExecutor(max_workers=2) as executor:
        profits_future = executor.submit(
            dr.retrieve_profits_data,
            start_date,
            end_date,
            wallets_config['data_cleaning']['min_wallet_inflows']
        )
        market_future = executor.submit(dr.retrieve_market_data)

        profits_df = profits_future.result()
        market_data_df = market_future.result()

    # Remove all records after the training period end to ensure no data leakage
    market_data_df = market_data_df[market_data_df['date']<=end_date]

    # Clean market_data_df
    market_data_df = market_data_df[market_data_df['coin_id'].isin(profits_df['coin_id'])]
    market_data_df = dr.clean_market_data(
        market_data_df,
        wallets_config,
        start_date,
        end_date
    )

    # Intelligently impute market cap data in market_data_df when good data is available
    market_data_df = dr.impute_market_cap(market_data_df,
                                        wallets_config['data_cleaning']['min_mc_imputation_coverage'],
                                        wallets_config['data_cleaning']['max_mc_imputation_multiple'])

    # Crudely fill all remaining gaps in market cap data
    market_data_df = wmc.force_fill_market_cap(market_data_df)

    # Remove coins that exceeded the initial market cap threshold at the start of the training period
    max_initial_market_cap = wallets_config['data_cleaning']['max_initial_market_cap']
    above_initial_threshold_coins = market_data_df[
        (market_data_df['date']==wallets_config['training_data']['training_period_start'])
        & (market_data_df['market_cap_filled']>max_initial_market_cap)
    ]['coin_id']
    market_data_df = market_data_df[~market_data_df['coin_id'].isin(above_initial_threshold_coins)]
    logger.info("Removed data for %s coins with a market cap above $%s at the start of the training period."
                ,len(above_initial_threshold_coins),dc.human_format(max_initial_market_cap))

    # Remove the filtered coins from profits_df
    profits_df = profits_df[profits_df['coin_id'].isin(market_data_df['coin_id'])]

    # Clean profits_df
    profits_df, _ = dr.clean_profits_df(profits_df, wallets_config['data_cleaning'])

    # Drop unneeded columns
    columns_to_drop = ['total_return']
    profits_df = profits_df.drop(columns_to_drop,axis=1)

    # Round relevant columns
    columns_to_round = [
        'profits_cumulative'
        ,'usd_balance'
        ,'usd_net_transfers'
        ,'usd_inflows'
        ,'usd_inflows_cumulative'
    ]
    profits_df[columns_to_round] = profits_df[columns_to_round].round(2)
    profits_df[columns_to_round] = profits_df[columns_to_round].replace(-0, 0)

    # Remove rows with a rounded 0 balance and 0 transfers
    profits_df = profits_df[
        ~((profits_df['usd_balance'] == 0) &
        (profits_df['usd_net_transfers'] == 0))
    ]

    # If a parquet file location is specified, store the files there and return nothing
    if parquet_prefix:
        # Store profits
        profits_file = f"{parquet_folder}/{parquet_prefix}_profits_df_full.parquet"
        profits_df.to_parquet(profits_file,index=False)
        logger.info(f"Stored profits_df with shape {profits_df.shape} to {profits_file}.")

        # Store market data
        market_data_file = f"{parquet_folder}/{parquet_prefix}_market_data_df_full.parquet"
        market_data_df.to_parquet(market_data_file,index=False)
        logger.info(f"Stored market_data_df with shape {market_data_df.shape} to {market_data_file}.")


    # Otherwise return the dfs
    else:
        return profits_df,market_data_df


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
def split_profits_df(training_profits_df,training_market_data_df,wallet_cohort):
    """
    Adds imputed rows at the start and end date of all windows
    """
    # Filter to only wallet cohort
    cohort_profits_df = training_profits_df[training_profits_df['wallet_address'].isin(wallet_cohort)]

    # Impute all training window dates
    training_window_dates = wtd.generate_training_window_dates()
    training_windows_profits_df = pri.impute_profits_for_multiple_dates(cohort_profits_df, training_market_data_df,
                                                               training_window_dates, n_threads=24)

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
def filter_modeling_period_wallets(modeling_period_profits_df):
    """
    Applies data cleaning filters to remove modeling period wallets without sufficient activity
    """
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

    # Remove wallets with below the minimum investment threshold
    base_wallets = len(modeling_wallets_df)
    modeling_wallets_df = modeling_wallets_df[
        modeling_wallets_df['max_investment'] >= min_modeling_investment]
    logger.info("Removed %s/%s wallets with modeling period investments below $%s.",
                base_wallets - len(modeling_wallets_df), base_wallets,
                min_modeling_investment)

    # Remove wallets with transaction counts below the threshold
    base_wallets = len(modeling_wallets_df)
    modeling_wallets_df = modeling_wallets_df[
        modeling_wallets_df['transaction_days'] >= min_modeling_transaction_days]
    logger.info("Removed %s/%s wallets with modeling period transaction days below %s.",
                base_wallets - len(modeling_wallets_df), base_wallets,
                min_modeling_transaction_days)

    logger.info("Selected wallet cohort of %s using %0.2f%% the %s training cohort wallets.",
                len(modeling_wallets_df), 100*len(modeling_wallets_df)/base_wallets, base_wallets)

    return modeling_wallets_df
