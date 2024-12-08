"""
Orchestrates groups of functions to generate wallet model pipeline
"""

import logging
from datetime import datetime,timedelta
import pandas as pd
import numpy as np
import pandas_gbq

# Local module imports
import training_data.data_retrieval as dr
import training_data.profits_row_imputation as pri
import wallet_modeling.wallet_training_data as wtd
import wallet_modeling.wallet_modeling as wm
import wallet_features.wallet_features as wf
import wallet_features.wallet_coin_features as wcf
import wallet_features.wallet_coin_date_features as wcdf
from wallet_modeling.wallets_config_manager import WalletsConfig

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()


def retrieve_datasets():
    """
    Retrieves market and profits data
    """
    earliest_date = wallets_config['training_data']['training_period_start']
    latest_date = wallets_config['training_data']['modeling_period_end']

    # Profits: retrieve for all wallets above lifetime inflows threshold
    profits_df = dr.retrieve_profits_data(earliest_date,latest_date,
                                        wallets_config['data_cleaning']['minimum_wallet_inflows'])

    # Market data: retrieve for all coins with transfer data
    market_data_df = dr.retrieve_market_data()
    market_data_df = market_data_df[market_data_df['coin_id'].isin(profits_df['coin_id'])]

    # Clean market_data_df
    market_data_df = dr.clean_market_data(
        market_data_df,
        wallets_config,
        earliest_date,
        latest_date
    )

    # Remove the filtered coins from profits_df
    profits_df = profits_df[profits_df['coin_id'].isin(market_data_df['coin_id'])]

    return profits_df,market_data_df


def define_wallet_cohort(profits_df,market_data_df):
    """
    Applies transformations and filters to identify wallets that pass data cleaning filters
    """
    # Impute the training period boundary dates
    training_period_boundary_dates = [
        wallets_config['training_data']['training_period_start'],
        wallets_config['training_data']['training_period_end']
    ]
    imputed_profits_df = pri.impute_profits_for_multiple_dates(profits_df, market_data_df,
                                                            training_period_boundary_dates, n_threads=24)

    # Create a training period only profits_df
    training_profits_df = imputed_profits_df[
        imputed_profits_df['date']<=wallets_config['training_data']['training_period_end']
        ].copy()

    # Add cash flows logic column
    training_profits_df = wcf.add_cash_flow_transfers_logic(training_profits_df)

    # Compute wallet level metrics over duration of training period
    training_wallet_metrics_df = wf.calculate_wallet_level_metrics(training_profits_df)

    # Apply filters based on wallet behavior during the training period
    filtered_training_wallet_metrics_df = wtd.apply_wallet_thresholds(training_wallet_metrics_df)

    # Identify cohort
    wallet_cohort = filtered_training_wallet_metrics_df.index.values

    # Upload the cohort to BigQuery for additional complex feature generation
    wtd.upload_wallet_cohort(wallet_cohort)

    return filtered_training_wallet_metrics_df,wallet_cohort


def split_profits_df(profits_df,market_data_df,wallet_cohort):
    """
    Adds imputed rows at the start and end date of all windows
    """
    # Filter to only wallet cohort
    cohort_profits_df = profits_df[profits_df['wallet_address'].isin(wallet_cohort)]

    # Impute all required dates
    imputation_dates = wtd.generate_imputation_dates()
    windows_profits_df = pri.impute_profits_for_multiple_dates(cohort_profits_df, market_data_df,
                                                               imputation_dates, n_threads=24)

    # Filter to only include training window rows
    windows_profits_df = (windows_profits_df[
        (windows_profits_df['date'] >= pd.to_datetime(min(imputation_dates)))
        & (windows_profits_df['date'] <= pd.to_datetime(max(imputation_dates)))
    ])

    # Split profits_df into training windows and the modeling period
    training_windows_profits_dfs, modeling_period_profits_df =  wtd.split_window_dfs(windows_profits_df)

    return training_windows_profits_dfs, modeling_period_profits_df


def generate_wallet_performance_features(training_windows_profits_dfs,training_wallet_metrics_df,wallet_cohort):
    """
    Generates wallet financial performance features for the full training period and each window
    """
    # Create training data df with full training period metrics
    training_data_df = wf.fill_missing_wallet_data(training_wallet_metrics_df, wallet_cohort)
    training_data_df = training_data_df.add_suffix("_all_windows")

    # Generate and join dfs for each training window
    for i, window_df in enumerate(training_windows_profits_dfs, 1):
        # Add metrics
        window_df = wcf.add_cash_flow_transfers_logic(window_df)
        window_wallets_df = wf.calculate_wallet_level_metrics(window_df)

        # Fill missing values and Join to training_data_df
        window_wallets_df = wf.fill_missing_wallet_data(window_wallets_df, wallet_cohort)
        window_wallets_df = window_wallets_df.add_suffix(f'_w{i}')  # no need for i+1 now
        training_data_df = training_data_df.join(window_wallets_df, how='left')

    return training_data_df
