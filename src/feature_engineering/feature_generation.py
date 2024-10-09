"""
functions used to build coin-level features from training data
"""
import os
import pandas as pd
import dreams_core.core as dc

# pylint: disable=E0401
# project module imports
import training_data.profits_row_imputation as pri
import coin_wallet_metrics.coin_wallet_metrics as cwm
import coin_wallet_metrics.indicators as ind
import feature_engineering.flattening as flt

# set up logger at the module level
logger = dc.setup_logger()


def generate_window_time_series_features(
        all_windows_time_series_df,
        dataset_name,
        config,
        dataset_metrics_config,
        modeling_config
    ):
    """
    Generates a window-specific feature set from the full all windows dataset. The window-specific
    features are saved as a csv and returned, along with the csv filepath.

    Params:
    - all_windows_time_series_df (DataFrame): df containing all metrics and indicators for a time
        series dataset.
    - dataset_name (string): key from config['datasets'] that will be added to the dataset's file
        and column names. If there are multiple keys, they should be split with a '-'.
            e.g. wallet_cohorts-whales, time_series-market_data
    - config (dict): config.yaml that has the dates for the specific time window
    - dataset_metrics_config (dict): The component of metrics_config relating to this dataset, e.g.
        metrics_config['time_series']['market_data']
    - modeling_config (dict): modeling_config.yaml

    Returns:
    - flattened_metrics_df (DataFrame): the flattened version of the original df, with columns for
        the configured aggregations and rolling metrics for all value columns and indicators.
    - flattened_metrics_filepath (string): the filepath to where the flattened_metrics_df is saved
    """
    # Filter input data to time window
    window_time_series_df, _ = cwm.split_dataframe_by_coverage(
        all_windows_time_series_df,
        config['training_data']['training_period_start'],
        config['training_data']['training_period_end'],
        id_column='coin_id',
        drop_outside_date_range=True
    )

    # Flatten the metrics DataFrame to be keyed only on coin_id
    flattened_metrics_df = flt.flatten_coin_date_df(
        window_time_series_df,
        dataset_metrics_config,
        config['training_data']['training_period_end']  # Ensure data is up to training period end
    )

    # Add time window modeling period start
    flattened_metrics_df.loc[:,'time_window'] = config['training_data']['modeling_period_start']

    # Add dataset_name as a prefix to all columns so their lineage is fully documented
    flattened_metrics_df = flattened_metrics_df.rename(
        columns=lambda x:
        f"{dataset_name.replace('-', '_')}_{x}"
        if x not in ['coin_id', 'time_window']
        else x)

    # Save the flattened output and retrieve the file path
    _, flattened_metrics_filepath = flt.save_flattened_outputs(
        flattened_metrics_df,
        os.path.join(
            modeling_config['modeling']['modeling_folder'],  # Folder to store flattened outputs
            'outputs/flattened_outputs'
        ),
        dataset_name,  # Descriptive metadata for the dataset
        config['training_data']['modeling_period_start']  # Ensure data starts from modeling period
    )

    return flattened_metrics_df, flattened_metrics_filepath



def generate_window_macro_trends_features(
        all_windows_macro_trends_df,
        dataset_name,
        config,
        metrics_config,
        modeling_config
    ):
    """
    Generates a window-specific feature set from the full all windows dataset. The window-specific
    features are saved as a csv and returned, along with the csv filepath.

    This function differs from the time_series set because it only flattens on date, since this
    dataset doesn't have coin_id.

    Params:
    - all_windows_time_series_df (DataFrame): df containing all metrics and indicators for a time
        series dataset.
    - dataset_name (string): key from config['datasets'] that will be added to the dataset's file
        and column names. If there are multiple keys, they should be split with a '-'.
            e.g. wallet_cohorts-whales, time_series-market_data
    - config: config.yaml that has the dates for the specific time window
    - metrics_config: metrics_config.yaml
    - modeling_config: modeling_config.yaml

    Returns:
    - flattened_metrics_df (DataFrame): the flattened version of the original df, with columns for
        the configured aggregations and rolling metrics for all value columns and indicators.
    - flattened_metrics_filepath (string): the filepath to where the flattened_metrics_df is saved
    """
    # Filter input data to time window
    window_macro_trends_df,_ = cwm.split_dataframe_by_coverage(all_windows_macro_trends_df,
                                                            config['training_data']['training_period_start'],
                                                            config['training_data']['training_period_end'],
                                                            id_column=None,
                                                            drop_outside_date_range=True)

    # Macro trends: flatten metrics
    flattened_features = flt.flatten_date_features(window_macro_trends_df,metrics_config[dataset_name])
    flattened_macro_trends_df = pd.DataFrame([flattened_features])

    # Add time window modeling period start
    flattened_macro_trends_df.loc[:,'time_window'] = config['training_data']['modeling_period_start']

    # Add dataset_name as a prefix to all columns so their lineage is fully documented
    flattened_macro_trends_df = flattened_macro_trends_df.rename(
        columns=lambda x:
        f"{dataset_name.replace('-', '_')}_{x}"
        if x not in ['coin_id', 'time_window']
        else x)

    # Save the flattened output and retrieve the file path
    _, flattened_metrics_filepath = flt.save_flattened_outputs(
        flattened_macro_trends_df,
        os.path.join(
            modeling_config['modeling']['modeling_folder'],  # Folder to store flattened outputs
            'outputs/flattened_outputs'
        ),
        dataset_name,  # Descriptive metadata for the dataset
        config['training_data']['modeling_period_start']  # Ensure data starts from modeling period
    )

    return flattened_macro_trends_df, flattened_metrics_filepath



def generate_window_wallet_cohort_features(
        profits_df,
        prices_df,
        config,
        metrics_config,
        modeling_config
    ):
    """
    Generates a window-specific feature set from the full all windows dataset. The window-specific
    features are saved as a csv and returned, along with the csv filepath.

    This function differs from the time_series set because it only flattens on date, since this
    dataset doesn't have coin_id.

    Params:
    - all_windows_time_series_df (DataFrame): df containing all metrics and indicators for a time
        series dataset.
    - config: config.yaml that has the dates for the specific time window
    - metrics_config: metrics_config.yaml
    - modeling_config: modeling_config.yaml

    Returns:
    - flattened_cohort_dfs (list of DataFrames): a list containing the flattened versions of each
        cohort's metrics, with columns for the configured aggregations and rolling metrics for all
        value columns and indicators.
    - flattened_cohorts_filepath (list of strings): a list containing the filepaths to where the
        flattened_cohort_dfs are saved
    """

    # 1. Impute all required dates
    # ----------------------------
    # Identify all required imputation dates
    imputation_dates = pri.identify_imputation_dates(config)

    # Impute all required dates
    window_profits_df = pri.impute_profits_for_multiple_dates(profits_df, prices_df, imputation_dates, n_threads=24)
    window_profits_df = (window_profits_df[(window_profits_df['date'] >= pd.to_datetime(min(imputation_dates))) &
                                        (window_profits_df['date'] <= pd.to_datetime(max(imputation_dates)))])


    # 2. Generate metrics and indicators for all cohorts
    # --------------------------------------------------
    # Set up lists to store flattened cohort data
    flattened_cohort_dfs = []
    flattened_cohort_filepaths = []

    for cohort_name in metrics_config['wallet_cohorts']:

        # load configs
        dataset_metrics_config = metrics_config['wallet_cohorts'][cohort_name]
        dataset_config = config['datasets']['wallet_cohorts'][cohort_name]

        # identify wallets in the cohort based on the full lookback period
        cohort_summary_df = cwm.classify_wallet_cohort(window_profits_df, dataset_config, cohort_name)
        cohort_wallets = cohort_summary_df[cohort_summary_df['in_cohort']]['wallet_address']

        # If no cohort members were identified, continue
        if len(cohort_wallets) == 0:
            logger.info("No wallets identified as members of cohort '%s'", cohort_name)
            continue

        # Generate cohort buysell_metrics
        cohort_metrics_df = cwm.generate_buysell_metrics_df(window_profits_df,
                                                            config['training_data']['training_period_end'],
                                                            cohort_wallets)

        # Generate cohort indicator metrics
        cohort_metrics_df = ind.generate_time_series_indicators(cohort_metrics_df,
                                                                metrics_config['wallet_cohorts'][cohort_name],
                                                                'coin_id')

        # Flatten cohort metrics
        flattened_cohort_df, flattened_cohort_filepath = generate_window_time_series_features(
            cohort_metrics_df,
            f'wallet_cohorts-{cohort_name}',
            config,
            dataset_metrics_config,
            modeling_config
        )

        flattened_cohort_dfs.extend([flattened_cohort_df])
        flattened_cohort_filepaths.extend([flattened_cohort_filepath])

    return flattened_cohort_dfs, flattened_cohort_filepaths
