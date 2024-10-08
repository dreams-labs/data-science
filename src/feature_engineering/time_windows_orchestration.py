"""
functions used to orchestrate time windows. the major steps are:
1. generating shared datasets that are unbounded by time windows
2. generating time window-specific features based off those values
3. merging all time window features together into a training data df
    ready for preprocessing
"""
import os
import re
from datetime import timedelta
from typing import List, Dict, Tuple
import pandas as pd
import dreams_core.core as dc

# pylint: disable=E0401  # unable to import modules from parent folders
# pylint: disable=C0413  # wrong import position
# project files
import training_data.data_retrieval as dr
import feature_engineering.feature_generation as fg
import feature_engineering.preprocessing as prp


# set up logger at the module level
logger = dc.setup_logger()


def generate_time_windows(config):
    """
    Generates the parameter dicts used by i.prepare_configs() to generate the full set
    of config files.

    Params:
        config (dict): config.yaml

    Returns:
        time_windows (list of dicts): a list of dicts that can be used to override the
        config.yaml settings for each time window.
    """
    start_date = pd.to_datetime(config['training_data']['modeling_period_start'])
    window_duration = timedelta(days=config['training_data']['training_period_duration'] +
                                     config['training_data']['modeling_period_duration'])

    time_windows = [
        {'config.training_data.modeling_period_start': start_date.strftime('%Y-%m-%d')}
    ]

    for _ in range(config['training_data']['additional_windows']):
        start_date -= window_duration
        time_windows.append({'config.training_data.modeling_period_start': start_date.strftime('%Y-%m-%d')})

    return time_windows



def retrieve_all_windows_training_data(config):
    """
    Retrieves, cleans and adds indicators to the all windows training data.

    Wallet cohort indicators will be generated for each window as the cohort groups will
    change based on the boundary dates of the cohort lookback_period specified in the config.

    Params:
    - config (dict): overall config.yaml without any window overrides

    Returns:
    - macro_trends_df (DataFrame): macro trends keyed on date only
    - market_data_df, prices_df (DataFrame): market data for coin-date pairs
    - profits_df (DataFrame): wallet transfer activity keyed on coin-date-wallet that will be used
        to compute wallet cohorts metrics
    """
    # 1. Data Retrieval, Cleaning, Indicator Calculation
    # --------------------------------------------------
    # Macro trends: retrieve and clean full history
    macro_trends_df = dr.retrieve_macro_trends_data()
    macro_trends_df = dr.clean_macro_trends(macro_trends_df, config)

    # Market data: retrieve and clean full history
    market_data_df = dr.retrieve_market_data()
    market_data_df = dr.clean_market_data(market_data_df, config)

    # Profits: retrieve and clean profits data spanning the earliest to latest training periods
    profits_df = dr.retrieve_profits_data(config['training_data']['earliest_cohort_lookback_start'],
                                        config['training_data']['training_period_end'],
                                        config['data_cleaning']['minimum_wallet_inflows'])
    profits_df, _ = dr.clean_profits_df(profits_df, config['data_cleaning'])


    # 2. Filtering based on dataset overlap
    # -------------------------------------
    # Filter market_data to only coins with transfers data if configured to
    if config['data_cleaning']['exclude_coins_without_transfers']:
        market_data_df = market_data_df[market_data_df['coin_id'].isin(profits_df['coin_id'])]
    # Create prices_df: lightweight reference for other functions
    prices_df = market_data_df[['coin_id','date','price']].copy()

    # Filter profits_df to remove records for any coins that were removed in data cleaning
    profits_df = profits_df[profits_df['coin_id'].isin(market_data_df['coin_id'])]


    return macro_trends_df, market_data_df, profits_df, prices_df



def generate_window_flattened_dfs(
        market_data_df,
        macro_trends_df,
        profits_df,
        prices_df,
        config,
        metrics_config,
        modeling_config
    ):
    """
    Takes the all windows datasets and filters and transforms them as needed to generate
    flattened dfs with all configured aggregations and indicators for all columns.

    Params:
    - market_data_df,macro_trends_df (DataFrames): all windows datasets with indicators added
        that will be flattened
    - profits_df,prices_df (DataFrames): all windows datasets that will be used to compute
        wallet cohorts metrics, which will then have indicators added and be flattened
    - config,metrics_config,modeling_config (dicts): full config files for the given window

    Returns:
    - window_flattened_dfs (list of DataFrames): all flattened dfs keyed on coin_id with columns
        for every specified aggregation and indicator
    - window_flattened_filepaths (list of strings): filepaths to csv versions of the flattened dfs
    """
    window_flattened_dfs = []
    window_flattened_filepaths = []

    # Market data: generate window-specific flattened metrics
    flattened_market_data_df, flattened_market_data_filepath = fg.generate_window_time_series_features(
        market_data_df,
        'market_data',
        config,
        metrics_config['time_series']['market_data'],
        modeling_config
    )
    window_flattened_dfs.extend([flattened_market_data_df])
    window_flattened_filepaths.extend([flattened_market_data_filepath])

    # Macro trends: generate window-specific flattened metrics
    flattened_macro_trends_df, flattened_macro_trends_filepath = fg.generate_window_macro_trends_features(
        macro_trends_df,
        'macro_trends',
        config,
        metrics_config,
        modeling_config
    )
    window_flattened_dfs.extend([flattened_macro_trends_df])
    window_flattened_filepaths.extend([flattened_macro_trends_filepath])

    # Cohorts: generate window-specific flattened metrics
    flattened_cohort_dfs, flattened_cohort_filepaths = fg.generate_window_wallet_cohort_features(
        profits_df,
        prices_df,
        config,
        metrics_config,
        modeling_config
    )
    window_flattened_dfs.extend(flattened_cohort_dfs)
    window_flattened_filepaths.extend(flattened_cohort_filepaths)

    return window_flattened_dfs, window_flattened_filepaths



def merge_dataset_time_windows_df(
        filepaths: List[str],
        modeling_config: Dict
    ) -> Dict[str, Tuple[pd.DataFrame, str]]:
    """
    For each dataset, concatenates the flattened dfs from all windows together and returns a
    tuple containing the merged df with its fill method.

    This function reads multiple CSV files, extracts their dataset prefixes,
    concatenates DataFrames with the same prefix, and determines the fill method for each dataset.
    It handles subgroups within datasets by prefixing column names with the subgroup name.

    Args:
        filepaths (List[str]): A list of file paths to the CSV files to be processed.
        modeling_config (Dict): Configuration dictionary containing preprocessing fill methods.

    Returns:
        concatenated_dfs (Dict[str, Tuple[pd.DataFrame, str]]): A dictionary where keys are
        dataset prefixes and values are tuples containing:
            - The concatenated DataFrame for each prefix
            - The fill method for the dataset
    """
    # Dictionary to store DataFrames grouped by dataset prefix
    grouped_dfs = {}
    fill_methods = {}

    parent_directory = os.path.join(modeling_config['modeling']['modeling_folder'],
                                    'outputs/preprocessed_outputs/')

    for filepath in filepaths:
        # Validate file existence
        if not os.path.exists(filepath):
            raise KeyError(f"File {filepath} does not exist.")

        # Extract the dataset prefix from the filename
        dataset_prefix = extract_dataset_key_from_filepath(filepath, parent_directory)

        # Read the CSV
        df = pd.read_csv(filepath)

        # Handle subgroups if present
        if '-' in dataset_prefix:
            dataset, subgroup = dataset_prefix.split('-')
            # Prefix the columns with the subgroup
            df = df.rename(
                columns={
                    col: (f'{subgroup}_' + col)
                    for col in df.columns
                    if col not in ['coin_id', 'time_window']}
            )
        else:
            dataset = dataset_prefix

        # Add the DataFrame to the appropriate group
        if dataset_prefix not in grouped_dfs:
            grouped_dfs[dataset_prefix] = []
            # Identify the fill method only once per dataset
            try:
                fill_methods[dataset_prefix] = modeling_config['preprocessing']['fill_methods'][dataset]
            except KeyError as exc:
                raise KeyError(f"Fill method not found for dataset: {dataset}") from exc

        grouped_dfs[dataset_prefix].append(df)

    # Concatenate DataFrames within each group and pair with fill method
    concatenated_dfs = {
        dataset_prefix: (pd.concat(dfs, ignore_index=True), fill_methods[dataset_prefix])
        for dataset_prefix, dfs in grouped_dfs.items()
    }

    return concatenated_dfs



def join_dataset_all_windows_dfs(concatenated_dfs):
    """
    Merges the all-windows dataframes of each dataset together according to the fill method
    specified in the model_config. The param is the format from tw.merge_dataset_time_windows_df().

    Params:
    - concatenated_dfs (Dict[str, Tuple[pd.DataFrame, str]]): A dictionary where keys are
        dataset prefixes and values are tuples containing:
            - The concatenated DataFrame for each prefix
            - The fill method for the dataset

    Returns
    - training_data_df (DataFrame): df multiindexed on time_window-coin_id, with columns for
        all flattened features for all datasets.
    - join_log_df (DataFrame): df that contains briefs logs of join methods and outcomes
    """
    # List to store all combinations
    all_coin_windows = []
    log_data = []

    # Pull a unique set of all coin_id-time_window pairs across all dfs
    for dataset, (df, _) in concatenated_dfs.items():
        if 'coin_id' in df.columns and 'time_window' in df.columns:
            combinations = df[['time_window', 'coin_id']].drop_duplicates()
            all_coin_windows.append(combinations)

    # Concatenate all combinations and remove duplicates
    training_data_df = pd.concat(all_coin_windows, ignore_index=True).drop_duplicates()
    training_data_df = training_data_df.set_index(['time_window', 'coin_id'])

    # Iterate through df_list and merge each one
    for dataset, (df, fill_method) in concatenated_dfs.items():
        rows_start = len(training_data_df)

        if df.isna().sum().sum() > 0:
            raise ValueError(f"All windows DataFrame for '{dataset}' unexpectedly contains null values.")

        if fill_method == 'drop_records':
            df = df.set_index(['time_window', 'coin_id'])
            training_data_df = training_data_df.join(df, on=['time_window', 'coin_id'], how='inner')

        elif fill_method == 'fill_zeros':
            df = df.set_index(['time_window', 'coin_id'])
            training_data_df = training_data_df.join(df, on=['time_window', 'coin_id'], how='left')
            training_data_df = training_data_df.fillna(0)

        elif fill_method == 'extend_coin_ids':
            df = df.set_index('time_window')
            training_data_df = training_data_df.join(df, on='time_window', how='inner')

        elif fill_method == 'extend_time_windows':
            df = df.set_index('coin_id')
            training_data_df = training_data_df.join(df, on='coin_id', how='inner')

        else:
            raise ValueError(f"Invalid fill method '{fill_method}' found in config.yaml.")

        rows_end = len(training_data_df)
        log_data.append({
            'dataset': dataset,
            'fill_method': fill_method,
            'rows_start': rows_start,
            'rows_end': rows_end,
            'rows_removed': rows_start - rows_end
        })

    if training_data_df.isna().sum().sum() > 0:
        raise ValueError("Merged training_data_df unexpectedly contains null values. "
                            f"See {training_data_df.isna().sum()}")

    join_log_df = pd.DataFrame(log_data)

    return training_data_df, join_log_df




def extract_dataset_key_from_filepath(filepath, parent_directory):
    """
    Helper function for merge_dataset_time_windows_df().

    Extracts the dataset key (e.g. 'time_series', 'wallet_cohorts', etc) from the
    flattened filepath.

    Params:
    - filepath (str): filepath output by generate_window_flattened_dfs().
    - parent_directory (string): the parent directory of the flattened files

    Returns:
    - dataset_key (string): dataset key that matches to the config and metrics_config
    """
    # Remove the parent directory part from the filepath
    relative_path = filepath.replace(parent_directory, '')

    # Split the remaining path into parts
    parts = relative_path.split(os.sep)

    # The filename should be the last part
    filename = parts[-1]

    # Define the date pattern regex
    date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}')

    # Find the date in the filename
    date_match = date_pattern.search(filename)

    if date_match:
        # Extract the part before the date
        dataset_key = filename[:date_match.start()].rstrip('_')
        return dataset_key
    else:
        raise ValueError(f"Could not parse dataset of file {filepath}")



def create_target_variables_for_all_time_windows(training_data_df, prices_df, config, modeling_config):
    """
    Create target variables for all time windows in training_data_df.

    Parameters:
    - training_data_df: DataFrame with multi-index (time_window, coin_id) and 'modeling_period_end' column
    - prices_df: DataFrame with 'coin_id', 'date', and 'price' columns
    - config: config.yaml
    - modeling_config: modeling_config.yaml

    Returns:
    - combined_target_variables: DataFrame with columns for 'time_window' and the configured
        target variable
    - combined_returns: DataFrame with columns 'returns' and 'time_window'
    """
    all_target_variables = []
    all_returns = []

    for time_window in training_data_df.index.get_level_values('time_window').unique():
        # Get the list of coin_ids for the current time_window
        current_coins = training_data_df.loc[time_window].index.get_level_values('coin_id').tolist()

        # Filter prices_df for the current coins
        current_prices_df = prices_df[prices_df['coin_id'].isin(current_coins)]

        # Create copy of config with time_window's modeling period dates
        current_training_data_config = config['training_data'].copy()
        current_training_data_config['modeling_period_start'] = time_window
        current_training_data_config['modeling_period_end'] = (
                pd.to_datetime(time_window) + timedelta(days=current_training_data_config['modeling_period_duration'])
                ).strftime('%Y-%m-%d')

        # Call create_target_variables function
        target_variables_df, returns_df = prp.create_target_variables(
            current_prices_df,
            current_training_data_config,
            modeling_config
        )

        # Add time_window information to the results
        target_variables_df['time_window'] = time_window
        returns_df['time_window'] = time_window

        # Store results
        all_target_variables.append(target_variables_df)
        all_returns.append(returns_df)

    # Combine and set indices for results
    combined_target_variables = pd.concat(all_target_variables, ignore_index=True)
    combined_target_variables = combined_target_variables.reset_index().set_index(['time_window','coin_id'])

    combined_returns = pd.concat(all_returns, ignore_index=False)
    combined_returns = combined_returns.reset_index().set_index(['time_window','coin_id'])

    return combined_target_variables, combined_returns
