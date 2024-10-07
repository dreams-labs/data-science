"""
functions used to perform high level analysis and performance assessments
"""
# pylint: disable=C0103 # X_train violates camelcase
# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false

import os
import sys
from datetime import timedelta
import hashlib
import json
import pandas as pd
import dreams_core.core as dc

# pylint: disable=E0401  # unable to import modules from parent folders
# pylint: disable=C0413  # wrong import position
# project files
import utils as u
import coin_wallet_metrics as cwm
import feature_engineering as fe
import modeling as m
sys.path.append('..')
from training_data import data_retrieval as dr
from training_data import profits_row_imputation as ri



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



def build_time_window_model_input(n, window, config, metrics_config, modeling_config):
    """
    Generates training data for each of the config.training_data.additional_windows.

    Params:
        n (int): The lookback number of the time window (e.g 0,1,2)
        window (Dict): The config override dict with the window's modeling_period_start
        config: config.yaml
        metrics_config: metrics_config.yaml
        modeling_config: modeling_config.yaml

    Returns:
        model_data (Dict): Dictionary containing all of the modeling features and variables:
            X_train, X_test (DataFrame): Model training features
            y_train, y_test (pd.Series): Model target variables
            returns_test (DataFrame): The actual returns of each coin_id in each time_window.
                - coin_id: Index (str)
                - time_window: Index (int)
                - returns: value column (float)
    """

    # Prepare the full configuration by applying overrides from the current trial config
    config, metrics_config, modeling_config = prepare_configs(modeling_config['modeling']['config_folder'], window)

    # Define window start and end dates
    start_date = config['training_data']['training_period_start']
    end_date = config['training_data']['modeling_period_end']

    # Rebuild market data
    market_data_df = dr.retrieve_market_data()
    market_data_df, _ = cwm.split_dataframe_by_coverage(market_data_df, start_date, end_date, id_column='coin_id')
    prices_df = market_data_df[['coin_id','date','price']].copy()

    # Retrieve macro trends data
    macro_trends_df = dr.retrieve_macro_trends_data()
    macro_trends_df = cwm.generate_macro_trends_features(macro_trends_df, config)

    # Rebuild profits_df
    if 'profits_df' not in locals():
        profits_df = None
    profits_df = rebuild_profits_df_if_necessary(config, prices_df, profits_df)

    # Build the configured model input data for the nth window
    X_train, X_test, y_train, y_test, returns_test = build_configured_model_input(
                                        profits_df,
                                        market_data_df,
                                        macro_trends_df,
                                        config,
                                        metrics_config,
                                        modeling_config)

    # Add time window indices to dfs with coin_ids
    X_train['time_window'] = n
    X_train.set_index('time_window', append=True, inplace=True)
    X_test['time_window'] = n
    X_test.set_index('time_window', append=True, inplace=True)
    returns_test['time_window'] = n
    returns_test.set_index('time_window', append=True, inplace=True)

    model_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'returns_test': returns_test
    }

    return model_data




def prepare_configs(config_folder, override_params):
    """
    Loads config files from the config_folder using load_config and applies overrides specified
    in override_params.

    Args:
    - config_folder (str): Path to the folder containing the configuration files.
    - override_params (dict): Dictionary of flattened parameters to override in the loaded configs.

    Returns:
    - config (dict): The main config file with overrides applied.
    - metrics_config (dict): The metrics configuration with overrides applied.
    - modeling_config (dict): The modeling configuration with overrides applied.

    Raises:
    - KeyError: if any key from override_params does not match an existing key in the
        corresponding config.
    """

    # Load the main config files using load_config
    config_path = os.path.join(config_folder, 'config.yaml')
    metrics_config_path = os.path.join(config_folder, 'metrics_config.yaml')
    modeling_config_path = os.path.join(config_folder, 'modeling_config.yaml')

    config = u.load_config(config_path)
    metrics_config = u.load_config(metrics_config_path)
    modeling_config = u.load_config(modeling_config_path)

    # Apply the flattened overrides to each config
    for full_key, value in override_params.items():
        if full_key.startswith('config.'):
            # Validate the key exists before setting the value
            validate_key_in_config(config, full_key[len('config.'):])
            set_nested_value(config, full_key[len('config.'):], value)
        elif full_key.startswith('metrics_config.'):
            # Validate the key exists before setting the value
            validate_key_in_config(metrics_config, full_key[len('metrics_config.'):])
            set_nested_value(metrics_config, full_key[len('metrics_config.'):], value)
        elif full_key.startswith('modeling_config.'):
            # Validate the key exists before setting the value
            validate_key_in_config(modeling_config, full_key[len('modeling_config.'):])
            set_nested_value(modeling_config, full_key[len('modeling_config.'):], value)
        else:
            raise ValueError(f"Unknown config section in key: {full_key}")

    # reapply the period boundary dates based on the current config['training_data'] params
    period_dates = u.calculate_period_dates(config['training_data'])
    config['training_data'].update(period_dates)

    return config, metrics_config, modeling_config

# helper function for prepare_configs()
def set_nested_value(config, key_path, value):
    """
    Sets a value in a nested dictionary based on a flattened key path.

    Args:
    - config (dict): The configuration dictionary to update.
    - key_path (str): The flattened key path (e.g., 'config.data_cleaning.inflows_filter').
    - value: The value to set at the given key path.
    """
    keys = key_path.split('.')
    sub_dict = config
    for key in keys[:-1]:  # Traverse down to the second-to-last key
        sub_dict = sub_dict.setdefault(key, {})
    sub_dict[keys[-1]] = value  # Set the value at the final key

# helper function for prepare_configs
def validate_key_in_config(config, key_path):
    """
    Validates that a given key path exists in the nested configuration.

    Args:
    - config (dict): The configuration dictionary to validate.
    - key_path (str): The flattened key path to check.
        (e.g. 'config.data_cleaning.inflows_filter')

    Raises:
    - KeyError: If the key path does not exist in the config.
    """
    keys = key_path.split('.')
    sub_dict = config
    for key in keys[:-1]:  # Traverse down to the second-to-last key
        if key not in sub_dict:
            raise KeyError(
                f"Key '{key}' not found in config at level '{'.'.join(keys[:-1])}'")
        sub_dict = sub_dict[key]
    if keys[-1] not in sub_dict:
        raise KeyError(
            f"Key '{keys[-1]}' not found in config at final level '{'.'.join(keys[:-1])}'")


def identify_imputation_dates(config):
    """
    Identifies all dates that must be imputed into profits_df.

    Params:
    - config (dict): config.yaml

    Returns:
    - required_dates (list of strings): list of all dates that need records showing unrealized
        profits as of that date
    """
    # Basic period boundary dates
    period_boundary_dates = [
        config['training_data']['modeling_period_end'],
        config['training_data']['modeling_period_start'],
        config['training_data']['training_period_end'],
        config['training_data']['training_period_start'],
    ]

    # Identify all unique cohort lookback periods
    cohort_lookback_periods = [
        cohort['lookback_period']
        for cohort in config['datasets']['wallet_cohorts'].values()
    ]

    # Determine the actual dates of the lookback period starts in this time window
    training_period_start = pd.to_datetime(config['training_data']['training_period_start'])
    lookback_start_dates = []
    for lbp in set(cohort_lookback_periods):
        lbp_start = training_period_start - timedelta(days=lbp)
        lookback_start_dates.append(lbp_start.strftime('%Y-%m-%d'))

    # Return combined list
    required_dates = period_boundary_dates + lookback_start_dates

    return required_dates


# module level config_cache dictionary for rebuild_profits_df_if_necessary()
config_cache = {"hash": None}

def rebuild_profits_df_if_necessary(config, prices_df, profits_df=None):
    """
    Checks if the config has changed and reruns time-intensive steps if needed.
    Args:
    - config (dict): The config containing training_data and data_cleaning.
    - prices_df (DataFrame): The prices dataframe needed to compute profits_df.
    - profits_df (DataFrame, optional): The profits dataframe passed in memory.
    Returns:
    - profits_df (DataFrame): The profits dataframe.
    """
    # Combine 'training_data' and 'data_cleaning' for a single hash
    relevant_configs = {**config['training_data'], **config['data_cleaning']}
    config_hash = generate_config_hash(relevant_configs)

    # If the hash hasn't changed and profits_df is passed, skip rerun
    if config_hash == config_cache["hash"] and profits_df is not None:
        logger.info("Using passed profits_df from memory.")
        return profits_df

    # Otherwise, rerun time-intensive steps
    if profits_df is None:
        logger.info("No profits_df found, rebuilding profits_df...")
    else:
        logger.info("Config changes detected, rebuilding profits_df...")

    # retrieve profits data
    profits_df = dr.retrieve_profits_data(config['training_data']['training_period_start'],
                                          config['training_data']['modeling_period_end'],
                                          config['data_cleaning']['minimum_wallet_inflows'])
    profits_df, _ = cwm.split_dataframe_by_coverage(profits_df,
                                                    config['training_data']['training_period_start'],
                                                    config['training_data']['modeling_period_end'],
                                                    id_column='coin_id')
    profits_df, _ = dr.clean_profits_df(profits_df, config['data_cleaning'])

    # DDA355 THESE GET REMOVED
    dates_to_impute = [
        config['training_data']['training_period_end'],
        config['training_data']['modeling_period_start'],
        config['training_data']['modeling_period_end'],
    ]
    profits_df = ri.impute_profits_for_multiple_dates(
                    profits_df,
                    prices_df,
                    dates_to_impute,
                    n_threads=24)

    # Update the cache for future comparisons
    config_cache["hash"] = config_hash

    return profits_df

# helper function for rebuild_profits_df_if_necessary()
def generate_config_hash(config):
    """
    Generates a hash for a given config section.

    Args:
    - config (dict): The configuration section.

    Returns:
    - str: A hash representing the config state.
    """
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode('utf-8')).hexdigest()

# helper function for rebuild_profits_df_if_necessary()
def handle_hash(config_hash, temp_folder, operation='load'):
    """
    Handles saving or loading the config hash for checking.

    Args:
    - config_hash (str): The generated hash to save or load.
    - temp_folder (str): The folder where the hash file is stored.
    - operation (str): Either 'save' or 'load' to perform the operation.

    Returns:
    - str: The loaded hash if operation is 'load', None if it doesn't exist.
    """
    hash_file = os.path.join(temp_folder, 'config_hash.txt')

    if operation == 'load':
        if os.path.exists(hash_file):
            with open(hash_file, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return None
    elif operation == 'save':
        with open(hash_file, 'w', encoding='utf-8') as f:
            f.write(config_hash)



def build_configured_model_input(
        profits_df,
        market_data_df,
        macro_trends_df,
        config,
        metrics_config,
        modeling_config):
    """
    Build the model input data (train/test sets) based on the configuration settings.

    Args:
    - profits_df (DataFrame): DataFrame containing profits information for wallets.
    - market_data_df (DataFrame): DataFrame containing market data for coins.
    - macro_trends_df (DataFrame): DataFrame containing macro trends data for dates.
    - config (dict): Overall configuration containing details for wallet cohorts and training
        data periods.
    - metrics_config (dict): Configuration for metric generation.
    - modeling_config (dict): Configuration for model training parameters.

    Returns:
    - X_train (DataFrame): Training feature set.
    - X_test (DataFrame): Testing feature set.
    - y_train (Series): Training target variable.
    - y_test (Series): Testing target variable.
    - returns_test_df (DataFrame): The actual returns of the test set with index 'coin_id' and
        column 'returns'
    """
    training_data_tuples = []

    # 1. Generate and merge features for all datasets
    # -------------------------------------
    # Time series features
    dataset_name = 'market_data'  # update to loop through all time series
    market_data_tuples, _ = fe.generate_time_series_features(
            dataset_name,
            market_data_df,
            config,
            metrics_config,
            modeling_config)
    training_data_tuples.extend(market_data_tuples)

    # Wallet cohort features
    wallet_cohort_tuples, _ = fe.generate_wallet_cohort_features(
            profits_df,
            config,
            metrics_config,
            modeling_config)
    training_data_tuples.extend(wallet_cohort_tuples)

    # Macro trends features
    macro_trends_tuples, _ = fe.generate_macro_trends_features(
            macro_trends_df,
            config,
            metrics_config,
            modeling_config
        )
    training_data_tuples.extend(macro_trends_tuples)

    # Merge all the features
    training_data_df, _ = fe.create_training_data_df(
                            modeling_config['modeling']['modeling_folder'],
                            training_data_tuples)


    # 2. Add target variable to training_data_df
    # ------------------------------------------
    # create the target variable df
    target_variables_df, returns_df, _ = fe.create_target_variables(
                                market_data_df,
                                config['training_data'],
                                modeling_config)

    # merge the two into the final model input df
    model_input_df = fe.prepare_model_input_df(
                                training_data_df,
                                target_variables_df,
                                modeling_config['modeling']['target_column'])


    # 3. Split into train/test datasets
    # ---------------------------------
    # split the df into train and test sets
    X_train, X_test, y_train, y_test = m.split_model_input(
        model_input_df,
        modeling_config['modeling']['target_column'],
        modeling_config['modeling']['train_test_split'],
        modeling_config['modeling']['random_state']
    )
    # Create returns_df for the test population by matching the X_test index
    returns_test_df = returns_df.loc[X_test.index].copy()

    return X_train, X_test, y_train, y_test, returns_test_df
