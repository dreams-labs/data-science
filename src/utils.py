"""
utility functions use in data science notebooks
"""
import time
import sys
import gc
import os
from datetime import datetime, timedelta
from typing import List,Dict,Any
import importlib
import itertools
import logging
import warnings
import functools
import yaml
import progressbar
import pandas as pd
import numpy as np
from pydantic import ValidationError
import pygame
import dreams_core.core as dc


# pylint: disable=E0401
# pylint: disable=E0611
# pydantic config files
import config_models.config as py_c
import config_models.metrics_config as py_mc
import config_models.modeling_config as py_mo
import config_models.experiments_config as py_e

# Reload the Pydantic config models to reflect any changes made to their definitions
importlib.reload(py_c)
importlib.reload(py_mc)
importlib.reload(py_mo)
importlib.reload(py_e)

# set up logger at the module level
logger = dc.setup_logger()


def load_all_configs(config_folder):
    """
    Loads and returns all config files
    """
    config = load_config(f'{config_folder}/config.yaml')
    metrics_config = load_config(f'{config_folder}/metrics_config.yaml')
    modeling_config = load_config(f'{config_folder}/modeling_config.yaml')
    experiments_config = load_config(f'{config_folder}/experiments_config.yaml')

    # Confirm that all datasets and metrics match across config and metrics_config
    validate_config_consistency(config,metrics_config,modeling_config)

    return config, metrics_config, modeling_config, experiments_config


def load_config(file_path='../notebooks/config.yaml'):
    """
    Load configuration from a YAML file. Automatically calculates and adds period dates
    if modeling_period_start is present in the training_data section.

    Args:
        file_path (str): Path to the config file.

    Returns:
        dict: Parsed YAML configuration with calculated date fields, if applicable.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        config_dict = yaml.safe_load(file)

    # Calculate and add period boundary dates into the config['training_data'] section
    if 'training_data' in config_dict and 'modeling_period_start' in config_dict['training_data']:
        period_dates = calculate_period_dates(config_dict)
        config_dict['training_data'].update(period_dates)

    # If a pydantic definition exists for the config, return it in json format
    filename = os.path.basename(file_path)

    with warnings.catch_warnings():
        # Suppresses pydantic "serialized value may not be as expected" warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        def get_base_filename(filename, valid_prefixes):
            """Remove any valid prefixes from filename."""
            for prefix in valid_prefixes:
                if filename.startswith(prefix):
                    return filename[len(prefix):]
            return filename

        try:
            valid_prefixes = ['test_', 'wallets_']
            base_filename = get_base_filename(filename, valid_prefixes)

            config_mapping = {
                'config.yaml': py_c.MainConfig,
                'metrics_config.yaml': py_mc.MetricsConfig,
                'modeling_config.yaml': py_mo.ModelingConfig,
                'experiments_config.yaml': py_e.ExperimentsConfig
            }

            if base_filename in config_mapping:
                config_model = config_mapping[base_filename]
                config_pydantic = config_model(**config_dict)
                config = config_pydantic.model_dump(mode="json", exclude_none=True)
            else:
                raise ValueError(f"Loading failed for unknown config type '{filename}'")

        except ValidationError as e:
            # Enhanced error reporting
            error_messages = []
            logger.error("Validation Error in %s:", filename)
            for err in e.errors():
                issue = err['msg']
                location = '.'.join(str(item) for item in err['loc'][:-1])
                missing_field = err['loc'][-1]

                error_message = (
                    f"Issue: {issue}\n"
                    f"Location: {location}\n"
                    f"Bad Field: {missing_field}\n"
                )
                error_messages.append(error_message)

            # Raise custom error with all gathered messages
            error_details = f"Validation Error in {filename}:\n" + "\n".join(error_messages)
            raise ValueError(error_details) from e

    return config


# helper function for load_config
def calculate_period_dates(config):
    """
    Calculate the training and modeling period start and end dates based on the provided
    durations and the modeling period start date. The calculated dates will include both
    the start and end dates, ensuring the correct number of days for each period.

    Args:
        config (dict): config.yaml which contains:
        ['training_data']
        - 'modeling_period_start' (str): Start date of the modeling period as 'YYYY-MM-DD'.
        - 'modeling_period_duration' (int): Duration of the modeling period in days.
        - 'training_period_duration' (int): Duration of the training period in days.
        ['wallet_cohorts']
        - [{cohort_name}]['lookback_period' (int): How far back a cohort's transaction history
            needs to extend prioer to the training_period_start

    Returns:
        dict: Dictionary containing:
        - 'training_period_start' (str): Calculated start date of the training period.
        - 'training_period_end' (str): Calculated end date of the training period.
        - 'modeling_period_end' (str): Calculated end date of the modeling period.
    """
    training_data_config = config['training_data']

    # Extract the config values
    modeling_period_start = datetime.strptime(training_data_config['modeling_period_start'],
                                              '%Y-%m-%d')
    modeling_period_duration = training_data_config['modeling_period_duration']  # in days
    training_period_duration = training_data_config['training_period_duration']  # in days

    # Training and Modeling Period Dates
    # ----------------------------------
    # Calculate modeling_period_end (inclusive of the start date)
    modeling_period_end = modeling_period_start + timedelta(days=modeling_period_duration - 1)

    # Calculate training_period_end (just before modeling_period_start)
    training_period_end = modeling_period_start - timedelta(days=1)

    # Calculate training_period_start (inclusive of the start date)
    training_period_start = training_period_end - timedelta(days=training_period_duration - 1)

    # Lookback Dates
    # --------------
    # Calculate the start date of the earliest window
    window_frequency = training_data_config['time_window_frequency']
    additional_windows = training_data_config['additional_windows']
    total_days_range = (
        # the total duration of modeling+training periods
        (modeling_period_duration + training_period_duration)
        # the number of lookback days added from the time windows
        + (window_frequency * additional_windows))
    earliest_window_start = pd.to_datetime(modeling_period_end) - timedelta(days=total_days_range)

    # Calculate the earliest cohort lookback date for the earliest window
    # Identify all unique cohort lookback periods
    cohort_lookback_periods = [
        cohort['lookback_period']
        for cohort in config['datasets']['wallet_cohorts'].values()
    ]
    earliest_cohort_lookback_start = (earliest_window_start -
                                        timedelta(days=max(cohort_lookback_periods)))


    # Return updated config with calculated values
    return {
        'training_period_start': training_period_start.strftime('%Y-%m-%d'),
        'training_period_end': training_period_end.strftime('%Y-%m-%d'),
        'modeling_period_end': modeling_period_end.strftime('%Y-%m-%d'),
        'earliest_window_start': earliest_window_start.strftime('%Y-%m-%d'),
        'earliest_cohort_lookback_start': earliest_cohort_lookback_start.strftime('%Y-%m-%d')
    }


def validate_config_consistency(config,metrics_config, modeling_config):
    """
    Validates the consistency between config.yaml and metrics_config.yaml.

    Args:
        config (Dict[str, Any]): Loaded config.yaml.
        metrics_config (Dict[str, Any]): Loaded metrics_config.yaml.

    Raises:
        ValueError: If there are inconsistencies between the configs.
    """
    config_datasets = config.get('datasets', {})

    # Check if metrics_config and config top-level keys match
    config_dataset_keys = set(config_datasets.keys())
    metrics_config_keys = set(metrics_config.keys())

    missing_in_metrics = config_dataset_keys - metrics_config_keys
    missing_in_config = metrics_config_keys - config_dataset_keys

    if missing_in_metrics or missing_in_config:
        error_msg = []
        if missing_in_metrics:
            error_msg.append(f"Missing in metrics_config.yaml: {', '.join(missing_in_metrics)}")
        if missing_in_config:
            error_msg.append(f"Missing in config.yaml datasets: {', '.join(missing_in_config)}")
        raise ValueError("Inconsistency between config files:\n" + "\n".join(error_msg))

    # Confirm all top-level keys have a modeling_config fill_method
    modeling_config_keys = set(modeling_config['preprocessing']['fill_methods'].keys())
    missing_in_modeling = config_dataset_keys - modeling_config_keys
    if missing_in_modeling:
        error_msg = (["Missing in modeling_config['preprocessing']"
                    f"['drop_features']: {', '.join(missing_in_modeling)}"])
        raise ValueError("Inconsistency between config files:\n" + "\n".join(error_msg))


    # Check if next-level keys match for each top-level key
    for key in config_dataset_keys:
        if key not in metrics_config:
            continue  # This case is already handled in the top-level check

        config_subkeys = set(config_datasets[key].keys())
        metrics_subkeys = set(metrics_config[key].keys())

        missing_in_metrics = config_subkeys - metrics_subkeys
        missing_in_config = metrics_subkeys - config_subkeys

        if missing_in_metrics or missing_in_config:
            error_msg = []
            if missing_in_metrics:
                error_msg.append(f"Missing in metrics_config.yaml[{key}]: {', '.join(missing_in_metrics)}")
            if missing_in_config:
                error_msg.append(f"Missing in config.yaml datasets[{key}]: {', '.join(missing_in_config)}")
            raise ValueError(f"Inconsistency in {key}:\n" + "\n".join(error_msg))

    # If we've made it this far, the configs are consistent
    logger.debug("Config files are consistent.")


def get_expected_columns(metrics_config: Dict[str, Any]) -> List[str]:
    """
    Generate a list of expected column names from the given metrics configuration.

    Args:
        metrics_config (Dict[str, Any]): The metrics configuration dictionary.

    Returns:
        List[str]: A list of expected column names.
    """
    expected_columns = []

    def recursive_parse(config: Dict[str, Any], prefix: str = ''):
        for key, value in config.items():
            new_prefix = f"{prefix}_{key}" if prefix else key

            if isinstance(value, dict):
                # Direct scaling for the current level
                if 'scaling' in value:
                    expected_columns.append(new_prefix)

                # Handle aggregations
                if 'aggregations' in value:
                    expected_columns.extend(
                        parse_aggregations(value['aggregations'], new_prefix)
                    )

                # Handle comparisons
                if 'comparisons' in value:
                    expected_columns.extend(
                        parse_comparisons(value['comparisons'], new_prefix)
                    )

                # Handle rolling metrics
                if 'rolling' in value:
                    expected_columns.extend(
                        parse_rolling(value['rolling'], new_prefix)
                    )

                # Handle indicators
                if 'indicators' in value:
                    expected_columns.extend(
                        parse_indicators(value['indicators'], new_prefix)
                    )

                # Exclude specific keys from recursion
                keys_to_exclude = {
                    'aggregations', 'comparisons', 'rolling',
                    'scaling', 'indicators', 'parameters', 'definition'
                }
                sub_config = {k: v for k, v in value.items() if k not in keys_to_exclude}

                # Recursive call for nested structures
                recursive_parse(sub_config, new_prefix)

            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        recursive_parse(item, new_prefix)

    def parse_aggregations(aggregations: Dict[str, Any], prefix: str) -> List[str]:
        columns = []
        for agg_type, _ in aggregations.items():
            column_name = f"{prefix}_{agg_type}"
            columns.append(column_name)
        return columns

    def parse_comparisons(comparisons: Dict[str, Any], prefix: str) -> List[str]:
        columns = []
        for comp_type, _ in comparisons.items():
            column_name = f"{prefix}_{comp_type}"
            columns.append(column_name)
        return columns

    def parse_rolling(rolling_config: Dict[str, Any], prefix: str) -> List[str]:
        columns = []
        window_duration = rolling_config['window_duration']
        lookback_periods = rolling_config['lookback_periods']

        if 'aggregations' in rolling_config:
            columns.extend(parse_rolling_aggregations(
                rolling_config['aggregations'], prefix, window_duration, lookback_periods
            ))
        if 'comparisons' in rolling_config:
            columns.extend(parse_rolling_comparisons(
                rolling_config['comparisons'], prefix, window_duration, lookback_periods
            ))
        return columns

    def parse_rolling_aggregations(aggregations: Dict[str, Any], prefix: str,
                                   window_duration: int, lookback_periods: int) -> List[str]:
        columns = []
        for agg_type in aggregations.keys():
            for period in range(1, lookback_periods + 1):
                column_name = f"{prefix}_{agg_type}_{window_duration}d_period_{period}"
                columns.append(column_name)
        return columns

    def parse_rolling_comparisons(comparisons: Dict[str, Any], prefix: str,
                                  window_duration: int, lookback_periods: int) -> List[str]:
        columns = []
        for comp_type in comparisons.keys():
            for period in range(1, lookback_periods + 1):
                column_name = f"{prefix}_{comp_type}_{window_duration}d_period_{period}"
                columns.append(column_name)
        return columns

    def parse_indicators(indicators: Dict[str, Any], prefix: str) -> List[str]:
        columns = []
        for indicator_type, indicator_config in indicators.items():
            indicator_prefix = f"{prefix}_{indicator_type}"

            # Handle parameters
            if 'parameters' in indicator_config:
                # Get parameter names and values
                param_names = list(indicator_config['parameters'].keys())  # pylint: disable=W0612
                param_values_list = list(indicator_config['parameters'].values())

                # Create combinations of parameters
                param_combinations = list(itertools.product(*param_values_list))

                for params in param_combinations:
                    # Build parameter string
                    param_str = '_'.join(map(str, params))
                    # Full indicator prefix with parameters
                    full_indicator_prefix = f"{indicator_prefix}_{param_str}"

                    # Handle aggregations within the indicator
                    if 'aggregations' in indicator_config:
                        columns.extend(parse_aggregations(
                            indicator_config['aggregations'], full_indicator_prefix
                        ))

                    # Handle rolling within the indicator
                    if 'rolling' in indicator_config:
                        columns.extend(parse_indicator_rolling(
                            indicator_config['rolling'], full_indicator_prefix
                        ))
            else:
                # No parameters, directly process aggregations
                full_indicator_prefix = indicator_prefix
                if 'aggregations' in indicator_config:
                    columns.extend(parse_aggregations(
                        indicator_config['aggregations'], full_indicator_prefix
                    ))
        return columns

    def parse_indicator_rolling(rolling_config: Dict[str, Any], prefix: str) -> List[str]:
        columns = []
        window_duration = rolling_config['window_duration']
        lookback_periods = rolling_config['lookback_periods']

        if 'aggregations' in rolling_config:
            columns.extend(parse_rolling_aggregations(
                rolling_config['aggregations'], prefix, window_duration, lookback_periods
            ))
        if 'comparisons' in rolling_config:
            columns.extend(parse_rolling_comparisons(
                rolling_config['comparisons'], prefix, window_duration, lookback_periods
            ))
        return columns

    recursive_parse(metrics_config)
    return expected_columns



def check_nan_values(series):
    """
    Check if NaN values are only at the start or end of the series.

    Returns:
    - True if NaN values are in the middle of the series
    - False if NaN values are only at start/end or if there are no NaN values
    """
    # Get the index of the first and last non-NaN value
    first_valid = series.first_valid_index()
    last_valid = series.last_valid_index()

    # Check if there are any NaN values between the first and last valid value
    has_nans_in_series = (series.loc[first_valid:last_valid] # selects the range between first and last valid value
                             .isna() # checks for NaN values in this range
                             .any()) # returns True if there are any NaN values in this range

    return has_nans_in_series


def safe_downcast(df, column, dtype):
    """
    Safe method to downcast a column datatype. If the column has no values that exceed the
    limits of the new dtype, it will be downcasted. If it has values that will result in
    overflow errors, it will raise an error.
    """
    # Check if the column is numeric
    if not pd.api.types.is_numeric_dtype(df[column]):
        logger.warning("Column '%s' is not numeric. Skipping downcast.", column)
        return df

    # Get the original dtype of the column
    original_dtype = df[column].dtype

    # Get the min and max values of the column
    col_min = df[column].min()
    col_max = df[column].max()

    # Get the limits of the target dtype
    if dtype in ['float32', 'float64']:
        type_info = np.finfo(dtype)
    elif dtype in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']:
        type_info = np.iinfo(dtype)
    else:
        logger.error("Unsupported dtype: %s", dtype)
        return df

    # Check if the column values are within the limits of the target dtype
    if col_min < type_info.min or col_max > type_info.max:
        logger.warning("Cannot safely downcast column '%s' to %s. "
                       "Values are outside the range of %s. "
                       "Min: %s, Max: %s",
                       column, dtype, dtype, col_min, col_max)
        return df

    # If we've made it here, it's safe to downcast
    df[column] = df[column].astype(dtype)

    logger.debug("Successfully downcasted column '%s' from %s to %s",
                 column, original_dtype, dtype)
    return df


def timing_decorator(func):
    """
    A decorator that logs the execution time of the decorated function.

    This decorator wraps the given function, times its execution, and logs
    the time taken using the logger of the module where the decorated function
    is defined. It uses lazy % formatting for efficient logging.

    Args:
        func (callable): The function to be decorated.

    Returns:
        callable: The wrapped function.

    Example:
        @timing_decorator
        def my_function(x, y):
            return x + y

    When my_function is called, it will log a message like:
    "my_function took 0.001 seconds to execute."
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get the logger
        function_logger = logging.getLogger(func.__module__)

        # Log the initiation of the function
        function_logger.debug('Initiating %s...', func.__name__)

        # Time the function execution
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Log the execution time
        function_logger.info(
            'Completed %s after %.2f seconds.',
            func.__name__,
            end_time - start_time
        )
        return result
    return wrapper



def create_progress_bar(total_items):
    """
    Creates and starts a progress bar for tracking the progress of for loops.

    Args:
    - total_items (int): The total number of items in the loop

    Returns:
    - progressbar.ProgressBar: An initialized and started progress bar.
    """
    _widgets = [
        'Completed ',  # Customizable label for context
        progressbar.SimpleProgress(),  # Displays 'current/total' format
        ' ',
        progressbar.Percentage(),
        ' ',
        progressbar.Bar(),
        ' ',
        progressbar.ETA()
    ]

    # Create and start the progress bar with widgets, redirecting stdout to make it "float"
    progress_bar = progressbar.ProgressBar(
        maxval=total_items,
        widgets=_widgets,
        redirect_stdout=True  # Redirect stdout to prevent new lines
    ).start()

    return progress_bar



def cw_filter_df(df, coin_id, wallet_address):
    """
    Filter DataFrame by coin_id and wallet_address.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        coin_id (str): The coin ID to filter by.
        wallet_address (str): The wallet address to filter by.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    filtered_df = df[
        (df['coin_id'] == coin_id) &
        (df['wallet_address'] == wallet_address)
    ]
    return filtered_df


def df_mem(df):
    """
    Checks how much memory a dataframe is using
    """
    # Memory usage of each column
    memory_usage = df.memory_usage(deep=True) / (1024 ** 2)
    print(memory_usage.round(2))

    # Total memory usage in bytes
    total_memory = df.memory_usage(deep=True).sum()
    print(f'Total memory usage: {total_memory / 1024 ** 2:.2f} MB')


def obj_mem():
    """
    Checks how much memory all objects are using

    name logic needs to be redone
    """
    objects = []
    for obj in gc.get_objects():
        # try:
        size = sys.getsizeof(obj)
        if size >= 1000:  # Filter out objects smaller than 1000 bytes
            obj_type = type(obj).__name__
            obj_name = str(getattr(obj, '__name__', 'Unnamed'))  # Get name if available
            objects.append((obj_name, obj_type, size / (1024 * 1024)))  # Convert size to MB
        # except:
        #     continue
    mem_df = pd.DataFrame(objects, columns=['Name', 'Type', 'Size (MB)'])
    mem_df = mem_df.sort_values(by='Size (MB)', ascending=False).reset_index(drop=True)
    return mem_df


def log_nan_counts(df):
    """
    utility function for testing
    """
    nan_counts = df.isna().sum()
    non_zero_nans = nan_counts[nan_counts > 0]

    if len(non_zero_nans) > 0:
        log_message = "NaN counts in columns:\n" + "\n".join(f"{col}: {count}" for col, count in non_zero_nans.items())
    else:
        log_message = "No NaN values found in any column."

    logger.critical(log_message)


def play_notification(sound_file_path=None):
    """
    Play a notification sound from a local audio file using pygame.
    Falls back to ALERT_SOUND_FILEPATH environment variable if no path provided.
    Returns early if no valid sound file path is found.

    Args:
        sound_file_path (str, optional): Path to the sound file (supports .mp3, .wav, etc.)
    """
    sound_file_path = sound_file_path or os.getenv('ALERT_SOUND_FILEPATH')
    if not sound_file_path:
        return "No sound file found."

    try:
        pygame.mixer.init()
        pygame.mixer.music.load(sound_file_path)
        pygame.mixer.music.play()

        # Wait for the sound to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    except Exception as e:  # pylint:disable=broad-exception-caught
        return f"Error playing sound: {e}"
    finally:
        pygame.mixer.quit()
