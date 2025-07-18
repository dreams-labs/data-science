"""
# data-science/utils.py

## Overview
Core utility functions supporting the data science pipeline, from configuration management
 to DataFrame optimization to development workflow tools. Provides essential infrastructure
 for handling large-scale crypto wallet analysis datasets and Jupyter notebook development.

## Key Functions

### utils.py Configuration Management
- **load_all_configs()** - Loads and validates all YAML configuration files
- **load_config()** - Individual config loading with Pydantic validation
- **calculate_period_dates()** - Automatic calculation of training/modeling period
    boundaries
- **validate_config_consistency()** - Cross-config validation for datasets and metrics

### DataFrame Optimization
- **df_downcast()** / **safe_downcast()** - Memory optimization through intelligent dtype
    reduction
- **ensure_index()** - Standardizes MultiIndex structure for coin_id-wallet_address-date
- **assert_period()** - Validates DataFrame dates against configured period boundaries
- **cw_filter()** / **cw_sample()** - Coin-wallet pair filtering and sampling utilities

### Development & Debugging
- **setup_notebook_logger()** - Configures multi-level logging with color formatting
- **obj_mem()** / **df_mem()** - Memory usage tracking for large datasets
- **export_code()** - Consolidates multiple Python files into single artifacts
- **notify()** / **AmbientPlayer** - Audio notifications for long-running processes

### Data Quality
- **check_nan_values()** - Validates NaN patterns in time series data
- **winsorize()** - Statistical outlier handling for financial metrics
- **assert_matching_indices()** - DataFrame index consistency validation

## Integration
Works closely with the modeling pipeline through standardized DataFrame indexing and period
 validation. Integrates with BigQuery for data persistence and supports the
 configuration-driven approach used throughout the project.

## Usage Patterns
Essential for notebook development workflows, particularly when working with profits_df
 datasets containing millions of rows. Memory optimization and logging functions are critical
 for managing computational resources during feature engineering and model training phases.
"""
import time
import sys
import os
import json
import gc
import inspect
import atexit
from pathlib import Path
import gzip
import shutil
import math
import threading
from datetime import datetime, timedelta
from typing import List,Dict,Any,Union
import itertools
import logging
import warnings
import functools
import yaml
import psutil
import pandas as pd
import numpy as np
from pydantic import ValidationError
import pygame
from IPython.display import display

# pylint: disable=E0401
# pylint: disable=E0611
# pylint: disable=too-many-lines

# set up logger at the module level
logger = logging.getLogger(__name__)



# ---------------------------------------- #
#         Config Related Functions
# ---------------------------------------- #

def load_all_configs(config_folder):
    """
    Loads and returns all config files needed for the Coin Flow Model
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
    # pydantic config files    # pylint:disable=import-outside-toplevel
    import config_models.config as py_c
    import config_models.metrics_config as py_mc
    import config_models.modeling_config as py_mo
    import config_models.experiments_config as py_e

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
                'coins_metrics_config.yaml': py_mc.MetricsConfig,
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

    # Coin Modeling Dates
    # -------------------
    coin_modeling_period_start = modeling_period_end + timedelta(days=1)
    coin_modeling_period_end = modeling_period_end + timedelta(days=modeling_period_duration)


    # Investing Period Dates
    # -------------------
    investing_period_start = coin_modeling_period_end + timedelta(days=1)
    investing_period_end = coin_modeling_period_end + timedelta(days=modeling_period_duration)


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
        'coin_modeling_period_start': coin_modeling_period_start.strftime('%Y-%m-%d'),
        'coin_modeling_period_end': coin_modeling_period_end.strftime('%Y-%m-%d'),
        'investing_period_start': investing_period_start.strftime('%Y-%m-%d'),
        'investing_period_end': investing_period_end.strftime('%Y-%m-%d'),
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
                error_msg.append("Config cohorts in [config][datasets][wallet_cohorts] are missing "
                                 "corresponding settings in [metrics_config][wallet_cohorts]: "
                                 f"[{', '.join(missing_in_metrics)}]")
            if missing_in_config:
                error_msg.append("Metrics config cohorts in [metrics_config][wallet_cohorts] are missing "
                                 "corresponding settings in [config][datasets][wallet_cohorts]: "
                                 f"[{', '.join(missing_in_metrics)}]")

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


class ConfigError(Exception):
    """Raised when configuration parameters are invalid or missing."""
    pass







# --------------------------------- #
#        Function Modifiers
# --------------------------------- #

def timing_decorator(func_or_level=logging.INFO):
    """
    A decorator that logs the execution time of the decorated function.

    Supports both @timing_decorator and @timing_decorator(level) syntax.


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
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug('Initiating %s...', func.__name__)

            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            record = logging.LogRecord(
                name=logger.name,
                level=level,
                pathname=func.__code__.co_filename,
                lineno=func.__code__.co_firstlineno,
                msg='(%.1fs) Completed %s.',
                args=(duration, func.__name__),
                exc_info=None
            )
            logger.handle(record)
            return result
        return wrapper

    if callable(func_or_level):
        # Used as @timing_decorator
        level = logging.INFO
        return decorator(func_or_level)
    else:
        # Used as @timing_decorator(level)
        level = func_or_level
        return decorator


def create_progress_bar(total_items):
    """
    Creates and starts a progress bar for tracking the progress of for loops.

    Args:
    - total_items (int): The total number of items in the loop

    Returns:
    - progressbar.ProgressBar: An initialized and started progress bar.
    """
    # lazy import at function level due to import size
    import progressbar  # pylint:disable=import-outside-toplevel

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








# ---------------------------------------- #
#     DataFrame Manipulation Functions
# ---------------------------------------- #

def safe_downcast(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Automatically downcast a numeric column to smallest safe dtype.

    Params:
    - df (DataFrame): Input dataframe
    - column (str): Column to downcast

    Returns:
    - DataFrame: DataFrame with downcasted column if safe
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        logger.debug(f"Column '{column}' is not numeric. Skipping downcast.")
        return df

    original_dtype = str(df[column].dtype)
    col_min = df[column].min()
    col_max = df[column].max()

    # Define downcast paths
    downcast_paths = {
        'int64': ['int32', 'int16'],
        'Int64': ['Int32', 'Int16'],
        'float64': ['float32', 'float16'],
        'Float64': ['Float32', 'Float16']
    }

    # Get downcast sequence for this dtype
    dtype_sequence = downcast_paths.get(original_dtype, [])
    if not dtype_sequence:
        return df

    # Try each downcast level
    for target_dtype in dtype_sequence:
        try:
            # Convert pandas dtype to numpy for limit checking
            np_dtype = target_dtype.lower()
            if target_dtype[0].isupper():
                np_dtype = np_dtype[1:]  # Remove 'I' from 'Int32'

            # Skip if we can't get type info
            if not np_dtype.startswith(('int', 'float')):
                continue

            # Get dtype limits
            if 'float' in np_dtype:
                type_info = np.finfo(np_dtype)
            else:
                type_info = np.iinfo(np_dtype)

            # Check if safe to downcast
            if col_min >= type_info.min and col_max <= type_info.max:
                df[column] = df[column].astype(target_dtype)
                logger.debug(f"Downcasted '{column}' from {original_dtype} to {target_dtype}")
                return df

        except (ValueError, TypeError) as e:
            logger.debug(f"Could not process {target_dtype} for column '{column}': {e}")
            continue

    return df


def df_downcast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to downcast all numeric columns to smallest safe dtype.

    Params:
    - df (DataFrame): Input dataframe to optimize

    Returns:
    - DataFrame: Optimized dataframe
    """
    start_time = time.time()

    for col in df.columns:
        df = safe_downcast(df, col)

    # Log duration
    if time.time() - start_time >= 1:
        logger.info('(%.1fs) Completed df_downcast.', time.time() - start_time)
    else:
        logger.debug('(%.1fs) Completed df_downcast.', time.time() - start_time)

    return df


def assert_period(df, period_start, period_end) -> None:
    """
    Validates if DataFrame dates fall within configured period boundaries.

    Params:
    - config (dict): config with dates in the 'training_data' section
    - df (DataFrame): Input DataFrame with 'date' column
    - period (str): Period type ('training'/'modeling'/'validation')

    Raises:
    - ValueError: If dates fall outside period boundaries
    """
    # Extract dates as datetimes
    period_start = pd.to_datetime(period_start)
    period_end = pd.to_datetime(period_end)
    period_starting_balance_date = period_start - timedelta(days=1)

    # Reset index so we can assess date as a column
    df = df.reset_index()

    # Confirm no null coin_id columns
    if df['coin_id'].isnull().any():
        raise ValueError("Found null values in coin_id column")

    # Confirm the earliest record is the period starting balance date
    earliest_date = df['date'].min()
    if earliest_date > period_starting_balance_date:
        raise ValueError(f"Earliest date is {earliest_date.strftime('%Y-%m-%d')} which "
                        "is later than the starting balance date of "
                        f"{period_starting_balance_date.strftime('%Y-%m-%d')}.")

    if earliest_date < period_starting_balance_date:
        raise ValueError(f"Earliest date is {earliest_date.strftime('%Y-%m-%d')} which "
                        "is earlier than the starting balance date of "
                        f"{period_starting_balance_date.strftime('%Y-%m-%d')}.")

    # Confirm the latest record is the period end
    latest_date = df['date'].max()
    if latest_date > period_end:
        raise ValueError(f"Found dates up to {latest_date.strftime('%Y-%m-%d')} which "
                        f"is later than the period end of {period_end.strftime('%Y-%m-%d')}.")
    if latest_date < period_end:
        raise ValueError(f"Latest record in dataframe is {latest_date.strftime('%Y-%m-%d')} which "
                        f"is earlier than the period end of {period_end.strftime('%Y-%m-%d')}.")

    # Checks for profits_df specific values
    if 'usd_balance' in df.columns:

        # Confirm no null wallet_address columns
        if df['wallet_address'].isnull().any():
            raise ValueError("Found null values in wallet_address column")

        # Confirm the profits_df columns are in df
        profits_df_columns = ['usd_balance', 'usd_net_transfers', 'usd_inflows', 'is_imputed']
        if not set(profits_df_columns).issubset(df.columns):
            raise ValueError("Dataframe contains 'usd_balance' column but does not have the "
                            f"expected profits_df columns of {profits_df_columns}.")

        # Confirm imputed dates are only at the starting balance and period end dates
        unexpected_imputed_dates = df[
            df['is_imputed'] &
            ~df['date'].isin([period_starting_balance_date, period_end])
        ]['date'].unique()
        if len(unexpected_imputed_dates):
            formatted_dates = pd.to_datetime(unexpected_imputed_dates).strftime("%Y-%m-%d")
            raise ValueError(f"Unexpected imputed dates found on {set(formatted_dates)}.")


        # Starting Balance Values Checks
        # ------------------------------
        starting_balance_profits_df = df[df['date']==period_starting_balance_date]

        # Confirm all transfers are set to $0
        if starting_balance_profits_df['usd_net_transfers'].sum() != 0:
            raise ValueError("Dataframe has non-zero usd_net_transfers on starting balance "
                                f"date of {period_starting_balance_date.strftime('%Y-%m-%d')}.")

        # Confirm all usd_inflows are set to $0
        if starting_balance_profits_df['usd_inflows'].sum() != 0:
            raise ValueError("Dataframe has non-zero usd_net_transfers on starting balance "
                                f"date of {period_starting_balance_date.strftime('%Y-%m-%d')}.")

        # Confirm all usd_inflows are set to $0
        if starting_balance_profits_df['usd_inflows'].sum() != 0:
            raise ValueError("Dataframe has non-zero usd_net_transfers on starting balance "
                                f"date of {period_starting_balance_date.strftime('%Y-%m-%d')}.")

        # Confirm all rows have is_imputed set to True
        if not starting_balance_profits_df['is_imputed'].all():
            raise ValueError("Found non-imputed records in starting balance")


def assert_matching_indices(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    """
    Assert that two DataFrames have identical indices (order-insensitive).

    Params
    ------
    df1 : DataFrame
    df2 : DataFrame

    Raises
    ------
    ValueError – if the indices differ.
    """
    # Ensure sorted, monotonic indices for comparison
    idx1 = df1.index if df1.index.is_monotonic_increasing else df1.index.sort_values()
    idx2 = df2.index if df2.index.is_monotonic_increasing else df2.index.sort_values()

    if np.array_equal(idx1.values, idx2.values):
        return  # everything matches

    # ─────────────────────── diagnostic message ───────────────────────
    msg_lines: list[str] = [
        "DataFrames have mismatched indices.",
        f"df1 shape: {df1.shape}, df2 shape: {df2.shape}",
        f"df1 index name: '{idx1.name}', df2 index name: '{idx2.name}'"
    ]

    # If numeric, add extra context (range & mean)
    if pd.api.types.is_numeric_dtype(idx1):
        arr1 = idx1.to_numpy(dtype=float)
        arr2 = idx2.to_numpy(dtype=float)
        msg_lines.append(f"df1 index range: [{arr1.min()}, {arr1.max()}], mean: {arr1.mean():.2f}")
        msg_lines.append(f"df2 index range: [{arr2.min()}, {arr2.max()}], mean: {arr2.mean():.2f}")

    raise ValueError("\n".join(msg_lines))

def ensure_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Params:
    - df (DataFrame): Input data.

    Returns:
    - DataFrame: Validated and properly indexed/sorted DataFrame.
    """
    start_time = time.time()

    required_cols_date = ['coin_id', 'wallet_address', 'date']
    required_cols_basic = ['coin_id', 'date']

    # Check if the DataFrame is already indexed correctly
    if df.index.names == required_cols_date or df.index.names == required_cols_basic:
        # Validate sorting
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        return df

    # Determine which columns are required based on the input format
    if all(col in df.columns for col in required_cols_date):
        index_cols = required_cols_date
    elif all(col in df.columns for col in required_cols_basic):
        index_cols = required_cols_basic
    else:
        raise ValueError(
            "Input DataFrame must have either an index or columns for ['coin_id', 'wallet_address', 'date'] "
            "or ['coin_id', 'wallet_address']."
        )

    # Convert to MultiIndex if needed
    if not isinstance(df.index, pd.MultiIndex):
        df = df.set_index(index_cols)

    # Ensure correct index names
    if df.index.names != index_cols:
        df = df.reorder_levels(index_cols)

    # Ensure sorting
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # Log duration
    if time.time() - start_time >= 1:
        logger.info('(%.1fs) Completed ensure_index.', time.time() - start_time)
    else:
        logger.debug('(%.1fs) Completed ensure_index.', time.time() - start_time)

    return df


def validate_column_consistency(
    df1: pd.DataFrame,
    df2: pd.DataFrame
) -> None:
    """
    Validate that two DataFrames have identical column names and order.

    Params:
    - df1, df2 (DataFrame): DataFrames to compare

    Raises:
    - ValueError: If columns don't match exactly in name and order
    """
    if df1.columns.tolist() != df2.columns.tolist():
        cols1 = df1.columns.tolist()
        cols2 = df2.columns.tolist()
        set1 = set(cols1)
        set2 = set(cols2)

        error_msg = "DataFrames have different column order.\n"
        error_msg += f"DF1: {len(cols1)} columns, DF2: {len(cols2)} columns\n"

        missing_in_2 = set1 - set2
        extra_in_2 = set2 - set1

        if missing_in_2:
            error_msg += f"Missing in DF2: {sorted(list(missing_in_2))}\n"
        if extra_in_2:
            error_msg += f"Extra in DF2: {sorted(list(extra_in_2))}\n"

        # Check if it's just an ordering issue
        if set1 == set2:
            error_msg += "Same columns but different order (sorting would fix this)\n"

        error_msg += f"First 5 DF1: {cols1[:5]}\n"
        error_msg += f"First 5 DF2: {cols2[:5]}"

        raise ValueError(error_msg)



def cw_filter(df, coin_id, wallet_address):
    """
    Filter DataFrame by coin_id and wallet_address, sort by date if available.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        coin_id (str): The coin ID to filter by.
        wallet_address (str): The wallet address to filter by.

    Returns:
        pd.DataFrame: Filtered DataFrame, optionally sorted by date.
    """
    filtered_df = df[
        (df['coin_id'] == coin_id) &
        (df['wallet_address'] == wallet_address)
    ]

    if 'date' in df.columns:
        filtered_df = filtered_df.sort_values('date')

    return filtered_df


def cw_sample(profits_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Retrieve all rows for n random coin-wallet pairs.

    Params:
    - profits_df (DataFrame): Input profits data with coin_id, wallet_address columns
    - n (int): Number of coin-wallet pairs to retrieve

    Returns:
    - subset_df (DataFrame): All rows for selected coin-wallet pairs
    """
    # Get unique combinations efficiently using drop_duplicates
    unique_pairs = profits_df[['coin_id', 'wallet_address']].drop_duplicates()

    # Sample n random pairs
    selected_pairs = unique_pairs.sample(n=n)

    # Create efficient boolean mask using isin() on multi-column condition
    mask = profits_df.set_index(['coin_id', 'wallet_address'])\
                    .index\
                    .isin(selected_pairs.set_index(['coin_id', 'wallet_address'])\
                    .index)

    # Return filtered dataframe
    subset_df = profits_df[mask].sort_values(['coin_id', 'wallet_address', 'date'])

    return subset_df


def df_nans(df,cells=False):
    """
    Returns rows and cells in a df that contain NaN values.
    """
    na_rows = df[df.isna().any(axis=1)]
    if cells is False:
        return na_rows

    na_cells = na_rows.loc[:, na_rows.isna().any()]
    return na_cells


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



def numpy_type_converter(obj):
    """Convert numpy and datetime types to JSON-friendly types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (float, np.floating)) and math.isinf(obj):
        return "Infinity" if obj > 0 else "-Infinity"
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):  # handle datetime
        return obj.isoformat()
    # Surface non-serializable types explicitly
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")



def display_full(df, sort_by=None, ascending=False):
    """
    Params:
    - df (DataFrame): dataframe to display with full visibility
    - sort_by (str): optional column name to sort by
    - ascending (bool): sort direction if sort_by is specified

    Returns:
    - None: displays the dataframe
    """
    display_df = df.sort_values(by=sort_by, ascending=ascending) if sort_by else df

    with pd.option_context('display.max_colwidth', None, 'display.max_columns', None, 'display.max_rows', None):
        display(display_df)



def to_parquet_safe(
        df: pd.DataFrame,
        file_path: str,
        sort_cols: bool = False,
        **kwargs
    ) -> None:
    """
    Write DataFrame to parquet using an intermediate temp file to prevent corruption.

    Should always be used to avoid saving partially written files.

    Params:
    - df: DataFrame to write
    - file_path: Final destination path
    - sort_cols: If True, sort columns alphabetically before saving
    - **kwargs: Additional arguments passed to to_parquet()
    """
    # Sort columns if requested
    if sort_cols and hasattr(df, 'columns'):
        df = df.sort_index(axis=1)

    temp_path = f"{file_path}.tmp"
    try:
        df.to_parquet(temp_path, **kwargs)
        os.rename(temp_path, file_path)
    except Exception:
        # Clean up temp file if write failed
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise



# ---------------------------------------- #
#      Series Manipulation Functions
# ---------------------------------------- #

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


def winsorize(data: pd.Series, cutoff: float = 0.01) -> pd.Series:
    """
    Winsorize a data series at specified cutoff levels.

    Args:
        data: Series to winsorize
        cutoff: Percentage (in decimal form) to cut from each tail

    Returns:
        Winsorized series
    """
    # Make a copy to avoid modifying original
    winsorized = data.copy()

    # Calculate bounds using non-null values
    valid_data = data[~np.isnan(data)]
    if valid_data.size == 0:
        return winsorized  # Return original series unchanged if no valid data

    lower_bound = np.percentile(valid_data, cutoff * 100, method='nearest')
    upper_bound = np.percentile(valid_data, (1 - cutoff) * 100, method='nearest')

    # Clip the data
    return np.clip(winsorized, lower_bound, upper_bound)




# ---------------------------------------- #
#     Audio Notebook Helper Functions
# ---------------------------------------- #

def notify(sound_name: Union[str, int] = None, prompt: str = None, voice_id: str = 'Tessa'):
    """
    Play alert sound followed by optional TTS message.

    Params:
    - sound_name (str|int): Name/index of sound from config
    - prompt (str, optional): Text to speak using TTS
    - voice_id (str, optional): Specific voice ID for TTS
    """
    # Load sound config from sounds directory environment variable
    sounds_directory = Path(os.environ.get('NOTIFICATION_SOUNDS_DIR', "../../../Local"))
    config_path = sounds_directory / "notification_sounds.yaml"

    # Overall adjustment to all sound levels; if set to 0.0 all sounds will be muted
    dampen_level = float(os.environ.get('NOTIFICATION_DAMPEN', '1.0'))

    try:
        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:  # pylint: disable=broad-exception-caught
        return f"Error loading sound config: {e}"

    sounds = config.get('notification_sounds', {})

    # Handle integer index
    if isinstance(sound_name, int):
        sound_keys = list(sounds.keys())
        if 0 <= sound_name < len(sound_keys):
            sound_name = sound_keys[sound_name]
        else:
            return f"Invalid sound index. Choose 0-{len(sound_keys)-1}"

    # Default to 'notify' if no sound specified
    if not sound_name:
        sound_name = 'notify'
    elif sound_name not in sounds:
        return f"Invalid sound name. Choose from: {', '.join(sounds.keys())}"

    sound_config = sounds[sound_name]

    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        # Use sounds_directory with the path from config
        dampen_level = .60
        volume_dampened = sound_config.get('volume', 0.5) * dampen_level
        sound_path = sounds_directory / sound_config['path']
        sound = pygame.mixer.Sound(sound_path)
        sound.set_volume(volume_dampened)
        sound.play()

        if prompt:
            time.sleep(0.8)
            voice_cmd = f"say -v {voice_id} -r 125 '[[ volm 0.6 ]] {prompt}'"
            os.system(voice_cmd)

    except Exception as e:  # pylint: disable=broad-exception-caught
        return f"Error with playback: {e}"


class AmbientPlayer:
    """
    Plays a sound file on loop until commanded to .stop(). Uses the same sound keys dict
     as u.notify(). Automatically stops on kernel shutdown/error.
    """
    _active_players = []  # Class variable to track all players

    def __init__(self):
        pygame.mixer.init()
        self.playing = False
        self.thread = None
        self.target_volume = 0.0
        self.current_volume = 0.0
        self.sound = None
        self.channel = None

        # Register this player for cleanup
        AmbientPlayer._active_players.append(self)

        # Register cleanup function (only once)
        if len(AmbientPlayer._active_players) == 1:
            atexit.register(AmbientPlayer._cleanup_all_players)

    @classmethod
    def stop_all_players(cls):
        """Stop all active players and clear the registry"""
        for player in cls._active_players[:]:  # Create copy to avoid modification during iteration
            try:
                player.stop()
            except:  # pylint:disable=bare-except
                pass  # Ignore errors during cleanup

        cls._active_players.clear()  # Clear the registry

        # Force stop all pygame channels and quit mixer
        pygame.mixer.stop()
        pygame.mixer.quit()

        # Reinitialize mixer for future use
        try:
            pygame.mixer.init()
        except:  # pylint:disable=bare-except
            pass

    @classmethod
    def _cleanup_all_players(cls):
        """Stop all active players on program exit"""
        cls.stop_all_players()

    def start(self, sound_name: str):
        """Start looping ambient audio in background thread"""
        if self.playing:
            return

        # Load sound config from sounds directory environment variable
        sounds_directory = Path(os.environ.get('NOTIFICATION_SOUNDS_DIR', "../../../Local"))
        config_path = sounds_directory / "notification_sounds.yaml"
        try:
            with open(config_path, encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Error loading sound config: {e}")
            return

        sounds = config.get('notification_sounds', {})
        file_path = sounds_directory / sounds[sound_name]['path']
        if not os.path.exists(file_path):
            logger.warning(f"couldn't find ambient file at '{file_path}'")
            return

        self.target_volume = sounds[sound_name]['volume']
        self.current_volume = 0.0
        self.sound = pygame.mixer.Sound(file_path)
        self.playing = True
        self.thread = threading.Thread(target=self._play_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop ambient audio with fade out"""
        if not self.playing:
            return

        # Fade out over 2 seconds
        fade_steps = 100
        step_duration = 2.0 / fade_steps
        volume_step = self.current_volume / fade_steps

        for _ in range(fade_steps):
            if not self.playing:  # Check if already stopped
                break
            self.current_volume = max(0, self.current_volume - volume_step)
            if self.sound:
                self.sound.set_volume(self.current_volume)
            threading.Event().wait(step_duration)

        self.playing = False
        if self.channel:
            self.channel.stop()
        if self.thread:
            self.thread.join(timeout=1)

        # Remove from active players
        if self in AmbientPlayer._active_players:
            AmbientPlayer._active_players.remove(self)

    def _play_loop(self):
        self.sound.set_volume(0.0)
        self.channel = self.sound.play(-1)  # -1 means loop indefinitely

        # Fade in over 2 seconds
        fade_steps = 100
        step_duration = 2.0 / fade_steps
        volume_step = self.target_volume / fade_steps

        for _ in range(fade_steps):
            if not self.playing:
                return
            self.current_volume = min(self.target_volume, self.current_volume + volume_step)
            self.sound.set_volume(self.current_volume)
            threading.Event().wait(step_duration)

        # Main playback loop - just wait while channel is playing
        while self.playing and self.channel and self.channel.get_busy():
            threading.Event().wait(0.1)


def notify_on_failure(shell, etype, value, tb, tb_offset=None):
    """
    Custom error handler that plays a notification sound
    and displays the traceback normally.
    """
    # Mute any ambient players
    player = AmbientPlayer()
    player.stop_all_players()

    try:
        # Play error notification
        notify('ui_sound')
    except Exception:
        pass  # Safely ignore any errors with notify()

    # Call the original traceback display method
    shell.showtraceback((etype, value, tb), tb_offset=tb_offset)






# ---------------------------------------- #
#     Misc Notebook Helper Functions
# ---------------------------------------- #

def df_mem(df):
    """
    Checks how much memory a dataframe is using
    """
    # Memory usage of each column
    memory_usage = df.memory_usage(deep=True) / (1024 ** 2)
    mem_df = pd.DataFrame(memory_usage)
    mem_df.columns = ['mb']
    mem_df['dtype'] = df.dtypes

    # Total memory usage in bytes
    total_memory = df.memory_usage(deep=True).sum()
    print(f'Total memory usage: {total_memory / 1024 ** 2:.2f} MB')

    return mem_df.sort_values(by='mb',ascending=False)


def obj_mem(return_details=False) -> pd.DataFrame:
    """
    Params:
    - return_details (bool): whether to return column names, types, and memory usage

    Tracks both object-level and process-level memory usage with enhanced metadata.
    Handles nested data structures better.
    """

    process = psutil.Process()
    total_rss = process.memory_info().rss / (1024 * 1024)

    # Get the calling frame to access its locals
    calling_frame = inspect.currentframe().f_back

    def find_name(obj):
        """Helper to find variable name across namespaces"""
        # Check globals first
        name = next((name for name, value in globals().items() if value is obj), None)
        if name:
            return name

        # Check calling frame's locals
        if calling_frame:
            name = next((name for name, value in calling_frame.f_locals.items() if value is obj), None)
            if name:
                return name

        return 'unnamed'

    # Track DataFrames in containers
    container_refs = {}
    for obj in gc.get_objects():
        if isinstance(obj, (list, dict, tuple)):
            try:
                name = find_name(obj)
                if name != 'unnamed':
                    for idx, item in enumerate(obj if isinstance(obj, (list, tuple)) else obj.values()):
                        if isinstance(item, pd.DataFrame):
                            container_refs[id(item)] = f"{name}[{idx}]"
            except:  # pylint:disable=bare-except
                continue

    # Gather all DataFrame info
    objects = []
    for obj in gc.get_objects():
        try:
            if not isinstance(obj, (pd.DataFrame, pd.Series, np.ndarray)):
                continue

            name = find_name(obj)
            if name == 'unnamed':
                name = container_refs.get(id(obj), 'unnamed')

            if hasattr(obj, 'memory_usage'):
                size = (obj.memory_usage(deep=True).sum() /
                        (1024 * 1024))  # Convert to MB
            else:
                size = sys.getsizeof(obj) / (1024 * 1024)

            if size >= 5:
                metadata = {
                    'name': name,
                    'type': type(obj).__name__,
                    'size_mb': round(size, 1),
                    'shape': str(getattr(obj, 'shape', None)),
                    'percent_of_total': round((size / total_rss) * 100, 1),
                }

                if isinstance(obj, pd.DataFrame):
                    metadata.update({
                        'columns': (str(list(obj.columns)[:3] + ['...']) if len(obj.columns) > 3
                                    else str(list(obj.columns))),
                        'memory_usage': str({col: f"{mem/1024/1024:.1f}MB"
                                           for col, mem in obj.memory_usage(deep=True).items()
                                           if mem/1024/1024 >= 1}),
                        'dtypes': str({col: str(dtype)
                                     for col, dtype in obj.dtypes.items()
                                     if obj[col].memory_usage(deep=True)/1024/1024 >= 1})
                    })

                objects.append(metadata)
        except:  # pylint:disable=bare-except
            continue

    # Clean up
    del calling_frame

    mem_df = pd.DataFrame(objects)

    summary = pd.DataFrame([{
        'name': 'TOTAL_PROCESS_MEMORY',
        'type': 'ProcessRSS',
        'size_mb': round(total_rss, 1),
        'shape': None,
        'percent_of_total': 100.0
    }])

    mem_df = pd.concat([summary, mem_df], ignore_index=True)
    mem_df = mem_df.sort_values('size_mb', ascending=False).reset_index(drop=True)

    cols = ['name', 'type', 'size_mb', 'shape', 'percent_of_total']
    extra_cols = [col for col in mem_df.columns if col not in cols]

    if return_details:
        # Return column details if requested
        return mem_df[cols + extra_cols]
    else:
        return mem_df[cols]


def purge_dict_dfs(epoch_dfs: Dict[datetime, pd.DataFrame]) -> None:
    """
    Wipe every DataFrame in the dict **and** any stray references in the
    current frame / interactive namespace, then run the GC.

    Params
    ------
    epoch_dfs : dict[datetime, DataFrame]
    """
    frame = inspect.currentframe().f_back            # calling frame

    # 1. break refs inside the dict
    for k in list(epoch_dfs):
        df = epoch_dfs.pop(k)
        df.drop(df.index, inplace=True)              # release BlockManager arrays
        del df                                       # delete local pointer

    epoch_dfs.clear()                                # just in case
    del epoch_dfs                                    # delete the dict itself

    # 2. nuke stray refs in locals / globals (REPL convenience vars, etc.)
    for scope in (frame.f_locals, frame.f_globals):
        for name, value in list(scope.items()):
            if isinstance(value, pd.DataFrame):
                scope[name] = None

    # 3. run the collector
    gc.collect()


# pylint: disable=dangerous-default-value
def export_code(
    code_directories=[],
    parent_directory="..//src",
    include_config=False,
    include_markdown=True,
    config_directory="..//config",
    notebook_directory="..//notebooks",
    ipynb_notebook=None,
    output_file="temp/consolidated_code.py"
):
    """
    Utility function used to compress specific code directories and relevant files into a single file.

    Consolidates all .py files in the specified code directories (relative to the parent directory),
    all .yaml files in the specified config directory, optionally .md files from all subdirectories,
    and optionally the cells from a Jupyter Notebook.

    Params:
    - parent_directory (str): Base directory for the code directories to consolidate.
    - code_directories (list): List of subdirectories containing .py files to consolidate.
    - include_config (bool): Whether to include the config files in the export
    - include_markdown (bool): Whether to include .md files from all subdirectories
    - config_directory (str): Path to the directory containing .yaml config files.
    - notebook_directory (str): Path to the directory containing the Jupyter Notebook (optional).
    - ipynb_notebook (str): Filename of the Jupyter Notebook to consolidate cells from (optional).
    - output_file (str): Path to the output consolidated .py file.
    """
    # Validate the parent directory
    if not os.path.exists(parent_directory):
        logger.error(f"Parent directory '{parent_directory}' does not exist.")
        raise FileNotFoundError(f"Parent directory '{parent_directory}' does not exist.")

    # If no specific code directories are provided, skip directory processing
    valid_directories = []
    if code_directories:
        # Validate each code directory
        for directory in code_directories:
            full_path = os.path.join(parent_directory, directory)
            if not os.path.exists(full_path):
                logger.warning(f"Code directory '{full_path}' does not exist. Skipping.")
            else:
                valid_directories.append(full_path)

        # Process the valid directories
        for directory in valid_directories:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)

    # Check the config directory
    if not os.path.exists(config_directory):
        logger.error(f"Config directory '{config_directory}' does not exist.")
        raise FileNotFoundError(f"Config directory '{config_directory}' does not exist.")

    # Open the output file and write linting disable lines at the top
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write("# pylint: skip-file\n")
        outfile.write("# pyright: reportUnusedImport=false\n\n")

        # Consolidate .py files from the specified code directories
        for directory in valid_directories:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)

                        if not os.path.isfile(file_path):
                            logger.warning(f"File '{file_path}' does not exist. Skipping.")
                            continue

                        relative_path = os.path.relpath(file_path, start=os.getcwd())
                        outfile.write(f"# {'-'*80}\n")
                        outfile.write(f"# Python File: {relative_path}\n")
                        outfile.write(f"# {'-'*80}\n\n")

                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())

                        outfile.write(f"\n# {'-'*80}\n")
                        outfile.write(f"# End of Python File: {relative_path}\n")
                        outfile.write(f"# {'-'*80}\n\n")

        # Consolidate .yaml files from the config directory
        if include_config:
            for root, _, files in os.walk(config_directory):
                for file in files:
                    if file.endswith('.yaml'):
                        file_path = os.path.join(root, file)

                        if not os.path.isfile(file_path):
                            logger.warning(f"File '{file_path}' does not exist. Skipping.")
                            continue

                        relative_path = os.path.relpath(file_path, start=os.getcwd())
                        outfile.write(f"# {'-'*80}\n")
                        outfile.write(f"# YAML Config File: {relative_path}\n")
                        outfile.write(f"# {'-'*80}\n\n")

                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())

                        outfile.write(f"\n# {'-'*80}\n")
                        outfile.write(f"# End of YAML Config File: {relative_path}\n")
                        outfile.write(f"# {'-'*80}\n\n")

        # Consolidate .md files from all subdirectories
        if include_markdown:
            markdown_files = []
            for root, _, files in os.walk(parent_directory):
                for file in files:
                    if file.endswith('.md'):
                        file_path = os.path.join(root, file)
                        markdown_files.append(file_path)

            for file_path in sorted(markdown_files):
                if not os.path.isfile(file_path):
                    logger.warning(f"File '{file_path}' does not exist. Skipping.")
                    continue

                relative_path = os.path.relpath(file_path, start=os.getcwd())
                outfile.write(f"# {'-'*80}\n")
                outfile.write(f"# Markdown File: {relative_path}\n")
                outfile.write(f"# {'-'*80}\n\n")

                with open(file_path, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())

                outfile.write(f"\n# {'-'*80}\n")
                outfile.write(f"# End of Markdown File: {relative_path}\n")
                outfile.write(f"# {'-'*80}\n\n")

        # Optionally consolidate cells from a Jupyter Notebook
        if notebook_directory and ipynb_notebook:
            notebook_path = os.path.join(notebook_directory, ipynb_notebook)

            if not os.path.isfile(notebook_path):
                logger.warning(f"Notebook '{notebook_path}' does not exist. Skipping.")
            else:
                outfile.write(f"# {'-'*80}\n")
                outfile.write(f"# Jupyter Notebook: {ipynb_notebook}\n")
                outfile.write(f"# {'-'*80}\n\n")

                with open(notebook_path, 'r', encoding='utf-8') as notebook_file:
                    notebook_data = json.load(notebook_file)
                    for cell in notebook_data.get('cells', []):
                        if cell.get('cell_type') == 'code':
                            code_lines = cell.get('source', [])
                            outfile.write("# Code from notebook cell:\n")
                            outfile.writelines(code_lines)
                            outfile.write("\n\n")

                outfile.write(f"# {'-'*80}\n")
                outfile.write(f"# End of Jupyter Notebook: {ipynb_notebook}\n")
                outfile.write(f"# {'-'*80}\n\n")

    logger.info(f"Consolidation complete. All files are saved in {output_file}")

# Define globally so other modules can safely import and use it
MILESTONE_LEVEL = 25
logging.MILESTONE = MILESTONE_LEVEL  # Make it accessible like logging.INFO
logging.addLevelName(MILESTONE_LEVEL, "MILESTONE")
def setup_notebook_logger(log_filepath: str = None) -> logging.Logger:
    """
    Sets up logging for notebook development with optional file output.
    Adds a custom MILESTONE log level (between INFO and WARNING), and colorizes terminal output.

    Params:
    - log_filepath (str, optional): Path for log file output
    """
    class ColorFormatter(logging.Formatter):
        """Sets colors for the logs output using tail commands"""
        COLOR_MAP = {
            "MILESTONE": "\033[92m",  # bright green
            "INFO": "\033[0m",        # default
            "WARNING": "\033[93m",    # yellow
            "ERROR": "\033[91m",      # red
            "DEBUG": "\033[90m",      # gray
            "CRITICAL": "\033[95m",   # magenta
        }
        RESET = "\033[0m"

        def format(self, record):
            base_msg = super().format(record)
            color = self.COLOR_MAP.get(record.levelname.replace(self.RESET, ""), "")
            return f"{color}{base_msg}{self.RESET}" if color else base_msg

    def milestone(self, message, *args, **kwargs):
        if self.isEnabledFor(MILESTONE_LEVEL):
            kwargs.setdefault("stacklevel", 2)  # point to the caller, not this function
            self._log(MILESTONE_LEVEL, message, args, **kwargs)  # pylint: disable=protected-access

    if not hasattr(logging.Logger, "milestone"):
        logging.Logger.milestone = milestone

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []

    # Terminal handler with color
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(ColorFormatter(
        '[%(asctime)s] %(levelname)s [%(module)s.%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%d/%b/%y %H:%M:%S'
    ))
    root_logger.addHandler(stream_handler)

    # Optional file handler without color
    if log_filepath:
        # full log
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(module)s.%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%d/%b/%Y %H:%M:%S'
        ))
        root_logger.addHandler(file_handler)

        # milestone-only log
        base, ext = os.path.splitext(log_filepath)
        milestone_path = f"{base}_milestone{ext}"
        milestone_handler = logging.FileHandler(milestone_path)
        milestone_handler.setLevel(MILESTONE_LEVEL)
        milestone_handler.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(module)s.%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%d/%b/%Y %H:%M:%S'
        ))
        root_logger.addHandler(milestone_handler)

    # Terminal display handler with colors and simplified format
    base, ext = os.path.splitext(log_filepath)
    terminal_path = f"{base}_display{ext}"
    terminal_handler = logging.FileHandler(terminal_path)

    # Custom formatter for terminal viewing - no filepath, includes colors
    class TerminalColorFormatter(ColorFormatter):
        """Assigns colors to logs"""
        def format(self, record):
            # Pad log level to 9 characters, left-aligned
            padded_level = f"{record.levelname:<9}"

            # Format timestamp
            timestamp = self.formatTime(record, '%H:%M:%S')

            # Simple format: [time] LEVEL     | message
            base_msg = f"[{timestamp}] {padded_level} | {record.getMessage()}"

            color = self.COLOR_MAP.get(record.levelname, "")
            return f"{color}{base_msg}{self.RESET}" if color else base_msg

    terminal_handler.setFormatter(TerminalColorFormatter())
    root_logger.addHandler(terminal_handler)

    # Milestone display handler - same format as terminal display but milestone+ only
    milestone_display_path = f"{base}_display_milestone{ext}"
    milestone_display_handler = logging.FileHandler(milestone_display_path)
    milestone_display_handler.setLevel(MILESTONE_LEVEL)
    milestone_display_handler.setFormatter(TerminalColorFormatter())
    root_logger.addHandler(milestone_display_handler)

    return root_logger


def archive_logs(log_filepath: str = "../logs/notebook_logs.log") -> bool:
    """
    Archives the 4 notebook log files by compressing and moving to archived/ directory.

    Params:
    - log_filepath (str): Base log filepath, defaults to "logs/notebook_logs.log"

    Returns:
    - bool: True if archival successful, False otherwise
    """
    # Extract base directory and filename components
    log_path = Path(log_filepath)
    log_dir = log_path.parent
    base_name = log_path.stem  # "notebook_logs"

    # Define the 4 files to archive
    files_to_archive = [
        f"{base_name}.log",
        f"{base_name}_milestone.log",
        f"{base_name}_display.log",
        f"{base_name}_display_milestone.log"
    ]

    # Verify archived directory exists
    archived_dir = log_dir / "archived"
    if not archived_dir.exists():
        raise FileNotFoundError(f"Archived directory not found: {archived_dir}")

    # Generate timestamp for archived filenames
    timestamp = datetime.now().strftime("%y%m%d")

    # Check for other log files and warn
    if log_dir.exists():
        all_log_files = [f.name for f in log_dir.glob("*.log")]
        non_archived_files = [f for f in all_log_files if f not in files_to_archive]

        if non_archived_files:
            logger.warning(f"Non-archived log files found: {non_archived_files}")

    archived_files = []

    try:
        # Archive each file
        for filename in files_to_archive:
            source_path = log_dir / filename

            # Skip if file doesn't exist
            if not source_path.exists():
                logger.warning(f"Expected log file not found: {source_path}")
                continue

            # Create archived filename
            name_parts = filename.split('.')
            archived_filename = f"{name_parts[0]}_{timestamp}.{name_parts[1]}.gz"
            archived_path = archived_dir / archived_filename

            # Compress and copy to archived directory
            with source_path.open('rb') as f_in:
                with gzip.open(archived_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Verify the archived file was created successfully
            if not archived_path.exists() or archived_path.stat().st_size == 0:
                logger.error(f"Failed to create archived file: {archived_path}")
                return False

            archived_files.append((source_path, archived_path))
            logger.info(f"Archived {filename} -> {archived_filename}")

        # Truncate original files only after all archives are verified
        for source_path, archived_path in archived_files:
            source_path.write_text("")  # Truncate file
            logger.info(f"Truncated {source_path.name}")

        # Only log success if we actually archived files
        if archived_files:
            logger.milestone(f"Successfully archived {len(archived_files)} log files with timestamp {timestamp}")
        else:
            logger.warning("No log files were archived - all expected files were missing")

    except Exception as e:
        logger.error(f"Error during log archival: {e}")
