"""
utility functions use in data science notebooks
"""
import time
import sys
import os
import json
import gc
import inspect
from datetime import datetime, timedelta
from typing import List,Dict,Any
import importlib
import itertools
import logging
import warnings
import functools
import yaml
import psutil
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



# ---------------------------------------- #
#         Config Related Functions
# ---------------------------------------- #

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








# --------------------------------- #
#        Function Modifiers
# --------------------------------- #

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








# ---------------------------------------- #
#     DataFrame Manipulation Functions
# ---------------------------------------- #

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

    # Get the min and max values of the colum
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


        # Confirm the profits_df columns are in df
        profits_df_columns = ['usd_balance', 'usd_net_transfers', 'usd_inflows', 'is_imputed']
        if not set(profits_df_columns).issubset(df.columns):
            raise ValueError("Dataframe contains 'usd_balance' column but does not have the "
                            f"expected profits_df columns of {profits_df_columns}.")

        # Confirm imputed dates are only at the starting balance and period end dates
        imputed_dates = set(df[df['is_imputed']]['date'])
        unexpected_imputed_dates = imputed_dates - set([period_starting_balance_date, period_end])
        if len(unexpected_imputed_dates) > 0:
            formatted_dates = {d.strftime("%Y-%m-%d") for d in unexpected_imputed_dates}
            raise ValueError(f"Unexpected imputed dates found on {formatted_dates}.")


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
        if not (starting_balance_profits_df['is_imputed'] == True).all():
            raise ValueError("Found non-imputed records in starting balance")




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


def df_nans(df):
    """
    Returns rows and cells in a df that contain NaN values.
    """
    na_rows = df[df.isna().any(axis=1)]
    na_cells = na_rows.loc[:, na_rows.isna().any()]

    return na_rows, na_cells


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
    lower_bound = np.percentile(valid_data, cutoff * 100, method='nearest')
    upper_bound = np.percentile(valid_data, (1 - cutoff) * 100, method='nearest')

    # Clip the data
    return np.clip(winsorized, lower_bound, upper_bound)







# ---------------------------------------- #
#     Misc Notebook Support Functions
# ---------------------------------------- #

# silence donation message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
def notify(sound_file_path=None):
    """
    Play a notification sound from a local audio file using pygame asynchronously.
    Falls back to ALERT_SOUND_FILEPATH environment variable if no path provided.
    Returns early if no valid sound file path is found.

    Args:
        sound_file_path (str, optional): Path to the sound file (supports .wav format)
    """
    sound_file_path = sound_file_path or os.getenv('ALERT_SOUND_FILEPATH')
    if not sound_file_path:
        return "No sound file found."

    try:
        if not pygame.mixer.get_init():  # Initialize mixer if not already done
            pygame.mixer.init()

        sound = pygame.mixer.Sound(sound_file_path)
        sound.play()

        # Don't wait for the sound to finish - return immediately
        return

    except Exception as e:  # pylint:disable=broad-exception-caught
        return f"Error playing sound: {e}"


# pylint: disable=dangerous-default-value
def export_code(
    code_directories=[],
    parent_directory="..//src",
    include_config=False,
    config_directory="..//config",
    notebook_directory="..//notebooks",
    ipynb_notebook=None,
    output_file="temp/consolidated_code.py"
):
    """
    Utility function used to compress specific code directories and relevant files into a single file.

    Consolidates all .py files in the specified code directories (relative to the parent directory),
    all .yaml files in the specified config directory, and optionally the cells from a Jupyter Notebook.

    Params:
    - parent_directory (str): Base directory for the code directories to consolidate.
    - code_directories (list): List of subdirectories containing .py files to consolidate.
    - include_config (bool): Whether to include the config files in the export
    - config_directory (str): Path to the directory containing .yaml config files.
    - notebook_directory (str): Path to the directory containing the Jupyter Notebook (optional).
    - ipynb_notebook (str): Filename of the Jupyter Notebook to consolidate cells from (optional).
    - output_file (str): Path to the output consolidated .py file.
    """
    # Validate the parent directory
    if not os.path.exists(parent_directory):
        logger.error(f"Parent directory '{parent_directory}' does not exist.")
        raise FileNotFoundError(f"Parent directory '{parent_directory}' does not exist.")

    # If no specific code directories are provided, default to all subdirectories
    if not code_directories:
        logger.warning("No code directories specified. Defaulting to all directories under the parent directory.")
        code_directories = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]

    # Validate each code directory
    valid_directories = []
    for directory in code_directories:
        full_path = os.path.join(parent_directory, directory)
        if not os.path.exists(full_path):
            logger.warning(f"Code directory '{full_path}' does not exist. Skipping.")
        else:
            valid_directories.append(full_path)

    if not valid_directories:
        logger.error("No valid code directories found.")
        raise ValueError("None of the specified code directories exist under the parent directory.")

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
