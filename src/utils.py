"""
utility functions use in data science notebooks
"""
import time
import os
from datetime import datetime, timedelta
import logging
import warnings
import functools
import yaml
import progressbar
from pydantic import ValidationError
import dreams_core.core as dc

# pydantic config files
from config_models.config import MainConfig
from config_models.metrics_config import MetricsConfig

# set up logger at the module level
logger = dc.setup_logger()


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
        period_dates = calculate_period_dates(config_dict['training_data'])
        config_dict['training_data'].update(period_dates)

    # If a pydantic definition exists for the config, return it in json format
    filename = os.path.basename(file_path)

    with warnings.catch_warnings():
        # Suppresses pydantic "serialized value may not be as expected" warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        try:
            if filename in ['config.yaml', 'test_config.yaml']:
                config_pydantic = MainConfig(**config_dict)
                config = config_pydantic.model_dump(mode="json", exclude_none=True)

            elif filename in ['metrics_config.yaml', 'test_metrics_config.yaml']:
                config_pydantic = MetricsConfig(**config_dict)
                config = config_pydantic.model_dump(mode="json", exclude_none=True)

            # Otherwise return the normal dict
            else:
                config = config_dict

        except ValidationError as e:
            # Enhanced error reporting
            logger.error("Validation Error in %s:", filename)
            for err in e.errors():
                issue = err['msg']
                print(f"Issue: {issue}")
                location = '.'.join(str(item) for item in err['loc'][:-1])
                print(f"Location: {location}")
                missing_field = err['loc'][-1]
                print(f"Bad Field: {missing_field}")
            raise e

    return config



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



# helper function for load_config
def calculate_period_dates(config):
    """
    Calculate the training and modeling period start and end dates based on the provided
    durations and the modeling period start date. The calculated dates will include both
    the start and end dates, ensuring the correct number of days for each period.

    Args:
        config (dict): Configuration dictionary containing:
        - 'modeling_period_start' (str): Start date of the modeling period in 'YYYY-MM-DD' format.
        - 'modeling_period_duration' (int): Duration of the modeling period in days.
        - 'training_period_duration' (int): Duration of the training period in days.

    Returns:
        dict: Dictionary containing:
        - 'training_period_start' (str): Calculated start date of the training period.
        - 'training_period_end' (str): Calculated end date of the training period.
        - 'modeling_period_end' (str): Calculated end date of the modeling period.
    """
    # Extract the config values
    modeling_period_start = datetime.strptime(config['modeling_period_start'], '%Y-%m-%d')
    modeling_period_duration = config['modeling_period_duration']  # in days
    training_period_duration = config['training_period_duration']  # in days

    # Calculate modeling_period_end (inclusive of the start date)
    modeling_period_end = modeling_period_start + timedelta(days=modeling_period_duration - 1)

    # Calculate training_period_end (just before modeling_period_start)
    training_period_end = modeling_period_start - timedelta(days=1)

    # Calculate training_period_start (inclusive of the start date)
    training_period_start = training_period_end - timedelta(days=training_period_duration - 1)

    # Return updated config with calculated values
    return {
        'training_period_start': training_period_start.strftime('%Y-%m-%d'),
        'training_period_end': training_period_end.strftime('%Y-%m-%d'),
        'modeling_period_end': modeling_period_end.strftime('%Y-%m-%d')
    }



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
