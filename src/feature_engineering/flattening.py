"""
functions used to build coin-level features from training data
"""
import os
from datetime import datetime
import time
import copy
import pandas as pd
import dreams_core.core as dc

# set up logger at the module level
logger = dc.setup_logger()


def flatten_coin_date_df(df, df_metrics_config, training_period_end):
    """
    Processes all coins in the DataFrame and flattens relevant time series metrics for each coin.

    Params:
    - df (pd.DataFrame): DataFrame containing time series data for multiple coins (coin_id-date).
    - df_metrics_config (dict): Configuration object showing the settings for the metrics
        that apply to the specific input df.
    - training_period_end (datetime): The end of the training period to ensure dates are filled
        until this date and that rolling windows end on the training_period_end

    Returns:
    - result (pd.DataFrame): A DataFrame of flattened features for all coins.

    Raises:
    - ValueError:
        - If the input DataFrame is empty.
        - If the time series data contains missing dates or NaN values.
    """

    # Step 1: Data Quality Checks
    # ---------------------------
    # Check if df is empty
    if df.empty:
        raise ValueError("Input DataFrame is empty. Check your data and ensure it's populated.")

    # Check that all coin-date pairs are complete for the full df
    missing_dates_dict = {}
    for coin_id in df['coin_id'].unique():

        coin_df = df[df['coin_id'] == coin_id]

        # Create the full date range for the coin, explicitly cast to pd.Timestamp
        full_date_range = pd.date_range(start=coin_df['date'].min(), end=training_period_end)

        # Get the existing dates for the coin, explicitly cast to pd.Timestamp
        existing_dates = set(pd.to_datetime(coin_df['date'].unique()).to_pydatetime())

        # Find the missing dates by subtracting existing from full date range
        missing_dates = set(full_date_range) - existing_dates

        # Store the missing dates for the current coin_id
        missing_dates_dict[coin_id] = sorted(missing_dates)

    # Convert to DataFrame for easier display
    missing_dates_df = pd.DataFrame(list(missing_dates_dict.items())
                                    , columns=['coin_id', 'missing_dates'])
    if any(len(missing_dates_df) > 0 for missing in missing_dates):
        raise ValueError(
            "Timeseries has missing dates. Ensure all dates are filled up to training_period_end."
            f"Missing dates: {missing_dates}"
        )

    # Check for NaN values
    if df.isnull().values.any():
        raise ValueError(
            "Timeseries contains NaN values. Ensure imputation is done upstream before flattening."
        )
    # Check if df columns have configurations in df_metrics_config
    configured_metrics = []
    for m in df_metrics_config.keys():
        if m in df.columns:
            configured_metrics.append(m)
    if len(configured_metrics) == 0:
        raise ValueError("No columns in the input_df were found in df_metrics_config.")


    # Step 2: Flatten the metrics
    # ---------------------------
    start_time = time.time()
    logger.debug("Flattening columns %s into coin-level features...", configured_metrics)

    all_flat_features = []

    # Loop through each unique coin_id
    for coin_id in df['coin_id'].unique():
        # Filter the data for the current coin
        coin_df = df[df['coin_id'] == coin_id].copy()

        # Flatten the features for this coin
        flat_features = flatten_date_features(coin_df, df_metrics_config)

        # Add coin_id to the flattened features
        flat_features['coin_id'] = coin_id

        all_flat_features.append(flat_features)

    # Convert the list of feature dictionaries into a DataFrame
    flattened_df = pd.DataFrame(all_flat_features)

    logger.debug('Flattened input df into coin-level features with shape %s after %.2f seconds.',
                flattened_df.shape, time.time() - start_time)


    return flattened_df



def flatten_date_features(time_series_df, df_metrics_config):
    """
    Flattens all relevant time series metrics for a single coin into a row of features.

    Params:
    - time_series_df (pd.DataFrame): DataFrame keyed on date with a time series dataset
    - df_metrics_config (dict): Configuration object with metric rules from the metrics file for
        the specific input df.

    Returns:
    - flat_features (dict): A dictionary of flattened features for the coin.

    Raises:
    - ValueError: If no columns were matched to the config.
    """
    flat_features = {}
    matched_columns = False

    # promote indicators to the same key level as primary metrics
    df_metrics_indicators_config = promote_indicators_to_metrics(df_metrics_config)

    # Apply global stats calculations for each metric
    for metric, config in df_metrics_indicators_config.items():
        if metric not in time_series_df.columns:
            continue

        matched_columns = True
        ts = time_series_df[metric].copy()  # Get the time series for this metric

        # Standard aggregations
        if 'aggregations' in config:
            for agg, agg_config in config['aggregations'].items():
                agg_value = calculate_aggregation(ts, agg)

                # Generate bucket columns if buckets are specified in the config
                if agg_config and 'buckets' in agg_config:
                    bucket_category = bucketize_value(agg_value, agg_config['buckets'])
                    flat_features[f'{metric}_{agg}_bucket'] = bucket_category

                # Return the aggregate metric if it is not bucketized
                else:
                    flat_features[f'{metric}_{agg}'] = agg_value


        # Rolling window calculations (unchanged)
        rolling = config.get('rolling', False)
        if rolling:

            rolling_aggregations = config['rolling'].get('aggregations', [])
            comparisons = config['rolling'].get('comparisons', [])
            window_duration = config['rolling']['window_duration']
            lookback_periods = config['rolling']['lookback_periods']

            # Calculate rolling metrics and update flat_features
            rolling_features = calculate_rolling_window_features(
                ts, window_duration, lookback_periods, rolling_aggregations, comparisons, metric)
            flat_features.update(rolling_features)

    if not matched_columns:
        raise ValueError("No metrics matched the columns in the DataFrame.")

    return flat_features


def promote_indicators_to_metrics(df_metrics_config):
    """
    Moves indicators to the same level as other columns in the config file so that their
    features can be generated the same way they are for primary metrics.

    If there are multiple windows for an indicator, a key will be generated for each of
    them with all metrics configured for the overall indicator.

    Params:
    - df_metrics_config (dict): Configuration object with metric rules from the metrics
        file for the specific input df.

    Returns:
    """
    # make a deep copy to avoid impacting the original dict
    df_metrics_indicators_config = copy.deepcopy(df_metrics_config)

    for key, value in df_metrics_config.items():
        # Check if indicators are present
        if 'indicators' in value:
            # If there are indicators...
            for indicator, indicator_metrics in value['indicators'].items():

                # ...define a new key for each window....
                for window in indicator_metrics['parameters']['window']:
                    new_key = f"{key}_{indicator}_{window}"

                # ...and create new top-level key for each window
                df_metrics_indicators_config[new_key] = indicator_metrics

            # Remove indicators from their original position
            df_metrics_indicators_config[key].pop('indicators')

    return df_metrics_indicators_config



def bucketize_value(value, buckets):
    """
    Categorizes a value based on the defined buckets.

    Params:
    - value (float): The value to categorize
    - buckets (list): List of dictionaries defining the buckets

    Returns:
    - str: The category the value falls into
    """
    for bucket in buckets:
        for category, threshold in bucket.items():
            if threshold == "remainder" or value <= threshold:
                return category

    # This should never be reached if buckets are properly defined
    raise ValueError(f"Value {value} does not fit in any defined bucket.")



def calculate_rolling_window_features(
        ts,
        window_duration,
        lookback_periods,
        rolling_aggregations,
        comparisons,
        metric_name
    ):
    """
    Calculates rolling window features and comparisons for a given time series based on
    configurable window duration and lookback periods.

    Params:
    - ts (pd.Series): The time series data for the metric.
    - window_duration (int): The size of each rolling window (e.g., 7 for 7 days).
    - lookback_periods (int): The number of lookback periods to calculate (e.g., 2 for two periods).
    - rolling_aggregations (list): The aggregations for each rolling window (e.g. ['sum', 'max']).
    - comparisons (list): The comparisons to make between the first and last value in the window.
    - metric_name (str): The name of the metric to include in the feature names.

    Returns:
    - features (dict): A dictionary containing calculated rolling window features.
    """

    features = {}  # Initialize an empty dictionary to store the rolling window features

    # Calculate the total number of complete periods that fit within the time series
    num_complete_periods = len(ts) // window_duration

    # Ensure we're not calculating more periods than we have data for
    actual_lookback_periods = min(lookback_periods, num_complete_periods)

    # Start processing from the last complete period, moving backwards
    for i in range(actual_lookback_periods):
        # Define the start and end of the current rolling window
        end_period = len(ts) - i * window_duration
        start_period = end_period - window_duration

        # Ensure that the start index is not out of bounds
        if start_period >= 0:
            rolling_window = ts.iloc[start_period:end_period]

            # Loop through each statistic to calculate for the rolling window
            for agg in rolling_aggregations:
                agg_key = f'{metric_name}_{agg}_{window_duration}d_period_{i+1}'
                features[agg_key] = calculate_aggregation(rolling_window, agg)

            # If the rolling window has enough data, calculate comparisons
            if len(rolling_window) > 0:
                for comparison in comparisons:
                    comparison_key = f'{metric_name}_{comparison}_{window_duration}d_period_{i+1}'
                    features[comparison_key] = calculate_comparisons(rolling_window, comparison)

    return features  # Return the dictionary of rolling window features



def calculate_adj_pct_change(start_value, end_value, cap=5, impute_value=1):
    """
    Calculates the adjusted percentage change between two values.
    Handles cases where the start_value is 0 by imputing a value and capping extreme
    percentage changes.

    Params:
    - start_value (float): The value at the beginning of the period.
    - end_value (float): The value at the end of the period.
    - cap (float): The maximum percentage change allowed (default = 1000%).
    - impute_value (float): The value to impute when the start_value is 0 (default = 1).

    Returns:
    - pct_change (float): The calculated or capped percentage change.
    """
    if start_value == 0:
        if end_value == 0:
            return 0  # 0/0 case
        else:
            # Use imputed value to maintain scale of increase
            return min((end_value / impute_value) - 1, cap)
    else:
        pct_change = (end_value / start_value) - 1
        return min(pct_change, cap)  # Cap the percentage change if it's too large


def calculate_aggregation(ts, stat):
    """
    Centralized function to calculate various aggregations on a time series.

    Parameters:
    - ts (pd.Series): The time series to aggregate.
    - stat (str): The type of aggregation to perform.

    Returns:
    - The calculated aggregation value.
    """
    agg_functions = {
        'sum': ts.sum,
        'mean': ts.mean,
        'median': ts.median,
        'std': ts.std,
        'max': ts.max,
        'min': ts.min,
        'first': lambda: ts.iloc[0],
        'last': lambda: ts.iloc[-1]
    }

    if stat not in agg_functions:
        raise KeyError(f"Unsupported aggregation type: '{stat}'.")

    return agg_functions[stat]()


# Helper function for comparison metric formulas
def calculate_comparisons(rolling_window, comparison):
    """
    helper function that calculates the comparison metric for the specific rolling window.
    this helps make the for loop more readable.

    params:
        - rolling_window (pd.Series): the metric series for the given window only
        - comparison (string): the type of comparison calculation to perform
    """
    if comparison == 'change':
        return rolling_window.iloc[-1] - rolling_window.iloc[0]
    elif comparison == 'pct_change':
        start_value, end_value = rolling_window.iloc[0], rolling_window.iloc[-1]
        return calculate_adj_pct_change(start_value, end_value)


def save_flattened_outputs(flattened_df, output_dir, metric_description, modeling_period_start):
    """
    Saves the flattened DataFrame with descriptive metrics into a CSV file.

    Params:
    - flattened_df (pd.DataFrame): The DataFrame containing flattened data.
    - output_dir (str): Directory where the CSV file will be saved.
    - metric_description (str): Description of metrics (e.g., 'buysell_metrics').
    - modeling_period_start (str): Start of the modeling period (e.g., '2023-01-01').

    Returns:
    - flattened_df (pd.DataFrame): The same DataFrame that was passed in.
    - output_path (str): The full path to the saved CSV file.
    """

    # if the coin_id column exists, confirm it is fully unique
    if 'coin_id' in flattened_df.columns:
        if not flattened_df['coin_id'].is_unique:
            raise ValueError("The 'coin_id' column must have fully unique values.")

    # Define filename with metric description and optional description
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    filename = f"{metric_description}_{timestamp}_model_period_{modeling_period_start}.csv"

    # Save file
    output_path = os.path.join(output_dir, filename)
    flattened_df.to_csv(output_path, index=False)

    logger.debug("Saved flattened outputs to %s", output_path)

    return flattened_df, output_path
