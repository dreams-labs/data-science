"""
functions used to build coin-level features from training data
"""
# pylint: disable=C0301 # line over 100 chars
# pylint: disable=C0303 # trailing whitespace

import time
import pandas as pd
import dreams_core.core as dc

# set up logger at the module level
logger = dc.setup_logger()


def flatten_coin_date_df(df, metrics_config, training_period_end):
    """
    Processes all coins in the DataFrame and flattens relevant time series metrics for each coin.

    Params:
    - df (pd.DataFrame): DataFrame containing time series data for multiple coins (coin_id-date).
    - metrics_config (dict): Configuration object with metric rules from the metrics file.
    - training_period_end (datetime): The end of the training period to ensure dates are filled until 
        this date and that rolling windows end on the training_period_end

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
        raise ValueError("Input DataFrame is empty. Check your data source and ensure it's populated correctly.")
    
    # Check that all coin-date pairs are complete for the full df
    missing_dates_dict = {}
    for coin_id in df['coin_id'].unique():

        coin_df = df[df['coin_id'] == coin_id]
        
        # Create the full date range for the coin, explicitly cast to pd.Timestamp
        full_date_range = pd.to_datetime(pd.date_range(start=coin_df['date'].min(), end=training_period_end)).to_pydatetime()

        # Get the existing dates for the coin, explicitly cast to pd.Timestamp
        existing_dates = set(pd.to_datetime(coin_df['date'].unique()).to_pydatetime())
        
        # Find the missing dates by subtracting existing from full date range
        missing_dates = set(full_date_range) - existing_dates
        
        # Store the missing dates for the current coin_id
        missing_dates_dict[coin_id] = sorted(missing_dates)

    # Convert to DataFrame for easier display
    missing_dates_df = pd.DataFrame(list(missing_dates_dict.items()), columns=['coin_id', 'missing_dates'])
    if any(len(missing_dates_df) > 0 for missing in missing_dates):
        raise ValueError(f"Timeseries contains missing dates. Ensure all dates are filled up to the training_period_end for all coins. Missing dates found: {missing_dates}")
    
    # Check for NaN values
    if df.isnull().values.any():
        raise ValueError("Timeseries contains NaN values. Ensure imputation is done upstream before calling flatten_coin_date_df().")

    # Check if df columns have configurations in metrics_config
    configured_metrics = []
    for m in metrics_config['metrics'].keys():
        if m in df.columns:
            configured_metrics.append(m)
    if len(configured_metrics) == 0:
        raise ValueError("No configurations were found in metrics_config for any columns in the input df.")


    # Step 2: Flatten the metrics
    # ---------------------------    
    start_time = time.time()
    logger.info("Flattening columns %s into coin-level features...", configured_metrics)

    all_flat_features = []

    # Loop through each unique coin_id
    for coin_id in df['coin_id'].unique():
        # Filter the data for the current coin
        coin_df = df[df['coin_id'] == coin_id].copy()

        # Flatten the features for this coin
        flat_features = flatten_coin_features(coin_df, metrics_config)
        all_flat_features.append(flat_features)
    
    # Convert the list of feature dictionaries into a DataFrame
    result = pd.DataFrame(all_flat_features)

    logger.info('Flattened input df into coin-level features with shape %s after %.2f seconds.', result.shape, time.time() - start_time)

    
    return result



def flatten_coin_features(coin_df, metrics_config):
    """
    Flattens all relevant time series metrics for a single coin into a row of features.

    Params:
    - coin_df (pd.DataFrame): DataFrame with time series data for a single coin (coin_id-date).
    - metrics_config (dict): Configuration object with metric rules from the metrics file.

    Returns:
    - flat_features (dict): A dictionary of flattened features for the coin.

    Raises:
    - KeyError: If a metric in the DataFrame is not found in the configuration.
    - ValueError: If an expected column (e.g., a metric) is missing from the DataFrame.
    """
    # Check if the required 'coin_id' column is present
    if 'coin_id' not in coin_df.columns:
        raise ValueError("The input DataFrame is missing the required 'coin_id' column.")
    
    flat_features = {'coin_id': coin_df['coin_id'].iloc[0]}  # Initialize with coin_id

    # Access the 'metrics' section directly from the config
    metrics_section = metrics_config.get('metrics', {})
    
    # Apply global stats calculations for each metric
    for metric, config in metrics_section.items():
        if metric not in coin_df.columns:
            raise ValueError(f"Metric '{metric}' is missing from the input DataFrame.")

        ts = coin_df[metric].copy()  # Get the time series for this metric

        # Standard aggregations
        if 'aggregations' in config:
            for agg in config['aggregations']:
                if agg == 'sum':
                    flat_features[f'{metric}_sum'] = ts.sum()
                elif agg == 'mean':
                    flat_features[f'{metric}_mean'] = ts.mean()
                elif agg == 'median':
                    flat_features[f'{metric}_median'] = ts.median()
                elif agg == 'min':
                    flat_features[f'{metric}_min'] = ts.min()
                elif agg == 'max':
                    flat_features[f'{metric}_max'] = ts.max()
                elif agg == 'std':
                    flat_features[f'{metric}_std'] = ts.std()
                else:
                    raise KeyError(f"Aggregation '{agg}' for metric '{metric}' is not recognized.")

        # Rolling window calculations
        rolling = config.get('rolling', False)
        if rolling:
            rolling_stats = config['rolling']['stats']  # Get rolling stats
            comparisons = config['rolling'].get('comparisons', [])  # Get comparisons
            window_duration = config['rolling']['window_duration']
            lookback_periods = config['rolling']['lookback_periods']

            # Calculate rolling metrics and update flat_features
            rolling_features = calculate_rolling_window_features(
                ts, window_duration, lookback_periods, rolling_stats, comparisons, metric)
            flat_features.update(rolling_features)

    return flat_features



def calculate_rolling_window_features(ts, window_duration, lookback_periods, rolling_stats, comparisons, metric_name):
    """
    Calculates rolling window features and comparisons for a given time series based on
    configurable window duration and lookback periods.

    Params:
    - ts (pd.Series): The time series data for the metric.
    - window_duration (int): The size of each rolling window (e.g., 7 for 7 days).
    - lookback_periods (int): The number of lookback periods to calculate (e.g., 2 for two periods).
    - rolling_stats (list): The statistics to calculate for each rolling window (e.g., ['sum', 'max']).
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
            # Slice the time series to get the data for the current rolling window
            rolling_window = ts.iloc[start_period:end_period]

            # Loop through each statistic to calculate for the rolling window
            for stat in rolling_stats:
                features[f'{metric_name}_{stat}_{window_duration}d_period_{i+1}'] = calculate_stat(rolling_window, stat)

            # If the rolling window has enough data, calculate comparisons
            if len(rolling_window) > 0:
                for comparison in comparisons:
                    if comparison == 'change':
                        features[f'{metric_name}_change_{window_duration}d_period_{i+1}'] = rolling_window.iloc[-1] - rolling_window.iloc[0]
                    elif comparison == 'pct_change':
                        start_value = rolling_window.iloc[0]
                        end_value = rolling_window.iloc[-1]
                        features[f'{metric_name}_pct_change_{window_duration}d_period_{i+1}'] = calculate_adj_pct_change(start_value, end_value)

    return features  # Return the dictionary of rolling window features



def calculate_adj_pct_change(start_value, end_value, cap=1000, impute_value=1):
    """
    Calculates the adjusted percentage change between two values.
    Handles cases where the start_value is 0 by imputing a value and caps extreme percentage changes.
    
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
            return min((end_value / impute_value - 1) * 100, cap)
    else:
        pct_change = (end_value / start_value - 1) * 100
        return min(pct_change, cap)  # Cap the percentage change if it's too large



def calculate_global_stats(ts, metric_name, config):
    """
    Calculates global statistics for a given time series based on the configuration.

    Params:
    - ts (pd.Series): The time series data for the metric.
    - metric_name (str): The name of the metric (e.g., 'buyers_new').
    - config (dict): The configuration object that specifies which statistics to calculate
                     for each metric.

    Returns:
    - stats (dict): A dictionary containing calculated statistics. The keys are in the format
                    of '{metric_name}_{aggregation}' (e.g., 'buyers_new_sum').
    
    Example:
    If the config specifies 'sum' and 'mean' for 'buyers_new', the result will include:
    {
        'buyers_new_sum': value,
        'buyers_new_mean': value
    }
    """
    stats = {}  # Initialize an empty dictionary to hold the calculated stats.
    
    # Get the aggregation functions for the given metric from the config.
    metric_config = config['metrics'].get(metric_name, [])
    
    # Loop through each aggregation function and calculate the stat.
    for agg in metric_config:
        # Use the helper function 'calculate_stat' to calculate each aggregation.
        stats[f'{metric_name}_{agg}'] = calculate_stat(ts, agg)
    
    return stats  # Return the dictionary of calculated statistics.



def calculate_stat(ts, stat):
    """
    Helper function to calculate a given statistic for a time series.

    Params:
    - ts (pd.Series): Time series data.
    - stat (str): The statistic to calculate (e.g., 'sum', 'mean').

    Returns:
    - The calculated statistic value.
    
    Raises:
    - KeyError: If the statistic is not recognized.
    """
    if stat == 'sum':
        return ts.sum()
    elif stat == 'mean':
        return ts.mean()
    elif stat == 'median':
        return ts.median()
    elif stat == 'std':
        return ts.std()
    elif stat == 'max':
        return ts.max()
    elif stat == 'min':
        return ts.min()
    else:
        raise KeyError(f"Invalid statistic: '{stat}'")
