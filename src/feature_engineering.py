"""
functions used to build coin-level features from training data
"""
# pylint: disable=C0301
# pylint: disable=C0303 trailing whitespace

import time
from datetime import datetime
from pytz import utc
import pandas as pd
import numpy as np
import pandas_gbq
from statsmodels.tsa.seasonal import seasonal_decompose
from dreams_core.googlecloud import GoogleCloud as dgc
import dreams_core.core as dc

# set up logger at the module level
logger = dc.setup_logger()

def calculate_global_stats(ts, metric_name, config):
    """
    Calculates statistics for a time series based on the configuration.
    
    Params:
    - ts (pd.Series): Time series data for a particular metric.
    - metric_name (str): The name of the metric (e.g., 'buyers_new').
    - config (dict): The configuration object containing metric aggregation rules.

    Returns:
    - stats (dict): A dictionary containing calculated statistics.
    """
    stats = {}
    metric_config = config['metrics'].get(metric_name, [])

    for agg in metric_config:
        if agg == 'sum':
            stats[f'{metric_name}_sum'] = ts.sum()
        elif agg == 'mean':
            stats[f'{metric_name}_mean'] = ts.mean()
        elif agg == 'median':
            stats[f'{metric_name}_median'] = ts.median()
        elif agg == 'std':
            stats[f'{metric_name}_std'] = ts.std()
        elif agg == 'max':
            stats[f'{metric_name}_max'] = ts.max()
        # Add any additional aggregation functions here

    return stats


def calculate_rolling_window_features(ts, window_duration, lookback_periods, rolling_stats, comparisons):
    """
    Calculates rolling window features and comparisons for a given time series based on 
    configurable window duration and lookback periods.

    Parameters:
    - ts (pd.Series): The time series of metrics (e.g., total_bought for a coin_id).
    - window_duration (int): The size of each rolling window (e.g., 7 for 7 days).
    - lookback_periods (int): The number of lookback periods to calculate (e.g., 2 for 2 periods of 7-day windows).
    - rolling_stats (list): The summary statistics to calculate (e.g., ['sum', 'max']).
    - comparisons (list): The comparative metrics to calculate (e.g., ['change', 'pct_change']).

    Returns:
    - features (dict): A dictionary of calculated rolling window features for each lookback period.
      The keys will include the rolling window stat and comparison names (e.g., 'sum_7d_period_1', 
      'change_7d_period_2') and the values are the computed metrics.
    
    Example:
    If rolling_stats includes 'sum' and 'max', and comparisons include 'change' and 'pct_change', 
    the output will include features like:
      - 'sum_7d_period_1', 'max_7d_period_1', 'change_7d_period_1', 'pct_change_7d_period_1'
      - 'sum_7d_period_2', 'max_7d_period_2', 'change_7d_period_2', 'pct_change_7d_period_2'
    
    The rolling window is calculated over each lookback period, and comparisons assess changes 
    between the first and last value in the window.
    """
    features = {}
    
    # Loop through the lookback periods
    for i in range(lookback_periods):
        end_period = len(ts) - i * window_duration
        start_period = end_period - window_duration
        
        if start_period >= 0:
            rolling_window = ts.iloc[start_period:end_period]
            
            # Calculate rolling stats for this window
            for stat in rolling_stats:
                if stat == 'sum':
                    features[f'sum_{window_duration}d_period_{i+1}'] = rolling_window.sum()
                elif stat == 'max':
                    features[f'max_{window_duration}d_period_{i+1}'] = rolling_window.max()
                elif stat == 'min':
                    features[f'min_{window_duration}d_period_{i+1}'] = rolling_window.min()
                elif stat == 'mean':
                    features[f'mean_{window_duration}d_period_{i+1}'] = rolling_window.mean()
                elif stat == 'std':
                    features[f'std_{window_duration}d_period_{i+1}'] = rolling_window.std()
                elif stat == 'median':
                    features[f'median_{window_duration}d_period_{i+1}'] = rolling_window.median()
            
            # Calculate comparisons (change, percentage change)
            if len(rolling_window) > 0:
                for comparison in comparisons:
                    if comparison == 'change':
                        features[f'change_{window_duration}d_period_{i+1}'] = rolling_window.iloc[-1] - rolling_window.iloc[0]
                    elif comparison == 'pct_change' and rolling_window.iloc[0] != 0:
                        features[f'pct_change_{window_duration}d_period_{i+1}'] = (rolling_window.iloc[-1] / rolling_window.iloc[0] - 1)

    return features


# def calculate_bollinger_bands(ts, bollinger_window):
#     """Calculates Bollinger Bands for a given time series and window."""
#     if len(ts) >= bollinger_window:
#         rolling_mean = ts.rolling(window=bollinger_window).mean().iloc[-1]
#         rolling_std = ts.rolling(window=bollinger_window).std().iloc[-1]
#         return {
#             'bollinger_upper': rolling_mean + (rolling_std * 2),
#             'bollinger_lower': rolling_mean - (rolling_std * 2)
#         }
#     return {}

# def calculate_decomposition_features(ts, decompose_model='additive', freq=None):
#     """Performs time series decomposition and returns the trend, seasonal, and residual features."""
#     if freq is not None and len(ts) >= freq * 2:
#         decomposition = seasonal_decompose(ts, model=decompose_model, period=freq)
#         return {
#             'trend_mean': decomposition.trend.mean() if decomposition.trend is not None else np.nan,
#             'seasonal_mean': decomposition.seasonal.mean() if decomposition.seasonal is not None else np.nan,
#             'residual_mean': decomposition.resid.mean() if decomposition.resid is not None else np.nan,
#             'trend_last_value': decomposition.trend.iloc[-1] if decomposition.trend is not None else np.nan,
#             'seasonal_last_value': decomposition.seasonal.iloc[-1] if decomposition.seasonal is not None else np.nan,
#             'residual_last_value': decomposition.resid.iloc[-1] if decomposition.resid is not None else np.nan
#         }
#     return {}


def flatten_coin_features(coin_df, metrics_config):
    """
    Flattens all relevant time series metrics for a single coin into a row of features.

    Params:
    - coin_df (pd.DataFrame): DataFrame with time series data for a single coin (coin_id-date).
    - metrics_config (dict): Configuration object with metric rules from the metrics file.

    Returns:
    - flat_features (dict): A dictionary of flattened features for the coin.
    """
    flat_features = {'coin_id': coin_df['coin_id'].iloc[0]}  # Initialize with coin_id

    # Access the 'metrics' section directly from the config
    metrics_section = metrics_config['metrics']

    # Apply global stats calculations for each metric
    for metric, config in metrics_section.items():  # Loop directly over the 'metrics' section
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
                elif agg == 'std':
                    flat_features[f'{metric}_std'] = ts.std()

        # Rolling window calculations
        rolling = config.get('rolling', False)
        if rolling:
            rolling_stats = config['rolling']['stats']  # Get rolling stats
            comparisons = config['rolling'].get('comparisons', [])  # Get comparisons
            window_duration = config['rolling']['window_duration']
            lookback_periods = config['rolling']['lookback_periods']

            # Calculate rolling metrics and update flat_features
            rolling_features = calculate_rolling_window_features(
                ts, window_duration, lookback_periods, rolling_stats, comparisons)
            flat_features.update(rolling_features)

    return flat_features


def flatten_coin_date_df(df, metrics_config):
    """
    Processes all coins in the DataFrame and flattens relevant time series metrics for each coin.

    Params:
    - df (pd.DataFrame): DataFrame containing time series data for multiple coins (coin_id-date).
    - metrics_config (dict): Configuration object with metric rules from the metrics file.

    Returns:
    - result (pd.DataFrame): A DataFrame of flattened features for all coins.
    """
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
    
    return result
