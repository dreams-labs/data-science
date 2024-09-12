"""
functions used to build coin-level features from training data
"""
# pylint: disable=C0301
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

def calculate_global_stats(ts):
    """Calculates basic statistics for a time series."""
    return {
        'mean': ts.mean(),
        'std': ts.std(),
        'min': ts.min(),
        'max': ts.max(),
        'median': ts.median(),
        # 'last_value': ts.iloc[-1],
        # 'kurtosis': ts.kurtosis(),
        # 'skewness': ts.skew(),
        # 'autocorrelation': ts.autocorr(),
        # 'z_score': (ts.iloc[-1] - ts.mean()) / ts.std() if ts.std() != 0 else np.nan,
        # 'change': ts.iloc[-1] - ts.iloc[0],
        # 'pct_change': (ts.iloc[-1] / ts.iloc[0] - 1) if ts.iloc[0] != 0 else np.nan
    }

def calculate_rolling_window_features(ts, window_sizes):
    """Calculates rolling window features for specified window sizes."""
    features = {}
    for window in window_sizes:
        if len(ts) >= window:
            rolling_window = ts.rolling(window=window)
            features[f'mean_{window}d'] = rolling_window.mean().iloc[-1]
            features[f'std_{window}d'] = rolling_window.std().iloc[-1]
            features[f'min_{window}d'] = rolling_window.min().iloc[-1]
            features[f'max_{window}d'] = rolling_window.max().iloc[-1]
            features[f'median_{window}d'] = rolling_window.median().iloc[-1]
            features[f'last_{window}d_change'] = ts.iloc[-1] - ts.iloc[-window]
            features[f'last_{window}d_pct_change'] = (ts.iloc[-1] / ts.iloc[-window] - 1) if ts.iloc[-window] != 0 else np.nan
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

# Now, we refactor the main flatten_time_series function
def flatten_time_series(ts, window_sizes=[7, 30], bollinger_window=20, decompose_model='additive', freq=None):
    """
    Convert a time series into a row of features using various window sizes, and also 
    include time series decomposition metrics (trend, seasonal, residual).
    
    Params:
    - ts (pd.Series): A single time series of numeric values (e.g., total_bought for a coin_id).
    - window_sizes (list): List of rolling window sizes to compute features over (default: [7, 30]).
    - bollinger_window (int): The window size for calculating Bollinger Bands (default: 20).
    - decompose_model (str): 'additive' or 'multiplicative' decomposition (default: 'additive').
    - freq (int): The frequency of the seasonality (e.g., 7 for weekly, 30 for monthly).
    
    Returns:
    - features (dict): A dictionary of features for the time series, where keys are feature names.
    """
    features = {}
    features.update(calculate_global_stats(ts))
    # features.update(calculate_rolling_window_features(ts, window_sizes))
    # features.update(calculate_bollinger_bands(ts, bollinger_window))
    # features.update(calculate_decomposition_features(ts, decompose_model, freq))

    return features



def flatten_coin_features(coin_df, metrics, window_sizes=[7, 30], 
                          bollinger_window=20, decompose_model='additive', freq=None):
    """
    Flattens all relevant time series metrics for a single coin into a row of features.
    
    Params:
    - coin_df (pd.DataFrame): DataFrame with time series data for a single coin (coin_id-date).
    - metrics (list): List of time series column names to flatten into features.
    - window_sizes (list): List of rolling window sizes to compute features over.
    - bollinger_window (int): The window size for calculating Bollinger Bands (default: 20).
    - decompose_model (str): 'additive' or 'multiplicative' decomposition (default: 'additive').
    - freq (int): The frequency of the seasonality (e.g., 7 for weekly, 30 for monthly).

    Returns:
    - flat_features (dict): A dictionary of flattened features for the coin.
    """
    flat_features = {'coin_id': coin_df['coin_id'].iloc[0]}  # Initialize with coin_id

    # Apply flatten_time_series to each time series (metric) in the metrics list
    for metric in metrics:
        ts = coin_df[metric].copy()
        ts_features = flatten_time_series(ts, window_sizes=window_sizes, 
                                          bollinger_window=bollinger_window, 
                                          decompose_model=decompose_model, 
                                          freq=freq)
        
        # Prefix each feature name with the metric to avoid name clashes
        for feature_name, feature_value in ts_features.items():
            flat_features[f'{metric}_{feature_name}'] = feature_value
    
    return flat_features



def process_all_coins(df, metrics, window_sizes=[7, 30], 
                      bollinger_window=20, decompose_model='additive', freq=None):
    """
    Processes all coins in the DataFrame and flattens relevant time series metrics for each coin.
    
    Params:
    - df (pd.DataFrame): DataFrame containing time series data for multiple coins (coin_id-date).
    - metrics (list): List of time series column names to flatten into features.
    - window_sizes (list): List of rolling window sizes to compute features over.
    - bollinger_window (int): The window size for calculating Bollinger Bands.
    - decompose_model (str): 'additive' or 'multiplicative' decomposition.
    - freq (int): The frequency of the seasonality (e.g., 7 for weekly, 30 for monthly).
    
    Returns:
    - result (pd.DataFrame): A DataFrame of flattened features for all coins.
    """
    all_flat_features = []

    # Loop through each unique coin_id
    for coin_id in df['coin_id'].unique():
        # Filter the data for the current coin
        coin_df = df[df['coin_id'] == coin_id].copy()

        # Flatten the features for this coin
        flat_features = flatten_coin_features(coin_df, metrics, window_sizes, 
                                              bollinger_window, decompose_model, freq)
        all_flat_features.append(flat_features)
    
    # Convert the list of feature dictionaries into a DataFrame
    result = pd.DataFrame(all_flat_features)
    
    return result
