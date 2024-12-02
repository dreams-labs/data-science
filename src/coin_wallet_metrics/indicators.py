'''
calculates metrics related to the distribution of coin ownership across wallets
'''
# pylint: disable=C0301 # line too long
import sys
from typing import Tuple,Optional
import pandas as pd
import numpy as np
import dreams_core.core as dc

# import coin_wallet_metrics as cwm
sys.path.append('..')
import utils as u  # pylint: disable=C0413  # import must be at top


# set up logger at the module level
logger = dc.setup_logger()


def generate_time_series_indicators(dataset_df, dataset_metrics_config, id_column):
    """
    Generates all indicators for a time series dataframe keyed on coin_id and date. This is
    a wrapper function to apply ind.generate_column_time_series_indicators() to each dataset
    column with indicator configurations.

    Params:
    - dataset_df (DataFrame): The df containing dataset metrics and a coin_id and date column,
        as well as columns needing indicator calculations.
    - dataset_metrics_config (dict): The subcomponent of metrics_config that has keys for the
        columns needing indicators, e.g. metrics_config['time_series']['market_data']
    - id_column: whether the input df has an id column that needs to be grouped on

    Returns:
    - dataset_indicators_df (DataFrame): The original dataset_df with added columns for all
        configured indicators.
    """
    # Calculate indicators for each value column
    for value_column in list(dataset_metrics_config.keys()):

        if 'indicators' in dataset_metrics_config[value_column].keys():
            dataset_df = generate_column_time_series_indicators(
                dataset_df,
                value_column,
                dataset_metrics_config[value_column]['indicators'],
                id_column
            )

    return dataset_df


def generate_column_time_series_indicators(
        time_series_df: pd.DataFrame,
        value_column: str,
        value_column_indicators_config: dict,
        id_column: Optional[str]=None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates time series metrics (e.g., SMA, EMA) based on the given config.
    Works for both multi-series (e.g., multiple coins) and single time series data.

    Params:
    - time_series_df (pd.DataFrame): The time series data with column {value_column} and
        no index.
    - value_column (string): The column in time_series_df that needs indicators
    - value_column_indicators_config: The metrics_config indicators dict for the column
        e.g. metrics_config['time_series']['market_data']['price']['indicators']
    - id_column (Optional[string]): The name of the column used to identify different
        series that should have indicators computed separately. If None, all values
        will be treated as part of the same series.
            e.g. SMA should not average 2 coins' prices together if they are next
            to each other in the df, so we need to group on coin_id

    Returns:
    - time_series_df (pd.DataFrame): Input df with all indicators added
    """
    # Confirm coin_id and date exist
    if 'date' not in time_series_df.columns:
        raise ValueError("Input df to generate_column_time_series_indicators() must have a "
                         "date column.")
    if id_column:
        if id_column not in time_series_df.columns:
            raise ValueError("Input df to generate_column_time_series_indicators() does not "
                             f"have id_column {id_column}.")

    # Confirm value column exists
    if value_column not in time_series_df.columns:
        raise KeyError(f"Input DataFrame does not include column '{value_column}'.")

    # Check for NaN values
    if id_column:
        for group_id, group in time_series_df.groupby(id_column, observed=True):
            if u.check_nan_values(group[value_column]):
                raise ValueError(f"The '{value_column}' column contains NaN values in the middle for {id_column} {group_id}.")
    else:
        if u.check_nan_values(time_series_df[value_column]):
            raise ValueError(f"The '{value_column}' column contains NaN values in the middle of the series.")

    # Defining index to group on
    # --------------------------
    # If there is an id_column, group indicator calculations on it
    if id_column:
        groupby_column = id_column
    # If there isn't, create a dummy_column for grouping and remove it later
    else:
        time_series_df['dummy_group'] = 1
        groupby_column = 'dummy_group'

    time_series_df = time_series_df.set_index([groupby_column,'date'])

    # Indicator Calculations
    # ----------------------
    # For each indicator, loop through all options and add the appropriate column
    for indicator, indicator_config in value_column_indicators_config.items():
        if indicator == 'sma':
            windows = indicator_config['parameters']['window']
            for w in windows:
                ind_series = time_series_df.groupby(level=groupby_column, observed=True)[value_column].transform(
                    lambda x, w=w: calculate_sma(x, w))
                time_series_df[f"{value_column}_{indicator}_{w}"] = ind_series

        elif indicator == 'ema':
            windows = indicator_config['parameters']['window']
            for w in windows:
                ind_series = time_series_df.groupby(level=groupby_column, observed=True)[value_column].transform(
                    lambda x, w=w: calculate_ema(x, w))
                time_series_df[f"{value_column}_{indicator}_{w}"] = ind_series

        elif indicator == 'rsi':
            windows = indicator_config['parameters']['window']
            for w in windows:
                ind_series = time_series_df.groupby(level=groupby_column, observed=True)[value_column].transform(
                    lambda x, w=w: calculate_rsi(x, w))
                time_series_df[f"{value_column}_{indicator}_{w}"] = ind_series

        elif indicator == 'bollinger_bands_upper':
            windows = indicator_config['parameters']['window']
            num_std = indicator_config['parameters'].get('num_std', None)
            for w in windows:
                ind_series = time_series_df.groupby(level=groupby_column, observed=True)[value_column].transform(
                    lambda x, w=w, num_std=num_std: calculate_bollinger_bands(x, 'upper', w, num_std))
                time_series_df[f"{value_column}_{indicator}_{w}"] = ind_series

        elif indicator == 'bollinger_bands_lower':
            windows = indicator_config['parameters']['window']
            num_std = indicator_config['parameters'].get('num_std', None)
            for w in windows:
                ind_series = time_series_df.groupby(level=groupby_column, observed=True)[value_column].transform(
                    lambda x, w=w, num_std=num_std: calculate_bollinger_bands(x, 'lower', w, num_std))
                time_series_df[f"{value_column}_{indicator}_{w}"] = ind_series

    # Reset index
    time_series_df = time_series_df.reset_index()

    # Remove the dummy column if it was created
    if groupby_column == 'dummy_group':
        time_series_df = time_series_df.drop('dummy_group', axis=1)


    # Recheck for NaN values in the middle of series
    if id_column:
        for group_id, group in time_series_df.groupby(id_column, observed=True):
            if u.check_nan_values(group[value_column]):
                raise ValueError(f"The '{value_column}' column contains NaN values in the middle for {id_column} {group_id}.")
    else:
        if u.check_nan_values(time_series_df[value_column]):
            raise ValueError(f"The '{value_column}' column contains NaN values in the middle of the series.")


    logger.info("Generated indicators for column '%s': %s",
                value_column,
                list(value_column_indicators_config.keys()))


    return time_series_df


def add_market_data_dualcolumn_indicators(market_data_df):
    """
    Adds multi-column indicators to market_data_df
    """
    market_data_df = add_mfi_column(market_data_df, price_col='price', volume_col='volume', window=14)
    market_data_df['obv'] = generalized_obv(market_data_df['price'], market_data_df['volume'])

    return market_data_df


# =====================================================================
# Single Series Input Indicators
# =====================================================================

# Price Level Indicators
# ----------------------
def calculate_sma(timeseries: pd.Series, window: int) -> pd.Series:
    """
    Simple Moving Average (SMA): Use SMA to smooth out price data over a set period by calculating
    the average price, helping to identify trends and support/resistance levels.

    The function returns NaN for records where there are fewer data points than the window size.

    Params:
    - timeseries (pd.Series): a series of numbers representing the time series data
    - window (int): the number of periods to use for the simple moving average

    Returns:
    - pd.Series: A series containing the calculated SMA, with NaNs for records where there are fewer than 'window' data points.
    """
    # Apply rolling().mean() with min_periods=window to ensure NaN for fewer than 'window' records
    sma = timeseries.rolling(window=window, min_periods=window).mean()

    return sma


def calculate_ema(timeseries: pd.Series, window: int) -> pd.Series:
    """
    Exponential Moving Average (EMA): EMA gives more weight to recent prices, making it more
    responsive to current price movements and useful for identifying momentum and trends.

    The function returns NaN for records where there are fewer data points than the window size.

    Params:
    - timeseries (pd.Series): a series of numbers representing the time series data
    - window (int): the number of periods to use for the exponential moving average

    Returns:
    - pd.Series: A series containing the calculated EMA, with NaNs for records where there are fewer than 'window' data points.
    """
    # Calculate the EMA using ewm(), but mask the first 'window - 1' values with NaN
    ema = timeseries.ewm(span=window, adjust=False).mean()

    # Set the first 'window - 1' values to NaN
    ema[:window-1] = np.nan

    return ema


def calculate_bollinger_bands(timeseries: pd.Series,
                              return_band: str,
                              window: int,
                              num_std: float = 2) -> pd.Series:
    """
    Bollinger Bands: Bollinger Bands measure volatility by placing bands above and below a moving
    average, indicating overbought or oversold conditions when prices touch the upper or lower bands.

    Params:
    - timeseries (pd.Series): a series of numbers that each represent a time series step
    - return_band (str): which band to return, either 'upper' or 'lower'
    - window (int): the number of periods to use for the moving average and standard deviation
    - num_std (float): the number of standard deviations for the upper and lower bands

    Returns:
    - pd.Series: The selected band (upper or lower) based on the return_band parameter
      Raises a ValueError if return_band is not 'upper' or 'lower'.
    """
    # Validate the return_band parameter
    if return_band not in ['upper', 'lower']:
        raise ValueError("Invalid return_band value. Must be 'upper' or 'lower'.")

    # Calculate the simple moving average (middle band)
    middle_band = timeseries.rolling(window=window).mean()

    # Calculate the standard deviation
    std_dev = timeseries.rolling(window=window).std(ddof=0)

    # Calculate the upper and lower bands
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)

    # Return the requested band
    if return_band == 'upper':
        return upper_band
    if return_band == 'lower':
        return lower_band



# Strength Indicators
# -------------------
def calculate_rsi(timeseries: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculates the RSI (Relative Strength Index) indicator. RSI looks at the average gains
    vs losses over the window duration and calculates the score based on a ratio between them.

    Params:
    - timeseries (pd.Series): a series of numbers that each represent a time series step
    - window (int): how many rows the lookback window should extend
    """
    delta = timeseries.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = np.where(loss != 0, gain / loss, np.inf)
    rsi = np.where(loss == 0, 1, 1 - (1 / (1 + rs)))
    return rsi


# =====================================================================
# Dual Series Input Indicators
# =====================================================================

def add_mfi_column(time_series_df, price_col='price', volume_col='volume', window=14, drop_price=False, drop_volume=False):
    """
    Adds a Money Flow Index (MFI) column to time_series_df based on the price and volume.
    The MFI is calculated over a lookback window and helps assess buying and selling pressure.

    Money Flow Index (MFI): MFI combines price and volume data to assess buying and selling pressure,
    signaling potential overbought (above 80) or oversold (below 20) conditions.

    Parameters:
    -----------
    time_series_df : pd.DataFrame
        The input DataFrame containing the time series data.
    price_col : str, optional (default='price')
        The name of the price column.
    volume_col : str, optional (default='volume')
        The name of the volume column.
    window : int, optional (default=14)
        The lookback window used to calculate the MFI.
    drop_price : bool, optional (default=False)
        If True, the price column will be dropped from the DataFrame after adding the MFI column.
    drop_volume : bool, optional (default=False)
        If True, the volume column will be dropped from the DataFrame after adding the MFI column.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with the new MFI column added and optional columns dropped.
    """
    # Define a function to apply MFI calculation to a group
    def apply_mfi(group):
        group['mfi'] = calculate_mfi(group[price_col], group[volume_col], window=window)
        return group.set_index(['coin_id', 'date'])

    # Apply the MFI calculation across each 'coin_id' group
    time_series_df = time_series_df.groupby('coin_id', group_keys=False, observed=True).apply(apply_mfi)

    # Drop price and volume columns if requested
    if drop_price:
        time_series_df = time_series_df.drop(columns=[price_col])
    if drop_volume:
        time_series_df = time_series_df.drop(columns=[volume_col])

    time_series_df = time_series_df.reset_index()

    return time_series_df

def calculate_mfi(price: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    """
    Money Flow Index (MFI): MFI combines price and volume data to assess buying and selling
    pressure, signaling potential overbought (above 80) or oversold (below 20) conditions.

    Params:
    - price (pd.Series): the close prices for each time step
    - volume (pd.Series): the trading volume for each time step
    - window (int): the lookback window for calculating the MFI (default is 14)

    Returns:
    - pd.Series: a series representing the MFI for each time step
    """

    # Step 1: Use the close price directly as the proxy for the typical price
    typical_price = price

    # Step 2: Calculate the Raw Money Flow (MF = TP * Volume)
    money_flow = typical_price * volume

    # Step 3: Calculate Positive and Negative Money Flow
    positive_money_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_money_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

    # Step 4: Calculate the Money Flow Ratio (MFR) over the window
    money_flow_ratio = positive_money_flow.rolling(window=window).sum() / negative_money_flow.rolling(window=window).sum()

    # Step 5: Calculate the Money Flow Index (MFI)
    mfi = 100 - (100 / (1 + money_flow_ratio))

    # Step 6: Forward fill NaN values, then fill remaining NaNs with 0.5
    mfi = mfi.ffill().fillna(0.5)

    return mfi


def add_crossover_column(time_series_df, col1, col2, drop_col1=False, drop_col2=False):
    """
    Adds a crossover column to time_series_df based on the crossovers between col1 and col2.
    The name of the new column is automatically generated as crossover_{col1}_{col2}. Crossover
    columns can be used as indicators to generate MACD features.

    Moving Average Convergence Divergence (MACD): MACD tracks the difference between two EMAs,
    highlighting momentum shifts and potential trend reversals when it crosses above or below
    its signal line.

    Parameters:
    -----------
    time_series_df : pd.DataFrame
        The input DataFrame containing the time series data.
    col1 : str
        The name of the first column (e.g., 'ema_12').
    col2 : str
        The name of the second column (e.g., 'ema_26').
    drop_col1 : bool, optional (default=False)
        If True, col1 will be dropped from the DataFrame after adding the crossover column.
    drop_col2 : bool, optional (default=False)
        If True, col2 will be dropped from the DataFrame after adding the crossover column.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with the new crossover column added and optional columns dropped.
    """

    crossover_colname = f"crossover_{col1}_{col2}"

    # Define a function that applies identify_crossovers to a group
    def apply_crossovers(group):
        # Reset the index inside the group to ensure proper alignment
        group = group.reset_index()
        group[crossover_colname] = identify_crossovers(group[col1], group[col2])
        return group.set_index(['coin_id', 'date'])  # Set the index back to original

    # Apply the crossover logic across each 'coin_id' group
    time_series_df = time_series_df.groupby('coin_id', group_keys=False, observed=True).apply(apply_crossovers)

    # Drop col1 and col2 if requested
    if drop_col1:
        time_series_df = time_series_df.drop(columns=[col1])
    if drop_col2:
        time_series_df = time_series_df.drop(columns=[col2])

    return time_series_df

def identify_crossovers(series1, series2):
    """
    Identifies crossovers between two time series (series1 and series2).

    - A crossover from below to above (series1 was less than or equal to series2 and
        becomes greater) is marked as 1.
    - A crossover from above to below (series1 was greater than or equal to series2 and
        becomes less) is marked as -1.
    - If there is no crossover or if the values remain equal, the value is 0.

    Parameters:
    -----------
    series1 : pd.Series
        The first time series to compare.
    series2 : pd.Series
        The second time series to compare.

    Returns:
    --------
    pd.Series
        A series indicating crossover points (1, -1, or 0).
    """

    # Calculate the difference between series1 and series2 at each time step
    diff = series1 - series2

    # Use np.where to detect crossover conditions
    # First condition: Check if series1 was less than or equal to series2 (including ties) and now crosses above
    # Second condition: Check if series1 was greater than or equal to series2 and now crosses below
    # Otherwise, return 0 (no crossover)
    crossovers = np.where(
        (diff.shift(1) <= 0) & (diff > 0),  # Crossover from below or tie to above
        1,  # Assign 1 for crossover upward
        np.where((diff.shift(1) >= 0) & (diff < 0),  # Crossover from above or tie to below
                 -1,  # Assign -1 for crossover downward
                 0)  # No crossover, remain 0
    )

    # Return the result as a Pandas Series
    return pd.Series(crossovers)



def generalized_obv(primary_series, secondary_series):
    """
    On-Balance Volume (OBV): OBV uses volume changes to predict price movements by showing whether
    volume is flowing into or out of an asset, indicating potential buying or selling pressure.

    - If the primary series increases, the secondary series is added.
    - If the primary series decreases, the secondary series is subtracted.
    - If the primary series remains unchanged, the secondary series is ignored.

    Parameters:
    -----------
    primary_series : pd.Series
        The time series that dictates whether the secondary series is added or subtracted.
    secondary_series : pd.Series
        The time series that is accumulated or decremented based on the primary series changes.

    Returns:
    --------
    pd.Series
        A cumulative series representing the generalized OBV-like metric.
    """

    # Calculate changes in the primary series
    primary_diff = primary_series.diff()

    # Define changes to the secondary series based on the primary series movements
    obv_changes = np.where(primary_diff > 0, secondary_series,  # If primary increases, add secondary
                           np.where(primary_diff < 0, -secondary_series, 0))  # If primary decreases, subtract secondary

    # Calculate the cumulative OBV
    obv = np.cumsum(obv_changes)

    # Return the cumulative series as a Pandas Series
    return pd.Series(obv, index=primary_series.index)
