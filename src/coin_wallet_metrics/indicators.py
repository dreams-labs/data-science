'''
calculates metrics related to the distribution of coin ownership across wallets
'''
# pylint: disable=C0301 # line too long

from typing import Tuple,Optional
import pandas as pd
import numpy as np
import dreams_core.core as dc

# import coin_wallet_metrics as cwm
# sys.path.append('..')
# from training_data import data_retrieval as dr
# from training_data import profits_row_imputation as ri


# set up logger at the module level
logger = dc.setup_logger()

def generate_time_series_indicators(
        time_series_df: pd.DataFrame,
        config: dict,
        value_column_indicators_config: dict,
        value_column: str,
        id_column: Optional[str]='coin_id'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates time series metrics (e.g., SMA, EMA) based on the given config.
    Works for both multi-series (e.g., multiple coins) and single time series data.

    Params:
    - time_series_df (pd.DataFrame): The input DataFrame with time series data.
    - config: The full general config file containing training_period_start and
        training_period_end.
    - value_column_indicators_config: The metrics_config subcomponent with the parameters for the
        value_column, e.g. metrics_config['time_series']['prices']['price']['indicators']
    - value_column (string): The column used to calculate the indicators (e.g., 'price').
    - id_column (Optional[string]): The name of the column used to identify different series
        (e.g., 'coin_id'). If None, assumes a single time series.

    Returns:
    - full_indicators_df (pd.DataFrame): Input df with additional columns for the specified
        indicators. Only includes series that had complete data for the period between
        training_period_start and training_period_end.
    - partial_time_series_indicators_df (pd.DataFrame): Input df with additional columns for the
        configured indicators. Only includes series that had partial data for the period.
    """
    # 1. Data Quality Checks and Formatting
    # -------------------------------------
    if value_column not in time_series_df.columns:
        raise KeyError(f"Input DataFrame does not include column '{value_column}'.")

    if time_series_df[value_column].isnull().any():
        raise ValueError(f"The '{value_column}' column contains null values, which are not allowed.")

    time_series_df = time_series_df.copy()
    time_series_df['date'] = pd.to_datetime(time_series_df['date'])
    training_period_start = pd.to_datetime(config['training_data']['training_period_start'])
    training_period_end = pd.to_datetime(config['training_data']['training_period_end'])

    time_series_df = time_series_df[(time_series_df['date'] >= training_period_start) &
                                    (time_series_df['date'] <= training_period_end)]

    # 2. Indicator Calculations
    # ----------------------
    if id_column:
        # Multi-series data (e.g., multiple coins)
        time_series_df = time_series_df.sort_values(by=[id_column, 'date'])
        groupby_column = id_column
    else:
        # Single time series data
        time_series_df = time_series_df.sort_values(by=['date'])
        groupby_column = lambda x: True # Group all rows on dummy column    # pylint: disable=C3001

    for _, group in time_series_df.groupby(groupby_column):
        for metric, config in value_column_indicators_config.items():
            period = config['parameters']['period']

            if metric == 'sma':
                sma = calculate_sma(group[value_column], period)
                time_series_df.loc[group.index, f"{value_column}_{metric}"] = sma
            elif metric == 'ema':
                ema = calculate_ema(group[value_column], period)
                time_series_df.loc[group.index, f"{value_column}_{metric}"] = ema

    # # Logging
    # logger.debug("Generated time series indicators data.")

    # return full_indicators_df, partial_time_series_indicators_df



# =====================================================================
# Single Series Input Indicators
# =====================================================================

# Price Level Indicators
# ----------------------
def calculate_sma(timeseries: pd.Series, window: int) -> pd.Series:
    """
    Simple Moving Average (SMA): Use SMA to smooth out price data over a set period by calculating
    the average price, helping to identify trends and support/resistance levels.
    """
    # Calculate the SMA for the first few values where data is less than the period
    sma = timeseries.expanding(min_periods=1).apply(lambda x: x.mean() if len(x) < window else np.nan)

    # Use rolling().mean() for the rest once the period is reached
    rolling_sma = timeseries.rolling(window=window, min_periods=window).mean()

    # Combine the two: use the expanding calculation until the period is reached, then use rolling()
    sma = sma.combine_first(rolling_sma)

    return sma

def calculate_ema(timeseries: pd.Series, window: int) -> pd.Series:
    """
    Exponential Moving Average (EMA): EMA gives more weight to recent prices, making it more
    responsive to current price movements and useful for identifying momentum and trends.
    """
    return timeseries.ewm(span=window, adjust=False).mean()



def add_bollinger_bands(time_series_df, price_col='price', window=20, num_std=2, include_middle=False):
    """
    Adds Bollinger Bands (middle, upper, lower) to time_series_df based on the specified price column.
    Bollinger Bands measure volatility by placing bands above and below a moving average, indicating
    overbought or oversold conditions when prices touch the upper or lower bands.

    Parameters:
    -----------
    time_series_df : pd.DataFrame
        The input DataFrame containing the time series data.
    price_col : str, optional (default='price')
        The name of the price column used for calculating Bollinger Bands.
    window : int, optional (default=20)
        The number of periods to use for the moving average and standard deviation.
    num_std : float, optional (default=2)
        The number of standard deviations for the upper and lower bands.
    include_middle : bool, optional (default=False)
        If True, the bolinger_band_middle will be returned. If False, it will be dropped.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with the new Bollinger Band columns added and the optional price column dropped.
    """

    # Define a function to apply Bollinger Bands calculation to a group
    def apply_bollinger_bands(group):
        group = group.reset_index()

        # Calculate Bollinger Bands
        middle_band, upper_band, lower_band = calculate_bollinger_bands(group[price_col], window=window, num_std=num_std)

        # Add the bands as new columns
        group['bollinger_band_middle'] = middle_band
        group['bollinger_band_upper'] = upper_band
        group['bollinger_band_lower'] = lower_band

        return group.set_index(['coin_id', 'date'])

    # Apply the Bollinger Bands calculation across each 'coin_id' group
    time_series_df = time_series_df.groupby('coin_id', group_keys=False, observed=True).apply(apply_bollinger_bands)

    # Drop middle band column if requested
    if not include_middle:
        time_series_df = time_series_df.drop(columns='bollinger_band_middle')

    return time_series_df

def calculate_bollinger_bands(timeseries: pd.Series,
                              window: int = 20,
                              num_std: float = 2) -> tuple:
    """
    Bollinger Bands: Bollinger Bands measure volatility by placing bands above and below a moving
    average, indicating overbought or oversold conditions when prices touch the upper or lower bands.

    Params:
    - timeseries (pd.Series): a series of numbers that each represent a time series step
    - window (int): the number of periods to use for the moving average and standard deviation
    - num_std (float): the number of standard deviations for the upper and lower bands

    Returns:
    - tuple: A tuple containing three Series (middle_band, upper_band, lower_band)
    """
    # Calculate the simple moving average (middle band)
    middle_band = timeseries.rolling(window=window).mean()

    # Calculate the standard deviation
    std_dev = timeseries.rolling(window=window).std()

    # Calculate the upper and lower bands
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)

    return middle_band, upper_band, lower_band


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

    rs = gain / loss
    rsi = 1 - (1 / (1 + rs))
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
        group = group.reset_index()
        group['mfi'] = calculate_mfi(group[price_col], group[volume_col], window=window)
        return group.set_index(['coin_id', 'date'])

    # Apply the MFI calculation across each 'coin_id' group
    time_series_df = time_series_df.groupby('coin_id', group_keys=False, observed=True).apply(apply_mfi)

    # Drop price and volume columns if requested
    if drop_price:
        time_series_df = time_series_df.drop(columns=[price_col])
    if drop_volume:
        time_series_df = time_series_df.drop(columns=[volume_col])

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
