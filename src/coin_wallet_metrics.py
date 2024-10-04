'''
calculates metrics related to the distribution of coin ownership across wallets
'''
# pylint: disable=C0301 # line too long

import time
from typing import Tuple,Optional
import pandas as pd
import numpy as np
import dreams_core.core as dc

# set up logger at the module level
logger = dc.setup_logger()



def classify_wallet_cohort(profits_df, wallet_cohort_config, cohort_name):
    """
    Classifies wallets into a cohort based on their activity across multiple coins.
    The function directly applies the inflows, profitability, and return thresholds at the wallet level.
    Outputs a DataFrame with aggregated wallet-level metrics and cohort classification.

    Parameters:
        profits_df (DataFrame): A DataFrame containing wallet-coin records with profits and returns data.
        wallet_cohort_config (dict): A configuration object containing cohort parameters, including:
            - 'wallet_minimum_inflows': Minimum total USD inflow to be eligible for the cohort.
            - 'wallet_maximum_inflows': Maximum total USD inflow to be eligible for the cohort.
            - 'coin_profits_win_threshold': Minimum total profits for a coin to be considered a "win".
            - 'coin_return_win_threshold': Minimum total return for a coin to be considered a "win".
            - 'wallet_min_coin_wins': Minimum number of coins that meet the "win" threshold for a wallet to join the cohort.
        cohort_name (string): The name of the cohort which is used for logging purposes

    Returns:
        wallet_cohort_df (DataFrame): A DataFrame containing wallets and summary metrics, including:
            - total inflows, total coins, total wins, win rate, and whether the wallet is in the cohort.
    """
    logger.debug("Classifying wallet cohort '%s' based on coin-level thresholds...", cohort_name)
    start_time = time.time()

    # Step 1: Aggregate wallet-level inflows and filter eligible wallets
    wallet_inflows_df = profits_df.groupby('wallet_address', observed=True)['usd_inflows'].sum().reset_index()
    eligible_wallets_df = wallet_inflows_df[
        (wallet_inflows_df['usd_inflows'] >= wallet_cohort_config['wallet_minimum_inflows']) &
        (wallet_inflows_df['usd_inflows'] <= wallet_cohort_config['wallet_maximum_inflows'])
    ]
    logger.debug("<Step 1> Aggregate wallet-level inflows and filter eligible wallets: %.2f seconds", time.time() - start_time)
    step_time = time.time()


    # Step 2: Group by wallet and coin to aggregate profits and return
    # filter profits_df for only eligible wallets
    eligible_wallets_profits_df = profits_df[profits_df['wallet_address'].isin(eligible_wallets_df['wallet_address'])].copy()
    eligible_wallets_profits_df = eligible_wallets_profits_df.sort_values(by=['wallet_address', 'coin_id', 'date'])

    # compute wallet-coin level metrics
    eligible_wallets_coins_df = eligible_wallets_profits_df.groupby(['wallet_address', 'coin_id'], observed=True).agg({
        'profits_cumulative': 'last',  # Use the last record for cumulative profits
        'total_return': 'last'  # Use the last record for total return
    }).reset_index()

    logger.debug("<Step 2> Group by wallet and coin to aggregate profits and return: %.2f seconds", time.time() - step_time)
    step_time = time.time()


    # Step 3: Apply wallet-coin-level thresholds (profits AND return) to each wallet and classify "win"s
    eligible_wallets_coins_df['is_profits_win'] = eligible_wallets_coins_df['profits_cumulative'] >= wallet_cohort_config['coin_profits_win_threshold']
    eligible_wallets_coins_df['is_returns_win'] = eligible_wallets_coins_df['total_return'] >= wallet_cohort_config['coin_return_win_threshold']

    # A coin is classified as a "win" if it meets both the profits AND returns threshold
    eligible_wallets_coins_df['is_coin_win'] = eligible_wallets_coins_df['is_profits_win'] & eligible_wallets_coins_df['is_returns_win']

    logger.debug("<Step 3> Classify coin wins: %.2f seconds", time.time() - step_time)
    step_time = time.time()


    # Step 4: Aggregate at the wallet level to count the number of "winning" coins and total coins
    wallet_wins_df = eligible_wallets_coins_df.groupby('wallet_address', observed=True)['is_coin_win'].sum().reset_index()
    wallet_wins_df.columns = ['wallet_address', 'winning_coins']

    wallet_coins_df = eligible_wallets_coins_df.groupby('wallet_address', observed=True)['coin_id'].nunique().reset_index()
    wallet_coins_df.columns = ['wallet_address', 'total_coins']

    logger.debug("<Step 4> Aggregate wins to wallet level: %.2f seconds", time.time() - step_time)
    step_time = time.time()


    # Step 5: Classify wallets into the cohort based on minimum number of coin wins
    wallet_wins_df['in_cohort'] = wallet_wins_df['winning_coins'] >= wallet_cohort_config['wallet_min_coin_wins']

    logger.debug("<Step 5> Classify cohorts: %.2f seconds", time.time() - step_time)
    step_time = time.time()


    # Step 6: Compute wallet-level metrics for output
    # Merge wallet inflows, wins, total coins, profits, and return rate
    wallet_cohort_df = wallet_inflows_df.merge(wallet_coins_df, on='wallet_address')
    wallet_cohort_df = wallet_cohort_df.merge(wallet_wins_df, on='wallet_address', how='left')
    wallet_cohort_df['winning_coins'] = wallet_cohort_df['winning_coins'].fillna(0)

    logger.debug("<Step 6a> Merge wallet inflows and coins: %.2f seconds", time.time() - step_time)
    step_time = time.time()

    # Calculate total profits (USD value)
    # Calculate profits for each coin that each wallet owns
    wallet_coin_profits_df = (profits_df.groupby(['wallet_address','coin_id'], observed=True)
                                   ['profits_cumulative'].last()
                                   .reset_index())
    # Sum each wallet's coin profits to get their total profits
    wallet_profits_df = (wallet_coin_profits_df.groupby('wallet_address')
                                               ['profits_cumulative'].sum()
                                               .reset_index())
    wallet_profits_df.columns = ['wallet_address', 'total_profits']

    # Calculate return rate: total profits / total inflows
    wallet_return_rate_df = wallet_profits_df.merge(wallet_inflows_df, on='wallet_address')
    wallet_return_rate_df['return_rate'] = wallet_return_rate_df['total_profits'] / wallet_return_rate_df['usd_inflows']

    # Merge profits and return rate into the cohort summary
    wallet_cohort_df = wallet_cohort_df.merge(wallet_profits_df, on='wallet_address', how='left')
    wallet_cohort_df = wallet_cohort_df.merge(wallet_return_rate_df[['wallet_address', 'return_rate']], on='wallet_address', how='left')

    logger.debug("<Step 6a> Merge profits and return rate: %.2f seconds", time.time() - step_time)
    step_time = time.time()

    # Log the count of wallets added to the cohort using % syntax
    logger.info("Wallet cohort '%s' classification complete after %.2f seconds.",
        cohort_name,
        time.time() - start_time
    )
    logger.info("Out of %s total wallets, %s met inflows requirements and %s met win rate conditions.",
        dc.human_format(len(wallet_inflows_df)),
        dc.human_format(len(wallet_cohort_df)),
        dc.human_format(len(wallet_cohort_df[wallet_cohort_df['in_cohort']]))
    )


    return wallet_cohort_df



def generate_buysell_metrics_df(profits_df,training_period_end,cohort_wallets):
    """
    Generates buysell metrics for all cohort coins by looping through each coin_id and applying
    calculate_buysell_coin_metrics_df().

    Parameters:
    - profits_df (pd.DataFrame): DataFrame containing profits data
    - training_period_end (string): Date on which the final metrics will be generated, formatted YYYY-MM-DD
    - cohort_wallets (array-like): List of wallet addresses to include.

    Returns:
    - buysell_metrics_df (pd.DataFrame): DataFrame keyed date-coin_id that includes various metrics
        about the cohort's buying, selling, and holding behavior. This date is filled to have a row for
        every coin_id-date pair through the training_period_end.
    """
    start_time = time.time()
    logger.debug('Preparing buysell_metrics_df...')


    # Step 1: Filter profits_df to cohort and conduct data quality checks
    # -------------------------------------------------------------------
    # Raise an error if either the wallet cohort or coin list is empty
    if len(cohort_wallets) == 0:
        raise ValueError("Wallet cohort is empty. Provide at least one wallet address.")

    # Create cohort_profits_df by filtering profits_df to only include the cohort coins and wallets
    # during the training period
    profits_df = profits_df[profits_df['date']<=training_period_end]
    cohort_profits_df = profits_df[profits_df['wallet_address'].isin(cohort_wallets)]

    cohort_profits_df = cohort_profits_df[['coin_id','wallet_address','date','usd_balance','usd_net_transfers']]

    # Raise an error if the filtered df is empty
    if cohort_profits_df.empty:
        raise ValueError("Cohort-filtered profits_df is empty. Please check input parameters")


    # Step 2: Add buy_sequence and sell_sequence columns
    # --------------------------------------------------
    # Initialize the buy and sell sequence columns
    cohort_profits_df['buy_sequence'] = np.where(cohort_profits_df['usd_net_transfers'] > 0, 1, np.nan)
    cohort_profits_df['sell_sequence'] = np.where(cohort_profits_df['usd_net_transfers'] < 0, 1, np.nan)

    # Calculate cumulative sum to simulate transfer sequence, skipping rows where usd_net_transfers == 0
    cohort_profits_df['buy_sequence'] = cohort_profits_df.groupby(['coin_id', 'wallet_address'], observed=True)['buy_sequence'].cumsum()
    cohort_profits_df['sell_sequence'] = cohort_profits_df.groupby(['coin_id', 'wallet_address'], observed=True)['sell_sequence'].cumsum()

    # Set buy_sequence and sell_sequence to null where usd_net_transfers == 0
    cohort_profits_df.loc[cohort_profits_df['usd_net_transfers'] == 0, ['buy_sequence', 'sell_sequence']] = np.nan


    # Step 3: Calculate coin metrics
    # ------------------------------
    # Initialize an empty list to store DataFrames for each coin
    coin_features_list = []

    # Loop through all unique coin_ids
    for c in cohort_profits_df['coin_id'].unique():
        # Filter cohort_profits_df for the current coin_id and create a copy
        coin_cohort_profits_df = cohort_profits_df[cohort_profits_df['coin_id'] == c].copy()

        # Call the feature calculation function
        coin_features_df = generate_coin_buysell_metrics_df(coin_cohort_profits_df)

        # Add coin_id back to the DataFrame to retain coin information
        coin_features_df['coin_id'] = c

        # Append the result to the list
        coin_features_list.append(coin_features_df)


    # Step 4: Consolidate all metrics into a filled DataFrame
    # -------------------------------------------------------
    # Concatenate all features DataFrames into a single DataFrame
    buysell_metrics_df = pd.concat(coin_features_list, ignore_index=True)

    # Ensure full date range coverage through the training_period_end for each coin-wallet pair
    buysell_metrics_df = fill_buysell_metrics_df(buysell_metrics_df, training_period_end)

    logger.info('Generated buysell_metrics_df after %.2f seconds.', time.time() - start_time)

    return buysell_metrics_df



def generate_coin_buysell_metrics_df(coin_cohort_profits_df):
    '''
    For a single coin_id, computes various buyer and seller metrics, including:
    - Number of new and repeat buyers/sellers
    - Total buyers/sellers (new + repeat)
    - Buyers-to-sellers ratio and new buyers-to-new sellers ratio
    - New vs. repeat buyer/seller ratios
    - Total bought/sold amounts
    - Market sentiment score

    params:
    - coin_cohort_profits_df (dataframe): df showing profits data for a single coin_id

    returns:
    - buysell_metrics_df (dataframe): df of metrics on new/repeat buyers/sellers and transaction totals on each date
    '''
    # Ensure 'date' column is of datetime type
    coin_cohort_profits_df['date'] = pd.to_datetime(coin_cohort_profits_df['date'])

    # Calculate buyer counts
    buyers_df = coin_cohort_profits_df.groupby('date').agg(
        buyers_new=('buy_sequence', lambda x: (x == 1).sum()),
        buyers_repeat=('buy_sequence', lambda x: (x > 1).sum())
    ).reset_index()

    # Calculate total buyers
    buyers_df['total_buyers'] = buyers_df['buyers_new'] + buyers_df['buyers_repeat']

    # Calculate seller counts
    sellers_df = coin_cohort_profits_df.groupby('date').agg(
        sellers_new=('sell_sequence', lambda x: (x == 1).sum()),
        sellers_repeat=('sell_sequence', lambda x: (x > 1).sum())
    ).reset_index()

    # Calculate total sellers
    sellers_df['total_sellers'] = sellers_df['sellers_new'] + sellers_df['sellers_repeat']

    # Calculate total bought, total sold, total net transfers, and total volume
    transactions_df = coin_cohort_profits_df.groupby('date').agg(
        total_bought=('usd_net_transfers', lambda x: x[x > 0].sum()),  # Sum of positive net transfers (buys)
        total_sold=('usd_net_transfers', lambda x: abs(x[x < 0].sum())),  # Sum of negative net transfers (sells as positive)
        total_net_transfers=('usd_net_transfers', 'sum'),  # Net of all transfers
        total_volume=('usd_net_transfers', lambda x: x[x > 0].sum() + abs(x[x < 0].sum()))  # Total volume: buys + sells
    ).reset_index()

    # Calculate total holders and total balance
    holders_df = coin_cohort_profits_df.groupby('date').agg(
        total_holders=('wallet_address', 'nunique'),  # Number of unique holders
        total_balance=('usd_balance', 'sum')  # Sum of balances for all wallets
    ).reset_index()

    # Merge buyers, sellers, transactions, and holders dataframes
    buysell_metrics_df = pd.merge(buyers_df, sellers_df, on='date', how='outer')
    buysell_metrics_df = pd.merge(buysell_metrics_df, transactions_df, on='date', how='outer')
    buysell_metrics_df = pd.merge(buysell_metrics_df, holders_df, on='date', how='outer')

    return buysell_metrics_df



def fill_buysell_metrics_df(buysell_metrics_df, training_period_end):
    """
    Fills missing dates in buysell_metrics_df and applies appropriate logic to fill NaN values
    for each metric.

    This function:
    - Adds rows with missing dates (if any) between the latest date in buysell_metrics_df and the
        training_period_end.
    - Fills NaN values for buy/sell metrics, balances, and other key metrics according to the
        following rules:
        - total_balance and total_holders:
                forward-filled (if no activity, assume balances/holders remain the same).
        - total_bought, total_sold, total_net_transfers, total_volume, buyers_new,
          buyers_repeat, sellers_new, sellers_repeat:
                filled with 0 (if no activity, assume no transactions).

    Parameters:
    - buysell_metrics_df: DataFrame containing buy/sell metrics keyed on coin_id-date
    - training_period_end: The end of the training period (datetime)

    Returns:
    - buysell_metrics_df with missing dates and NaN values appropriately filled.
    """

    # Identify the expected rows for all coin_id-date pairs
    min_date = buysell_metrics_df['date'].min()
    full_date_range = pd.date_range(start=min_date, end=training_period_end, freq='D')
    all_combinations = pd.MultiIndex.from_product(
        [buysell_metrics_df['coin_id'].unique(), full_date_range], names=['coin_id', 'date']
    )

    # Identify missing rows by comparing the existing rows to the expected
    existing_combinations = pd.MultiIndex.from_frame(buysell_metrics_df[['coin_id', 'date']])
    missing_combinations = all_combinations.difference(existing_combinations)
    missing_rows = pd.DataFrame(index=missing_combinations).reset_index()

    # Add the missing rows to buysell_metrics_df and sort by coin_id,date to ensure correct fill logic
    buysell_metrics_df = pd.concat([buysell_metrics_df, missing_rows], ignore_index=True)
    buysell_metrics_df = buysell_metrics_df.sort_values(by=['coin_id', 'date']).reset_index(drop=True)


    # Apply the appropriate fill logic per metric, with groupby to ensure correct forward-filling within each coin_id:
    buysell_metrics_df = buysell_metrics_df.groupby('coin_id', group_keys=False).apply(
        lambda group: group.assign(
            # Preserve coin_id as an output column
            coin_id=buysell_metrics_df['coin_id'],

            # Forward-fill for balance and holders (each coin_id has independent forward-filling logic)
            # After the forward-fill, fill 0 for dates earlier than the first balance/holders records
            total_balance=buysell_metrics_df.groupby('coin_id')['total_balance'].ffill().fillna(0),
            total_holders=buysell_metrics_df.groupby('coin_id')['total_holders'].ffill().fillna(0),

            # Fill 0 for metrics related to transactions (buying, selling) that should be 0 when there's no activity
            total_bought=buysell_metrics_df['total_bought'].fillna(0),
            total_sold=buysell_metrics_df['total_sold'].fillna(0),
            total_net_transfers=buysell_metrics_df['total_net_transfers'].fillna(0),
            total_volume=buysell_metrics_df['total_volume'].fillna(0),

            # Fill 0 for buyer/seller counts on days with no transactions
            buyers_new=buysell_metrics_df['buyers_new'].fillna(0),
            buyers_repeat=buysell_metrics_df['buyers_repeat'].fillna(0),
            total_buyers=buysell_metrics_df['total_buyers'].fillna(0),
            sellers_new=buysell_metrics_df['sellers_new'].fillna(0),
            sellers_repeat=buysell_metrics_df['sellers_repeat'].fillna(0),
            total_sellers=buysell_metrics_df['total_sellers'].fillna(0)
        )
        ,include_groups=False  # Exclude the grouping columns (coin_id) from the operation
    ).reset_index(drop=True)

    return buysell_metrics_df



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


    # 3. Split records by complete vs partial time series coverage
    # ------------------------------------------------------------
    full_indicators_df, partial_time_series_indicators_df = split_dataframe_by_coverage(
        time_series_df, training_period_start, training_period_end, id_column
    )

    # Logging
    logger.debug("Generated time series indicators data.")

    return full_indicators_df, partial_time_series_indicators_df


def split_dataframe_by_coverage(
        time_series_df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        id_column: Optional[str] = 'coin_id'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the input DataFrame into full coverage and partial coverage based on date range.
    Works for both multi-series and single-series datasets.

    Params:
    - time_series_df (pd.DataFrame): The input DataFrame with time series data.
    - start_date (pd.Timestamp): Start date of the training period.
    - end_date (pd.Timestamp): End date of the modeling period.
    - id_column (Optional[str]): The name of the column used to identify different series.

    Returns:
    - full_coverage_df (pd.DataFrame): DataFrame with series having complete data for the period.
    - partial_coverage_df (pd.DataFrame): DataFrame with series having partial data for the period.
    """
    # Convert params to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Define a function to check if a date range has full coverage
    def has_full_coverage(min_date, max_date):
        return (min_date <= start_date) and (max_date >= end_date)

    if id_column:
        # Multi-series data
        series_data_range = time_series_df.groupby(id_column, observed=True)['date'].agg(['min', 'max'])
        full_duration_series = series_data_range[series_data_range.apply(lambda x: has_full_coverage(x['min'], x['max']), axis=1)].index
    else:
        # Single-series data
        series_data_range = time_series_df['date'].agg(['min', 'max'])
        full_duration_series = [0] if has_full_coverage(series_data_range['min'], series_data_range['max']) else []

    # Calculate coverage statistics
    full_coverage_count = len(full_duration_series)

    # Split the dataframe
    if id_column:
        # Convert id column to categorical to reduce memory usage
        time_series_df[id_column] = time_series_df[id_column].astype('category')
        full_coverage_df = time_series_df[time_series_df[id_column].isin(full_duration_series)]
        partial_coverage_df = time_series_df[~time_series_df[id_column].isin(full_duration_series)]
    else:
        full_coverage_df = time_series_df if full_coverage_count else pd.DataFrame(columns=time_series_df.columns)
        partial_coverage_df = time_series_df if not full_coverage_count else pd.DataFrame(columns=time_series_df.columns)

    logger.info("Split df with dimensions %s into %s full coverage records and %s partial coverage records.",
                time_series_df.shape,
                dc.human_format(len(full_coverage_df)),
                dc.human_format(len(partial_coverage_df))
    )

    return full_coverage_df, partial_coverage_df



def calculate_sma(timeseries: pd.Series, period: int) -> pd.Series:
    """
    Calculates Simple Moving Average (SMA) for a given time series, with handling for imputing
    null values on dates that occur when there are fewer data points than the period duration.
    """
    # Calculate the SMA for the first few values where data is less than the period
    sma = timeseries.expanding(min_periods=1).apply(lambda x: x.mean() if len(x) < period else np.nan)

    # Use rolling().mean() for the rest once the period is reached
    rolling_sma = timeseries.rolling(window=period, min_periods=period).mean()

    # Combine the two: use the expanding calculation until the period is reached, then use rolling()
    sma = sma.combine_first(rolling_sma)

    return sma

def calculate_ema(timeseries: pd.Series, period: int) -> pd.Series:
    """Calculates Exponential Moving Average (EMA) for a given time series."""
    return timeseries.ewm(span=period, adjust=False).mean()
