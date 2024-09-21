'''
calculates metrics related to the distribution of coin ownership across wallets
'''
# pylint: disable=C0301 # line too long
# pylint: disable=C0303 # trailing whitespace

import time
import pandas as pd
import numpy as np
import dreams_core.core as dc

# set up logger at the module level
logger = dc.setup_logger()


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
    logger.info('Preparing buysell_metrics_df...')


    # Step 1: Filter profits_df to cohort and conduct data quality checks
    # -------------------------------------------------------------------
    # Raise an error if either the wallet cohort or coin list is empty
    if len(cohort_wallets) == 0:
        raise ValueError("Wallet cohort is empty. Provide at least one wallet address.")

    # Create cohort_profits_df by filtering projects_df to only include the cohort coins and wallets during the training period
    profits_df = profits_df[profits_df['date']<=training_period_end]
    cohort_profits_df = profits_df[profits_df['wallet_address'].isin(cohort_wallets)]
    cohort_profits_df = cohort_profits_df[['coin_id','wallet_address','date','balance','net_transfers']]

    # Raise an error if the filtered df is empty
    if cohort_profits_df.empty:
        raise ValueError("Cohort-filtered profits_df is empty. Please check input parameters")


    # Step 2: Add buy_sequence and sell_sequence columns
    # --------------------------------------------------
    # Initialize the buy and sell sequence columns
    cohort_profits_df['buy_sequence'] = np.where(cohort_profits_df['net_transfers'] > 0, 1, np.nan)
    cohort_profits_df['sell_sequence'] = np.where(cohort_profits_df['net_transfers'] < 0, 1, np.nan)

    # Calculate cumulative sum to simulate transfer sequence, skipping rows where net_transfers == 0
    cohort_profits_df['buy_sequence'] = cohort_profits_df.groupby(['coin_id', 'wallet_address'])['buy_sequence'].cumsum()
    cohort_profits_df['sell_sequence'] = cohort_profits_df.groupby(['coin_id', 'wallet_address'])['sell_sequence'].cumsum()

    # Set buy_sequence and sell_sequence to null where net_transfers == 0
    cohort_profits_df.loc[cohort_profits_df['net_transfers'] == 0, ['buy_sequence', 'sell_sequence']] = np.nan


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
    start_time = time.time()

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
        total_bought=('net_transfers', lambda x: x[x > 0].sum()),  # Sum of positive net transfers (buys)
        total_sold=('net_transfers', lambda x: abs(x[x < 0].sum())),  # Sum of negative net transfers (sells as positive)
        total_net_transfers=('net_transfers', 'sum'),  # Net of all transfers
        total_volume=('net_transfers', lambda x: x[x > 0].sum() + abs(x[x < 0].sum()))  # Total volume: buys + sells
    ).reset_index()

    # Calculate total holders and total balance
    holders_df = coin_cohort_profits_df.groupby('date').agg(
        total_holders=('wallet_address', 'nunique'),  # Number of unique holders
        total_balance=('balance', 'sum')  # Sum of balances for all wallets
    ).reset_index()

    # Merge buyers, sellers, transactions, and holders dataframes
    buysell_metrics_df = pd.merge(buyers_df, sellers_df, on='date', how='outer')
    buysell_metrics_df = pd.merge(buysell_metrics_df, transactions_df, on='date', how='outer')
    buysell_metrics_df = pd.merge(buysell_metrics_df, holders_df, on='date', how='outer')

    logger.debug('New vs repeat buyer/seller counts, transaction totals, and holder metrics complete after %.2f seconds', time.time() - start_time)

    return buysell_metrics_df



def fill_buysell_metrics_df(buysell_metrics_df, training_period_end):
    """
    Fills missing dates in buysell_metrics_df and applies appropriate logic to fill NaN values for each metric.

    This function:
    - Adds rows with missing dates (if any) between the latest date in buysell_metrics_df and the training_period_end.
    - Fills NaN values for buy/sell metrics, balances, and other key metrics according to the following rules:
      - total_balance and total_holders: forward-filled (if no activity, assume balances/holders remain the same).
      - total_bought, total_sold, total_net_transfers, total_volume, buyers_new, buyers_repeat, sellers_new, sellers_repeat: filled with 0 (if no activity, assume no transactions).

    Parameters:
    - buysell_metrics_df: DataFrame containing buy/sell metrics keyed on coin_id-date
    - training_period_end: The end of the training period (datetime)

    Returns:
    - buysell_metrics_df with missing dates and NaN values appropriately filled.
    """

    # Identify the expected rows for all coin_id-date pairs
    min_date = buysell_metrics_df['date'].min()
    full_date_range = pd.date_range(start=min_date, end=training_period_end, freq='D')
    all_combinations = pd.MultiIndex.from_product([buysell_metrics_df['coin_id'].unique(), full_date_range], names=['coin_id', 'date'])

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



def generate_time_series_metrics(
        time_series_df: pd.DataFrame,
        metrics_config: dict,
        dataset_key: str,
        colname: str
    ) -> pd.DataFrame:
    """
    Generates time series metrics (e.g., SMA, EMA) based on the given config.

    Params:
    - time_series_df (pd.DataFrame): The input DataFrame with time series data.
    - metrics_config: The full metrics_config file with a time_series key that matches the dataset_key param.
    - dataset_key (string): The dataset's key in the metrics_config['time_series'] section.
    - colname (string): The name of the column that the metrics should be calculated for (e.g., 'price').

    Returns:
    - time_series_metrics_df (pd.DataFrame): The input DataFrame with the configured metrics added as new columns.
    """
    # 1. Data Quality Checks and Formatting
    # -------------------------------------
    # Confirm that the colname is in the input df
    if not colname in time_series_df.columns:
        raise KeyError(f"Input DataFrame does not include column '{colname}'.")

    # Confirm there are no null values in the input column
    if time_series_df[colname].isnull().any():
        raise ValueError(f"The '{colname}' column contains null values, which are not allowed.")

    # Retrieve relevant metrics configuration and raise error if the key doesn't exist
    try:
        time_series_metrics_config = metrics_config['time_series'][dataset_key]
    except KeyError as exc:
        raise KeyError(f"Key [{dataset_key}] not found in metrics_config['time_series']") from exc

    # Raise error if there are no metrics for the key
    if not metrics_config['time_series'][dataset_key]:
        raise KeyError(f"No metrics are specified for key [{dataset_key}] in metrics_df. ")

    # Ensure date is in datetime format and sorted by coin_id and date
    time_series_df['date'] = pd.to_datetime(time_series_df['date'])
    time_series_df = time_series_df.sort_values(by=['coin_id', 'date'])


    # 2. Metric Calculations
    # -------------------------------------
    # Loop over each coin_id group
    for _, group in time_series_df.groupby('coin_id'):
        for metric, config in time_series_metrics_config.items():
            period = config['parameters']['period']

            # Create the dynamic column name, e.g., 'prices_sma_20'
            metric_colname = f'{dataset_key}_{metric}_{period}'

            # Calculate the corresponding metric
            if metric == 'sma':
                sma = calculate_sma(group[colname], period)
                time_series_df.loc[group.index, metric_colname] = sma

            elif metric == 'ema':
                ema = calculate_ema(group[colname], period)
                time_series_df.loc[group.index, metric_colname] = ema

    # Include all dynamically created columns in the return DataFrame
    dynamic_cols = [col for col in time_series_df.columns if col.startswith(f'{dataset_key}_')]
    time_series_metrics_df = time_series_df[['coin_id', 'date', colname] + dynamic_cols]

    return time_series_metrics_df

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



# def prepare_cohort_profits_df(profits_df,cohort_wallets,cohort_coins):
#     '''
#     Prepare a simplified version of profits_df for analysis based on the given wallet cohort and coin list.

#     runs two bigquery queries to retrieve the dfs necessary for wallet metric calculation.
#     note that the all_balances_df is very large as it contains all transfer-days for all coins,
#     which is why metadata is stored in a separate much smaller table.

#     Parameters:
#     - cohort_wallets (array-like): List of wallet addresses to include.
#     - cohort_coins (array-like): List of coin IDs to include.

#     returns:
#     - metadata_df (df): metadata about each coin, with total supply necessary to calculate metrics
#     - all_balances_df (df): daily wallet activity necessary to calculate relevant metrics

#     '''
#     start_time = time.time()
#     logger.info('Preparing cohort_profits_df...')

#         # Raise an error if either the wallet cohort or coin list is empty
#     if len(cohort_wallets) == 0:
#         raise ValueError("Wallet cohort is empty. Provide at least one wallet address.")
#     if len(cohort_coins) == 0:
#         raise ValueError("Coin list is empty. Provide at least one coin ID.")


#     cohort_profits_df = profits_df[
#         (profits_df['wallet_address'].isin(cohort_wallets)) &
#         (profits_df['coin_id'].isin(cohort_coins))
#     ]
#     cohort_profits_df = cohort_profits_df[['coin_id','wallet_address','date','balance','net_transfers']]

#     # SQL query to retrieve metadata for the coins, filtered by the coin list
#     metadata_sql = f'''
#         select c.coin_id
#         ,c.symbol
#         ,c.total_supply
#         from `core.coins` c
#         where c.coin_id in {tuple(cohort_coins)}  -- Filter on the coin list
#         and c.total_supply is not null
#     '''
#     metadata_df = dgc().run_sql(metadata_sql)
#     logger.debug('Coin metadata retrieved after %.2f seconds.', time.time() - start_time)
#     step_time = time.time()

#     # convert coin_id string column to categorical to reduce memory usage
#     cohort_profits_df['coin_id'] = cohort_profits_df['coin_id'].astype('category')
#     logger.debug('Converted coin_ids column from string to categorical after %.2f seconds.', time.time() - step_time)

#     logger.info('Generated cohort_profits_df after %.2f seconds.', time.time() - start_time)

#     return cohort_profits_df,metadata_df



# def resample_profits_df(cohort_profits_df, resampling_period=3):
#     '''
#     Resamples the cohort_profits_df over a specified resampling time period by aggregating coin-wallet
#     pair activity within each resampling period.

#     Parameters:
#     - cohort_profits_df (pd.DataFrame): DataFrame containing profits data for wallet-coin pairs.
#     - resampling_period (int): Number of days to group for resampling (default is 3 days).

#     Returns:
#     - resampled_df (pd.DataFrame): DataFrame resampled over the specified period with balance and net_transfers.
#     '''
#     start_time = time.time()
#     logger.info('Preparing resampled_profits_df...')

#     # Ensure 'date' column is of datetime type
#     cohort_profits_df['date'] = pd.to_datetime(cohort_profits_df['date'])

#     # Set 'date' as index for resampling
#     cohort_profits_df.set_index('date', inplace=True)

#     # Step 1: Group by wallet_address and coin_id
#     grouped_df = cohort_profits_df.groupby(['wallet_address', 'coin_id'])

#     # Step 2: Resample within each group over the specified period (e.g., 3 days)
#     # This will generate rows for periods without any data, so we need to handle this.
#     resampled_df = grouped_df.resample(f'{resampling_period}D').agg({
#         'balance': 'last',         # Retain the last balance for each period
#         'net_transfers': 'sum'     # Sum net transfers for each period
#     }).reset_index()

#     # Step 3: Exclude rows where net_transfers is exactly 0
#     resampled_df = resampled_df[resampled_df['net_transfers'] != 0]

#     logger.info('Generated resampled_profits_df after %.2f seconds.', time.time() - start_time)

#     return resampled_df



# def calculate_coin_metrics(metadata_df,balances_df):
#     '''
#     Calculate various metrics for a specific cryptocurrency coin and merge them into a single df.

#     Parameters:
#     - all_balances_df (dataframe): Contains balance information on all dates for all coins.
#     - all_metadata_df (dataframe): Contains containing metadata for all coins.
#     - coin_id (str): The coin ID to calculate metrics for.

#     Returns:
#     - coin_metrics_df (dataframe): Contains the calculated metrics and metadata for the specified coin.
#     '''
#     logger.info('Calculating metrics for %s', metadata_df['symbol'].iloc[0])
#     total_supply = metadata_df['total_supply'].iloc[0]
#     coin_id = metadata_df['coin_id'].iloc[0]

#     # Calculate Metrics
#     # -----------------

#     # Metric 1: Wallets by Ownership
#     # Shows what whale wallets and small wallets are doing
#     wallets_by_ownership_df = calculate_wallet_counts(balances_df, total_supply)

#     # Metric 2: Buyers New vs Repeat
#     # Shows how many daily buyers are first-time vs repeat buyers
#     buyers_new_vs_repeat_df = calculate_buyer_counts(balances_df)

#     # Metric 3: Gini Coefficients
#     # Gini coefficients based on wallet balances
#     gini_df = calculate_daily_gini(balances_df)
#     gini_df_excl_mega_whales = calculate_gini_without_mega_whales(balances_df, total_supply)


#     # Merge All Metrics into One DataFrame
#     # ------------------------------------
#     logger.debug('Merging all metrics into one df...')
#     metrics_dfs = [
#         wallets_by_ownership_df,
#         buyers_new_vs_repeat_df,
#         gini_df,
#         gini_df_excl_mega_whales
#     ]
#     coin_metrics_df = metrics_dfs[0]
#     for df in metrics_dfs[1:]:
#         coin_metrics_df = coin_metrics_df.join(df, how='outer')

#     # reset index
#     coin_metrics_df = coin_metrics_df.reset_index().rename(columns={'index': 'date'})

#     # add coin_id
#     coin_metrics_df['coin_id'] = coin_id

#     return coin_metrics_df



# def calculate_wallet_counts(balances_df,total_supply):
#     '''
#     Consolidates wallet transactions into a daily count of wallets that control a certain \
#         percentage of total supply

#     params:
#     - balances_df (dataframe): df showing daily wallet balances of a coin_id token that \
#         has been filtered to only include one coin_id.
#     - total_supply (float): the total supply of the coin

#     returns:
#     - wallets_df (dataframe): df of wallet counts based on percent of total supply
#     '''
#     start_time = time.time()

#     # Calculate wallet bin sizes from total supply
#     wallet_bins, wallet_labels = generate_wallet_bins(total_supply)

#     logger.debug('Calculating daily balances for each wallet...')
#     start_time = time.time()

#     # Forward fill balances to ensure each date has the latest balance for each wallet
#     balances_df = balances_df.sort_values(by=['wallet_address', 'date'])
#     balances_df['balance'] = balances_df.groupby('wallet_address')['balance'].ffill()

#     # Classify each balance into ownership percentage bins
#     balances_df['wallet_types'] = pd.cut(balances_df['balance'], bins=wallet_bins, labels=wallet_labels)

#     # Group by date and wallet type, then count the number of occurrences
#     wallets_df = balances_df.groupby(['date', 'wallet_types'], observed=False).size().unstack(fill_value=0)

#     # Add rows for dates with 0 transactions
#     date_range = pd.date_range(start=wallets_df.index.min(), end=wallets_df.index.max(), freq='D')
#     wallets_df = wallets_df.reindex(date_range, fill_value=0)

#     # Fill empty cells with 0s
#     wallets_df.fillna(0, inplace=True)

#     logger.debug('Daily balance calculations complete after %.2f seconds', time.time() - start_time)

#     return wallets_df



# def generate_wallet_bins(total_supply):
#     '''
#     defines bins for wallet balances based on what percent of total supply they own

#     params:
#     - total_supply (float): total supply of the coin

#     returns:
#     - wallet_bins (list of floats): the number of tokens a wallet needs to be included in a bin
#     - wallet_labels (list of strings): the label for each bin
#     '''

#     # defining bin boundaries
#     percentages = [
#         0.00000001,
#         0.0000010,
#         0.0000018,
#         0.0000032,
#         0.0000056,
#         0.0000100,
#         0.0000180,
#         0.0000320,
#         0.0000560,
#         0.0001000,
#         0.0001800,
#         0.0003200,
#         0.0005600,
#         0.0010000,
#         0.0018000,
#         0.0032000,
#         0.0056000,
#         0.0100000
#     ]

#     wallet_bins = [total_supply * pct for pct in percentages] + [np.inf]

#     # defining labels
#     wallet_labels = [
#         'wallets_0p000001_pct',
#         'wallets_0p00010_pct',
#         'wallets_0p00018_pct',
#         'wallets_0p00032_pct',
#         'wallets_0p00056_pct',
#         'wallets_0p0010_pct',
#         'wallets_0p0018_pct',
#         'wallets_0p0032_pct',
#         'wallets_0p0056_pct',
#         'wallets_0p010_pct',
#         'wallets_0p018_pct',
#         'wallets_0p032_pct',
#         'wallets_0p056_pct',
#         'wallets_0p10_pct',
#         'wallets_0p18_pct',
#         'wallets_0p32_pct',
#         'wallets_0p56_pct',
#         'wallets_1p0_pct'
#     ]

#     return wallet_bins, wallet_labels



# def calculate_daily_gini(balances_df):
#     '''
#     Calculates the Gini coefficient for the distribution of wallet balances for each date.

#     params:
#     - balances_df (dataframe): df showing daily wallet balances of a coin_id token that \
#         has been filtered to only include one coin_id.

#     returns:
#     - gini_df (dataframe): df with dates as the index and the Gini coefficients as the values.
#     '''
#     start_time = time.time()

#     # Get the most recent balance for each wallet each day
#     balances_df = balances_df.sort_values(by=['wallet_address', 'date'])
#     daily_balances = balances_df.drop_duplicates(subset=['wallet_address', 'date'], keep='last')


#     # Calculate Gini coefficient for each day
#     gini_coefficients = daily_balances.groupby('date')['balance'].apply(efficient_gini)

#     # Convert the Series to a DataFrame for better readability
#     gini_df = gini_coefficients.reset_index(name='gini_coefficient')
#     gini_df.set_index('date', inplace=True)

#     logger.debug('Daily gini coefficients complete after %.2f seconds', time.time() - start_time)
#     return gini_df



# def calculate_gini_without_mega_whales(balances_df, total_supply):
#     '''
#     computes the gini coefficient while ignoring wallets that have ever held >5% of \
#         the total supply. the hope is that this removes treasury accounts, burn addresses, \
#         and other wallets that are not likely to be wallet owners.


#     params:
#     - balances_df (dataframe): df showing daily wallet balances of a coin_id token that \
#         has been filtered to only include one coin_id.
#     - total_supply (float): the total supply of the coin

#     returns:
#     - gini_filtered_df (dataframe): df of gini coefficient without mega whales
#     '''
#     # filter out addresses that have ever owned 5% or more supply
#     mega_whales = balances_df.loc[balances_df['balance'] >= (total_supply * 0.05), 'wallet_address'].unique()
#     balances_df_filtered = balances_df[~balances_df['wallet_address'].isin(set(mega_whales))]

#     # calculate gini
#     gini_filtered_df = calculate_daily_gini(balances_df_filtered)
#     gini_filtered_df.rename(columns={'gini_coefficient': 'gini_coefficient_excl_mega_whales'}, inplace=True)

#     return gini_filtered_df



# def efficient_gini(arr):
#     """
#     Calculate the Gini coefficient, a measure of inequality, for a given array.

#     The Gini coefficient ranges between 0 and 1, where 0 indicates perfect equality
#     (everyone has the same value), and 1 indicates maximum inequality (all the value
#     is held by one entity).

#     Parameters:
#     arr : numpy.ndarray or list
#         A 1D array or list containing the values for which the Gini coefficient
#         should be calculated. These values represent a distribution (e.g., income, wealth).

#     Returns:
#     float or None:
#         - Gini coefficient rounded to 6 decimal places if the array is non-empty and contains positive values.
#         - None if the sum of values is 0 or the array is empty (which means Gini cannot be calculated).

#     Notes:
#     - The array is first sorted because the Gini coefficient requires ordered data.
#     - This method uses an efficient, vectorized approach for computation.
#     """
#     # Sort the input array in ascending order
#     arr = np.sort(arr)

#     # Get the number of elements in the array
#     n = len(arr)

#     # Return None if the total sum of the array or number of elements is zero
#     if (n * np.sum(arr)) == 0:
#         return None

#     # Return None if there are negative balances in the array
#     if np.any(arr < 0):
#         return None

#     # Create an index array starting from 1 to n
#     index = np.arange(1, n + 1)

#     # Calculate the Gini coefficient
#     gini = (2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr))

#     # Return the Gini coefficient rounded to 6 decimal places
#     return round(gini, 6)



# def generate_coin_wallet_metrics(cohort_wallets,cohort_coins):
#     '''
#     HTTP-triggered Cloud Function that calculates and uploads metrics related to the
#     distribution of coin ownership across wallets for all tracked coins.

#     Steps:
#     1. Retrieves required datasets from BigQuery:
#        - Coin metadata (e.g., total supply, chain ID, token address)
#        - Daily wallet balances and transaction details for each coin.
#     2. For each coin, calculates several metrics:
#        - Wallet ownership distribution: Classifies wallets into bins based on percentage of total coin supply held.
#        - New vs. repeat buyers: Counts first-time and repeat buyers for each day.
#        - Gini coefficient: Measures wealth inequality among wallets.
#        - Gini coefficient excluding "mega whales": Filters out wallets holding more than 5% of total supply to refine inequality analysis.
#     3. Aggregates metrics for all coins into a single DataFrame.
#     4. Uploads the results to the BigQuery table `core.coin_wallet_metrics`.

#     Raises:
#     - May raise errors related to data retrieval, computation, or BigQuery upload if any step fails.
#     '''
#     # retrieve full sets of metadata and daily wallet balances
#     all_metadata_df,all_balances_df = prepare_datasets(cohort_wallets,cohort_coins)

#     # filter unique_coin_ids to include only those that have corresponding metadata
#     metadata_coin_ids = all_metadata_df['coin_id'].unique().tolist()
#     balances_coin_ids = all_balances_df['coin_id'].drop_duplicates().tolist()
#     unique_coin_ids = [c for c in balances_coin_ids if c in metadata_coin_ids]

#     coin_metrics_df_list = []

#     logger.info('Starting generation of metrics for each coin...')
#     # generate metrics for all coins
#     for c in unique_coin_ids:
#         # retrieve coin-specific dfs; balances_df will be altered so it needs the slower .copy()
#         logger.debug('Filtering coin-specific data from all_balances_df...')
#         balances_df = all_balances_df.loc[all_balances_df['coin_id'] == c].copy()
#         metadata_df = all_metadata_df.loc[all_metadata_df['coin_id'] == c]

#         # skip the coin if we do not have total supply from metadata_df, we cannot calculate all metrics
#         if metadata_df.empty:
#             logger.debug("skipping coin_id %s as no matching metadata found.", c)
#             continue

#         # calculate and merge metrics
#         coin_metrics_df = calculate_coin_metrics(metadata_df,balances_df)
#         coin_metrics_df_list.append(coin_metrics_df)
#         logger.debug('Successfully retrieved coin_metrics_df.')


#     # fill zeros for missing dates (currently impacts buyer behavior and gini columns)
#     all_coin_metrics_df = pd.concat(coin_metrics_df_list, ignore_index=True)
#     all_coin_metrics_df.fillna(0, inplace=True)

#     return 'finished refreshing core.coin_wallet_metrics.'
