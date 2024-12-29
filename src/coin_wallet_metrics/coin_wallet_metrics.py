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

    logger.info("No wallet cohort activity found for %s coins during the window.",
                len(profits_df['coin_id'].unique()) - len(cohort_profits_df['coin_id'].unique()))

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



def split_dataframe_by_coverage(
        time_series_df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        id_column: Optional[str] = 'coin_id',
        drop_outside_date_range: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the input DataFrame into full coverage and partial coverage based on date range.
    Works for both multi-series and single-series datasets.

    Params:
    - time_series_df (pd.DataFrame): The input DataFrame with time series data.
    - start_date (pd.Timestamp): Start date of the training period.
    - end_date (pd.Timestamp): End date of the modeling period.
    - id_column (Optional[str]): The name of the column used to identify different series.
    - drop_outside_date_range (Optional[bool]): Whether to remove all rows that are outside of
        the start_date and end_date

    Returns:
    - full_coverage_df (pd.DataFrame): DataFrame with series having complete data for the period.
    - partial_coverage_df (pd.DataFrame): DataFrame with series having partial data for the period.
    """
    # Convert params to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Create copy of df
    time_series_df = time_series_df.copy()

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

    logger.debug("Split df with dimensions %s into %s full coverage records and %s partial coverage records.",
                time_series_df.shape,
                len(full_coverage_df),
                len(partial_coverage_df))

    if drop_outside_date_range:
        # Remove rows outside the date range for both dataframes
        full_coverage_df = (full_coverage_df[(full_coverage_df['date'] >= start_date) &
                                             (full_coverage_df['date'] <= end_date)])
        partial_coverage_df = (partial_coverage_df[(partial_coverage_df['date'] >= start_date) &
                                                   (partial_coverage_df['date'] <= end_date)])

        # Log the number of remaining records
        total_remaining = len(full_coverage_df) + len(partial_coverage_df)
        logger.debug("After removing records outside the date range, %s records remain.",
                    total_remaining)

    return full_coverage_df, partial_coverage_df



def apply_period_boundaries(
        time_series_df: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
    """
    Splits DataFrame into records that fall within the date range.

    Params:
    - time_series_df (DataFrame): Input DataFrame with 'date' column
    - start_date, end_date (strings): the earliest and latest dates to retain formatted as YYYY-MM-DD

    Returns:
    - in_range_df
    """
    # Convert params to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    in_range_df = time_series_df[
        (time_series_df['date'] >= start_date)
        & (time_series_df['date'] <= end_date)
    ]

    return in_range_df



def generate_coin_metadata_features(metadata_df, config):
    """
    Generate model-friendly coin metadata features.

    Args:
    - metadata_df: DataFrame containing coin_id, categories, and chain information.
    - config: Configuration dict that includes the chain threshold.

    Returns:
    - A DataFrame with coin_id, boolean category columns, and boolean chain columns.
    """
    metadata_df = metadata_df.copy()

    # Step 1: Create boolean columns for each unique category
    logger.debug("Creating boolean columns for each category...")

    # Get all unique categories from the categories column
    all_categories = set(cat for sublist in metadata_df['categories'] for cat in sublist)

    # Create boolean columns for each category
    for category in all_categories:
        column_name = f"category_{category.lower()}"
        metadata_df[column_name] = metadata_df['categories'].apply(
            lambda cats, category=category: category.lower() in [c.lower() for c in cats]
        )

    # Step 2: Process chain data and apply threshold
    logger.debug("Processing chain data and applying chain threshold...")

    # Lowercase chain values
    metadata_df['chain'] = metadata_df['chain'].str.lower()

    # Count number of coins per chain
    chain_counts = metadata_df['chain'].value_counts()
    chain_threshold = config['datasets']['coin_facts']['coin_metadata']['chain_threshold']

    # Create boolean columns for chains above the threshold
    for chain, count in chain_counts.items():
        if count >= chain_threshold:
            metadata_df[f'chain_{chain}'] = metadata_df['chain'] == chain

    # Create chain_other column for chains below the threshold or missing chain data
    metadata_df['chain_other'] = metadata_df['chain'].apply(
        lambda x: chain_counts.get(x, 0) < chain_threshold if pd.notna(x) else True
    )

    # Step 3: Return the final DataFrame with boolean columns
    # Keep only relevant columns
    columns_to_keep = ['coin_id'] + [
        col for col in metadata_df.columns
        if col.startswith('category_') or col.startswith('chain_')
    ]
    metadata_features_df = metadata_df[columns_to_keep].copy()
    logger.info("Generated coin_metadata_features_df.")

    return metadata_features_df



def generate_macro_trends_features(macro_trends_df, config):
    """
    Generate model-friendly macro trends features. This means filtering the df to only
    include columns with metrics configurations and trimming the dataset to the period
    range.

    Params:
    - macro_trends_df (DataFrame): DataFrame with macro trends columns for all available
        metrics and rows for all available dates
    - config (dict): config.yaml
    """
    # Retrieve config variables
    selected_columns = config['datasets']['macro_trends'].keys()
    start_date = config['training_data']['training_period_start']
    end_date = config['training_data']['modeling_period_end']

    # Filter to only the columns with metrics configurations
    macro_trends_df = macro_trends_df[macro_trends_df.columns.intersection(selected_columns)]

    # Filter to only current window
    macro_trends_df = macro_trends_df.loc[start_date:end_date]
    columns_with_missing = macro_trends_df.columns[macro_trends_df.isnull().any()].tolist()

    # If there are columns with missing values, log a warning and drop them
    if columns_with_missing:
        for column in columns_with_missing:
            logger.warning(
                "Dropping macro trends column '%s' due to missing values "
                "in time window %s to %s.",
                column, start_date, end_date)

        macro_trends_df = macro_trends_df.drop(columns=columns_with_missing)

    macro_trends_df = macro_trends_df.reset_index()

    return macro_trends_df
