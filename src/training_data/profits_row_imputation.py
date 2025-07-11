"""
Functions used to impute additional rows in profits_df for all coin-wallet pairs,
which is needed because rows at the period start and end dates are required for
many calculations.
"""
import time
from datetime import timedelta
import warnings
import threading
import queue
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dreams_core import core as dc

# Local module imports
import utils as u
from utils import ConfigError

# set up logger at the module level
logger = dc.setup_logger()

# ____________________________________________________________________________
#                  Multi-Threaded profits_df Row Imputation
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
# All functions below this point relate to multi-threaded imputation of new rows
# for profits_df, which are needed to ensure we have complete data at the start
# and end of the training and test periods. As this is dealing with >10^8 rows,
# it is computationally expensive and is greatly sped up by using multithreading
# to split the calculations between cores and by using only vectorized operations
# to manipulate dfs.



# -----------------------------------
#       Main Interface Function
# -----------------------------------

def impute_profits_for_multiple_dates(profits_df, prices_df, dates, n_threads, reset_index=True):
    """
    Wrapper function to impute profits for multiple dates using multithreaded processing.

    Params:
    - profits_df (DataFrame): DataFrame containing dated profits data
    - prices_df (DataFrame): DataFrame containing price information
    - dates (list): List of dates for imputation
    - n_threads (int): Number of threads to use
    - reset_index (bool): Whether to reset index before returning

    Returns:
    - DataFrame: Updated profits_df with imputed rows (or original if no imputation needed)
    """
    if not isinstance(dates, list):
        raise TypeError(f"dates parameter '{dates}' must be a list.")

    if n_threads > 12:
        raise ConfigError("Thread counts for 'impute_profits_for_multiple_dates()' should be at or "
                          "below 12 to maintain reasonably sized batches.")

    start_time = time.time()
    logger.debug("Starting profits_df imputation for %s dates...", len(dates))

    # Create indices for vectorized operations
    profits_df = u.ensure_index(profits_df)
    prices_df = u.ensure_index(prices_df)

    new_rows_list = []

    for date in dates:
        new_rows_df = multithreaded_impute_profits_rows(profits_df, prices_df, date, n_threads)

        if not new_rows_df.empty:
            if new_rows_df.isna().any().any():
                raise ValueError(f"NaN values found in imputed rows for date: {date}")
            new_rows_list.append(new_rows_df)

    # If no new rows were generated, return original DataFrame
    if not new_rows_list:
        logger.debug("No new rows generated during imputation. Returning original DataFrame.")
        return profits_df.reset_index() if reset_index else profits_df

    # Concatenate and append new rows
    all_new_rows = pd.concat(new_rows_list, ignore_index=False)
    updated_profits_df = pd.concat([profits_df, all_new_rows], ignore_index=False)

    if reset_index:
        updated_profits_df = updated_profits_df.reset_index()

    logger.info("Completed new row generation after %.2f seconds. Total rows: %s",
                time.time() - start_time,
                updated_profits_df.shape[0])

    return updated_profits_df




# ---------------------------------
#         Helper Functions
# ---------------------------------

def multithreaded_impute_profits_rows(profits_df, prices_df, target_date, n_threads):
    """
    Imputes new profits_df rows as of the taget_date by using multithreading to maximize
    performance, and then merges and returns the results. First, profits_df is split on coin_id
    into n_threads partitions. Then a thread is started and impute_profits_df_rows() is
    calculated by all threads before being merged into all_new_rows_df.

    Args:
        profits_df (pd.DataFrame): DataFrame containing dated profits data for coin-wallet pairs
        prices_df (pd.DataFrame): DataFrame containing price information
        target_date (str or datetime): The date for which to impute rows
        n_threads (int): The number of threads to run impute_profits_df_rows() with
        reset_index (bool): Whether to reset the index before returning the df

    Returns:
        all_new_rows_df (pd.DataFrame): The set of new rows to be added to profits_df to
            ensure that every coin-wallet pair has a record on the given date.
    """
    # Partition profits_df on coin_id
    logger.debug("Splitting profits_df into %s partitions for date %s...",n_threads,target_date)
    profits_df_partitions = create_partitions(profits_df, n_threads)

    # Create a thread-safe queue to store results
    result_queue = queue.Queue()

    # Create a list to hold thread objects
    threads = []

    # Create and start a thread for each partition
    logger.debug("Initiating multithreading calculations for date %s...",target_date)

    for partition in profits_df_partitions:
        thread = threading.Thread(
            target=worker,
            args=(partition, prices_df, target_date, result_queue)
        )
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    # Merge results together
    warnings.simplefilter(action='ignore', category=FutureWarning) # ignore NaN dtype warnings
    if len(results) == 0:
        logger.warning("No new rows imputed for %s.", target_date)
        all_new_rows_df = profits_df.head(0)
    else:
        all_new_rows_df = pd.concat(results, ignore_index=False)
        logger.debug("Generated %s new rows for date %s.",
                    len(all_new_rows_df),
                    target_date)

    return all_new_rows_df



def impute_profits_df_rows(
        profits_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        target_date: str
    ) -> pd.DataFrame:
    """
    Impute rows for all coin-wallet pairs in profits_df on the target date using only
    vectorized functions, i.e. there are no groupby statements or for loops/lambda
    functions that iterate over each row. This is necessary due to the size and memory
    requirements of the input df.

    This function performs the following steps:
    1. Splits profits_df into records before and after the target date
    2. Filters for pairs needing new rows
    3. Identifies the last date for each coin-wallet pair
    4. Appends price columns for the last date and target date
    5. Calculates new values for pairs needing rows
    6. Concatenates the new rows with the original dataframe

    Args:
        profits_df (pd.DataFrame): DataFrame containing profit information
        prices_df (pd.DataFrame): DataFrame containing price information
        target_date (str or datetime): The date for which to impute rows

    Returns:
        profits_df_filled (pd.DataFrame): Updated profits DataFrame with imputed rows

    Raises:
        ValueError: If joining prices_df removes rows from profits_df
    """
    start_time = time.time()
    logger.debug('%s Imputing rows for all coin-wallet pairs in profits_df on %s...',
                profits_df.shape,
                target_date)

    # Data Checks and Typecasting
    # ---------------------------
    # Convert date to datetime
    target_date = pd.to_datetime(target_date)

    # Get data boundaries for detailed error reporting
    profits_start = pd.to_datetime(profits_df.index.get_level_values('date').min())
    profits_end = pd.to_datetime(profits_df.index.get_level_values('date').max())
    prices_start = pd.to_datetime(prices_df.index.get_level_values('date').min())
    prices_end = pd.to_datetime(prices_df.index.get_level_values('date').max())

    # Check if target_date is earlier than all profits_df dates
    if target_date < profits_start:
        logger.warning(
            f"Target date {target_date.strftime('%Y-%m-%d')} is earlier than all dates in profits_df.\n"
            f"Profits data coverage: {profits_start.strftime('%Y-%m-%d')} to {profits_end.strftime('%Y-%m-%d')}\n"
            f"Profits shape: {profits_df.shape}"
        )
        return profits_df.head(0)

    # Check if target_date is later than all prices_df dates
    if pd.to_datetime(target_date) > prices_end:
        logger.warning(
            f"Target date {target_date.strftime('%Y-%m-%d')} is later than all dates in prices_df.\n"
            f"Prices data coverage: {prices_start.strftime('%Y-%m-%d')} to {prices_end.strftime('%Y-%m-%d')}\n"
            f"Prices shape: {prices_df.shape}"
        )
        return profits_df.head(0)

    # Step 1: Split profits_df records before and after the target_date
    # -----------------------------------------------------------------
    profits_df_after_target = profits_df.xs(
        slice(target_date + pd.Timedelta('1 day'), None),
        level=2,
        drop_level=False)
    profits_df = profits_df.xs(slice(None, target_date), level=2, drop_level=False)

    logger.debug("%s <Step 1> Split profits_df into %s rows through the target_date and %s after "
                    "target_date: %.2f seconds",
                    profits_df.shape,
                    len(profits_df),
                    len(profits_df_after_target),
                    time.time() - start_time)
    step_time = time.time()


    # Step 2: Filter profits_df to only pairs that need new rows
    # ----------------------------------------------------------
    # Create a boolean mask for rows at the target_date
    target_date_mask = profits_df.index.get_level_values('date') == target_date

    # Create a boolean mask for pairs that don't have a row at the target_date
    pairs_mask = ~profits_df.index.droplevel('date').isin(
        profits_df[target_date_mask].index.droplevel('date')
    )
    profits_df = profits_df[pairs_mask].sort_index()

    logger.debug("%s <Step 2> Identified %s coin-wallet pairs that need imputed rows: %.2f seconds",
                    profits_df.shape,
                    len(profits_df),
                    time.time() - step_time)
    step_time = time.time()


    # Step 3: Identify the last date for each coin-wallet pair
    # --------------------------------------------------------
    # The logic here is that every row that doesn't have the same coin_id-wallet_address
    # combination as the previous row must indicate that the previous coin-wallet pair
    # just had its last date.

    # Create shifted index
    shifted_index = profits_df.index.to_frame().shift(-1)

    # Create boolean mask for last dates
    is_last_date = (profits_df.index.get_level_values('coin_id') != shifted_index['coin_id']) | \
            (profits_df.index.get_level_values('wallet_address') != shifted_index['wallet_address'])

    # Filter for last dates
    profits_df = profits_df[is_last_date]

    logger.debug("%s <Step 3> Filtered profits_df to only the last dates for each coin-wallet "
                    "pair: %.2f seconds",
                    profits_df.shape,
                    time.time() - step_time)
    step_time = time.time()


    # Step 4: Append columns for previous_price (on last date) and price (on the target_date)
    # ---------------------------------------------------------------------------------------
    # Add price_previous by joining the price as of the last date for each coin-wallet pair
    prejoin_size = len(profits_df)
    profits_df = profits_df.join(prices_df['price'], on=['coin_id', 'date'], how='inner')
    profits_df = profits_df.rename(columns={'price': 'price_previous'})

    # Add price by joining the price as of the target_date
    prices_target_date = prices_df.xs(target_date, level='date')
    profits_df = profits_df.join(prices_target_date['price'], on='coin_id', how='inner')

    if len(profits_df) != prejoin_size:
        raise ValueError(
            f"Inner join to prices_df on coin_id-date removed {prejoin_size - len(profits_df)} rows "
            f"from profits_df with original length {len(profits_df)}. There should be complete coverage "
            "for all rows in profits_df."
        )

    logger.debug("%s <Step 4> Joined prices_df and added price and previous_price helper "
                    "columns: %.2f seconds",
                    profits_df.shape,
                    time.time() - step_time)
    step_time = time.time()


    # Step 5: Calculate new values for pairs needing rows
    # ---------------------------------------------------
    # Profits calculations are in a separate helper function
    new_rows_df = calculate_new_profits_values(profits_df, target_date)

    logger.debug("%s <Step 5> Calculated %s new rows: %.2f seconds",
                    profits_df.shape,
                    len(new_rows_df),
                    time.time() - step_time)
    step_time = time.time()

    logger.debug("%s Successfully generated new_rows_df with shape %s after %.2f total seconds.",
                new_rows_df.shape,
                new_rows_df.shape,
                time.time() - start_time)

    # confirm there are no empty values
    if new_rows_df.isna().any().any():
        raise ValueError("new_rows_df contains unexpected empty values after the calculation.")


    return new_rows_df



def calculate_new_profits_values(profits_df, target_date):
    """
    Calculate new financial metrics for imputed rows.

    Args:
        profits_df (pd.DataFrame): DataFrame with price_previous and price columns
        target_date (datetime): The date for which to impute rows

    Returns:
        pd.DataFrame: DataFrame with calculated financial metrics
    """
    new_rows_df = pd.DataFrame(index=profits_df.index)

    # new profits_cumulative = (price change * usd_balance) + previous profits_cumulative
    new_rows_df['profits_cumulative'] = (
         ((profits_df['price'] / profits_df['price_previous'] - 1) * profits_df['usd_balance'])
         + profits_df['profits_cumulative'])

    # % change in price times the USD balance
    new_rows_df['usd_balance'] = ((profits_df['price'] / profits_df['price_previous'])
                                  * profits_df['usd_balance'])

    # these are zero since the transactionless day is being imputed
    new_rows_df['usd_net_transfers'] = np.float32(0)
    new_rows_df['usd_inflows'] = np.float32(0)

    # no new usd_inflows so cumulative remains the same
    new_rows_df['usd_inflows_cumulative'] = profits_df['usd_inflows_cumulative']

    # Set the date index to be the target_date
    if not new_rows_df.empty:
        new_rows_df = override_date_level(new_rows_df, target_date)

    # Log the column as imputed if the lineage column was added from log_column=True
    if 'is_imputed' in profits_df.columns:
        new_rows_df['is_imputed']=True

    return new_rows_df



def create_partitions(profits_df, n_partitions):
    """
    Partition an indexed DataFrame into multiple subsets based on unique coin_ids.

    Params:
    - profits_df (DataFrame): MultiIndexed on ['coin_id', 'wallet_address', 'date']
    - n_partitions (int): Number of partitions to create

    Returns:
    - partition_dfs (List[DataFrame]): List of partitioned DataFrames
    """
    # Get unique coin_ids from the index
    unique_coin_ids = profits_df.index.get_level_values('coin_id').unique().tolist()

    # Shuffle coin_ids
    np.random.seed(88)
    np.random.shuffle(unique_coin_ids)

    # Calculate coins per partition
    coins_per_partition = len(unique_coin_ids) // n_partitions

    # Create partitions
    partition_dfs = []
    for i in range(n_partitions):
        start_idx = i * coins_per_partition
        end_idx = start_idx + coins_per_partition if i < n_partitions - 1 else None
        partition_coin_ids = unique_coin_ids[start_idx:end_idx]

        # Use IndexSlice to filter the MultiIndex
        idx = pd.IndexSlice
        partition_df = profits_df.loc[idx[partition_coin_ids, :, :], :].copy()
        partition_dfs.append(partition_df)

    return partition_dfs



def worker(partition, prices_df, target_date, result_queue):
    """
    Worker function to process a partition and put the result in the queue.
    """
    # Call impute_profits_df_rows with the copy
    result = impute_profits_df_rows(partition, prices_df, target_date)

    # Put the result in the queue
    result_queue.put(result)



# ----------------------------------
#         Utility Functions
# ----------------------------------

def identify_imputation_dates(config):
    """
    Identifies all dates that must be imputed into profits_df.

    Params:
    - config (dict): config.yaml

    Returns:
    - imputation_dates (list of strings): list of all dates that need records showing unrealized
        profits as of that date
    """
    # Basic period boundary dates
    period_boundary_dates = [
        config['training_data']['training_period_end'],
        config['training_data']['training_period_start'],
    ]

    # Identify all unique cohort lookback periods
    cohort_lookback_periods = [
        cohort['lookback_period']
        for cohort in config['datasets']['wallet_cohorts'].values()
    ]

    # Determine the actual dates of the lookback period starts in this time window
    training_period_start = pd.to_datetime(config['training_data']['training_period_start'])
    lookback_start_dates = []
    for lbp in set(cohort_lookback_periods):
        lbp_start = training_period_start - timedelta(days=lbp)
        lookback_start_dates.append(lbp_start.strftime('%Y-%m-%d'))

    # Return combined list
    imputation_dates = set(period_boundary_dates + lookback_start_dates)
    imputation_dates = sorted(imputation_dates)

    return imputation_dates



def test_partition_performance(profits_df, prices_df, target_date, partition_numbers):
    """
    (Development function not used in production pipelines)

    Used to test the performance of the multithreaded_impute_profits function with different
    numbers of partitions.

    This function iterates through the provided partition numbers, running the
    multithreaded_impute_profits function for each. It measures the execution time, logs
    the result size, and generates a performance plot.

    Args:
        profits_df (pd.DataFrame): The input DataFrame containing profit information.
        prices_df (pd.DataFrame): The input DataFrame containing price information.
        target_date (str or datetime): The target date for imputation.
        partition_numbers (list of int): A list of partition numbers to test.

    Returns:
        list of tuple: A list of (partition_number, execution_time) tuples.

    Raises:
        ValueError: If the size of any result DataFrame differs from the others.

    Side effects:
        - Prints the execution time for each partition number to the console.
        - Logs the size of the result DataFrame for each partition number.
        - Generates and displays a plot of execution time vs. number of partitions.
    """
    results = []
    expected_size = None

    for n_partitions in partition_numbers:
        start_time = time.time()

        partitions = create_partitions(profits_df, n_partitions)
        result = multithreaded_impute_profits_rows(
            partitions,
            prices_df,
            target_date,
            partition_numbers)

        # Check size consistency
        current_size = result.shape[0]
        if expected_size is None:
            expected_size = current_size
        elif current_size != expected_size:
            raise ValueError(f"Inconsistent result size detected. Expected {expected_size} rows, "
                             f"but got {current_size} rows for {n_partitions} partitions.")

        end_time = time.time()
        execution_time = end_time - start_time

        results.append((n_partitions, execution_time))
        logger.info("Partitions: %s, Result Shape: %s , Time: %.2f seconds"
                    ,n_partitions
                    ,result.shape
                    ,execution_time)

    # Generate the plot
    partitions, times = zip(*results)
    plt.plot(partitions, times, marker='o')
    plt.xlabel('Number of Partitions')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Performance vs Number of Partitions')
    plt.show()

    return results


def override_date_level(df: pd.DataFrame, new_date) -> pd.DataFrame:
    """
    Params:
    - df (DataFrame): input data with a MultiIndex including 'date'.
    - new_date: the value to assign to all date levels.

    Returns:
    - df (DataFrame): updated DataFrame with 'date' index overridden.
    """
    # Get the position of the 'date' level in the MultiIndex
    date_level = df.index.names.index('date')
    # Create a new levels list: replace the 'date' level with a list containing only new_date
    new_levels = list(df.index.levels)
    new_levels[date_level] = [new_date]
    # Create new codes: set all codes for 'date' to 0 so they all point to new_date
    new_codes = list(df.index.codes)
    new_codes[date_level] = np.zeros(len(df.index), dtype=int)
    # Build a new MultiIndex with the updated level and codes
    df.index = pd.MultiIndex(levels=new_levels, codes=new_codes, names=df.index.names)
    return df
