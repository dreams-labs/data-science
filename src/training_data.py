"""
functions used in generating training data for the models
"""
# pylint: disable=W1203 # fstrings in logs
# pylint: disable=C0301 # line over 100 chars

import time
import pandas as pd
import numpy as np
from dreams_core.googlecloud import GoogleCloud as dgc
from dreams_core import core as dc

# set up logger at the module level
logger = dc.setup_logger()


def retrieve_prices_data():
    """
    Retrieves prices data from the core.coin_market_data table and converts coin_id to categorical

    Returns:
    - prices_df: DataFrame containing coin prices with 'coin_id' as a categorical column.
    """
    # SQL query to retrieve prices data
    query_sql = '''
        select cmd.coin_id
        ,date
        ,cast(cmd.price as float64) as price
        from core.coin_market_data cmd
        order by 1,2
    '''

    # Run the SQL query using dgc's run_sql method
    logger.debug('retrieving prices data...')
    prices_df = dgc().run_sql(query_sql)

    # Convert coin_id column to categorical to reduce memory usage
    prices_df['coin_id'] = prices_df['coin_id'].astype('category')

    prices_df['date'] = pd.to_datetime(prices_df['date'])

    logger.info('retrieved prices data with shape %s',prices_df.shape)

    return prices_df



def fill_prices_gaps(prices_df, max_gap_days):
    """
    Forward-fills small gaps in price data for each coin_id.
    Parameters:
    - prices_df: DataFrame containing price data with columns 'coin_id', 'date', and 'price'.
    - max_gap_days: The maximum allowed consecutive missing days for forward-filling.

    Returns:
    - prices_filled_df: DataFrame with small gaps forward-filled, excluding coins with gaps 
        too large to fill.
    - outcomes_df: DataFrame tracking the outcome for each coin_id.
    """
    # Get unique coin_ids
    unique_coins = prices_df['coin_id'].unique()

    # List to store results
    filled_results = []

    # List to track outcomes for each coin
    outcomes = []

    # Iterate over each coin_id
    for coin_id in unique_coins:
        # Step 1: Reindex to create rows for all missing dates
        coin_df = prices_df[prices_df['coin_id'] == coin_id].copy()

        # Create the full date range
        full_date_range = pd.date_range(start=coin_df['date'].min(), end=coin_df['date'].max(), freq='D')

        # Reindex to get all dates
        coin_df = coin_df.set_index('date').reindex(full_date_range).rename_axis('date').reset_index()
        coin_df['coin_id'] = coin_id  # Fills coin_id in the newly created rows

        # Step 2: Count the number of sequential missing dates
        missing_values = coin_df['price'].isnull().astype(int)
        consecutive_groups = coin_df['price'].notnull().cumsum()
        coin_df['missing_gap'] = missing_values.groupby(consecutive_groups).cumsum()

        # Check if there are no gaps at all
        if coin_df['missing_gap'].max() == 0:
            outcomes.append({'coin_id': coin_id, 'outcome': 'no gaps'})
            filled_results.append(coin_df)
            continue

        # Check if any gaps exceed max_gap_days
        if coin_df['missing_gap'].max() > max_gap_days:
            outcomes.append({'coin_id': coin_id, 'outcome': 'gaps above threshold'})
            continue

        # Step 3: Forward-fill any gaps that are smaller than max_gap_days
        coin_df['price'] = coin_df['price'].ffill(limit=max_gap_days)

        # Remove rows with larger gaps that shouldn't be filled (already handled by check above)
        coin_df = coin_df[coin_df['missing_gap'] <= max_gap_days]


        # Append to the result list
        filled_results.append(coin_df)
        outcomes.append({'coin_id': coin_id, 'outcome': 'gaps below threshold'})

    # Concatenate all results
    if filled_results:
        prices_filled_df = pd.concat(filled_results).reset_index(drop=True)
    else:
        prices_filled_df = pd.DataFrame()  # Handle case where no coins were filled

    # Drop the temporary 'missing_gap' column
    prices_filled_df = prices_filled_df.drop(columns=['missing_gap'])

    # Convert outcomes to DataFrame
    outcomes_df = pd.DataFrame(outcomes)

    # Log summary based on outcomes_df
    no_gaps_count = len(outcomes_df[outcomes_df['outcome'] == 'no gaps'])
    gaps_below_threshold_count = len(outcomes_df[outcomes_df['outcome'] == 'gaps below threshold'])
    gaps_above_threshold_count = len(outcomes_df[outcomes_df['outcome'] == 'gaps above threshold'])

    logger.debug("retained %s coins.", no_gaps_count + gaps_below_threshold_count)
    logger.info("%s coins had no gaps, %s coins had gaps filled, and %s coins were dropped due to large gaps.",
                no_gaps_count, gaps_below_threshold_count, gaps_above_threshold_count)

    return prices_filled_df, outcomes_df



def create_target_variables(prices_df, modeling_period_start, modeling_period_end, moon_threshold, dump_threshold):
    """
    Creates a DataFrame with target variable 'is_moon' for each coin based on price performance 
    during the modeling period.

    Parameters:
    - prices_df: DataFrame containing price data with columns 'coin_id', 'date', and 'price'.
    - modeling_period_start: Start date of the modeling period.
    - modeling_period_end: End date of the modeling period.
    - moon_threshold: The return a coin must have during the modeling period for is_moon=True
    - dump_threshold: The loss a coin must have during the modeling period for is_dump=True

    Returns:
    - target_variable_df: DataFrame with columns 'coin_id' and 'is_moon'.
    - outcomes_df: DataFrame tracking outcomes for each coin.
    """
    # Convert modeling period start and end to datetime
    modeling_period_start = pd.to_datetime(modeling_period_start)
    modeling_period_end = pd.to_datetime(modeling_period_end)

    # Filter for the modeling period and sort the df
    modeling_period_df = prices_df[(prices_df['date'] >= modeling_period_start) & (prices_df['date'] <= modeling_period_end)]
    modeling_period_df = modeling_period_df.sort_values(by=['coin_id', 'date'])

    # Process coins with data
    target_data = []
    outcomes = []
    for coin_id, group in modeling_period_df.groupby('coin_id'):
        # Get the price on the start and end dates
        price_start = group[group['date'] == modeling_period_start]['price'].values
        price_end = group[group['date'] == modeling_period_end]['price'].values

        # Check if both start and end prices exist
        if len(price_start) > 0 and len(price_end) > 0:
            # create the target variable
            price_start = price_start[0]
            price_end = price_end[0]
            is_moon = int(price_end >= (1+moon_threshold) * price_start)
            is_dump = int(price_end <= (1+dump_threshold) * price_start)
            target_data.append({'coin_id': coin_id, 'is_moon': is_moon, 'is_dump': is_dump})
            outcomes.append({'coin_id': coin_id, 'outcome': 'target variable created'})

        else:
            # log coins with price data that does not overlap with the full modeling period
            if len(price_start) == 0 and len(price_end) == 0:
                outcomes.append({'coin_id': coin_id, 'outcome': 'missing both'})
            elif len(price_start) == 0:
                outcomes.append({'coin_id': coin_id, 'outcome': 'missing start price'})
            else:
                outcomes.append({'coin_id': coin_id, 'outcome': 'missing end price'})

    # Log outcomes for coins with no data in the modeling period
    coins_with_no_data = set(prices_df['coin_id'].unique()) - set(modeling_period_df['coin_id'].unique())
    for coin_id in coins_with_no_data:
        outcomes.append({'coin_id': coin_id, 'outcome': 'missing both'})

    # Convert target data and outcomes to DataFrames
    target_variables_df = pd.DataFrame(target_data)
    outcomes_df = pd.DataFrame(outcomes)

    # Log summary based on outcomes
    target_variable_count = len(outcomes_df[outcomes_df['outcome'] == 'target variable created'])
    missing_start_count = len(outcomes_df[outcomes_df['outcome'] == 'missing start price'])
    missing_end_count = len(outcomes_df[outcomes_df['outcome'] == 'missing end price'])
    missing_both_count = len(outcomes_df[outcomes_df['outcome'] == 'missing both'])
    moons = target_variables_df[target_variables_df['is_moon']==1].shape[0]
    dumps = target_variables_df[target_variables_df['is_dump']==1].shape[0]
    logger.info(
        "Target variables created for %s coins with %s/%s (%s) moons and %s/%s (%s) dumps.",target_variable_count,
        moons, target_variable_count, dc.human_format(100*moons/target_variable_count)+'%',
        dumps, target_variable_count, dc.human_format(100*dumps/target_variable_count)+'%'
    )
    logger.info(
        "Target variables not created for %s coins missing start price, %s missing end price, and %s missing both.",
        missing_start_count, missing_end_count, missing_both_count
    )

    return target_variables_df,outcomes_df



def retrieve_transfers_data(training_period_start,modeling_period_start,modeling_period_end):
    """
    Retrieves wallet transfers data from the core.coin_wallet_transfers table and converts 
    columns to categorical for calculation efficiency. 

    New rows are added for every coin-wallet pair start as of the start and end of the training 
    and modeling periods. If there was no existing row then the transfer is counted as 0 and the
    existing balance is carried forward. 
      
    These imputed rows are needed for profitability calculations and are very compute intensive to 
    add using pandas. 
    
    Params:
    - training_period_start: String with format 'YYYY-MM-DD'
    - modeling_period_start: String with format 'YYYY-MM-DD'
    - modeling_period_end: String with format 'YYYY-MM-DD'

    Returns:
    - transfers_df: DataFrame with columns ['coin_id', 'wallet_address', 'date', 'net_transfers', 'balance']
    """

    query_sql = f'''
        -- STEP 1: retrieve transfers data through the end of the modeling period
        -------------------------------------------------------------------------
        with transfers_base as (
            -- start with the same query that generates transfers_df
            select cwt.coin_id
            ,cwt.wallet_address
            ,cwt.date
            ,cast(cwt.net_transfers as float64) as net_transfers
            ,cast(cwt.balance as float64) as balance
            from `core.coin_wallet_transfers` cwt

            -- inner join to filter onto only coins with price data
            join (
                select coin_id
                from `core.coin_market_data`
                group by 1
            ) cmd on cmd.coin_id = cwt.coin_id
            
            -- remove some of the largest coins that add many records and aren't altcoins
            where cwt.coin_id not in (
                '06b54bc2-8688-43e7-a49a-755300a4f995' -- SHIB
                ,'eb52375e-f394-4632-a9cf-3a9291a8ebf7' -- COMP
                ,'88779ad0-f8c3-448c-bc6c-699c2a692514' -- CRV
            )

            -- don't retrieve transfers through after modeling period
            and cwt.date <= '{modeling_period_end}'
        ),


        -- STEP 2: create new records for all coin-wallet pairs as of the training_period_start
        ---------------------------------------------------------------------------------------
        -- any coins that had existing balances when the training period begins will have these
        -- balances reflected as a transfer in in the net_transfers column

        training_start_existing_rows as (
            -- identify coin-wallet pairs that already have a balance as of the period end
            select *
            from transfers_base
            where date = '{training_period_start}'
        ),
        training_start_needs_rows as (
            -- for coin-wallet pairs that don't have existing records, identify the row closest to the period end date
            select t.*
            ,row_number() over (partition by t.coin_id,t.wallet_address order by t.date desc) as rn
            from transfers_base t
            left join training_start_existing_rows e on e.coin_id = t.coin_id 
                and e.wallet_address = t.wallet_address
            where t.date < '{training_period_start}'
            and e.coin_id is null
        ),
        training_start_new_rows as (
            -- create a new row for the period end date by carrying the balance from the closest existing record
            select coin_id
            ,wallet_address
            ,cast('{training_period_start}' as date) as date
            ,cast(balance as float64) as net_transfers -- treat starting balances as a transfer in
            ,cast(balance as float64) as balance
            from training_start_needs_rows
            where rn=1
        ),


        -- STEP 3: create new records for all coin-wallet pairs as of the end of the training period
        --------------------------------------------------------------------
        training_end_existing_rows as (
            -- identify coin-wallet pairs that already have a balance as of the period end
            select *
            from transfers_base
            where date = date_sub('{modeling_period_start}', interval 1 day)
        ),
        training_end_needs_rows as (
            -- for coin-wallet pairs that don't have existing records, identify the row closest to the period end date
            select t.*
            ,row_number() over (partition by t.coin_id,t.wallet_address order by t.date desc) as rn
            from transfers_base t
            left join training_end_existing_rows e on e.coin_id = t.coin_id 
                and e.wallet_address = t.wallet_address
            where t.date < date_sub('{modeling_period_start}', interval 1 day)
            and e.coin_id is null
        ),
        training_end_new_rows as (
            -- create a new row for the period end date by carrying the balance from the closest existing record
            select coin_id
            ,wallet_address
            ,date_sub('{modeling_period_start}', interval 1 day) as date
            ,0 as net_transfers
            ,cast(balance as float64) as balance
            from training_end_needs_rows
            where rn=1
        ),


        -- STEP 4: create new records for all coin-wallet pairs as of the start of the modeling period
        --------------------------------------------------------------------
        modeling_start_existing_rows as (
            -- identify coin-wallet pairs that already have a balance as of the period start
            select *
            from transfers_base
            where date = '{modeling_period_start}'
        ),
        modeling_start_needs_rows as (
            -- for coin-wallet pairs that don't have existing records, identify the row closest to the period start date
            select t.*
            ,row_number() over (partition by t.coin_id,t.wallet_address order by t.date desc) as rn
            from transfers_base t
            left join modeling_start_existing_rows e on e.coin_id = t.coin_id 
                and e.wallet_address = t.wallet_address
            where t.date < '{modeling_period_start}'
            and e.coin_id is null
        ),
        modeling_start_new_rows as (
            -- create a new row for the period start date by carrying the balance from the closest existing record
            select coin_id
            ,wallet_address
            ,cast('{modeling_period_start}' as date) as date
            ,0 as net_transfers
            ,cast(balance as float64) as balance
            from modeling_start_needs_rows
            where rn=1
        ),

        -- STEP 5: create new records for all coin-wallet pairs as of the end of the modeling period
        --------------------------------------------------------------------
        modeling_end_existing_rows as (
            -- identify coin-wallet pairs that already have a balance as of the period end
            select *
            from transfers_base
            where date = '{modeling_period_end}'
        ),
        modeling_end_needs_rows as (
            -- for coin-wallet pairs that don't have existing records, identify the row closest to the period end date
            select t.*
            ,row_number() over (partition by t.coin_id,t.wallet_address order by t.date desc) as rn
            from transfers_base t
            left join modeling_end_existing_rows e on e.coin_id = t.coin_id 
                and e.wallet_address = t.wallet_address
            where t.date < '{modeling_period_end}'
            and e.coin_id is null
        ),
        modeling_end_new_rows as (
            -- create a new row for the period end date by carrying the balance from the closest existing record
            select coin_id
            ,wallet_address
            ,cast('{modeling_period_end}' as date) as date
            ,0 as net_transfers
            ,cast(balance as float64) as balance
            from modeling_end_needs_rows
            where rn=1
        )

        -- STEP 6: merge all rows together
        --------------------------------------------------------------------
        select * from transfers_base
        where date >= '{training_period_start}' -- transfers prior to the training period are summarized in training_start_new_rows

        union all
        select * from training_start_new_rows
        union all
        select * from training_end_new_rows
        union all
        select * from modeling_start_new_rows
        union all
        select * from modeling_end_new_rows

        order by coin_id, wallet_address, date

    '''

    # Run the SQL query using dgc's run_sql method
    start_time = time.time()
    logger.debug('retrieving transfers data...')
    transfers_df = dgc().run_sql(query_sql)

    # Convert column types
    transfers_df['coin_id'] = transfers_df['coin_id'].astype('category')
    transfers_df['date'] = pd.to_datetime(transfers_df['date'])

    logger.info('retrieved transfers_df with shape %s after %s seconds.',
                transfers_df.shape, round(time.time()-start_time,1))

    return transfers_df



def prepare_profits_data(transfers_df, prices_df):
    """
    Prepares a DataFrame (profits_df) by merging wallet transfer data with coin price data,
    ensuring valid pricing data is available for each transaction, and handling cases where
    wallets had balances prior to the first available pricing data.

    The function performs the following steps:
    1. Merges the `transfers_df` and `prices_df` on 'coin_id' and 'date'.
    2. Identifies wallets with transfer records before the first available price for each coin.
    3. Creates new records for these wallets, treating the balance as a net transfer on the 
       first price date.
    4. Removes original records with missing price data.
    5. Appends the newly created records and sorts the resulting DataFrame.

    Parameters:
    - transfers_df (pd.DataFrame): 
        A DataFrame containing wallet transaction data with columns:
        - coin_id: The ID of the coin/token.
        - wallet_address: The unique identifier of the wallet.
        - date: The date of the transaction.
        - net_transfers: The net tokens transferred in or out of the wallet on that date.
        - balance: The token balance in the wallet at the end of the day.

    - prices_df (pd.DataFrame): 
        A DataFrame containing price data with columns:
        - coin_id: The ID of the coin/token.
        - date: The date of the price record.
        - price: The price of the coin/token on that date.

    Returns:
    - pd.DataFrame: 
        A merged DataFrame containing profitability data, with new records added for wallets
        that had balances prior to the first available price date for each coin.
    """
    logger.info("Preparing profits_df data...")
    start_time = time.time()

    # Raise an error if either df is empty
    if transfers_df.empty or prices_df.empty:
        raise ValueError("Input DataFrames cannot be empty.")

    # 1. Format dataframes and merge on 'coin_id' and 'date'
    # ----------------------------------------------------------
    # set dates to datetime and coin_ids to categorical
    transfers_df = transfers_df.copy()
    prices_df = prices_df.copy()

    transfers_df['date'] = pd.to_datetime(transfers_df['date'])
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    transfers_df['coin_id'] = transfers_df['coin_id'].astype('category')
    prices_df['coin_id'] = prices_df['coin_id'].astype('category')

    # merge datasets
    profits_df = pd.merge(transfers_df, prices_df, on=['coin_id', 'date'], how='left')
    logger.debug(f"<Step 1> merge transfers and prices: {time.time() - start_time:.2f} seconds")
    step_time = time.time()


    # 2. Attach data showing the first price record of all coins
    # ----------------------------------------------------------
    # Identify the earliest pricing data for each coin and merge to get the first price date
    first_prices_df = prices_df.groupby('coin_id',observed=True).agg({
        'date': 'min',
        'price': 'first'  # Assuming we want the first available price on the first_price_date
    }).reset_index()
    first_prices_df.columns = ['coin_id', 'first_price_date', 'first_price']

    # Merge the first price data into profits_df
    profits_df = profits_df.merge(first_prices_df, on='coin_id', how='left')
    logger.debug(f"<Step 2> identify first prices of coins: {time.time() - step_time:.2f} seconds")
    step_time = time.time()


    # 3. Create new records for the first_price_date for each wallet-coin pair
    # ------------------------------------------------------------------------
    # Identify wallets with transfer data before first_price_date
    pre_price_transfers = profits_df[profits_df['date'] < profits_df['first_price_date']]

    # Group by coin_id and wallet_address to ensure only one record per pair
    grouped_pre_price_transfers = pre_price_transfers.groupby(['coin_id', 'wallet_address'],observed=True).agg({
        'first_price_date': 'first',
        'balance': 'last',  # Get the balance as of the latest transfer before first_price_date
        'first_price': 'first'  # Retrieve the first price for this coin
    }).reset_index()

    # Create the new records to reflect transfer in of balance as of first_price_date
    new_records = grouped_pre_price_transfers.copy()
    new_records['date'] = new_records['first_price_date']
    new_records['net_transfers'] = new_records['balance']  # Treat the balance as a net transfer
    new_records['price'] = new_records['first_price']  # Use the first price on the first_price_date

    # # Select necessary columns for new records
    new_records = new_records[['coin_id', 'wallet_address', 'date', 'net_transfers', 'balance', 'price', 'first_price_date', 'first_price']]
    logger.debug(f"<Step 3> created new records as of the first_price_date: {time.time() - step_time:.2f} seconds")
    step_time = time.time()


    # 4. Remove the rows prior to pricing data append the new records
    # ---------------------------------------------------------------
    # Remove original records with no price data (NaN in 'price' column)
    profits_df = profits_df[profits_df['price'].notna()]

    # Append new records to the original dataframe
    profits_df = pd.concat([profits_df, new_records], ignore_index=True)

    # Sort by coin_id, wallet_address, and date to maintain order
    profits_df = profits_df.sort_values(by=['coin_id', 'wallet_address', 'date']).reset_index(drop=True)
    logger.debug(f"<Step 4> merge new records into profits_df: {time.time() - step_time:.2f} seconds")
    step_time = time.time()


    # 5. Remove all records before a coin-wallet pair has any priced tokens
    # ---------------------------------------------------------------------
    # these are artifacts resulting from activity prior to price data availability. if wallets purchased 
    # and sold all coins in these pre-data eras, their first record will be of a zero-balance state.

    # calculate cumulative token inflows
    profits_df['token_inflows'] = profits_df['net_transfers'].where(profits_df['net_transfers'] > 0, 0)
    profits_df['token_inflows_cumulative'] = profits_df.groupby(['coin_id', 'wallet_address'],observed=True)['token_inflows'].cumsum()

    # remove records prior to positive token_inflows
    profits_df = profits_df[profits_df['token_inflows_cumulative']>0]

    logger.debug(f"<Step 5> removed records prior to each wallet's first token inflows: {time.time() - step_time:.2f} seconds")
    step_time = time.time()


    # 6. Tidy up and return profits_df
    # ---------------------------------
    # remove helper columns
    profits_df.drop(columns=['first_price_date', 'first_price', 'token_inflows', 'token_inflows_cumulative'], inplace=True)

    # Reset the index
    profits_df = profits_df.reset_index(drop=True)

    logger.debug(f"generated profits_df after {time.time() - start_time:.2f} total seconds")

    return profits_df



def calculate_wallet_profitability(profits_df):
    """
    Calculates the profitability metrics for each wallet-coin pair by analyzing changes in price 
    and balance over time. The function computes both daily profitability changes and cumulative 
    profitability, along with additional metrics such as USD inflows and returns.

    Process Summary:
    1. Ensure there are no missing prices in the `profits_df`.
    2. Calculate the daily profitability based on price changes and previous balances.
    3. Compute the cumulative profitability for each wallet-coin pair using `cumsum()`.
    4. Calculate USD balances, net transfers, inflows, and the overall rate of return.

    Parameters:
    - profits_df (pd.DataFrame): 
        A DataFrame containing merged wallet transaction and price data with columns:
        - coin_id: The ID of the coin/token.
        - wallet_address: The unique identifier of the wallet.
        - date: The date of the transaction.
        - net_transfers: The net tokens transferred in or out of the wallet on that date.
        - balance: The token balance in the wallet at the end of the day.
        - price: The price of the coin/token on that date.

    Returns:
    - pd.DataFrame: 
        A DataFrame with the following additional columns:
        - profits_change: The daily change in profitability, calculated as the difference between 
          the current price and the previous price, multiplied by the previous balance.
        - profits_cumulative: The cumulative profitability for each wallet-coin pair over time.
        - usd_balance: The USD value of the wallet's balance, based on the current price.
        - usd_net_transfers: The USD value of the net transfers on a given day.
        - usd_inflows: The USD value of net transfers into the wallet (positive transfers only).
        - usd_inflows_cumulative: The cumulative USD inflows for each wallet-coin pair.
        - total_return: The total return, calculated as cumulative profits divided by total USD inflows.

    Raises:
    - ValueError: If any missing prices are found in the `profits_df`.
    """
    logger.debug("Starting generation of profits_df...")
    start_time = time.time()

    # Raise an error if there are any missing prices
    if profits_df['price'].isnull().any():
        raise ValueError("Missing prices found for some transfer dates. This indicates an issue with the price data generation.")

    # create offset price and balance rows to easily calculate changes between periods
    profits_df['previous_price'] = profits_df.groupby(['coin_id', 'wallet_address'],observed=True)['price'].shift(1)
    profits_df['previous_price'] = profits_df['previous_price'].fillna(profits_df['price'])
    profits_df['previous_balance'] = profits_df.groupby(['coin_id', 'wallet_address'],observed=True)['balance'].shift(1).fillna(0)

    logger.debug(f"Offset prices and balances for profitability logic: {time.time() - start_time:.2f} seconds")
    step_time = time.time()

    # calculate the profitability change in each period and sum them to get cumulative profitability
    profits_df['profits_change'] = (profits_df['price'] - profits_df['previous_price']) * profits_df['previous_balance']
    profits_df['profits_cumulative'] = profits_df.groupby(['coin_id', 'wallet_address'],observed=True)['profits_change'].cumsum()

    logger.debug(f"Calculate profitability: {time.time() - step_time:.2f} seconds")
    step_time = time.time()

    # Calculate USD inflows, balances, and rate of return
    profits_df['usd_balance'] = profits_df['balance'] * profits_df['price']
    profits_df['usd_net_transfers'] = profits_df['net_transfers'] * profits_df['price']
    profits_df['usd_inflows'] = profits_df['usd_net_transfers'].where(profits_df['usd_net_transfers'] > 0, 0)
    profits_df['usd_inflows_cumulative'] = profits_df.groupby(['coin_id', 'wallet_address'],observed=True)['usd_inflows'].cumsum()
    profits_df['total_return'] = profits_df['profits_cumulative'] / profits_df['usd_inflows_cumulative'].where(profits_df['usd_inflows_cumulative'] != 0, np.nan)

    logger.debug(f"Calculate rate of return {time.time() - step_time:.2f} seconds")
    step_time = time.time()

    # Drop helper columns
    profits_df.drop(columns=['previous_price', 'previous_balance'], inplace=True)

    total_time = time.time() - start_time
    logger.info(f"Generated profits df after {total_time:.2f} seconds")

    return profits_df



def clean_profits_df(profits_df, data_cleaning_config):
    """
    Clean the profits DataFrame by excluding all records for any wallet_addresses that either have: 
     - aggregate profitabiilty above profitability_filter (abs value of gains or losses).
     - aggregate USD inflows above the inflows_filter
    this catches outliers such as minting/burning addresses, contract addresses, etc and ensures
    they are not included in the wallet behavior training data. 
    
    Parameters:
    - profits_df: DataFrame with columns ['coin_id', 'wallet_address', 'date', 'profits_cumulative']
    - data_cleaning_config:
        - profitability_filter: Threshold value to exclude pairs with profits or losses exceeding this value
        - inflows_filter: Threshold value to exclude pairs with USD inflows
        
    Returns:
    - Cleaned DataFrame with records for coin_id-wallet_address pairs filtered out.
    """
    logger.debug("Starting generation of profits_cleaned_df...")
    start_time = time.time()

    # 1. Remove wallets with higher or lower total profits than the profitability_filter
    # ----------------------------------------------------------------------------------
    # Group by wallet_address and calculate the total profitability
    wallet_profits_agg_df = profits_df.groupby('wallet_address')['profits_change'].sum().reset_index()

    # Identify wallet_addresses with total profitability that exceeds the threshold
    exclusions_profits_df = wallet_profits_agg_df[
        (wallet_profits_agg_df['profits_change'] >= data_cleaning_config['profitability_filter']) |
        (wallet_profits_agg_df['profits_change'] <= -data_cleaning_config['profitability_filter'])
    ][['wallet_address']]

    # Merge to filter out the records with those wallet addresses
    profits_cleaned_df = profits_df.merge(exclusions_profits_df, on='wallet_address', how='left', indicator=True)
    profits_cleaned_df = profits_cleaned_df[profits_cleaned_df['_merge'] == 'left_only']
    profits_cleaned_df.drop(columns=['_merge'], inplace=True)

    # 2. Remove wallets with higher total inflows than the inflows_filter
    # -------------------------------------------------------------------
    # Group by wallet_address and calculate the total inflows
    wallet_inflows_agg_df = profits_df.groupby('wallet_address')['usd_inflows'].sum().reset_index()

    # Identify wallet addresses where total inflows exceed the threshold
    exclusions_inflows_df = wallet_inflows_agg_df[
        wallet_inflows_agg_df['usd_inflows'] >= data_cleaning_config['inflows_filter']
    ][['wallet_address']]

    # Merge to filter out the records with those wallet addresses
    profits_cleaned_df = profits_cleaned_df.merge(exclusions_inflows_df, on='wallet_address', how='left', indicator=True)
    profits_cleaned_df = profits_cleaned_df[profits_cleaned_df['_merge'] == 'left_only']
    profits_cleaned_df.drop(columns=['_merge'], inplace=True)


    # 3. Prepare exclusions_df and output logs
    # ----------------------------------------
    # prepare exclusions_logs_df
    exclusions_profits_df['profits_exclusion'] = True
    exclusions_inflows_df['inflows_exclusion'] = True
    exclusions_logs_df = exclusions_profits_df.merge(exclusions_inflows_df, on='wallet_address', how='outer')

    # Fill NaN values with False for missing exclusions
    exclusions_logs_df['profits_exclusion'] = exclusions_logs_df['profits_exclusion'].astype(bool).fillna(False)
    exclusions_logs_df['inflows_exclusion'] = exclusions_logs_df['inflows_exclusion'].astype(bool).fillna(False)

    # log outputs
    total_time = time.time() - start_time
    logger.info("Finished cleaning profits_df after %.2f seconds.",total_time)
    logger.debug("Identified %s coin-wallet pairs beyond profit threshold of $%s and %s pairs beyond inflows filter of %s.",
        exclusions_profits_df.shape[0], dc.human_format(data_cleaning_config['profitability_filter']),
        exclusions_inflows_df.shape[0], dc.human_format(data_cleaning_config['inflows_filter'])
    )

    return profits_cleaned_df,exclusions_logs_df



def classify_shark_coins(profits_df, training_data_config):
    """
    Classify wallet-coin pairs as sharks based on their profitability and return performance, filtered 
    to the shark eligibility criteria. Pairs are classified either based on gross profits (is_profits_shark)
    or based on return (is_returns_shark). Pairs that match either criteria are given is_shark=True. 

    Steps:
    1. Filter wallets that have inflows above a predefined threshold (shark eligibility).
    2. Identify wallets as sharks based on either:
    a. Total profits exceeding a configured threshold.
    b. Total return exceeding a configured threshold.
    3. Add a combined 'is_shark' column for wallets that meet either profits or returns criteria.

    Parameters:
        profits_df (DataFrame): A DataFrame containing wallet transactions and profits data.
        training_data_config (dict): A configuration object containing modeling parameters, including:
            - 'modeling_period_start': Start date for filtering data.
            - 'shark_coin_minimum_inflows': Minimum total USD inflow to be eligible as a shark.
            - 'shark_coin_profits_threshold': Minimum total profits to be considered a profits shark.
            - 'shark_coin_return_threshold': Minimum total return to be considered a returns shark.

    Returns:
        shark_coins_df (DataFrame): A DataFrame of coin-wallet pairs classified as sharks, including 
        flags for whether the wallet meets the profits or return criteria.
    """
    logger.debug('identifying coin-level sharks...')

    # Step 1. Remove modeling period data and wallets that do not meet shark eligibility
    # ----------------------------------------------------------------------------------
    # Filter out transfers that occured after the end of the training period
    filtered_profits_df = profits_df[profits_df['date'] < training_data_config['modeling_period_start']].copy()

    # Identify coin-wallet pairs that have deposited enough total USD to be considered sharks
    eligible_wallets_df = filtered_profits_df.groupby(['coin_id', 'wallet_address'])['usd_inflows_cumulative'].max().reset_index()
    eligible_wallets_df = eligible_wallets_df[eligible_wallets_df['usd_inflows_cumulative'] >= training_data_config['shark_coin_minimum_inflows']]

    # Filter profits_df to only include eligible wallets
    filtered_profits_df = filtered_profits_df.merge(eligible_wallets_df[['coin_id', 'wallet_address']],
                                                    on=['coin_id', 'wallet_address'],
                                                    how='inner')

    # Step 2. Determine which eligible wallets are classified as sharks
    # -----------------------------------------------------------------------------------
    # identify sharks based on absolute balance of profits
    total_profits_df = filtered_profits_df.sort_values('date').groupby(['coin_id', 'wallet_address']).last()['profits_cumulative'].reset_index()
    shark_coins_df = eligible_wallets_df.merge(total_profits_df, on=['coin_id', 'wallet_address'])
    shark_coins_df['is_profits_shark'] = shark_coins_df['profits_cumulative'] >= training_data_config['shark_coin_profits_threshold']

    # identify sharks based on percentage return
    total_returns_df = filtered_profits_df.sort_values('date').groupby(['coin_id', 'wallet_address']).last()['total_return'].reset_index()
    shark_coins_df = shark_coins_df.merge(total_returns_df, on=['coin_id', 'wallet_address'])
    shark_coins_df['is_returns_shark'] = shark_coins_df['total_return'] >= training_data_config['shark_coin_return_threshold']

    # add a combined column for wallets that meet either criteria
    shark_coins_df['is_shark'] = shark_coins_df['is_profits_shark'] | shark_coins_df['is_returns_shark']

    logger.info('creation of shark_coins_df complete.')

    return shark_coins_df



def classify_shark_wallets(shark_coins_df, training_data_config):
    """
    Classifies wallets as sharks based on their activity across multiple coins.

    Parameters:
        shark_coins_df (DataFrame): A DataFrame containing wallet-coin records and shark classification.
        training_data_config (dict): A configuration object containing thresholds for megashark classification:
            - 'shark_wallet_type': Column indicating whether a wallet is a shark.
            - 'shark_wallet_min_coins': Minimum number of coins for megashark classification.
            - 'shark_wallet_min_shark_rate': Minimum shark rate for megashark classification.

    Returns:
        shark_wallets_df (DataFrame): A DataFrame containing wallets classified as megasharks.
    """
    shark_wallet_type = training_data_config['shark_wallet_type']
    shark_wallet_min_coins = training_data_config['shark_wallet_min_coins']
    shark_wallet_min_shark_rate = training_data_config['shark_wallet_min_shark_rate']

    # Calculate the total number of coins each wallet has records with
    all_coins_df = shark_coins_df.groupby('wallet_address')['coin_id'].count().reset_index()
    all_coins_df.columns = ['wallet_address', 'total_coins']

    # Calculate the number of coins each wallet is a shark on
    shark_coins_df = shark_coins_df[shark_coins_df[shark_wallet_type]].groupby('wallet_address')['coin_id'].count().reset_index()
    shark_coins_df.columns = ['wallet_address', 'shark_coins']

    # Merge the two DataFrames to get total and shark coins for each wallet
    shark_wallets_df = all_coins_df.merge(shark_coins_df, on='wallet_address', how='left')
    shark_wallets_df['shark_coins'] = shark_wallets_df['shark_coins'].fillna(0)

    # Calculate the shark_rate for each wallet
    shark_wallets_df['shark_rate'] = shark_wallets_df['shark_coins'] / shark_wallets_df['total_coins']

    # Classify wallets as megasharks based on minimum coins and shark rate thresholds
    shark_wallets_df['is_shark'] = (
        (shark_wallets_df['total_coins'] >= shark_wallet_min_coins)
        & (shark_wallets_df['shark_rate'] >= shark_wallet_min_shark_rate)
    )

    return shark_wallets_df



def calculate_shark_performance(transfers_df, prices_df, shark_wallets_df, config):
    """
    Calculate shark and non-shark performance based on profitability and inflows during the modeling period.

    Args:
        transfers_df (pd.DataFrame): DataFrame containing transfer data.
        prices_df (pd.DataFrame): DataFrame containing price data.
        shark_wallets_df (pd.DataFrame): DataFrame containing shark wallet classification.
        config (dict): Configuration dictionary containing modeling period details.

    Returns:
        pd.DataFrame: DataFrame summarizing shark performance with aggregated return.
    """

    # Filter transfers for the modeling period
    modeling_period_transfers_df = transfers_df[
        (transfers_df['date'] >= config['training_data']['modeling_period_start']) &
        (transfers_df['date'] <= config['training_data']['modeling_period_end'])
    ]

    # Create profits_df for the modeling period
    logger.info("Creating modeling period profits data...")
    modeling_period_profits_df = prepare_profits_data(modeling_period_transfers_df, prices_df)
    modeling_period_profits_df = calculate_wallet_profitability(modeling_period_profits_df)

    # Retrieve profit state at the end of the period for each coin-wallet pair
    modeling_end_profits_df = modeling_period_profits_df[
        modeling_period_profits_df['date'] == config['training_data']['modeling_period_end']
    ]

    # Aggregate wallet-level metrics by summing usd inflows and profits
    modeling_end_wallet_profits_df = modeling_end_profits_df.groupby('wallet_address')[
        ['usd_inflows_cumulative', 'profits_cumulative']
    ].sum()

    # Add prefix to the additional columns in shark_wallets_df
    shark_wallets_df_prefixed = shark_wallets_df.add_prefix('training_period_')
    shark_wallets_df_prefixed['wallet_address'] = shark_wallets_df['wallet_address']
    shark_wallets_df_prefixed['is_shark'] = shark_wallets_df['is_shark']

    # Classify wallets by shark status and compare their performance
    shark_wallets_performance_df = shark_wallets_df_prefixed.merge(
        modeling_end_wallet_profits_df,
        on='wallet_address',
        how='left'
    )

    # Replace NaNs with 0s for wallets that had no inflows and profits in the modeling period
    shark_wallets_performance_df['usd_inflows_cumulative'] = shark_wallets_performance_df['usd_inflows_cumulative'].fillna(0)
    shark_wallets_performance_df['profits_cumulative'] = shark_wallets_performance_df['profits_cumulative'].fillna(0)

    # aggregate metrics on is_shark
    shark_agg_performance_df = shark_wallets_performance_df.groupby('is_shark').agg(
        count_wallets=('wallet_address', 'size'),
        median_inflows=('usd_inflows_cumulative', 'median'),
        median_profits=('profits_cumulative', 'median'),
        mean_inflows=('usd_inflows_cumulative', 'mean'),
        min_inflows=('usd_inflows_cumulative', 'min'),
        max_inflows=('usd_inflows_cumulative', 'max'),
        percentile_25_inflows=('usd_inflows_cumulative', lambda x: np.percentile(x.dropna(), 25) if len(x) > 1 else np.nan),
        percentile_75_inflows=('usd_inflows_cumulative', lambda x: np.percentile(x.dropna(), 75) if len(x) > 1 else np.nan),
        mean_profits=('profits_cumulative', 'mean'),
        min_profits=('profits_cumulative', 'min'),
        max_profits=('profits_cumulative', 'max'),
        percentile_25_profits=('profits_cumulative', lambda x: np.percentile(x.dropna(), 25) if len(x) > 1 else np.nan),
        percentile_75_profits=('profits_cumulative', lambda x: np.percentile(x.dropna(), 75) if len(x) > 1 else np.nan),
        total_inflows=('usd_inflows_cumulative', 'sum'),
        total_profits=('profits_cumulative', 'sum')
    )
    # Calculate median return
    shark_agg_performance_df['median_return'] = np.divide(
        shark_agg_performance_df['median_profits'],
        shark_agg_performance_df['median_inflows'],
        out=np.zeros_like(shark_agg_performance_df['median_profits']),
        where=shark_agg_performance_df['median_inflows'] != 0
    )
    # Calculate aggregate return
    shark_agg_performance_df['return_aggregate'] = np.divide(
        shark_agg_performance_df['total_profits'],
        shark_agg_performance_df['total_inflows'],
        out=np.zeros_like(shark_agg_performance_df['total_profits']),
        where=shark_agg_performance_df['total_inflows'] != 0
    )

    # calculate separate metrics for wallets that had transactions during the modeling period
    nonzero_shark_wallets_performance_df = shark_wallets_performance_df[shark_wallets_performance_df['profits_cumulative']!=0]

    nonzero_shark_agg_performance_df = nonzero_shark_wallets_performance_df.groupby('is_shark').agg(
        count_wallets=('wallet_address', 'size'),
        median_inflows=('usd_inflows_cumulative', 'median'),
        median_profits=('profits_cumulative', 'median'),
        percentile_25_inflows=('usd_inflows_cumulative', lambda x: np.percentile(x.dropna(), 25) if len(x) > 1 else np.nan),
        percentile_75_inflows=('usd_inflows_cumulative', lambda x: np.percentile(x.dropna(), 75) if len(x) > 1 else np.nan),
        percentile_25_profits=('profits_cumulative', lambda x: np.percentile(x.dropna(), 25) if len(x) > 1 else np.nan),
        percentile_75_profits=('profits_cumulative', lambda x: np.percentile(x.dropna(), 75) if len(x) > 1 else np.nan),
    )
    # Calculate median return
    nonzero_shark_agg_performance_df['median_return'] = np.divide(
        nonzero_shark_agg_performance_df['median_profits'],
        nonzero_shark_agg_performance_df['median_inflows'],
        out=np.zeros_like(nonzero_shark_agg_performance_df['median_profits']),
        where=nonzero_shark_agg_performance_df['median_inflows'] != 0
    )


    # Prefix the nonzero DataFrame columns
    nonzero_shark_agg_performance_df = nonzero_shark_agg_performance_df.add_prefix('nonzero_')

    # Merge the two DataFrames on 'is_shark'
    shark_agg_performance_df = shark_agg_performance_df.merge(
        nonzero_shark_agg_performance_df, on='is_shark', how='left'
    )

    return shark_agg_performance_df,shark_wallets_performance_df
