"""
functions used in generating training data for the models
"""
# pylint: disable=C0301
# pylint: disable=W1203

import time
import pandas as pd
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
    logger.info('retrieving prices data...')
    prices_df = dgc().run_sql(query_sql)

    # Convert coin_id column to categorical to reduce memory usage
    prices_df['coin_id'] = prices_df['coin_id'].astype('category')

    prices_df['date'] = pd.to_datetime(prices_df['date'])

    return prices_df



def fill_prices_gaps(prices_df, max_gap_days=3):
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
        coin_df['price'] = coin_df['price'].fillna(method='ffill', limit=max_gap_days)

        # Remove rows with larger gaps that shouldn't be filled (already handled by check above)
        coin_df = coin_df[coin_df['missing_gap'] <= max_gap_days]

        # Drop the temporary 'missing_gap' column
        coin_df = coin_df.drop(columns=['missing_gap'])

        # Append to the result list
        filled_results.append(coin_df)
        outcomes.append({'coin_id': coin_id, 'outcome': 'gaps below threshold'})

    # Concatenate all results
    if filled_results:
        prices_filled_df = pd.concat(filled_results).reset_index(drop=True)
    else:
        prices_filled_df = pd.DataFrame()  # Handle case where no coins were filled

    # Convert outcomes to DataFrame
    outcomes_df = pd.DataFrame(outcomes)

    # Log summary based on outcomes_df
    no_gaps_count = len(outcomes_df[outcomes_df['outcome'] == 'no gaps'])
    gaps_below_threshold_count = len(outcomes_df[outcomes_df['outcome'] == 'gaps below threshold'])
    gaps_above_threshold_count = len(outcomes_df[outcomes_df['outcome'] == 'gaps above threshold'])

    logger.info("retained %s coins.", no_gaps_count + gaps_below_threshold_count)
    logger.info("%s coins had no gaps, %s coins had gaps filled, and %s coins were dropped due to large gaps.",
                no_gaps_count, gaps_below_threshold_count, gaps_above_threshold_count)

    return prices_filled_df, outcomes_df



def create_target_variable(prices_df, modeling_period_start, modeling_period_end, moon_threshold, dump_threshold):
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



def retrieve_transfers_data(modeling_period_start):
    """
    Retrieves wallet transfers data from the core.coin_wallet_transfers table and converts 
    columns to categorical for calculation efficiency. 

    Params:
    - modeling_period_start: String with format 'YYYY-MM-DD'
        Date after which data should be used to create target variables instead of training data. 
        In this function it is used to create transfer rows for all coin_id-wallet pairs so that 
        we can accurately calculate profitability as of the training period end. 

    Returns:
    - transfers_df: DataFrame with columns ['coin_id', 'wallet_address', 'date', 'net_transfers', 'balance']
    """
    # SQL query to retrieve prices data
    query_sql = f'''
        with transfers_base as (
            -- start with the same query that generates transfers_df
            select cwt.coin_id
            ,cwt.wallet_address
            ,cwt.date
            ,cast(cwt.net_transfers as float64) as net_transfers
            ,cast(cwt.balance as float64) as balance
            from `core.coin_wallet_transfers` cwt
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
        ),

        existing_rows as (
            -- identify coin-wallet pairs that already have a balance as of the training period end
            select *
            from transfers_base
            where date = '{modeling_period_start}'
        ),

        needs_rows as (
            -- for coin-wallet pairs that don't have existing records, identify the row closest to the period end date
            select t.*
            ,row_number() over (partition by t.coin_id,t.wallet_address order by t.date desc) as rn
            from transfers_base t
            left join existing_rows e on e.coin_id = t.coin_id 
                and e.wallet_address = t.wallet_address
            where t.date < '{modeling_period_start}'
            and e.coin_id is null
        ),

        transfers_new_rows as (
            -- create a new row for the period end date by carrying the balance from the closest existing record
            select coin_id
            ,wallet_address
            ,date_sub('{modeling_period_start}', interval 1 day) as date
            ,0 as net_transfers
            ,cast(balance as float64) as balance
            from needs_rows
            where rn=1
        ) 

        select * from transfers_base
        union all
        select * from transfers_new_rows
        order by coin_id, wallet_address, date
    '''

    # Run the SQL query using dgc's run_sql method
    start_time = time.time()
    logger.info('retrieving transfers data...')
    transfers_df = dgc().run_sql(query_sql)

    # Convert column types
    transfers_df['coin_id'] = transfers_df['coin_id'].astype('category')
    transfers_df['date'] = pd.to_datetime(transfers_df['date'])

    logger.info('retrieved transfers_df with shape %s after %s seconds.',
                transfers_df.shape, round(time.time()-start_time,1))

    return transfers_df



def calculate_wallet_profitability(transfers_df, prices_df):
    """
    Calculate the profitability of wallets by merging transaction (transfers) data with price data 
    and computing daily and cumulative profitability for each wallet-coin pair. The balance as of 
    the first price record is treated as an initial "transfer in" for calculating profitability.

    Parameters:
    - transfers_df (pd.DataFrame): 
        - coin_id: The ID of the coin/token.
        - wallet_address: The unique identifier of the wallet.
        - date: The date of the transaction.
        - net_transfers: The net tokens transferred in or out of the wallet on that date.
        - balance: The token balance in the wallet at the end of the day.
    - prices_df (pd.DataFrame): 
        - coin_id: The ID of the coin/token.
        - date: The date for the price record.
        - price: The price of the coin/token on that date.

    Returns:
    - pd.DataFrame: 
        A DataFrame with profitability metrics, including:
        - profitability_change: The change in profitability for each transaction, calculated as the 
          difference between the current price and the previous price, multiplied by the previous balance.
        - profitability_cumulative: The cumulative profitability for each wallet-coin pair over time.
    
    Process:
    1. Merge the `transfers_df` and `prices_df` on 'coin_id' and 'date'.
    2. Remove any transaction records earlier than the first available price date for each coin.
    3a. Calculate daily profitability based on price changes and previous balances.
    3b. Compute the cumulative profitability for each wallet-coin pair using `cumsum()`.
    """
    logger.info("Starting generation of profits_df...")
    start_time = time.time()

    # 1. Merge transfers and prices data on 'coin_id' and 'date'
    # ----------------------------------------------------------
    transfers_df['date'] = pd.to_datetime(transfers_df['date'])
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    profits_df = pd.merge(transfers_df, prices_df, on=['coin_id', 'date'], how='left')
    logger.debug(f"<Step 1> (Merge transfers and prices): {time.time() - start_time:.2f} seconds")
    step_time = time.time()

    # 2. Remove transfers history earlier than the first pricing data
    # ---------------------------------------------------------------
    # identify the earliest pricing data for each coin
    first_prices_df = prices_df.groupby('coin_id')['date'].min().reset_index()
    first_prices_df.columns = ['coin_id', 'first_price_date']

    # remove transfer data that occurred when we don't know the coin price
    profits_df = pd.merge(profits_df, first_prices_df, on='coin_id', how='left')
    profits_df = profits_df[profits_df['date'] >= profits_df['first_price_date']]
    logger.debug(f"<Step 2> (Remove records before first price date): {time.time() - step_time:.2f} seconds")
    step_time = time.time()

    # 3. Calculate profitability
    # --------------------------
    # create offset price and balance rows to easily calculate changes between periods
    profits_df['previous_price'] = profits_df.groupby(['coin_id', 'wallet_address'])['price'].shift(1)
    profits_df['previous_price'].fillna(profits_df['price'], inplace=True)
    profits_df['previous_balance'] = profits_df.groupby(['coin_id', 'wallet_address'])['balance'].shift(1).fillna(0)

    # calculate the profitability change in each period and sum them to get cumulative profitability
    profits_df['profitability_change'] = (profits_df['price'] - profits_df['previous_price']) * profits_df['previous_balance']
    profits_df['profitability_cumulative'] = profits_df.groupby(['coin_id', 'wallet_address'])['profitability_change'].cumsum()
    logger.debug(f"<Step 3> Calculate profitability: {time.time() - step_time:.2f} seconds")

    # Drop helper columns
    profits_df.drop(columns=['first_price_date', 'previous_price', 'previous_balance'], inplace=True)

    total_time = time.time() - start_time
    logger.info(f"Generated profits df after {total_time:.2f} seconds")

    return profits_df



def clean_profits_df(profits_df, profitability_filter=10000000):
    """
    Clean the profits DataFrame by excluding all records for any coin_id-wallet_address pair
    if any single day's profitability exceeds the profitability_filter (positive or negative).
    
    Parameters:
    - profits_df: DataFrame with columns ['coin_id', 'wallet_address', 'date', 'profitability_cumulative']
    - profitability_filter: Threshold value to exclude pairs with profits or losses exceeding this value
    
    Returns:
    - Cleaned DataFrame with records for coin_id-wallet_address pairs filtered out.
    """
    logger.info("Starting generation of profits_cleaned_df...")
    start_time = time.time()

    # Identify coin_id-wallet_address pairs where any single day's profitability exceeds the threshold
    exclusions_profits_df = profits_df[
        (profits_df['profitability_cumulative'] > profitability_filter) |
        (profits_df['profitability_cumulative'] < -profitability_filter)
    ][['coin_id', 'wallet_address']].drop_duplicates()

    # Merge to filter out the records with those pairs
    profits_cleaned_df = profits_df.merge(exclusions_profits_df, on=['coin_id', 'wallet_address'], how='left', indicator=True)

    # Keep only the records where the pair was not in the exclusion list
    profits_cleaned_df = profits_cleaned_df[profits_cleaned_df['_merge'] == 'left_only']

    # Drop the merge indicator column
    profits_cleaned_df.drop(columns=['_merge'], inplace=True)

    total_time = time.time() - start_time
    logger.info("Finished cleaning profits_df after %.2f seconds. Removed %s coin-wallet pairs that breached profit or loss threshold of $%s",
        total_time, exclusions_profits_df.shape[0], dc.human_format(profitability_filter))

    return profits_cleaned_df,exclusions_profits_df



def classify_sharks(profits_df, *, profitability_threshold, modeling_period_start, balance_threshold):
    """
    Classify wallets as sharks based on lifetime profitability during the progeny period and USD balance thresholds.
    
    Parameters:
    - profits_df: DataFrame with columns ['coin_id', 'wallet_address', 'date', 'profitability_cumulative', 'balance', 'price']
    - profitability_threshold: Threshold to classify a wallet as a shark based on lifetime profitability at the end of the progeny period
    - modeling_period_start: Date after which data should be excluded from the training set (format: 'YYYY-MM-DD')
    - balance_threshold: Minimum USD balance required to be considered a shark
    
    Returns:
    - DataFrame with an additional column 'is_shark' indicating if the wallet is classified as a shark.
    """
    logger.info('identifying shark wallets...')

    # Filter out data from the period
    modeling_period_start = pd.to_datetime(modeling_period_start)
    filtered_profits_df = profits_df[profits_df['date'] < modeling_period_start].copy()

    # Calculate USD transfers and balances
    filtered_profits_df['usd_balance'] = filtered_profits_df['balance'] * filtered_profits_df['price']
    filtered_profits_df['usd_net_transfers'] = filtered_profits_df['net_transfers'] * filtered_profits_df['price']

    # Filter out wallets that have never reached the minimum USD balance_threshold for a coin_id
    eligible_wallets_df = filtered_profits_df.groupby(['coin_id', 'wallet_address'])['usd_balance'].max().reset_index()
    eligible_wallets_df = eligible_wallets_df[eligible_wallets_df['usd_balance'] >= balance_threshold]

    # Get the last profitability for each wallet-coin pair before the modeling period start
    last_profitability = filtered_profits_df.sort_values('date').groupby(['coin_id', 'wallet_address']).last()['profitability_cumulative'].reset_index()

    # Calculate rate of return
    filtered_profits_df['usd_total_inflows'] = filtered_profits_df['net_transfers'].where(filtered_profits_df['net_transfers'] > 0, 0)
    filtered_profits_df['rate_of_return'] = filtered_profits_df['profitability_cumulative'] / filtered_profits_df['usd_total_inflows']

    # Classify wallets as sharks based on lifetime profitability and eligibility
    sharks_df = eligible_wallets_df.merge(last_profitability, on=['coin_id', 'wallet_address'])
    sharks_df['is_shark'] = sharks_df['profitability_cumulative'] >= profitability_threshold

    logger.info('creation of sharks_df complete.')

    return sharks_df
