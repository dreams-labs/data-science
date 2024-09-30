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

# project file imports
from utils import timing_decorator  # pylint: disable=E0401 # can't find utils import


# set up logger at the module level
logger = dc.setup_logger()


def retrieve_market_data():
    """
    Retrieves market data from the core.coin_market_data table and converts coin_id to categorical

    Returns:
    - market_data_df: DataFrame containing market data with 'coin_id' as a categorical column.
    """
    start_time = time.time()
    logger.debug('Retrieving market data...')


    # SQL query to retrieve market data
    query_sql = '''
        select cmd.coin_id
        ,date
        ,cast(cmd.price as float64) as price
        ,cast(cmd.volume as int64) as volume
        ,cast(coalesce(cmd.market_cap,cmd.fdv) as int64) as market_cap
        from core.coin_market_data cmd
        order by 1,2
    '''

    # Run the SQL query using dgc's run_sql method
    logger.debug('retrieving market data...')
    market_data_df = dgc().run_sql(query_sql)

    # Convert coin_id column to categorical to reduce memory usage
    market_data_df['coin_id'] = market_data_df['coin_id'].astype('category')

    # Downcast numeric columns to reduce memory usage
    market_data_df['price'] = pd.to_numeric(market_data_df['price'], downcast='float')
    market_data_df['volume'] = pd.to_numeric(market_data_df['volume'], downcast='integer')
    market_data_df['market_cap'] = pd.to_numeric(market_data_df['market_cap'], downcast='integer')

    # Dates as dates
    market_data_df['date'] = market_data_df['date'].dt.date
    market_data_df['date'] = pd.to_datetime(market_data_df['date'])

    logger.info('Retrieved market_data_df with %s unique coins and %s rows after %s seconds',
                len(set(market_data_df['coin_id'])),
                len(market_data_df),
                round(time.time()-start_time,1))

    return market_data_df



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

    # Convert column types to memory efficient ones
    transfers_df['coin_id'] = transfers_df['coin_id'].astype('category')
    transfers_df['wallet_address'] = transfers_df['wallet_address'].astype('category')
    transfers_df['wallet_address'] = transfers_df['wallet_address'].cat.codes.astype('uint32')
    transfers_df['date'] = pd.to_datetime(transfers_df['date'])

    # Attempt to downcast to float32 to reduce memory usage
    transfers_df['net_transfers'] = pd.to_numeric(transfers_df['net_transfers'], downcast='float')
    transfers_df['balance'] = pd.to_numeric(transfers_df['balance'], downcast='float')

    logger.info('Retrieved transfers_df with %s unique coins and %s rows after %s seconds',
                len(set(transfers_df['coin_id'])),
                len(transfers_df),
                round(time.time()-start_time,1))

    return transfers_df


@timing_decorator
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

    # merge datasets and confirm coin_id is categorical
    profits_df = pd.merge(transfers_df, prices_df, on=['coin_id', 'date'], how='inner')
    profits_df = recategorize_columns(profits_df)

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
    first_prices_df['coin_id'] = first_prices_df['coin_id'].astype('category')

    # Merge the first price data into profits_df
    profits_df = profits_df.merge(first_prices_df, on='coin_id', how='inner')
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
    grouped_pre_price_transfers = recategorize_columns(grouped_pre_price_transfers)

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
    if not new_records.empty:
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
    profits_df['token_inflows'] = profits_df['net_transfers'].clip(lower=0)
    profits_df['token_inflows_cumulative'] = profits_df.groupby(['coin_id', 'wallet_address'],observed=True)['token_inflows'].cumsum()
    profits_df = recategorize_columns(profits_df)

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

    return profits_df



@timing_decorator
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
    start_time = time.time()

    # Raise an error if there are any missing prices
    if profits_df['price'].isnull().any():
        raise ValueError("Missing prices found for transfer dates which should not be possible.")

    # create offset price and balance rows to easily calculate changes between periods
    profits_df['previous_price'] = profits_df.groupby(['coin_id', 'wallet_address'], observed=True)['price'].shift(1)
    profits_df['previous_price'] = profits_df['previous_price'].fillna(profits_df['price'])
    profits_df = recategorize_columns(profits_df)

    profits_df['previous_balance'] = profits_df.groupby(['coin_id', 'wallet_address'], observed=True)['balance'].shift(1).fillna(0)
    profits_df = recategorize_columns(profits_df)

    logger.debug(f"Offset prices and balances for profitability logic: {time.time() - start_time:.2f} seconds")
    step_time = time.time()

    # calculate the profitability change in each period and sum them to get cumulative profitability
    profits_df['profits_change'] = (profits_df['price'] - profits_df['previous_price']) * profits_df['previous_balance']
    profits_df['profits_cumulative'] = profits_df.groupby(['coin_id', 'wallet_address'], observed=True)['profits_change'].cumsum()
    profits_df = recategorize_columns(profits_df)

    logger.debug(f"Calculate profitability: {time.time() - step_time:.2f} seconds")
    step_time = time.time()

    # Calculate USD inflows, balances, and rate of return
    profits_df['usd_balance'] = profits_df['balance'] * profits_df['price']
    profits_df['usd_net_transfers'] = profits_df['net_transfers'] * profits_df['price']
    profits_df['usd_inflows'] = profits_df['usd_net_transfers'].where(profits_df['usd_net_transfers'] > 0, 0)
    profits_df['usd_inflows_cumulative'] = profits_df.groupby(['coin_id', 'wallet_address'], observed=True)['usd_inflows'].cumsum()
    profits_df['total_return'] = (
        profits_df['profits_cumulative']
        / profits_df['usd_inflows_cumulative']
        .where(profits_df['usd_inflows_cumulative'] != 0, np.nan)
    )

    # Final recategorization
    profits_df = recategorize_columns(profits_df)

    logger.debug(f"Calculate rate of return {time.time() - step_time:.2f} seconds")
    step_time = time.time()

    # Drop helper columns
    profits_df.drop(columns=['previous_price', 'previous_balance'], inplace=True)

    return profits_df

def recategorize_columns(df):
    """
    Helper function to reoptimize column dtypes after groupby operations. This can reduce
    memory usage by up to 75% in some cases.

    Params: df (DataFrame): dataframe with wallet_address and coin_id columns
    Returns: df (DataFrame): dataframe with wallet_address and coin_id columns formatted
        as categories
    """
    recat_time = time.time()
    df['coin_id'] = df['coin_id'].astype('category')
    df['wallet_address'] = df['wallet_address'].astype('category')
    df['wallet_address'] = df['wallet_address'].cat.codes.astype('uint32')
    df['date'] = df['date'].dt.date
    logger.debug(f"Recategorization took {time.time() - recat_time:.2f} seconds.")

    return df


@timing_decorator
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

    # 1. Remove wallets with higher or lower total profits than the profitability_filter
    # ----------------------------------------------------------------------------------
    # Group by wallet_address and calculate the total profitability
    wallet_profits_agg_df = profits_df.groupby('wallet_address', observed=True)['profits_change'].sum().reset_index()

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
    wallet_inflows_agg_df = profits_df.groupby('wallet_address', observed=True)['usd_inflows'].sum().reset_index()

    # Identify wallet addresses where total inflows exceed the threshold
    exclusions_inflows_df = wallet_inflows_agg_df[
        wallet_inflows_agg_df['usd_inflows'] >= data_cleaning_config['inflows_filter']
    ][['wallet_address']]

    # Merge to filter out the records with those wallet addresses
    profits_cleaned_df = profits_cleaned_df.merge(exclusions_inflows_df, on='wallet_address', how='left', indicator=True)
    profits_cleaned_df = profits_cleaned_df[profits_cleaned_df['_merge'] == 'left_only']
    profits_cleaned_df.drop(columns=['_merge'], inplace=True)

    # Convert coin_id to categorical
    profits_df['coin_id'] = profits_df['coin_id'].astype('category')
    profits_df['wallet_address'] = profits_df['wallet_address'].astype('category')

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
    logger.debug("Identified %s coin-wallet pairs beyond profit threshold of $%s and %s pairs beyond inflows filter of %s.",
        exclusions_profits_df.shape[0], dc.human_format(data_cleaning_config['profitability_filter']),
        exclusions_inflows_df.shape[0], dc.human_format(data_cleaning_config['inflows_filter'])
    )

    return profits_cleaned_df,exclusions_logs_df



def retrieve_metadata_data():
    """
    Retrieves metadata from the core.coin_facts_metadata table.

    Returns:
    - metadata_df: DataFrame containing coin_id-keyed metadata
    """
    # SQL query to retrieve prices data
    query_sql = '''
        select c.coin_id
        ,md.categories
        ,c.chain
        from core.coins c
        join core.coin_facts_metadata md on md.coin_id = c.coin_id
    '''

    # Run the SQL query using dgc's run_sql method
    logger.debug('retrieving metadata data...')
    metadata_df = dgc().run_sql(query_sql)

    logger.info('retrieved metadata_df with shape %s',metadata_df.shape)

    return metadata_df



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
        col for col in metadata_df.columns if col.startswith('category_') or col.startswith('chain_')
    ]
    metadata_features_df = metadata_df[columns_to_keep].copy()
    logger.info("Generated coin_metadata_features_df.")

    return metadata_features_df



def retrieve_google_trends_data():
    """
    Retrieves Google Trends data from the macro_trends dataset. Because the data is weekly, it also
    resamples to daily records using linear interpolation.

    Returns:
    - google_trends_df: DataFrame keyed on date containing Google Trends values for multiple terms
    """
    query_sql = '''
        select *
        from `macro_trends.google_trends`
        order by date
    '''

    # Run the SQL query using dgc's run_sql method
    google_trends_df = dgc().run_sql(query_sql)
    logger.info('retrieved Google Trends data with shape %s',google_trends_df.shape)

    # Convert the date column to datetime format
    google_trends_df['date'] = pd.to_datetime(google_trends_df['date'])

    # Resample the df to fill in missing days by using date as the index
    google_trends_df.set_index('date', inplace=True)
    google_trends_df = google_trends_df.resample('D').interpolate(method='linear')
    google_trends_df.reset_index(inplace=True)

    return google_trends_df
