"""
functions used in generating training data for the models
"""
# pylint: disable=W1203 # fstrings in logs
# pylint: disable=C0301 # line over 100 chars
import time
import pandas as pd
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
    query_sql = """
        select cmd.coin_id
        ,date
        ,cast(cmd.price as float64) as price
        ,cast(cmd.volume as int64) as volume
        ,cast(coalesce(cmd.market_cap,cmd.fdv) as int64) as market_cap
        from core.coin_market_data cmd
        order by 1,2
    """

    # Run the SQL query using dgc's run_sql method
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



def retrieve_profits_data(start_date,end_date):
    """
    Retrieves data from the core.coin_wallet_profits table and converts columns to
    memory-efficient formats. Records prior to the start_date are excluded but a new
    row is imputed for the coin_id-wallet_address pair that summarizes their historical
    performance (see Step 2 CTEs).

    Params:
    - start_date (String): The earliest date to retrieve records for with format 'YYYY-MM-DD'
    - start_date (String): The latest date to retrieve records for with format 'YYYY-MM-DD'

    Returns:
    - profits_df (DataFrame): contains coin-wallet-date keyed profits data denominated in USD
    - wallet_address_mapping (pandas Index): mapping that will allow us to convert wallet_address
        back to the original strings, rather than the integer values they get mapped to for
        memory optimization
    """
    start_time = time.time()
    logger.info('Retrieving profits data...')

    # SQL query to retrieve profits data
    query_sql = f"""
        -- STEP 1: retrieve profits data through the end of the modeling period
        -----------------------------------------------------------------------
        with profits_base as (
            select coin_id
            ,date
            ,wallet_address
            ,profits_change
            ,profits_cumulative
            ,usd_balance
            ,usd_net_transfers
            ,usd_inflows
            ,usd_inflows_cumulative
            ,total_return
            from core.coin_wallet_profits
            where date <= '{end_date}'
        ),

        -- STEP 2: create new records for all coin-wallet pairs as of the training_period_start
        ---------------------------------------------------------------------------------------
        -- compute the starting profits and balances as of the training_period_start
        training_start_existing_rows as (
            -- identify coin-wallet pairs that already have a balance as of the period end
            select *
            from profits_base
            where date = '{start_date}'
        ),
        training_start_needs_rows as (
            -- for coin-wallet pairs that don't have existing records, identify the row closest to the period end date
            select t.*
            ,cmd_previous.price as price_previous
            ,cmd_training.price as price_current
            ,row_number() over (partition by t.coin_id,t.wallet_address order by t.date desc) as rn
            from profits_base t
            left join training_start_existing_rows e on e.coin_id = t.coin_id
                and e.wallet_address = t.wallet_address

            -- obtain the last price used to compute the balance and profits data
            join core.coin_market_data cmd_previous on cmd_previous.coin_id = t.coin_id and cmd_previous.date = t.date

            -- obtain the training_period_start price so we can update the calculations
            join core.coin_market_data cmd_training on cmd_training.coin_id = t.coin_id and cmd_training.date = '{start_date}'
            where t.date < '{start_date}'
            and e.coin_id is null
        ),
        training_start_new_rows as (
            -- create a new row for the period end date by carrying the balance from the closest existing record
            select t.coin_id
            ,cast('{start_date}' as datetime) as date
            ,t.wallet_address

            -- profits_change is the USD change in price minus the previous usd_balance
            ,((t.price_current / t.price_previous) * t.usd_balance) - t.usd_balance as profits_change
            -- profits_cumulative is the previous profits_cumulative + profits_change
            ,((t.price_current / t.price_previous) * t.usd_balance) - t.usd_balance + t.profits_cumulative as profits_cumulative
            -- usd_balance is 1+% change in price times previous balance
            ,(t.price_current / t.price_previous) * t.usd_balance as usd_balance
            -- there were no transfers
            ,0 as usd_net_transfers
            -- there were no inflows
            ,0 as usd_inflows
            -- no change since there were no inflows
            ,usd_inflows_cumulative as usd_inflows_cumulative
            -- total_return is profits_cumumlative / usd_inflows_cumulative
            ,(((t.price_current / t.price_previous) * t.usd_balance) - t.usd_balance + t.profits_cumulative) -- profits cumulative
            / usd_inflows_cumulative
            as total_return

            from training_start_needs_rows t
            where rn=1
            and usd_balance > 0
        ),

        -- STEP 3: merge all records together and round relevant columns
        ----------------------------------------------------------------
        profits_merged as (
            select * from profits_base
            -- transfers prior to the training period are summarized in training_start_new_rows
            where date >= '{start_date}'

            union all

            select * from training_start_new_rows
        )

        -- round values before exporting to reduce memory usage
        select coin_id
        ,date
        -- replace the memory-intensive address strings with integers
        ,DENSE_RANK() OVER (ORDER BY wallet_address) as wallet_address
        ,round(profits_change,2) as profits_change
        ,round(profits_cumulative,2) as profits_cumulative
        ,round(usd_balance,2) as usd_balance
        ,round(usd_net_transfers,2) as usd_net_transfers
        ,round(usd_inflows,2) as usd_inflows
        -- set a floor of $0.01 to avoid divide by 0 errors caused by rounding
        ,greatest(0.01,round(usd_inflows_cumulative,2)) as usd_inflows_cumulative
        ,round(total_return,5) as total_return
        from profits_merged
    """

    # Run the SQL query using dgc's run_sql method
    profits_df = dgc().run_sql(query_sql)

    logger.info('Converting columns to memory-optimized formats...')

    # Convert coin_id to categorical and date to date
    profits_df['coin_id'] = profits_df['coin_id'].astype('category')
    profits_df['date'] = pd.to_datetime(profits_df['date'])

    # Convert all numerical columns to float32
    profits_df['wallet_address'] = profits_df['wallet_address'].astype('float32')
    profits_df['profits_change'] = profits_df['profits_change'].astype('float32')
    profits_df['profits_cumulative'] = profits_df['profits_cumulative'].astype('float32')
    profits_df['usd_balance'] = profits_df['usd_balance'].astype('float32')
    profits_df['usd_net_transfers'] = profits_df['usd_net_transfers'].astype('float32')
    profits_df['usd_inflows'] = profits_df['usd_inflows'].astype('float32')
    profits_df['usd_inflows_cumulative'] = profits_df['usd_inflows_cumulative'].astype('float32')
    profits_df['total_return'] = profits_df['total_return'].astype('float32')

    logger.info('Retrieved profits_df with %s unique coins and %s rows after %.2f seconds',
                len(set(profits_df['coin_id'])),
                len(profits_df),
                time.time()-start_time)

    return profits_df



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
