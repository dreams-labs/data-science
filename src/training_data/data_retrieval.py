"""
functions used in generating training data for the models
"""
import sys
import time
import logging
import pandas as pd
from dreams_core.googlecloud import GoogleCloud as dgc
from dreams_core import core as dc

# import coin_wallet_metrics as cwm
sys.path.append('..')
import utils as u  # pylint: disable=C0413  # import must be at top

# Set up logger at the module level
logger = logging.getLogger(__name__)




# ------------------------------------
#     Primary Interface Functions
# ------------------------------------

def retrieve_market_data(end_date: str, dataset='prod'):
    """
    Retrieves market data from the core.coin_market_data table and converts coin_id to categorical
    Params:
    - end_date (str): YYYY-MM-DD string for the last day data should be retrieved
    - dataset (str): 'prod' targets core dataset, 'dev' targets dev_core dataset

    Returns:
    - market_data_df: DataFrame containing market data with 'coin_id' as a categorical column.
    """
    start_time = time.time()

    if dataset == 'prod':
        core_dataset = 'core'
    elif dataset == 'dev':
        core_dataset = 'dev_core'
    else:
        raise ValueError("Invalid dataset parameter '%s', dataset must be 'dev' or 'prod'.")

    # SQL query to retrieve market data
    logger.info(f"Retrieving market data from {dataset} dataset '{core_dataset}'...")
    query_sql = f"""
        select cmd.coin_id
        ,date
        ,cast(cmd.price as float64) as price
        ,cast(cmd.volume as int64) as volume
        ,cast(cmd.market_cap as int64) as market_cap
        ,days_imputed
        from {core_dataset}.coin_market_data cmd
        where date <= '{end_date}'
        order by 1,2
    """
    # Run the SQL query using dgc's run_sql method
    market_data_df = dgc().run_sql(query_sql)
    logger.debug('Base market data retrieved, beginning memory optimized formatting...')

    # Convert coin_id column to categorical to reduce memory usage
    market_data_df['coin_id'] = market_data_df['coin_id'].astype('category')

    # Downcast numeric columns to reduce memory usage
    market_data_df = u.df_downcast(market_data_df)

    # Dates as dates
    market_data_df['date'] = pd.to_datetime(market_data_df['date'])

    # Check for negative values in 'price', 'volume', and 'market_cap'
    if (market_data_df['price'] < 0).any():
        raise ValueError("Negative values found in 'price' column.")
    if (market_data_df['volume'] < 0).any():
        raise ValueError("Negative values found in 'volume' column.")
    if (market_data_df['market_cap'] < 0).any():
        raise ValueError("Negative values found in 'market_cap' column.")

    logger.info('Retrieved market_data_df with %s unique coins and %s rows after %s seconds',
                len(set(market_data_df['coin_id'])),
                len(market_data_df),
                round(time.time()-start_time,1))

    return market_data_df


def retrieve_profits_data(start_date, end_date, min_wallet_inflows, dataset='prod'):
    """
    Retrieves data from the core.coin_wallet_profits table and converts columns to
    memory-efficient formats. Records prior to the start_date are excluded but a new
    row is imputed for the coin_id-wallet_address pair that summarizes their historical
    performance (see Step 2 CTEs).

    Params:
    - start_date (String): The earliest date to retrieve records for with format 'YYYY-MM-DD'
    - start_date (String): The latest date to retrieve records for with format 'YYYY-MM-DD'
    - min_wallet_inflows (Float): Wallets with fewer than this amount of total USD inflows
        across all coins will be removed from the profits_df dataset
    - dataset (String): determines bigquery dataset. 'prod' targets 'core._'; 'dev' targets 'dev_core._'


    Returns:
    - profits_df (DataFrame): contains coin-wallet-date keyed profits data denominated in USD
    - wallet_address_mapping (pandas Index): mapping that will allow us to convert wallet_address
        back to the original strings, rather than the integer values they get mapped to for
        memory optimization
    """
    start_time = time.time()

    if dataset == 'prod':
        core_dataset = 'core'
    elif dataset == 'dev':
        core_dataset = 'dev_core'
    else:
        raise ValueError("Invalid dataset parameter '%s', dataset must be 'dev' or 'prod'.")


    # SQL query to retrieve profits data
    logger.info(f"Retrieving profits data from {dataset} dataset '{core_dataset}'...")

    query_sql = f"""
        -- STEP 1: retrieve profits data and apply USD inflows filter
        -------------------------------------------------------------
        with profits_full as (
            select coin_id
            ,date
            ,wallet_address
            ,profits_cumulative
            ,usd_balance
            ,usd_net_transfers
            ,usd_inflows
            ,usd_inflows_cumulative
            from {core_dataset}.coin_wallet_profits
            where date <= '{end_date}'
        ),

        usd_inflows_filter as (
            select coin_id
            ,wallet_address
            ,max(usd_inflows_cumulative) as total_usd_inflows
            from profits_full
            group by 1,2
        ),

        profits_base as (
            select pf.*
            from profits_full pf

            -- filter to remove wallet-coin pairs below the min_wallet_inflows
            join usd_inflows_filter f on f.coin_id = pf.coin_id
                and f.wallet_address = pf.wallet_address
            where f.total_usd_inflows >= {min_wallet_inflows}
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
            join {core_dataset}.coin_market_data cmd_previous on cmd_previous.coin_id = t.coin_id and cmd_previous.date = t.date

            -- obtain the training_period_start price so we can update the calculations
            join {core_dataset}.coin_market_data cmd_training on cmd_training.coin_id = t.coin_id and cmd_training.date = '{start_date}'
            where t.date < '{start_date}'
            and e.coin_id is null
        ),
        training_start_new_rows as (
            -- create a new row for the period end date by carrying the balance from the closest existing record
            select t.coin_id
            ,cast('{start_date}' as datetime) as date
            ,t.wallet_address
            -- profits_cumulative is the previous profits_cumulative + the change in profits up to the start_date
            ,((t.price_current / t.price_previous) - 1) * t.usd_balance + t.profits_cumulative as profits_cumulative
            -- usd_balance is previous balance * (1 + % change in price)
            ,(t.price_current / t.price_previous) * t.usd_balance as usd_balance
            -- there were no transfers
            ,0 as usd_net_transfers
            -- there were no inflows
            ,0 as usd_inflows
            -- no change since there were no inflows
            ,usd_inflows_cumulative as usd_inflows_cumulative

            from training_start_needs_rows t
            where rn=1

        ),

        -- STEP 3: merge all records together
        -------------------------------------
        profits_merged as (
            select *,
            False as is_imputed
            from profits_base
            -- transfers prior to the training period are summarized in training_start_new_rows
            where date >= '{start_date}'

            union all

            select *,
            True as is_imputed
            from training_start_new_rows
        )

        select coin_id
        ,date

        -- replace the memory-intensive address strings with integers
        ,id.wallet_id as wallet_address

        ,profits_cumulative
        ,usd_balance
        ,usd_net_transfers
        ,usd_inflows
        -- set a floor of $0.01 to avoid divide by 0 errors caused by rounding
        ,greatest(0.01,usd_inflows_cumulative) as usd_inflows_cumulative
        ,is_imputed
        from profits_merged pm
        join reference.wallet_ids id on id.wallet_address = pm.wallet_address
        order by coin_id, wallet_address, date
    """

    # Run the SQL query using dgc's run_sql method
    profits_df = dgc().run_sql(query_sql)

    logger.debug('Converting columns to memory-optimized formats...')

    # Convert coin_id to categorical and date to date
    profits_df['coin_id'] = profits_df['coin_id'].astype('category')
    profits_df['date'] = pd.to_datetime(profits_df['date'])

    # Convert all numerical columns to 32 bit, using safe_downcast to avoid overflow
    profits_df = u.df_downcast(profits_df)

    logger.info('Retrieved profits_df with %s unique coins and %s rows after %.2f seconds',
                len(set(profits_df['coin_id'])),
                len(profits_df),
                time.time()-start_time)

    return profits_df


def retrieve_macro_trends_data(query_sql = None):
    """
    Retrieves Google Trends data from the macro_trends dataset. Because the data is weekly, it also
    resamples to daily records using linear interpolation.

    Params:
    - override_sql (str): Overrides the base query. Must have column 'date'.

    Returns:
    - google_trends_df: DataFrame keyed on date containing Google Trends values for multiple terms
    """
    if query_sql is None:
        # query to retrieve all macro trends data at once
        query_sql = """
            with all_dates as (
                select date from `macro_trends.bitcoin_indicators`
                union distinct
                select date from `macro_trends.crypto_global_market`
                union distinct
                select date from `macro_trends.google_trends`
            )
            select d.date
            ,bi.btc_price
            ,bi.mvrv_z_score as btc_mvrv_z_score
            ,bi.vdd_multiple as btc_vdd_multiple
            ,gm.market_cap as global_market_cap
            ,gm.total_volume as global_volume

            -- casts to ensure all are numeric because sometimes the output is "<1"
            --  todo: eventually this should be done when data is ingested
            ,cast(replace(cast(gt.altcoin_worldwide as string),'<','') as int64) as gtrends_altcoin_worldwide
            ,cast(replace(cast(gt.cryptocurrency_worldwide as string),'<','') as int64) as gtrends_cryptocurrency_worldwide
            ,cast(replace(cast(gt.solana_us as string),'<','') as int64) as gtrends_solana_us
            ,cast(replace(cast(gt.cryptocurrency_us as string),'<','') as int64) as gtrends_cryptocurrency_us
            ,cast(replace(cast(gt.bitcoin_us as string),'<','') as int64) as gtrends_bitcoin_us
            ,cast(replace(cast(gt.solana_worldwide as string),'<','') as int64) as gtrends_solana_worldwide
            ,cast(replace(cast(gt.coinbase_us as string),'<','') as int64) as gtrends_coinbase_us
            ,cast(replace(cast(gt.bitcoin_worldwide as string),'<','') as int64) as gtrends_bitcoin_worldwide
            ,cast(replace(cast(gt.altcoin_us as string),'<','') as int64) as gtrends_altcoin_us
            ,cast(replace(cast(gt.coinbase_worldwide as string),'<','') as int64) as gtrends_coinbase_worldwide
            ,cast(replace(cast(gt.memecoin_worldwide as string),'<','') as int64) as gtrends_memecoin_worldwide
            ,cast(replace(cast(gt.memecoin_us as string),'<','') as int64) as gtrends_memecoin_us
            from all_dates d
            left join `macro_trends.bitcoin_indicators` bi on bi.date = d.date
            left join `macro_trends.crypto_global_market` gm on gm.date = d.date
            left join `macro_trends.google_trends` gt on gt.date = d.date
            order by date desc
            """

    # Run the SQL query using dgc's run_sql method
    logger.info("Retrieving macro trends data from prod schema 'macro_trends'...")
    macro_trends_df = dgc().run_sql(query_sql)
    logger.info('Retrieved macro trends data with shape %s',macro_trends_df.shape)

    # Create sorted datetime index
    macro_trends_df['date'] = pd.to_datetime(macro_trends_df['date'])
    macro_trends_df = macro_trends_df.set_index('date')
    macro_trends_df = macro_trends_df.sort_index()

    # Convert all numerical columns to 32 bit, using safe_downcast to avoid overflow
    macro_trends_df = u.df_downcast(macro_trends_df)

    return macro_trends_df



# -----------------------------------
#        Market Data Helpers
# -----------------------------------

def detect_price_data_staleness(market_data_df, config):
    """
    Find concerning patterns in data freshness where imputed prices count rises and stays
    high. Issues a logger.warning if issues above the data cleaning flags are detected.

    Params:
    - market_data_df (DataFrame): Dataframe with dates and days_imputed column
    - lookback_days (int): Number of days to look back for minimum value

    Returns:
    - bool: True if concerning staleness pattern detected
    """
    # Define thresholds
    count_threshold = config['data_cleaning']['price_coverage_warning_min_coin_increase']
    percent_threshold = config['data_cleaning']['price_coverage_warning_min_pct_increase']
    audit_window=config['data_cleaning']['coverage_decrease_audit_window']


    # Summarize as daily records
    df = market_data_df.groupby('date')['days_imputed'].count().to_frame()
    total_coins = market_data_df.groupby('date')['coin_id'].nunique()

    # Convert index to datetime if not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(pd.to_datetime(df.index))

    # Get most recent date and lookback period
    latest_date = df.index.max()
    lookback_date = latest_date - pd.Timedelta(days=audit_window)

    # Get relevant values
    latest_count = df.loc[latest_date, 'days_imputed']
    min_date = df.loc[lookback_date:latest_date, 'days_imputed'].idxmin()
    recent_min = df.loc[min_date, 'days_imputed']
    latest_total = total_coins.loc[latest_date]

    # Calculate the increases
    count_increase = latest_count - recent_min
    pct_increase = (count_increase / recent_min) if recent_min > 0 else float('inf')

    # Calculate percentages
    latest_pct = (latest_count / latest_total)
    min_pct = (recent_min / total_coins.loc[min_date])

    if count_increase > count_threshold and pct_increase > percent_threshold:
        logging.warning(
            f"Price data freshness issue detected on {latest_date.date()}:\n"
            f"- {count_increase:.0f} records have become stale since {min_date.date()}.\n"
            f"- Imputed records increased from {min_pct*100:.1f}% ({recent_min} coins) on {min_date.date()} to "
            f"{latest_pct*100:.1f}% ({latest_count:.0f} coins) on {latest_date.date()}."
        )
        return True

    return False


def clean_market_data(market_data_df, config, earliest_date, latest_date):
    """
    Removes all market data records for
        1. coins with a longer price gap than the config['data_cleaning']['max_gap_days']
        2. coins with lower daily mean volume than the minimum_daily_volume between the
            earliest_window_start and the end of the last modeling period.

    Note: Does NOT filter records based on date, only filters on gap days and volume.

    Params:
    - market_data_df (DataFrame): DataFrame containing market data as well as the
        days_imputed column, which represents the number of days a real price has been
        forwardfilled in a row to ensure a complete time series.
    - config (dict): Config with data cleaning parmas for market data
    - earliest_date (string): The earliest date relevant to the dataset formatted as
        'YYYY-MM-DD'. Data cleaning filters will not be applied to records prior to this.
    - latest_date (string): The latest date relevant to the dataset formatted as
        'YYYY-MM-DD'. Data cleaning filters will not be applied to records after this.
    """
    # Declare thresholds
    max_gap_days = config['data_cleaning']['max_gap_days']
    min_daily_volume = config['data_cleaning']['min_daily_volume']

    # Create mask for evaluation period
    date_mask = ((market_data_df['date'] >= earliest_date) &
                 (market_data_df['date'] <= latest_date))

    # Identify coin_ids with gaps that exceed the maximum within date range
    filtered_df = market_data_df[date_mask]
    gap_coin_ids = (filtered_df.groupby('coin_id', observed=True)['days_imputed']
                    .max()[lambda x: x > max_gap_days].index.tolist())

    # Identify coin_ids with insufficient volume within date range
    mean_volume = filtered_df.groupby('coin_id', observed=True)['volume'].mean()
    low_volume_coins = mean_volume[mean_volume < min_daily_volume].index.tolist()

    # Combine problematic coin lists
    coins_to_remove = list(set(gap_coin_ids + low_volume_coins))

    # Remove ALL records for problematic coins
    cleaned_df = market_data_df[~market_data_df['coin_id'].isin(coins_to_remove)]

    logger.info("Removed %s coins (%s for gaps, %s for volume) and %s total records.",
                len(coins_to_remove),
                len(gap_coin_ids),
                len(low_volume_coins),
                len(market_data_df) - len(cleaned_df))

    # Assess potential staleness of prices based on imputation trends
    _ = detect_price_data_staleness(cleaned_df[cleaned_df['date'] <= latest_date],config)

    # Drop imputation lineage column
    cleaned_df = cleaned_df.drop(columns='days_imputed')

    return cleaned_df


def impute_market_cap(market_data_df, min_coverage=0.7, max_multiple=1.0):
    """
    Create a new column with imputed market cap values based on price movements.
    Handles missing values at start, middle, and end of time series.

    Params:
    - market_data_df (DataFrame): DataFrame with columns ['coin_id', 'date', 'price', 'market_cap']
    - min_coverage (Float): Minimum coverage threshold (0-1) for coins to be processed
    - max_multiple (Float): The maximum multiple that an imputed market cap can reach relative to the
        maximum known market cap. Rows exceeding this threshold will be set to np.nan

    Returns:
    - df_copy (DataFrame): DataFrame with new 'market_cap_imputed' column containing original and
        imputed values as int64
    """
    # Make a copy of input data
    df_copy = market_data_df.copy()
    df_copy = df_copy.sort_values(['coin_id','date'])

    # Calculate coverage and historical maximums per coin
    coverage = df_copy.groupby('coin_id', observed=True).agg(
        records=('price', 'count'),
        has_cap=('market_cap', 'count'),
        max_cap=('market_cap', 'max')
    )
    coverage['coverage'] = coverage['has_cap'] / coverage['records']

    # Get eligible coins
    eligible_coins = coverage[
        (coverage['coverage'] >= min_coverage) &
        (coverage['coverage'] < 1)
    ].index

    # Initialize imputed column with original values as int64
    df_copy['market_cap_imputed'] = df_copy['market_cap'].astype('Int64')

    # Process only eligible coins
    mask_eligible = df_copy['coin_id'].isin(eligible_coins)

    # Calculate ratio for all valid records of eligible coins
    df_copy.loc[mask_eligible, 'ratio'] = (
        df_copy.loc[mask_eligible, 'market_cap'] /
        df_copy.loc[mask_eligible, 'price']
    )

    # Backfill and forward fill ratios within each coin group
    df_copy['ratio'] = df_copy.groupby('coin_id',observed=True)['ratio'].bfill()
    df_copy['ratio'] = df_copy.groupby('coin_id',observed=True)['ratio'].ffill()

    # Calculate imputed market caps using the filled ratios
    mask_missing = df_copy['market_cap_imputed'].isna() & mask_eligible
    df_copy.loc[mask_missing, 'market_cap_imputed'] = (
        (df_copy.loc[mask_missing, 'price'] *
         df_copy.loc[mask_missing, 'ratio']).round().astype('Int64')
    )

    # Join max historical values and apply max_multiple check vectorized
    df_copy = df_copy.merge(
        coverage[['max_cap']],
        left_on='coin_id',
        right_index=True,
        how='left'
    )

    # Set imputed values exceeding max_multiple * historical max to np.nan
    mask_exceeds_max = (
        df_copy['market_cap_imputed'] >
        (df_copy['max_cap'] * max_multiple)
    )
    df_copy.loc[mask_exceeds_max, 'market_cap_imputed'] = pd.NA

    # Drop temporary columns
    df_copy = df_copy.drop(['ratio', 'max_cap'], axis=1)

    # Logger calculations and output
    all_rows = len(df_copy)
    known = df_copy['market_cap'].count()
    imputed = df_copy['market_cap_imputed'].count()
    logger.info("Imputation increased market cap coverage by %.1f%% to %.1f%% (%s/%s) vs base of %.1f%% (%s/%s)",
                100*(imputed-known)/all_rows,
                100*imputed/all_rows, dc.human_format(imputed), dc.human_format(all_rows),
                100*known/all_rows, dc.human_format(known), dc.human_format(all_rows))


    return df_copy





# ------------------------------------
#        Profits Data Helpers
# ------------------------------------

def check_coin_transfers_staleness(profits_df, data_cleaning_config) -> None:
    """
    Warns if recent counts of coins with transfers data has dramatically decreased
    in recent periods, which could indicate partial data staleness.

    e.g. if  Dune hasn't been updated in 5 days but Ethereum chain transfers have,
    catastrophic training data inconsistencies will be created.

    Params:
    - profits_df (df): df showing coin-wallet-date records where transers exist
    """
    # Extract thresholds
    count_threshold=data_cleaning_config['transfers_coverage_warning_min_coin_increase'],
    percent_threshold=data_cleaning_config['transfers_coverage_warning_min_pct_increase']
    audit_window=data_cleaning_config['coverage_decrease_audit_window']

    # Create counts of coins with transfers
    daily_counts = profits_df.groupby('date')['coin_id'].nunique().copy()
    latest_count = daily_counts.iloc[-1]
    latest_date = daily_counts.index[-1]
    cutoff_date = latest_date - pd.Timedelta(days=audit_window)
    week_data = daily_counts.loc[cutoff_date:]

    count_decrease = week_data.max() - latest_count
    min_date = week_data.idxmin()
    pct_decrease = count_decrease / week_data.max() * 100

    if count_decrease > count_threshold and pct_decrease > percent_threshold:
        logging.warning(
            f"Transfers data coverage alert on {latest_date.date()}:\n"
            f"- {count_decrease:.0f} coins have become stale since {min_date.date()}."
            f"- Transfers coverage decreased from {week_data.max():.0f} coins ({min_date.date()}) "
            f"to {latest_count:.0f} coins ({latest_date.date()}), "
            f"a {pct_decrease:.1f}% decrease."
        )



def clean_profits_df(profits_df, data_cleaning_config):
    """
    Clean the profits DataFrame by excluding all records for any wallet_addresses that
    have aggregate USD inflows above the max_wallet_coin_inflows.
    The goal is to filter outliers such as minting/burning/contract addresses.

    Parameters:
    - profits_df: DataFrame with columns 'coin_id', 'wallet_address', 'date', 'usd_inflows_cumulative'
    - data_cleaning_config:
        - max_wallet_coin_inflows: exclude wallets with total all time USD inflows above this level

    Returns:
    - Cleaned DataFrame with records for coin_id-wallet_address pairs filtered out.
    """
    # 0. Check coin coverage to see if transfers data may be stale
    # ------------------------------------------------------------
    check_coin_transfers_staleness(profits_df,data_cleaning_config)


    # 1. Calculate total inflows for each wallet across all coins
    # -----------------------------------------------------------
    # Group by wallet-coin pair and calculate the ending total inflows
    wallet_coin_agg_df = (profits_df.sort_values('date')
                                    .groupby(['wallet_address', 'coin_id'], observed=True)
                                    .agg({
                                        'usd_inflows_cumulative': 'last'
                                    })
                                    .reset_index())

    # Sum the inflows of all coins associated with each wallet
    wallet_agg_df = (wallet_coin_agg_df.groupby('wallet_address')
                                       .agg({
                                           'usd_inflows_cumulative': 'sum'
                                       })
                                       .reset_index())

    # 2. Identify the wallets to be excluded
    # --------------------------------------
    # Identify wallet addresses where total inflows exceed the threshold
    wallet_exclusions_inflows = wallet_agg_df[
        wallet_agg_df['usd_inflows_cumulative'] >= data_cleaning_config['max_wallet_inflows']
    ]['wallet_address']

    # Remove records from profits_df where wallet_address is in the exclusion list
    profits_cleaned_df = profits_df[~profits_df['wallet_address'].isin(wallet_exclusions_inflows)]

    # 3. Prepare exclusions_df and output logs
    # ----------------------------------------
    # prepare exclusions_logs_df
    exclusions_logs_df = pd.DataFrame({
        'wallet_address': wallet_exclusions_inflows.unique()
    })
    exclusions_logs_df['inflows_exclusion'] = True

    # log outputs
    logger.debug("Identified %s wallets exceeding inflows filter of $%s.",
                 len(wallet_exclusions_inflows),
                 dc.human_format(data_cleaning_config['max_wallet_inflows']))

    return profits_cleaned_df, exclusions_logs_df


def retrieve_metadata_data():
    """
    Retrieves metadata from the core.coin_facts_metadata table.

    Returns:
    - metadata_df: DataFrame containing coin_id-keyed metadata
    """
    # SQL query to retrieve prices data
    query_sql = """
        select c.coin_id
        ,md.categories
        ,c.chain
        from core.coins c
        join core.coin_facts_metadata md on md.coin_id = c.coin_id
    """

    # Run the SQL query using dgc's run_sql method
    logger.debug('retrieving metadata data...')
    metadata_df = dgc().run_sql(query_sql)

    logger.info('Retrieved metadata_df with shape %s',metadata_df.shape)

    return metadata_df






# ----------------------------------
#       Macro Trends Helpers
# ----------------------------------

def clean_macro_trends(macro_trends_df, macro_trends_cols, start_date=None, end_date=None):
    """
    Cleans macro trends data by filtering to specified columns and date range,
    validating data quality, and imputing missing values.

    Params:
    - macro_trends_df (DataFrame): df with macro trends data keyed on date
    - macro_trends_cols (list): list of macro trends columns to clean
    - start_date (str/datetime, optional): start date to filter data
    - end_date (str/datetime, optional): end date to filter data

    Returns:
    - filtered_df (DataFrame): cleaned dataframe with date as index
    """
    # 1. Filter to only relevant columns
    # ----------------------------------
    # Check if all required columns exist in the dataframe
    missing_columns = [col for col in macro_trends_cols if col not in macro_trends_df.columns]
    if missing_columns:
        raise ValueError(f"The following columns are missing from the dataframe: {', '.join(missing_columns)}")

    # 2. Validate numeric columns
    # ----------------------------------
    for col in macro_trends_cols:
        if not pd.api.types.is_numeric_dtype(macro_trends_df[col]):
            raise ValueError(f"Column {col} is not numeric")

    # 3. Filter to date range and set index
    # ----------------------------------
    filtered_df = macro_trends_df[macro_trends_cols].copy()

    if start_date:
        start_date = pd.to_datetime(start_date)
        filtered_df = filtered_df[filtered_df.index >= start_date]

    if end_date:
        end_date = pd.to_datetime(end_date)
        filtered_df = filtered_df[filtered_df.index <= end_date]

    if len(filtered_df) == 0:
        raise ValueError("No data available in the specified date range")

    # 4. Check for gaps > 6 days
    # ----------------------------------
    if len(filtered_df) > 1:
        date_diffs = filtered_df.index.to_series().diff().dt.days[1:]
        max_gap = date_diffs.max()
        if max_gap > 6:
            gap_locations = filtered_df.index[1:][date_diffs > 6]
            raise ValueError(f"Gaps larger than 6 days found in the data at: {gap_locations.tolist()}")

    # 5. Impute missing values
    # ----------------------------------
    # Resample to daily frequency and forward fill
    filtered_df = filtered_df.resample('D').asfreq().ffill()

    return filtered_df
