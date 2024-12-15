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


def retrieve_market_data():
    """
    Retrieves market data from the core.coin_market_data table and converts coin_id to categorical

    Returns:
    - market_data_df: DataFrame containing market data with 'coin_id' as a categorical column.
    """
    start_time = time.time()
    logger.info('Retrieving market data...')

    # SQL query to retrieve market data
    query_sql = """
        select cmd.coin_id
        ,date
        ,cast(cmd.price as float64) as price
        ,cast(cmd.volume as int64) as volume
        ,cast(cmd.market_cap as int64) as market_cap
        ,days_imputed
        from core.coin_market_data cmd
        order by 1,2
    """

    # Run the SQL query using dgc's run_sql method
    market_data_df = dgc().run_sql(query_sql)

    # Convert coin_id column to categorical to reduce memory usage
    market_data_df['coin_id'] = market_data_df['coin_id'].astype('category')

    # Downcast numeric columns to reduce memory usage
    market_data_df['price'] = market_data_df['price'].astype('float32')

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



def clean_market_data(market_data_df, config, earliest_date, latest_date):
    """
    Removes all market data records for
        1. coins with a longer price gap than the config['data_cleaning']['max_gap_days']
        2. coins with lower daily mean volume than the minimum_daily_volume between the
            earliest_window_start and the end of the last modeling period.

    Params:
    - market_data_df (DataFrame): DataFrame containing market data as well as the
        days_imputed column, which represents the number of days a real price has been
        forwardfilled in a row to ensure a complete time series.
    - max_gap_days (int): The maximum allowable number of forwardfilled dates before
        all records for the coin are removed
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
    low_volume_coins = mean_volume[mean_volume <= min_daily_volume].index.tolist()

    # Combine problematic coin lists
    coins_to_remove = list(set(gap_coin_ids + low_volume_coins))

    # Remove ALL records for problematic coins
    cleaned_df = market_data_df[~market_data_df['coin_id'].isin(coins_to_remove)]

    # Drop helper column
    cleaned_df = cleaned_df.drop(columns='days_imputed')

    logger.info("Removed %s coins (%s for gaps, %s for volume) and %s total records.",
                len(coins_to_remove),
                len(gap_coin_ids),
                len(low_volume_coins),
                len(market_data_df) - len(cleaned_df))

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



def retrieve_profits_data(start_date, end_date, min_wallet_inflows):
    """
    Retrieves data from the core.coin_wallet_profits table and converts columns to
    memory-efficient formats. Records prior to the start_date are excluded but a new
    row is imputed for the coin_id-wallet_address pair that summarizes their historical
    performance (see Step 2 CTEs).

    Params:
    - start_date (String): The earliest date to retrieve records for with format 'YYYY-MM-DD'
    - start_date (String): The latest date to retrieve records for with format 'YYYY-MM-DD'
    - min_wallet_inflows (Float): Wallets with fewer than this amount of lifetime
        USD inflows will be removed from the profits_df dataset

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
        -- STEP 1: retrieve profits data and apply USD inflows filter
        -------------------------------------------------------------
        with profits_base as (
            select coin_id
            ,date
            ,wallet_address
            ,profits_cumulative
            ,usd_balance
            ,usd_net_transfers
            ,usd_inflows
            ,usd_inflows_cumulative
            from core.coin_wallet_profits
            where date <= '{end_date}'
        ),

        usd_inflows_filter as (
            select coin_id
            ,wallet_address
            ,max(usd_inflows_cumulative) as total_usd_inflows
            from profits_base
            -- we don't need to include coin-wallet pairs that have no transactions between
            -- the start and end dates
            group by 1,2
        ),

        profits_base_filtered as (
            select pb.*
            from profits_base pb

            -- filter to remove wallet-coin pairs below the min_wallet_inflows
            join usd_inflows_filter f on f.coin_id = pb.coin_id
                and f.wallet_address = pb.wallet_address
                and f.total_usd_inflows >= {min_wallet_inflows}
        ),


        -- STEP 2: create new records for all coin-wallet pairs as of the training_period_start
        ---------------------------------------------------------------------------------------
        -- compute the starting profits and balances as of the training_period_start
        training_start_existing_rows as (
            -- identify coin-wallet pairs that already have a balance as of the period end
            select *
            from profits_base_filtered
            where date = '{start_date}'
        ),
        training_start_needs_rows as (
            -- for coin-wallet pairs that don't have existing records, identify the row closest to the period end date
            select t.*
            ,cmd_previous.price as price_previous
            ,cmd_training.price as price_current
            ,row_number() over (partition by t.coin_id,t.wallet_address order by t.date desc) as rn
            from profits_base_filtered t
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
            from profits_base_filtered
            -- transfers prior to the training period are summarized in training_start_new_rows
            where date >= '{start_date}'
            and (usd_balance != 0 or usd_net_transfers != 0)

            union all

            select *,
            True as is_imputed
            from training_start_new_rows
            where (usd_balance != 0 or usd_net_transfers != 0)

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
    """

    # Run the SQL query using dgc's run_sql method
    profits_df = dgc().run_sql(query_sql)

    logger.debug('Converting columns to memory-optimized formats...')

    # Convert coin_id to categorical and date to date
    profits_df['coin_id'] = profits_df['coin_id'].astype('category')
    profits_df['date'] = pd.to_datetime(profits_df['date'])

    # Add total_return column
    profits_df['total_return'] = (profits_df['profits_cumulative']
                                   / profits_df['usd_inflows_cumulative'])

    # Convert all numerical columns to 32 bit, using safe_downcast to avoid overflow
    profits_df = u.safe_downcast(profits_df, 'wallet_address', 'int32')
    profits_df = u.safe_downcast(profits_df, 'profits_cumulative', 'float32')
    profits_df = u.safe_downcast(profits_df, 'usd_balance', 'float32')
    profits_df = u.safe_downcast(profits_df, 'usd_net_transfers', 'float32')
    profits_df = u.safe_downcast(profits_df, 'usd_inflows', 'float32')
    profits_df = u.safe_downcast(profits_df, 'usd_inflows_cumulative', 'float32')
    profits_df = u.safe_downcast(profits_df, 'total_return', 'float32')

    # Round relevant columns
    columns_to_round = [
        'profits_cumulative'
        ,'usd_balance'
        ,'usd_net_transfers'
        ,'usd_inflows'
        ,'usd_inflows_cumulative'
    ]
    profits_df[columns_to_round] = profits_df[columns_to_round].round(2)
    profits_df[columns_to_round] = profits_df[columns_to_round].replace(-0, 0)

    # Remove rows with a rounded 0 balance and 0 transfers
    profits_df = profits_df[
        ~((profits_df['usd_balance'] == 0) &
        (profits_df['usd_net_transfers'] == 0))
    ]

    logger.info('Retrieved profits_df with %s unique coins and %s rows after %.2f seconds',
                len(set(profits_df['coin_id'])),
                len(profits_df),
                time.time()-start_time)

    return profits_df


@u.timing_decorator
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



def retrieve_macro_trends_data():
    """
    Retrieves Google Trends data from the macro_trends dataset. Because the data is weekly, it also
    resamples to daily records using linear interpolation.

    Returns:
    - google_trends_df: DataFrame keyed on date containing Google Trends values for multiple terms
    """
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
        ,bi.cdd_terminal_adjusted_90dma as btc_cdd_terminal_adjusted_90dma
        ,bi.fear_and_greed as btc_fear_and_greed
        ,bi.mvrv_z_score as btc_mvrv_z_score
        ,bi.vdd_multiple as btc_vdd_multiple
        ,gm.market_cap as global_market_cap
        ,gm.total_volume as global_volume
        ,gt.altcoin_worldwide as gtrends_altcoin_worldwide
        ,gt.cryptocurrency_worldwide as gtrends_cryptocurrency_worldwide
        ,gt.solana_us as gtrends_solana_us
        ,gt.cryptocurrency_us as gtrends_cryptocurrency_us
        ,gt.bitcoin_us as gtrends_bitcoin_us
        ,gt.solana_worldwide as gtrends_solana_worldwide
        ,gt.coinbase_us as gtrends_coinbase_us
        ,gt.bitcoin_worldwide as gtrends_bitcoin_worldwide
        ,gt.ethereum_worldwide as gtrends_ethereum_worldwide
        ,gt.ethereum_us as gtrends_ethereum_us
        ,gt.altcoin_us as gtrends_altcoin_us
        ,gt.coinbase_worldwide as gtrends_coinbase_worldwide
        ,gt.memecoin_worldwide as gtrends_memecoin_worldwide
        ,gt.memecoin_us as gtrends_memecoin_us
        from all_dates d
        left join `macro_trends.bitcoin_indicators` bi on bi.date = d.date
        left join `macro_trends.crypto_global_market` gm on gm.date = d.date
        left join `macro_trends.google_trends` gt on gt.date = d.date
        order by date desc
        """

    # Run the SQL query using dgc's run_sql method
    macro_trends_df = dgc().run_sql(query_sql)
    logger.debug('Retrieved macro trends data with shape %s',macro_trends_df.shape)

    # Convert the date column to datetime format
    macro_trends_df['date'] = pd.to_datetime(macro_trends_df['date'])

    # Resample the df to fill in missing days by using date as the index
    macro_trends_df = macro_trends_df.set_index('date')
    macro_trends_df = macro_trends_df.resample('D').interpolate(method='time', limit_area='inside')

    # Reset index
    macro_trends_df = macro_trends_df.reset_index()

    return macro_trends_df


def clean_macro_trends(macro_trends_df, config):
    """
    Basic function to only retain the columns in macro_trends_df that have metrics described in
    the config files.

    Params:
    - macro_trends_df (DataFrame): df with macro trends data keyed on date
    - config (dict): config.yaml

    Returns:
    - filtered_macro_trends_df (DataFrame): input df with non-metric configured columns removed
    """
    # 1. Filter to only relevant columns
    # ----------------------------------
    # Get the keys from the config dictionary
    metric_columns = list(config['datasets']['macro_trends'].keys()) if 'macro_trends' in config['datasets'] else []

    # Ensure 'date' is included
    required_columns = ['date'] + metric_columns

    # Check if all required columns exist in the dataframe
    missing_columns = [col for col in required_columns if col not in macro_trends_df.columns]

    if missing_columns:
        raise ValueError(f"The following columns are missing from the dataframe: {', '.join(missing_columns)}")

    # Filter the dataframe to only the required columns
    filtered_macro_trends_df = macro_trends_df[required_columns]


    # 2. Confirm there are no mid-series nan values
    # ---------------------------------------------
    nan_checks = []
    nan_cols = []

    for col in filtered_macro_trends_df.columns:
        # Check if there are NaNs in the middle of any columns
        nan_check = u.check_nan_values(filtered_macro_trends_df[col])
        nan_checks.append(nan_check)

        # Store column name if there are NaNs in the middle of the series
        if nan_check:
            nan_cols.append(col)

    if sum(nan_checks) > 0:
        raise ValueError(f"NaN values found in macro_trends_df columns: {nan_cols}")

    return filtered_macro_trends_df
