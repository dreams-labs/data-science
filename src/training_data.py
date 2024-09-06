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
        ,case when cwt.coin_id is not null then true else false end as has_transfers_data
        from core.coin_market_data cmd
        left join (
            select coin_id
            from core.coin_wallet_transfers cwt 
            group by 1
        ) cwt on cwt.coin_id = cmd.coin_id
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
