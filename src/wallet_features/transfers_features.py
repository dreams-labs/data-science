"""
Calculates metrics aggregated at the wallet-coin level
"""

import logging
import pandas as pd
import numpy as np
import pandas_gbq
from dreams_core.googlecloud import GoogleCloud as dgc

# Local module imports
import wallet_features.performance_features as wpf
import wallet_features.trading_features as wtf
from wallet_modeling.wallets_config_manager import WalletsConfig
import utils as u

# set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()


# -------------------------------------------------
#          Transfers Sequencing Features
# -------------------------------------------------
# these identify which buyer number a wallet was to a given coin

def retrieve_transfers_sequencing(hybridize_wallet_ids: bool) -> pd.DataFrame:
    """
    Returns the buyer number for each wallet-coin pairing, where the first buyer
    receives rank 1 and the count increases for each subsequence wallet.

    Buyer numbers are calculated for all wallets but the returned df only includes
    wallets that were uploaded to the temp.wallet_modeling_training_cohort table.

    Params:
    - hybridize_wallet_ids (bool): whether the IDs are regular wallet_ids or hybrid wallet-coin IDs

    Returns:
    - buyer_numbers_df (df): dataframe showing that buyer number a wallet was for
        the associated coin_id.
    """
    # Wallet transactions below this threshold will not be included in the buyer sequencing
    minimum_transaction_size = wallets_config['features']['timing_metrics_min_transaction_size']

    # All data after the training period must be ignored to avoid data leakage
    training_period_end = wallets_config['training_data']['training_period_end']

    sequencing_sql_ctes = f"""
        with transaction_rank as (
            select coin_id
            ,wallet_address
            ,min(date) as first_transaction
            from core.coin_wallet_profits cwp
            where cwp.usd_net_transfers > {minimum_transaction_size}
            and cwp.date <= '{training_period_end}'
            group by 1,2
        ),

        buy_ordering as (
            select tr.coin_id
            ,tr.wallet_address
            ,first_transaction
            ,rank() over (partition by coin_id order by first_transaction asc) as buyer_number
            from transaction_rank tr
        )
        """

    if hybridize_wallet_ids is False:
        sequencing_sql = f"""
        {sequencing_sql_ctes}

        select wc.wallet_id as wallet_address -- rename to match the rest of the pipeline
        ,o.coin_id
        ,o.first_transaction
        ,o.buyer_number
        from buy_ordering o
        join reference.wallet_ids xw on xw.wallet_address = o.wallet_address
        join temp.wallet_modeling_training_cohort wc on wc.wallet_id = xw.wallet_id
        """
    else:
        sequencing_sql = f"""
        {sequencing_sql_ctes}

        select wc.hybrid_id as wallet_address -- rename to match the rest of the pipeline
        ,o.coin_id
        ,o.first_transaction
        ,o.buyer_number
        from buy_ordering o
        join temp.wallet_modeling_training_cohort wc on wc.wallet_address = o.wallet_address
            and wc.coin_id = o.coin_id
        """

    transfers_sequencing_df = dgc().run_sql(sequencing_sql)
    logger.info("Retrieved transfers data for %s wallet-coin pairs associated with %s wallets "
                "in temp.wallet_modeling_training_cohort.",
                len(transfers_sequencing_df), len(transfers_sequencing_df['wallet_address'].unique()))

    # Convert coin_id column to categorical to reduce memory usage
    transfers_sequencing_df['coin_id'] = transfers_sequencing_df['coin_id'].astype('category')
    transfers_sequencing_df = u.df_downcast(transfers_sequencing_df)

    return transfers_sequencing_df


@u.timing_decorator
def calculate_transfers_sequencing_features(profits_df, transfers_sequencing_df):
    """
    Retrieves facts about the wallet's transfer activity based on blockchain data.
    Period boundaries are defined by the dates in profits_df through the inner join.

    Params:
        profits_df (df): the profits_df for the period that the features will reflect
        transfers_sequencing_df (df): each wallet's lifetime transfers data

    Returns:
        transfers_sequencing_features_df (df): dataframe indexed on wallet_address with
        transfers feature columns
    """

    # Inner join lifetime transfers with the profits_df window to filter on date
    window_transfers_data_df = pd.merge(
        profits_df,
        transfers_sequencing_df,
        left_on=['coin_id', 'date', 'wallet_address'],
        right_on=['coin_id', 'first_transaction', 'wallet_address'],
        how='inner'
    )

    # Append buyer numbers to the merged_df
    transfers_sequencing_features_df = window_transfers_data_df.groupby('wallet_address').agg({
        'buyer_number': ['count', 'mean', 'median', 'min']
    })
    transfers_sequencing_features_df.columns = [
        'new_coin_buy_counts',
        'avg_buyer_number',
        'median_buyer_number',
        'min_buyer_number'
    ]

    return transfers_sequencing_features_df




# -------------------------------------------------
#            Holding Behavior Features
# -------------------------------------------------

# these identify which how long a wallet held their tokens

# def retrieve_transfers():
#     """
#     Returns the buyer number for each wallet-coin pairing, where the first buyer
#     receives rank 1 and the count increases for each subsequence wallet.

#     Buyer numbers are calculated for all wallets but the returned df only includes
#     wallets that were uploaded to the temp.wallet_modeling_training_cohort table.

#     Returns:
#     - buyer_numbers_df (df): dataframe showing that buyer number a wallet was for
#         the associated coin_id.
#     """
#     # All data after the training period must be ignored to avoid data leakage
#     training_period_end = wallets_config['training_data']['training_period_end']

#     transfers_sql = f"""select cwt.coin_id
#     ,cwt.wallet_address
#     ,cwt.date
#     ,cwt.net_transfers
#     ,cwt.balance
#     from core.coin_wallet_transfers cwt
#     join temp.wallet_modeling_training_cohort wmc on wmc.wallet_address = cwt.wallet_address
#     and cwt.date <= '{training_period_end}'
#     order by 1,2,3"""

#     transfers_df = dgc().run_sql(transfers_sql)

#     logger.info("Retrieved transfers data for %s wallet-coin-date records.",
#                 len(transfers_df))

#     transfers_df = u.df_downcast(transfers_df)

#     return transfers_df


# # needs to be moved to profits_df with a USD value floor
# def calculate_days_since_last_buy(transfers_df):
#     """
#     Calculate days since last buy for a single wallet-coin pair as of each date.

#     Parameters:
#     transfers_df: pandas DataFrame with columns [date, wallet_address, coin_id, net_transfers]
#         date can be string or datetime format
#         net_transfers should be positive for buys, negative for sells

#     Returns:
#     DataFrame with additional column 'days_since_last_buy'
#     """
#     # Create a copy to avoid modifying original dataframe
#     result_df = transfers_df.copy()

#     # Convert dates to datetime if they aren't already
#     result_df['date'] = pd.to_datetime(result_df['date'])

#     # Mark buy transactions and get last buy date for each wallet-coin combination
#     result_df['is_buy'] = result_df['net_transfers'] > 0

#     # First create the series of dates where is_buy is True, then group and forward fill
#     buy_dates = result_df['date'].where(result_df['is_buy'])
#     result_df['last_buy_date'] = buy_dates.groupby([result_df['wallet_address'], result_df['coin_id']]).ffill()

#     # Calculate days since last buy
#     result_df['days_since_last_buy'] = (
#         result_df['date'] - result_df['last_buy_date']
#     ).dt.days

#     # Clean up intermediate columns
#     columns_to_keep = ['coin_id', 'wallet_address', 'date', 'days_since_last_buy']
#     result_df = result_df[columns_to_keep]


#     return result_df



def calculate_average_holding_period(transfers_df):
    """
    Calculate the average holding period based on the dates included for a
    single wallet-coin pair.

    The calculation handles:
    - New tokens starting with 0 days held
    - Aging of existing tokens over time
    - Proportional reduction in holding days when tokens are sold

    Parameters:
    - transfers_df: pandas DataFrame containing columns [date, net_transfers, balance]

    Returns:
    - holding_days_df: DataFrame with columns [date, average_holding_period]
    """
    holding_days_df = transfers_df[['date','net_transfers','balance']].copy().sort_values('date')
    holding_days_df['date'] = pd.to_datetime(holding_days_df['date'])

    avg_age = 0
    balance = 0
    last_date = None
    ages = []

    for _, row in holding_days_df.iterrows():
        current_date = row['date']
        net = row['net_transfers']

        # Calculate days passed
        days_passed = (current_date - last_date).days if last_date else 0
        last_date = current_date

        # Age current holdings
        if balance > 0:
            avg_age += days_passed

        # Adjust for buy/sell
        if net > 0:  # Buy: blend with new coins (age=0)
            total_coins = balance + net
            avg_age = (avg_age * (balance / total_coins)) + (0 * (net / total_coins))
            balance = total_coins
        elif net < 0:  # Sell: remove coins at current avg age
            balance += net  # net is negative
            # Average age stays the same since we remove equally aged coins

        # Compute average holding period
        current_avg = avg_age if balance > 0 else 0
        ages.append(current_avg)

    holding_days_df['average_holding_period'] = ages

    # Clean up intermediate columns if desired
    columns_to_keep = ['date', 'average_holding_period']
    holding_days_df = holding_days_df[columns_to_keep]

    return holding_days_df




# -------------------------------------------------
#            Ideal Performance Features
# -------------------------------------------------


@u.timing_decorator
def calculate_transfers_hypothetical_features(
    training_profits_df: pd.DataFrame,
    period_start_date: str,
    period_end_date: str
) -> pd.DataFrame:
    """
    Calculates hypothetical wallet transfer features based on ideal price points.

    Params:
    - training_profits_df (DataFrame): Historical profits data
    - period_start_date (str): Start of analysis period
    - period_end_date (str): End of analysis period

    Returns:
    - hypothetical_features_df (DataFrame): Wallet-level hypothetical transfer features
    """
    # Upload profits data to temporary storage
    upload_profits_df_dates(training_profits_df)

    # Calculate ideal price points for each transfer
    ideal_transfers_df = get_ideal_transfers_df(
        period_start_date,
        period_end_date
    )

    # Enrich with actual transfer history
    ideal_transfers_df = append_profits_data(
        ideal_transfers_df,
        training_profits_df
    )

    # Generate wallet features
    hypothetical_features_df = generate_hypothetical_features(
        ideal_transfers_df,
        period_start_date,
        period_end_date
    )

    return hypothetical_features_df



@u.timing_decorator
def upload_profits_df_dates(training_profits_df: pd.DataFrame) -> None:
    """
    Uploads all coin/wallet/date combinations in profits_df to BigQuery temp table.

    Params:
    - training_profits_df (DataFrame): Source profits data
    - project_id (str): GCP project identifier
    """
    upload_df = training_profits_df[['coin_id', 'date', 'wallet_address']].copy()

    project_id = 'western-verve-411004'
    table_id = f"{project_id}.temp.training_cohort_coin_dates"
    schema = [
        {'name': 'coin_id', 'type': 'string'},
        {'name': 'date', 'type': 'date'},
        {'name': 'wallet_address', 'type': 'integer'}
    ]

    pandas_gbq.to_gbq(
        upload_df,
        table_id,
        project_id=project_id,
        if_exists='replace',
        table_schema=schema,
        progress_bar=False
    )



@u.timing_decorator
def get_ideal_transfers_df(training_starting_balance_date: str,
                           training_period_end: str) -> pd.DataFrame:
    """
    Get wallet transfer data with price ranges.

    Params:
    - training_starting_balance_date (str): Starting balance date for training period
    - training_period_end (str): End date for training period

    Returns:
    - ideal_transfers_df (DataFrame): Transfer data with min/max prices
    """
    sql_query = f"""
        with first_price_dates as (
            -- Get the first price date for coins which we need to fill starting_balances
            select cmd.coin_id,
            min(date) as first_price_date
            from core.coin_market_data cmd
            group by 1
        ),

        starting_balances as (
            -- Get most recent balance ON OR BEFORE the starting balance date for each wallet-coin pair
            -- This will be used to fill the initial balance for training period start
            select cwt.wallet_address, cwt.coin_id, cwt.balance, cwt.date,
            row_number() over(partition by cwt.wallet_address, cwt.coin_id order by date desc) as rn
            from core.coin_wallet_transfers cwt
            join first_price_dates fpd on fpd.coin_id = cwt.coin_id

            -- if a transfer was during the period but before price data, it becomes the starting_balance
            where (cwt.date <= greatest('{training_starting_balance_date}',fpd.first_price_date))
        ),

        date_ranges as (
            select
                wc.wallet_id,
                wcd.coin_id,
                wcd.date,
                -- Use starting balance for first date, leave other nulls for python forward-fill
                case
                    when (
                        -- cases when the first transfer was before the period
                        wcd.date = '{training_starting_balance_date}' OR
                        -- cases when the first transfer was before the period AND the first price was within the period
                        -- (rows that exists in profits_df but not in the transfers table were imputed on the first price date)
                        cwt.date is null
                    ) then sb.balance
                    else last_value(cwt.balance ignore nulls) over (
                        partition by wc.wallet_id, wcd.coin_id
                        order by wcd.date
                        rows between unbounded preceding and current row
                    )
                end as balance,
                coalesce(cwt.net_transfers, 0) as net_transfers,
                COALESCE(
                    -- All available transfers will return the day before the following transfer
                    LEAD(wcd.date) OVER (PARTITION BY xw.wallet_address, wcd.coin_id ORDER BY wcd.date) - interval 1 day,
                    -- The most recent transfer will return null from the lead, so coalesce-fill with training_period_end
                    '{training_period_end}'
                ) as date_range
            from temp.wallet_modeling_training_cohort wc
            join temp.training_cohort_coin_dates wcd on wcd.wallet_address = wc.wallet_id
            join reference.wallet_ids xw on xw.wallet_id = wc.wallet_id
            left join core.coin_wallet_transfers cwt on cwt.wallet_address = xw.wallet_address
                and cwt.coin_id = wcd.coin_id
                and cwt.date = wcd.date
            left join starting_balances sb on sb.wallet_address = xw.wallet_address
                and sb.coin_id = wcd.coin_id
                and sb.rn = 1
            join core.coin_market_data cmd on cmd.coin_id = wcd.coin_id
                and cmd.date = wcd.date
            where wcd.date <= '{training_period_end}'
        )

        select dr.wallet_id as wallet_address
        ,dr.coin_id
        ,dr.date
        ,dr.balance as token_balance
        ,dr.net_transfers as token_net_transfers
        ,max(cmd.price) as max_price
        ,min(cmd.price) as min_price
        from date_ranges dr
        join core.coin_market_data cmd on cmd.coin_id = dr.coin_id
            and cmd.date between dr.date and dr.date_range
        where cmd.date <= '{training_period_end}'
        group by 1,2,3,4,5
        order by date,wallet_id,coin_id
    """
    ideal_transfers_df = dgc().run_sql(sql_query)

    # Handle column dtypes
    ideal_transfers_df['coin_id'] = ideal_transfers_df['coin_id'].astype('category')
    ideal_transfers_df['date'] = ideal_transfers_df['date'].astype('datetime64[ns]')
    ideal_transfers_df = u.df_downcast(ideal_transfers_df)

    # Forward fill balances to cover imputed period end rows
    ideal_transfers_df['token_balance'] = (ideal_transfers_df.sort_values('date')
                                           .groupby(['wallet_address', 'coin_id'])['token_balance']
                                           .ffill())

    # Confirm no nulls
    if ideal_transfers_df.isna().sum().sum() > 0:
        raise ValueError(f"Null values found in ideal_transfers_df. Review query:{sql_query}")


    return ideal_transfers_df



@u.timing_decorator
def append_profits_data(ideal_transfers_df: pd.DataFrame,
                        training_profits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge profits data with ideal transfers, enforcing data consistency and type constraints.

    Params:
    - ideal_transfers_df (DataFrame): Ideal transfers to append
    - training_profits_df (DataFrame): Source profits data

    Returns:
    - merged_df (DataFrame): Merged and validated transfers data
    """
    # Validate input sizes
    if len(ideal_transfers_df) != len(training_profits_df):
        raise ValueError(
            f"Dataframe sizes do not match: ideal_transfers_df {ideal_transfers_df.shape} vs "
            f"training_profits_df {training_profits_df.shape}"
        )

    # Standardize date types
    for df in [ideal_transfers_df, training_profits_df]:
        df['date'] = df['date'].astype('datetime64[ns]')

    # Align categorical types
    common_categories = pd.CategoricalDtype(
        categories=training_profits_df['coin_id'].cat.categories,
        ordered=False
    )
    training_profits_df['coin_id'] = training_profits_df['coin_id'].astype(common_categories)
    ideal_transfers_df['coin_id'] = ideal_transfers_df['coin_id'].astype(common_categories)

    # Execute merge
    cols_to_merge = ['wallet_address', 'coin_id', 'date', 'usd_net_transfers', 'usd_balance']
    merged_df = pd.merge_asof(
        training_profits_df.sort_values(['date', 'wallet_address', 'coin_id'])[cols_to_merge],
        ideal_transfers_df,
        by=['wallet_address', 'coin_id'],
        on='date',
        direction='backward'
    )

    # Validate output quality
    merged_df = u.ensure_index(merged_df)
    merged_df = u.df_downcast(merged_df)
    if merged_df.isna().sum().sum() > 0:
        raise ValueError("Null values found in merged output")

    return merged_df



@u.timing_decorator
def generate_scenario_features(scenario_profits_df: pd.DataFrame,
                               period_start_date: str,
                               period_end_date: str,) -> pd.DataFrame:
    """
    Generate trading and profit features for a given transfer scenario.

    Params:
    - scenario_profits_df (DataFrame): DataFrame with columns 'usd_balance' and 'usd_net_transfers'
    - period_start_date (str): Training period start date
    - period_end_date (str): Training period end date

    Returns:
    - scenario_features_df (DataFrame): Performance features for the profits_df scenario
    """

    # Generate performance features
    scenario_profits_df = wtf.calculate_crypto_balance_columns(
        scenario_profits_df, period_start_date, period_end_date
    )
    scenario_trading_df = wtf.calculate_gain_and_investment_columns(scenario_profits_df)
    scenario_performance_df = wpf.calculate_performance_features(
        scenario_trading_df, include_twb_metrics=False
    )

    # Convert to the Hypothetical feature set
    features = wallets_config['features']['hypothetical_performance_features']
    if len(features) != len(set(features)):
        raise ValueError("Duplicate features detected in hypothetical_performance_features")
    hypothetical_features_df = scenario_performance_df[features]

    # Remove '/' delimiters for better importance analysis parsing
    hypothetical_features_df.columns = hypothetical_features_df.columns.str.replace('/', '_')

    return hypothetical_features_df



@u.timing_decorator
def generate_hypothetical_features(ideal_transfers_df: pd.DataFrame,
                                   period_start_date: str,
                                   period_end_date: str) -> pd.DataFrame:
    """
    Generate features for best and worst case selling scenarios.

    Params:
    - ideal_transfers_df (DataFrame): Transfer data with min/max price columns
    - period_start_date (str): Start date of analysis period
    - period_end_date (str): End date of analysis period

    Returns:
    - hypothetical_features_df (DataFrame): Combined best/worst case features
    """
    # Generate best case scenario (sells at highest price)
    best_profits_df = ideal_transfers_df[['usd_balance']].assign(
        usd_net_transfers=np.where(
            ideal_transfers_df['token_net_transfers'] < 0,
            ideal_transfers_df['token_net_transfers'] * ideal_transfers_df['max_price'],
            ideal_transfers_df['usd_net_transfers']
        )
    )
    best_features = generate_scenario_features(best_profits_df, period_start_date, period_end_date)
    best_features = best_features.add_prefix('sells_best/')

    # Generate worst case scenario (sells at lowest price)
    worst_profits_df = ideal_transfers_df[['usd_balance']].assign(
        usd_net_transfers=np.where(
            ideal_transfers_df['token_net_transfers'] < 0,
            ideal_transfers_df['token_net_transfers'] * ideal_transfers_df['min_price'],
            ideal_transfers_df['usd_net_transfers']
        )
    )
    worst_features = generate_scenario_features(worst_profits_df, period_start_date, period_end_date)
    worst_features = worst_features.add_prefix('sells_worst/')

    hypothetical_features_df = pd.concat([best_features, worst_features], axis=1)

    return hypothetical_features_df
