import logging
from datetime import datetime
import pandas as pd
import numpy as np
import pandas_gbq
from dreams_core.googlecloud import GoogleCloud as dgc

# Local module imports
import wallet_features.performance_features as wpf
import wallet_features.trading_features as wtf
import utils as u

# set up logger at the module level
logger = logging.getLogger(__name__)

# -----------------------------------
#       Main Interface Function
# -----------------------------------

@u.timing_decorator
def calculate_scenario_features(
    training_profits_df: pd.DataFrame,
    training_market_indicators_df: pd.DataFrame,
    performance_features_df: pd.DataFrame,
    period_start_date: str,
    period_end_date: str,
    wallets_config: dict
) -> pd.DataFrame:
    """
    Calculates scenario wallet transfer features based on ideal price points.

    Params:
    - training_profits_df (DataFrame): Historical profits data
    - training_market_indicators_df (DataFrame): df with coin_id,date,price columns
    - performance_features_df (DataFrame): base performance of each wallet that we will
        compare vs ideal performance
    - period_start_date (str): Start of analysis period
    - period_end_date (str): End of analysis period

    Returns:
    - scenario_features_df (DataFrame): Wallet-level scenario transfer features
    """
    # Config params
    table_reference_date = datetime.strptime(wallets_config['training_data']['modeling_period_start'],
                                             '%Y-%m-%d').strftime('%Y%m%d')
    use_hybrid_ids = wallets_config['training_data']['hybridize_wallet_ids']

    # Upload profits data to temporary storage
    upload_profits_df_dates(
        training_profits_df,
        table_reference_date
    )

    # Calculate ideal price points for each transfer
    ideal_transfers_df = get_ideal_transfers_df(
        period_end_date,
        use_hybrid_ids,
        table_reference_date
    )

    # Enrich with actual transfer history
    ideal_transfers_df = append_profits_data(
        ideal_transfers_df,
        training_profits_df,
        training_market_indicators_df
    )

    # Generate performance features
    performance_features_df = generate_scenario_features(
        ideal_transfers_df,
        period_start_date,
        period_end_date,
        wallets_config
    )
    scenario_features_df = performance_features_df

    # Data completeness check
    profits_wallets = training_profits_df['wallet_address'].drop_duplicates()
    feature_wallets = scenario_features_df.index.get_level_values('wallet_address')
    if len(profits_wallets) != len(feature_wallets):
        raise ValueError("Wallets in profits_df were not found in ideal_transfers_df.")

    return scenario_features_df




# -----------------------------------
#          Helper Functions
# -----------------------------------

@u.timing_decorator
def upload_profits_df_dates(
        training_profits_df: pd.DataFrame,
        table_reference_date: str = '') -> None:
    """
    Uploads all coin/wallet/date combinations in profits_df to BigQuery temp table.

    Params:
    - training_profits_df (DataFrame): Source profits data
    - table_reference_date (str): Modifier to the upload table name so offsets don't collide.
    """
    upload_df = training_profits_df[['coin_id', 'date', 'wallet_address']].copy()

    project_id = 'western-verve-411004'
    table_id = f"{project_id}.temp.training_cohort_coin_dates_{table_reference_date}"
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
def get_ideal_transfers_df(
        training_period_end: str,
        use_hybrid_ids: bool,
        table_reference_date: str
    ) -> pd.DataFrame:
    """
    Get wallet transfer data with price ranges.

    Params:
    - training_period_end (str): End date for training period
    - use_hybrid_ids (bool): Whether to use hybrid_cw_ids instead of wallet_ids
    - table_reference_date (str): Reference date for temp table naming

    Returns:
    - ideal_transfers_df (DataFrame): Transfer data with min/max prices
    """
    # Different logic for date_ranges based on hybrid_ids setting
    if use_hybrid_ids:
        date_ranges_logic = f"""
        date_ranges as (
            select
                xw_cw.hybrid_cw_id as wallet_address,   -- mask hybrid_cw_id as "wallet_address"
                wcd.coin_id,
                wcd.date,
                COALESCE(
                    LEAD(wcd.date) OVER (PARTITION BY wc.wallet_address, wcd.coin_id ORDER BY wcd.date) - interval 1 day,
                    '{training_period_end}'
                ) as date_range

            -- the "wallet_address" in training cohort table is really a hybrid_cw_id
            from temp.wallet_modeling_training_cohort_{table_reference_date} wc
            join reference.wallet_coin_ids xw_cw on xw_cw.hybrid_cw_id = wc.wallet_address

            -- the "wallet_address" in coin-dates table is really a hybrid_cw_id
            join temp.training_cohort_coin_dates_{table_reference_date} wcd on wcd.wallet_address = xw_cw.hybrid_cw_id

            join core.coin_market_data cmd on cmd.coin_id = wcd.coin_id
                and cmd.date = wcd.date
            where wcd.date <= '{training_period_end}'
        )"""
    else:
        date_ranges_logic = f"""
        date_ranges as (
            select
                wc.wallet_address,  -- mask wallet_id as "wallet_address"
                wcd.coin_id,
                wcd.date,
                COALESCE(
                    LEAD(wcd.date) OVER (PARTITION BY wc.wallet_address, wcd.coin_id ORDER BY wcd.date) - interval 1 day,
                    '{training_period_end}'
                ) as date_range

            -- the "wallet_address" in training cohort table is really a wallet_id
            from temp.wallet_modeling_training_cohort_{table_reference_date} wc
            join temp.training_cohort_coin_dates_{table_reference_date} wcd on wcd.wallet_address = wc.wallet_id

            join reference.wallet_ids xw on xw.wallet_id = wc.wallet_id
            join core.coin_market_data cmd on cmd.coin_id = wcd.coin_id
                and cmd.date = wcd.date
            where wcd.date <= '{training_period_end}'
        )"""

    sql_query = f"""
        with {date_ranges_logic}

        select dr.wallet_address  -- this is actually the ID selected in the CTE, not a wallet_address
        ,dr.coin_id
        ,dr.date
        ,DATE_DIFF(dr.date_range, dr.date, DAY) as days_until_next_transfer
        ,max(cmd.price) as max_price
        ,min(cmd.price) as min_price
        from date_ranges dr
        join core.coin_market_data cmd on cmd.coin_id = dr.coin_id
            and cmd.date between dr.date and dr.date_range
        where cmd.date <= '{training_period_end}'
        group by 1,2,3,4
        order by date, wallet_address, coin_id
        """
    print(sql_query)
    ideal_transfers_df = dgc().run_sql(sql_query)

    # Handle column dtypes
    ideal_transfers_df['coin_id'] = ideal_transfers_df['coin_id'].astype('category')
    ideal_transfers_df['date'] = ideal_transfers_df['date'].astype('datetime64[ns]')
    ideal_transfers_df = u.df_downcast(ideal_transfers_df)

    # Confirm no nulls
    if ideal_transfers_df.isna().sum().sum() > 0:
        raise ValueError(f"Null values found in ideal_transfers_df. Review query:{sql_query}")

    return ideal_transfers_df



@u.timing_decorator
def append_profits_data(ideal_transfers_df: pd.DataFrame,
                        training_profits_df: pd.DataFrame,
                        training_market_indicators_df: pd.DataFrame,
                        ) -> pd.DataFrame:
    """
    Merge profits data with ideal transfers, enforcing data consistency and type constraints.

    Params:
    - ideal_transfers_df (DataFrame): Ideal transfers to append
    - training_profits_df (DataFrame): Source profits data
    - training_market_indicators_df (DataFrame): df with coin_id,date,price columns

    Returns:
    - merged_df (DataFrame): Merged and validated transfers data
    """
    # Merge profits and market data
    merged_df = training_profits_df[['date', 'wallet_address', 'coin_id', 'usd_balance', 'usd_net_transfers']].merge(
        training_market_indicators_df[['date', 'coin_id', 'price']],
        on=['date', 'coin_id'],
        how='inner'
    )
    if len(merged_df) != len(training_profits_df):
        raise ValueError("Merge of profits_df and market_data_df did not fully align")

    # Merge ideal_transfers_df
    merged_df = merged_df.merge(
        ideal_transfers_df,
        on=['date', 'coin_id', 'wallet_address'],
        how='inner'
    )
    if len(merged_df) != len(ideal_transfers_df):
        raise ValueError("Merge of profits_df and market_data_df did not fully align")

    # Add ratio columns
    merged_df['token_balance'] = merged_df['usd_balance'] / merged_df['price']
    merged_df['token_net_transfers'] = merged_df['usd_net_transfers'] / merged_df['price']

    # Validate output quality
    merged_df = u.ensure_index(merged_df)
    merged_df = u.df_downcast(merged_df)
    if merged_df.isna().sum().sum() > 0:
        raise ValueError("Null values found in merged output")

    return merged_df



@u.timing_decorator
def generate_scenario_performance(
        scenario_profits_df: pd.DataFrame,
        period_start_date: str,
        period_end_date: str,
        wallets_config: dict
    ) -> pd.DataFrame:
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
        scenario_trading_df,
        wallets_config
    )

    # Convert to the Hypothetical feature set
    features = wallets_config['features']['scenario_performance_features']
    if len(features) != len(set(features)):
        raise ValueError("Duplicate features detected in scenario_performance_features")
    scenario_features_df = scenario_performance_df[features]

    # Remove '/' delimiters for better importance analysis parsing
    scenario_features_df.columns = scenario_features_df.columns.str.replace('/', '_')

    return scenario_features_df



@u.timing_decorator
def generate_scenario_features(
        ideal_transfers_df: pd.DataFrame,
        period_start_date: str,
        period_end_date: str,
        wallets_config: dict
    ) -> pd.DataFrame:
    """
    Generate features for best and worst case selling scenarios.

    Params:
    - ideal_transfers_df (DataFrame): Transfer data with min/max price columns
    - training_profits_df (DataFrame): Source profits data
    - period_start_date (str): Start date of analysis period
    - period_end_date (str): End date of analysis period

    Returns:
    - scenario_features_df (DataFrame): Combined best/worst case features
    """
    # Generate best case sells scenario (sells at highest price)
    best_sells_profits_df = ideal_transfers_df[['usd_balance']].assign(
        usd_net_transfers=np.where(
            ideal_transfers_df['token_net_transfers'] < 0,
            ideal_transfers_df['token_net_transfers'] * ideal_transfers_df['max_price'],
            ideal_transfers_df['usd_net_transfers']
        )
    )
    best_sells_features = generate_scenario_performance(
        best_sells_profits_df,
        period_start_date,
        period_end_date,
        wallets_config
    )
    best_sells_features = best_sells_features.add_prefix('sells_best/')

    # Merge all together
    scenario_features_df = best_sells_features

    # Data quality checks
    all_wallets = set(ideal_transfers_df.index.get_level_values('wallet_address'))
    if len(all_wallets) != len(scenario_features_df):
        raise ValueError("Missing wallets identified in scenario_performance_features.")
    if scenario_features_df.isna().sum().sum() > 0:
        raise ValueError("Null values found in scenario_performance_features.")

    return scenario_features_df
