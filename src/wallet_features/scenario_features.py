"""
Generates performance metrics for wallets under given scenarios.

The first scenario support is the "ideal timing" scenario, which compares
 each wallets' trading performance with how they would have done if they
 sold at the highest available price and bought at the lowest available price,
 defined as all prices between their previous transaction (or period start)
 and the actual transaction date.
"""
import logging
import copy
from datetime import datetime
import pandas as pd
import numpy as np
from google.cloud import bigquery

# Local module imports
import wallet_features.performance_features as wpf
import wallet_features.trading_features as wtf
import utils as u
import utilities.bq_utils as bqu

# set up logger at the module level
logger = logging.getLogger(__name__)

# -----------------------------------
#       Main Interface Function
# -----------------------------------

@u.timing_decorator
def calculate_scenario_features(
    training_profits_df: pd.DataFrame,
    training_market_indicators_df: pd.DataFrame,
    trading_features_df: pd.DataFrame,
    performance_features_df: pd.DataFrame,
    period_start_date: str,
    period_end_date: str,
    wallets_config: dict
) -> pd.DataFrame:
    """
    Calculates scenario wallet transfer features based on scenario price points.

    Params:
    - training_profits_df (DataFrame): Historical profits data
    - training_market_indicators_df (DataFrame): df with coin_id,date,price columns
    - trading_features_df (DataFrame): actual trading features to compare vs scenario
    - performance_features_df (DataFrame): actual performance features to compare vs scenario
    - period_start_date (str): Start of analysis period
    - period_end_date (str): End of analysis period

    Returns:
    - scenario_features_df (DataFrame): Wallet-level scenario transfer features
    """
    # Deep copy to avoid thread collision
    profits_df = copy.deepcopy(training_profits_df)
    market_indicators_df = copy.deepcopy(training_market_indicators_df)
    base_trading_df = copy.deepcopy(trading_features_df)
    base_performance_df = copy.deepcopy(performance_features_df)
    u.assert_matching_indices(base_trading_df,base_performance_df)

    # Config params
    cohort_reference_date = datetime.strptime(wallets_config['training_data']['modeling_period_start'],
                                             '%Y-%m-%d').strftime('%Y%m%d')
    use_hybrid_ids = wallets_config['training_data']['hybridize_wallet_ids']

    # Create profits reference from period dates
    profits_reference = (
        f"{cohort_reference_date}_"
        f"{datetime.strptime(period_start_date, '%Y-%m-%d').strftime('%y%m%d')}_"
        f"{datetime.strptime(period_end_date, '%Y-%m-%d').strftime('%y%m%d')}_"
        f"hybridized_{use_hybrid_ids}"
    )
    # Upload profits data to temporary storage
    upload_profits_df_dates(
        profits_df,
        profits_reference
    )

    # Calculate scenario price points for each transfer
    scenario_prices_df = get_scenario_prices_df(
        period_start_date,
        period_end_date,
        use_hybrid_ids,
        cohort_reference_date,
        profits_reference,
        wallets_config['training_data']['dataset']
    )

    # Create proideafits_df for scenario performance scenario and calculate base features
    scenario_profits_df = generate_scenario_profits_df(
        scenario_prices_df,
        profits_df,
        market_indicators_df,
        period_end_date
    )
    scenario_trading_df = wtf.calculate_wallet_trading_features(
        scenario_profits_df, period_start_date, period_end_date)
    scenario_performance_df = wpf.calculate_performance_features(
        scenario_trading_df, wallets_config)
    try:
        u.assert_matching_indices(base_trading_df, scenario_trading_df)
        u.assert_matching_indices(base_performance_df, scenario_performance_df)
    except Exception as e:
        # surface context in the logs
        logger.error("Index mismatch between base/scenario features: %s", e, exc_info=True)
        raise                       # bubble the error so the epoch aborts
    del profits_df, market_indicators_df

    # Compute metrics comparing actual performance vs scenario timing performance
    lost_trading_df = calculate_scenario_trading_metrics(
        base_trading_df.copy(),
        scenario_trading_df,
        wallets_config
    )
    lost_performance_df = calculate_scenario_performance_metrics(
        base_performance_df.copy(),
        scenario_performance_df
    )
    scenario_features_df = lost_trading_df.join(lost_performance_df)

    return scenario_features_df




# -----------------------------------
#          Helper Functions
# -----------------------------------

@u.timing_decorator
def upload_profits_df_dates(
        profits_df: pd.DataFrame,
        profits_reference: str) -> None:
    """
    Uploads all coin/wallet/date combinations in profits_df to BigQuery temp table.

    Params:
    - profits_df (DataFrame): Source profits data
    - profits_reference (str): Modifier for the upload table name to avoid collisions.

    Uses the shared upload_dataframe utility to respect global concurrency limits.
    """
    # select relevant columns
    upload_df = profits_df[['coin_id', 'date', 'wallet_address']]

    project_id = 'western-verve-411004'
    table_id = f"{project_id}.temp.training_cohort_coin_dates_{profits_reference}"

    # define load job configuration with explicit schema and truncate semantics
    job_config = bigquery.LoadJobConfig(
        schema=[
            bigquery.SchemaField('coin_id', 'STRING'),
            bigquery.SchemaField('date', 'DATE'),
            bigquery.SchemaField('wallet_address', 'INTEGER')
        ],
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
    )

    # perform upload through centralized util (blocks if limit reached)
    bqu.upload_dataframe(
        dataframe=upload_df,
        destination_table=table_id,
        job_config=job_config
    )

    logger.info(f"Completed profits dates upload to {table_id}.")



@u.timing_decorator
def get_scenario_prices_df(
        training_period_start: str,
        training_period_end: str,
        use_hybrid_ids: bool,
        cohort_reference_date: str,
        profits_reference: str,
        dataset: str = 'prod'
    ) -> pd.DataFrame:
    """
    Get wallet transfer data with price ranges by querying BigQuery.

    Query output fields:
    - wallet_address,coin_id: same as input df
    - open_date: the earliest date within the period that the wallet could
        have transacted
    - date: the actual date the wallet transacted
    - open_days: how many days are between the open_date and date
    - max_price: the maximum token price during the open period
    - min_price: the minimum token price during the open period

    Params:
    - training_period_start (str): Start date for training period
    - training_period_end (str): End date for training period
    - use_hybrid_ids (bool): Whether to use hybrid_cw_ids instead of wallet_ids
    - cohort_reference_date (str): Reference date for the cohort temp table
    - profits_reference (str): Reference date for the profits pairs temp table
    - dataset (str): Determines whether to query core or dev_core schema

    Returns:
    - scenario_prices_df (DataFrame): Transfer data with min/max prices
    """
    # Define schema
    core_schema = 'core' if dataset == 'prod' else 'dev_core'


    # Different logic for date_ranges based on hybrid_ids setting
    if use_hybrid_ids:
        # for hybridized runs, the 'wallet_address' df column is the database 'hybrid_cw_id'
        wallet_addresses_cte = f"""
        with masked_wallet_addresses as (
            select
                -- mask hybrid_cw_id as "wallet_address" for data science pipeline
                xw_cw.hybrid_cw_id as wallet_address,
                wcd.coin_id,
                wcd.date
            from temp.training_cohort_coin_dates_{profits_reference} wcd
            -- the "wallet_address" in training cohort table is really a hybrid_cw_id
            join reference.wallet_coin_ids xw_cw on xw_cw.hybrid_cw_id = wcd.wallet_address
            join temp.wallet_modeling_training_cohort_{cohort_reference_date} wc
                on wc.wallet_address = xw_cw.wallet_address
            where wcd.date <= '{training_period_end}'
        )"""
    else:
        # for non-hybridized runs, the 'wallet_address' df column is the database 'wallet_id'
        wallet_addresses_cte = f"""
        with masked_wallet_addresses as (
            select
                -- mask wallet_id as "wallet_address"
                wc.wallet_id as wallet_address,
                wcd.coin_id,
                wcd.date

            -- the "wallet_address" in training cohort table is really a wallet_id
            from temp.wallet_modeling_training_cohort_{cohort_reference_date} wc
            join temp.training_cohort_coin_dates_{profits_reference} wcd on wcd.wallet_address = wc.wallet_id
            where wcd.date <= '{training_period_end}'
        )"""

    sql_query = f"""
        {wallet_addresses_cte}

        ,date_ranges as (
            select wallet_address,  -- masked wallet_id as "wallet_address"
            coin_id,
            COALESCE(
                -- selects the date of the previous transaction
                LAG(date) OVER (PARTITION BY wallet_address, coin_id ORDER BY date),

                -- if there is no previous transaction, use the starting balance date
                ('{training_period_start}' - interval 1 day)
            ) as open_date,
            date
            from masked_wallet_addresses
        )

        select dr.wallet_address  -- this is actually the ID selected in the CTE, not a wallet_address
        ,dr.coin_id
        ,dr.open_date
        ,dr.date
        ,DATE_DIFF(dr.date, dr.open_date, DAY) as open_days
        ,max(cmd.price) as max_price
        ,min(cmd.price) as min_price
        from date_ranges dr
        join {core_schema}.coin_market_data cmd on cmd.coin_id = dr.coin_id
            and (
                -- join prices
                (cmd.date between dr.open_date and dr.date)
                or (cmd.date = dr.date)
            )
        group by 1,2,3,4,5
        order by 1,2,4
        """
    scenario_prices_df = bqu.run_query(sql_query)

    # Handle column dtypes
    scenario_prices_df['coin_id'] = scenario_prices_df['coin_id'].astype('category')
    scenario_prices_df['date'] = scenario_prices_df['date'].astype('datetime64[ns]')
    scenario_prices_df = u.df_downcast(scenario_prices_df)

    # Confirm no nulls
    if scenario_prices_df.isna().sum().sum() > 0:
        raise ValueError(f"Null values found in scenario_prices_df. Review query:{sql_query}")

    logger.info(f"Retrieved scenario_prices_df with shape {scenario_prices_df.shape}")

    return scenario_prices_df



def append_profits_data(
        scenario_prices_df: pd.DataFrame,
        profits_df: pd.DataFrame,
        market_indicators_df: pd.DataFrame,
    ) -> pd.DataFrame:
    """
    Merge profits data with scenario prices, enforcing data consistency and type constraints.

    Params:
    - scenario_prices_df (DataFrame): Scenario prices to append
    - profits_df (DataFrame): Source profits data
    - market_indicators_df (DataFrame): df with coin_id,date,price columns

    Returns:
    - merged_df (DataFrame): Merged and validated transfers data
    """
    # Merge profits and market data
    profits_prices_df = (profits_df.copy()[
        ['date', 'wallet_address', 'coin_id', 'usd_balance', 'usd_net_transfers']]
        .merge(
            market_indicators_df[['date', 'coin_id', 'price']],
            on=['date', 'coin_id'],
            how='inner'
        ))
    if len(profits_prices_df) != len(profits_df):
        raise ValueError("Merge of profits_df and market_data_df did not fully align")

    # Merge scenario_prices_df
    merged_df = profits_prices_df.merge(
        scenario_prices_df.copy(),
        on=['date', 'coin_id', 'wallet_address'],
        how='inner'
    )
    if len(merged_df) != len(scenario_prices_df):
        raise ValueError("Merge of profits_df and scenario_prices_df did not fully align")

    # Add ratio columns
    merged_df['token_balance'] = merged_df['usd_balance'] / merged_df['price']
    merged_df['token_net_transfers'] = merged_df['usd_net_transfers'] / merged_df['price']

    # Validate output quality
    merged_df = u.df_downcast(merged_df)
    if merged_df.isna().sum().sum() > 0:
        raise ValueError("Null values found in merged output")

    return merged_df



def generate_scenario_profits_df(
        scenario_prices_df: pd.DataFrame,
        profits_df: pd.DataFrame,
        market_indicators_df: pd.DataFrame,
        period_end_date: str
    ) -> pd.DataFrame:
    """
    Generate scenario profits dataframe with adjusted transfer timing and final
        balance value.

    Params:
    - scenario_prices_df (DataFrame): DataFrame from get_scenario_transfers_df with
        min/max price columns
    - profits_df (DataFrame): Source profits data for merging actual transfer history
    - market_indicators_df (DataFrame): df with coin_id,date,price columns
    - period_end_date (str): End date to calculate scenario final balance

    Returns:
    - scenario_profits_df (DataFrame): Profits data with scenario transfers and balances
    """
    # Enrich with actual transfer history
    scenario_df = append_profits_data(
        scenario_prices_df,
        profits_df,
        market_indicators_df
    ).sort_index()

    # Calculate ideal transfers
    scenario_df['ideal_net_transfers'] = np.where(
        scenario_df['token_net_transfers'] < 0,
        # Ideal sells were sold at the maximum price
        scenario_df['token_net_transfers'] * scenario_df['max_price'],
        # Ideal buys were bought at the minimum price
        scenario_df['token_net_transfers'] * scenario_df['min_price']
    )

    # Calculate ideal ending balance
    scenario_df['ideal_usd_balance'] = np.where(
        scenario_df['date'] == period_end_date,
        # The ideal ending balance was cashed out at the max price
        scenario_df['token_balance'] * scenario_df['max_price'],
        scenario_df['usd_balance']
    )

    # Data quality checks
    validate_scenario_df(scenario_df)

    # Create final output dataframe
    scenario_profits_df = scenario_df[['date','wallet_address','coin_id']].copy()
    scenario_profits_df['usd_balance'] = scenario_df['ideal_usd_balance']
    scenario_profits_df['usd_net_transfers'] = scenario_df['ideal_net_transfers']
    scenario_profits_df['usd_inflows'] = np.where(
        scenario_profits_df['usd_net_transfers'] > 0,
        scenario_profits_df['usd_net_transfers'],
        0
    )
    scenario_profits_df['is_imputed'] = profits_df['is_imputed']

    # Add at the start of the function, after docstring
    null_check = scenario_profits_df[['date', 'wallet_address', 'coin_id']].isnull()
    if null_check.any().any():
        total_records = len(scenario_profits_df)
        null_counts = null_check.sum()
        raise ValueError(
            f"Null values detected in required columns. "
            f"Total records: {total_records:,}. "
            f"Nulls found - date: {null_counts['date']:,}, "
            f"wallet_address: {null_counts['wallet_address']:,}, "
            f"coin_id: {null_counts['coin_id']:,}"
        )

    return scenario_profits_df



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



def generate_scenario_features(
        scenario_prices_df: pd.DataFrame,
        period_start_date: str,
        period_end_date: str,
        wallets_config: dict
    ) -> pd.DataFrame:
    """
    Generate features for best and worst case selling scenarios.

    Params:
    - scenario_prices_df (DataFrame): Transfer data with min/max price columns
    - period_start_date (str): Start date of analysis period
    - period_end_date (str): End date of analysis period

    Returns:
    - scenario_features_df (DataFrame): Combined best/worst case features
    """
    # Generate best case sells scenario (sells at highest price)
    best_sells_profits_df = scenario_prices_df[['usd_balance']].assign(
        usd_net_transfers=np.where(
            scenario_prices_df['token_net_transfers'] < 0,
            scenario_prices_df['token_net_transfers'] * scenario_prices_df['max_price'],
            scenario_prices_df['usd_net_transfers']
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
    all_wallets = set(scenario_prices_df.index.get_level_values('wallet_address'))
    if len(all_wallets) != len(scenario_features_df):
        raise ValueError("Missing wallets identified in scenario_performance_features.")
    if scenario_features_df.isna().sum().sum() > 0:
        raise ValueError("Null values found in scenario_performance_features.")

    return scenario_features_df



def validate_scenario_df(ideal_df: pd.DataFrame) -> None:
    """
    Validates scenario_df for data integrity with floating point tolerance.

    Checks performed:
    1. Confirm the ideal USD balance is always >= than the base balance
    2. Confirms ideal USD balance is positive
    3. Confirms ideal_net_transfers are all <= than the base net transfers.
        If the wallet is buying, they should never pay more than the actual.
        If the wallet is selling, they should always receive more USD than actual,
            given that sells are reflected as negative numbers.

    Params:
    - ideal_df (DataFrame): DataFrame with ideal balance calculations

    Raises:
    - ValueError: If validation constraints are violated
    """
    # Calculate tolerance: max of 0.01% of value or 0.01
    balance_tolerance = np.maximum(ideal_df['usd_balance'] * 0.0001, 0.01)
    transfer_tolerance = np.maximum(ideal_df['usd_net_transfers'].abs() * 0.0001, 0.01)

    # Check for ideal_usd_balance < usd_balance (with tolerance)
    balance_violation = ideal_df['ideal_usd_balance'] < (ideal_df['usd_balance'] - balance_tolerance)
    if balance_violation.any():
        raise ValueError("Ideal USD balance cannot be less than actual USD balance. "
                         f"Found {balance_violation.sum()} violations.")

    # Check for negative ideal_usd_balance (with tolerance)
    negative_ideal = ideal_df['ideal_usd_balance'] < -0.01
    if negative_ideal.any():
        raise ValueError("Ideal USD balance cannot be negative. "
                         f"Found {negative_ideal.sum()} violations.")

    # Check for ideal_net_transfers > usd_net_transfers (with tolerance)
    transfer_violation = ideal_df['ideal_net_transfers'] > (ideal_df['usd_net_transfers'] + transfer_tolerance)
    if transfer_violation.any():
        raise ValueError("Ideal net transfers cannot exceed actual net transfers. "
                         f"Found {transfer_violation.sum()} violations.")



def calculate_scenario_trading_metrics(
        base_trading_df: pd.DataFrame,
        scenario_trading_df: pd.DataFrame,
        wallets_config: dict
    ) -> pd.DataFrame:
    """
    Calculates underperformance in trading performance vs scenario timed trades.

    Params:
    - base_trading_df (DataFrame): actual trading metrics
    - scenario_trading_df (DataFrame): scenario trading metrics
    - wallets_config (dict): config with usd_materiality and returns_winsorization

    Returns:
    - lost_df (DataFrame): lost trading metrics with ranks
    """
    u.assert_matching_indices(base_trading_df, scenario_trading_df)

    # Define dfs and vars
    lost_df = pd.DataFrame()
    lost_df.index = base_trading_df.index
    trading_diff_df = base_trading_df.sort_index() - scenario_trading_df.sort_index()
    usd_materiality = wallets_config['features']['usd_materiality']
    winsorization_pct = wallets_config['features']['returns_winsorization']

    def add_lost_metric(lost_df: pd.DataFrame, diff_col: str, base_col: str, metric_name: str) -> None:
        """Calculate winsorized loss metric with rank."""
        lost_df[metric_name] = abs(trading_diff_df[diff_col])
        pct_col = f"{metric_name}_pct"
        # Calculate winsorized percent diff for material wallets
        lost_df[pct_col] = u.winsorize(np.where(
            base_trading_df[base_col] >= usd_materiality,
            lost_df[metric_name] / base_trading_df[base_col], 0
        ), winsorization_pct)
        lost_df[f"{pct_col}_rank"] = lost_df[pct_col].rank(pct=True)

    # Lost overpays: how much they paid relative to the min price
    add_lost_metric(lost_df, 'crypto_cash_buys', 'crypto_cash_buys', 'lost_overpays')

    # Lost undersells: how much they sold for relative to the max price
    add_lost_metric(lost_df, 'crypto_cash_sells', 'crypto_cash_sells', 'lost_undersells')

    # Lost gains: how much they *could have* sold for at the max price
    add_lost_metric(lost_df, 'crypto_outflows', 'crypto_outflows', 'lost_gains')

    # Lost net gains: total net losses from overpays and ROI on max investment
    add_lost_metric(lost_df, 'crypto_net_gain', 'max_investment', 'lost_net_gains')
    lost_df.rename(columns={
        'lost_net_gains_pct': 'lost_roi',
        'lost_net_gains_pct_rank': 'lost_roi_pct_rank'
    }, inplace=True)

    # Lost total product: lost overpays * lost gains, which penalizes double losses more
    lost_df['lost_combined_pct'] = u.winsorize(
        ((lost_df['lost_overpays_pct']+1) * (lost_df['lost_gains_pct']+1)) - 1,
        winsorization_pct)
    lost_df['lost_combined_pct_rank'] = lost_df['lost_combined_pct'].rank(pct=True)

    return lost_df



def calculate_scenario_performance_metrics(
        base_performance_df: pd.DataFrame,
        scenario_performance_df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Params:
    - base_performance_df (DataFrame): actual performance metrics
    - scenario_performance_df (DataFrame): scenario performance metrics

    Returns:
    - lost_performance_df (DataFrame): lost performance metrics with renamed columns
    """
    # Calculate lost performance for all metrics
    lost_performance_df = base_performance_df.sort_index() - scenario_performance_df.sort_index()

    include_metrics = {
        "crypto_net_gain/max_investment": "lost_net_gain_max_inv",
        "crypto_net_gain/crypto_inflows": "lost_net_gain_inflows",
    }

    # Filter to only included cols
    filtered_cols = [col for col in lost_performance_df.columns
                     if any(col.startswith(metric) for metric in include_metrics)]
    lost_performance_df = lost_performance_df[filtered_cols]

    # Rename cols to abbreviated prefixes
    rename_map = {}
    for old_pref, new_pref in include_metrics.items():
        for col in lost_performance_df.columns:
            if col.startswith(f"{old_pref}/"):
                # keep the suffix (e.g. "base", "winsorized", "rank")
                suffix = col[len(old_pref) + 1 :]
                rename_map[col] = f"{new_pref}/{suffix}"

    lost_performance_df = lost_performance_df.rename(columns=rename_map)

    return lost_performance_df
