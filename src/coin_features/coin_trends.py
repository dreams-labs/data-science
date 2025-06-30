"""
Functions for generating flattened macroeconomic and market time series features for coins.

This module provides utilities to aggregate and transform macroeconomic and market indicator
time series data into single-row, coin-level features. These features are designed to be
cross-joined onto each coin's record for downstream modeling and analysis.

Main functionalities include:
- Generating flattened macroeconomic features from time series data.
- Generating flattened market features from time series data.
- Renaming feature columns for clarity and consistency.

These functions support the feature engineering pipeline for coin-level predictive modeling.
"""
import os
import logging
import pandas as pd

# Local module imports
import feature_engineering.flattening as flt
from utilities import bq_utils as bqu
import utils as u
from utils import ConfigError

# set up logger at the module level
logger = logging.getLogger(__name__)



# --------------------------------------
#        Features Main Interface
# --------------------------------------

def generate_coin_trends_features(
    wallets_config: dict,
    period_end_date: str,
    coin_trends_metrics_config: dict
) -> pd.DataFrame:
    """
    Generates flattened coin trends data time series features for coins.

    Params:
    - wallets_config (dict): from yaml
    - period_end_date (str): latest date to include in the source data
    - coin_trends_metrics_config (dict): defines the time series features that will
        be output. Defined at wallets_coins_metrics_config['time_series']['coin_trends']

    Returns:
    - trends_features_df (df): coin_id-indexed features about coin holder and
        price trends. Column details are provided in the docstring of
        retrieve_comprehensive_coin_trends().
    """
    # Retrieve comprehensive file
    comprehensive_trends_df = retrieve_comprehensive_coin_trends(
        wallets_config['features']['usd_materiality'],
        wallets_config['training_data']['reference_dfs_folder'],
        wallets_config['training_data']['dataset'],
        wallets_config['features'].get('force_rebuild_coin_trends', False)
    )

    # Filter on end date
    trends_df = comprehensive_trends_df[
        comprehensive_trends_df.index.get_level_values('date') <= period_end_date]

    # Calculate all specified metrics
    trends_features_df = flt.flatten_coin_date_df(
        trends_df.reset_index(),
        coin_trends_metrics_config,
        period_end_date
    )
    trends_features_df = trends_features_df.set_index('coin_id')

    # Remove '_last' suffixes from the columns
    mapping = {
        col: col[:-5]
        for col in trends_features_df.columns
        if col.endswith("_last")
    }
    trends_features_df.rename(columns=mapping)

    return trends_features_df




# -------------------------------
#        Helper Functions
# -------------------------------

def validate_metrics_config(
    metrics_config: dict
) -> list:
    """
    # Confirms all metrics are only _last. Other metrics could be calculated
    #  simply by modifying the config, but the current code logic assumes each
    #  column only has a single last aggregation.

    Params:
    - metrics_config (dict): mapping metric names to their config dict.

    Returns:
    - invalid_metrics (list): metric names that don’t follow aggregations.last structure.
    """
    invalid_metrics = []
    for metric_name, cfg in metrics_config.items():
        # check there's only an 'aggregations' key
        if set(cfg.keys()) != {"aggregations"}:
            invalid_metrics.append(metric_name)
            continue

        aggr = cfg["aggregations"]
        # check there's only a 'last' key under aggregations
        if set(aggr.keys()) != {"last"}:
            invalid_metrics.append(metric_name)
            continue

        last = aggr["last"]
        # check there's only a 'scaling' key under last
        if set(last.keys()) != {"scaling"}:
            invalid_metrics.append(metric_name)

    if len(invalid_metrics) > 0:
        raise ConfigError("Non-last metrics found.")


def retrieve_comprehensive_coin_trends(
        materiality: int,
        reference_dfs_folder: str,
        dataset: str = 'prod',
        force_rebuild: bool = False
    ) -> pd.DataFrame:
    """
    Downloads and stores a comprehensive list of every coin's lifetime trends for
     holders and current price vs its all-time high. The results are stored to parquet
     rather than returned.

    This query is expensive (~$2 per run) so we will be archiving comprehensive results
     that can then be filtered for the actual modeling periods. This means changes in
     materiality will not be reflected until the full query is rerun.

    Params:
    - materiality (int): Wallets with balances below this level will not count as holders,
        and wallets with profits below it will not count towards current_holders_in_profit.
    - reference_dfs_folder (str): Where to save the parquet file
    - dataset (str): can be set to 'dev' to query dev_core schema. If 'dev', the comprehensive
        file will be stored with a '_dev' suffix.
    - force_rebuild (bool): If True, the file will be regenerated regardless of whether a
        version exists already.

    Returns:
    - coin_trends_df (df): df with the below structure that includes records for all dates in
        the database.

    Structure of coin_trends_df
    * Multiindex
        - coin_id (categorical)
        - date (datetime)
    * Feature columns
        - current_holders (int): number of wallets with balance > materiality.
        - current_holders_in_profit (int): wallets with balance > materiality and
            cumulative profit > materiality.
        - current_holders_in_profit_pct (float): current_holders_in_profit ÷ current_holders.
        - lifetime_holders (int): total unique wallets ever above materiality to date.
        - current_holders_pct_of_lifetime (float): current_holders ÷ lifetime_holders.
        - days_since_launch (int): days since the coin’s first material transfer.
        - days_since_ath (int): days since the coin’s all-time-high price.
    """
    if dataset not in ['prod','dev']:
        raise ValueError(f"Invalid dataset value '{dataset}' found. Dataset must be 'prod' or 'dev'")

    # Identify file location
    suffix = '_dev' if dataset == 'dev' else ''
    output_file = f"{reference_dfs_folder}/comprehensive_coin_trends{suffix}.parquet"

    # Escape: return existing file if override isn't configured
    if os.path.exists(output_file):
        if not force_rebuild:
            logger.info(f"Returning existing coin trends file at '{output_file}' skipping query.")

            return pd.read_parquet(output_file)
        else:
            logger.warning("Force rebuilding comprehensive_coin_trends.parquet to override "
                           f"existing file at '{output_file}'.")

    core_schema = 'core' if dataset == 'prod' else 'dev_core'

    coin_series_sql = f"""
        -- ALL TIME HIGH LOGIC --
        WITH running_ath AS (
            SELECT
                coin_id,
                date,
                price,
                MAX(price) OVER (
                    PARTITION BY coin_id
                    ORDER BY date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS ath_price
            FROM
                {core_schema}.coin_market_data
        ),

        days_since_ath AS (
            SELECT
                coin_id,
                date,
                price,
                ath_price,
                DATE_DIFF(
                    date,
                    LAST_VALUE(
                        CASE
                            WHEN price = ath_price THEN date
                            ELSE NULL
                        END
                        IGNORE NULLS
                    ) OVER (
                        PARTITION BY coin_id
                        ORDER BY date
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ),
                    DAY
                ) AS days_since_ath,
                CASE
                    WHEN ath_price > 0 THEN price / ath_price
                    ELSE NULL
                END AS price_pct_of_ath
            FROM
                running_ath
        ),

        -- FIRST TRANSFER LOGIC --
        first_transfer AS (
            SELECT
                coin_id,
                MIN(date) AS launch_date
            FROM
                {core_schema}.coin_wallet_profits
            WHERE
                usd_balance >= {materiality}
            GROUP BY
                coin_id
        ),

        -- CURRENT HOLDERS DELTAS LOGIC --
        -- 1) bring in prev_day balances
        wallet_history AS (
            SELECT
                coin_id,
                date,
                wallet_address,
                usd_balance,
                profits_cumulative,
                LAG(usd_balance) OVER (
                    PARTITION BY coin_id, wallet_address
                    ORDER BY date
                ) AS prev_usd_balance,
                LAG(profits_cumulative) OVER (
                    PARTITION BY coin_id, wallet_address
                    ORDER BY date
                ) AS prev_profits_cumulative
            FROM
                {core_schema}.coin_wallet_profits
        ),

        -- 2) compute net holder/profit changes per day
        daily_changes AS (
            SELECT
                coin_id,
                date,
                -- USD balance crossings (treat NULL prev as 0)
                SUM(
                    CASE
                        WHEN COALESCE(prev_usd_balance, 0) <= {materiality}
                            AND usd_balance > {materiality} THEN 1
                        WHEN COALESCE(prev_usd_balance, 0) > {materiality}
                            AND usd_balance <= {materiality} THEN -1
                        ELSE 0
                    END
                ) AS net_holder_change,
                -- joint balance + profit state crossings (null-safe)
                SUM(
                    CASE
                        WHEN (COALESCE(prev_profits_cumulative, 0) <= {materiality}
                            OR COALESCE(prev_usd_balance, 0) <= {materiality})
                            AND profits_cumulative > {materiality}
                            AND usd_balance > {materiality} THEN 1
                        WHEN COALESCE(prev_profits_cumulative, 0) > {materiality}
                            AND COALESCE(prev_usd_balance, 0) > {materiality}
                            AND (profits_cumulative <= {materiality}
                                OR usd_balance <= {materiality}) THEN -1
                        ELSE 0
                    END
                ) AS net_profit_change
            FROM
                wallet_history
            GROUP BY
                coin_id,
                date
        ),

        -- 3) running totals on tx-days
        daily_totals AS (
            SELECT
                coin_id,
                date,
                SUM(net_holder_change) OVER (
                    PARTITION BY coin_id
                    ORDER BY date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS current_holders,
                SUM(net_profit_change) OVER (
                    PARTITION BY coin_id
                    ORDER BY date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS current_holders_in_profit
            FROM
                daily_changes
        ),

        -- 4) join to full date list and forward‐fill
        holder_metrics AS (
            SELECT
                dsa.coin_id,
                dsa.date,
                dt.current_holders,
                dt.current_holders_in_profit,
                LAST_VALUE(dt.current_holders IGNORE NULLS) OVER (
                    PARTITION BY dsa.coin_id
                    ORDER BY dsa.date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS current_holders_filled,
                LAST_VALUE(dt.current_holders_in_profit IGNORE NULLS) OVER (
                    PARTITION BY dsa.coin_id
                    ORDER BY dsa.date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS current_holders_in_profit_filled
            FROM
                days_since_ath dsa
            LEFT JOIN
                daily_totals dt USING (coin_id, date)
        ),

        -- NEW AND LIFETIME HOLDERS LOGIC --
        -- 1) compute how many new holders joined per day
        holder_entries AS (
            SELECT
                coin_id,
                date,
                SUM(
                    CASE
                        WHEN COALESCE(prev_usd_balance, 0) <= {materiality}
                            AND usd_balance > {materiality} THEN 1
                        ELSE 0
                    END
                ) AS new_holders
            FROM
                wallet_history
            GROUP BY
                coin_id,
                date
        ),

        -- 2) running total of those new_holders
        holder_metrics_with_lifetime AS (
            SELECT
                dsa.coin_id,
                dsa.date,
                SUM(COALESCE(he.new_holders, 0)) OVER (
                    PARTITION BY dsa.coin_id
                    ORDER BY dsa.date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS lifetime_holders
            FROM
                days_since_ath dsa
            LEFT JOIN
                holder_entries he USING (coin_id, date)
        )

        SELECT
            hm.coin_id,
            hm.date,
            hm.current_holders_filled           AS current_holders,
            hm.current_holders_in_profit_filled  AS current_holders_in_profit,
            COALESCE(
                SAFE_DIVIDE(
                    hm.current_holders_in_profit_filled,
                    hm.current_holders_filled
                ),
                0
            )                                  AS current_holders_in_profit_pct,
            hml.lifetime_holders,
            COALESCE(
                SAFE_DIVIDE(
                    hm.current_holders_filled,
                    hml.lifetime_holders
                ),
                0
            )                                  AS current_holders_pct_of_lifetime,
            -- if the first day had only immaterial transfers this can be negative
            DATE_DIFF(hm.date, ft.launch_date, DAY) AS days_since_launch,
            ath.days_since_ath,
            ath.price_pct_of_ath
        FROM
            holder_metrics hm
        JOIN
            days_since_ath ath USING (coin_id, date)
        JOIN
            first_transfer ft  USING (coin_id)
        JOIN
            holder_metrics_with_lifetime hml USING (coin_id, date)
        WHERE
            hml.lifetime_holders > 0
        ORDER BY
            hm.coin_id,
            hm.date;
        """

    print(coin_series_sql) # TODO: remove print

    logger.info(f"<{dataset.upper()}> Retrieving comprehensive coin trends data...")
    cs_df = bqu.run_query(coin_series_sql)

    # Convert coin_id to categorical
    cs_df['coin_id'] = cs_df['coin_id'].astype('category')

    # Set index
    cs_df = cs_df.set_index(['coin_id','date'])

    # Data Validation
    # ---------------
    # 1) Confirm that for each coin, row_count == (max_date – min_date + 1)
    diffs = (
        cs_df
        .reset_index()  # bring coin_id,date back to columns
        .groupby('coin_id', observed=False)['date']
        .agg(min_date='min', max_date='max', row_count='count')
    )
    diffs['expected_count'] = (diffs['max_date'] - diffs['min_date']).dt.days + 1

    # find any coins that don’t line up
    bad = diffs[diffs['row_count'] != diffs['expected_count']]
    if not bad.empty:
        bad_coins = bad.index.tolist()
        raise ValueError(
            f"Date continuity check failed for {len(bad_coins)} coins: {bad_coins[:5]}…"
        )

    # 2) Validate no negatives in any numeric column
    numeric_cols = cs_df.select_dtypes(include=["number"]).columns
    neg_counts = (cs_df[numeric_cols] < 0).sum()
    # keep only columns with one or more negatives
    neg_counts = neg_counts[neg_counts > 0]
    if not neg_counts.empty:
        # format like “current_holders: 3; days_since_ath: 1”
        details = "; ".join(f"{col}: {count}" for col, count in neg_counts.items())
        raise ValueError(f"Negative values detected in comprehensive_coin_trends: {details}")

    # Save to reference folder
    u.to_parquet_safe(cs_df, output_file)
    logger.info(f"Stored comprehensive coin_trends_df with shape {cs_df.shape} to {output_file}.")

    return cs_df

