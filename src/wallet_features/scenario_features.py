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


# -----------------------------------
#       Main Interface Function
# -----------------------------------

@u.timing_decorator
def calculate_scenario_features(
    training_profits_df: pd.DataFrame,
    training_market_indicators_df: pd.DataFrame,
    period_start_date: str,
    period_end_date: str
) -> pd.DataFrame:
    """
    Calculates scenario wallet transfer features based on ideal price points.

    Params:
    - training_profits_df (DataFrame): Historical profits data
    - training_market_indicators_df (DataFrame): df with coin_id,date,price columns
    - period_start_date (str): Start of analysis period
    - period_end_date (str): End of analysis period

    Returns:
    - scenario_features_df (DataFrame): Wallet-level scenario transfer features
    """
    # Upload profits data to temporary storage
    upload_profits_df_dates(training_profits_df)

    # Calculate ideal price points for each transfer
    ideal_transfers_df = get_ideal_transfers_df(period_end_date)

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
        period_end_date
    )
    scenario_features_df = performance_features_df

    # FeatureRemoval not predictive
    # # Generate ratio features
    # performance_ratios_df = generate_delay_wtd_relative_performance(ideal_transfers_df)
    # scenario_features_df = performance_features_df.join(performance_ratios_df,how='inner')

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
def get_ideal_transfers_df(training_period_end: str) -> pd.DataFrame:
    """
    Get wallet transfer data with price ranges.

    Params:
    - training_starting_balance_date (str): Starting balance date for training period
    - training_period_end (str): End date for training period

    Returns:
    - ideal_transfers_df (DataFrame): Transfer data with min/max prices
    """
    sql_query = f"""
        with date_ranges as (
            select
                wc.wallet_id,
                wcd.coin_id,
                wcd.date,
                COALESCE(
                    LEAD(wcd.date) OVER (PARTITION BY xw.wallet_address, wcd.coin_id ORDER BY wcd.date) - interval 1 day,
                    '{training_period_end}'
                ) as date_range
            from temp.wallet_modeling_training_cohort wc
            join temp.training_cohort_coin_dates wcd on wcd.wallet_address = wc.wallet_id
            join reference.wallet_ids xw on xw.wallet_id = wc.wallet_id
            join core.coin_market_data cmd on cmd.coin_id = wcd.coin_id
                and cmd.date = wcd.date
            where wcd.date <= '{training_period_end}'
        )

        select dr.wallet_id as wallet_address
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
        order by date,wallet_id,coin_id
        """
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
def generate_scenario_performance(scenario_profits_df: pd.DataFrame,
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
    features = wallets_config['features']['scenario_performance_features']
    if len(features) != len(set(features)):
        raise ValueError("Duplicate features detected in scenario_performance_features")
    scenario_features_df = scenario_performance_df[features]

    # Remove '/' delimiters for better importance analysis parsing
    scenario_features_df.columns = scenario_features_df.columns.str.replace('/', '_')

    return scenario_features_df



@u.timing_decorator
def generate_scenario_features(ideal_transfers_df: pd.DataFrame,
                               period_start_date: str,
                               period_end_date: str) -> pd.DataFrame:
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
    best_sells_features = generate_scenario_performance(best_sells_profits_df, period_start_date, period_end_date)
    best_sells_features = best_sells_features.add_prefix('sells_best/')

    # Generate worst case buys scenario (sells at lowest price)
    worst_sells_profits_df = ideal_transfers_df[['usd_balance']].assign(
        usd_net_transfers=np.where(
            ideal_transfers_df['token_net_transfers'] < 0,
            ideal_transfers_df['token_net_transfers'] * ideal_transfers_df['min_price'],
            ideal_transfers_df['usd_net_transfers']
        )
    )
    worst_sells_features = generate_scenario_performance(worst_sells_profits_df, period_start_date, period_end_date)
    worst_sells_features = worst_sells_features.add_prefix('sells_worst/')

    # Generate best case buys scenario (buys at lowest price)
    best_buys_profits_df = ideal_transfers_df[['usd_balance']].assign(
        usd_net_transfers=np.where(
            ideal_transfers_df['token_net_transfers'] > 0,
            ideal_transfers_df['token_net_transfers'] * ideal_transfers_df['min_price'],
            ideal_transfers_df['usd_net_transfers']
        )
    )
    best_buys_features = generate_scenario_performance(best_buys_profits_df, period_start_date, period_end_date)
    best_buys_features = best_buys_features.add_prefix('buys_best/')

    # Generate worst case buys scenario (buys at highest price)
    worst_buys_profits_df = ideal_transfers_df[['usd_balance']].assign(
        usd_net_transfers=np.where(
            ideal_transfers_df['token_net_transfers'] > 0,
            ideal_transfers_df['token_net_transfers'] * ideal_transfers_df['max_price'],
            ideal_transfers_df['usd_net_transfers']
        )
    )
    worst_buys_features = generate_scenario_performance(worst_buys_profits_df, period_start_date, period_end_date)
    worst_buys_features = worst_buys_features.add_prefix('buys_worst/')

    # Merge all together
    scenario_features_df = best_sells_features
    scenario_features_df = pd.concat([scenario_features_df, worst_sells_features], axis=1)
    scenario_features_df = pd.concat([scenario_features_df, best_buys_features], axis=1)
    scenario_features_df = pd.concat([scenario_features_df, worst_buys_features], axis=1)

    # Data quality checks
    all_wallets = set(ideal_transfers_df.index.get_level_values('wallet_address'))
    if len(all_wallets) != len(scenario_features_df):
        raise ValueError("Missing wallets identified in scenario_performance_features.")
    if scenario_features_df.isna().sum().sum() > 0:
        raise ValueError("Null values found in scenario_performance_features.")

    return scenario_features_df



# FeatureRemoval not predictive
# def generate_delay_wtd_relative_performance(ideal_transfers_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Calculates delay-weighted relative performance metrics for buys and sells.

#     Params:
#     - ideal_transfers_df (DataFrame): Must contain columns: usd_net_transfers, min_price,
#       max_price, price, days_until_next_transfer, wallet_address

#     Returns:
#     - DataFrame: Wallet-level metrics with columns:
#       - buy_delay_wtd_performance: Buy timing efficiency weighted by delay until next transfer
#       - sell_delay_wtd_performance: Sell timing efficiency weighted by delay until next transfer
#     """
#     relative_df = pd.DataFrame(index=ideal_transfers_df.index)

#     # Buy efficiency: ratio of minimum price to actual purchase price
#     is_buy = ideal_transfers_df['usd_net_transfers'] > 0
#     relative_df['buy_pct_of_ideal'] = np.where(
#         is_buy,
#         ideal_transfers_df['min_price'] / ideal_transfers_df['price'],
#         np.nan
#     )
#     relative_df['buy_delays'] = np.where(
#         is_buy,
#         ideal_transfers_df['days_until_next_transfer'],
#         np.nan
#     )
#     relative_df['buy_wtd'] = (
#         relative_df['buy_pct_of_ideal'] *
#         relative_df['buy_delays']
#     )

#     # Sell efficiency: ratio of actual sale price to maximum price
#     is_sell = ideal_transfers_df['usd_net_transfers'] < 0
#     relative_df['sell_pct_of_ideal'] = np.where(
#         is_sell,
#         ideal_transfers_df['price'] / ideal_transfers_df['max_price'],
#         np.nan
#     )
#     relative_df['sell_delays'] = np.where(
#         is_sell,
#         ideal_transfers_df['days_until_next_transfer'],
#         np.nan
#     )
#     relative_df['sell_wtd'] = (
#         relative_df['sell_pct_of_ideal'] *
#         relative_df['sell_delays']
#     )

#     # Aggregate to wallet level
#     wtd_avgs_df = relative_df.reset_index().groupby('wallet_address').agg({
#         'buy_wtd': 'sum',
#         'buy_delays': 'sum',
#         'sell_wtd': 'sum',
#         'sell_delays': 'sum',
#     })

#     # Safe division for weighted performance ratios
#     wtd_avgs_df['buy_delay_wtd_performance'] = np.where(
#         wtd_avgs_df['buy_delays'] > 0,
#         wtd_avgs_df['buy_wtd'] / wtd_avgs_df['buy_delays'],
#         np.nan
#     )
#     wtd_avgs_df['sell_delay_wtd_performance'] = np.where(
#         wtd_avgs_df['sell_delays'] > 0,
#         wtd_avgs_df['sell_wtd'] / wtd_avgs_df['sell_delays'],
#         np.nan
#     )

#     # Validate relative performance metrics are between 0 and 1
#     for col in ['buy_delay_wtd_performance', 'sell_delay_wtd_performance']:
#         if not (
#             wtd_avgs_df[col].dropna().between(0, 1, inclusive='both').all()
#         ):
#             raise ValueError(
#                 f"Found {col} values outside expected range [0,1]. "
#                 f"Min: {wtd_avgs_df[col].min():.3f}, Max: {wtd_avgs_df[col].max():.3f}"
#             )

#     return wtd_avgs_df[['buy_delay_wtd_performance', 'sell_delay_wtd_performance']]



# FeatureRemoval not predictive
# def add_scenario_vs_base_columns(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Adds columns to the DataFrame by dividing each 'category/metric' column
#     by the corresponding 'base/metric' column to calculate category vs base ratios.

#     Params:
#     - df (pd.DataFrame): Input DataFrame with category/metric column format.

#     Returns:
#     - pd.DataFrame: Updated DataFrame with additional ratio columns.
#     """
#     # Extract unique category prefixes and metric suffixes
#     categories = set(col.split('/', 1)[0] for col in df.columns if '/' in col)
#     metric_suffixes = set(col.split('/', 1)[1] for col in df.columns if '/' in col)

#     # Ensure 'base/' exists in categories
#     if 'base' not in categories:
#         raise ValueError("The DataFrame must contain 'base/' category columns for comparison.")

#     # Loop through categories excluding 'base' and calculate the ratios
#     for category in categories - {'base'}:
#         for suffix in metric_suffixes:
#             base_col = f"base/{suffix}"
#             category_col = f"{category}/{suffix}"
#             new_col = f"{category}_v_base/{suffix}"

#             # Check if both columns exist before dividing
#             if base_col in df.columns and category_col in df.columns:
#                 # Add the ratio column
#                 df[new_col] = df[category_col] - df[base_col]
#             else:
#                 raise KeyError(f"Required columns missing: {base_col} or {category_col}")

#     # Drop all 'base/' columns
#     df = df.drop(columns=[col for col in df.columns if col.startswith('base/')])

#     return df
