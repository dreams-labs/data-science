"""
Calculates metrics aggregated at the wallet-coin level
"""

import logging
import pandas as pd
from dreams_core.googlecloud import GoogleCloud as dgc

# set up logger at the module level
logger = logging.getLogger(__name__)



def add_cash_flow_transfers_logic(profits_df):
    """
    Adds a cash_flow_transfers column to profits_df that can be used to compute
    the wallet's gain and investment amount by converting their starting and ending
    balances to cash flow equivilants.

    Params:
    - profits_df (df): profits_df that needs wallet investment peformance computed
        based on the earliest and latest dates in the df.

    Returns:
    - adj_profits_df (df): input df with the cash_flow_transfers column added
    """

    def adjust_end_transfers(df, target_date):
        df.loc[df['date'] == target_date, 'cash_flow_transfers'] -= df.loc[df['date'] == target_date, 'usd_balance']
        return df

    def adjust_start_transfers(df, target_date):
        df.loc[df['date'] == target_date, 'cash_flow_transfers'] = df.loc[df['date'] == target_date, 'usd_balance']
        return df

    # Copy df and add cash flow column
    adj_profits_df = profits_df.copy()
    adj_profits_df['cash_flow_transfers'] = adj_profits_df['usd_net_transfers']

    # Modify the records on the start and end dates to reflect the balances
    start_date = adj_profits_df['date'].min()
    end_date = adj_profits_df['date'].max()

    adj_profits_df = adjust_start_transfers(adj_profits_df,start_date)
    adj_profits_df = adjust_end_transfers(adj_profits_df,end_date)

    return adj_profits_df



def retrieve_buyer_numbers():
    """
    Returns the buyer number for each wallet-coin pairing, where the first buyer
    receives rank 1 and the count increases for each subsequence wallet.

    Buyer numbers are calculated for all wallets but the returned df only includes
    wallets that were uploaded to the temp.wallet_modeling_cohort table.

    Returns:
    - buyer_numbers_df (df): dataframe showing that buyer number a wallet was for
        the associated coin_id.
    """
    buyer_numbers_sql = """
        with transaction_rank as (
            select coin_id
            ,wallet_address
            ,min(date) as first_transaction
            from core.coin_wallet_transfers cwt
            group by 1,2
        ),

        buy_ordering as (
            select tr.coin_id
            ,tr.wallet_address
            ,rank() over (partition by coin_id order by first_transaction asc) as buyer_number
            from transaction_rank tr
        )

        select wc.wallet_id
        ,o.coin_id
        ,o.buyer_number
        from buy_ordering o
        join reference.wallet_ids xw on xw.wallet_address = o.wallet_address
        join temp.wallet_modeling_cohort wc on wc.wallet_id = xw.wallet_id
        """

    buyer_numbers_df = dgc().run_sql(buyer_numbers_sql)
    logger.info("Retrieved buyer numbers for %s wallet-coin pairs.", len(buyer_numbers_df))

    return buyer_numbers_df



def calculate_timing_features_for_column(df, metric_column):
    """
    Calculate timing features for a single metric column from pre-merged DataFrame.

    Args:
        df (pd.DataFrame): Pre-merged DataFrame with columns [wallet_address, usd_net_transfers, metric_column]
        metric_column (str): Name of the column to analyze

    Returns:
        pd.DataFrame: DataFrame indexed by wallet_address with columns:
            - {metric_column}_buy_weighted
            - {metric_column}_buy_mean
            - {metric_column}_sell_weighted
            - {metric_column}_sell_mean
    """
    # Split into buys and sells
    buys = df[df['usd_net_transfers'] > 0].copy()
    sells = df[df['usd_net_transfers'] < 0].copy()

    features = pd.DataFrame(index=df['wallet_address'].unique())

    # Vectorized buy calculations
    if not buys.empty:
        # Regular mean
        features[f"{metric_column}_buy_mean"] = (
            buys.groupby('wallet_address')[metric_column].mean()
        )

        # Weighted mean: First compute the products, then group
        buys['weighted_values'] = buys[metric_column] * abs(buys['usd_net_transfers'])
        weighted_sums = buys.groupby('wallet_address')['weighted_values'].sum()
        weight_sums = buys.groupby('wallet_address')['usd_net_transfers'].apply(abs).sum()
        features[f"{metric_column}_buy_weighted"] = weighted_sums / weight_sums

    # Similar for sells
    if not sells.empty:
        features[f"{metric_column}_sell_mean"] = (
            sells.groupby('wallet_address')[metric_column].mean()
        )

        sells['weighted_values'] = sells[metric_column] * abs(sells['usd_net_transfers'])
        weighted_sums = sells.groupby('wallet_address')['weighted_values'].sum()
        weight_sums = sells.groupby('wallet_address')['usd_net_transfers'].apply(abs).sum()
        features[f"{metric_column}_sell_weighted"] = weighted_sums / weight_sums

    return features



def generate_all_timing_features(
    profits_df,
    market_timing_df,
    relative_change_columns,
    min_transaction_size=0
):
    """
    Generate timing features for multiple market metric columns.

    Args:
        profits_df (pd.DataFrame): DataFrame with columns [coin_id, date, wallet_address, usd_net_transfers]
        market_timing_df (pd.DataFrame): DataFrame with market timing metrics indexed by (coin_id, date)
        relative_change_columns (list): List of column names from market_timing_df to analyze
        min_transaction_size (float): Minimum absolute USD value of transaction to consider

    Returns:
        pd.DataFrame: DataFrame indexed by wallet_address with generated features for each input column
    """
    # Filter by minimum transaction size
    filtered_profits = profits_df[
        abs(profits_df['usd_net_transfers']) >= min_transaction_size
    ].copy()

    # Perform the merge once
    timing_profits_df = filtered_profits.merge(
        market_timing_df[relative_change_columns + ['coin_id', 'date']],
        on=['coin_id', 'date'],
        how='left'
    )

    # Initialize empty result with wallet_address index
    all_features = []

    # Calculate features for each column
    for col in relative_change_columns:
        logger.info("Generating timing performance features for %s...", col)
        col_features = calculate_timing_features_for_column(
            timing_profits_df,
            col
        )
        all_features.append(col_features)

    # Combine all feature sets
    if all_features:
        result = pd.concat(all_features, axis=1)
    else:
        # Return empty DataFrame with correct index if no features generated
        result = pd.DataFrame(index=filtered_profits['wallet_address'].unique())

    return result
