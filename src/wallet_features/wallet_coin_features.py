"""
Calculates metrics aggregated at the wallet-coin level
"""

import logging
import pandas as pd
from dreams_core.googlecloud import GoogleCloud as dgc

# Local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig

# set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()


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



def retrieve_transfers_data():
    """
    Returns the buyer number for each wallet-coin pairing, where the first buyer
    receives rank 1 and the count increases for each subsequence wallet.

    Buyer numbers are calculated for all wallets but the returned df only includes
    wallets that were uploaded to the temp.wallet_modeling_cohort table.

    Returns:
    - buyer_numbers_df (df): dataframe showing that buyer number a wallet was for
        the associated coin_id.
    """
    # Wallet transactions below this threshold will not be included in the buyer sequencing
    minimum_transaction_size = wallets_config['features']['timing_metrics_min_transaction_size']

    # All data after the training period must be ignored to avoid data leakage
    training_period_end = wallets_config['training_data']['training_period_end']

    transfers_data_sql = f"""
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

        select wc.wallet_id
        ,o.coin_id
        ,o.first_transaction
        ,o.buyer_number
        from buy_ordering o
        join reference.wallet_ids xw on xw.wallet_address = o.wallet_address
        join temp.wallet_modeling_cohort wc on wc.wallet_id = xw.wallet_id
        """

    transfers_data_df = dgc().run_sql(transfers_data_sql)
    logger.info("Retrieved transfers data for %s wallet-coin pairs associated with %s wallets "
                "in temp.wallet_modeling_cohort.",
                len(transfers_data_df), len(transfers_data_df['wallet_id'].unique()))

    return transfers_data_df



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
    # Get list of wallets to include
    wallet_addresses = profits_df['wallet_address'].unique()

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

    # Calculate features for each column
    all_features = []
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
        result = pd.DataFrame(
            index=pd.Index(wallet_addresses, name='wallet_address')
        )

    # Ensure all wallet addresses are included and fill NaNs
    result = pd.DataFrame(
        result,
        index=pd.Index(wallet_addresses, name='wallet_address')
    ).fillna(0)

    return result
