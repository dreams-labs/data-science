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



def retrieve_transfers_sequencing():
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
    minimum_transaction_size = 0

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

    transfers_sequencing_df = dgc().run_sql(transfers_data_sql)
    logger.info("Retrieved transfers data for %s wallet-coin pairs associated with %s wallets "
                "in temp.wallet_modeling_cohort.",
                len(transfers_sequencing_df), len(transfers_sequencing_df['wallet_id'].unique()))

    return transfers_sequencing_df



def calculate_transfers_sequencing_features(profits_df, transfers_sequencing_df):
    """
    Retrieves facts about the wallet's transfer activity based on blockchain data.

    Params:
        profits_df (df): the profits_df for the period that the features will reflect
        transfers_sequencing_df (df): each wallet's lifetime transfers data

    Returns:
        transfers_sequencing_features_df (df): dataframe indexed on wallet_id with transfers feature columns
    """
    # Inner join lifetime transfers with the profits_df window to filter on date
    window_transfers_data_df = pd.merge(
        profits_df,
        transfers_sequencing_df,
        left_on=['coin_id', 'date', 'wallet_address'],
        right_on=['coin_id', 'first_transaction', 'wallet_id'],
        how='inner'
    )

    # Append buyer numbers to the merged_df
    transfers_sequencing_features_df = window_transfers_data_df.groupby('wallet_id').agg({
        'buyer_number': ['count', 'mean', 'median', 'min']
    })
    transfers_sequencing_features_df.columns = [
        'new_coin_buy_counts',
        'avg_buyer_number',
        'median_buyer_number',
        'min_buyer_number'
    ]

    # Rename to the wallet_id index to "wallet_address" to be consistent with the other functions
    transfers_sequencing_features_df.index.name = 'wallet_address'

    return transfers_sequencing_features_df



def calculate_days_since_last_buy(transfers_df):
    """
    Calculate days since last buy for each wallet-coin combination at each date.

    Parameters:
    transfers_df: pandas DataFrame with columns [date, wallet_address, coin_id, net_transfers]
        date can be string or datetime format
        net_transfers should be positive for buys, negative for sells

    Returns:
    DataFrame with additional column 'days_since_last_buy'
    """
    # Create a copy to avoid modifying original dataframe
    result_df = transfers_df.copy()

    # Convert dates to datetime if they aren't already
    result_df['date'] = pd.to_datetime(result_df['date'])

    # Mark buy transactions and get last buy date for each wallet-coin combination
    result_df['is_buy'] = result_df['net_transfers'] > 0

    # First create the series of dates where is_buy is True, then group and forward fill
    buy_dates = result_df['date'].where(result_df['is_buy'])
    result_df['last_buy_date'] = buy_dates.groupby([result_df['wallet_address'], result_df['coin_id']]).ffill()

    # Calculate days since last buy
    result_df['days_since_last_buy'] = (
        result_df['date'] - result_df['last_buy_date']
    ).dt.days

    # Clean up intermediate columns
    result_df = result_df.drop(['is_buy', 'last_buy_date'], axis=1)

    return result_df



def calculate_average_holding_period(transfers_df):
    """
    Calculate the average holding period for tokens in each wallet at each timestamp.

    The calculation handles:
    - New tokens starting with 0 days held
    - Aging of existing tokens over time
    - Proportional reduction in holding days when tokens are sold

    Parameters:
    df: pandas DataFrame with columns [date, net_transfers]
        date: timestamp of the transfer
        net_transfers: denominated in tokens. positive for buys, negative for sells.
        balance: number of tokens a wallet holds on the date

    Returns:
    DataFrame with additional columns including average_holding_period
    """
    result_df = transfers_df.copy()

    # Ensure date is in datetime format
    result_df['date'] = pd.to_datetime(result_df['date'])

    # Calculate running balance after each transfer
    result_df['balance'] = result_df['net_transfers'].cumsum()

    # Calculate time elapsed since previous transaction
    result_df['previous_date'] = result_df['date'].shift(1)
    result_df['days_passed'] = (result_df['date'] - result_df['previous_date']).dt.days.fillna(0)

    # Track balance at start of each period
    result_df['opening_balance'] = result_df['balance'].shift(1).fillna(0)

    # Calculate holding days added during this period
    # This is: (opening balance) * (days since last transaction)
    result_df['new_hdays'] = result_df['opening_balance'] * result_df['days_passed']

    # Handle sells by reducing holding days proportionally
    # If we sell 50% of tokens, we reduce holding days by 50%
    result_df['sold_tokens'] = result_df['net_transfers'].clip(upper=0)  # Keep only sells (negative values)
    result_df['sold_hdays'] = result_df['sold_tokens'] * result_df['days_passed']  # Reduce holding days proportionally

    # Combine effects of time passing and sales
    result_df['net_change_hdays'] = result_df['new_hdays'] + result_df['sold_hdays']
    result_df['previous_change_hdays'] = result_df['net_change_hdays'].shift(1).fillna(0)

    # Calculate cumulative holding days
    result_df['closing_hdays'] = result_df['net_change_hdays'] + result_df['previous_change_hdays']
    result_df['opening_hdays'] = result_df['closing_hdays'].shift(1).fillna(0)
    result_df['hdays'] = result_df['opening_hdays'] + result_df['net_change_hdays']

    # Calculate average holding period
    # When balance is 0, set average holding period to 0 to avoid division by zero
    result_df['average_holding_period'] = ((result_df['hdays'] / result_df['balance'])
                                           .where(result_df['balance'] != 0, 0))

    # Clean up intermediate columns if desired
    columns_to_keep = ['date', 'net_transfers', 'balance', 'average_holding_period']
    result_df = result_df[columns_to_keep]

    return result_df


