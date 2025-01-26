import logging
import pandas as pd
from dreams_core.googlecloud import GoogleCloud as dgc

# Local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig
import utils as u

# set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()



# -----------------------------------
#       Main Interface Function
# -----------------------------------

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
