import logging
from typing import List
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
#       Data Retrieval Function
# -----------------------------------

def retrieve_transfers_sequencing(hybridize_wallet_ids: bool = False) -> pd.DataFrame:
    """
    Returns buyer and seller sequence numbers for each wallet-coin pair, where the first
    buyer/seller receives rank 1. Only includes wallets from wallet_modeling_training_cohort.

    Params:
    - hybridize_wallet_ids (bool): Whether to use hybrid wallet-coin IDs vs regular wallet IDs

    Returns:
    - sequence_df (DataFrame): Columns: wallet_address, coin_id, first_buy, first_sell,
        buyer_number, seller_number
    """
    # Minimum USD value to filter out dust/airdrops
    min_txn_size = wallets_config['features']['timing_metrics_min_transaction_size']
    training_end = wallets_config['training_data']['training_period_end']

    # Set join table based on ID type
    if hybridize_wallet_ids is False:
        # non-hybridized wallet_ids need to be converted to wallet_address
        id_column = 'wallet_id'
        join_sequence = """
            join (
                select wc.wallet_id,
                xw.wallet_address,
                from temp.wallet_modeling_training_cohort wc
                join reference.wallet_ids xw on xw.wallet_id = wc.wallet_id
            ) wc using(wallet_address)
            """
    else:
        # hybridized ids already have wallet_address included in their table
        id_column = 'hybrid_id'
        join_sequence = """
            join temp.wallet_modeling_training_cohort wc using(wallet_address, coin_id)
            """

    sequencing_sql = f"""
    with transaction_rank as (
        select coin_id
        ,wallet_address
        ,min(case when usd_net_transfers >= {min_txn_size} then date end) as first_buy
        ,min(case when usd_net_transfers <= -{min_txn_size} then date end) as first_sell
        from core.coin_wallet_profits cwp
        where abs(cwp.usd_net_transfers) >= {min_txn_size}
        and cwp.date <= '{training_end}'
        group by 1,2
    ),


    buyer_ranks as (
        select coin_id
        ,wallet_address
        ,first_buy
        ,RANK() OVER (PARTITION BY coin_id ORDER BY first_buy ASC) as buyer_number
        from transaction_rank
        where first_buy is not null
    ),
    seller_ranks as (
        select coin_id
        ,wallet_address
        ,first_sell
        ,RANK() OVER (PARTITION BY coin_id ORDER BY first_sell ASC) as seller_number
        from transaction_rank
        where first_sell is not null
    ),
    sequence_ordering as (
        select tr.coin_id
        ,tr.wallet_address
        ,tr.first_buy
        ,tr.first_sell
        ,b.buyer_number
        ,s.seller_number
        from transaction_rank tr
        left join buyer_ranks b using (coin_id, wallet_address)
        left join seller_ranks s using (coin_id, wallet_address)
    ),

    base_ordering as (
        select
            so.*,
            wc.{id_column} as final_wallet_id
        from sequence_ordering so
        {join_sequence}
    )

    select
        final_wallet_id as wallet_address,
        coin_id,
        first_buy,
        first_sell,
        buyer_number,
        seller_number
    from base_ordering
    order by 1,2,3,4
    """
    print(sequencing_sql)
    sequence_df = dgc().run_sql(sequencing_sql)

    # Log retrieval stats
    logger.info("Retrieved sequence data for %s wallet-coin pairs across %s wallets",
                len(sequence_df), len(sequence_df['wallet_address'].unique()))

    # Optimize memory usage
    sequence_df['coin_id'] = sequence_df['coin_id'].astype('category')
    sequence_df = u.df_downcast(sequence_df)

    return sequence_df



# --------------------------------------
#        Features Main Interface
# --------------------------------------

@u.timing_decorator
def calculate_transfers_features(profits_df, transfers_sequencing_df):
    """
    Retrieves facts about the wallet's transfer activity based on blockchain data.
    Period boundaries are defined by the dates in profits_df through the inner join.

    Params:
    - profits_df (df): the profits_df for the period that the features will reflect
        transfers_sequencing_df (df): each wallet's lifetime transfers data

    Returns:
    - transfers_sequencing_features_df (df): dataframe indexed on wallet_address with
        transfers feature columns
    """
    # Get list of requested features
    include_features = wallets_config['features']['include_transfers_features']
    if len(include_features) == 0:
        return pd.DataFrame()

    # Assign index
    profits_df = u.ensure_index(profits_df)

    # Calculate initial_hold_time in days
    transfers_sequencing_df['initial_hold_time'] = (
        transfers_sequencing_df['first_sell'] - transfers_sequencing_df['first_buy']
    ).dt.days

    # Features related to the wallet-coin pairs' first buy and sell transfers
    first_transfers_features_df = calculate_first_transfers_features(profits_df,
                                                                     transfers_sequencing_df,
                                                                     include_features)

    # Merge all together
    combined_features_df = first_transfers_features_df

    return combined_features_df


# --------------------------------------
#       Features Helper Functions
# --------------------------------------

def calculate_first_transfers_features(profits_df: pd.DataFrame,
                                   transfers_sequencing_df: pd.DataFrame,
                                   include_features: List) -> pd.DataFrame:
    """
    Params:
    - profits_df (DataFrame): profits data defining period boundaries
    - transfers_sequencing_df (DataFrame): lifetime transfers data for each wallet
    - include_features (List): list of features to calculate

    Returns:
    - first_transfers_features_df (DataFrame): wallet transfer features indexed by wallet_address
    """
    # Define transfer types and their corresponding parameters
    feature_params = {
        'first_buy': {
            'date_col': 'first_buy',
            'number_col': 'buyer_number'
        },
        'first_sell': {
            'date_col': 'first_sell',
            'number_col': 'seller_number'
        },
        'initial_hold_time': {
            'date_col': 'first_sell',
            'number_col': 'initial_hold_time'
        }
    }

    # Validate features
    invalid_features = set(include_features) - set(feature_params.keys())
    if invalid_features:
        raise ValueError(f"Features not found in feature_params: {invalid_features}")

    feature_dfs = []

    for transfer_type in include_features:
        params = feature_params[transfer_type]
        transfers_df = pd.merge(
            profits_df,
            transfers_sequencing_df,
            left_index=True,
            right_on=['coin_id', 'wallet_address', params['date_col']],
            how='inner'
        )

        # Aggregate features
        features_df = transfers_df.groupby('wallet_address').agg({
            params['number_col']: ['count', 'mean', 'median', 'min']
        })

        # Rename columns and add prefix
        features_df.columns = [
            'new_coin_transaction_counts',
            'avg_wallet_rank',
            'median_avg_wallet_rank',
            'min_avg_wallet_rank'
        ]
        features_df = features_df.add_prefix(f'{transfer_type}/')
        feature_dfs.append(features_df)

    first_transfers_features_df = pd.concat(feature_dfs, axis=1)

    return first_transfers_features_df


def calculate_initial_hold_time(
    transfers_sequencing_df: pd.DataFrame,
    profits_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculates the initial hold time (in days) for each record in the transfers_sequencing_df,
    but only includes records where first_sell falls between the min_date and max_date
    defined by the profits_df.

    Params:
    - transfers_sequencing_df (DataFrame): Contains columns such as wallet_address, coin_id,
      first_buy, first_sell, buyer_number, seller_number.
    - profits_df (DataFrame): Defines the valid date range with its index, which must contain
      min and max boundaries.

    Returns:
    - hold_time_df (DataFrame): Original DataFrame with an additional initial_hold_time column
      for valid records.
    """
    # Extract min_date and max_date from profits_df
    min_date = profits_df.index.min()
    max_date = profits_df.index.max()

    # Filter records where first_sell is within the valid date range
    filtered_transfers_df = transfers_sequencing_df[
        (transfers_sequencing_df['first_sell'] >= min_date) &
        (transfers_sequencing_df['first_sell'] <= max_date)
    ].copy()

    # Calculate initial_hold_time in days
    filtered_transfers_df['initial_hold_time'] = (
        filtered_transfers_df['first_sell'] - filtered_transfers_df['first_buy']
    ).dt.days

    return filtered_transfers_df







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
