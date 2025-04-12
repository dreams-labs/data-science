import logging
import pandas as pd
from dreams_core.googlecloud import GoogleCloud as dgc

# Local module imports
import utils as u

# set up logger at the module level
logger = logging.getLogger(__name__)



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


# # --------------------------------------
# #       Features Helper Functions
# # --------------------------------------

# def calculate_first_transfers_features(profits_df: pd.DataFrame,
#                                    transfers_sequencing_df: pd.DataFrame,
#                                    include_features: List) -> pd.DataFrame:
#     """
#     Params:
#     - profits_df (DataFrame): profits data defining period boundaries
#     - transfers_sequencing_df (DataFrame): lifetime transfers data for each wallet
#     - include_features (List): list of features to calculate

#     Returns:
#     - first_transfers_features_df (DataFrame): wallet transfer features indexed by wallet_address
#     """
#     # Define transfer types and their corresponding parameters
#     feature_params = {
#         'first_buy': {
#             'date_col': 'first_buy',
#             'number_col': 'buyer_number'
#         },
#         'first_sell': {
#             'date_col': 'first_sell',
#             'number_col': 'seller_number'
#         },
#         'initial_hold_time': {
#             'date_col': 'first_sell',
#             'number_col': 'initial_hold_time'
#         }
#     }

#     # Validate features
#     invalid_features = set(include_features) - set(feature_params.keys())
#     if invalid_features:
#         raise ValueError(f"Features not found in feature_params: {invalid_features}")

#     feature_dfs = []

#     for transfer_type in include_features:
#         params = feature_params[transfer_type]
#         transfers_df = pd.merge(
#             profits_df,
#             transfers_sequencing_df,
#             left_index=True,
#             right_on=['coin_id', 'wallet_address', params['date_col']],
#             how='inner'
#         )

#         # Aggregate features
#         features_df = transfers_df.groupby('wallet_address').agg({
#             params['number_col']: ['count', 'mean', 'median', 'min']
#         })

#         # Rename columns and add prefix
#         features_df.columns = [
#             'new_coin_transaction_counts',
#             'avg_wallet_rank',
#             'median_avg_wallet_rank',
#             'min_avg_wallet_rank'
#         ]
#         features_df = features_df.add_prefix(f'{transfer_type}/')
#         feature_dfs.append(features_df)

#     first_transfers_features_df = pd.concat(feature_dfs, axis=1)

#     return first_transfers_features_df


# def calculate_initial_hold_time(
#     transfers_sequencing_df: pd.DataFrame,
#     profits_df: pd.DataFrame
# ) -> pd.DataFrame:
#     """
#     Calculates the initial hold time (in days) for each record in the transfers_sequencing_df,
#     but only includes records where first_sell falls between the min_date and max_date
#     defined by the profits_df.

#     Params:
#     - transfers_sequencing_df (DataFrame): Contains columns such as wallet_address, coin_id,
#       first_buy, first_sell, buyer_number, seller_number.
#     - profits_df (DataFrame): Defines the valid date range with its index, which must contain
#       min and max boundaries.

#     Returns:
#     - hold_time_df (DataFrame): Original DataFrame with an additional initial_hold_time column
#       for valid records.
#     """
#     # Extract min_date and max_date from profits_df
#     min_date = profits_df.index.min()
#     max_date = profits_df.index.max()

#     # Filter records where first_sell is within the valid date range
#     filtered_transfers_df = transfers_sequencing_df[
#         (transfers_sequencing_df['first_sell'] >= min_date) &
#         (transfers_sequencing_df['first_sell'] <= max_date)
#     ].copy()

#     # Calculate initial_hold_time in days
#     filtered_transfers_df['initial_hold_time'] = (
#         filtered_transfers_df['first_sell'] - filtered_transfers_df['first_buy']
#     ).dt.days

#     return filtered_transfers_df
