"""
Calculates metrics aggregated at the wallet-coin level
"""

import logging
from dreams_core.googlecloud import GoogleCloud as dgc

# Local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig

# set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()



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

    transfers_data_df = dgc().run_sql(transfers_data_sql)
    logger.info("Retrieved transfers data for %s wallet-coin pairs associated with %s wallets "
                "in temp.wallet_modeling_cohort.",
                len(transfers_data_df), len(transfers_data_df['wallet_id'].unique()))

    return transfers_data_df
