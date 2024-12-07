"""
Calculates metrics aggregated at the wallet-coin level
"""

import dreams_core.core as dc
from dreams_core.googlecloud import GoogleCloud as dgc

# set up logger at the module level
logger = dc.setup_logger()


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
