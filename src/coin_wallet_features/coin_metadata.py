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
def retrieve_metadata_df() -> pd.DataFrame:
    """
    Returns coin metadata from the blockchain. We cannot use Coingecko fields because
     a successful coin is more likely to have Coingecko support, categories, etc which
     will cause data leakage.

    Params:
    - min_txn_size (int): Minimum USD value to filter out dust/airdrops
    - training_end (str): Training period end as YYYY-MM-DD string
    - epoch_reference_date (str): Suffix added to table for each epoch
    - hybridize_wallet_ids (bool): Whether to use hybrid wallet-coin IDs vs regular wallet IDs
    - dataset (str): Set to 'prod' or 'dev' to alter query schema

    Returns:
    - sequence_df (DataFrame): Columns: wallet_address, coin_id, first_buy, first_sell,
        buyer_number, seller_number
    """
    logger.info("Querying for coin metadata...")

    sql = f"""
        select c.coin_id,
        c.chain as blockchain,
        c.total_supply,
        c.decimals,
        case when upper(description) like '%MEME%' then 1 else 0 end as described_meme
        -- note: we cannot use the cm.categories field due to survivorship bias, as
        --  a coin that reached larger size is more likely to have a category

        from core.coins c
        join `core.coin_facts_metadata` cm using (coin_id)
        where c.has_wallet_transfer_data = True
        and c.has_market_data = True
    """
    metadata_df = dgc().run_sql(sql)

    # Log retrieval stats
    logger.info("Retrieved metadata data for %s coins.", len(metadata_df))

    # Set index
    metadata_df = metadata_df.set_index('coin_id')

    return metadata_df
