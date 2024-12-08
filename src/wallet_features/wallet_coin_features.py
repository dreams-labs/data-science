"""
Calculates metrics aggregated at the wallet-coin level
"""

import logging
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
