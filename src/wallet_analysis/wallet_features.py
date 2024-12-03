'''
calculates metrics related to the distribution of coin ownership across wallets
'''
# pylint: disable=C0301 # line too long

import pandas as pd
import dreams_core.core as dc

# set up logger at the module level
logger = dc.setup_logger()


def calculate_wallet_investment_return(profits_df):
    """
    Calculates the return on investment for the wallet, defined below.

    Profits_df must have initial balances reflected as positive usd_net_transfers
    and ending balances reflected as negative usd_net_transfers for calculations to
    be accurately reflected.

    - Invested: the maximum amount of cumulative net inflows for the wallet. This
        that a wallet that has coin x outflows followed by equal coin y inflows,
        the amount invested will show no change.
    - Return: All net transfers summed together, showing the combined change
        in assets and balance

    Params:
    - profits_df (pd.DataFrame): df showing all usd net transfers for coin-wallet pairs,
        with columns coin_id,wallet_address,date

    Returns:
    - wallet_performance_df (pd.DataFrame): df keyed on wallet_address with columns
        'ingested', 'net_gain' , 'return'
    """

    profits_df = profits_df.set_index('wallet_address')

    # Calculate amount invested
    wallet_invested_df = pd.DataFrame(
        profits_df
        .groupby(level='wallet_address')['usd_net_transfers'].cumsum()
        .groupby(level='wallet_address').max()
    )
    wallet_invested_df.columns = ['invested']

    # Calculate net gains
    wallet_gain_df = pd.DataFrame(
        -profits_df.groupby(level='wallet_address')['usd_net_transfers'].sum()
    )
    wallet_gain_df.columns = ['net_gain']

    # Join dfs
    wallet_performance_df = wallet_invested_df.join(wallet_gain_df)

    # Compute return
    wallet_performance_df['return'] = wallet_performance_df['net_gain']/wallet_performance_df['invested']

    return wallet_performance_df
