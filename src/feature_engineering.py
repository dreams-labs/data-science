"""
functions used to build coin-level features from training data
"""
# pylint: disable=W1203 # fstrings in logs
# pylint: disable=C0301 # line over 100 chars

import time
import pandas as pd
import numpy as np
from dreams_core.googlecloud import GoogleCloud as dgc
from dreams_core import core as dc

# set up logger at the module level
logger = dc.setup_logger()


def calculate_shark_wallet_concentration(profits_df, shark_wallet_addresses):
    """
    Calculate the shark wallet concentration for each coin and date in the DataFrame.
    
    :param df: DataFrame containing wallet data (profits_df)
    :param shark_wallets: Pandas Series of wallet addresses considered as shark wallets
    :return: DataFrame with coin_id, shark_concentration
    """
    # Create a boolean mask for shark wallets
    profits_df = profits_df.reset_index().copy()
    profits_df['is_shark'] = profits_df['wallet_address'].isin(shark_wallet_addresses)

    # Group by coin_id and date, then calculate concentrations
    grouped = profits_df.groupby(['coin_id','is_shark'])['balance'].sum().unstack().fillna(0)
    grouped.columns = ['non_shark_balance', 'shark_balance']

    # Calculate total balance and shark concentration
    grouped['total_balance'] = grouped['non_shark_balance'] + grouped['shark_balance']
    grouped['shark_concentration'] = grouped['shark_balance'] / grouped['total_balance']

    concentration_df = grouped['shark_concentration'].reset_index()

    return concentration_df
