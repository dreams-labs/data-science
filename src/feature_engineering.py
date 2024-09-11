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


def calculate_cohort_concentration(profits_df, cohort_wallet_addresses):
    """
    Calculate the cohort concentration for each coin and date in the DataFrame.
    
    :param df: DataFrame containing wallet data (profits_df)
    :param cohort_wallet_addresses: Pandas Series of wallet addresses representing the cohort
    :return: DataFrame with coin_id, cohort_concentration
    """
    # Create a boolean mask for cohort wallets
    profits_df = profits_df.reset_index().copy()
    profits_df['in_cohort'] = profits_df['wallet_address'].isin(cohort_wallet_addresses)

    # Group by coin_id and date, then calculate concentrations
    grouped = profits_df.groupby(['coin_id','in_cohort'])['balance'].sum().unstack().fillna(0)
    grouped.columns = ['non_cohort_balance', 'cohort_balance']

    # Calculate total balance and cohort concentration
    grouped['total_balance'] = grouped['non_cohort_balance'] + grouped['cohort_balance']
    grouped['cohort_concentration'] = grouped['cohort_balance'] / grouped['total_balance']

    concentration_df = grouped['cohort_concentration'].reset_index()

    return concentration_df


