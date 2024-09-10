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


def build_shark_coin_features(shark_coins_df):
    """
    creates a series of coin-keyed metrics based on shark behavior
    """
    # Step 1: Coin-Level Metrics - Counting the number of sharks per coin
    coin_shark_count = shark_coins_df.groupby('coin_id')['is_shark'].sum().reset_index()
    coin_shark_count.columns = ['coin_id', 'num_sharks']

    # Step 2: Total inflows by sharks for each coin
    coin_shark_inflows = shark_coins_df[shark_coins_df['is_shark']].groupby('coin_id')['usd_inflows_cumulative'].sum().reset_index()
    coin_shark_inflows.columns = ['coin_id', 'total_shark_inflows']

    # Step 3: Merge the coin-level shark metrics
    coin_shark_metrics_df = pd.merge(coin_shark_count, coin_shark_inflows, on='coin_id', how='left')

    return coin_shark_metrics_df
