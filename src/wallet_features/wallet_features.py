"""
Calculates metrics aggregated at the wallet level
"""
import time
import logging
from pathlib import Path
import pandas as pd
import yaml

# Local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig
import wallet_modeling.wallet_modeling as wm
import wallet_features.wallet_coin_date_features as wcdf
import wallet_features.trading_features as wtf
import wallet_features.transfers_features as wts
import wallet_features.market_timing_features as wmt
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()
wallets_metrics_config = u.load_config('../config/wallets_metrics_config.yaml')
wallets_features_config = yaml.safe_load(Path('../config/wallets_features_config.yaml').read_text(encoding='utf-8'))


def calculate_wallet_features(profits_df, market_indicators_data_df, transfers_sequencing_df, wallet_cohort):
    """
    Calculates all features for the wallets in a given profits_df

    Params:
    - profits_df (df): for the window over which the metrics should be computed
    - market_indicators_data_df (df): the full market data df with indicators added
    - transfers_sequencing_df (df): each wallet's lifetime transfers data
    - wallet_cohort (array-like): Array of all wallet addresses that should be present

    Returns:
    - wallet_features_df (df): df indexed on wallet_address with all features
    """
    # Create a DataFrame with all wallets that should exist
    wallet_features_df = pd.DataFrame(index=wallet_cohort)
    wallet_features_df.index.name = 'wallet_address'

    # Trading features (inner join, custom fill)
    profits_df = wtf.add_cash_flow_transfers_logic(profits_df)
    trading_features = wtf.calculate_wallet_trading_features(profits_df)
    trading_features = wtf.fill_trading_features_data(trading_features, wallet_cohort)
    wallet_features_df = wallet_features_df.join(trading_features, how='inner')

    # Market timing features (fill zeros)
    timing_features = wmt.calculate_market_timing_features(profits_df, market_indicators_data_df)
    wallet_features_df = wallet_features_df.join(timing_features, how='left')\
        .fillna({col: 0 for col in timing_features.columns})

    # Market cap features (fill zeros)
    market_features = wcdf.calculate_market_cap_features(profits_df, market_indicators_data_df)
    wallet_features_df = wallet_features_df.join(market_features, how='left')\
        .fillna({col: 0 for col in market_features.columns})

    # Transfers features (fill -1)
    transfers_features = wts.calculate_transfers_sequencing_features(profits_df, transfers_sequencing_df)
    wallet_features_df = wallet_features_df.join(transfers_features, how='left')\
        .fillna({col: -1 for col in transfers_features.columns})

    # Performance features (inner join, no fill)
    performance_features = wm.generate_target_variables(wallet_features_df)
    wallet_features_df = wallet_features_df.join(
        performance_features.drop(['invested', 'net_gain'], axis=1),
        how='inner'
    )

    return wallet_features_df
