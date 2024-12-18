"""
Calculates metrics aggregated at the wallet level
"""
import logging
from pathlib import Path
import pandas as pd
import yaml

# Local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig
import wallet_features.performance_features as wpf
import wallet_features.market_cap_features as wmc
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

    # Store feature sets with their prefixes for bulk renaming
    feature_column_names = {}

    # Trading features (inner join, custom fill)
    profits_df = wtf.add_cash_flow_transfers_logic(profits_df)
    trading_features_df = wtf.calculate_wallet_trading_features(profits_df)
    feature_column_names['trading_'] = trading_features_df.columns
    wallet_features_df = wallet_features_df.join(trading_features_df, how='left')\
        .fillna({col: 0 for col in trading_features_df.columns})

    # Market timing features (fill zeros)
    timing_features_df = wmt.calculate_market_timing_features(profits_df, market_indicators_data_df)
    feature_column_names['timing_'] = timing_features_df.columns
    wallet_features_df = wallet_features_df.join(timing_features_df, how='left')\
        .fillna({col: 0 for col in timing_features_df.columns})

    # Market cap features (fill zeros)
    market_features_df = wmc.calculate_market_cap_features(profits_df, market_indicators_data_df)
    feature_column_names['mktcap_'] = market_features_df.columns
    wallet_features_df = wallet_features_df.join(market_features_df, how='left')\
        .fillna({col: 0 for col in market_features_df.columns})

    # Transfers features (fill -1)
    transfers_features_df = wts.calculate_transfers_sequencing_features(profits_df, transfers_sequencing_df)
    feature_column_names['transfers_'] = transfers_features_df.columns
    wallet_features_df = wallet_features_df.join(transfers_features_df, how='left')\
        .fillna({col: -1 for col in transfers_features_df.columns})

    # Performance features (inner join, no fill)
    performance_features_df = (wpf.calculate_performance_features(wallet_features_df)
                                  .drop(['max_investment', 'total_net_flows'], axis=1))  # already exist as trading features
    feature_column_names['performance_'] = performance_features_df.columns
    wallet_features_df = wallet_features_df.join(performance_features_df, how='inner')

    # Bulk rename all columns with their respective prefixes to make data lineage clear
    rename_map = {}
    for prefix, cols in feature_column_names.items():
        rename_map.update({col: f"{prefix}{col}" for col in cols})

    wallet_features_df = wallet_features_df.rename(columns=rename_map)

    return wallet_features_df
