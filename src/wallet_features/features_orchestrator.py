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
    Calculates all features for the wallets in a given profits_df.

    Imputed Row Dependencies:
    - Trading Features: Requires starting_balance_date and period_end_date for performance calculation
    - Performance Features: Inherits from trading features
    - Market Cap Features:
        - Volume weighted: Uses only real transfers (~is_imputed)
        - Balance weighted: Uses period_end_date balances
    - Market Timing Features: Uses only real transfers (~is_imputed)
    - Transfers Features: Uses only real transfers (~is_imputed)

    Function Dependencies:
    1. Trading features must precede Performance features
    2. All other features can be calculated independently

    Params:
    - profits_df (df): Daily profits with imputed rows on:
        1. starting_balance_date (period start reference)
        2. period_end_date (period end reference)
    - market_indicators_data_df (df): Market data with technical indicators
    - transfers_sequencing_df (df): Lifetime transfers data
    - wallet_cohort (array-like): All wallet addresses to include

    Returns:
    - wallet_features_df (df): Wallet-indexed features dataframe
    """
    # Initialize output dataframe
    wallet_features_df = pd.DataFrame(index=wallet_cohort)
    wallet_features_df.index.name = 'wallet_address'
    feature_column_names = {}

    # Trading features (inner join)
    # Requires both starting_balance_date and period_end_date imputed rows
    # -----------------------------------------------------------------------
    profits_df = wtf.add_cash_flow_transfers_logic(profits_df)
    trading_features_df = wtf.calculate_wallet_trading_features(profits_df)
    feature_column_names['trading_'] = trading_features_df.columns
    wallet_features_df = wallet_features_df.join(trading_features_df, how='inner')

    # Performance features (inner join)
    # Inherits trading features imputed rows requirement
    # -----------------------------------------------------------------------
    performance_features_df = (wpf.calculate_performance_features(wallet_features_df)
                            .drop(['max_investment', 'crypto_net_gain'], axis=1))
    feature_column_names['performance_'] = performance_features_df.columns
    wallet_features_df = wallet_features_df.join(performance_features_df, how='inner')

    # Market timing features (left join, fill 0s)
    # Uses only real transfers (~is_imputed)
    # -----------------------------------------------------------------------
    timing_features_df = wmt.calculate_market_timing_features(profits_df, market_indicators_data_df)
    feature_column_names['timing_'] = timing_features_df.columns
    wallet_features_df = wallet_features_df.join(timing_features_df, how='left')\
        .fillna({col: 0 for col in timing_features_df.columns})

    # Market cap features (left join, fill 0s)
    # Volume weighted uses real transfers, balance weighted uses period_end_date
    # -----------------------------------------------------------------------
    market_features_df = wmc.calculate_market_cap_features(profits_df, market_indicators_data_df)
    feature_column_names['mktcap_'] = market_features_df.columns
    wallet_features_df = wallet_features_df.join(market_features_df, how='left')\
        .fillna({col: 0 for col in market_features_df.columns})

    # Transfers features (left join, fill -1s)
    # Uses only real transfers (~is_imputed)
    # -----------------------------------------------------------------------
    transfers_features_df = wts.calculate_transfers_sequencing_features(profits_df, transfers_sequencing_df)
    feature_column_names['transfers_'] = transfers_features_df.columns
    wallet_features_df = wallet_features_df.join(transfers_features_df, how='left')\
        .fillna({col: -1 for col in transfers_features_df.columns})

    # Apply feature prefixes
    rename_map = {col: f"{prefix}{col}"
                for prefix, cols in feature_column_names.items()
                for col in cols}

    return wallet_features_df.rename(columns=rename_map)
