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



# ------------------------------------------
#      Primary Orchestration Functions
# ------------------------------------------

def calculate_wallet_features(profits_df, market_indicators_data_df, transfers_sequencing_df,
                              wallet_cohort, period_start_date, period_end_date):
    """
    Calculates all features for the wallet_cohort in a given profits_df, returning a df with a
    row for every wallet in the cohort.

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
    - period_start_date (str): Period start in 'YYYY-MM-DD' format
    - period_end_date (str): Period end in 'YYYY-MM-DD' format

    Returns:
    - wallet_features_df (df): Wallet-indexed features dataframe with a row for every wallet_cohort
    """
    logger.info('starting data validation...')
    # Validate inputs
    u.assert_period(profits_df, period_start_date, period_end_date)
    logger.info('data validation1')
    validate_inputs(profits_df, market_indicators_data_df, transfers_sequencing_df)
    logger.info('data validation2')

    # Downcast to ensure optimal memory usage
    profits_df = u.df_downcast(profits_df)
    market_indicators_data_df = u.df_downcast(market_indicators_data_df)
    transfers_sequencing_df = u.df_downcast(transfers_sequencing_df)
    logger.info('data validation complete.')

    # Initialize output dataframe
    wallet_features_df = pd.DataFrame(index=wallet_cohort)
    wallet_features_df.index.name = 'wallet_address'
    feature_column_names = {}

    # Trading features (left join, fill 0s)
    # Requires both starting_balance_date and period_end_date imputed rows
    # -----------------------------------------------------------------------
    trading_features_df = wtf.calculate_wallet_trading_features(profits_df,
        period_start_date,period_end_date,
        wallets_config['features']['include_twb_metrics']
    )
    feature_column_names['trading|'] = trading_features_df.columns
    wallet_features_df = wallet_features_df.join(trading_features_df, how='left')\
        .fillna({col: 0 for col in trading_features_df.columns})

    # Performance features (left join, do not fill)
    # Requires both starting_balance_date and period_end_date imputed rows (same as trading)
    # -----------------------------------------------------------------------
    performance_features_df = wpf.calculate_performance_features(wallet_features_df,
        wallets_config['features']['include_twb_metrics']
    )
    feature_column_names['performance|'] = performance_features_df.columns
    wallet_features_df = wallet_features_df.join(performance_features_df, how='left')

    # Market timing features (left join, fill 0s)
    # Uses only real transfers (~is_imputed)
    # -----------------------------------------------------------------------
    timing_features_df = wmt.calculate_market_timing_features(profits_df, market_indicators_data_df)
    feature_column_names['timing|'] = timing_features_df.columns
    wallet_features_df = wallet_features_df.join(timing_features_df, how='left')\
        .fillna({col: 0 for col in timing_features_df.columns})

    # Market cap features (left join, do not full)
    # Volume weighted uses real transfers, balance weighted uses period_end_date
    # -----------------------------------------------------------------------
    market_features_df = wmc.calculate_market_cap_features(profits_df, market_indicators_data_df)
    feature_column_names['mktcap|'] = market_features_df.columns
    wallet_features_df = wallet_features_df.join(market_features_df, how='left')

    # Transfers features (left join, do not fill)
    # Uses only real transfers (~is_imputed)
    # -----------------------------------------------------------------------
    transfers_features_df = wts.calculate_transfers_sequencing_features(profits_df, transfers_sequencing_df)
    feature_column_names['transfers|'] = transfers_features_df.columns
    wallet_features_df = wallet_features_df.join(transfers_features_df, how='left')

    # Apply feature prefixes
    rename_map = {col: f"{prefix}{col}"
                for prefix, cols in feature_column_names.items()
                for col in cols}

    return wallet_features_df.rename(columns=rename_map)



# ----------------------------------
#         Utility Functions
# ----------------------------------

def validate_inputs(profits_df, market_data_df, transfers_sequencing_df):
    """
    Validates pre-indexed DataFrames for the feature calculation pipeline.

    Params:
    - profits_df (DataFrame): Indexed by (coin_id, wallet_address, date)
    - market_data_df (DataFrame): Indexed by (coin_id, date)
    - transfers_sequencing_df (DataFrame): Contains wallet_address column

    Raises:
    - ValueError: For data quality issues
    - AssertionError: For missing market data coverage
    """
    # NaN checks use index-aware operations
    if profits_df.isnull().any().any():
        raise ValueError("profits_df contains NaN values.")

    if market_data_df[['price', 'volume', 'market_cap_filled']].isnull().any().any():
        raise ValueError("market_data_df contains NaN values in critical columns.")

    # Check market data coverage using index operations
    profits_dates = profits_df.index.droplevel('wallet_address').unique()
    market_dates = market_data_df.index
    missing_pairs = profits_dates.difference(market_dates)

    if len(missing_pairs) > 0:
        raise AssertionError(f"Found {len(missing_pairs)} coin_id-date pairs missing in market_data_df")

    # Wallet presence check using index
    if not set(profits_df.index.get_level_values('wallet_address')).issubset(transfers_sequencing_df['wallet_address']):
        raise ValueError("profits_df has wallets not in transfers_sequencing_df.")

    logger.debug("All input dataframes passed validation checks.")
