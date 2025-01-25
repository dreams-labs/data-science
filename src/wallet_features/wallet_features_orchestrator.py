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

@u.timing_decorator
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
    # Add indices and validate inputs
    prepare_dataframes(profits_df,market_indicators_data_df,transfers_sequencing_df,
                       period_start_date,period_end_date)

    # Initialize output dataframe
    wallet_features_df = pd.DataFrame(index=wallet_cohort)
    wallet_features_df.index.name = 'wallet_address'
    feature_column_names = {}

    # Trading features (left join, fill 0s)
    # Requires both starting_balance_date and period_end_date imputed rows
    # -----------------------------------------------------------------------

    trading_features_df = wtf.calculate_wallet_trading_features(profits_df,
        period_start_date,period_end_date,
        wallets_config['features']['include_twb_metrics'],
        wallets_config['features']['include_twr_metrics']
    )
    feature_column_names['trading|'] = trading_features_df.columns
    wallet_features_df = wallet_features_df.join(trading_features_df, how='left')\
        .fillna({col: 0 for col in trading_features_df.columns})

    profits_df.reset_index(inplace=True)
    market_indicators_data_df.reset_index(inplace=True)

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
    transfers_sequencing_features_df = wts.calculate_transfers_sequencing_features(profits_df, transfers_sequencing_df)
    transfers_hypothetical_features_df = wts.calculate_transfers_hypothetical_features(profits_df,
                                                                           period_start_date,period_end_date)
    logger.info(f'transfers_sequencing_features_df {transfers_sequencing_features_df.shape}')
    logger.info(f'transfers_hypothetical_features_df {transfers_hypothetical_features_df.shape}')
    transfers_features_df = transfers_sequencing_features_df.join(transfers_hypothetical_features_df,how='left')

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


@u.timing_decorator
def prepare_dataframes(profits_df: pd.DataFrame,
                       market_indicators_df: pd.DataFrame,
                       transfers_sequencing_df: pd.DataFrame,
                       period_start_date: str,
                       period_end_date: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Orchestrates dataframe preparation: validates inputs, optimizes indices, and downcasts dtypes.

    Params:
    - profits_df (DataFrame): profits data with coin_id, wallet_address, date
    - market_indicators_df (DataFrame): market indicators with coin_id, date
    - transfers_sequencing_df (DataFrame): transfers sequencing data
    - period_start_date (str): start of analysis period
    - period_end_date (str): end of analysis period

    Returns:
    - tuple of prepared DataFrames: (profits_df, market_indicators_df, transfers_df)
    """
    required_profits_cols = ['coin_id', 'wallet_address', 'date']
    required_market_cols = ['coin_id', 'date']

    # Validate profits_df structure
    if not all(col in profits_df.columns or col in profits_df.index.names
              for col in required_profits_cols):
        raise ValueError(f"profits_df missing required columns: {required_profits_cols}")

    # Set indices if needed
    if not isinstance(profits_df.index, pd.MultiIndex):
        profits_df.set_index(required_profits_cols, inplace=True, verify_integrity=True)

    if not isinstance(market_indicators_df.index, pd.MultiIndex):
        market_indicators_df.set_index(required_market_cols, inplace=True, verify_integrity=True)

    # Sort indices if needed
    if not profits_df.index.is_monotonic_increasing:
        profits_df.sort_index(inplace=True)

    if not market_indicators_df.index.is_monotonic_increasing:
        market_indicators_df.sort_index(inplace=True)

    # Run required validations
    u.assert_period(profits_df, period_start_date, period_end_date)
    validate_inputs(profits_df, market_indicators_df, transfers_sequencing_df)

    # Downcast all dataframes
    profits_df = u.df_downcast(profits_df)
    market_indicators_df = u.df_downcast(market_indicators_df)
    transfers_sequencing_df = u.df_downcast(transfers_sequencing_df)

    return profits_df, market_indicators_df, transfers_sequencing_df
