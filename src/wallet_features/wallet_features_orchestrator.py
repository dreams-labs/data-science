"""
Calculates metrics aggregated at the wallet level
"""
import logging
from typing import List
import pandas as pd

# Local module imports
import wallet_features.performance_features as wpf
import wallet_features.market_cap_features as wmc
import wallet_features.trading_features as wtf
import wallet_features.transfers_features as wts
import wallet_features.market_timing_features as wmt
import wallet_features.scenario_features as wsc
import wallet_features.balance_features as wbf
import wallet_features.time_series_features as wfts
import coin_features.coin_trends as cfct
import wallet_modeling.wallet_training_data_orchestrator as wtdo
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)



# --------------------------
#      Class Definition
# --------------------------

class WalletFeaturesOrchestrator:
    """
    Orchestrates the calculation of wallet-level features across different feature modules.

    This class encapsulates the feature calculation pipeline that was previously implemented
    as standalone functions, providing better state management and reusability.
    """
    def __init__(
            self,
            wallets_config,
            wallets_metrics_config,
            wallets_features_config,
            complete_hybrid_cw_id_df
        ):
        """
        Initialize the WalletFeaturesOrchestrator.

        Loads configuration objects that are shared across feature calculations.

        Params:
        - 3 configs (dicts)
        - complete_hybrid_cw_id_df (DataFrame): Mapping between coin_id and hybrid_cw_id
        """
        # Load configs at instance level for reuse
        self.wallets_config = wallets_config
        self.wallets_metrics_config = wallets_metrics_config
        self.wallets_features_config = wallets_features_config

        # Store hybrid mapping for coin trends features if configured
        self.complete_hybrid_cw_id_df = complete_hybrid_cw_id_df


    # --------------------------------------
    #      Primary Orchestration Method
    # --------------------------------------

    @u.timing_decorator
    def calculate_wallet_features(
            self,
            profits_df: pd.DataFrame,
            market_indicators_data_df: pd.DataFrame,
            macro_indicators_df: pd.DataFrame,
            transfers_sequencing_df: pd.DataFrame,
            wallet_cohort: List[int],
            period_start_date: str,
            period_end_date: str
        ) -> pd.DataFrame:
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
        - macro_indicators_df (df): Macroeconomic data with technical indicators
        - transfers_sequencing_df (df): Lifetime transfers data
        - wallet_cohort (array-like): All wallet addresses to include
        - period_start_date (str): Period start in 'YYYY-MM-DD' format
        - period_end_date (str): Period end in 'YYYY-MM-DD' format

        Returns:
        - wallet_features_df (df): Wallet-indexed features dataframe with a row for every wallet_cohort
        """
        # Add indices and validate inputs
        profits_df, market_indicators_data_df, transfers_sequencing_df = prepare_dataframes(
            profits_df,market_indicators_data_df,transfers_sequencing_df,
            period_start_date,period_end_date)

        # Initialize output dataframe
        wallet_features_df = pd.DataFrame(index=wallet_cohort)
        wallet_features_df.index.name = 'wallet_address'
        feature_column_names = {}

        # Trading features (left join, fill 0s)
        trading_features_df = wtf.calculate_wallet_trading_features(profits_df,
            period_start_date,period_end_date,
            self.wallets_config['features']['include_twb_metrics'],
            self.wallets_config['features']['include_twr_metrics']
        )
        feature_column_names['trading|'] = trading_features_df.columns
        wallet_features_df = wallet_features_df.join(trading_features_df, how='left')\
            .fillna({col: 0 for col in trading_features_df.columns})

        # Performance features (left join, do not fill)
        performance_features_df = wpf.calculate_performance_features(
            trading_features_df,
            self.wallets_config
        )
        feature_column_names['performance|'] = performance_features_df.columns
        wallet_features_df = wallet_features_df.join(performance_features_df, how='left')

        # Transfers features (left join, do not fill)
        if self.wallets_config['features']['toggle_transfers_features']:
            transfers_sequencing_features_df = wts.calculate_transfers_features(
                profits_df,
                transfers_sequencing_df
            )
            feature_column_names['transfers|'] = transfers_sequencing_features_df.columns
            wallet_features_df = wallet_features_df.join(transfers_sequencing_features_df, how='left')

        # Balance features (left join, do not fill)
        if self.wallets_config['features']['toggle_balance_features']:
            balance_features_df = wbf.calculate_balance_features(
                self.wallets_config,
                profits_df
            )
            feature_column_names['balance|'] = balance_features_df.columns
            wallet_features_df = wallet_features_df.join(balance_features_df, how='left')

        # Macroeconomic features (cross join)
        macroeconomic_features_df = wfts.calculate_macro_features(
            macro_indicators_df,
            self.wallets_metrics_config['time_series']['macro_trends']
        )
        feature_column_names['macro|'] = macroeconomic_features_df.columns
        wallet_features_df = (wallet_features_df.reset_index()
                            .merge(macroeconomic_features_df, how='cross')
                            .set_index('wallet_address'))

        # BELOW FUNCTIONS DO NOT WORK WITH INDICES AND SHOULD BE EVENTUALLY REFACTORED
        profits_df.reset_index(inplace=True)
        market_indicators_data_df.reset_index(inplace=True)

        # Market timing features (left join, fill 0s)
        timing_features_df = wmt.calculate_market_timing_features(
            profits_df,
            market_indicators_data_df,
            macro_indicators_df
        )
        feature_column_names['timing|'] = timing_features_df.columns
        wallet_features_df = wallet_features_df.join(timing_features_df, how='left')\
            .fillna({col: 0 for col in timing_features_df.columns})

        # Market cap features (left join, do not full)
        market_features_df = wmc.calculate_market_cap_features(
            self.wallets_config,
            profits_df,
            market_indicators_data_df
        )
        feature_column_names['mktcap|'] = market_features_df.columns
        wallet_features_df = wallet_features_df.join(market_features_df, how='left')

        # Scenario transfers features (left join, do not fill)
        if self.wallets_config['features']['toggle_scenario_features']:

            scenario_features_df = wsc.calculate_scenario_features(
                profits_df.copy(),
                market_indicators_data_df.copy(),
                trading_features_df.copy(),
                performance_features_df.copy(),
                period_start_date,
                period_end_date,
                self.wallets_config,
            )
            feature_column_names['scenario|'] = scenario_features_df.columns
            wallet_features_df = wallet_features_df.join(scenario_features_df, how='left')

        # Coin trends features (dehybridized coin_id join)
        if (
            self.wallets_config['training_data']['hybridize_wallet_ids']
            and self.wallets_config['features'].get('toggle_coin_trends_features',False)
        ):
            coin_trends_features_df = cfct.generate_coin_trends_features(
                self.wallets_config,
                self.wallets_config['training_data']['training_period_end'],
                self.wallets_metrics_config['time_series']['coin_trends']
            )
            feature_column_names['coin_trends|'] = coin_trends_features_df.columns
            wallet_features_df = self._join_coin_trends_features(
                wallet_features_df,
                coin_trends_features_df
            )

        # Apply feature prefixes
        rename_map = {col: f"{prefix}{col}"
                    for prefix, cols in feature_column_names.items()
                    for col in cols}
        wallet_features_df = wallet_features_df.rename(columns=rename_map)

        return wallet_features_df



# ----------------------------------
#    Utility Methods + Functions
# ----------------------------------

    def _join_coin_trends_features(
            self,
            wallet_features_df: pd.DataFrame,
            coin_trends_features_df: pd.DataFrame
        ) -> pd.DataFrame:
        """
        Join coin_id-indexed coin trends features onto hybridized wallet_address-indexed wallet features.

        This function dehybridizes the wallet addresses to extract coin_ids, joins the coin trends
        features on coin_id, then restores the original hybrid wallet_address indexing.

        Params:
        - wallet_features_df (DataFrame): Features indexed on hybridized wallet_address
        - coin_trends_features_df (DataFrame): Features indexed on coin_id

        Returns:
        - wallet_features_df (DataFrame): Original features with coin trends features added,
            maintaining hybrid wallet_address index

        Raises:
        - ValueError: If any hybrid wallet addresses lack corresponding coin trends data
        """
        # Store original hybrid index for restoration
        original_hybrid_index = wallet_features_df.index.copy()

        # Dehybridize to extract coin_id and wallet_address components
        dehybridized_df = wtdo.dehybridize_wallet_address(
            wallet_features_df.reset_index(),
            self.complete_hybrid_cw_id_df
        )

        # Extract unique coin_ids from dehybridized data
        wallet_coin_ids = set(dehybridized_df['coin_id'].unique())
        trends_coin_ids = set(coin_trends_features_df.index.unique())

        # Check for missing coin trends data
        missing_coin_ids = wallet_coin_ids - trends_coin_ids
        if missing_coin_ids:
            raise ValueError(
                f"Found {len(missing_coin_ids)} coin_ids in hybrid wallet addresses "
                f"without corresponding coin trends data. Missing coin_ids: {sorted(missing_coin_ids)[:10]}"
            )

        # Join coin trends features on coin_id
        merged_df = dehybridized_df.merge(
            coin_trends_features_df,
            left_on='coin_id',
            right_index=True,
            how='left'
        )

        # Verify all rows got coin trends data
        coin_trends_cols = coin_trends_features_df.columns
        missing_trends = merged_df[coin_trends_cols].isna().any(axis=1)
        if missing_trends.any():
            # Grab the exact hybrid addresses and coin_ids that didn’t match
            failed_pairs = merged_df.loc[missing_trends, ['wallet_address', 'coin_id']]
            failed_list = list(failed_pairs.itertuples(index=False, name=None))
            logger.warning(
                f"Coin trends join failed for {len(failed_list)}/{len(merged_df)} hybrid rows. "
                f"Missing pairs: {failed_list}"
            )

        # Restore original hybrid wallet_address index
        merged_df = merged_df.set_index('wallet_address').drop(columns=['coin_id'])
        merged_df.index = original_hybrid_index

        return merged_df



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
    #  No nulls
    if profits_df.isnull().any().any():
        raise ValueError("profits_df contains NaN values.")

    if market_data_df[['price', 'volume', 'market_cap_filled']].isnull().any().any():
        raise ValueError("market_data_df contains NaN values in critical columns.")

    # Unique indices
    if not profits_df.index.is_unique:
        raise ValueError("profits_df index has duplicate (coin_id, wallet_address, date) entries.")
    if not market_data_df.index.is_unique:
        raise ValueError("market_data_df index has duplicate (coin_id, date) entries.")
    if not transfers_sequencing_df.index.is_unique:
        raise ValueError("transfers_sequencing_df index has duplicate entries.")

    # Dates overlap
    profits_dates = profits_df.index.droplevel('wallet_address').unique()
    market_dates = market_data_df.index.unique()
    missing_pairs = profits_dates.difference(market_dates)  # Faster than NumPy set operations

    if missing_pairs.size > 0:
        raise AssertionError(f"Found {missing_pairs.size} coin_id-date pairs missing in market_data_df")

    # If transfers features are toggled on, confirm wallets in transfers_df exist in profits_df
    if not transfers_sequencing_df.empty:
        wallets_in_profits = profits_df.index.get_level_values('wallet_address').unique()
        wallets_in_transfers = transfers_sequencing_df['wallet_address'].unique()
        common_wallets = wallets_in_profits.intersection(wallets_in_transfers)
        coverage = len(common_wallets) / len(wallets_in_profits)
        if coverage < 0.99:
            raise ValueError(f"Only {coverage:.2%} of wallets in profits_df are in transfers_sequencing_df.")

    # All done
    logger.debug("All input dataframes passed validation checks.")



@u.timing_decorator
def prepare_dataframes(
        profits_df: pd.DataFrame,
        market_indicators_df: pd.DataFrame,
        transfers_sequencing_df: pd.DataFrame,
        period_start_date: str,
        period_end_date: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    # Ensure index
    profits_df = u.ensure_index(profits_df)
    market_indicators_df = u.ensure_index(market_indicators_df)

    # Run required validations
    u.assert_period(profits_df, period_start_date, period_end_date)
    validate_inputs(profits_df, market_indicators_df, transfers_sequencing_df)

    # Downcast all dataframes
    profits_df = u.df_downcast(profits_df)
    market_indicators_df = u.df_downcast(market_indicators_df)
    transfers_sequencing_df = u.df_downcast(transfers_sequencing_df)

    return profits_df, market_indicators_df, transfers_sequencing_df
