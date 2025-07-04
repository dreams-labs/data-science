"""
Orchestrates the end-to-end coin feature engineering pipeline for coin-level modeling.

This module defines the CoinFeaturesOrchestrator class, which coordinates the loading,
transformation, and aggregation of data to produce a comprehensive set of coin-level features
for predictive modeling. The orchestrator integrates multiple feature sources—including
wallet-based metrics, macroeconomic and market time series, coin metadata, and Coin Flow
model outputs—into a unified DataFrame indexed by coin_id.

Business logic and high-level architecture:
- The orchestrator manages configuration and cohort information for the modeling process.
- It executes a multi-stage pipeline:
    1. **Wallet-based features:** Aggregates wallet-level trading and balance metrics into
       coin-level features, using wallet segmentation and flattening logic.
    2. **Time series features:** Generates and merges macroeconomic and market data features,
       ensuring alignment and uniqueness of columns.
    3. **Coin metadata:** Retrieves and merges blockchain-derived metadata, avoiding
       data leakage from external sources.
    4. **Coin Flow model features:** Optionally generates and merges features from the
       Coin Flow model, ensuring period alignment and feature consistency.
- The orchestrator provides methods for generating features, calculating target variables,
  and merging feature sets, with robust validation and logging at each step.
- Utility functions are included for parsing and structuring feature names for analysis.

This architecture enables modular, reproducible, and extensible coin feature engineering,
supporting downstream machine learning workflows.
"""


import logging
import pandas as pd

# Local module imports
import feature_engineering.coin_flow_features_orchestrator as cffo
import wallet_modeling.wallets_config_manager as wcm
import coin_features.wallet_segmentation as cws
import coin_features.wallet_metrics as cfwm
import coin_features.wallet_metrics_flattening as cfwmf
import coin_features.coin_time_series as cfts
import coin_features.coin_metadata as cfmd
import coin_insights.coin_validation_analysis as civa
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)


# -------------------------------------------------
#              CoinFeaturesOrchestrator
# -------------------------------------------------

class CoinFeaturesOrchestrator:
    """
    Orchestrates coin-model pipeline: data loading, feature engineering,
    target computation, modeling, and artifact saving.
    """
    def __init__(
        self,
        wallets_config: dict,
        wallets_coin_config: dict,
        wallets_coins_metrics_config: dict,
        coin_flow_config: dict,
        coin_flow_modeling_config: dict,
        coin_flow_metrics_config: dict,
        training_coin_cohort: pd.Series,
    ):
        # Store configs
        self.wallets_config = wallets_config
        self.wallets_coin_config = wallets_coin_config
        self.wallets_coins_metrics_config = wallets_coins_metrics_config
        self.coin_flow_config = coin_flow_config
        self.coin_flow_modeling_config = coin_flow_modeling_config
        self.coin_flow_metrics_config = coin_flow_metrics_config

        # Store cohort
        self.training_coin_cohort = training_coin_cohort

        # Placeholders for intermediate DataFrames
        self.modeling_profits_df = None
        self.modeling_market_data_df = None
        self.como_profits_df = None
        self.como_market_data_df = None

    # -----------------------------------------
    #       Primary Orchestration Methods
    # -----------------------------------------

    @u.timing_decorator
    def generate_coin_features(
        self,
        profits_df: pd.DataFrame,
        wallet_training_data_df: pd.DataFrame,
        market_indicators_df: pd.DataFrame,
        macro_indicators_df: pd.DataFrame,
        period: str,
        prefix: str
    ) -> pd.DataFrame:
        """
        Compute coin-level features for the epoch.

        Features are split into three categories:
        1. Wallet-based features: flatten coin-wallet pair activity to the coin level,
            such as aggregated trading behavior or holder scores.
        2. Time series-based features: macroeconomic indicators and market data such as price,
            volume, and market cap.
        3. Coin Flow features: generated from the Coin Flow model.


        Params:
        - profits_df (DataFrame): profits data for the specified period
        - wallet_training_data_df (DataFrame): wallet training features built with data that
            extends up to the coin_modeling_period_start.
        - market_indicators_df (DataFrame): date-indexed market data indicators
        - macro_indicators_df (DataFrame): date-indexed macroeconomic indicators
        - period (str): period identifier (e.g., 'coin_modeling')
        - prd (str): abbreviated prefix for saved files (e.g., 'como')

        Returns:
        - coin_training_data_df_full (DataFrame): coin_id-indexed feature set
        """
        logger.info("Beginning coin feature generation...")
        u.notify('intro_4')

        # Guard: profits_df covers expected date range
        u.assert_period(
            profits_df,
            self.wallets_config['training_data'][f'{period}_period_start'],
            self.wallets_config['training_data'][f'{period}_period_end']
        )
        # Guard: wallet_training_data_df has unique wallet rows (needed for segmentation)
        if wallet_training_data_df.index.duplicated().any():
            raise ValueError("wallet_training_data_df contains duplicated wallet rows.")


        # Wallet-Based Features
        # ---------------------
        # Generate metrics for coin-wallet pairs in wallet_training_data_df
        cw_metrics_df = cfwm.compute_coin_wallet_metrics(
            self.wallets_coin_config,
            profits_df,
            wallet_training_data_df,
            self.wallets_config['training_data'][f'{period}_period_start'],
            self.wallets_config['training_data'][f'{period}_period_end']
        )

        # Assign wallets in wallet_training_data_df to segments
        wallet_segmentation_df = cws.build_wallet_segmentation(
            self.wallets_coin_config,
            wallet_training_data_df
        )

        # Flatten cw_metrics into single values for each coin-segment pair
        coin_wallet_features_df = cfwmf.flatten_cw_to_coin_segment_features(
            cw_metrics_df,
            wallet_segmentation_df,
            self.training_coin_cohort,
            self.wallets_coin_config['features']['score_distributions'],
            self.wallets_coin_config['n_threads']['cw_flattening_threads']
        )

        # Instantiate full features df
        coin_training_data_df_full = coin_wallet_features_df
        if not coin_training_data_df_full.columns.is_unique:
            raise ValueError("Duplicate columns found in coin_training_data_df_full.")


        # Time Series Features
        # --------------------
        # Macroeconomic features
        if self.wallets_coin_config['features']['toggle_macro_features']:
            macro_features_df = cfts.generate_macro_features(
                macro_indicators_df,
                self.wallets_coins_metrics_config['time_series']['macro_trends']
            )
            macro_features_df = macro_features_df.add_prefix('macro|')
            # cross join
            coin_training_data_df_full = (
                coin_training_data_df_full.reset_index()
                .merge(macro_features_df, how='cross')
                .set_index('coin_id')
            )
            if not coin_training_data_df_full.columns.is_unique:
                raise ValueError("Duplicate columns found in coin_training_data_df_full.")

        # Market data features
        if self.wallets_coin_config['features']['toggle_market_features']:
            market_features_df = cfts.generate_market_features(
                market_indicators_df,
                self.wallets_coins_metrics_config['time_series']['market_data']
            )
            market_features_df = market_features_df.set_index('coin_id').add_prefix('market_data|')
            # join on coin_id
            u.assert_matching_indices(market_features_df,coin_training_data_df_full)
            coin_training_data_df_full = coin_training_data_df_full.join(market_features_df)
            if not coin_training_data_df_full.columns.is_unique:
                raise ValueError("Duplicate columns found in coin_training_data_df_full.")


        # Coin Metadata Features
        # ----------------------
        metadata_features_df = cfmd.retrieve_metadata_df()
        metadata_features_df = metadata_features_df.add_prefix('metadata|')
        coin_training_data_df_full = coin_training_data_df_full.join(metadata_features_df)


        # Coin Flow Model Features
        # ------------------------
        # Generate and merge Coin Flow Model features if configured
        if self.wallets_coin_config['features']['toggle_coin_flow_model_features']:

            # Generate and merge all features
            coin_flows_model_features_df = self._generate_coin_flow_model_features()
            u.to_parquet_safe(
                coin_flows_model_features_df,
                f"{self.wallets_coin_config['training_data']['parquet_folder']}"
                f"/{prefix}_coin_flows_model_features_df.parquet",
                index=True
            )
            coin_training_data_df_full = self._merge_all_features(
                coin_wallet_features_df,
                coin_flows_model_features_df
            )
            if not coin_training_data_df_full.columns.is_unique:
                raise ValueError("Duplicate columns found in coin_training_data_df_full.")

        u.notify('notification_toast')
        logger.info("Successfully generated coin_training_data_df with shape "
                    f"({coin_training_data_df_full.shape}).")

        return coin_training_data_df_full


    def calculate_target_variables(
        self,
        market_data_df: pd.DataFrame,
        period_start: str,
        period_end: str,
        coin_cohort: pd.Series
    ) -> pd.DataFrame:
        """
        Params:
        - market_data_df (DataFrame): Market data for target variable period.
        - period_start (str): Period start date (YYYY-MM-DD).
        - period_end (str): Period end date (YYYY-MM-DD).

        Returns:
        - coin_performance_df (DataFrame): Coin performance metrics with target variables.
        """
        u.assert_period(market_data_df, period_start, period_end)

        # Filter to cohort
        market_data_df = market_data_df[market_data_df.index.get_level_values('coin_id').isin(coin_cohort)]

        # Calculate coin return performance during validation period
        coin_performance_df = civa.calculate_coin_performance(
            market_data_df,
            period_start,
            period_end
        )

        # Drop columns with np.nan coin_return values, which indicate a 0 starting price
        coin_performance_df = coin_performance_df.dropna()

        # Add winsorized return
        coin_performance_df['coin_return_winsorized'] = u.winsorize(
            coin_performance_df['coin_return'],
            self.wallets_coin_config['coin_modeling']['returns_winsorization']
        )

        # Add full percentile (meaning it's a percentile of all coins prior to any population filtering)
        coin_performance_df['coin_return_pctile_full'] = (
            coin_performance_df['coin_return'].rank(pct=True, ascending=True)
        )

        # Validation: check if any coin_ids missing from final features
        missing_coins = coin_cohort - set(coin_performance_df.index)
        if missing_coins:
            raise ValueError(
                f"Found {len(missing_coins)} coin_ids in training_data_df without validation period target variables."
            )

        return coin_performance_df.sort_index()




    # ----------------------------------
    #           Helper Methods
    # ----------------------------------

    def _generate_coin_flow_model_features(
        self,
    ) -> pd.DataFrame:
        """
        Generate the “non-wallet” (time-window) coin features and save them to parquet.

        Params:
        - modeling_profits_df (DataFrame): profits for modeling period
        - time_window_generator (callable): cffo.generate_all_time_windows_model_inputs

        Returns:
        - coin_non_wallet_features_df (DataFrame): index=coin_id, features from all windows
        """
        # Confirm config alignment
        wcm.validate_config_alignment(
            self.coin_flow_config,
            self.wallets_config,
            self.wallets_coin_config)

        # Confirm period boundaries align
        model_start = self.coin_flow_config['training_data']['modeling_period_start']
        val_start = self.wallets_config['training_data']['coin_modeling_period_start']
        model_end = self.coin_flow_config['training_data']['modeling_period_end']
        val_end = self.wallets_config['training_data']['coin_modeling_period_end']
        if not (model_start == val_start and model_end == val_end):
            raise ValueError(
                f"Coin features modeling period must align with wallet features validation period:\n"
                f"Wallet-coin model coin_modeling_period boundaries: {val_start} to {val_end} \n"
                f"Coin Flow Model modeling_period boundaries: {model_start} to {model_end}"
            )

        # Generate features based on the coin config files
        coin_flow_features_orchestrator = cffo.CoinFlowFeaturesOrchestrator(
            self.coin_flow_config,
            self.coin_flow_metrics_config,
            self.coin_flow_modeling_config
        )
        coin_features_df, _, _ = coin_flow_features_orchestrator.generate_all_time_windows_model_inputs()

        # Remove time window index since we aren't using that for now
        coin_features_df = coin_features_df.reset_index(level='time_window', drop=True)

        u.notify('ui_1')

        return coin_features_df



    def _merge_all_features(
        self,
        coin_wallet_features_df: pd.DataFrame,
        coin_flows_model_features_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Inner-join wallet_df & non_wallet_df into final coin-level feature set.

        Params:
        - wallet_df (DataFrame): coin-level features derived from wallet segments.
        - non_wallet_df (DataFrame): coin-level features from time-window engine.

        Returns:
        - merged_df (DataFrame): inner-joined on coin_id, only coins present in both.
        """
        if coin_flows_model_features_df is not None:
            # Confirm overlap
            coin_features_ids = coin_flows_model_features_df.index
            coin_wallet_features_ids = coin_wallet_features_df.index
            wallet_features_only_ids = set(coin_wallet_features_ids) - set(coin_features_ids)

            if len(wallet_features_only_ids) == 0:
                logger.info("All %s coins with wallet features were found in the non wallet coin features set.",
                            len(coin_wallet_features_ids))
            else:
                logger.warning(f"Wallet features contain {len(wallet_features_only_ids)} coins "
                            "not in the non wallet coin features")

            # Join together
            coin_training_data_df_full = coin_wallet_features_df.join(coin_flows_model_features_df,how='inner')

        else:
            # Just return base features if Coin Flow Model features aren't input
            coin_training_data_df_full = coin_wallet_features_df

        logger.info("Final features shape: %s",coin_training_data_df_full.shape)

        return coin_training_data_df_full




# ----------------------------------
#         Utility Functions
# ----------------------------------

def parse_feature_names(
        coin_training_data_df: pd.DataFrame,
        retain_col: str = None
    ) -> pd.DataFrame:
    """Parse feature names from training dataframe into structured components.

    Params:
    - coin_training_data_df (DataFrame): DataFrame containing features to parse
    - retain_col (str): The column in coin_training_data_df with this name is
        appended to the output. e.g. used to retain Importances

    Returns:
    - feature_details_df (DataFrame): DataFrame with parsed feature components
    """
    # Create dataframe of column names
    df = pd.DataFrame(coin_training_data_df)

    # Split on pipe delimiters
    split_df = df['feature'].str.split('|', expand=True)
    split_df = split_df.reindex(columns=range(4))  # Pad with NaN if needed
    split_df.columns = ['segment_category', 'segment_family', 'metric', 'transformation']

    # Split nested components
    segment_families = split_df['segment_family'].str.split('/', expand=True)
    segment_families.columns = ['segment_family', 'segment_value']

    metrics = split_df['metric'].str.split('/', expand=True)
    metrics.columns = ['metric', 'metric_detail']

    transformations = split_df['transformation'].str.split('/', expand=True)
    transformations.columns = ['transformation_category', 'transformation_base', 'transformation_method']

    # Combine all components
    feature_details_df = pd.concat([
        split_df['segment_category'],
        segment_families,
        metrics,
        transformations,
    ], axis=1)
    feature_details_df['feature_full'] = df['feature']

    # Add retain cols if configured
    if retain_col is not None:
        feature_details_df[retain_col] = coin_training_data_df[retain_col]

    return feature_details_df
