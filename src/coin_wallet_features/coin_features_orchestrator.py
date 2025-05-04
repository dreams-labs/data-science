"""
Orchestrates groups of functions to generate wallet model pipeline
"""
import logging
import importlib
import pandas as pd

# Local module imports
import feature_engineering.time_windows_orchestration as tw
import coin_wallet_features.coin_features_orchestrator as cfo
import coin_wallet_features.wallet_segmentation as cws
import coin_wallet_features.wallet_metrics as cwwm
import coin_wallet_features.wallet_metrics_flattening as cwwmf
import coin_insights.coin_validation_analysis as civa
import utils as u
importlib.reload(cfo)

# Set up logger at the module level
logger = logging.getLogger(__name__)

class CoinTrainingDataOrchestrator:
    """
    Orchestrates coin-model pipeline: data loading, feature engineering,
    target computation, modeling, and artifact saving.
    """
    def __init__(
        self,
        wallets_config: dict,
        wallets_coin_config: dict,
        metrics_config: dict,
        coins_config: dict,
        coins_modeling_config: dict,
        training_coin_cohort: pd.Series
    ):
        # Store configs
        self.wallets_config = wallets_config
        self.wallets_coin_config = wallets_coin_config
        self.metrics_config = metrics_config
        self.coins_config = coins_config
        self.coins_modeling_config = coins_modeling_config

        # Store cohort
        self.training_coin_cohort = training_coin_cohort

        # placeholders for intermediate DataFrames
        self.modeling_profits_df = None
        self.modeling_market_data_df = None
        self.como_profits_df = None
        self.como_market_data_df = None


    # -----------------------------------------
    #       Primary Orchestration Methods
    # -----------------------------------------

    def generate_coin_features_for_period(
        self,
        profits_df: pd.DataFrame,
        period: str,
        prd: str
    ) -> pd.DataFrame:
        """
        Compute coin-wallet metrics for a specific period and build coin features.

        Params:
        - profits_df (DataFrame): profits data for the specified period
        - period (str): period identifier (e.g., 'coin_modeling')
        - prd (str): abbreviated prefix for saved files (e.g., 'como')

        Returns:
        - coin_training_data_df_full (DataFrame): full coin-level feature set
        """
        # Generate metrics for coin-wallet pairs
        cw_metrics_df = cwwm.compute_coin_wallet_metrics(
            self.wallets_coin_config,
            profits_df,
            self.wallets_config['training_data'][f'{period}_period_start'],
            self.wallets_config['training_data'][f'{period}_period_end']
        )

        # Assign wallets to segments
        wallet_segmentation_df = cws.build_wallet_segmentation(
            self.wallets_coin_config,
            self.wallets_config,
            score_suffix=f'|{prd}'
        )

        # Flatten cw_metrics into single values for each coin-segment pair
        coin_wallet_features_df = cwwmf.flatten_cw_to_coin_segment_features(
            cw_metrics_df,
            wallet_segmentation_df,
            self.training_coin_cohort
        )

        # Generate and merge prior model features if configured
        if self.wallets_coin_config['wallet_features']['toggle_prior_model_features']:
            prior_coin_model_features_df = self._generate_prior_coin_model_features()
            prior_coin_model_features_df.to_parquet(
                f"{self.wallets_coin_config['training_data']['parquet_folder']}"
                f"/{prd}_prior_coin_model_features_df.parquet",
                index=True
            )
            coin_training_data_df_full = self._merge_all_features(
                coin_wallet_features_df,
                prior_coin_model_features_df
            )
        else:
            coin_training_data_df_full = coin_wallet_features_df

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

    def _generate_prior_coin_model_features(
        self,
    ) -> pd.DataFrame:
        """
        Generate the “non-wallet” (time-window) coin features and save them to parquet.

        Params:
        - modeling_profits_df (DataFrame): profits for modeling period
        - time_window_generator (callable): tw.generate_all_time_windows_model_inputs

        Returns:
        - coin_non_wallet_features_df (DataFrame): index=coin_id, features from all windows
        """
        # Confirm period boundaries align
        model_start = self.coins_config['training_data']['modeling_period_start']
        val_start = self.wallets_config['training_data']['coin_modeling_period_start']
        model_end = self.coins_config['training_data']['modeling_period_end']
        val_end = self.wallets_config['training_data']['coin_modeling_period_end']
        if not (model_start == val_start and model_end == val_end):
            raise ValueError(
                f"Coin features modeling period must align with wallet features validation period:\n"
                f"Wallet-coin model coin_modeling_period boundaries: {val_start} to {val_end} \n"
                f"Prior coin model modeling_period boundaries: {model_start} to {model_end}"
            )

        # Generate features based on the coin config files
        coin_features_df, _, _ = tw.generate_all_time_windows_model_inputs(
            self.coins_config,
            self.metrics_config,
            self.coins_modeling_config
        )

        # Remove time window index since we aren't using that for now
        coin_features_df = coin_features_df.reset_index(level='time_window', drop=True)

        u.notify('ui_1')

        return coin_features_df

    def _merge_all_features(
        self,
        coin_wallet_features_df: pd.DataFrame,
        prior_coin_model_features_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Inner-join wallet_df & non_wallet_df into final coin-level feature set.

        Params:
        - wallet_df (DataFrame): coin-level features derived from wallet segments.
        - non_wallet_df (DataFrame): coin-level features from time-window engine.

        Returns:
        - merged_df (DataFrame): inner-joined on coin_id, only coins present in both.
        """
        if prior_coin_model_features_df is not None:
            # Confirm overlap
            coin_features_ids = prior_coin_model_features_df.index
            coin_wallet_features_ids = coin_wallet_features_df.index
            wallet_features_only_ids = set(coin_wallet_features_ids) - set(coin_features_ids)

            if len(wallet_features_only_ids) == 0:
                logger.info("All %s coins with wallet features were found in the non wallet coin features set.",
                            len(coin_wallet_features_ids))
            else:
                logger.warning(f"Wallet features contain {len(wallet_features_only_ids)} coins "
                            "not in the non wallet coin features")

            # Join together
            coin_training_data_df_full = coin_wallet_features_df.join(prior_coin_model_features_df,how='inner')

        else:
            # Just return base features if prior model features aren't input
            coin_training_data_df_full = coin_wallet_features_df

        logger.info("Final features shape: %s",coin_training_data_df_full.shape)

        return coin_training_data_df_full




# ------------------------------
#         Helper Functions
# ------------------------------

def load_wallet_scores(wallet_scores: list, wallet_scores_path: str, score_suffix: str = None) -> pd.DataFrame:
    """
    Params:
    - wallet_scores (list): List of score names to merge
    - wallet_scores_path (str): Base path for score parquet files

    Returns:
    - wallet_scores_df (DataFrame):
        wallet_address (index): contains all wallet addresses included in any score
        score|{score_name} (float): the predicted score
        residual|{score_name} (float): the residual of the score
    """
    wallet_scores_df = pd.DataFrame()

    for score_name in wallet_scores:
        score_df = pd.read_parquet(f"{wallet_scores_path}/{score_name}{score_suffix}.parquet")
        feature_cols = []

        # Add scores column
        score_df[f'scores|{score_name}_score'] = score_df[f'score|{score_name}']
        feature_cols.append(f'scores|{score_name}_score')

        # Add residuals column
        if wallets_coin_config['wallet_segments']['wallet_scores_residuals_segments'] is True:
            score_df[f'scores|{score_name}_residual'] = (
                score_df[f'score|{score_name}'] - score_df[f'actual|{score_name}']
            )
            feature_cols.append(f'scores|{score_name}_residual')

        # Add confidence if provided
        if ((wallets_coin_config['wallet_segments']['wallet_scores_confidence_segments'] is True)
            & (f'confidence|{score_name}' in score_df.columns)
            ):
            score_df[f'scores|{score_name}_confidence'] = score_df[f'confidence|{score_name}']
            feature_cols.append(f'scores|{score_name}_confidence')

        # Full outer join with existing results
        wallet_scores_df = (
            score_df[feature_cols] if wallet_scores_df.empty
            else wallet_scores_df.join(score_df[feature_cols], how='outer')
        )

    return wallet_scores_df
