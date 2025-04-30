"""
Orchestrates groups of functions to generate wallet model pipeline
"""
import gc
import logging
import pandas as pd

# Local module imports
import feature_engineering.time_windows_orchestration as tw
import coin_wallet_features.coin_features_orchestrator as cfo
import coin_wallet_features.wallet_segmentation as cws
import utils as u

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
    ):
        self.wallets_config = wallets_config
        self.wallets_coin_config = wallets_coin_config
        self.metrics_config = metrics_config
        self.coins_config = coins_config
        self.coins_modeling_config = coins_modeling_config

        # placeholders for intermediate DataFrames
        self.modeling_profits_df = None
        self.modeling_market_data_df = None
        self.como_profits_df = None
        self.como_market_data_df = None
        self.training_coin_cohort = None


    # ----------------------------------
    #           Helper Methods
    # ----------------------------------

    def generate_prior_coin_model_features(
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
        coin_features_training_data_df, _, _ = tw.generate_all_time_windows_model_inputs(
            self.coins_config,
            self.metrics_config,
            self.coins_modeling_config
        )

        # Remove time window index since we aren't using that for now
        coin_features_training_data_df = coin_features_training_data_df.reset_index(level='time_window', drop=True)

        # Save to parquet
        coin_features_training_data_df.to_parquet(
            f"{self.wallets_coin_config['training_data']['parquet_folder']}"
            "/coin_non_wallet_features_training_data_df.parquet",index=True
        )

        u.notify('ui_1')


    def build_wallet_segmentation(self) -> pd.DataFrame:
        """
        Build wallet segmentation DataFrame with score quantiles and optional clusters.
        """
        # Load wallet scores
        wallet_scores_df = cfo.load_wallet_scores(
            self.wallets_coin_config['wallet_segments']['wallet_scores'],
            self.wallets_coin_config['wallet_segments']['wallet_scores_path']
        )
        wallet_segmentation_df = wallet_scores_df.copy()

        # Add "all" segment for full-population aggregations
        wallet_segmentation_df['all_wallets|all'] = 'all'
        wallet_segmentation_df['all_wallets|all'] = wallet_segmentation_df['all_wallets|all'].astype('category')

        # Assign score quantiles
        wallet_segmentation_df = cws.assign_wallet_score_quantiles(
            wallet_segmentation_df,
            self.wallets_coin_config['wallet_segments']['wallet_scores'],
            self.wallets_coin_config['wallet_segments']['score_segment_quantiles']
        )

        # If configured, add training-period cluster labels
        cluster_groups = self.wallets_coin_config['wallet_segments'].get('training_period_cluster_groups')
        if cluster_groups:
            # Load full-window wallet features to generate clusters
            training_df_path = (
                f"{self.wallets_config['training_data']['parquet_folder']}"
                "/wallet_training_data_df_full.parquet"
            )
            training_data_df = pd.read_parquet(training_df_path)
            wallet_clusters_df = cws.assign_cluster_labels(
                training_data_df,
                cluster_groups
            )
            # Join and verify no rows dropped
            orig_len = len(wallet_segmentation_df)
            wallet_segmentation_df = wallet_segmentation_df.join(wallet_clusters_df, how='inner')
            joined_len = len(wallet_segmentation_df)
            if joined_len < orig_len:
                raise ValueError(
                    f"Join dropped {orig_len - joined_len} rows from original {orig_len} rows"
                )
            # Clean up
            del training_data_df
            gc.collect()

        return wallet_segmentation_df
