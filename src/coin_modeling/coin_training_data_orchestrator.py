"""
Orchestrates groups of functions to generate wallet model pipeline
"""
import gc
import logging
import importlib
import pandas as pd

# Local module imports
import feature_engineering.time_windows_orchestration as tw
import coin_wallet_features.coin_features_orchestrator as cfo
import coin_wallet_features.wallet_segmentation as cws
import coin_wallet_features.wallet_base_metrics as cwbm
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
        coin_features_df, _, _ = tw.generate_all_time_windows_model_inputs(
            self.coins_config,
            self.metrics_config,
            self.coins_modeling_config
        )

        # Remove time window index since we aren't using that for now
        coin_features_df = coin_features_df.reset_index(level='time_window', drop=True)

        u.notify('ui_1')

        return coin_features_df


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


    def compute_coin_wallet_metrics(self, modeling_profits_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute coin-wallet–level metrics: balances and trading metrics.

        Params:
        - modeling_profits_df (DataFrame): must include columns ['coin_id','wallet_address',…].

        Returns:
        - cw_metrics_df (DataFrame): MultiIndex [coin_id, wallet_address] with
          'balances/...’ and 'trading/...’ feature columns.
        """
        # 1) Build base index of all (coin, wallet) pairs
        idx = (
            modeling_profits_df[['coin_id', 'wallet_address']]
            .drop_duplicates()
            .set_index(['coin_id', 'wallet_address'])
            .index
        )
        cw_metrics_df = pd.DataFrame(index=idx)

        # 2) Validate configured balance dates
        valid_balance_dates = [
            self.wallets_config['training_data']['modeling_starting_balance_date'],
            self.wallets_config['training_data']['modeling_period_end']
        ]
        bd = self.wallets_coin_config['wallet_features']['wallet_balance_dates']
        if not all(d in valid_balance_dates for d in bd):
            raise ValueError(
                f"wallet_balance_dates {bd} must be one of {valid_balance_dates}"
            )

        # 3) Calculate balances
        balances_df = cwbm.calculate_coin_wallet_balances(
            modeling_profits_df,
            bd
        ).add_prefix('balances/')
        cw_metrics_df = (
            cw_metrics_df
            .join(balances_df, how='left')
            .fillna({col: 0 for col in balances_df.columns})
        )

        # 4) Calculate trading metrics
        trading_df = cwbm.calculate_coin_wallet_trading_metrics(
            modeling_profits_df,
            self.wallets_config['training_data']['modeling_period_start'],
            self.wallets_config['training_data']['modeling_period_end'],
            self.wallets_coin_config['wallet_features']['drop_trading_metrics']
        ).add_prefix('trading/')
        cw_metrics_df = (
            cw_metrics_df
            .join(trading_df, how='left')
            .fillna({col: 0 for col in trading_df.columns})
        )

        return cw_metrics_df


    def flatten_cw_to_coin_segment_features(
        self,
        cw_metrics_df: pd.DataFrame,
        wallet_segmentation_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Flatten coin-wallet metrics into coin-level features across segments.

        Params:
        - cw_metrics_df (DataFrame): indexed by (coin_id, wallet_address) with 'balances/...'
          and 'trading/...' columns.
        - wallet_segmentation_df (DataFrame): indexed by wallet_address with segment assignments.

        Returns:
        - coin_wallet_features_df (DataFrame): indexed by coin_id, joined features for each
          metric × segment family.
        """
        # start with an empty coin-level df
        coin_wallet_features_df = pd.DataFrame(index=self.training_coin_cohort)
        coin_wallet_features_df.index.name = 'coin_id'

        # identify which segmentation columns to use
        segmentation_families = wallet_segmentation_df.columns[
            ~wallet_segmentation_df.columns.str.startswith('scores|')
        ]

        # loop through each metric × segment and join
        total_metrics = len(cw_metrics_df.columns)
        for i, metric_column in enumerate(cw_metrics_df.columns, start=1):
            for segment_family in segmentation_families:
                # generate coin-level features for this metric & segment
                segment_df = cfo.flatten_cw_to_coin_features(
                    cw_metrics_df,
                    metric_column,
                    wallet_segmentation_df,
                    segment_family,
                    self.training_coin_cohort
                )
                coin_wallet_features_df = coin_wallet_features_df.join(segment_df, how='inner')
            logger.info("Flattened metric %s/%s: %s", i, total_metrics, metric_column)

        logger.info(
            "All wallet-based features flattened. Final shape: %s",
            coin_wallet_features_df.shape
        )
        return coin_wallet_features_df


    def merge_all_features(
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
