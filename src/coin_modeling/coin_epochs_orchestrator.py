"""
Orchestrates the creation of wallet training data, models, and scores for each coin lookback
 window, then converts them into coin features that are merged into a single dataframe along
 with their epoch reference date.
"""
import os
import logging
from pathlib import Path
import copy
from datetime import datetime, timedelta
import pandas as pd

# Local module imports
import wallet_modeling.wallet_epochs_orchestrator as weo
import wallet_modeling.wallets_config_manager as wcm
import wallet_modeling.wallet_model_orchestrator as wmo

# import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)


# ----------------------------------------
#       Primary Orchestration Class
# ----------------------------------------

class CoinEpochsOrchestrator:
    """
    Orchestrates training data generation across multiple epochs by
    offsetting base config dates and managing the resulting datasets.
    """
    def __init__(
        self,
        wallets_coin_config: dict,
        wallets_config: dict,
        wallets_metrics_config: dict,
        wallets_features_config: dict,
        wallets_epochs_config: dict,
        complete_profits_df: pd.DataFrame = None,
        complete_market_data_df: pd.DataFrame = None,
        complete_macro_trends_df: pd.DataFrame = None,
    ):
        # Coin Params
        self.wallets_coin_config = wallets_coin_config

        # WalletEpochsOrchestrator Configs and DataFrames
        self.wallets_config = wallets_config
        self.wallets_metrics_config = wallets_metrics_config
        self.wallets_features_config = wallets_features_config
        self.wallets_epochs_config = wallets_epochs_config
        self.complete_profits_df = complete_profits_df
        self.complete_market_data_df = complete_market_data_df
        self.complete_macro_trends_df = complete_macro_trends_df


    # -----------------------------------
    #         Primary Interface
    # -----------------------------------

    def load_complete_raw_datasets(self) -> None:
        """
        Identifies the earliest training start date from the earliest coin epoch
        and generates complete dfs for them.
        """
        # Return existing dfs if available
        if self.complete_profits_df is not None:
            logger.info("Stored complete dfs will be used.")
            return

        # Load the existing files if avilable
        parquet_folder = self.wallets_config['training_data']['parquet_folder']
        if os.path.exists(f"{parquet_folder}/complete_profits_df.parquet"):
            parquet_folder = self.wallets_config['training_data']['parquet_folder']

            # Store in self
            self.complete_profits_df = pd.read_parquet(f"{parquet_folder}/complete_profits_df.parquet")
            self.complete_market_data_df = pd.read_parquet(f"{parquet_folder}/complete_market_data_df.parquet")
            self.complete_macro_trends_df = pd.read_parquet(f"{parquet_folder}/complete_macro_trends_df.parquet")
            logger.info("Loaded complete dfs from parquet.")
            return

        # Generate wallets_lookback_config with the earliest lookback date
        coins_earliest_epoch = min(self.wallets_coin_config['training_data']['coin_epoch_lookbacks'])
        wallets_lookback_config = copy.deepcopy(self.wallets_config)
        earliest_modeling_period_start = (
            pd.to_datetime(wallets_lookback_config['training_data']['modeling_period_start'])
            + timedelta(days=coins_earliest_epoch)
        ).strftime('%Y-%m-%d')
        wallets_lookback_config_dict = copy.deepcopy(wallets_lookback_config.config)
        wallets_lookback_config_dict['training_data']['modeling_period_start'] = earliest_modeling_period_start
        wallets_lookback_config = wcm.add_derived_values(wallets_lookback_config_dict)

        # Load and store complete dfs
        wallet_orchestrator = weo.WalletEpochsOrchestrator(
            wallets_lookback_config, # lookback config
            self.wallets_metrics_config,
            self.wallets_features_config,
            self.wallets_epochs_config,
        )
        wallet_orchestrator.load_complete_raw_datasets()

        self.complete_profits_df = wallet_orchestrator.complete_profits_df
        self.complete_market_data_df = wallet_orchestrator.complete_market_data_df
        self.complete_macro_trends_df = wallet_orchestrator.complete_macro_trends_df
        logger.info("Successfully retrieved complete dfs.")



    def orchestrate_coin_epochs(self) -> None:
        """
        Orchestrate coin-level epochs by iterating through lookbacks and processing each epoch.
        """
        for lookback in self.wallets_coin_config['training_data']['coin_epoch_lookbacks']:
            self._process_coin_epoch(lookback)


    # Primary helper
    def _process_coin_epoch(self, lookback_duration: int) -> None:
        """
        Process a single coin epoch: generate data, train wallet models, generate coin modeling data, and score wallets.
        """
        # 1) Generate epoch-specific WalletEpochsOrchestrator and training data
        epoch_weo, epoch_training_dfs = self._generate_coin_epoch_training_data(lookback_duration)

        # 2) Prepare coin config with date suffix folders
        epoch_coins_config = self._prepare_epoch_coins_config(epoch_weo)

        # 3) Train and score wallet models for this epoch
        models_dict, wamo_training_data_df = self._train_and_score_wallet_epoch(
            epoch_weo, epoch_coins_config, epoch_training_dfs
        )




    # -----------------------------------
    #           Helper Methods
    # -----------------------------------

    def _generate_coin_epoch_training_data(self, lookback_duration: int):
        """
        Generates a coin epoch's training data for all wallet windows and epochs.

        Params:
         - lookback_duration (int): How many days the coin lookback is offset vs the base config.

         Returns:
         - epoch_orch (WalletEpochsOrchestrator): orchestrator with config set for lookback
         - (
                wallet_training_data_df,modeling_wallet_features_df,
                validation_training_data_df,validation_wallet_features_df
            ) Tuple(pd.DataFrames): training_data_df and target variables to the epoch

        # 1) Build a fresh epoch-offset config
        """
        epoch_wallets_config = self._prepare_coin_epoch_base_config(lookback_duration)

        # 2) Instantiate a new WalletEpochsOrchestrator for this epoch
        epoch_orch = weo.WalletEpochsOrchestrator(
            base_config=epoch_wallets_config,
            metrics_config=self.wallets_metrics_config,
            features_config=self.wallets_features_config,
            epochs_config=self.wallets_epochs_config,
            complete_profits_df=self.complete_profits_df,
            complete_market_data_df=self.complete_market_data_df,
            complete_macro_trends_df=self.complete_macro_trends_df,
        )

        # 3) Generate its epoch configs
        epoch_orch.all_epochs_configs = epoch_orch.generate_epoch_configs()

        # 4) Generate and return its training & modeling data
        return epoch_orch, epoch_orch.generate_epochs_training_data()



    def _prepare_coin_epoch_base_config(self, lookback_duration: int):
        """
        Creates a config with dates offset by a wallets_coin_config lookback_duration.

        Params:
         - lookback_duration (int): How many days the coin lookback is offset vs the base config.
        """
        # Cache parsed base dates and base folder path
        base_training_data = copy.deepcopy(self.wallets_config.config['training_data'])
        base_modeling_start = datetime.strptime(base_training_data['modeling_period_start'], '%Y-%m-%d')
        base_modeling_end = datetime.strptime(base_training_data['modeling_period_end'], '%Y-%m-%d')
        base_training_window_starts = [
            datetime.strptime(dt, '%Y-%m-%d')
            for dt in base_training_data['training_window_starts']
        ]
        base_parquet_folder_base = Path(base_training_data['parquet_folder'])

        # Load and store complete dfs
        wallet_epochs_orchestrator = weo.WalletEpochsOrchestrator(
            self.wallets_config.config, # base config
            self.wallets_metrics_config,
            self.wallets_features_config,
            self.wallets_epochs_config
        )
        coin_epoch_base_config = wallet_epochs_orchestrator.build_epoch_config(
            lookback_duration,
            'coin_modeling',
            base_modeling_start,
            base_modeling_end,
            base_training_window_starts,
            base_parquet_folder_base
        )

        return coin_epoch_base_config



    def _prepare_epoch_coins_config(self, epoch_weo) -> dict:
        """
        Prepare epoch-specific coins config with date suffix folders.
        """
        # Build a date suffix from the modeling_period_start
        wamo_date_suffix = pd.to_datetime(
            epoch_weo.base_config['training_data']['modeling_period_start']
        ).strftime('%Y%m%d')

        # Deep-copy the coin config and adjust folder paths
        epoch_coins_config = copy.deepcopy(self.wallets_coin_config.config)
        base_folder = epoch_coins_config['training_data']['parquet_folder']

        parquet_folder = f"{base_folder}/{wamo_date_suffix}"
        epoch_coins_config['training_data']['parquet_folder'] = parquet_folder
        Path(parquet_folder).mkdir(exist_ok=True)

        scores_folder = f"{parquet_folder}/scores"
        epoch_coins_config['training_data']['coins_wallet_scores_folder'] = scores_folder
        Path(scores_folder).mkdir(exist_ok=True)
        return epoch_coins_config



    def _train_and_score_wallet_epoch(
        self,
        epoch_weo,
        epoch_coins_config: dict,
        epoch_training_dfs: tuple
    ) -> tuple[dict, pd.DataFrame]:
        """
        Train wallet models for a single epoch and score wallets.

        Returns:
        - models_dict (dict): trained model objects per score.
        - wamo_training_data_df (DataFrame): training data used for scoring.
        """
        # Instantiate the WalletModelOrchestrator for this epoch
        epoch_wmo = wmo.WalletModelOrchestrator(
            epoch_weo.base_config,
            self.wallets_metrics_config,
            self.wallets_features_config,
            self.wallets_epochs_config,
            epoch_coins_config
        )

        # Train wallet models using all epoch dfs
        models_dict = epoch_wmo.train_wallet_models(*epoch_training_dfs)

        # Score wallets on the first training df
        wamo_training_data_df = epoch_training_dfs[0]
        epoch_wmo.predict_and_store(models_dict, wamo_training_data_df)

        return models_dict, wamo_training_data_df
