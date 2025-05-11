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
import utils as u

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

        # Wallet WalletEpochsOrchestrator() Configs and DataFrames
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

    @u.timing_decorator(logging.MILESTONE)  # pylint: disable=no-member
    def load_complete_raw_datasets(self) -> None:
        """
        Identifies the earliest training start date from the earliest coin window
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
            self.complete_profits_df = pd.read_parquet(f"{parquet_folder}/complete_profits_df.parquet")
            self.complete_market_data_df = pd.read_parquet(f"{parquet_folder}/complete_market_data_df.parquet")
            self.complete_macro_trends_df = pd.read_parquet(f"{parquet_folder}/complete_macro_trends_df.parquet")
            logger.info("Loaded complete dfs from parquet.")
            return

        # Generate wallets_lookback_config with the earliest lookback date
        coins_earliest_lookback = max(self.wallets_coin_config['training_data']['training_window_lookbacks'])
        wallets_lookback_config = copy.deepcopy(self.wallets_config)
        wallets_lookback_config.config['training_data']['modeling_period_start'] = (
            pd.to_datetime(wallets_lookback_config['training_data']['modeling_period_start'])
            - timedelta(days=coins_earliest_lookback)
        ).strftime('%Y-%m-%d')
        wallets_lookback_config.reload()

        # Load and store complete dfs
        epochs_orchestrator = weo.WalletEpochsOrchestrator(
            wallets_lookback_config, # lookback config
            self.wallets_metrics_config,
            self.wallets_features_config,
            self.wallets_epochs_config
        )
        epochs_orchestrator.load_complete_raw_datasets()

        self.complete_profits_df = epochs_orchestrator.complete_profits_df
        self.complete_market_data_df = epochs_orchestrator.complete_market_data_df
        self.complete_macro_trends_df = epochs_orchestrator.complete_macro_trends_df
        logger.info("Successfully retrieved complete dfs.")



    # -----------------------------------
    #           Helper Methods
    # -----------------------------------

    def _prepare_coin_epoch_base_config(self, lookback_duration):
        """
        Creates a config with the following changes:
         - dates offset by a wallets_coin_config training_window_lookback
         - parquet_folder subdirectory added
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
        epochs_orchestrator = weo.WalletEpochsOrchestrator(
            self.wallets_config.config, # lookback config
            self.wallets_metrics_config,
            self.wallets_features_config,
            self.wallets_epochs_config
        )
        coin_epoch_base_config = epochs_orchestrator.build_epoch_config(
            lookback_duration,
            'coin_modeling',
            base_modeling_start,
            base_modeling_end,
            base_training_window_starts,
            base_parquet_folder_base
        )

        coin_epoch_base_config['training_data']['parquet_folder'] = (
            f"{coin_epoch_base_config['training_data']['parquet_folder']}_coin_epoch"
        )

        return coin_epoch_base_config
