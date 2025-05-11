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

        self.wallet_epochs_orchestrator = None



    # -----------------------------------
    #         Primary Interface
    # -----------------------------------

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
        self.wallet_epochs_orchestrator = weo.WalletEpochsOrchestrator(
            wallets_lookback_config, # lookback config
            self.wallets_metrics_config,
            self.wallets_features_config,
            self.wallets_epochs_config
        )
        self.wallet_epochs_orchestrator.load_complete_raw_datasets()

        self.complete_profits_df = self.wallet_epochs_orchestrator.complete_profits_df
        self.complete_market_data_df = self.wallet_epochs_orchestrator.complete_market_data_df
        self.complete_macro_trends_df = self.wallet_epochs_orchestrator.complete_macro_trends_df
        logger.info("Successfully retrieved complete dfs.")



    # -----------------------------------
    #           Helper Methods
    # -----------------------------------

    def _generate_coin_epoch_training_data(self, lookback_duration: int):
        """
        Generates a coin epoch's training data for all wallet windows and epochs.

        Params:
         - lookback_duration (int): How many days the coin lookback is offset vs the base config.
        """
        # Overwrite base config in wallets orchestrator
        coin_epoch_wallets_config = self._prepare_coin_epoch_base_config(lookback_duration)
        self.wallet_epochs_orchestrator.wallets_config = coin_epoch_wallets_config

        # Identify filepaths
        parquet_folder = coin_epoch_wallets_config['training_data']['parquet_folder']
        period = coin_epoch_wallets_config['training_data']['modeling_period_start']
        wallet_td_df_path = f"{parquet_folder}/{period}_multiwindow_wallet_training_data_df.parquet"
        modeling_wf_df_path = f"{parquet_folder}/{period}_multiwindow_modeling_wallet_features_df.parquet"
        validation_td_df_path = f"{parquet_folder}/{period}_multiwindow_validation_training_data_df.parquet"
        validation_wf_df_path = f"{parquet_folder}/{period}_multiwindow_validation_wallet_features_df.parquet"

        # Load and return existing files if they exist
        if os.path.exists(wallet_td_df_path):
            logger.info("Successfully loaded existing wallet epoch dfs.")
            return (
                pd.load_parquet(wallet_td_df_path),
                pd.load_parquet(modeling_wf_df_path),
                pd.load_parquet(validation_td_df_path),
                pd.load_parquet(validation_wf_df_path),
            )

        # If not, generate and save training and modeling dfs for all windows
        (
            wallet_training_data_df,
            modeling_wallet_features_df,
            validation_training_data_df,
            validation_wallet_features_df
        ) = self.wallet_epochs_orchestrator.generate_epochs_training_data()
        wallet_training_data_df.to_parquet(wallet_td_df_path, index=True)
        modeling_wallet_features_df.to_parquet(modeling_wf_df_path, index=True)
        validation_training_data_df.to_parquet(validation_td_df_path, index=True)
        validation_wallet_features_df.to_parquet(validation_wf_df_path, index=True)

        logger.info("Successfully generated and saved wallet epoch dfs.")
        return (
            wallet_training_data_df,
            modeling_wallet_features_df,
            validation_training_data_df,
            validation_wallet_features_df
        )


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
            self.wallets_config.config, # lookback config
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
