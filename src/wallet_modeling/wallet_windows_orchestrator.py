"""Orchestrates the creation of training_data_dfs for multiple training windows"""
import os
import logging
from pathlib import Path
import copy
from typing import List, Dict
from datetime import datetime, timedelta
import pandas as pd

# Local module imports
import wallet_modeling.wallets_config_manager as wcm
import wallet_modeling.wallet_modeling_orchestrator as wmo
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)


# ----------------------------------------
#       Primary Orchestration Class
# ----------------------------------------

class MultiWindowOrchestrator:
    """
    Orchestrates training data generation across multiple time windows by
    offsetting base config dates and managing the resulting datasets.
    """
    def __init__(
        self,
        base_config: dict,
        metrics_config: dict,
        features_config: dict,
        windows_config: dict
    ):
        # Param Configs
        self.base_config = base_config
        self.metrics_config = metrics_config
        self.features_config = features_config
        self.windows_config = windows_config

        # Generated objects
        self.all_windows_configs = None
        self.window_training_data_filepaths = None
        self.output_dfs = {}


    @u.timing_decorator
    def generate_windows_training_data(self) -> Dict[str, pd.DataFrame]:
        """
        Generates training data for all window configs in parallel.

        Returns:
        - Dict[str, pd.DataFrame]: Dictionary mapping window start dates to their training DFs
        """
        if not self.all_windows_configs:
            self.all_windows_configs = self._generate_window_configs()

        window_training_data_filepaths = {}

        for window_config in self.all_windows_configs:
            model_start = window_config['training_data']['modeling_period_start']
            logger.info(f"Generating training data for window starting {model_start}")

            # Generate name of parquet folder and create it if necessary
            parquet_folder = window_config['training_data']['parquet_folder']
            os.makedirs(parquet_folder, exist_ok=True)

            try:
                # Initialize data generator with window-specific config
                data_generator = wmo.WalletTrainingDataOrchestrator(
                    window_config,
                    self.metrics_config,
                    self.features_config
                )

                # 1. Retrieve base data
                _,_,_ = data_generator.retrieve_period_datasets(
                    window_config['training_data']['training_period_start'],
                    window_config['training_data']['training_period_end'],
                    parquet_prefix='training'
                )

                # 2. Select cohort and prepare training data
                training_profits_df_full = pd.read_parquet(f"{parquet_folder}/training_profits_df_full.parquet")
                training_market_data_df_full = pd.read_parquet(f"{parquet_folder}/training_market_data_df_full.parquet")
                data_generator.prepare_training_data(training_profits_df_full,training_market_data_df_full)

                # 3. Generate training features
                parquet_folder = window_config['training_data']['parquet_folder']
                training_profits_df = pd.read_parquet(f"{parquet_folder}/training_profits_df.parquet")
                training_market_indicators_df = pd.read_parquet(f"{parquet_folder}/training_market_indicators_data_df.parquet")
                training_transfers_df = pd.read_parquet(f"{parquet_folder}/training_transfers_sequencing_df.parquet")

                data_generator.generate_training_features(
                    training_profits_df,
                    training_market_indicators_df,
                    training_transfers_df
                )

                # Store with model start date as key
                training_data_filepath = f"{parquet_folder}/wallet_training_data_df_full.parquet"
                window_training_data_filepaths[model_start] = training_data_filepath

                logger.info(f"Successfully generated training data for {model_start}")

            except Exception as e:
                logger.error(f"Failed to generate training data for {model_start}: {str(e)}")
                raise

        self.window_training_data_filepaths = window_training_data_filepaths



    # -----------------------------------
    #           Helper Methods
    # -----------------------------------

    def _generate_window_configs(self) -> List[Dict]:
        """
        Generates config dicts for each window, including the base (0 offset) and offset windows.

        Returns:
        - List[Dict]: List of config dicts, one per window.
        """
        all_windows_configs = []
        base_training_data = self.base_config['training_data']
        # Include 0 offset for the base config along with other offsets
        offsets = [0] + self.windows_config['offset_windows']['offsets']

        for offset_days in offsets:
            # Deep copy base config to prevent mutations
            window_config = copy.deepcopy(self.base_config)

            # Offset key dates
            for date_key in [
                'modeling_period_start',
                'modeling_period_end',
                'validation_period_end'
            ]:
                base_date = datetime.strptime(base_training_data[date_key], '%Y-%m-%d')
                new_date = base_date + timedelta(days=offset_days)
                window_config['training_data'][date_key] = new_date.strftime('%Y-%m-%d')

            # Offset every date in training_window_starts
            window_config['training_data']['training_window_starts'] = [
                (datetime.strptime(dt, '%Y-%m-%d') + timedelta(days=offset_days)).strftime('%Y-%m-%d')
                for dt in base_training_data['training_window_starts']
            ]

            # Update parquet folder based on new modeling_period_start
            model_start = window_config['training_data']['modeling_period_start']
            folder_suffix = datetime.strptime(model_start, '%Y-%m-%d').strftime('%y%m%d')
            base_folder = Path(base_training_data['parquet_folder'])
            window_config['training_data']['parquet_folder'] = str(base_folder / folder_suffix)

            # Use WalletsConfig to add derived values
            window_config = wcm.add_derived_values(window_config)

            # Log window configuration
            logger.info(
                f"Generated config for {offset_days} day offset window: "
                f"modeling_period={window_config['training_data']['modeling_period_start']} "
                f"to {window_config['training_data']['modeling_period_end']}"
            )

            all_windows_configs.append(window_config)

        return all_windows_configs