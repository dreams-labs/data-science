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
        self.output_dfs = {}


    @u.timing_decorator
    def generate_windows_training_data(self) -> pd.DataFrame:
        """
        Generates training data for all window configs in parallel.

        Returns:
        - merged_df (pd.DataFrame): DF indexed on wallet_address,
        """
        if not self.all_windows_configs:
            self.all_windows_configs = self._generate_window_configs()

        window_training_data_filepaths = {}
        window_modeling_data_filepaths = {}

        for window_config in self.all_windows_configs:
            model_start = window_config['training_data']['modeling_period_start']
            logger.info(f"Generating training data for window starting {model_start}")

            # Generate name of parquet folder and create it if necessary
            window_parquet_folder = window_config['training_data']['parquet_folder']
            os.makedirs(window_parquet_folder, exist_ok=True)

            try:
                # 1. Initialize data generator for window
                # ---------------------------------------
                data_generator = wmo.WalletTrainingDataOrchestrator(
                    window_config,
                    self.metrics_config,
                    self.features_config
                )

                # 2. Generate TRAINING_DATA_DFs
                # -----------------------------
                # Retrieve base data
                training_profits_df_full,training_market_data_df_full,training_coin_cohort = data_generator.retrieve_period_datasets(
                    window_config['training_data']['training_period_start'],
                    window_config['training_data']['training_period_end']
                )

                # Select cohort and prepare training data
                (training_profits_df, training_market_indicators_df, training_transfers_df
                 ) = data_generator.prepare_training_data(training_profits_df_full, training_market_data_df_full, True)

                # Generate features
                data_generator.generate_training_features(
                    training_profits_df,
                    training_market_indicators_df,
                    training_transfers_df
                )

                # Store training data filepath
                window_training_data_filepaths[model_start] = f"{window_parquet_folder}/wallet_training_data_df_full.parquet"


                # 3. Generate MODELING_DATA_DFs
                # -----------------------------
                # Retrieve base data
                modeling_profits_df_full,_,_ = data_generator.retrieve_period_datasets(
                    window_config['training_data']['modeling_period_start'],
                    window_config['training_data']['modeling_period_end'],
                    training_coin_cohort)

                # Generate modeling wallet features
                hybrid_cw_id_map = None
                if window_config['training_data']['hybridize_wallet_ids']:
                    hybrid_cw_id_map = pd.read_pickle(f"{window_parquet_folder}/hybrid_cw_id_map.pkl")
                _ = data_generator.prepare_modeling_features(
                    modeling_profits_df_full,
                    hybrid_cw_id_map
                )

                # Store modeling data filepath
                window_modeling_data_filepaths[model_start] = f"{window_parquet_folder}/modeling_wallet_features_df.parquet"


                logger.info(f"Successfully generated training data for window {model_start}")
                u.notify('click_2')


            except Exception as e:
                logger.error(f"Failed to generate training data for {model_start}: {str(e)}")
                raise

        merged_df = self._merge_window_training_data(window_training_data_filepaths)
        logger.info("Generated multi-window TRAINING_DATA_DF with shape %s", merged_df.shape)
        u.notify('level_up')

        return merged_df



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



    def _merge_window_training_data(self,window_training_data_filepaths: Dict[str, pd.DataFrame]
        ) -> pd.DataFrame:
        """
        Merges training data from multiple windows into a single DataFrame with window dates in index.
        Uses parquet files generated by generate_windows_training_data().

        Returns:
        - merged_df (DataFrame): MultiIndexed DataFrame with (wallet_address, window_start_date)
        """
        logger.info("Beginning merge of %d window training datasets...",
                    len(window_training_data_filepaths))

        # Get window dates from config
        window_dates = [
            datetime.strptime(date, '%Y-%m-%d')
            for date in self.base_config['training_data']['training_window_starts']
        ]

        # Create mapping of model start dates to file paths
        date_file_map = dict(zip(window_dates, window_training_data_filepaths.values()))

        merged_dfs = []
        for window_date, filepath in date_file_map.items():
            # Read window training data
            window_df = pd.read_parquet(filepath)

            # Add window date to index
            window_df = window_df.set_index(
                pd.MultiIndex.from_product(
                    [window_df.index, [window_date]],
                    names=['wallet_address', 'window_start_date']
                )
            )

            merged_dfs.append(window_df)

        # Combine all windows
        merged_df = pd.concat(merged_dfs, axis=0)

        # Save merged result
        output_path = f"{self.base_config['training_data']['parquet_folder']}/merged_training_data.parquet"
        merged_df.to_parquet(output_path)
        logger.info("Saved merged training data with shape %s to %s", merged_df.shape, output_path)

        return merged_df
