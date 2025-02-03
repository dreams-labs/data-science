import logging
from pathlib import Path
import copy
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import pandas as pd

# Local module imports
import wallet_modeling.wallets_config_manager as wcm
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)


# ----------------------------------------
#       Primary Orchestration Class
# ----------------------------------------

class MultiWindowTrainingOrchestrator:
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
        self.base_config = base_config
        self.metrics_config = metrics_config
        self.features_config = features_config
        self.windows_config = windows_config
        self.output_dfs = {}


    def generate_window_configs(self) -> List[Dict]:
        """
        Generates config dicts for each offset window.

        Returns:
        - List[Dict]: List of config dicts, one per offset window
        """
        window_configs = []
        base_training_data = self.base_config['training_data']

        for offset_days in self.windows_config['offset_windows']['offsets']:
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

            # Create window-specific parquet folder
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

            window_configs.append(window_config)

        return window_configs



    # -----------------------------------
    #           Helper Functions
    # -----------------------------------
