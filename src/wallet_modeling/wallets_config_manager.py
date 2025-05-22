"""Allows the config file to be imported to other wallet .py files"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import pandas as pd

from coin_modeling.coin_config_manager import WalletsCoinConfig

logger = logging.getLogger(__name__)



# -----------------------------------------------------------
#         Orchestrator: Reload Wallets + Coin Configs
# -----------------------------------------------------------

def load_all_wallets_configs(config_dir: str):
    """
    Reload both wallets and wallets-coin configurations from their YAML files,
    and validate consistency between their datasets and folder suffixes.
    Params:
    - config_dir (str): Path to directory containing 'wallets_config.yaml' and 'wallets_coin_config.yaml'.
    Returns:
    - wallets_config (WalletsConfig): Reloaded wallets configuration instance.
    - wallets_coin_config (WalletsCoinConfig): Reloaded wallets-coin configuration instance.
    """
    # Identify paths
    base_path = Path(config_dir)
    wallets_yaml_path = base_path / 'wallets_config.yaml'
    wallets_coin_yaml_path = base_path / 'wallets_coin_config.yaml'

    # Reload configs
    wallets_config = WalletsConfig.load_from_yaml(wallets_yaml_path)
    wallets_coin_config = WalletsCoinConfig.load_from_yaml(wallets_coin_yaml_path)

    # Populate wallets_coin_config['training_data']
    # ---------------------------------------------
    base_wc_config = wallets_coin_config.config

    # Fill [dataset]
    base_wc_config['training_data']['dataset'] = wallets_config['training_data']['dataset']

    # Fill [parquet_folder]
    base_folder = '/'.join(wallets_config['training_data']['parquet_folder'].split('/')[:-2])
    instance_folder = wallets_config['training_data']['parquet_folder'].split('/')[-1]
    wc_folder = f"{base_folder}/coin_modeling_dfs/{instance_folder}"
    base_wc_config['training_data']['parquet_folder'] = wc_folder

    # Fill [coins_wallet_scores_folder]
    coins_wallet_scores_folder = f"{wc_folder}/scores"
    Path(coins_wallet_scores_folder).mkdir(parents=True, exist_ok=True)
    base_wc_config['training_data']['coins_wallet_scores_folder'] = coins_wallet_scores_folder

    # Fill [model_artifacts_folder]
    path_without_suffix = wallets_config['training_data']['model_artifacts_folder'].rsplit('/', 1)[0]
    base_wc_config['training_data']['model_artifacts_folder'] = f"{path_without_suffix}/coin_modeling"

    # Store updated config
    wallets_coin_config.config = base_wc_config

    # Validate no overlap between coin training and validation epochs
    wallets_coin_config.validate_epochs_validation_overlap(
        wallets_config['training_data']['modeling_period_duration']
    )

    return wallets_config, wallets_coin_config



# -------------------------
#       Primary Class
# -------------------------

class WalletsConfig:
    """
    A singleton class to manage configuration data loaded from a YAML file.

    This class ensures only one instance of the configuration exists across
    the entire application. It loads configuration data from a YAML file
    and provides dictionary-style access to the configuration values.

    Attributes:
        config (dict): The configuration data loaded from the YAML file

    Example:
        # In your main script or notebook:
        config = WalletsConfig.load_from_yaml('config.yaml')

        # In other files:
        config = WalletsConfig()  # Gets the same instance
        value = config['some_key']
    """

    # Class variable to store the singleton instance
    _instance = None
    _yaml_path = None

    def __new__(cls):
        # Ensure only one instance is created (singleton pattern)
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Initialize the config dict only if it doesn't exist
        # This prevents re-initialization when getting existing instance
        if not hasattr(self, 'config'):
            self.config = {}

    @classmethod
    def load_from_yaml(cls, yaml_path):
        """
        Load configuration from a YAML file and store it in the singleton instance.

        Params:
        - yaml_path (str or Path): Path to the YAML configuration file

        Returns:
        - WalletsConfig: The singleton instance with loaded configuration
        """
        instance = cls()
        cls._yaml_path = Path(yaml_path)  # Store the path
        instance.reload()  # Use reload method to load the config
        return instance

    def get(self, key, default=None):
        """
        Safely get a configuration value with a default if the key doesn't exist.

        Args:
            key: The configuration key to look up
            default: Value to return if key is not found (default: None)

        Returns:
            The value for the key if it exists, otherwise the default value

        Example:
            value = config.get('some_key', 'default_value')
        """
        return self.config.get(key, default)

    def reload(self):
        """
        Reload the configuration from the YAML file.
        Raises FileNotFoundError if no YAML file has been specified yet.
        """
        if self._yaml_path is None:
            raise FileNotFoundError("No config file path specified. Call load_from_yaml first.")
        raw_config = yaml.safe_load(self._yaml_path.read_text(encoding='utf-8'))

        # Append _dev suffix to folder if applicable
        if (
            raw_config['training_data']['dataset'] == 'dev' and
            raw_config['training_data']['parquet_folder'][-4:] != '_dev'
        ):
            dev_folder = f"{raw_config['training_data']['parquet_folder']}_dev"
            raw_config['training_data']['parquet_folder'] = dev_folder

        # Impute model_scores_folder
        raw_config['training_data']['model_scores_folder'] = (
            raw_config['training_data']['parquet_folder']
            .replace('wallet_modeling_dfs','wallet_modeling_score_dfs')
        )

        # Add derived values
        self.config = add_derived_values(raw_config)

        # Confirm modeling period is later than all windows
        first_window_start = self.config['training_data']['training_window_starts'][-1]
        modeling_period_start = self.config['training_data']['modeling_period_start']
        if first_window_start > modeling_period_start:
            raise ValueError(f"First window start date of {first_window_start} is later "
                             f"than modeling period start of {modeling_period_start}.")

    def __getitem__(self, key):
        # Enable dictionary-style access with square brackets (config['key'])
        return self.config[key]

    def __contains__(self, key):
        # Enable 'in' operator to check if a key exists (if 'key' in config)
        return key in self.config

    def __str__(self):
        # Provide a string representation of the config for debugging
        return f"WalletsConfig(keys={list(self.config.keys())})"


# -----------------------------------
#           Utility Functions
# -----------------------------------

def add_derived_values(config_dict: dict) -> dict:
    """
    Add calculated values to a config dict including period boundaries and balance dates.

    Params:
    - config_dict (dict): Config dictionary containing training_data section

    Returns:
    - dict: New config dict with added derived values
    """
    if 'training_data' not in config_dict:
        return config_dict.copy()

    # Create a fresh copy to avoid modifying original
    cfg = {k: v.copy() if isinstance(v, dict) else v for k, v in config_dict.items()}
    td = cfg['training_data']

    # Convert training lookbacks to dates
    modeling_start = datetime.strptime(td['modeling_period_start'], "%Y-%m-%d")
    lookbacks = td['training_window_lookbacks']
    td['training_window_starts'] = (modeling_start - pd.to_timedelta(lookbacks, unit='d')).strftime('%Y-%m-%d').tolist()
    first_window = min(td['training_window_starts'])

    # Training Period Boundaries
    training_start = datetime.strptime(first_window, "%Y-%m-%d")
    td['training_period_start'] = first_window
    td['training_starting_balance_date'] = (training_start - timedelta(days=1)).strftime("%Y-%m-%d")
    td['training_period_end'] = (modeling_start - timedelta(days=1)).strftime("%Y-%m-%d")

    # Modeling Period Boundaries
    modeling_duration = td['modeling_period_duration']
    modeling_end = modeling_start + timedelta(days=modeling_duration - 1) # -1 as period is inclusive of start/end dates
    td['modeling_period_end'] = modeling_end.strftime("%Y-%m-%d")
    td['modeling_starting_balance_date'] = td['training_period_end']
    if (td['modeling_period_end'] > td['validation_period_end']
        and not td['training_data_only']):
        raise ValueError(f"Validation period end of {td['validation_period_end']} is earlier than "
                         f"the modeling period end of {td['modeling_period_end']}.")

    # Validation Period Boundaries
    validation_period_start_dt = modeling_end + timedelta(days=modeling_duration)
    td['validation_period_start'] = validation_period_start_dt
    td['validation_starting_balance_date'] = (validation_period_start_dt - timedelta(days=1)).strftime("%Y-%m-%d")

    # Coin Modeling Period Boundaries
    td['coin_modeling_period_start'] = (modeling_end + timedelta(days=1)).strftime("%Y-%m-%d")
    td['coin_modeling_period_end'] = (modeling_end + timedelta(days=modeling_duration)).strftime("%Y-%m-%d")

    # Investing Period Boundaries
    td['investing_period_start'] = (modeling_end + timedelta(days=modeling_duration + 1)).strftime("%Y-%m-%d")
    td['investing_period_end'] = (modeling_end + timedelta(days=2 * modeling_duration)).strftime("%Y-%m-%d")

    return cfg


def validate_config_alignment(config: dict, wallets_config: dict, wallets_coin_config: dict) -> None:
    """
    Validates that configuration objects are properly aligned.

    Params:
    - config (dict): Main configuration object
    - wallets_config (dict): Wallet features configuration
    - wallets_coin_config (dict): Wallet-coin configuration

    Returns:
    - None: Raises ValueError if configurations are misaligned
    """
    # First validate dataset alignment between wallet configs
    if not (wallets_config['training_data']['dataset'] ==
            wallets_coin_config['training_data']['dataset']):
        raise ValueError("Config datasets not aligned:\n"
            f" - wallets_config: {wallets_config['training_data']['dataset']}\n"
            f" - wallets_coin_config: {wallets_coin_config['training_data']['dataset']}\n"
        )

    # If coin flow model features enabled, perform additional validation
    if wallets_coin_config['features']['toggle_coin_flow_model_features']:
        # Confirm period boundaries align
        model_start = config['training_data']['modeling_period_start']
        val_start = wallets_config['training_data']['coin_modeling_period_start']
        model_end = config['training_data']['modeling_period_end']
        val_end = wallets_config['training_data']['coin_modeling_period_end']

        if not (model_start == val_start and model_end == val_end):
            raise ValueError(
                f"Coin features modeling period must align with wallet features validation period:\n"
                f"Wallet-coin model coin_modeling_period boundaries: {val_start} to {val_end} \n"
                f"Coin Flow Model modeling_period boundaries: {model_start} to {model_end}"
            )

        # Validate all three configs have aligned datasets
        if not (wallets_config['training_data']['dataset'] ==
                wallets_coin_config['training_data']['dataset'] ==
                config['training_data']['dataset']):
            raise ValueError("Config datasets not aligned:\n"
                f" - wallets_config: {wallets_config['training_data']['dataset']}\n"
                f" - wallets_coin_config: {wallets_coin_config['training_data']['dataset']}\n"
                f" - config: {config['training_data']['dataset']}"
            )
