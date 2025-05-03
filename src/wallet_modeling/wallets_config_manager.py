"""Allows the config file to be imported to other wallet .py files"""
from datetime import datetime, timedelta
from pathlib import Path
import yaml



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

    # Training Period Boundaries
    first_window = min(td['training_window_starts'])
    td['training_period_start'] = first_window
    training_start = datetime.strptime(first_window, "%Y-%m-%d")
    td['training_starting_balance_date'] = (training_start - timedelta(days=1)).strftime("%Y-%m-%d")

    # Modeling Period Boundaries
    modeling_start = datetime.strptime(td['modeling_period_start'], "%Y-%m-%d")
    td['training_period_end'] = (modeling_start - timedelta(days=1)).strftime("%Y-%m-%d")
    td['modeling_starting_balance_date'] = td['training_period_end']

    # Validation Period Boundaries
    # 1. Calculate the modeling period duration in days.
    modeling_end = datetime.strptime(td['modeling_period_end'], "%Y-%m-%d")
    modeling_duration = (modeling_end - modeling_start).days + 1  # period is inclusive of start/end dates

    # 2. Extract the raw validation_period_end from td.
    validation_period_end = datetime.strptime(td['validation_period_end'], "%Y-%m-%d")

    # 3. Create validation_period_start by subtracting the modeling duration from validation_period_end.
    validation_period_start_dt = validation_period_end - timedelta(days=modeling_duration)
    td['validation_period_start'] = validation_period_start_dt

    # 4. Calculate the starting balance date (one day before the period start).
    td['validation_starting_balance_date'] = (validation_period_start_dt - timedelta(days=1)).strftime("%Y-%m-%d")

    # Coin Modeling Period Boundaries
    # -------------------------------
    td['coin_modeling_period_start'] = (modeling_end + timedelta(days=1)).strftime("%Y-%m-%d")
    td['coin_modeling_period_end'] = (modeling_end + timedelta(days=modeling_duration)).strftime("%Y-%m-%d")

    # Investing Period Boundaries
    td['investing_period_start'] = (
        modeling_end + timedelta(days=modeling_duration + 1)
    ).strftime("%Y-%m-%d")
    td['investing_period_end'] = (
        modeling_end + timedelta(days=2 * modeling_duration)
    ).strftime("%Y-%m-%d")

    return cfg
