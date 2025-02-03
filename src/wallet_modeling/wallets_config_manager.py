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
        cls._yaml_path = Path(yaml_path)
        instance.reload()
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

def add_derived_values(config: dict) -> dict:
    """
    Add calculated values to a config dict including period boundaries and balance dates.

    Params:
    - config (dict): Config dictionary containing training_data section

    Returns:
    - dict: Config with added derived values
    """
    if 'training_data' not in config:
        return config

    cfg = config.copy()

    # Training Period Boundaries
    first_window = min(cfg['training_data']['training_window_starts'])
    cfg['training_data']['training_period_start'] = first_window

    # Training balance date (1 day before period start)
    training_start = datetime.strptime(first_window, "%Y-%m-%d")
    training_balance_date = (training_start - timedelta(days=1)).strftime("%Y-%m-%d")
    cfg['training_data']['training_starting_balance_date'] = training_balance_date

    # Modeling Period Boundaries
    modeling_start = datetime.strptime(cfg['training_data']['modeling_period_start'], "%Y-%m-%d")
    training_end = (modeling_start - timedelta(days=1)).strftime("%Y-%m-%d")
    cfg['training_data']['training_period_end'] = training_end

    # Modeling Period Balance Date
    cfg['training_data']['modeling_starting_balance_date'] = training_end

    # Validation Period Boundaries
    modeling_end = datetime.strptime(cfg['training_data']['modeling_period_end'], "%Y-%m-%d")
    validation_start = (modeling_end + timedelta(days=1)).strftime("%Y-%m-%d")
    cfg['training_data']['validation_period_start'] = validation_start

    # Validation Period Balance Date
    cfg['training_data']['validation_starting_balance_date'] = modeling_end

    return cfg