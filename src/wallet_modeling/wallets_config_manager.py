"""Allows the config file to be imported to other wallet .py files"""

from datetime import datetime, timedelta
from pathlib import Path
import yaml

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

        Args:
            yaml_path (str or Path): Path to the YAML configuration file

        Returns:
            WalletsConfig: The singleton instance with loaded configuration

        Example:
            config = WalletsConfig.load_from_yaml('../config/config.yaml')
        """
        instance = cls()
        cls._yaml_path = Path(yaml_path)  # Store the path
        instance.reload()  # Use reload method to load the config
        instance._add_derived_values()  # Add derived training period boundary dates
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
        self.config = yaml.safe_load(self._yaml_path.read_text(encoding='utf-8'))
        self._add_derived_values()  # Add derived training period boundary dates

    def _add_derived_values(self):
        """Add calculated values to the config."""
        if 'training_data' in self.config:
            # Training Period Boundaries
            # Get the first training window start date
            first_window = min(self.config['training_data']['training_window_starts'].values())
            self.config['training_data']['training_period_start'] = first_window

            # Get the day before modeling period start
            modeling_start = datetime.strptime(
                self.config['training_data']['modeling_period_start'],
                "%Y-%m-%d"
            )
            training_end = (modeling_start - timedelta(days=1)).strftime("%Y-%m-%d")
            self.config['training_data']['training_period_end'] = training_end

            # Validation Period Boundaries
            modeling_end = datetime.strptime(
                self.config['training_data']['modeling_period_end'],
                "%Y-%m-%d"
            )
            validation_start = (modeling_end + timedelta(days=1)).strftime("%Y-%m-%d")
            self.config['training_data']['validation_period_start'] = validation_start


    def __getitem__(self, key):
        # Enable dictionary-style access with square brackets (config['key'])
        return self.config[key]

    def __contains__(self, key):
        # Enable 'in' operator to check if a key exists (if 'key' in config)
        return key in self.config

    def __str__(self):
        # Provide a string representation of the config for debugging
        return f"WalletsConfig(keys={list(self.config.keys())})"
