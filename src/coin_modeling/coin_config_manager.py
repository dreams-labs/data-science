"""
Allows the coin config file to be imported to other coin‐modeling .py files.
"""

from pathlib import Path
import yaml


class WalletsCoinConfig:
    """
    A singleton class to manage configuration data loaded from a YAML file.

    This class ensures only one instance of the configuration exists across
    the application. It loads configuration data from a YAML file and provides
    dict-style access to the configuration values.
    """

    _instance = None
    _yaml_path = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize the config dict once
        if not hasattr(self, 'config'):
            self.config = {}

    @classmethod
    def load_from_yaml(cls, yaml_path):
        """
        Load configuration from a YAML file and store it in the singleton.

        Params:
        - yaml_path (str or Path): path to the YAML config file

        Returns:
        - WalletsCoinConfig: the singleton instance with loaded config
        """
        instance = cls()
        cls._yaml_path = Path(yaml_path)
        instance.reload()
        return instance


    def reload(self):
        """
        Reload the configuration from the YAML file.

        Raises:
        - FileNotFoundError: if no YAML path has been set via load_from_yaml.
        """
        if self._yaml_path is None:
            raise FileNotFoundError(
                "No config file path specified. Call load_from_yaml first."
            )
        raw_config = yaml.safe_load(self._yaml_path.read_text(encoding='utf-8'))

        self.config = raw_config
        # Validate that distributions correspond to defined score parameters
        self._validate_score_distribution_features()


    def get(self, key, default=None):
        """
        Safely get a configuration value with a default.

        Params:
        - key: config key to look up
        - default: value to return if key isn’t present

        Returns:
        - The config value or default
        """
        return self.config.get(key, default)

    def __getitem__(self, key):
        return self.config[key]

    def __contains__(self, key):
        return key in self.config

    def __str__(self):
        return f"WalletsCoinConfig(keys={list(self.config.keys())})"

    def _validate_score_distribution_features(self):
        """
        Validate that each score in wallet_features.score_distributions
        has a corresponding key in wallet_scores.score_params.
        """
        features = self.config.get('wallet_features', {}).get('score_distributions', [])
        params_keys = set(self.config.get('wallet_scores', {}).get('score_params', {}).keys())
        missing = [s for s in features if s not in params_keys]
        if missing:
            raise ValueError(
                f"Invalid score_distributions entries not found in score_params: {missing}"
            )
