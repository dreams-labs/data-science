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
        Validate that each feature in features.score_distributions
        has a corresponding key in wallet_scores.score_params.
        """
        # List of distributions to validate
        distributions = self.config['features']['score_distributions']
        # Valid score parameters
        params_keys = set(self.config['wallet_scores']['score_params'].keys())
        # Identify any missing distributions
        missing = [d for d in distributions if d not in params_keys]
        if missing:
            raise ValueError(
                f"Invalid score_distributions entries not found in score_params: {missing}"
            )


    def validate_epochs_validation_overlap(self, period_duration: int):
        """
        Validates that there is no overlap between coin training and validation epochs

        Params:
        - period_duration (int): How long a modeling period lasts for, as defined in
            wallets_config['training_data']['modeling_period_duration']
        """
        td_epochs = self.config['training_data']['coin_epoch_lookbacks']
        val_epochs = self.config['training_data']['coin_epochs_validation']
        latest_td_end = max(td_epochs) + period_duration
        earliest_val_start = min(val_epochs)
        if latest_td_end > earliest_val_start:
            raise ValueError(f"Latest training epoch offset ends at day {latest_td_end} which overlaps "
                             f"with the earliest validation epoch offset of {earliest_val_start}.")
