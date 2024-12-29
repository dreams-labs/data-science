# """
# Calculates metrics aggregated at the wallet level
# """
# import logging
# from pathlib import Path
# import pandas as pd
# import yaml

# # Local module imports
# from wallet_modeling.wallets_config_manager import WalletsConfig
# import coin_wallet_features.wallet_balance_features as cwb
# import utils as u

# # Set up logger at the module level
# logger = logging.getLogger(__name__)

# # Load wallets_config at the module level
# wallets_config = WalletsConfig()
# wallets_metrics_config = u.load_config('../config/wallets_metrics_config.yaml')
# wallets_features_config = yaml.safe_load(Path('../config/wallets_features_config.yaml').read_text(encoding='utf-8'))
