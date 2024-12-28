import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig
import wallet_insights.wallet_model_evaluation as wime
import coin_insights.coin_validation_analysis as civa
import utils as u


# pylint:disable=invalid-name  # X doesn't conform to snake case

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()
