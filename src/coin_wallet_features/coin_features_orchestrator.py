"""
Calculates metrics aggregated at the coin level
"""
import logging
from pathlib import Path
import pandas as pd
import yaml

# Local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig
# import coin_wallet_features.wallet_balance_features as cwb
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()
wallets_metrics_config = u.load_config('../config/wallets_metrics_config.yaml')
wallets_features_config = yaml.safe_load(Path('../config/wallets_features_config.yaml').read_text(encoding='utf-8'))



def load_wallet_scores(wallet_scores: list, wallet_scores_path: str) -> pd.DataFrame:
    """
    Params:
    - wallet_scores (list): List of score names to merge
    - wallet_scores_path (str): Base path for score parquet files

    Returns:
    - wallet_scores_df (DataFrame):
        wallet_address (index): contains all wallet addresses included in any score
        score|{score_name} (float): the predicted score
        residual|{score_name} (float): the residual of the score
    """
    wallet_scores_df = pd.DataFrame()

    for score_name in wallet_scores:
        score_df = pd.read_parquet(f"{wallet_scores_path}/{score_name}.parquet")

        # Add residuals column
        score_df[f'residual|{score_name}'] = (
            score_df[f'score|{score_name}'] - score_df[f'actual|{score_name}']
        )

        # Select only needed columns
        score_subset = score_df[[
            f'score|{score_name}',
            f'residual|{score_name}'
        ]]

        # Full outer join with existing results
        wallet_scores_df = (
            score_subset if wallet_scores_df.empty
            else wallet_scores_df.join(score_subset, how='outer')
        )

    return wallet_scores_df