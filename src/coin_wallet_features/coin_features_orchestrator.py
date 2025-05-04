"""
Calculates metrics aggregated at the coin level
"""
import logging
from pathlib import Path
import pandas as pd
import yaml

# Local module imports
# import coin_wallet_features.wallet_balance_features as cwb

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_coin_config = yaml.safe_load(Path('../config/wallets_coin_config.yaml').read_text(encoding='utf-8'))







# ------------------------------
#         Helper Functions
# ------------------------------

def load_wallet_scores(wallet_scores: list, wallet_scores_path: str, score_suffix: str = None) -> pd.DataFrame:
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
        score_df = pd.read_parquet(f"{wallet_scores_path}/{score_name}{score_suffix}.parquet")
        feature_cols = []

        # Add scores column
        score_df[f'scores|{score_name}_score'] = score_df[f'score|{score_name}']
        feature_cols.append(f'scores|{score_name}_score')

        # Add residuals column
        if wallets_coin_config['wallet_segments']['wallet_scores_residuals_segments'] is True:
            score_df[f'scores|{score_name}_residual'] = (
                score_df[f'score|{score_name}'] - score_df[f'actual|{score_name}']
            )
            feature_cols.append(f'scores|{score_name}_residual')

        # Add confidence if provided
        if ((wallets_coin_config['wallet_segments']['wallet_scores_confidence_segments'] is True)
            & (f'confidence|{score_name}' in score_df.columns)
            ):
            score_df[f'scores|{score_name}_confidence'] = score_df[f'confidence|{score_name}']
            feature_cols.append(f'scores|{score_name}_confidence')

        # Full outer join with existing results
        wallet_scores_df = (
            score_df[feature_cols] if wallet_scores_df.empty
            else wallet_scores_df.join(score_df[feature_cols], how='outer')
        )

    return wallet_scores_df
