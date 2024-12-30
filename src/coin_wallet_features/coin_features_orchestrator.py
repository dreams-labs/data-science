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


def load_merge_feature_scores(feature_scores: list, base_path: str) -> pd.DataFrame:
    """
    Params:
    - feature_scores (list): List of feature score types to load
    - base_path (str): Base path to parquet files directory

    Returns:
    - merged_df (DataFrame): Combined features dataframe with de-duplicated columns
    """
    # Load first df to get index structure
    merged_df = pd.read_parquet(f"{base_path}/{feature_scores[0]}.parquet")
    base_cols = set(merged_df.columns)

    # Merge remaining dfs with suffix handling for overlaps
    for score in feature_scores[1:]:
        curr_df = pd.read_parquet(f"{base_path}/{score}.parquet")
        overlap_cols = list(set(curr_df.columns) & base_cols)

        # Only add suffix if there are overlapping columns
        suffix = (None, f'_{score}') if overlap_cols else (None, None)
        merged_df = merged_df.merge(curr_df, left_index=True, right_index=True,
                                  suffixes=suffix)

    return merged_df
