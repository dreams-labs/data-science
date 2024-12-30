"""
Functions for assigning wallets to different segments or cohorts, which can then be
used to compare metrics between the features.
"""
import logging
from pathlib import Path
import yaml
import pandas as pd

# local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Locate the config directory
current_dir = Path(__file__).parent
config_directory = current_dir / '..' / '..' / 'config'

# Load wallets_config at the module level
wallets_config = WalletsConfig()
wallets_coin_config = yaml.safe_load((config_directory / 'wallets_coin_config.yaml').read_text(encoding='utf-8'))  # pylint:disable=line-too-long



def assign_wallet_quantiles(wallet_scores_df: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    """
    Assigns each wallet to a single quantile bucket based on score.

    Params:
    - wallet_scores_df (DataFrame): Wallet scores data indexed by wallet_address
        Must contain 'score' column
    - quantiles (list[float]): List of quantile thresholds in ascending order (e.g. [0.4, 0.6, 0.8])
        Higher values represent better scores

    Returns:
    - DataFrame: Original wallet_scores_df with new 'score_quantile' column as strings
        indicating which quantile bucket the wallet belongs to (e.g. '0_40pct')

    Raises:
    - ValueError: If any wallets are missing segments or segment count is incorrect
    """
    # Validate and sort quantiles
    quantiles = sorted(quantiles)
    if not all(0 < q < 1 for q in quantiles):
        raise ValueError("Quantiles must be between 0 and 1")

    # Create bin edges including -inf and 1
    bin_edges = [-float('inf')] + quantiles + [1]

    # Create labels for each bin (e.g. '0_40pct', '40_60pct', etc)
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        start_pct = int(max(0, bin_edges[i]) * 100)
        end_pct = int(bin_edges[i + 1] * 100)
        bin_labels.append(f'{start_pct}_{end_pct}pct')

    # Assign quantile labels using pd.cut with ascending order and convert to string
    result_df = wallet_scores_df.copy()
    result_df['score_quantile'] = pd.cut(
        result_df['score'],
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True
    ).astype(str)

    # Validate all wallets have segments and segment count is correct
    unique_segments = result_df['score_quantile'].unique()
    if result_df['score_quantile'].isna().any():
        raise ValueError("Some wallets are missing segment assignments")
    if len(unique_segments) != len(quantiles) + 1:
        raise ValueError(f"Expected {len(quantiles) + 1} segments but found {len(unique_segments)}")

    return result_df[['score_quantile']]
