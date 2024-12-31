"""
Functions for assigning wallets to different segments or cohorts, which can then be
used to compare metrics between the features.
"""
import logging
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

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



def assign_wallet_quantiles(score_series: pd.Series, quantiles: list[float]) -> pd.DataFrame:
    """
    Assigns each wallet to a single quantile bucket based on score.

    Params:
    - score_series (Series): Score values indexed by wallet_address
    - quantiles (list[float]): List of quantile thresholds in ascending order (e.g. [0.4, 0.6, 0.8])
        Represents percentile breakpoints for binning

    Returns:
    - DataFrame: DataFrame with new score quantile column named 'score_{score_name}_quantile'
        indicating which quantile bucket the wallet belongs to (e.g. '0_40pct')
    """
    # Validate and sort quantiles
    quantiles = sorted(quantiles)
    if not all(0 < q < 1 for q in quantiles):
        raise ValueError("Quantiles must be between 0 and 1")

    # Calculate score thresholds based on data distribution
    score_thresholds = score_series.quantile(quantiles)

    # Create bin edges using actual score distribution
    bin_edges = [-float('inf')] + list(score_thresholds) + [float('inf')]

    # Create labels for each bin (e.g. '0_40pct', '40_60pct', etc)
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        start_pct = int(quantiles[i-1] * 100) if i > 0 else 0
        end_pct = int(quantiles[i] * 100) if i < len(quantiles) else 100
        bin_labels.append(f'{start_pct}_{end_pct}pct')

    # Create result DataFrame with quantile assignments
    result_df = pd.DataFrame(index=score_series.index)
    column_name = f'score_quantile|{score_series.name.split("|")[1]}'
    result_df[column_name] = pd.cut(
        score_series,
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True
    ).astype(str)

    # Validate all wallets have segments and segment count is correct
    unique_segments = result_df[column_name].unique()
    if result_df[column_name].isna().any():
        raise ValueError("Some wallets are missing segment assignments")
    if len(unique_segments) != len(quantiles) + 1:
        raise ValueError(f"Expected {len(quantiles) + 1} segments but found {len(unique_segments)}: {unique_segments}")

    return result_df



def quantize_all_wallet_scores(wallet_segmentation_df, wallet_scores, score_segment_quantiles):
    """
    Generates categorical quantiles for each wallet across each score and residual.

    Params:
    - wallet_segmentation_df (df): index wallet_address with columns score|{} and residual|{}
        for all scores in wallet_scores param
    - wallet_scores (list of strings): the names of the scores and residuals to quantized
    - score_segment_quantiles (list of floats): the percentiles to use as quantile boundaries

    Returns:
    - wallet_segmentation_df (df): the param df but with added score_quantile|{} columns
        for each score and residual
    """
    for score_name in wallet_scores:

        # Append score quantiles
        wallet_quantiles = assign_wallet_quantiles(wallet_segmentation_df[f'scores|{score_name}/score'],
                                                   score_segment_quantiles)
        wallet_segmentation_df = wallet_segmentation_df.join(wallet_quantiles,how='inner')

        # Append residual quantiles
        wallet_quantiles = assign_wallet_quantiles(wallet_segmentation_df[f'scores|{score_name}/residual'],
                                                   score_segment_quantiles)
        wallet_segmentation_df = wallet_segmentation_df.join(wallet_quantiles,how='inner')

    return wallet_segmentation_df



def add_quantile_columns(validation_coin_wallet_features_df: pd.DataFrame, n_quantiles: int) -> pd.DataFrame:
    """
    Adds quantile columns for each base metric in the dataframe.

    Params:
    - validation_coin_wallet_features_df (DataFrame): Input features dataframe
    - n_quantiles (int): Number of quantiles to split into (e.g. 4 for quartiles)

    Returns:
    - DataFrame: Original dataframe with added quantile columns
    """
    result_df = validation_coin_wallet_features_df.copy()
    base_cols = [col for col in result_df.columns if col != 'coin_return']

    for col in base_cols:
        quantile_col = f"{col}/quantile_{n_quantiles}"
        try:
            # Convert categorical output to int
            result_df[quantile_col] = pd.qcut(
                result_df[col],
                q=n_quantiles,
                labels=False,  # Returns 0-based indices
                duplicates='drop'
            ) + 1  # Shift to 1-based indices
        except ValueError:
            bins = np.linspace(
                result_df[col].min(),
                result_df[col].max(),
                n_quantiles + 1
            )
            result_df[quantile_col] = pd.cut(
                result_df[col],
                bins=bins,
                labels=False,
                include_lowest=True
            ) + 1

    return result_df
