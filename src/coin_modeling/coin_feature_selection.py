import logging
import pandas as pd
import numpy as np


# Set up logger at the module level
logger = logging.getLogger(__name__)

# pylint:disable=invalid-name  # X_test isn't camelcase

def remove_low_variance_features(
    training_df: pd.DataFrame,
    variance_threshold: float = 0.01
) -> pd.DataFrame:
    """
    Remove features with variance below threshold.

    Params:
    - training_df (DataFrame): Input feature matrix
    - variance_threshold (float): Minimum variance to keep feature

    Returns:
    - reduced_df (DataFrame): DataFrame with low variance features removed
    """
    # Calculate variances without recomputing multiple times
    feature_variances = training_df.var()

    # Get features exceeding threshold
    high_variance_features = feature_variances[
        feature_variances > variance_threshold
    ].index.tolist()

    # Select columns efficiently
    reduced_df = training_df[high_variance_features]

    logger.info(f"Removed {training_df.shape[1] - reduced_df.shape[1]} low variance features")

    return reduced_df

def remove_correlated_features(
    training_df: pd.DataFrame,
    correlation_threshold: float = 0.95
    ) -> pd.DataFrame:
    """
    Remove highly correlated features, keeping first occurrence.

    Params:
    - training_df (DataFrame): Input feature matrix
    - correlation_threshold (float): Maximum allowed correlation

    Returns:
    - reduced_df (DataFrame): DataFrame with correlated features removed
    """
    # Calculate correlation matrix
    logger.info("Calculating feature correlation...")
    corr_matrix = training_df.corr().abs()

    # Get upper triangle indices excluding diagonal
    upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

    # Find features to drop
    to_drop = []
    for _, j in zip(*np.where(upper & (corr_matrix > correlation_threshold))):
        to_drop.append(corr_matrix.index[j])

    # Remove duplicates and create filtered df
    to_drop = list(set(to_drop))
    reduced_df = training_df.drop(columns=to_drop)

    logger.info(f"Removed {len(to_drop)} highly correlated features")
    return reduced_df
