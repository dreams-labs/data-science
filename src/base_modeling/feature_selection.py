import logging
import pandas as pd
import numpy as np


# Set up logger at the module level
logger = logging.getLogger(__name__)

# pylint:disable=invalid-name  # X_test isn't camelcase

def remove_low_variance_features(
    training_df: pd.DataFrame,
    variance_threshold: float = 0.01,
    protected_features: list = None
) -> pd.DataFrame:
    """
    Remove features with variance below threshold, preserving protected features.

    Params:
    - training_df (DataFrame): Input feature matrix
    - variance_threshold (float): Minimum variance to keep feature
    - protected_features (list): Prefixes of features to preserve regardless of variance

    Returns:
    - reduced_df (DataFrame): DataFrame with low variance features removed
    """
    # Calculate variances without recomputing multiple times
    feature_variances = training_df.var()

    # Get features exceeding threshold
    high_variance_features = feature_variances[
        feature_variances > variance_threshold
    ].index.tolist()

    # Add protected features regardless of variance
    if protected_features:
        protected_cols = [
            col for col in training_df.columns
            if any(col.startswith(p) for p in protected_features)
        ]
        high_variance_features = list(set(high_variance_features + protected_cols))

    # Select columns efficiently
    reduced_df = training_df[high_variance_features]

    logger.info(
        f"Removed {training_df.shape[1] - reduced_df.shape[1]} low variance features"
        f" while protecting {len(protected_cols) if protected_features else 0} features"
    )

    return reduced_df


def remove_correlated_features(
    training_df: pd.DataFrame,
    correlation_threshold: float = 0.95,
    protected_features: list = None
) -> pd.DataFrame:
    """
    Remove highly correlated features, preserving protected features.

    Params:
    - training_df (DataFrame): Input feature matrix
    - correlation_threshold (float): Maximum allowed correlation
    - protected_features (list): Prefixes of features to preserve regardless of correlation

    Returns:
    - reduced_df (DataFrame): DataFrame with correlated features removed
    """
    # Calculate correlation matrix
    logger.info("Calculating feature correlation...")
    corr_matrix = training_df.corr().abs()

    # Get upper triangle indices excluding diagonal
    upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

    # Find features to drop, excluding protected ones
    to_drop = []
    for _, j in zip(*np.where(upper & (corr_matrix > correlation_threshold))):
        col_name = corr_matrix.index[j]
        if not protected_features or not any(col_name.startswith(p) for p in protected_features):
            to_drop.append(col_name)

    # Remove duplicates and create filtered df
    to_drop = list(set(to_drop))
    reduced_df = training_df.drop(columns=to_drop)

    protected_count = len([c for c in training_df.columns
                        if protected_features and any(c.startswith(p) for p in protected_features)])

    logger.info(
        f"Removed {len(to_drop)} highly correlated features "
        f"while protecting {protected_count if protected_features else 0} features"
    )
    return reduced_df
