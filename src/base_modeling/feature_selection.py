"""
Utilities for removing low-variance and highly correlated features.
Supports pattern matching with wildcards and protected feature lists.
Used by FeatureSelector and DropColumnPatterns pipeline steps.
"""
import logging
import fnmatch
from typing import List,Set
import pandas as pd
import numpy as np


# Set up logger at the module level
logger = logging.getLogger(__name__)

# pylint:disable=invalid-name  # X_test isn't camelcase


# ------------------------------------
#      Column Dropping Functions
# ------------------------------------

def identify_matching_columns(
        column_patterns: List[str],
        all_columns: List[str],
        protected_columns: List[str] = None
    ) -> Set[str]:
    """
    Match columns that contain all non-wildcard parts of patterns, preserving sequence and structure.

    Params:
    - column_patterns: List of patterns with * wildcards.
    - all_columns: List of actual column names.
    - protected_columns: List of columns to exclude from results.

    Returns:
    - matched_columns: Set of columns matching any pattern, minus protected columns.
    """
    matched = set()
    for pattern in column_patterns:
        for column in all_columns:
            # Match using fnmatch to preserve structure and sequence
            if fnmatch.fnmatch(column, pattern):
                matched.add(column)

    if protected_columns:
        matched = matched - set(protected_columns)

    return matched


def validate_drop_params(df: pd.DataFrame, wallets_config: dict):
    """
    Utility function that identifies feature_selector.drop_patterns that don't
    match any columns.

    Params:
    - df (pd.DataFrame): df with columns that should match the drop_patterns
    - wallets_config (dict): from yaml
    """
    base_patterns = wallets_config['modeling']['feature_selection']['drop_patterns']

    # check for patterns that match nothing
    all_cols = df.columns.tolist()
    missing = [p for p in base_patterns
               if not identify_matching_columns([p], all_cols)]

    if missing:
        missing_str = "\n".join(missing)
        logger.info(f"Base drop patterns with no matches:\n{missing_str}")
    else:
        logger.info("All base patterns match at least one column.")



# ------------------------------------
#    Variance/Correlation Functions
# ------------------------------------

def remove_low_variance_features(
    training_df: pd.DataFrame,
    variance_threshold: float = None,
    protected_features: list = None,
    scale_before_selection: bool = True
) -> pd.DataFrame:
    """
    Remove features with variance below threshold, preserving protected features.
    If scale_before_selection is True, applies variance threshold to standardized data.

    Params:
    - training_df (DataFrame): Input feature matrix
    - variance_threshold (float): Minimum variance to keep feature
    - protected_features (list): Prefixes of features to preserve regardless of variance
    - scale_before_selection (bool): Whether to standardize before variance calculation

    Returns:
    - reduced_df (DataFrame): DataFrame with low variance features removed
    """
    if variance_threshold is None or variance_threshold < 0:
        logger.info("Skipping variance-based feature selection...")
        return training_df  # no changes

    # Work on a copy so we don’t mutate in-place
    _df = training_df.copy()

    # Either scale first or just compute raw variances
    if scale_before_selection:
        scaled = (_df - _df.mean()) / _df.std(ddof=0)  # ddof=0 to avoid 1/n bias
        variances = scaled.var()
    else:
        variances = _df.var()

    # Identify columns with variance <= threshold
    columns_to_drop = variances[variances <= variance_threshold].index.tolist()

    # Remove protected columns from that list
    if protected_features:
        columns_to_drop = [
            col
            for col in columns_to_drop
            if not any(col.startswith(prefix) for prefix in protected_features)
        ]

    # Short-circuit if we end up with zero columns to drop
    if not columns_to_drop:
        logger.info("No columns found below the variance threshold. Returning original data unchanged.")
        return training_df

    # Otherwise drop them
    logger.info(f"Dropping {len(columns_to_drop)} columns with variance <= {variance_threshold}")
    return training_df.drop(columns=columns_to_drop)


def remove_correlated_features(
    training_df: pd.DataFrame,
    correlation_threshold: float = None,
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
    # If the threshold is 1.0 then don't compute anything
    if correlation_threshold is None or correlation_threshold > 1.0:
        logger.info("Skipping correlation-based feature selection...")
        return training_df

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
        f"while protecting {protected_count if protected_features else 0} features."
    )
    return reduced_df
