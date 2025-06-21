"""
Functions for generating flattened macroeconomic and market time series features for coins.

This module provides utilities to aggregate and transform macroeconomic and market indicator
time series data into single-row, coin-level features. These features are designed to be
cross-joined onto each coin's record for downstream modeling and analysis.

Main functionalities include:
- Generating flattened macroeconomic features from time series data.
- Generating flattened market features from time series data.
- Renaming feature columns for clarity and consistency.

These functions support the feature engineering pipeline for coin-level predictive modeling.
"""

import logging
import pandas as pd

# Local module imports
import wallet_features.time_series_features as wfts

# set up logger at the module level
logger = logging.getLogger(__name__)



# --------------------------------------
#        Features Main Interface
# --------------------------------------

def generate_macro_features(
    macro_indicators_df: pd.DataFrame,
    macro_trends_metrics_config: dict
) -> pd.DataFrame:
    """
    Generates flattened macroeconomic time series features for coins.

        Params:
    - macro_indicators_df (DataFrame): date-indexed macroeconomic indicators
    - macro_trends_metrics_config (dict): defines the time series features that will
        be output. Defined at wallets_coins_metrics_config['time_series']['macro_trends']

    Returns:
    - macro_features_df (pd.DataFrame): single row dataframe containing a
        column for each macro feature. These flattened features will be
        cross joined onto every coin's record
    """
    macro_features_df = wfts.calculate_macro_features(
        macro_indicators_df,
        macro_trends_metrics_config
    )

    # Rename macro feature columns: replace first underscore after key with '/'
    macro_keys = macro_trends_metrics_config.keys()
    rename_map = {}
    for col in macro_features_df.columns:
        for key in macro_keys:
            prefix = f"{key}_"
            if col.startswith(prefix):
                rename_map[col] = f"{key}/{col[len(prefix):]}"
                break
    if rename_map:
        macro_features_df = macro_features_df.rename(columns=rename_map)

    return macro_features_df



def generate_market_features(
    market_indicators_df: pd.DataFrame,
    marlet_data_metrics_config: dict,
) -> pd.DataFrame:
    """
    Generates flattened market data time series features for coins.

    Params:
    - market_indicators_df (DataFrame): date-indexed marketeconomic indicators
    - marlet_data_metrics_config (dict): defines the time series features that will
        be output. Defined at wallets_coins_metrics_config['time_series']['market_data']

    Returns:
    - market_features_df (pd.DataFrame): single row dataframe containing a
        column for each market feature. These flattened features will be
        cross joined onto every coin's record
    """
    market_features_df = wfts.calculate_market_data_features(
        market_indicators_df,
        marlet_data_metrics_config
    )

    # Rename market feature columns: replace first underscore after key with '/'
    market_keys = marlet_data_metrics_config.keys()
    rename_map = {}
    for col in market_features_df.columns:
        for key in market_keys:
            prefix = f"{key}_"
            if col.startswith(prefix):
                rename_map[col] = f"{key}/{col[len(prefix):]}"
                break
    if rename_map:
        market_features_df = market_features_df.rename(columns=rename_map)

    return market_features_df
