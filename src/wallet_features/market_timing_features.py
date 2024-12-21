"""
Calculates metrics aggregated at the wallet level
"""
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yaml


# Local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Locate the config directory
current_dir = Path(__file__).parent
config_directory = current_dir / '..' / '..' / 'config'

# Load wallets_config at the module level
wallets_config = WalletsConfig()
wallets_metrics_config = u.load_config(config_directory / 'wallets_metrics_config.yaml')
wallets_features_config = yaml.safe_load((config_directory / 'wallets_features_config.yaml').read_text(encoding='utf-8'))  # pylint:disable=line-too-long

@u.timing_decorator
def calculate_market_timing_features(profits_df, market_indicators_data_df):
    """
    Calculate features capturing how wallet transaction timing aligns with future market movements.

    This function performs a sequence of transformations to assess wallet timing performance:
    1. Enriches market data with technical indicators (RSIs, SMAs) on price and volume
    2. Calculates future values of these indicators at different time offsets (e.g., 7, 14, 30 days ahead)
    3. Computes relative changes between current and future indicator values
    4. For each wallet's transactions, calculates value-weighted and unweighted averages of these future changes,
        treating buys and sells separately

    Args:
        profits_df (pd.DataFrame): Wallet transaction data with columns:
            - wallet_address (pd.Categorical): Unique wallet identifier
            - coin_id (str): Identifier for the traded coin
            - date (pd.Timestamp): Transaction date
            - usd_net_transfers (float): USD value of transaction (positive=buy, negative=sell)

        market_indicators_data_df (pd.DataFrame): Raw market data with columns:
            - coin_id (str): Identifier for the coin
            - date (pd.Timestamp): Date of market data
            - price (float): Asset price
            - volume (float): Trading volume
            - all of the indicators specific in wallets_metrics_config

    Returns:
        pd.DataFrame: Features indexed by wallet_address with columns for each market timing metric:
            {indicator}_vs_lead_{n}_{direction}_{type}
            where:
            - indicator: The base metric (e.g., price_rsi_14, volume_sma_7)
            - n: The forward-looking period (e.g., 7, 14, 30 days)
            - direction: 'buy' or 'sell'
            - type: 'weighted' (by USD value) or 'mean'

            All wallet_addresses from the categorical index are included with zeros for missing data.
    """

    # add timing offset features
    market_timing_df = calculate_offsets(market_indicators_data_df)
    market_timing_df,relative_change_columns = calculate_relative_changes(market_timing_df)

    # flatten the wallet-coin-date transactions into wallet-indexed features
    wallet_timing_features_df = generate_all_timing_features(
        profits_df,
        market_timing_df,
        relative_change_columns,
        wallets_config['features']['timing_metrics_min_transaction_size'],
    )

    return wallet_timing_features_df



class FeatureConfigError(Exception):
    """Custom exception for feature configuration errors."""
    pass

def calculate_offsets(
    market_timing_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate offset values for specified columns in market timing dataframe.

    Args:
        market_timing_df: DataFrame containing market timing data

    Returns:
        DataFrame with added offset columns

    Raises:
        FeatureConfigError: If configuration is invalid or required columns are missing
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    result_df = market_timing_df.copy()

    # Get the offsets configuration
    try:
        offset_config = wallets_features_config['market_timing']['offsets']
    except KeyError as e:
        raise FeatureConfigError("Config key ['market_timing']['offsets'] was not found in " \
                               "wallets_features_config.") from e

    # Process each column and its offsets
    for column, column_config in offset_config.items():
        # Check if the column exists in the DataFrame
        if column not in result_df.columns:
            raise FeatureConfigError(f"Column '{column}' not found in DataFrame")

        # Extract offsets from the new config structure
        try:
            # Handle both old and new config formats
            if isinstance(column_config, dict):
                offsets = column_config['offsets']
            else:
                # Backward compatibility for old format
                offsets = column_config
        except KeyError as e:
            raise FeatureConfigError(f"'offsets' key not found in configuration for column '{column}'") from e
        except Exception as e:
            raise FeatureConfigError(f"Invalid configuration format for column '{column}': {str(e)}") from e

        # Calculate offset for each specified lead value
        for lead in offsets:
            new_column = f"{column}_lead_{lead}"
            try:
                result_df[new_column] = result_df.groupby('coin_id',observed=True)[column].shift(-lead)
            except Exception as e:
                raise FeatureConfigError(f"Error calculating offset for column '{column}' " \
                                      f"with lead {lead}: {str(e)}") from e

    return result_df


def calculate_relative_changes(
    market_timing_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate relative changes between base columns and their corresponding offset columns.

    For each base column with offset columns (e.g., price_rsi_14 and price_rsi_14_lead_30),
    calculates the percentage change between them and stores it in a new column
    (e.g., price_rsi_14_vs_lead_30). If retain_base_columns is False, drops the original
    base and offset columns after calculating the changes. All changes are winsorized
    using the configured coefficient.

    Args:
        market_timing_df: DataFrame containing market timing data with offset columns

    Returns:
        DataFrame with added relative change columns and optionally dropped base/offset columns
        relative_change_columns: List of the relative change columns created

    Raises:
        FeatureConfigError: If configuration is invalid or required columns are missing
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    result_df = market_timing_df.copy()

    # Get the offsets configuration
    try:
        offset_config = wallets_features_config['market_timing']['offsets']
        winsor_coef = wallets_config['features']['offset_winsorization']
    except KeyError as e:
        raise FeatureConfigError("Required config key not found in wallets_config['features']: " + str(e)) from e

    # Keep track of columns to drop
    columns_to_drop = set()
    # Keep track of columns to winsorize
    relative_change_columns = []

    # Process each column and calculate relative changes
    for base_column, column_config in offset_config.items():
        # Check if the base column exists in the DataFrame
        if base_column not in result_df.columns:
            raise FeatureConfigError(f"Base column '{base_column}' not found in DataFrame")

        try:
            # Extract configuration
            if isinstance(column_config, dict):
                offsets = column_config['offsets']
                retain_base_columns = column_config.get('retain_base_columns', True)
            else:
                # Backward compatibility
                offsets = column_config
                retain_base_columns = True
        except KeyError as e:
            raise FeatureConfigError(f"Invalid configuration for column '{base_column}': {str(e)}") from e

        # Calculate relative change for each offset
        for lead in offsets:
            offset_column = f"{base_column}_lead_{lead}"

            # Check if offset column exists
            if offset_column not in result_df.columns:
                raise FeatureConfigError(f"Offset column '{offset_column}' not found in DataFrame. " \
                                      "Run calculate_offsets() first.")

            # Create new column name for relative change
            change_column = f"{base_column}_vs_lead_{lead}"

            try:
                # Calculate percentage change
                # Formula: (offset_value - base_value) / base_value
                result_df[change_column] = (
                    (result_df[offset_column] - result_df[base_column]) /
                    result_df[base_column]
                )

                # Handle division by zero cases
                result_df[change_column] = result_df[change_column].replace([np.inf, -np.inf], np.nan)

                # Add column to winsorization list
                relative_change_columns.append(change_column)

                # If retain_base_columns is False, mark columns for dropping
                if not retain_base_columns:
                    columns_to_drop.add(base_column)
                    columns_to_drop.add(offset_column)

            except Exception as e:
                raise FeatureConfigError(f"Error calculating relative change between '{base_column}' " \
                                      f"and '{offset_column}': {str(e)}") from e

    # Winsorize all change columns
    for column in relative_change_columns:
        result_df[column] = u.winsorize(result_df[column], winsor_coef)

    # Drop marked columns if any
    if columns_to_drop:
        result_df = result_df.drop(columns=list(columns_to_drop))

    return result_df,relative_change_columns




def calculate_timing_features_for_column(df, metric_column):
    """
    Calculate timing features for a single metric column from pre-merged DataFrame.

    Args:
        df (pd.DataFrame): Pre-merged DataFrame with columns [wallet_address, usd_net_transfers, metric_column]
        metric_column (str): Name of the column to analyze

    Returns:
        pd.DataFrame: DataFrame indexed by wallet_address with columns:
            - {metric_column}_buy_weighted
            - {metric_column}_buy_mean
            - {metric_column}_sell_weighted
            - {metric_column}_sell_mean
    """
    # Split into buys and sells
    buys = df[df['usd_net_transfers'] > 0].copy()
    sells = df[df['usd_net_transfers'] < 0].copy()

    features = pd.DataFrame(index=df['wallet_address'].unique())

    # Vectorized buy calculations
    if not buys.empty:
        # Regular mean
        features[f"{metric_column}_buy_mean"] = (
            buys.groupby('wallet_address')[metric_column].mean()
        )

        # Weighted mean: First compute the products, then group
        buys['weighted_values'] = buys[metric_column] * abs(buys['usd_net_transfers'])
        weighted_sums = buys.groupby('wallet_address')['weighted_values'].sum()
        weight_sums = buys.groupby('wallet_address')['usd_net_transfers'].apply(abs).sum()
        features[f"{metric_column}_buy_weighted"] = weighted_sums / weight_sums

    # Similar for sells
    if not sells.empty:
        features[f"{metric_column}_sell_mean"] = (
            sells.groupby('wallet_address')[metric_column].mean()
        )

        sells['weighted_values'] = sells[metric_column] * abs(sells['usd_net_transfers'])
        weighted_sums = sells.groupby('wallet_address')['weighted_values'].sum()
        weight_sums = sells.groupby('wallet_address')['usd_net_transfers'].apply(abs).sum()
        features[f"{metric_column}_sell_weighted"] = weighted_sums / weight_sums

    return features


def generate_all_timing_features(
    profits_df,
    market_timing_df,
    relative_change_columns,
    min_transaction_size=0
):
    """
    Generate timing features for multiple market metric columns.

    Args:
        profits_df (pd.DataFrame): DataFrame with columns [coin_id, date, wallet_address, usd_net_transfers]
        market_timing_df (pd.DataFrame): DataFrame with market timing metrics indexed by (coin_id, date)
        relative_change_columns (list): List of column names from market_timing_df to analyze
        min_transaction_size (float): Minimum absolute USD value of transaction to consider

    Returns:
        pd.DataFrame: DataFrame indexed by wallet_address with generated features for each input column
    """
    # Get list of wallets to include
    wallet_addresses = profits_df['wallet_address'].unique()

    # Filter by minimum transaction size
    filtered_profits = profits_df[
        abs(profits_df['usd_net_transfers']) >= min_transaction_size
    ].copy()

    # Perform the merge once
    timing_profits_df = filtered_profits.merge(
        market_timing_df[relative_change_columns + ['coin_id', 'date']],
        on=['coin_id', 'date'],
        how='left'
    )

    # Calculate features for each column
    all_features = []
    for col in relative_change_columns:
        logger.debug("Generating timing performance features for %s...", col)
        col_features = calculate_timing_features_for_column(
            timing_profits_df,
            col
        )
        all_features.append(col_features)

    # Combine all feature sets
    if all_features:
        result = pd.concat(all_features, axis=1)
    else:
        result = pd.DataFrame(
            index=pd.Index(wallet_addresses, name='wallet_address')
        )

    # Ensure all wallet addresses are included and fill NaNs
    result = pd.DataFrame(
        result,
        index=pd.Index(wallet_addresses, name='wallet_address')
    ).fillna(0)

    return result
