"""
Calculates metrics aggregated at the wallet-coin-date level
"""
import logging
import time
from typing import Dict, Any
import pandas as pd
import numpy as np

# Local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig
import utils as u

# set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()


def force_fill_market_cap(market_data_df):
    """
    Creates a column 'market_cap_filled' with complete coverage by:
    1. Back- and forwardfilling any existing values which breached thresholds in
        of data_retrieval.impute_market_cap().
    2. Blanket filling the remaining empty values with a constant value from the config.

    Params:
    - market_data_df (df): dataframe with column date, coin_id, market_cap, market_cap_imputed
    Returns:
    - market_data_df (df): input dataframe with new column 'market_cap_filled'
    """

    # Force fill market cap to generate a complete feature set
    market_data_df['market_cap_filled'] = market_data_df['market_cap_imputed']

    # Backfill and forward fill if there are any values at all, since these missing values failed imputation checks
    market_data_df['market_cap_filled'] = market_data_df.groupby('coin_id')['market_cap_filled'].bfill()
    market_data_df['market_cap_filled'] = market_data_df.groupby('coin_id')['market_cap_filled'].ffill()

    # For coins with no market cap data at all, bulk fill with a constant
    default_market_cap = wallets_config['data_cleaning']['market_cap_default_fill']
    market_data_df['market_cap_filled'] = market_data_df['market_cap_filled'].fillna(default_market_cap)

    return market_data_df


def calculate_volume_weighted_market_cap(profits_market_features_df):
    """
    Calculate volume-weighted average market cap for each wallet address.
    If volume is 0 for all records of a wallet, uses simple average instead.

    Parameters:
    - profits_market_features_df (df): DataFrame containing wallet_address, volume, and market_cap_filled

    Returns:
    pandas.DataFrame: DataFrame with wallet addresses and their weighted average market caps
    """
    # Create a copy to avoid modifying original
    df_calc = profits_market_features_df.copy()

    # Group by wallet_address
    grouped = df_calc.groupby('wallet_address')

    # Calculate weighted averages
    results = []

    for wallet, group in grouped:
        total_volume = group['volume'].sum()

        if total_volume > 0:
            # If there's volume, calculate volume-weighted average
            weighted_avg = np.average(
                group['market_cap_filled'],
                weights=group['volume']
            )
        else:
            # If no volume, use simple average
            weighted_avg = group['market_cap_filled'].mean()

        results.append({
            'wallet_address': wallet,
            'total_volume': total_volume,
            'volume_wtd_market_cap': weighted_avg
        })

    volume_wtd_df = pd.DataFrame(results)
    volume_wtd_df = volume_wtd_df.set_index('wallet_address')

    return volume_wtd_df


def calculate_balance_weighted_market_cap(profits_market_features_df):
    """
    Calculate USD balance-weighted average market cap for each wallet address
    using only the most recent date's data.

    Parameters:
    - profits_market_features_df (df): DataFrame containing wallet_address, usd_balance,
        date, and market_cap_filled columns

    Returns:
    - balance_wtd_df (df): DataFrame with wallet addresses and their balance-weighted
        average market caps
    """
    # Create a copy to avoid modifying original
    df_calc = profits_market_features_df.copy()

    # Get the latest date
    latest_date = df_calc['date'].max()

    # Filter for only the latest date
    latest_df = df_calc[df_calc['date'] == latest_date]

    # Group by wallet_address
    results = []

    for wallet, group in latest_df.groupby('wallet_address'):
        total_balance = group['usd_balance'].sum()

        if total_balance > 0:
            # If there's balance, calculate balance-weighted average
            weighted_avg = np.average(
                group['market_cap_filled'],
                weights=group['usd_balance']
            )
        else:
            # If no balance, use simple average
            weighted_avg = group['market_cap_filled'].mean()

        results.append({
            'wallet_address': wallet,
            'ending_portfolio_usd': total_balance,
            'portfolio_wtd_market_cap': weighted_avg
        })

    balance_wtd_df = pd.DataFrame(results)
    balance_wtd_df = balance_wtd_df.set_index('wallet_address')

    return balance_wtd_df


def calculate_market_data_features(profits_df,market_data_df):
    """
    Calculates each wallet's total volume and ending balance, and the average market cap of coins
    they interacted with weighted by the volume and ending balances.

    Params:
    - profits_df
    - market_data_df

    Returns:
    - market_features_df (df): dataframe indexed on wallet_address that contains columns
        total_volume, volume_wtd_market_cap, ending_portfolio_usd, portfolio_wtd_market_cap

    """
    start_time = time.time()
    logger.debug("Calculating market data features...")

    # Fully fill market cap data
    filled_market_cap_df = force_fill_market_cap(market_data_df)

    # Generate simplified profits df
    profits_market_features_df = profits_df[['coin_id','date','wallet_address','usd_balance']].copy()
    profits_market_features_df['volume'] = abs(profits_df['usd_net_transfers'])

    # Append the filled market data to the simplified profits
    profits_market_features_df = profits_market_features_df.merge(
        filled_market_cap_df[['date', 'coin_id', 'market_cap_filled']],
        on=['date', 'coin_id'],
        how='inner'
    )

    # Calculate weighted metrics
    volume_wtd_df = calculate_volume_weighted_market_cap(profits_market_features_df)
    balance_wtd_df = calculate_balance_weighted_market_cap(profits_market_features_df)

    # Merge into a wallet-indexed df of features
    market_features_df = volume_wtd_df.copy()
    market_features_df = market_features_df.join(balance_wtd_df)

    logger.debug("Successfully calculated market data features after %.2f.",
                 time.time() - start_time)

    return market_features_df


class FeatureConfigError(Exception):
    """Custom exception for feature configuration errors."""
    pass

def calculate_offsets(
    market_timing_df: pd.DataFrame,
    wallets_features_config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calculate offset values for specified columns in market timing dataframe.

    Args:
        market_timing_df: DataFrame containing market timing data
        wallets_features_config: Configuration dictionary containing offset specifications
            in the format:
            market_timing:
              offsets:
                column_name:
                  offsets: [list_of_offsets]
                  retain_base_columns: bool

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
                result_df[new_column] = result_df.groupby('coin_id')[column].shift(-lead)
            except Exception as e:
                raise FeatureConfigError(f"Error calculating offset for column '{column}' " \
                                      f"with lead {lead}: {str(e)}") from e

    return result_df


def calculate_relative_changes(
    market_timing_df: pd.DataFrame,
    wallets_features_config: Dict[str, Any]
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
        wallets_features_config: Configuration dictionary containing offset specifications,
            retention settings, and winsorization coefficient

    Returns:
        DataFrame with added relative change columns and optionally dropped base/offset columns

    Raises:
        FeatureConfigError: If configuration is invalid or required columns are missing
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    result_df = market_timing_df.copy()

    # Get the offsets configuration
    try:
        offset_config = wallets_features_config['market_timing']['offsets']
        winsor_coef = wallets_config['data_cleaning']['offset_winsorization']
    except KeyError as e:
        raise FeatureConfigError("Required config key not found in wallets_features_config: " + str(e)) from e

    # Keep track of columns to drop
    columns_to_drop = set()
    # Keep track of columns to winsorize
    columns_to_winsorize = []

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
                columns_to_winsorize.append(change_column)

                # If retain_base_columns is False, mark columns for dropping
                if not retain_base_columns:
                    columns_to_drop.add(base_column)
                    columns_to_drop.add(offset_column)

            except Exception as e:
                raise FeatureConfigError(f"Error calculating relative change between '{base_column}' " \
                                      f"and '{offset_column}': {str(e)}") from e

    # Winsorize all change columns
    for column in columns_to_winsorize:
        result_df[column] = u.winsorize(result_df[column], winsor_coef)

    # Drop marked columns if any
    if columns_to_drop:
        result_df = result_df.drop(columns=list(columns_to_drop))

    return result_df
