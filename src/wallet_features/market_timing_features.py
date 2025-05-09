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
wallets_features_config = yaml.safe_load((config_directory / 'wallets_features_config.yaml').read_text(encoding='utf-8'))  # pylint:disable=line-too-long


# -----------------------------
# Main Interface
# -----------------------------
@u.timing_decorator
def calculate_market_timing_features(
        profits_df,
        market_indicators_data_df,
        macro_indicators_df = None
    ):
    """
    Calculate features capturing how wallet transaction timing aligns with future market movements.

    This function performs a sequence of transformations to assess wallet timing performance:
    1. Enriches market data with technical indicators (RSIs, SMAs) on price and volume
    2. Calculates relative indicator changes between transaction dates and offset dates:
        - Lead offsets (positive n): The percentage change in an indicator from transaction date to n days later
        - Lag offsets (negative n): The percentage change in an indicator from n days before to the transaction date
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

        macro_indicators_df (pd.DataFrame): Macroeconomic data with columns:
            - date (INDEX) (pd.Timestamp): Date of macroeconomic data
            - all of the fields and indicators specified in wallet_metrics_config


    Returns:
        pd.DataFrame: Features indexed by wallet_address with columns for each market timing metric:
            {indicator}/lead_{n}_{direction}_{type}
            where:
            - indicator: The base metric (e.g., price_rsi_14, volume_sma_7)
            - n: The forward-looking period (e.g., 7, 14, 30 days)
            - direction: 'buy' or 'sell'
            - type: 'weighted' (by USD value) or 'mean'

            All wallet_addresses from the categorical index are included with zeros for missing data.
    """
    # Merge date-indexed macroeconomic indicators onto the coin_id-date keyed market indicators
    if macro_indicators_df is not None:
        market_indicators_data_df = market_indicators_data_df.merge(macro_indicators_df.reset_index(), on='date')

    # add timing offset features
    market_timing_df = calculate_offsets(market_indicators_data_df)
    market_timing_df,features_columns = calculate_relative_changes(market_timing_df)

    # flatten the wallet-coin-date transactions into wallet-indexed features
    wallet_timing_features_df = generate_all_timing_features(
        profits_df,
        market_timing_df,
        features_columns,
        wallets_config['features']['timing_metrics_min_transaction_size'],
    )

    return wallet_timing_features_df



# -----------------------------
# Component Functions
# -----------------------------

def calculate_offsets(
    market_indicators_data_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate offset values for specified columns in market timing dataframe.

    Args:
        market_indicators_data_df: DataFrame containing market timing data

    Returns:
        market_timing_df: DataFrame with added offset columns

    Raises:
        FeatureConfigError: If configuration is invalid or required columns are missing
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    market_timing_df = market_indicators_data_df.copy()

    # Get the offsets configuration
    try:
        offset_config = wallets_features_config['market_timing']['offsets']
    except KeyError as e:
        raise FeatureConfigError("Config key ['market_timing']['offsets'] was not found in " \
                                "wallets_features_config.") from e

    # Process each column and its offsets
    for column, column_config in offset_config.items():
        # Check if the column exists in the DataFrame
        if column not in market_timing_df.columns:
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
        for offset in offsets:
            if offset > 0:
                new_column = f"{column}_lead_{offset}"
            elif offset < 0:
                new_column = f"{column}_lag_{-offset}"
            else:
                raise ValueError(f"Invalid wallet_features_config offset param {offset} found. "
                                "Offsets must be non-zero integers.")


            try:
                market_timing_df[new_column] = market_timing_df.groupby('coin_id',observed=True)[column].shift(-offset)
            except Exception as e:
                raise FeatureConfigError(f"Error calculating offset for column '{column}' " \
                                        f"with lead {offset}: {str(e)}") from e

    return market_timing_df



def calculate_relative_changes(
    market_timing_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate relative changes between base columns and their corresponding offset columns.

    For each base column with offset columns (e.g., price_rsi_14 and price_rsi_14_lead_30),
    calculates the percentage change between them and stores it in a new column
    (e.g., price_rsi_14/lead_30). If retain_base_columns is False, drops the original
    base and offset columns after calculating the changes. All changes are winsorized
    using the configured coefficient.

    Args:
        market_timing_df: DataFrame containing market timing data with offset columns

    Returns:
        DataFrame with added relative change columns and optionally dropped base/offset columns
        features_columns: List of the columns to generate features for

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

    # Keep track of columns to be used in feature generation
    relative_change_columns = []
    retained_base_columns = []

    # Process each column and calculate relative changes
    for base_column, column_config in offset_config.items():
        # Check if the base column exists in the DataFrame
        if base_column not in result_df.columns:
            raise FeatureConfigError(f"Base column '{base_column}' not found in DataFrame")

        try:
            offsets = column_config['offsets']

            # If retain_base_columns is False, mark columns for dropping
            retain_base_columns = column_config.get('retain_base_columns', False)
            if retain_base_columns:
                retained_base_columns.append(base_column)
            else:
                columns_to_drop.add(base_column)

        except KeyError as e:
            raise FeatureConfigError(f"Invalid configuration for column '{base_column}': {str(e)}") from e

        # Calculate relative change for each offset
        for offset in offsets:

            # Identify column and confirm it exists
            if offset > 0:
                offset_str = f'lead_{offset}'
            else:
                offset_str = f'lag_{-offset}'

            offset_column = f'{base_column}_{offset_str}'
            if offset_column not in result_df.columns:
                raise FeatureConfigError(f"Offset column '{offset_column}' not found in DataFrame. " \
                                        "Run calculate_offsets() first.")

            # Create new column name for relative change
            change_column = f"{base_column}/{offset_str}"

            try:
                # Calculate percentage change
                # Formula: (offset_value - base_value) / base_value
                result_df[change_column] = (
                    (result_df[offset_column] - result_df[base_column]) /
                    result_df[base_column]
                )

                # Flip the sign if the offset is negative (compares present v past instead of future vs present)
                if offset < 0:
                    result_df[change_column] = result_df[change_column] * -1

                # Handle division by zero cases
                result_df[change_column] = result_df[change_column].replace([np.inf, -np.inf], np.nan)

                # Add column to winsorization list
                relative_change_columns.append(change_column)

            except Exception as e:
                raise FeatureConfigError(f"Error calculating relative change between '{base_column}' " \
                                        f"and '{offset_column}': {str(e)}") from e

    # Winsorize all change columns
    for column in relative_change_columns:
        result_df[column] = u.winsorize(result_df[column], winsor_coef)

    # Drop marked columns if any
    if columns_to_drop:
        result_df = result_df.drop(columns=list(columns_to_drop))

    # Consolidate columns to generate features for
    features_columns = retained_base_columns + relative_change_columns

    return result_df,features_columns


# def calculate_timing_features_for_column(df: pd.DataFrame, metric_column: str) -> pd.DataFrame:
#     """
#     Calculate buy_weighted feature for a single metric column.

#     FEATUREREMOVAL:
#      previously we computed all of these columns and only buy_weighted
#      proved to be predictive. the other features were removed as part of
#      ticket DDA-768:
#         {metric_column}/buy_weighted,
#         {metric_column}/buy_mean,
#         {metric_column}/sell_weighted
#         {metric_column}/sell_mean,



#     Params:
#     - df (DataFrame): input profits data.
#     - metric_column (str): column name to calculate weighted average for.

#     Returns:
#     - DataFrame with wallet_address index and {metric_column}/buy_weighted column.
#     """
#     # Filter to only buy transactions (positive net transfers)
#     buy_df = df[df['usd_net_transfers'] > 0].copy()

#     # Skip calculation if no buy transactions
#     if buy_df.empty:
#         return pd.DataFrame(columns=['wallet_address', f'{metric_column}/buy_weighted'])

#     # Calculate weighted values directly
#     buy_df['weighted_values'] = buy_df[metric_column] * buy_df['usd_net_transfers']

#     # Group by wallet_address
#     result = buy_df.groupby('wallet_address').agg(
#         sum_weighted_values=('weighted_values', 'sum'),
#         sum_weights=('usd_net_transfers', 'sum')
#     ).reset_index()

#     # Compute weighted average
#     result[f'{metric_column}/buy_weighted'] = result['sum_weighted_values'] / result['sum_weights']

#     # Keep only needed columns
#     return result[['wallet_address', f'{metric_column}/buy_weighted']].set_index('wallet_address')


# FEATUREREMOVAL only buy_weighted was predictive
def calculate_timing_features_for_column(df: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    """
    Calculate timing features (mean, weighted mean) for a single metric column in a single pass.
    Result columns: {metric_column}/buy_mean, {metric_column}/buy_weighted,
                    {metric_column}/sell_mean, {metric_column}/sell_weighted.
    """
    # Filter out rows with zero net transfers
    df = df[df['usd_net_transfers'] != 0].copy()

    # Label transaction_side = 'buy' or 'sell'
    df['transaction_side'] = np.where(df['usd_net_transfers'] > 0, 'buy', 'sell')

    # Precompute abs transfers & weighted values
    df['abs_net_transfers'] = df['usd_net_transfers'].abs()
    df['weighted_values'] = df[metric_column] * df['abs_net_transfers']

    # Group once on [wallet_address, transaction_side]
    grouped = df.groupby(['wallet_address', 'transaction_side'], observed=True).agg(
        mean_val=(metric_column, 'mean'),
        sum_weighted_values=('weighted_values', 'sum'),
        sum_weights=('abs_net_transfers', 'sum'),
    ).reset_index()

    # Compute weighted average
    grouped['weighted_val'] = grouped['sum_weighted_values'] / grouped['sum_weights']

    # Pivot to separate 'buy' and 'sell' columns
    pivoted = grouped.pivot(index='wallet_address', columns='transaction_side')

    # We only want mean and weighted columns
    # pivoted["mean_val"] => buy / sell means
    # pivoted["weighted_val"] => buy / sell weighted
    pivoted_mean = pivoted['mean_val'].copy()
    pivoted_weighted = pivoted['weighted_val'].copy()

    # Rename columns
    pivoted_mean.columns = [f'{metric_column}/{side}_mean' for side in pivoted_mean.columns]
    pivoted_weighted.columns = [f'{metric_column}/{side}_weighted' for side in pivoted_weighted.columns]

    # Combine back
    final_df = pd.concat([pivoted_mean, pivoted_weighted], axis=1)

    return final_df



def generate_all_timing_features(
    profits_df,
    market_timing_df,
    features_columns,
    min_transaction_size=0
):
    """
    Generate timing features for multiple market metric columns.

    Args:
        profits_df (pd.DataFrame): DataFrame with columns [coin_id, date, wallet_address, usd_net_transfers]
        market_timing_df (pd.DataFrame): DataFrame with market timing metrics indexed by (coin_id, date)
        features_columns (list): List of column names from market_timing_df to analyze
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
        market_timing_df[features_columns + ['coin_id', 'date']],
        on=['coin_id', 'date'],
        how='left'
    )

    # Calculate features for each column
    all_features = []
    for col in features_columns:
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



# -----------------------------
# Helper Functions
# -----------------------------

class FeatureConfigError(Exception):
    """Custom exception for feature configuration errors."""
    pass
