"""
Calculates metrics aggregated at the wallet level
"""
import logging
from typing import List,Tuple
from pathlib import Path
import pandas as pd
import numpy as np


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


# -----------------------------
# Main Interface
# -----------------------------
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
            {indicator}/lead_{n}_{direction}_{type}
            where:
            - indicator: The base metric (e.g., price_rsi_14, volume_sma_7)
            - n: The forward-looking period (e.g., 7, 14, 30 days)
            - direction: 'buy' or 'sell'
            - type: 'weighted' (by USD value) or 'mean'

            All wallet_addresses from the categorical index are included with zeros for missing data.
    """
    # identify indicator columns based on the config file
    indicator_columns = identify_indicator_columns(wallets_metrics_config['time_series']['market_data'])

    # add timing offset features
    market_timing_df = calculate_offsets(market_indicators_data_df, indicator_columns)
    market_timing_df, relative_change_columns = calculate_relative_changes(market_timing_df, indicator_columns)


    # Flatten the wallet-coin-date transactions into wallet-indexed features
    timing_profits_df, factorization_info = prepare_timing_data(profits_df,
                                                                market_timing_df,
                                                                relative_change_columns)
    wallet_timing_features_df = calculate_wallet_timing_features(timing_profits_df,
                                                                 relative_change_columns,
                                                                 factorization_info)


    return wallet_timing_features_df



# -----------------------------
# Component Functions
# -----------------------------


def identify_indicator_columns(config: dict) -> list:
    """
    Generates column names from nested indicator config.

    Params:
    - config (dict): Nested config with metrics, indicators and parameters

    Returns:
    - cols (list): List of generated column names
    """
    cols = []
    for metric, metric_config in config.items():
        for indicator, ind_config in metric_config['indicators'].items():
            for _, values in ind_config['parameters'].items():
                for value in values:
                    cols.append(f"{metric}_{indicator}_{value}")

    return cols



def calculate_offsets(
    market_indicators_data_df: pd.DataFrame,
    indicator_columns: List
) -> pd.DataFrame:
    """
    Calculate offset values for specified columns in market timing dataframe.

    Args:
        market_indicators_data_df: DataFrame containing market timing data
        indicator_columns: List of the names of the indicator columns

    Returns:
        market_timing_df: DataFrame with added offset columns
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    market_timing_df = market_indicators_data_df.copy()

    # Get the offsets configuration
    offsets = wallets_config['features']['market_timing_offsets']

    # Process each column and its offsets
    for column in indicator_columns:

        # Calculate offset for each specified value
        for offset in offsets:
            if offset > 0:
                new_column = f"{column}_lead_{offset}"
            else:
                new_column = f"{column}_lag_{-offset}"

            market_timing_df[new_column] = market_timing_df.groupby('coin_id',observed=True)[column].shift(-offset)

    return market_timing_df



def calculate_relative_changes(
    market_timing_df: pd.DataFrame,
    indicator_columns: List
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
        indicator_columns: List of the names of the indicator columns

    Returns:
        DataFrame with added relative change columns and optionally dropped base/offset columns
        relative_change_columns: List of the relative change columns created
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    result_df = market_timing_df.copy()
    offsets = wallets_config['features']['market_timing_offsets']
    offset_winsorization = wallets_config['features']['market_timing_offset_winsorization']

    # Keep track of columns to drop
    columns_to_drop = set()
    # Keep track of columns to winsorize
    relative_change_columns = []

    # Process each column and calculate relative changes
    for base_column in indicator_columns:

        # Calculate relative change for each offset
        for offset in offsets:

            # Define column names
            if offset > 0:
                offset_str = f'lead_{offset}'
            else:
                offset_str = f'lag_{-offset}'

            offset_column = f'{base_column}_{offset_str}'

            # Create new column name for relative change
            change_column = f"{base_column}/{offset_str}"

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

    # Winsorize all change columns
    for column in relative_change_columns:
        result_df[column] = u.winsorize(result_df[column], offset_winsorization)

    # Drop marked columns if any
    if columns_to_drop:
        result_df = result_df.drop(columns=list(columns_to_drop))

    return result_df,relative_change_columns




def prepare_timing_data(profits_df: pd.DataFrame,
                       market_timing_df: pd.DataFrame,
                       relative_change_columns: list
                   ) -> Tuple[pd.DataFrame, dict]:
    """
    Prepare and merge profits data with market timing metrics.

    Params:
    - profits_df (DataFrame): Raw profits data
    - market_timing_df (DataFrame): Market timing metrics data
    - relative_change_columns (list): Columns to include from market timing data

    Returns:
    - timing_profits_df (DataFrame): Merged and preprocessed timing data
    - factorization_info (dict): Pre-computed factorization data
    """
    # Filter out transactions below materiality threshold
    filtered_profits = profits_df[
        abs(profits_df['usd_net_transfers']) >= wallets_config['data_cleaning']['usd_materiality']
    ].copy()

    # Perform the merge once
    timing_profits_df = filtered_profits.merge(
        market_timing_df[relative_change_columns + ['coin_id', 'date']],
        on=['coin_id', 'date'],
        how='left'
    )

    # Filter out rows with zero net transfers
    timing_profits_df = timing_profits_df[timing_profits_df['usd_net_transfers'] != 0].copy()

    # Label transaction_side = 'buy' or 'sell'
    timing_profits_df['transaction_side'] = np.where(timing_profits_df['usd_net_transfers'] > 0, 'buy', 'sell')

    # Pre-compute factorization
    # see the below Optimization Notes for an explantion of this sequence
    wallet_codes, wallet_uniques = pd.factorize(timing_profits_df['wallet_address'])
    side_codes, side_uniques = pd.factorize(timing_profits_df['transaction_side'])
    n_sides = len(side_uniques)
    combined_codes = wallet_codes * n_sides + side_codes

    # Precompute abs transfers
    timing_profits_df['abs_net_transfers'] = timing_profits_df['usd_net_transfers'].abs()

    factorization_info = {
        'wallet_uniques': wallet_uniques,
        'side_uniques': side_uniques,
        'combined_codes': combined_codes,
        'n_sides': n_sides
    }

    return timing_profits_df, factorization_info


def calculate_wallet_timing_features(timing_profits_df: pd.DataFrame,
                                   relative_change_columns: list,
                                   factorization_info: dict) -> pd.DataFrame:
    """
    Generate timing features for each wallet from prepared timing data.

    Params:
    - timing_profits_df (DataFrame): Preprocessed timing data
    - relative_change_columns (list): Columns to generate features for
    - factorization_info (dict): Pre-computed factorization data

    Returns:
    - result (DataFrame): Wallet-level timing features
    """
    # Use pre-computed wallet addresses from factorization
    wallet_uniques = factorization_info['wallet_uniques']

    # Calculate features for each column
    all_features = []
    for col in relative_change_columns:
        logger.debug("Generating timing performance features for %s...", col)
        col_features = calculate_timing_features_for_column(
            timing_profits_df,
            col,
            factorization_info
        )
        all_features.append(col_features)

    # Combine all feature sets
    if all_features:
        result = pd.concat(all_features, axis=1)
    else:
        result = pd.DataFrame(
            index=pd.Index(wallet_uniques, name='wallet_address')
        )

    # Ensure all wallet addresses are included and fill NaNs
    result = pd.DataFrame(
        result,
        index=pd.Index(wallet_uniques, name='wallet_address')
    ).fillna(0)

    return result


# pylint:disable=pointless-string-statement
"""
Optimization Notes: Integer-Based Aggregation vs GroupBy
------------------------------------------------------
The below code uses pd.factorization along with np.bincount to dramatically
improve performance over a standard groupby.agg() sequence.

Original Approach:

    grouped = df.groupby(['wallet_address', 'transaction_side'], observed=True).agg(
        mean_val=(metric_column, 'mean'),
        sum_weighted_values=('weighted_values', 'sum'),
        sum_weights=('abs_net_transfers', 'sum'),
    ).reset_index()


The initial implementation used pandas groupby() to calculate metrics for each
wallet's buy/sell transactions. While intuitive, groupby operations have
significant overhead as they:
1. Create separate groups in memory
2. Apply operations to each group independently
3. Recombine results back into a DataFrame



New Approach: Integer-Based Aggregation
-------------------------------------
The optimized version uses factorization to convert the problem into pure
numerical operations:

1. Factorization: Convert categorical data to dense integer arrays
   Example:
   wallet_addresses = ['wallet_a', 'wallet_a', 'wallet_b']
   Becomes:
   codes = [0, 0, 1]  # Dense integers
   uniques = ['wallet_a', 'wallet_b']  # Lookup table

2. Combined Keys: Create single integer key for wallet-side combinations
   With 2 sides (buy/sell):
   wallet 0, side 1 = 0 * 2 + 1 = 1
   wallet 1, side 0 = 1 * 2 + 0 = 2

3. Fast Aggregation: Use numpy's bincount for efficient integer array operations
   - Sum values: bincount(combined_codes, weights=values)
   - Count occurrences: bincount(combined_codes)
   - Calculate means: divide sums by counts

Performance Benefits:
- Memory Efficiency: Works with simple integer arrays instead of group objects
- Vectorized Operations: Uses optimized numpy functions
- Single Pass: Scans data once for all aggregations
- Pre-computation: Factorizes once at start, avoiding repeated string comparisons

This approach shows significant performance improvements over groupby when
dealing with large datasets (~200M rows).
"""


def calculate_timing_features_for_column(df: pd.DataFrame,
                                       metric_column: str,
                                       factorization_info: dict) -> pd.DataFrame:
    """
    Calculate timing features using pre-computed factorization.
    Result columns: {metric_column}/buy_mean, {metric_column}/buy_weighted,
                    {metric_column}/sell_mean, {metric_column}/sell_weighted.
    """
    # Unpack pre-computed factorization data
    wallet_uniques = factorization_info['wallet_uniques']
    side_uniques = factorization_info['side_uniques']
    combined_codes = factorization_info['combined_codes']
    n_sides = factorization_info['n_sides']

    # Calculate aggregations using bincount
    sum_values = np.bincount(combined_codes, weights=df[metric_column].to_numpy(),
                           minlength=len(wallet_uniques) * n_sides)
    counts = np.bincount(combined_codes, minlength=len(wallet_uniques) * n_sides)
    weighted_values = df[metric_column] * df['abs_net_transfers']
    sum_weighted = np.bincount(combined_codes, weights=weighted_values.to_numpy(),
                             minlength=len(wallet_uniques) * n_sides)
    sum_weights = np.bincount(combined_codes, weights=df['abs_net_transfers'].to_numpy(),
                            minlength=len(wallet_uniques) * n_sides)

    # Calculate means with proper NaN handling
    mean_matrix = np.divide(sum_values, counts,
                          out=np.full_like(sum_values, np.nan, dtype=float),
                          where=counts > 0).reshape(len(wallet_uniques), n_sides)
    weighted_matrix = np.divide(sum_weighted, sum_weights,
                              out=np.full_like(sum_weighted, np.nan, dtype=float),
                              where=sum_weights > 0).reshape(len(wallet_uniques), n_sides)

    # Create final DataFrame
    mean_df = pd.DataFrame(
        mean_matrix,
        index=wallet_uniques,
        columns=[f'{metric_column}/{side}_mean' for side in side_uniques]
    )
    weighted_df = pd.DataFrame(
        weighted_matrix,
        index=wallet_uniques,
        columns=[f'{metric_column}/{side}_weighted' for side in side_uniques]
    )

    return pd.concat([mean_df, weighted_df], axis=1)
