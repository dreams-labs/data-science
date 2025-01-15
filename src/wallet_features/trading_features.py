"""
Calculates metrics related to trading performance

Intended function sequence:

# Base feature calculation
profits_df = wtf.add_cash_flow_transfers_logic(profits_df)
trading_features = wtf.calculate_wallet_trading_features(profits_df)
"""
import logging
from datetime import datetime,timedelta
import pandas as pd
import numpy as np

# Local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()


# -----------------------------------
# Main Interface Function
# -----------------------------------

@u.timing_decorator
def calculate_wallet_trading_features(
    profits_df: pd.DataFrame,
    period_start_date: str,
    period_end_date: str,
    include_twb_metrics: bool = True,
    include_btc_metrics: bool = False
) -> pd.DataFrame:
    """
    Calculates comprehensive crypto trading metrics for each wallet.

    Params:
    - profits_df (DataFrame): Daily profits data
    - period_start_date (str): Period start in 'YYYY-MM-DD' format
    - period_end_date (str): Period end in 'YYYY-MM-DD' format
    - include_twb_metrics (bool): whether to calculate time weighted balance metrics

    Required columns: wallet_address, coin_id, date, usd_balance,
                    usd_net_transfers, is_imputed

    Returns:
    - trading_features_df (DataFrame): Trading metrics keyed on wallet_address with columns:
        - total_crypto_buys: Sum of positive balance changes
        - total_crypto_sells: Sum of negative balance changes
        - net_crypto_investment: Net sum of all balance changes
        - crypto_net_gain: Final unrealized gain
        - transaction_days: Number of days with activity
        - unique_coins_traded: Number of unique coins
        - total_volume: Sum of absolute balance changes
        - average_transaction: Mean absolute balance change
        - activity_density: Transaction days / period duration
        - volume_vs_twb_ratio: Volume relative to time-weighted balance
    """
    # Validate profits_df
    profits_df = ensure_index(profits_df, period_start_date, period_end_date)

    # Calculate configured metrics
    # ----------------------------
    # Add crypto balance/transfers/gain helper columns
    profits_df = calculate_crypto_balance_columns(profits_df, period_start_date)
    profits_df = profits_df.reset_index()

    # Calculate net_gain and max_investment columns
    gain_and_investment_df = calculate_gain_and_investment_columns(profits_df)

    # Calculated metrics that ignore imputed transactions
    observed_activity_df = calculate_observed_activity_columns(profits_df,period_start_date,period_end_date)

    # Merge together
    trading_features_df = gain_and_investment_df.join(observed_activity_df)

    # Add twb if configured to do so
    if include_twb_metrics:

        # Calculate time weighted balance using the cost basis
        time_weighted_df = aggregate_time_weighted_balance(profits_df)

        # Join all full metric dataframes
        trading_features_df = trading_features_df.join(time_weighted_df)

        # Calculate volume to time-weighted balance ratio
        trading_features_df['volume_vs_twb_ratio'] = np.where(
            trading_features_df['time_weighted_balance'] > 0,
            trading_features_df['total_volume'] / trading_features_df['time_weighted_balance'],
            0
        )

    # Fill missing values and handle edge cases
    trading_features_df = trading_features_df.fillna(0).replace(-0, 0)

    return trading_features_df


# -----------------------------------
# Support Functions
# -----------------------------------


def calculate_crypto_balance_columns(profits_df: pd.DataFrame,
                                   period_start_date: str
                                   ) -> pd.DataFrame:
    """
    Adds crypto balance change columns using multiindex operations.

    Params:
    - profits_df (DataFrame): Daily profits data with multiindex (coin_id, wallet_address, date)
    - period_start_date (str): Period start in 'YYYY-MM-DD' format

    Returns:
    - adj_profits_df (DataFrame): Input df with crypto balance columns added
    """
    profits_df['crypto_balance_change'] = profits_df['usd_net_transfers']
    profits_df = buy_crypto_start_balance(profits_df, period_start_date)

    # Use index-aware cumsum() since data is already sorted by (coin_id, wallet_address, date)
    profits_df['crypto_cumulative_transfers'] = (profits_df
                                                 .groupby(level=['coin_id', 'wallet_address'],observed=True)
                                                 ['crypto_balance_change'].cumsum())

    profits_df['crypto_cumulative_net_gain'] = (profits_df['usd_balance']
                                                - profits_df['crypto_cumulative_transfers'])

    return profits_df



def buy_crypto_start_balance(df: pd.DataFrame, period_start_date: str) -> pd.DataFrame:
    """
    Sets start date crypto balance change using multiindex operations.

    Params:
    - df (DataFrame): Input df with multiindex (coin_id, wallet_address, date)
    - period_start_date (str): Period start in 'YYYY-MM-DD'

    Returns:
    - df (DataFrame): DataFrame with adjusted crypto_balance_change
    """
    period_start_date = datetime.strptime(period_start_date, '%Y-%m-%d')
    starting_balance_date = period_start_date - timedelta(days=1)

    # Use IndexSlice to efficiently select the starting balance date
    idx = pd.IndexSlice
    start_slice = idx[:, :, starting_balance_date]
    df.loc[start_slice, 'crypto_balance_change'] = df.loc[start_slice, 'usd_balance']

    return df



def calculate_gain_and_investment_columns(profits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates net gain and max investment for each wallet.

    Params:
    - profits_df (DataFrame): Profits data with crypto_cumulative_net_gain
                            and crypto_cumulative_transfers columns

    Returns:
    - gain_and_investment_df (DataFrame): Metrics keyed on wallet_address with columns:
        - max_investment: Maximum cumulative transfers (floored at 0)
        - crypto_net_gain: Final unrealized gain per wallet
    """
    # Retrieve the cumulative net gain at the end of the period
    last_rows = profits_df.drop_duplicates(subset=['wallet_address', 'coin_id'], keep='last')
    gains_df = last_rows.groupby('wallet_address', as_index=False).agg(
        crypto_net_gain=('crypto_cumulative_net_gain', 'sum')
    ).set_index('wallet_address')

    # Calculate buy/sell metrics from full dataset, including opening balances
    profits_df['positive_changes'] = np.where(profits_df['crypto_balance_change'] > 0,
                                            profits_df['crypto_balance_change'], 0)
    profits_df['negative_changes'] = np.where(profits_df['crypto_balance_change'] < 0,
                                            -profits_df['crypto_balance_change'], 0)

    # Group and aggregate for max investment
    investments_df = profits_df.groupby('wallet_address').agg(
        max_investment=('crypto_cumulative_transfers', 'max'),
        total_crypto_buys=('positive_changes', 'sum'),
        total_crypto_sells=('negative_changes', 'sum'),
        net_crypto_investment=('crypto_balance_change', 'sum'),
    )

    # Set floor of 0 on max_investment to match business constraint
    investments_df['max_investment'] = investments_df['max_investment'].clip(lower=0)

    # Combine metrics
    gain_and_investment_df = investments_df.join([gains_df])

    return gain_and_investment_df



def calculate_observed_activity_columns(profits_df: pd.DataFrame,
                                        period_start_date: str,
                                        period_end_date: str) -> pd.DataFrame:
    """
    Calculates metrics based on actual trading activity, excluding imputed rows.

    Params:
    - profits_df (DataFrame): Profits data with crypto_balance_change column
    - period_start_date (str): Period start in 'YYYY-MM-DD' format
    - period_end_date (str): Period end in 'YYYY-MM-DD' format

    Returns:
    - activity_df (DataFrame): Activity metrics keyed on wallet_address with columns:
        - unique_coins_traded: Number of unique coins traded
        - total_volume: Sum of absolute balance changes
        - total_crypto_buys: Sum of positive balance changes
        - total_crypto_sells: Sum of negative balance changes
        - net_crypto_investment: Net sum of all balance changes
        - transaction_days: Number of days with activity
        - average_transaction: Mean absolute balance change
        - activity_density: Transaction days / period duration
    """
    # Filter to actual transaction activity (excluding imputed rows)
    observed_profits_df = profits_df.loc[~profits_df['is_imputed'],
                                ['wallet_address', 'date', 'coin_id', 'crypto_balance_change']]

    # Precompute absolute balance changes
    observed_profits_df['abs_balance_change'] = observed_profits_df['crypto_balance_change'].abs()

    # Calculate observed activity metrics
    observed_activity_df = observed_profits_df.groupby('wallet_address', as_index=False).agg(
        unique_coins_traded=('coin_id', 'nunique'),
        total_volume=('abs_balance_change', 'sum'),
        transaction_days=('date', 'nunique'),
        average_transaction=('abs_balance_change', 'mean'),
    ).set_index('wallet_address')

    # Add activity density based on period duration
    period_duration = (datetime.strptime(period_end_date, '%Y-%m-%d') -
                        datetime.strptime(period_start_date, '%Y-%m-%d')).days + 1
    observed_activity_df['activity_density'] = observed_activity_df['transaction_days'] / period_duration

    return observed_activity_df



@u.timing_decorator
def aggregate_time_weighted_balance(profits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates time weighted average balance for each wallet.
    Includes both total TWB and active-only TWB (periods with non-zero balance).

    Params:
    - profits_df (DataFrame): Daily profits data with balances and dates

    Returns:
    - trading_features_df (DataFrame): Time weighted metrics by wallet
    """
    active_period_threshold = wallets_config['features']['timing_metrics_min_transaction_size']

    # Calculate cost basis for each wallet-coin pair and merge into main dataframe
    cost_basis_df = get_cost_basis_df(profits_df)
    profits_df = profits_df.merge(
        cost_basis_df,
        on=['wallet_address', 'coin_id', 'date'],
        how='left'
    )

    # Calculate days held for each balance level
    logger.debug('a')
    profits_df['next_date'] = (profits_df.groupby(['wallet_address', 'coin_id'],observed=True)
                               ['date'].shift(-1))
    profits_df['days_held'] = (
        profits_df['next_date'] - profits_df['date']
    ).dt.total_seconds() / (24 * 60 * 60)

    logger.debug('b')
    # There is no hold time for the closing balance on the final day
    last_mask = profits_df['next_date'].isna()
    profits_df.loc[last_mask, 'days_held'] = 0

    logger.debug('c')
    # Calculate weighted cost for both total and active periods
    profits_df['weighted_cost'] = profits_df['crypto_cost_basis'] * profits_df['days_held']
    profits_df['active_weighted_cost'] = np.where(
        profits_df['crypto_cost_basis'] > active_period_threshold,
        profits_df['weighted_cost'],
        0
    )
    profits_df['active_days'] = np.where(
        profits_df['crypto_cost_basis'] > active_period_threshold,
        profits_df['days_held'],
        0
    )

    logger.debug('d')
    # Group by wallet-coin and calculate both versions
    coin_level_metrics = profits_df.groupby(['wallet_address', 'coin_id'],observed=True).agg({
        'days_held': 'sum',
        'active_days': 'sum',
        'weighted_cost': 'sum',
        'active_weighted_cost': 'sum'
    })

    logger.debug('e')
    # Calculate TWB for both versions
    epsilon = 1e-10
    coin_level_metrics['coin_twb'] = (
        coin_level_metrics['weighted_cost'] /
        (coin_level_metrics['days_held'] + epsilon)
    )
    coin_level_metrics['active_coin_twb'] = np.where(
        coin_level_metrics['active_days'] > 0,
        coin_level_metrics['active_weighted_cost'] / coin_level_metrics['active_days'],
        0
    )

    logger.debug('f')
    # Sum to wallet level
    wallet_metrics = coin_level_metrics.groupby('wallet_address').agg({
        'coin_twb': 'sum',
        'active_coin_twb': 'sum'
    })

    return pd.DataFrame({
        'time_weighted_balance': wallet_metrics['coin_twb'],
        'active_time_weighted_balance': wallet_metrics['active_coin_twb']
    })



def get_cost_basis_df(profits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimized cost basis calculation focusing on maintaining original loop efficiency.

    Params:
    - profits_df (DataFrame): profits data with required columns

    Returns:
    - DataFrame with cost basis for each wallet-coin-date
    """
    df = profits_df.copy()

    # Calculate opening balance before transfers
    df['opening_balance'] = df['usd_balance'] - df['usd_net_transfers']

    # Calculate % sold when transfers are negative
    df['pct_sold'] = np.where(
        df['usd_net_transfers'] < 0,
        -df['usd_net_transfers'] / df['opening_balance'],
        0
    ).astype('float64')

    # Track new cost basis from buys
    df['cost_basis_bought'] = np.where(
        df['crypto_balance_change'] > 0,
        df['crypto_balance_change'],
        0
    ).astype('float64')

    # Pre-sort the DataFrame to simplify iteration
    df = df.sort_values(['wallet_address', 'coin_id', 'date']).reset_index(drop=True)

    # Initialize an array for the cost basis
    crypto_cost_basis = np.zeros(len(df))

    # Efficient iteration: Avoid grouping by working directly on pre-sorted indices
    current_wallet_coin = None
    cumulative_cost_basis = 0

    for i in range(len(df)):
        wallet_coin = (df.at[i, 'wallet_address'], df.at[i, 'coin_id'])

        if wallet_coin != current_wallet_coin:
            # New wallet-coin group
            cumulative_cost_basis = 0
            current_wallet_coin = wallet_coin

        # Update cumulative cost basis
        cumulative_cost_basis = (
            cumulative_cost_basis * (1 - df.at[i, 'pct_sold']) +
            df.at[i, 'cost_basis_bought']
        )
        crypto_cost_basis[i] = cumulative_cost_basis

    # Assign the calculated values back to the DataFrame
    df['crypto_cost_basis'] = crypto_cost_basis

    result_df = df[['wallet_address', 'coin_id', 'date', 'crypto_cost_basis']]

    # Validation check to ensure the result DataFrame has the same number of rows as the input
    if len(profits_df) != len(result_df):
        raise ValueError('Record count mismatch')

    return result_df



# -----------------------------------
#          Utility Functions
# -----------------------------------

def ensure_index(profits_df: pd.DataFrame,
                       period_start_date: str,
                       period_end_date: str
) -> pd.DataFrame:
    """
    Params:
    - profits_df (DataFrame): Input profits data
    - period_start_date (str): Period start date
    - period_end_date (str): Period end date

    Returns:
    - DataFrame: Validated and properly indexed/sorted profits_df
    """
    required_cols = ['coin_id', 'wallet_address', 'date']

    # Check existence of required columns
    missing_cols = [col for col in required_cols
                   if col not in profits_df.columns
                   and col not in profits_df.index.names]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Convert to MultiIndex if needed
    if not isinstance(profits_df.index, pd.MultiIndex):
        index_cols = [col for col in required_cols if col in profits_df.columns]
        profits_df = profits_df.set_index(index_cols)

    # Ensure correct index names
    if profits_df.index.names != required_cols:
        profits_df = profits_df.reorder_levels(required_cols)

    # Sort if needed
    if not profits_df.index.is_monotonic_increasing:
        profits_df = profits_df.sort_index()

    # Assert period
    u.assert_period(profits_df, period_start_date, period_end_date)

    return profits_df