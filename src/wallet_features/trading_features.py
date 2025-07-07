"""
Calculates metrics related to trading activity
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
#       Main Interface Function
# -----------------------------------

@u.timing_decorator
def calculate_wallet_trading_features(
    profits_df: pd.DataFrame,
    period_start_date: str,
    period_end_date: str,
    include_twb_metrics: bool = False,
    include_twr_metrics: bool = False,
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
        - crypto_inflows: Sum of positive balance changes
        - crypto_outflows: Sum of negative balance changes
        - crypto_net_flows: Net sum of all balance changes
        - crypto_net_gain: Final unrealized gain
        - transaction_days: Number of days with activity
        - unique_coins_traded: Number of unique coins
        - total_volume: Sum of absolute balance changes
        - average_transaction: Mean absolute balance change
        - activity_density: Transaction days / period duration
        - volume_vs_twb_ratio: Volume relative to time-weighted balance
    """
    # Validate profits_df
    profits_df = profits_df.copy()
    profits_df = u.ensure_index(profits_df)
    u.assert_period(profits_df,period_start_date, period_end_date)

    # 1. Calculate Base Metrics
    # Add crypto balance/transfers/gain helper columns
    profits_df = calculate_crypto_balance_columns(profits_df,
                                                  period_start_date, period_end_date)

    # Calculate net_gain and max_investment columns
    gain_and_investment_df = calculate_gain_and_investment_columns(profits_df)

    # Calculated metrics that ignore imputed transactions
    observed_activity_df = calculate_observed_activity_columns(profits_df,
                                                               period_start_date, period_end_date)

    # Make trading_features_df via merge
    trading_features_df = gain_and_investment_df.join(observed_activity_df)


    # 2. Calculate Time Weighted Return if configured to do so
    if include_twr_metrics:
        twr_df = calculate_wallet_time_weighted_returns(profits_df)
        trading_features_df = trading_features_df.join(twr_df)


    # 3. Calculate Time Weighted Balance if configured to do so
    if include_twb_metrics:
        twb_df = aggregate_time_weighted_balance(profits_df)
        profits_df = u.ensure_index(profits_df)
        trading_features_df = trading_features_df.join(twb_df)

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
#    Base Metrics Helper Functions
# -----------------------------------

def calculate_crypto_balance_columns(profits_df: pd.DataFrame,
                                   period_start_date: str,
                                   period_end_date: str,
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
    profits_df = sell_crypto_end_balance(profits_df, period_end_date)

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


def sell_crypto_end_balance(df: pd.DataFrame, period_end_date: str) -> pd.DataFrame:
    """
    Adjusts crypto_balance_change by subtracting usd_balance and sets usd_balance to 0
    for the ending balance date.

    Params:
    - df (DataFrame): Input df with multiindex (coin_id, wallet_address, date).
    - period_end_date (str): Period end in 'YYYY-MM-DD'.

    Returns:
    - df (DataFrame): DataFrame with adjusted crypto_balance_change and usd_balance.
    """
    ending_balance_date = datetime.strptime(period_end_date, '%Y-%m-%d')

    # Use IndexSlice to efficiently select the ending balance date
    idx = pd.IndexSlice
    end_slice = idx[:, :, ending_balance_date]

    # Adjust crypto_balance_change by subtracting usd_balance
    df.loc[end_slice, 'crypto_balance_change'] -= df.loc[end_slice, 'usd_balance']

    # Set usd_balance to 0
    df.loc[end_slice, 'usd_balance'] = 0

    return df


def calculate_gain_and_investment_columns(profits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates net gain and max investment using multiindex operations.

    Params:
    - profits_df (DataFrame): Profits data with multiindex (coin_id, wallet_address, date)
                            and required columns

    Returns:
    - gain_and_investment_df (DataFrame): Wallet metrics with columns:
        - max_investment: Maximum cumulative transfers (floored at 0)
        - crypto_net_gain: Final unrealized gain per wallet
        - crypto_inflows: Sum of positive balance changes
        - crypto_outflows: Sum of negative balance changes
        - crypto_net_flows: Net balance changes
    """
    # Get last entry per wallet-coin using index levels
    last_rows = profits_df.groupby(level=['coin_id', 'wallet_address'], observed=True).last()

    # Sum gains across coins for each wallet
    gains_df = last_rows.groupby(level='wallet_address', observed=True).agg(
        crypto_net_gain=('crypto_cumulative_net_gain', 'sum')
    )

    # Add buy/sell columns
    balance_changes = profits_df['crypto_balance_change']
    profits_df['positive_changes'] = balance_changes.clip(lower=0)
    profits_df['negative_changes'] = (-balance_changes).clip(lower=0)

    # Calculate all investment metrics in one groupby
    investments_df = profits_df.groupby(level='wallet_address', observed=True).agg(
        max_investment=('crypto_cumulative_transfers', 'max'),
        crypto_inflows=('positive_changes', 'sum'),
        crypto_outflows=('negative_changes', 'sum'),
        crypto_net_flows=('crypto_balance_change', 'sum')
    )
    investments_df['max_investment'] = investments_df['max_investment'].clip(lower=0)

    # Combine metrics - both DataFrames are indexed on wallet_address
    gain_and_investment_df = investments_df.join(gains_df)

    return gain_and_investment_df


# pylint:disable=unused-argument  # params used for # FeatureRemoval features
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
    - activity_df (DataFrame): Activity metrics keyed on wallet_address
    """
    # Filter and add absolute changes
    observed_profits_df = profits_df.loc[~profits_df['is_imputed']].copy()
    observed_profits_df['abs_balance_change'] = observed_profits_df['usd_net_transfers'].abs()

    # Add buy/sell columns
    balance_changes = profits_df['usd_net_transfers']
    profits_df['positive_changes'] = balance_changes.clip(lower=0)
    profits_df['negative_changes'] = (-balance_changes).clip(lower=0)

    # Combine metrics in a single groupby where possible
    metrics_df = observed_profits_df.groupby(level='wallet_address', observed=True).agg(
        total_volume=('abs_balance_change', 'sum'),
        average_transaction=('abs_balance_change', 'mean'),
        crypto_cash_buys=('positive_changes', 'sum'),
        crypto_cash_sells=('negative_changes', 'sum'),
        crypto_net_cash_flows=('crypto_balance_change', 'sum'),
    )
    observed_activity_df = metrics_df

    # Extract index levels once
    index_frame = observed_profits_df.index.to_frame(index=False)

    # Calculate unique counts from index_frame in single operation
    unique_counts = index_frame.groupby('wallet_address').agg(
        unique_coins_traded=('coin_id', 'nunique'),
        # FeatureRemoval due to no predictiveness
        # transaction_days=('date', 'nunique')
    )

    # # Combine metrics
    observed_activity_df = observed_activity_df.join(unique_counts)

    # FeatureRemoval due to no predictiveness
    # # Add activity density
    # period_duration = (datetime.strptime(period_end_date, '%Y-%m-%d') -
    #                   datetime.strptime(period_start_date, '%Y-%m-%d')).days + 1
    # observed_activity_df['activity_density'] = observed_activity_df['transaction_days'] / period_duration

    return observed_activity_df



# --------------------------------------------
#    Time Weighted Balance Helper Functions
# --------------------------------------------

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
    active_period_threshold = wallets_config['features']['usd_materiality']

    # Calculate cost basis for each wallet-coin pair and merge into main dataframe
    cost_basis_df = get_cost_basis_df(profits_df)

    profits_df.reset_index(inplace=True)

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


@u.timing_decorator
def get_cost_basis_df(profits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimized cost basis calculation focusing on maintaining original loop efficiency.

    Params:
    - profits_df (DataFrame): profits data with required columns

    Returns:
    - DataFrame with cost basis for each wallet-coin-date
    """
    # Confirm index is sorted
    if not profits_df.index.is_monotonic_increasing:
        raise ValueError("profits_df index should be sorted")
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

    # Initialize an array for the cost basis
    crypto_cost_basis = np.zeros(len(df))

    # Efficient iteration: Avoid grouping by working directly on pre-sorted indices
    current_wallet_coin = None
    cumulative_cost_basis = 0

    # TO DO: convert rest of function to use index calculations
    df.reset_index(inplace=True)
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




# --------------------------------------------
#    Time Weighted Returns Helper Functions
# --------------------------------------------

@u.timing_decorator
def calculate_wallet_time_weighted_returns(profits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates time-weighted return metrics at wallet level.

    Params:
    - twr_df (DataFrame): Contains wallet_address, coin_id, and metrics

    Returns:
    - wallet_stats (DataFrame): Wallet-level statistics
    """
    # Calculate wallet-coin level time weighted returns
    wc_twr_df = calculate_wallet_coin_time_weighted_returns(profits_df)

    metrics = ['days_held', 'time_weighted_return', 'annualized_twr']
    agg_funcs = ['mean', 'min', 'max', 'median', 'std']

    # Vectorized aggregation
    wallet_twr_df = wc_twr_df.groupby('wallet_address')[metrics].agg(agg_funcs)

    # Flatten column names
    wallet_twr_df.columns = [f'{metric}/{func}'
                          for metric in metrics
                          for func in agg_funcs]

    return wallet_twr_df


def calculate_wallet_coin_time_weighted_returns(profits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates time-weighted returns (TWR) for coin-wallet pairs using actual holding period
    in days.

    This could be modified to calculate wallet level TWR if a complete cash flows and balance
    series was made available. This would involve imputing balances for all the wallet's coins on
    any day the wallet had a transaction with any coin so the balance data would be complete.

    Params:
    - profits_df (DataFrame): Daily profits data

    Returns:
    - twr_df (DataFrame): TWR metrics keyed on wallet_address
    """
    # 1. Validate and format df
    # -------------------------
    profits_df = profits_df.copy()

    # Remove rows below materiality threshold
    profits_df = profits_df[~(
        (profits_df['usd_balance'] < wallets_config['features']['usd_materiality']) &
        (abs(profits_df['usd_net_transfers']) < wallets_config['features']['usd_materiality'])
    )]

    # Reset date index to use in calculations
    profits_df = profits_df.reset_index('date')


    # 2. Calculate base metrics
    # -------------------------
    # Calculate holding period returns
    profits_df['pre_transfer_balance'] = profits_df['usd_balance'] - profits_df['usd_net_transfers']
    profits_df['prev_balance'] = (profits_df.groupby(['wallet_address', 'coin_id'], observed=True)
                                  ['usd_balance']
                                  .shift().fillna(0))
    profits_df['days_held'] = (profits_df.groupby(['wallet_address', 'coin_id'], observed=True)
                              ['date']
                              .diff().dt.days.fillna(0))

    # Calculate period returns and weights
    profits_df['period_return'] = np.where(
        profits_df['usd_net_transfers'] != 0,
        profits_df['pre_transfer_balance'] / profits_df['prev_balance'],
        profits_df['usd_balance'] / profits_df['prev_balance']
    )
    # For initial buys, fill with a 100% return value i.e. equivalent to balance == prev_balance
    profits_df['period_return'] = profits_df['period_return'].replace([np.inf, -np.inf], 1).fillna(1)

    # Weight by holding period duration
    profits_df['weighted_return'] = (profits_df['period_return'] - 1) * profits_df['days_held']


    # 3. Calculate TWR and return it
    # ------------------------------
    # Get wallet-coin level date ranges
    wallet_coin_dates = profits_df.groupby(['wallet_address', 'coin_id'],observed=True)['date'].agg(['min', 'max'])
    total_days = (wallet_coin_dates['max'] - wallet_coin_dates['min']).dt.days

    # Calculate TWR per wallet-coin pair
    wallet_coin_weighted_returns = (profits_df.groupby(['wallet_address', 'coin_id'],observed=True)
                                    ['weighted_return']
                                    .sum().fillna(0))

    # Build TWR DataFrame maintaining wallet-coin granularity
    twr_df = pd.DataFrame(index=total_days.index)  # Index is now MultiIndex (wallet, coin)
    twr_df['days_held'] = total_days
    twr_df['time_weighted_return'] = wallet_coin_weighted_returns / twr_df['days_held']

    # Annualize returns
    twr_df['annualized_twr'] = (((1 + twr_df['time_weighted_return']) ** (365 / twr_df['days_held']) - 1)
                                .clip(upper=wallets_config['features']['twr_max_annual_return']))
    twr_df = twr_df.replace([np.inf, -np.inf], np.nan)

    # Winsorize output
    returns_winsorization = wallets_config['features']['returns_winsorization']
    if returns_winsorization > 0:
        twr_df['time_weighted_return'] = u.winsorize(twr_df['time_weighted_return'],returns_winsorization)
        twr_df['annualized_twr'] = u.winsorize(twr_df['annualized_twr'],returns_winsorization)

    return twr_df
