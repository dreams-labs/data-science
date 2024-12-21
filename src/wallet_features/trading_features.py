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
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)



@u.timing_decorator
def calculate_wallet_trading_features(
    base_profits_df: pd.DataFrame,
    period_start_date: str,
    period_end_date: str
) -> pd.DataFrame:
    """
    Calculates comprehensive crypto trading metrics for each wallet.

    Params:
    - base_profits_df (DataFrame): Daily profits data
    - period_start_date (str): Period start in 'YYYY-MM-DD' format
    - period_end_date (str): Period end in 'YYYY-MM-DD' format

    Required columns: wallet_address, coin_id, date, usd_balance,
                    usd_net_transfers, is_imputed

    Returns:
    - wallet_metrics_df (DataFrame): Trading metrics keyed on wallet_address with columns:
        - total_crypto_buys: Sum of positive balance changes
        - total_crypto_sells: Sum of negative balance changes
        - net_crypto_investment: Net sum of all balance changes
        - current_gain: Final unrealized gain
        - transaction_days: Number of days with activity
        - unique_coins_traded: Number of unique coins
        - total_volume: Sum of absolute balance changes
        - average_transaction: Mean absolute balance change
        - activity_density: Transaction days / period duration
        - volume_vs_twb_ratio: Volume relative to time-weighted balance
    """
    profits_df = base_profits_df.copy()

    # Calculate additional columns
    profits_df = calculate_crypto_balance_columns(profits_df, period_start_date)

    profits_df['date'] = pd.to_datetime(profits_df['date'])
    profits_df = profits_df.sort_values(['wallet_address', 'coin_id', 'date'])

    # Get last rows per coin-wallet for final gain calculation
    last_rows = profits_df.sort_values('date').groupby(['wallet_address', 'coin_id']).last()
    gains_df = last_rows.groupby('wallet_address').agg(
        current_gain=('crypto_cumulative_net_gain', 'sum')
    )

    # Calculate transaction metrics at wallet level
    transaction_metrics_df = profits_df.groupby('wallet_address').agg(
        total_crypto_buys=('crypto_balance_change', lambda x: abs(x[x > 0].sum())),
        total_crypto_sells=('crypto_balance_change', lambda x: abs(x[x < 0].sum())),
        net_crypto_investment=('crypto_balance_change', 'sum')
    )

    # Combine the metrics
    base_metrics_df = transaction_metrics_df.join(gains_df)

    # Calculate metrics for actual transaction activity (excluding imputed rows)
    observed_metrics_df = profits_df[~profits_df['is_imputed']].groupby('wallet_address').agg(
        transaction_days=('date', 'nunique'),
        unique_coins_traded=('coin_id', 'nunique'),
        total_volume=('crypto_balance_change', lambda x: abs(x).sum()),
        average_transaction=('crypto_balance_change', lambda x: abs(x).mean())
    )

    # Calculate time weighted balance
    time_weighted_df = aggregate_time_weighted_balance(profits_df)

    # Combine all metrics and handle edge cases
    wallet_trading_features_df = base_metrics_df.join([
        observed_metrics_df,
        time_weighted_df
    ])
    wallet_trading_features_df = wallet_trading_features_df.fillna(0)
    wallet_trading_features_df = wallet_trading_features_df.replace(-0, 0)

    # Calculate activity density (frequency of trading)
    start = datetime.strptime(period_start_date, '%Y-%m-%d')
    end = datetime.strptime(period_end_date, '%Y-%m-%d')
    period_duration = (end - start).days + 1
    wallet_trading_features_df['activity_density'] = (
        wallet_trading_features_df['transaction_days'] / period_duration
    )

    # Calculate volume to time-weighted balance ratio
    wallet_trading_features_df['volume_vs_twb_ratio'] = np.where(
        wallet_trading_features_df['time_weighted_balance'] > 0,
        wallet_trading_features_df['total_volume'] / wallet_trading_features_df['time_weighted_balance'],
        0
    )

    return wallet_trading_features_df


@u.timing_decorator
def buy_crypto_start_balance(df: pd.DataFrame, period_start_date: str) -> pd.DataFrame:
    """
    Sets start date crypto balance change as the initial balance value.

    Params:
    - df (DataFrame): Input dataframe with usd_balance and crypto_balance_change
    - period_start_date (str): The start date of the period

    Returns:
    - df (DataFrame): DataFrame with adjusted crypto_balance_change

    Example Case
    ------------
    Opening balance of $75 with $0 transfer results in:
    crypto_balance_change = +$75 (increase in crypto holdings)
    """
    period_start_date = datetime.strptime(period_start_date,'%Y-%m-%d')
    starting_balance_date = period_start_date - timedelta(days=1)

    mask = df['date'] == starting_balance_date
    target_balances = df.loc[mask, 'usd_balance']
    df.loc[mask, 'crypto_balance_change'] = target_balances

    return df




@u.timing_decorator
def get_cost_basis_df(profits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized cost basis calculation matching original logic.

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

    # Calculate running cost basis by group
    for _, group in df.groupby(['wallet_address', 'coin_id']):
        pct_kept = 1 - group['pct_sold']
        cost_basis = np.zeros(len(group))

        for i in range(len(group)):
            if i == 0:
                cost_basis[i] = group['cost_basis_bought'].iloc[i]
            else:
                cost_basis[i] = (cost_basis[i-1] * pct_kept.iloc[i] +
                               group['cost_basis_bought'].iloc[i])

        df.loc[group.index, 'crypto_cost_basis'] = cost_basis

    result_df = df[['wallet_address', 'coin_id', 'date', 'crypto_cost_basis']].copy()

    if len(profits_df) != len(result_df):
        raise ValueError('Record count mismatch')

    return result_df


@u.timing_decorator
def calculate_crypto_balance_columns(profits_df: pd.DataFrame,
                                   period_start_date: str
                                   ) -> pd.DataFrame:
    """
    Adds crypto_balance_change column tracking changes in crypto holdings.
    A positive value indicates an increase in crypto holdings, negative indicates decrease.

    Params:
    - profits_df (DataFrame): Daily profits data to compute balance changes for
    - period_start_date (str): Period start in 'YYYY-MM-DD' format

    Returns:
    - adj_profits_df (DataFrame): Input df with crypto_balance_change column added
    """
    profits_df = profits_df.copy()

    # Sort by wallet, coin, and date for accurate cumulative calculations
    profits_df = profits_df.sort_values(['wallet_address', 'coin_id', 'date'])

    # Crypto balance change equals usd_net_transfers except on the starting_balance_date
    profits_df['crypto_balance_change'] = profits_df['usd_net_transfers']
    profits_df = buy_crypto_start_balance(profits_df, period_start_date)

    # Calculate crypto cost based on lifetime usd transfers
    profits_df['crypto_cumulative_transfers'] = (profits_df
                                       .groupby(['wallet_address', 'coin_id'],
                                               observed=True,
                                               sort=False)  # sort=False since we pre-sorted
                                       ['crypto_balance_change']
                                       .cumsum())

    # Calculate cost basis for each wallet-coin pair and merge into main dataframe
    cost_basis_df = get_cost_basis_df(profits_df)
    profits_df = profits_df.merge(
        cost_basis_df,
        on=['wallet_address', 'coin_id', 'date'],
        how='left'
    )

    # Net gain is the current balance less the cost basis
    profits_df['crypto_cumulative_net_gain'] = profits_df['usd_balance'] - profits_df['crypto_cumulative_transfers']

    return profits_df



@u.timing_decorator
def aggregate_time_weighted_balance(profits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates time weighted average balance for each wallet by:
    1. Computing TWB for each coin-wallet combination
    2. Summing the TWBs up to wallet level

    Params:
    - profits_df (DataFrame): Daily profits data with balances and dates
    - period_end_date (string): Period end as YYYY-MM-DD

    Returns:
    - wallet_metrics_df (DataFrame): Time weighted metrics by wallet
    """
    # Sort by date and calculate holding period for each balance
    profits_df = profits_df.sort_values(['wallet_address', 'coin_id', 'date'])

    # Calculate days held for each balance level
    profits_df['next_date'] = profits_df.groupby(['wallet_address', 'coin_id'])['date'].shift(-1)
    profits_df['days_held'] = (
        profits_df['next_date'] - profits_df['date']
    ).dt.total_seconds() / (24 * 60 * 60)

    # There is no hold time for the closing balance on the final day
    last_mask = profits_df['next_date'].isna()
    profits_df.loc[last_mask, 'days_held'] = 0

    # Calculate weighted cost (balance * days held)
    profits_df['weighted_cost'] = profits_df['crypto_cost_basis'] * profits_df['days_held']

    # First group by wallet-coin to get coin-level TWB
    coin_level_twb = profits_df.groupby(['wallet_address', 'coin_id']).agg(
        total_days=('days_held', 'sum'),
        sum_weighted_cost=('weighted_cost', 'sum')
    )
    coin_level_twb['coin_twb'] = (
        coin_level_twb['sum_weighted_cost'] / coin_level_twb['total_days']
    )

    # Then sum up to wallet level
    wallet_twb = coin_level_twb.groupby('wallet_address')['coin_twb'].sum()
    return pd.DataFrame(wallet_twb).rename(columns={'coin_twb': 'time_weighted_balance'})
