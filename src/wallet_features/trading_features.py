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

    logger.info("1")
    # Extract the last row per wallet-coin from the pre-sorted DataFrame
    last_rows = profits_df.drop_duplicates(subset=['wallet_address', 'coin_id'], keep='last')

    # Group by wallet to calculate the total current gain
    gains_df = last_rows.groupby('wallet_address', as_index=False).agg(
        current_gain=('crypto_cumulative_net_gain', 'sum')
    ).set_index('wallet_address')

    logger.info("2")
    # Precompute metrics in a single pass
    profits_df['positive_changes'] = np.where(profits_df['crypto_balance_change'] > 0, profits_df['crypto_balance_change'], 0)
    profits_df['negative_changes'] = np.where(profits_df['crypto_balance_change'] < 0, -profits_df['crypto_balance_change'], 0)

    # Group and aggregate precomputed columns
    transaction_metrics_df = profits_df.groupby('wallet_address').agg(
        total_crypto_buys=('positive_changes', 'sum'),
        total_crypto_sells=('negative_changes', 'sum'),
        net_crypto_investment=('crypto_balance_change', 'sum')
    )

    logger.info("3")
    # Combine the metrics
    base_metrics_df = transaction_metrics_df.join(gains_df)

    logger.info("4")
    # Calculate metrics for actual transaction activity (excluding imputed rows)
    # Filter only necessary rows and columns
    filtered_df = profits_df.loc[~profits_df['is_imputed'],
                                 ['wallet_address', 'date', 'coin_id', 'crypto_balance_change']]

    # Precompute absolute balance changes
    filtered_df['abs_balance_change'] = filtered_df['crypto_balance_change'].abs()

    # Group by wallet and aggregate metrics
    observed_metrics_df = filtered_df.groupby('wallet_address', as_index=False).agg(
        transaction_days=('date', 'nunique'),
        unique_coins_traded=('coin_id', 'nunique'),
        total_volume=('abs_balance_change', 'sum'),
        average_transaction=('abs_balance_change', 'mean')
    ).set_index('wallet_address')

    logger.info("5")
    # Calculate time weighted balance
    time_weighted_df = aggregate_time_weighted_balance(profits_df)

    logger.info("6")
    # Combine all metrics and handle edge cases
    wallet_trading_features_df = base_metrics_df.join([
        observed_metrics_df,
        time_weighted_df
    ])
    wallet_trading_features_df = wallet_trading_features_df.fillna(0)
    wallet_trading_features_df = wallet_trading_features_df.replace(-0, 0)

    logger.info("7")
    # Calculate activity density (frequency of trading)
    start = datetime.strptime(period_start_date, '%Y-%m-%d')
    end = datetime.strptime(period_end_date, '%Y-%m-%d')
    period_duration = (end - start).days + 1
    wallet_trading_features_df['activity_density'] = (
        wallet_trading_features_df['transaction_days'] / period_duration
    )

    logger.info("8")
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
    # Calculate days held for each balance level
    logger.debug('a')
    profits_df['next_date'] = profits_df.groupby(['wallet_address', 'coin_id'])['date'].shift(-1)
    profits_df['days_held'] = (
        profits_df['next_date'] - profits_df['date']
    ).dt.total_seconds() / (24 * 60 * 60)

    logger.debug('b')
    # There is no hold time for the closing balance on the final day
    last_mask = profits_df['next_date'].isna()
    profits_df.loc[last_mask, 'days_held'] = 0

    logger.debug('c')
    # Calculate weighted cost (balance * days held)
    profits_df['weighted_cost'] = profits_df['crypto_cost_basis'] * profits_df['days_held']

    logger.debug('d')
    # Precompute totals using a single aggregation for each metric
    total_days = profits_df.groupby(['wallet_address', 'coin_id'])['days_held'].sum()
    sum_weighted_cost = profits_df.groupby(['wallet_address', 'coin_id'])['weighted_cost'].sum()

    # Combine results into a single DataFrame
    coin_level_twb = pd.concat([total_days, sum_weighted_cost], axis=1)
    coin_level_twb.columns = ['total_days', 'sum_weighted_cost']

    logger.debug('e')
    coin_level_twb['coin_twb'] = (
        coin_level_twb['sum_weighted_cost'] / coin_level_twb['total_days']
    )

    logger.debug('f')
    # Then sum up to wallet level
    wallet_twb = coin_level_twb.groupby('wallet_address')['coin_twb'].sum()
    return pd.DataFrame(wallet_twb).rename(columns={'coin_twb': 'time_weighted_balance'})
