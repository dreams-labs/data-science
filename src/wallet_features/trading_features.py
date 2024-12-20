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

    # Calculate core wallet metrics
    base_metrics_df = profits_df.groupby('wallet_address').agg(
        total_crypto_buys=('crypto_balance_change', lambda x: abs(x[x > 0].sum())),
        total_crypto_sells=('crypto_balance_change', lambda x: abs(x[x < 0].sum())),
        net_crypto_investment=('crypto_balance_change', 'sum'),
        current_gain=('crypto_cumulative_net_gain', lambda x: x.iloc[-1])
    )

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

    # Add a column showing crypto balance change from real and imputed transfers
    profits_df['crypto_balance_change'] = profits_df['usd_net_transfers']
    profits_df = buy_crypto_start_balance(profits_df, period_start_date)

    # Calculate crypto cost based on lifetime usd transfers
    profits_df['crypto_balance_cost'] = (profits_df
                                       .groupby(['wallet_address', 'coin_id'],
                                               observed=True,
                                               sort=False)  # sort=False since we pre-sorted
                                       ['crypto_balance_change']
                                       .cumsum())

    # Net gain is the current balance less the cost basis
    profits_df['crypto_cumulative_net_gain'] = profits_df['usd_balance'] - profits_df['crypto_balance_cost']

    return profits_df



def aggregate_time_weighted_balance(profits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates time weighted average balance for each wallet.
    Uses crypto_balance_cost which represents capital deployed.

    Params:
    - profits_df (DataFrame): Daily profits data with balances and dates

    Returns:
    - wallet_metrics_df (DataFrame): Time weighted metrics by wallet
    """
    # Sort by date and calculate holding period for each balance
    profits_df = profits_df.sort_values(['wallet_address', 'date'])

    # Calculate days held for each balance level
    profits_df['next_date'] = profits_df.groupby('wallet_address')['date'].shift(-1)
    profits_df['days_held'] = (
        profits_df['next_date'] - profits_df['date']
    ).dt.total_seconds() / (24 * 60 * 60)  # convert to days

    # For the final period, use the last known date
    last_mask = profits_df['next_date'].isna()
    profits_df.loc[last_mask, 'days_held'] = 1  # or we could exclude last period

    # Calculate weighted cost (balance * days held)
    profits_df['weighted_cost'] = profits_df['crypto_balance_cost'] * profits_df['days_held']

    # Group by wallet to get time weighted average
    time_weighted_df = profits_df.groupby('wallet_address').agg(
        total_days=('days_held', 'sum'),
        sum_weighted_cost=('weighted_cost', 'sum')
    )

    time_weighted_df['time_weighted_balance'] = (
        time_weighted_df['sum_weighted_cost'] / time_weighted_df['total_days']
    )

    return time_weighted_df[['time_weighted_balance']]



# def adjust_start_transfers(df: pd.DataFrame, target_date: str) -> pd.DataFrame:
#     """
#     Sets start date cash flow as the negative period_start_date balance.

#     Params:
#     - df (DataFrame): Input dataframe with usd_balance and cash_flow_transfers
#     - target_date (str): Date to adjust transfers for

#     Returns:
#     - df (DataFrame): DataFrame with adjusted cash_flow_transfers

#     Example Case
#     ------------
#     1. Opening balance of $75 with a $0 transfer. Net cash flows are -$75 out of the bank account.
#     No other scenario is possible as the period_starting_balance is always imputed with $0 transfers.
#     """
#     mask = df['date'] == target_date
#     target_balances = df.loc[mask, 'usd_balance']
#     df.loc[mask, 'cash_flow_transfers'] = -target_balances

#     return df

# def adjust_end_transfers(df, target_date):
#     """
#     Sets end date cash flow to a positive balance plus transfers, since transfers happen
#     before the balance is measured.

#     Example Cases
#     -------------
#     1. Imputation Case: Ending balance of $75 with a $0 transfer: Net flow is +$75 end + $0 buy = $75 inflows
#     2. Buy Case: Ending balance of $100 after a $30 buy. Net flow is +$100 end -$30 buy = $70 inflows
#     3. Sell Partial Case: Ending balance of $80 after a $40 sell. Net flow is +$80 end +$40 sell = $120 inflows
#     4. Sell All Case: Ending balance of $0 after a $50 sell. Net flow is $0 end +$50 sell = $50 inflows

#     """
#     end_mask = df['date'] == target_date
#     df.loc[end_mask, 'cash_flow_transfers'] = (
#         df.loc[end_mask, 'usd_balance'] - df.loc[end_mask, 'usd_net_transfers']
#     )
#     return df

# def add_cash_flows_logic(profits_df):
#     """
#     Adds a cash_flow_transfers column that converts starting/ending balances to cash flow equivalents
#     for performance calculations. Handles the timing of transfers by treating:
#     - Start date: Balance is the only inflow since balances already have transfers added
#     - End date: Both final balance and transfers are outflows since the transfers are
#         removed from the end balance

#     Params:
#     - profits_df (df): Daily profits data to compute performance metrics for

#     Returns:
#     - adj_profits_df (df): Input df with cash_flow_transfers column added

#     """
#     adj_profits_df = profits_df.copy()
#     adj_profits_df['cash_flow_transfers'] = -1 * adj_profits_df['usd_net_transfers']

#     start_date = adj_profits_df['date'].min()
#     end_date = adj_profits_df['date'].max()

#     adj_profits_df = adjust_start_transfers(adj_profits_df, start_date)
#     adj_profits_df = adjust_end_transfers(adj_profits_df, end_date)

#     return adj_profits_df




# @u.timing_decorator
# def calculate_wallet_trading_features(profits_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Calculates comprehensive trading metrics by combining base trading features with
#     time-weighted and realized returns.

#     Params:
#     - profits_df (DataFrame): Daily profits with columns: wallet_address, coin_id, date,
#       cash_flow_transfers, usd_net_transfers, usd_balance, is_imputed

#     Returns:
#     - wallet_metrics_df (DataFrame): Trading metrics keyed on wallet_address
#     """

#     # Calculate base trading features
#     profits_df['date'] = pd.to_datetime(profits_df['date'])
#     profits_df = profits_df.sort_values(['wallet_address', 'coin_id', 'date'])

#     # Precompute necessary transformations
#     profits_df['abs_usd_net_transfers'] = profits_df['usd_net_transfers'].abs()
#     profits_df['cumsum_cash_flow_transfers'] = (profits_df.sort_values(by='date')
#                                             .groupby('wallet_address')['cash_flow_transfers']
#                                             .cumsum())

#     # Metrics that incorporate imputed rows
#     base_metrics_df = profits_df.groupby('wallet_address').agg(
#         total_inflows=('cash_flow_transfers', lambda x: x[x > 0].sum()),
#         total_outflows=('cash_flow_transfers', lambda x: abs(x[x < 0].sum())),
#         total_net_flows=('cash_flow_transfers', lambda x: -x.sum()),
#         max_investment=('cumsum_cash_flow_transfers', 'max'),
#     )

#     # Set floor of 0 on max_investment to match the business constraint that wallet balances cannot be negative.
#     # This handles edge cases where very small initial balances (< $0.01) get rounded to 0 and subsequent outflows
#     # create artificial negative "max_investment" values.
#     base_metrics_df['max_investment'] = base_metrics_df['max_investment'].clip(lower=0)

#     # Observed activity metrics
#     observed_metrics_df = profits_df[~profits_df['is_imputed']].groupby('wallet_address').agg(
#         transaction_days=('date', 'nunique'),
#         unique_coins_traded=('coin_id', 'nunique'),
#         cash_buy_inflows=('usd_net_transfers', lambda x: x[x > 0].sum()),
#         cash_sell_outflows=('usd_net_transfers', lambda x: abs(x[x < 0].sum())),
#         cash_net_flows=('usd_net_transfers', lambda x: -x.sum()),
#         total_volume=('abs_usd_net_transfers', 'sum'),
#         average_transaction=('abs_usd_net_transfers', 'mean'),
#     )

#     # Combine all metrics
#     wallet_trading_features_df = base_metrics_df.join(observed_metrics_df)

#     # Fill missing values and clean up
#     wallet_trading_features_df = wallet_trading_features_df.fillna(0)
#     wallet_trading_features_df = wallet_trading_features_df.replace(-0, 0)

#     # Calculate wallet-level ratio metrics
#     period_duration = (profits_df['date'].max() - profits_df['date'].min()).days + 1
#     wallet_trading_features_df['activity_density'] = (
#         wallet_trading_features_df['transaction_days'] / period_duration
#     )
#     wallet_trading_features_df['volume_vs_investment_ratio'] = np.where(
#         wallet_trading_features_df['max_investment'] > 0,
#         wallet_trading_features_df['total_volume'] / wallet_trading_features_df['max_investment'],
#         0
#     )

#     return wallet_trading_features_df


