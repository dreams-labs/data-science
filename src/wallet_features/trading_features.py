"""
Calculates metrics related to trading performance

Intended function sequence:

# Base feature calculation
profits_df = wtf.add_cash_flow_transfers_logic(profits_df)
trading_features = wtf.calculate_wallet_trading_features(profits_df)

# Fills values for wallets in wallet_cohort but not in profits_df
trading_features = wtf.fill_trading_features_data(trading_features, wallet_cohort)
"""
import time
import logging
import pandas as pd
import numpy as np

# Set up logger at the module level
logger = logging.getLogger(__name__)


def add_cash_flow_transfers_logic(profits_df):
    """
    Adds a cash_flow_transfers column that converts starting/ending balances to cash flow equivalents
    for performance calculations. Handles the timing of transfers by treating:
    - Start date: Balance is the only inflow since balances already have transfers added
    - End date: Both final balance and transfers are outflows since the transfers are
        removed from the end balance

    Params:
    - profits_df (df): Daily profits data to compute performance metrics for

    Returns:
    - adj_profits_df (df): Input df with cash_flow_transfers column added
    """

    def adjust_start_transfers(df, target_date):
        """
        Sets start date cash flow to just the balance, ignoring transfers since they
        are already added to the balance.

        Example Cases
        -------------
        Imputation Case: Opening balance of $75 with a $0 buy. Net flow is $75 start + $0 buy = $75 inflows.
        Buy Case: Opening balance of $100 with a $30 buy. Net flow is $70 start + $30 buy = $100 inflows.
        Sell Partial Case: Opening balance of $80 after a $40 sell. Net flow is $120 start - $40 sell = $80 inflows.
        Sell All Case: Opening balance of $0 after a $50 sell. Net flow is $50 start - $50 sell = $0 inflows.
        """
        df.loc[df['date'] == target_date, 'cash_flow_transfers'] = df.loc[df['date'] == target_date, 'usd_balance']
        return df

    def adjust_end_transfers(df, target_date):
        """
        Sets end date cash flow to negative balance plus transfers, since transfers happen
        before the balance is measured.

        Example Cases
        -------------
        Imputation Case: Ending balance of $75 with a $0 buy: Net flow is -$75 end + $0 buy = -$75 outflows
        Buy Case: Ending balance of $100 after a $30 buy. Net flow is -$100 end + $30 buy = -$70 outflows
        Sell Partial Case: Ending balance of $80 after a $40 sell. Net flow is -$80 end - $40 sell = $120 outflows
        Sell All Case: Ending balance of $0 after a $50 sell. Net flow is $0 end - $50 sell = -$50 outflows

        """
        end_mask = df['date'] == target_date
        df.loc[end_mask, 'cash_flow_transfers'] = -(
            df.loc[end_mask, 'usd_balance'] - df.loc[end_mask, 'usd_net_transfers']
        )
        return df

    adj_profits_df = profits_df.copy()
    adj_profits_df['cash_flow_transfers'] = adj_profits_df['usd_net_transfers']

    start_date = adj_profits_df['date'].min()
    end_date = adj_profits_df['date'].max()

    adj_profits_df = adjust_start_transfers(adj_profits_df, start_date)
    adj_profits_df = adjust_end_transfers(adj_profits_df, end_date)

    return adj_profits_df



def calculate_wallet_trading_features(profits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates comprehensive trading metrics by combining base trading features with
    time-weighted and realized returns.

    Params:
    - profits_df (DataFrame): Daily profits with columns: wallet_address, coin_id, date,
      cash_flow_transfers, usd_net_transfers, usd_balance, is_imputed

    Returns:
    - wallet_metrics_df (DataFrame): Trading metrics keyed on wallet_address
    """
    start_time = time.time()
    logger.debug("Calculating wallet trading features...")

    # Calculate base trading features
    profits_df['date'] = pd.to_datetime(profits_df['date'])
    profits_df = profits_df.sort_values(['wallet_address', 'coin_id', 'date'])

    # Precompute necessary transformations
    profits_df['abs_usd_net_transfers'] = profits_df['usd_net_transfers'].abs()
    profits_df['cumsum_cash_flow_transfers'] = profits_df.groupby('wallet_address')['cash_flow_transfers'].cumsum()

    # Metrics that incorporate imputed rows
    base_metrics_df = profits_df.groupby('wallet_address').agg(
        imputed_deposits=('cash_flow_transfers', lambda x: x[x > 0].sum()),
        imputed_withdrawals=('cash_flow_transfers', lambda x: abs(x[x < 0].sum())),
        imputed_net_cash_flows=('cash_flow_transfers', lambda x: -x.sum()),
        invested=('cumsum_cash_flow_transfers', 'max'),
        net_gain=('cash_flow_transfers', lambda x: -x.sum()),
    )

    # Calculate additional return metrics using existing functions
    twr_df = calculate_time_weighted_returns(profits_df)

    # Observed activity metrics
    observed_metrics_df = profits_df[~profits_df['is_imputed']].groupby('wallet_address').agg(
        transaction_days=('date', 'nunique'),
        unique_coins_traded=('coin_id', 'nunique'),
        realized_deposits=('usd_net_transfers', lambda x: x[x > 0].sum()),
        realized_withdrawals=('usd_net_transfers', lambda x: abs(x[x < 0].sum())),
        realized_net_cash_flows=('usd_net_transfers', lambda x: -x.sum()),
        total_volume=('abs_usd_net_transfers', 'sum'),
        average_transaction=('abs_usd_net_transfers', 'mean'),
    )

    # Combine all metrics
    wallet_trading_features_df = (
        base_metrics_df
        .join(observed_metrics_df)
        .join(twr_df)
    )

    # Fill missing values and clean up
    wallet_trading_features_df = wallet_trading_features_df.fillna(0)
    wallet_trading_features_df = wallet_trading_features_df.replace(-0, 0)

    # Calculate activity density
    period_duration = (profits_df['date'].max() - profits_df['date'].min()).days + 1
    wallet_trading_features_df['activity_density'] = (
        wallet_trading_features_df['transaction_days'] / period_duration
    )

    logger.info(f"Wallet trading features computed after {time.time() - start_time:.2f} seconds")

    return wallet_trading_features_df



def fill_trading_features_data(wallet_trading_features_df, wallet_cohort):
    """
    Fill missing wallet data for all wallets in wallet_cohort that are not in window_wallets_df.

    Parameters:
    - wallet_trading_features_df (pd.DataFrame): DataFrame with wallet trading features
    - wallet_cohort (array-like): Array of all wallet addresses that should be present

    Returns:
    - pd.DataFrame: Complete DataFrame with filled values for missing wallets
    """

    # Create the fill value dictionary
    fill_values = {
        'invested': 0,
        'net_gain': 0,
        'unique_coins_traded': 0,
        'transaction_days': 0,
        'total_volume': 0,
        'average_transaction': 0,
        'activity_days': 0,
        'activity_density': 0
    }

    # Create a DataFrame with all wallets that should exist
    complete_df = pd.DataFrame(index=wallet_cohort)
    complete_df.index.name = 'wallet_address'

    # Add the fill value columns
    for column, fill_value in fill_values.items():
        complete_df[column] = fill_value

    # Update with actual values where they exist
    complete_df.update(wallet_trading_features_df)

    return complete_df



def calculate_time_weighted_returns(profits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates time-weighted returns (TWR) using actual holding period in days.

    Params:
    - profits_df (DataFrame): Daily profits data

    Returns:
    - twr_df (DataFrame): TWR metrics keyed on wallet_address
    """
    profits_df = profits_df.sort_values(['wallet_address', 'coin_id', 'date'])

    # Calculate holding period returns
    profits_df['pre_transfer_balance'] = profits_df['usd_balance'] - profits_df['usd_net_transfers']
    profits_df['prev_balance'] = profits_df.groupby(['wallet_address', 'coin_id'])['usd_balance'].shift()
    profits_df['days_held'] = profits_df.groupby(['wallet_address', 'coin_id'])['date'].diff().dt.days

    # Calculate period returns and weights
    profits_df['period_return'] = np.where(
        profits_df['usd_net_transfers'] != 0,
        profits_df['pre_transfer_balance'] / profits_df['prev_balance'],
        profits_df['usd_balance'] / profits_df['prev_balance']
    )
    profits_df['period_return'] = profits_df['period_return'].replace([np.inf, -np.inf], 1).fillna(1)

    # Weight by holding period duration
    profits_df['weighted_return'] = (profits_df['period_return'] - 1) * profits_df['days_held']

    # Get total days for each wallet
    total_days = profits_df.groupby('wallet_address')['date'].agg(lambda x: (x.max() - x.min()).days)

    # Calculate TWR using total days held
    def safe_twr(weighted_returns, wallet):
        if len(weighted_returns) == 0 or weighted_returns.isna().all():
            return 0
        days = max(total_days[wallet], 1)  # Get days for this wallet, minimum 1
        return weighted_returns.sum() / days

    # Aggregate by wallet
    twr_df = profits_df.groupby('wallet_address').agg(
        time_weighted_return=('weighted_return', lambda x: safe_twr(x, x.name)),
        days_held=('date', lambda x: max((x.max() - x.min()).days, 1))
    )

    # Annualize returns
    twr_df['annualized_twr'] = ((1 + twr_df['time_weighted_return']) ** (365 / twr_df['days_held'])) - 1
    twr_df = twr_df.replace([np.inf, -np.inf], np.nan)

    return twr_df



def calculate_realized_returns(profits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates realized returns using cash flow transfers through a simple sum of
    inflows vs outflow.

    Params:
    - profits_df (DataFrame): input profits data with cash_flow_transfers column

    Returns:
    - realized_returns_df (DataFrame): Metrics for each wallet:
        - imputed_deposits: Sum of positive cash flows
        - imputed_withdrawals: Sum of negative cash flows (converted to positive)
        - realized_return: (withdrawals/investments) - 1
    """
    realized_returns_df = profits_df.groupby('wallet_address').agg(
        imputed_deposits=('cash_flow_transfers', lambda x: x[x > 0].sum()),
        imputed_withdrawals=('cash_flow_transfers', lambda x: -x[x < 0].sum())
    )

    realized_returns_df['realized_return'] = (
        realized_returns_df['imputed_withdrawals'] /
        realized_returns_df['imputed_deposits']
    ) - 1

    # Handle edge case where wallet has no investments
    realized_returns_df.loc[realized_returns_df['imputed_deposits'] == 0, 'realized_return'] = 0

    return realized_returns_df
