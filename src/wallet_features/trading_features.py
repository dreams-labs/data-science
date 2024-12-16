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
        1. Imputation Case: Opening balance of $75 with a $0 transfer. Net flow is $75 start + $0 buy = $75 inflows.
        2. Buy Case: Opening balance of $100 with a $30 buy. Net flow is $70 start + $30 buy = $100 inflows.
        3. Sell Partial Case: Opening balance of $80 after a $40 sell. Net flow is $120 start - $40 sell = $80 inflows.
        4. Sell All Case: Opening balance of $0 after a $50 sell. Net flow is $50 start - $50 sell = $0 inflows.
        """
        df.loc[df['date'] == target_date, 'cash_flow_transfers'] = df.loc[df['date'] == target_date, 'usd_balance']
        return df

    def adjust_end_transfers(df, target_date):
        """
        Sets end date cash flow to negative balance plus transfers, since transfers happen
        before the balance is measured.

        Example Cases
        -------------
        1. Imputation Case: Ending balance of $75 with a $0 transfer: Net flow is -$75 end + $0 buy = -$75 outflows
        2. Buy Case: Ending balance of $100 after a $30 buy. Net flow is -$100 end + $30 buy = -$70 outflows
        3. Sell Partial Case: Ending balance of $80 after a $40 sell. Net flow is -$80 end - $40 sell = $120 outflows
        4. Sell All Case: Ending balance of $0 after a $50 sell. Net flow is $0 end - $50 sell = -$50 outflows

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
    logger.info("Calculating wallet trading features...")

    # Calculate base trading features
    profits_df['date'] = pd.to_datetime(profits_df['date'])
    profits_df = profits_df.sort_values(['wallet_address', 'coin_id', 'date'])

    # Precompute necessary transformations
    profits_df['abs_usd_net_transfers'] = profits_df['usd_net_transfers'].abs()
    profits_df['cumsum_cash_flow_transfers'] = profits_df.groupby('wallet_address')['cash_flow_transfers'].cumsum()

    # Metrics that incorporate imputed rows
    base_metrics_df = profits_df.groupby('wallet_address').agg(
        total_inflows=('cash_flow_transfers', lambda x: x[x > 0].sum()),
        total_outflows=('cash_flow_transfers', lambda x: abs(x[x < 0].sum())),
        total_net_flows=('cash_flow_transfers', lambda x: -x.sum()),
        max_investment=('cumsum_cash_flow_transfers', 'max'),
    )

    # Observed activity metrics
    observed_metrics_df = profits_df[~profits_df['is_imputed']].groupby('wallet_address').agg(
        transaction_days=('date', 'nunique'),
        unique_coins_traded=('coin_id', 'nunique'),
        cash_buy_inflows=('usd_net_transfers', lambda x: x[x > 0].sum()),
        cash_sell_outflows=('usd_net_transfers', lambda x: abs(x[x < 0].sum())),
        cash_net_flows=('usd_net_transfers', lambda x: -x.sum()),
        total_volume=('abs_usd_net_transfers', 'sum'),
        average_transaction=('abs_usd_net_transfers', 'mean'),
    )

    # Combine all metrics
    wallet_trading_features_df = base_metrics_df.join(observed_metrics_df)

    # Fill missing values and clean up
    wallet_trading_features_df = wallet_trading_features_df.fillna(0)
    wallet_trading_features_df = wallet_trading_features_df.replace(-0, 0)

    # Calculate wallet-level ratio metrics
    period_duration = (profits_df['date'].max() - profits_df['date'].min()).days + 1
    wallet_trading_features_df['activity_density'] = (
        wallet_trading_features_df['transaction_days'] / period_duration
    )
    wallet_trading_features_df['volume_vs_investment_ratio'] = np.where(
        wallet_trading_features_df['max_investment'] > 0,
        wallet_trading_features_df['total_volume'] / wallet_trading_features_df['max_investment'],
        0
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
        'max_investment': 0,
        'total_net_flows': 0,
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
