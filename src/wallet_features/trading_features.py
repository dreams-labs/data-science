"""
Calculates metrics related to trading performance

Intended function sequence:

# Base feature calculation
profits_df = wtf.add_cash_flow_transfers_logic(profits_df)
trading_features = wtf.calculate_wallet_trading_features(profits_df)

# Only if records for wallets not in profits_df need to be filled
trading_features = wtf.fill_trading_features_data(trading_features, wallet_cohort)
"""
import time
import logging
from pathlib import Path
import pandas as pd
import yaml

# Local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()
wallets_metrics_config = u.load_config('../config/wallets_metrics_config.yaml')
wallets_features_config = yaml.safe_load(Path('../config/wallets_features_config.yaml').read_text(encoding='utf-8'))



def add_cash_flow_transfers_logic(profits_df):
    """
    Adds a cash_flow_transfers column to profits_df that can be used to compute
    the wallet's gain and investment amount by converting their starting and ending
    balances to cash flow equivilants.

    Params:
    - profits_df (df): profits_df that needs wallet investment peformance computed
        based on the earliest and latest dates in the df.

    Returns:
    - adj_profits_df (df): input df with the cash_flow_transfers column added
    """

    def adjust_end_transfers(df, target_date):
        df.loc[df['date'] == target_date, 'cash_flow_transfers'] -= df.loc[df['date'] == target_date, 'usd_balance']
        return df

    def adjust_start_transfers(df, target_date):
        df.loc[df['date'] == target_date, 'cash_flow_transfers'] = df.loc[df['date'] == target_date, 'usd_balance']
        return df

    # Copy df and add cash flow column
    adj_profits_df = profits_df.copy()
    adj_profits_df['cash_flow_transfers'] = adj_profits_df['usd_net_transfers']

    # Modify the records on the start and end dates to reflect the balances
    start_date = adj_profits_df['date'].min()
    end_date = adj_profits_df['date'].max()

    adj_profits_df = adjust_start_transfers(adj_profits_df,start_date)
    adj_profits_df = adjust_end_transfers(adj_profits_df,end_date)

    return adj_profits_df



def calculate_wallet_trading_features(profits_df):
    """
    Calculates the return on investment for the wallet and additional aggregation metrics,
    ensuring proper date-based ordering for cumulative calculations.

    Profits_df must have initial balances reflected as positive cash_flow_transfers
    and ending balances reflected as negative cash_flow_transfers for calculations to
    be accurately reflected.

    - Invested: the maximum amount of cumulative net inflows for the wallet,
        properly ordered by date to ensure accurate running totals
    - Return: All net transfers summed together, showing the combined change
        in assets and balance

    Params:
    - profits_df (pd.DataFrame): df showing all usd net transfers for coin-wallet pairs,
        with columns coin_id, wallet_address, date, cash_flow_transfers, usd_net_transfers,
        and is_imputed

    Returns:
    - wallet_metrics_df (pd.DataFrame): df keyed on wallet_address with columns
        'invested', 'net_gain', 'return', and additional aggregation metrics
    """
    start_time = time.time()
    logger.info("Calculating wallet trading features...")

    # Ensure date is in datetime format
    profits_df['date'] = pd.to_datetime(profits_df['date'])

    # Sort by date and wallet_address to ensure proper cumulative calculations
    profits_df = profits_df.sort_values(['wallet_address', 'date'])

    # Precompute necessary transformations
    profits_df['abs_usd_net_transfers'] = profits_df['usd_net_transfers'].abs()

    # Calculate cumsum by wallet, respecting date order
    profits_df['cumsum_cash_flow_transfers'] = profits_df.groupby('wallet_address')['cash_flow_transfers'].cumsum()

    # Calculate per-coin cumulative flows to catch potential issues
    profits_df['cumsum_by_coin'] = profits_df.groupby(['wallet_address', 'coin_id'])['cash_flow_transfers'].cumsum()

    # Metrics that take into account imputed rows/profits
    logger.debug("Calculating wallet metrics based on imputed performance...")
    imputed_metrics_df = profits_df.groupby('wallet_address').agg(
        invested=('cumsum_cash_flow_transfers', 'max'),
        net_gain=('cash_flow_transfers', lambda x: -1 * x.sum()),  # outflows reflect profit-taking
        unique_coins_traded=('coin_id', 'nunique'),
        first_activity=('date', 'min'),
        last_activity=('date', 'max')
    )

    # Metrics only based on observed activity
    logger.debug("Calculating wallet metrics based on observed behavior...")
    observed_metrics_df = profits_df[~profits_df['is_imputed']].groupby('wallet_address').agg(
        transaction_days=('date', 'nunique'),  # Changed from count to nunique for actual trading days
        total_volume=('abs_usd_net_transfers', 'sum'),
        average_transaction=('abs_usd_net_transfers', 'mean')
    )

    # Join all metrics together
    wallet_trading_features_df = imputed_metrics_df.join(observed_metrics_df)

    # Fill 0s for wallets without observed activity
    wallet_trading_features_df = wallet_trading_features_df.fillna(0)

    # Remove any negative 0s
    wallet_trading_features_df = wallet_trading_features_df.replace(-0,0)

    # Compute additional derived metrics
    wallet_trading_features_df['activity_days'] = (wallet_trading_features_df['last_activity'] -
                                        wallet_trading_features_df['first_activity']).dt.days + 1
    wallet_trading_features_df['activity_density'] = (wallet_trading_features_df['transaction_days']
                                                      / wallet_trading_features_df['activity_days'])

    logger.info(f"Wallet trading features computed after {time.time() - start_time:.2f} seconds")

    return wallet_trading_features_df



def fill_trading_features_data(wallet_trading_features_df, wallet_cohort):
    """
    Fill missing wallet data for all wallets in wallet_cohort that are not in window_wallets_df.

    Parameters:
    wallet_trading_features_df (pd.DataFrame): DataFrame with wallet trading features
    wallet_cohort (array-like): Array of all wallet addresses that should be present

    Returns:
    pd.DataFrame: Complete DataFrame with filled values for missing wallets
    """

    # Create the fill value dictionary
    fill_values = {
        'invested': 0,
        'net_gain': 0,
        'unique_coins_traded': 0,
        'transaction_days': 0,
        'total_volume': 0,
        'average_transaction': 0,
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
