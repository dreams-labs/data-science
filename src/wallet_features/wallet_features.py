"""
Calculates metrics aggregated at the wallet level
"""

import time
import logging
import pandas as pd

import wallet_modeling.wallet_modeling as wm
import wallet_features.wallet_coin_features as wcf
import wallet_features.wallet_coin_date_features as wcdf

# set up logger at the module level
logger = logging.getLogger(__name__)



def calculate_wallet_features(profits_df, market_data_df, wallet_cohort):
    """
    Calculates all features for the wallets in a given profits_df

    Params:
    - profits_df (df): for the window over which the metrics should be computed
    - market_data_df (df): the full dataset
    - wallet_cohort (array-like): Array of all wallet addresses that should be present

    Returns:
    - wallet_features_df (df): df indexed on wallet_address with all features
    """
    # Add transaction metrics
    profits_df = wcf.add_cash_flow_transfers_logic(profits_df)
    wallet_features_df = calculate_wallet_level_metrics(profits_df)

    # Fill missing values in transaction data
    wallet_features_df = fill_missing_wallet_data(wallet_features_df, wallet_cohort)

    # Add market data metrics
    market_features_df = wcdf.calculate_market_data_features(profits_df,market_data_df)
    wallet_features_df = wallet_features_df.join(
        market_features_df.drop('total_volume',axis=1)
        ,how='left'
        ).fillna(0)

    # Add performance metrics
    performance_df = wm.generate_target_variables(wallet_features_df)
    performance_df = performance_df.drop(['invested','net_gain'],axis=1)
    wallet_features_df = wallet_features_df.join(performance_df)

    return wallet_features_df



def calculate_wallet_level_metrics(profits_df):
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
    wallet_metrics_df = imputed_metrics_df.join(observed_metrics_df)

    # Fill 0s for wallets without observed activity
    wallet_metrics_df = wallet_metrics_df.fillna(0)

    # Remove any negative 0s
    wallet_metrics_df = wallet_metrics_df.replace(-0,0)

    # Compute additional derived metrics
    wallet_metrics_df['activity_days'] = (wallet_metrics_df['last_activity'] -
                                        wallet_metrics_df['first_activity']).dt.days + 1
    wallet_metrics_df['activity_density'] = wallet_metrics_df['transaction_days'] / wallet_metrics_df['activity_days']

    logger.debug(f"Wallet metric computation complete after {time.time() - start_time:.2f} seconds")

    return wallet_metrics_df



def fill_missing_wallet_data(window_wallets_df, wallet_cohort):
    """
    Fill missing wallet data for all wallets in wallet_cohort that are not in window_wallets_df.

    Parameters:
    window_wallets_df (pd.DataFrame): DataFrame with wallet metrics
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
    complete_df.update(window_wallets_df)

    return complete_df
