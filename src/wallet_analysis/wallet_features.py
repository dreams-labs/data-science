"""
Calculates metrics aggregated at the wallet level
"""

import time
import dreams_core.core as dc

# set up logger at the module level
logger = dc.setup_logger()


def calculate_wallet_level_metrics(profits_df):
    """
    Calculates the return on investment for the wallet (defined below) and
    additional aggregation metrics.

    Profits_df must have initial balances reflected as positive cash_flow_transfers
    and ending balances reflected as negative cash_flow_transfers for calculations to
    be accurately reflected.

    - Invested: the maximum amount of cumulative net inflows for the wallet. This
        that a wallet that has coin x outflows followed by equal coin y inflows,
        the amount invested will show no change.
    - Return: All net transfers summed together, showing the combined change
        in assets and balance

    Params:
    - profits_df (pd.DataFrame): df showing all usd net transfers for coin-wallet pairs,
        with columns coin_id,wallet_address,date

    Returns:
    - wallet_metrics_df (pd.DataFrame): df keyed on wallet_address with columns
        'ingested', 'net_gain' , 'return', and additional aggregation metrics
    """
    start_time = time.time()

    # Precompute necessary transformations
    profits_df['abs_usd_net_transfers'] = profits_df['usd_net_transfers'].abs()
    profits_df['cumsum_cash_flow_transfers'] = profits_df.groupby('wallet_address')['cash_flow_transfers'].cumsum()

    # Metrics that take into account imputed rows/profits
    logger.debug("Calculating wallet metrics based on imputed performance...")
    imputed_metrics_df = profits_df.groupby('wallet_address').agg(
        invested=('cumsum_cash_flow_transfers', 'max'),
        net_gain=('cash_flow_transfers', lambda x: -1 * x.sum()),  # outflows reflect profit-taking
        unique_coins_held=('coin_id', 'nunique')
    )

    # Metrics only based on observed activity
    logger.debug("Calculating wallet metrics based on observed behavior...")
    observed_metrics_df = profits_df[~profits_df['is_imputed']].groupby('wallet_address').agg(
        transaction_days=('date', 'count'),
        total_volume=('abs_usd_net_transfers', 'sum'),
        average_transaction=('abs_usd_net_transfers', 'mean')
    )

    # Join all metrics together
    wallet_metrics_df = imputed_metrics_df.join(observed_metrics_df)

    # Fill 0s for wallets without observed activity
    wallet_metrics_df = wallet_metrics_df.fillna(0)

    # Compute additional derived metrics
    wallet_metrics_df['return'] = wallet_metrics_df['net_gain'] / wallet_metrics_df['invested']

    logger.debug("Wallet metric computation complete after %.2f.", start_time-time.time())


    return wallet_metrics_df
