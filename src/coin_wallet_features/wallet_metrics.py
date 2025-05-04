import logging
import pandas as pd
import wallet_features.trading_features as wtf
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)







# -----------------------------------
#       Main Interface Function
# -----------------------------------

def compute_coin_wallet_metrics(
        wallets_coin_config: dict,
        profits_df: pd.DataFrame,
        period_start: str,
        period_end: str
        ) -> pd.DataFrame:
    """
    Compute coin-wallet–level metrics: balances and trading metrics.

    Params:
    - wallets_coin_config (dict): dict from .yaml file
    - profits_df (DataFrame): must include columns ['coin_id','wallet_address',…].
    - period_start,period_end str(YYYY-MM-DD): period start and end dates

    Returns:
    - cw_metrics_df (DataFrame): MultiIndex [coin_id, wallet_address] with
        'balances/...’ and 'trading/...’ feature columns.
    """
    # Confirm time period is correct
    u.assert_period(profits_df, period_start, period_end)

    # 1) Build base index of all (coin, wallet) pairs
    idx = (
        profits_df[['coin_id', 'wallet_address']]
        .drop_duplicates()
        .set_index(['coin_id', 'wallet_address'])
        .index
    )
    cw_metrics_df = pd.DataFrame(index=idx)

    # 2) Calculate balances
    balances_df = calculate_coin_wallet_ending_balances(
        profits_df
    ).add_prefix('balances/')
    cw_metrics_df = (
        cw_metrics_df
        .join(balances_df, how='left')
        .fillna({col: 0 for col in balances_df.columns})
    )

    # 3) Calculate trading metrics
    trading_df = calculate_coin_wallet_trading_metrics(
        profits_df,
        period_start,
        period_end,
        wallets_coin_config['wallet_features']['drop_trading_metrics']
    ).add_prefix('trading/')
    cw_metrics_df = (
        cw_metrics_df
        .join(trading_df, how='left')
        .fillna({col: 0 for col in trading_df.columns})
    )

    return cw_metrics_df




# ------------------------------
#         Helper Functions
# ------------------------------

def calculate_coin_wallet_ending_balances(profits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate metric value for each wallet-coin pair on specified dates.

    Params:
    - profits_df (DataFrame): Input profits data

    Returns:
    - balances_df (DataFrame): Wallet-coin level data with metric values for each specified date
        with indices ('coin_id', 'wallet_address').
    """
    # Create a DataFrame with coin_id and wallet_address as indices
    balances_df = profits_df[['coin_id', 'wallet_address']].drop_duplicates().set_index(['coin_id', 'wallet_address'])

    # Starting Balance
    period_starting_balance = profits_df['date'].min()
    starting_balances = profits_df.loc[
        profits_df['date'] == period_starting_balance,
        ['coin_id', 'wallet_address', 'usd_balance']
    ].set_index(['coin_id', 'wallet_address'])

    # Ending Balance
    period_end = profits_df['date'].max()
    ending_balances = profits_df.loc[
        profits_df['date'] == period_end,
        ['coin_id', 'wallet_address', 'usd_balance']
    ].set_index(['coin_id', 'wallet_address'])

    # Add the balance column to balances_df
    balances_df['usd_balance_starting'] = starting_balances['usd_balance']
    balances_df['usd_balance_ending'] = ending_balances['usd_balance']

    return balances_df



def calculate_coin_wallet_trading_metrics(profits_df, start_date, end_date, drop_trading_metrics):
    """
    Creates a coin-wallet multiindexed df with trading metrics for each pair.

    Params:
    - profits_df (df): df with period boundaries set to start and end dates
    - start_date, end_date (str): YYYY-MM-DD dates
    - drop_trading_metrics (list of strings): columns to drop

    Returns:
    - cw_trading_metrics_df (df): df multiindexed on coin_id,wallet_address with trading metrics
    """
    # Assert period and copy
    u.assert_period(profits_df, start_date, end_date)
    rekeyed_profits_df = profits_df.copy()

    # Create profits_df keyed on hybrid coin_id|wallet_address
    rekeyed_profits_df['wallet_address'] = (rekeyed_profits_df['coin_id'].astype(str) + '|'
                                            + rekeyed_profits_df['wallet_address'].astype(int).astype(str))

    # Calculate trading features on the hybrid index level
    cw_trading_metrics_df = wtf.calculate_wallet_trading_features(rekeyed_profits_df,
                                                                  start_date,end_date,
                                                                  include_twb_metrics=False)

    # Split the hybrid index back into wallet_address and coin_id
    wallet_coin = cw_trading_metrics_df.index.str.split('|', expand=True)
    cw_trading_metrics_df.index = pd.MultiIndex.from_tuples(wallet_coin, names=['coin_id', 'wallet_address'])

    # Reconvert the wallet_address back to int
    cw_trading_metrics_df.index = pd.MultiIndex.from_tuples(
        [(c, int(w)) for c, w in cw_trading_metrics_df.index],
        names=['coin_id', 'wallet_address']
    )

    # Drop metrics if configured to do so
    if len(drop_trading_metrics) > 0:
        existing_cols = [col for col in drop_trading_metrics if col in cw_trading_metrics_df.columns]
        missing_cols = set(drop_trading_metrics) - set(existing_cols)

        if existing_cols:
            cw_trading_metrics_df = cw_trading_metrics_df.drop(columns=existing_cols)
        if missing_cols:
            logger.warning(f"Trading drop metrics not found: {missing_cols}")


    return cw_trading_metrics_df
