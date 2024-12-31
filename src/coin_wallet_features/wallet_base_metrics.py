import logging
from typing import Tuple,List
import pandas as pd
import wallet_features.trading_features as wtf
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)



def calculate_coin_wallet_balances(
    profits_df: pd.DataFrame,
    balance_date: str
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate metric value for each wallet-coin pair on specified date.

    Params:
    - profits_df (DataFrame): Input profits data
    - metric_column (str): Column name to aggregate (e.g. 'usd_balance', 'volume')
    - balance_date (str): Date for metric calculation

    Returns:
    - balances_df (DataFrame): Wallet-coin level data with metric value for specified date
        with indices (coin_id','wallet_address)
    """
    balance_date = pd.to_datetime(balance_date)
    balance_date_str = balance_date.strftime('%y%m%d')

    # Filter to date and select only needed columns
    balances_df = profits_df[profits_df['date'] == balance_date][
        ['coin_id', 'wallet_address', 'usd_balance']
    ].copy().set_index(['coin_id','wallet_address'])

    # Rename the metric column to include the date
    col_name = f'usd_balance_{balance_date_str}'
    balances_df = balances_df.rename(
        columns={'usd_balance': col_name}
    )

    return balances_df



def calculate_coin_wallet_trading_metrics(profits_df, start_date, end_date):
    """
    Creates a coin-wallet multiindexed df with trading metrics for each pair.

    Params:
    - profits_df (df): df with period boundaries set to start and end dates
    - start_date, end_date (str): YYYY-MM-DD dates

    Returns:
    - cw_trading_metrics_df (df): df multiindexed on coin_id,wallet_address with trading metrics
    """
    # Assert period and copy
    u.assert_period(profits_df, start_date, end_date)
    rekeyed_profits_df = profits_df.copy()

    # Create profits_df keyed on hybrid coin_id|wallet_address
    rekeyed_profits_df['wallet_address'] = (rekeyed_profits_df['coin_id'].astype(str) + '|'
                                            + rekeyed_profits_df['wallet_address'].astype(str))


    # Calculate trading features on the hybrid index level
    cw_trading_metrics_df = wtf.calculate_wallet_trading_features(rekeyed_profits_df,start_date,end_date)

    # Split the hybrid index back into wallet_address and coin_id
    wallet_coin = cw_trading_metrics_df.index.str.split('|', expand=True)
    cw_trading_metrics_df.index = pd.MultiIndex.from_tuples(wallet_coin, names=['coin_id', 'wallet_address'])

    # Reconvert the wallet_address back to int
    cw_trading_metrics_df.index = pd.MultiIndex.from_tuples(
        [(c, int(w)) for c, w in cw_trading_metrics_df.index],
        names=['coin_id', 'wallet_address']
    )

    return cw_trading_metrics_df
