import logging
from pathlib import Path
import pandas as pd

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Locate the config directory
current_dir = Path(__file__).parent
config_directory = current_dir / '..' / '..' / 'config'


def calculate_coin_wallet_balances(
    profits_df: pd.DataFrame,
    balance_date: str
) -> pd.DataFrame:
    """Calculate metric value for each wallet-coin pair on specified date.

    Params:
    - profits_df (DataFrame): Input profits data
    - metric_column (str): Column name to aggregate (e.g. 'usd_balance', 'volume')
    - balance_date (str): Date for metric calculation

    Returns:
    - balances_df (DataFrame): Wallet-coin level data with metric value for specified date
    - col_name (str): the name of the balance column
    """
    balance_date = pd.to_datetime(balance_date)
    balance_date_str = balance_date.strftime('%y%m%d')

    # Filter to date and select only needed columns
    balances_df = profits_df[profits_df['date'] == balance_date][
        ['coin_id', 'wallet_address', 'usd_balance']
    ].copy()

    # Rename the metric column to include the date
    col_name = f'usd_balance/{balance_date_str}'
    balances_df = balances_df.rename(
        columns={'usd_balance': col_name}
    )

    return balances_df,col_name
