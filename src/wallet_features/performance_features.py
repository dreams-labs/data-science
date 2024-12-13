"""
Calculates wallet-level financial performance metrics
"""
import logging
import pandas as pd
import numpy as np

# local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()


@u.timing_decorator
def calculate_performance_features(wallet_features_df):
    """
    Generates various target variables for modeling wallet performance.

    Parameters:
    - wallet_features_df: pandas DataFrame with columns ['total_net_flows', 'max_investment']

    Returns:
    - DataFrame with additional target variables
    """
    metrics_df = wallet_features_df[['max_investment','total_net_flows','cash_net_flows']].copy().round(6)
    returns_winsorization = wallets_config['modeling']['returns_winsorization']
    epsilon = 1e-10

    # Calculate base return, including unrealized price change impacts
    metrics_df['return'] = np.where(abs(metrics_df['max_investment']) == 0,0,
                                    metrics_df['total_net_flows'] / metrics_df['max_investment'])

    # Calculate realized return, based on actual cash flows only
    metrics_df['realized_return'] = np.where(abs(metrics_df['max_investment']) == 0,0,
                                    metrics_df['cash_net_flows'] / metrics_df['max_investment'])

    # Apply winsorization
    if returns_winsorization > 0:
        metrics_df['return_unwinsorized'] = metrics_df['return']
        metrics_df['return'] = u.winsorize(metrics_df['return'],returns_winsorization)

    # Normalize returns
    metrics_df['norm_return'] = (metrics_df['return'] - metrics_df['return'].min()) / \
        (metrics_df['return'].max() - metrics_df['return'].min())

    # Normalize logged investments
    log_invested = np.log10(metrics_df['max_investment'] + epsilon)
    metrics_df['norm_invested'] = (log_invested - log_invested.min()) / \
        (log_invested.max() - log_invested.min())

    # Performance score
    return_weight = wallets_config['modeling']['target_var_params']['performance_score_return_weight']
    metrics_df['performance_score'] = (return_weight * metrics_df['norm_return'] +
                                     (1-return_weight) * metrics_df['norm_invested'])

    # Size-adjusted rank
    # Create mask for zero values
    zero_mask = metrics_df['max_investment'] == 0

    # Create quartiles series initialized with 'q0' for zero values
    quartiles = pd.Series('q0', index=metrics_df.index)

    # Calculate quartiles for non-zero values
    non_zero_quartiles = pd.qcut(metrics_df['max_investment'][~zero_mask],
                                q=4,
                                labels=['q1', 'q2', 'q3', 'q4'])

    # Assign the quartiles to non-zero values
    quartiles[~zero_mask] = non_zero_quartiles

    # Calculate size-adjusted rank within each quartile
    metrics_df['size_adjusted_rank'] = metrics_df.groupby(quartiles)['return'].rank(pct=True)


    # Clean up intermediate columns
    cols_to_drop = ['norm_return', 'norm_invested', 'norm_gain', 'cash_net_flows']
    metrics_df = metrics_df.drop(columns=[c for c in cols_to_drop
                                        if c in metrics_df.columns])

    # Verify all input wallets exist in output
    missing_wallets = set(wallet_features_df.index) - set(metrics_df.index)
    if missing_wallets:
        raise ValueError(f"Found {len(missing_wallets)} wallets in input that are missing from output")

    return metrics_df.round(6)



# TO DO: ADD FULL PRICE DATA
# --------------------------
# the balances of each coin-wallet pair needs to be imputed for every date that the wallet
# has a transaction in order for this calculation to work correctly against production data.

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

    # Compute TWR and days_held using vectorized operations
    twr_df = profits_df.groupby('wallet_address').agg(
        time_weighted_return=('weighted_return',
                              lambda x: safe_twr(x, profits_df.loc[x.index, 'wallet_address'].iloc[0])),
        days_held=('date', lambda x: max((x.max() - x.min()).days, 1))
    )

    # Annualize returns
    twr_df['annualized_twr'] = ((1 + twr_df['time_weighted_return']) ** (365 / twr_df['days_held'])) - 1
    twr_df = twr_df.replace([np.inf, -np.inf], np.nan)

    # Winsorize output
    returns_winsorization = wallets_config['modeling']['returns_winsorization']
    if returns_winsorization > 0:
        twr_df['time_weighted_return'] = u.winsorize(twr_df['time_weighted_return'],returns_winsorization)
        twr_df['annualized_twr'] = u.winsorize(twr_df['annualized_twr'],returns_winsorization)

    return twr_df
