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




# -----------------------------------
# Main Interface Function
# -----------------------------------

@u.timing_decorator
def calculate_performance_features(trading_features_df,
                                   include_twb_metrics: bool = True) -> pd.DataFrame:

    """
    Calculates a set of profit numerators, investment denominators, ratio combinations,
    and transformations of ratios to create a matrix of performance scores.

    Params:
    - trading_features_df (DataFrame): Required columns:
        - max_investment: Peak portfolio value
        - time_weighted_balance: Average balance including zeros
        - active_time_weighted_balance: Average non-zero balance
        - activity_density: Trading frequency
        - transaction_days: Days with activity
        - average_transaction: Mean transaction size
    - include_twb_metrics (bool): whether to include 'twb' and 'active_twb'
        as balance features
    """
    trading_features_df = trading_features_df.copy()

    # Numerator features reflecting the profit/return rate/cash inflows/inflow
    profits_features_df = calculate_profits_features(trading_features_df)
    profits_features_df = profits_features_df.add_prefix('profits_')

    # Demoniminator features reflecting the balance/investment/outlays
    balance_features_df = calculate_balance_features(trading_features_df,
                                                     include_twb_metrics)
    balance_features_df = balance_features_df.add_prefix('balance_')

    # Combine to make ratios
    ratios_features_df = profits_features_df.join(balance_features_df)
    ratios_features_df = calculate_performance_ratios(ratios_features_df)

    # Generate features using transformations of ratios
    performance_features_df = transform_performance_ratios(ratios_features_df,
                                                           balance_features_df)

    # Check null values
    null_check = performance_features_df.isnull().sum()
    if null_check.any():
        raise ValueError(f"Null values found in columns: {null_check[null_check > 0].index.tolist()}")

    # Check for infinite values
    inf_columns = (
        performance_features_df.columns[
            performance_features_df.isin([np.inf, -np.inf]).any()
        ].tolist())
    if inf_columns:
        raise ValueError(f"Infinite values found in columns: {inf_columns}")

    # Check wallet_address index consistency
    if not performance_features_df.index.equals(trading_features_df.index):
        raise ValueError("Wallet address mismatch between trading_features_df and performance_features_df")

    return performance_features_df




# ---------------------------------
#         Helper Functions
# ---------------------------------

def calculate_profits_features(wallet_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates candidate profit profits_features for return calculations.

    Params:
    - wallet_features_df (DataFrame): Required columns:
        - crypto_net_gain: Total gain including unrealized
        - crypto_net_flows: Net realized gain
        - crypto_inflows: Sum of buy transactions
        - crypto_outflows: Sum of sell transactions

    Returns:
    - profits_features_df (DataFrame): Profit profits_features with columns:
        - total_gain: Total gain including unrealized P&L
        - realized_gain: Net cash flows from trading
        - unrealized_gain: Mark-to-market gains not yet realized

    """
    profits_features_df = pd.DataFrame(index=wallet_features_df.index)

    # Primary gain profits_features
    profits_features_df['crypto_net_gain'] = wallet_features_df['crypto_net_gain']
    profits_features_df['crypto_net_flows'] = wallet_features_df['crypto_net_flows']

    # FeatureRemoval values found to be nonpredictive
    # profits_features_df['crypto_net_cash_flows'] = wallet_features_df['crypto_net_cash_flows']

    # Verify no nulls produced
    null_check = profits_features_df.isnull().sum()
    if null_check.any():
        raise ValueError(f"Null values found in columns: {null_check[null_check > 0].index.tolist()}")

    return profits_features_df



def calculate_balance_features(trading_features_df: pd.DataFrame,
                               include_twb_metrics: bool = True) -> pd.DataFrame:
    """
    Generates candidate denominator features capturing different aspects of wallet scale.

    Params:
    - trading_features_df (DataFrame): Required columns:
        - max_investment: Peak portfolio value
        - time_weighted_balance: Average balance including zeros
        - active_time_weighted_balance: Average non-zero balance
        - activity_density: Trading frequency
        - transaction_days: Days with activity
        - average_transaction: Mean transaction size
    - include_twb_metrics (bool): whether to include 'twb' and 'active_twb'
        as balance features

    Returns:
    - balance_df (DataFrame): Scale features with columns:
        Size features:
        - max_investment: Peak portfolio value
        - time_weighted_balance: Average including zeros
        - active_time_weighted_balance: Average excluding zeros

        Hybrid size/activity features:
        - activity_weighted_balance: TWB * activity_density
        - transaction_weighted_balance: TWB * transaction_count
        - velocity_weighted_balance: TWB * (volume/twb ratio)

        Risk exposure features:
        - peak_exposure: max(buy_volume, sell_volume)
        - sustained_exposure: twb * sqrt(transaction_days)
        - turnover_exposure: total_volume * (twb/max_investment)
    """
    balance_features_df = pd.DataFrame(index=trading_features_df.index)

    # Basic size features
    balance_features_df['max_investment'] = trading_features_df['max_investment']
    balance_features_df['crypto_inflows'] = trading_features_df['crypto_inflows']

    # FeatureRemoval values found to be nonpredictive
    # balance_features_df['crypto_cash_buys'] = trading_features_df['crypto_cash_buys']

    # Add twb metrics if configured to
    if include_twb_metrics:
        balance_features_df['twb'] = trading_features_df['time_weighted_balance']
        balance_features_df['active_twb'] = trading_features_df['active_time_weighted_balance']

    # Verify no nulls produced
    null_check = balance_features_df.isnull().sum()
    if null_check.any():
        raise ValueError(f"Null values found in columns: {null_check[null_check > 0].index.tolist()}")

    return balance_features_df



def calculate_performance_ratios(performance_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates ratio columns for each profits metric divided by each balance metric.

    Params:
    - performance_features_df (DataFrame): Combined profits and balance metrics

    Returns:
    - performance_ratios_df (DataFrame): DataFrame with columns for each profits/balance ratio
    """
    # Extract profits and balance columns using string matching
    profit_cols = [col for col in performance_features_df.columns if col.startswith('profits_')]
    balance_cols = [col for col in performance_features_df.columns if col.startswith('balance_')]

    # Initialize empty DataFrame with same index
    performance_ratios_df = pd.DataFrame(index=performance_features_df.index)

    # Vectorized calculation of all ratios
    for p_col in profit_cols:
        for b_col in balance_cols:
            # Create ratio column name
            p_feature = p_col.replace('profits_', '')
            b_feature = b_col.replace('balance_', '')
            ratio_name = f'{p_feature}/{b_feature}'

            # Calculate ratio
            performance_ratios_df[ratio_name] = np.where(
                performance_features_df[b_col] == 0, 0,
                performance_features_df[p_col] / performance_features_df[b_col],
            )
    if performance_ratios_df.isin([np.inf, -np.inf]).any().any():
        print("Infinite values found in ratio columns")
    if performance_ratios_df.isnull().any().any():
        print("NaN values found in ratio columns")

    return performance_ratios_df



def transform_performance_ratios(performance_ratios_df: pd.DataFrame,
                                 balance_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies transformations to performance ratios.

    Params:
    - performance_ratios_df (DataFrame): Raw performance ratios
    - balance_features_df (DataFrame): Balance features for ntile calculation

    Returns:
    - performance_features_df (DataFrame): For each input ratio creates:
        - {ratio}_base: unmodified ratio
        - {ratio}_rank: the wallet's rank out of all wallets for the ratio
        - {ratio}_log: signed log of the ratio
        - {ratio}_winsorized: winsorized ratio using the config param 'returns_winsorization'
        - {ratio}_ntile_rank: wallets ranking within its balance_feature ntile, with the ntile
            count set in config param 'ranking_ntiles'

    """
    # Extract params from config
    ntile_count = wallets_config['features']['ranking_ntiles']
    returns_winsorization = wallets_config['features']['returns_winsorization']

    # Create complete index in empty df
    performance_features_df = pd.DataFrame(index=performance_ratios_df.index)

    # For each ratio column, calculate multiple feature columns using transformations
    for col in performance_ratios_df.columns:

        series = performance_ratios_df[col]

        # Unmodified ratio
        performance_features_df[f'{col}/base'] = series

        # Rank of ratio
        performance_features_df[f'{col}/rank'] = series.rank(method='average', pct=True)

        # Signed log of ratio
        performance_features_df[f'{col}/log'] = np.sign(series) * np.log1p(series.abs())

        # Winsorized ratio
        performance_features_df[f'{col}/winsorized'] = u.winsorize(series,returns_winsorization)

        # Ntile rank of ratio
        # Extract denominator and create ntils of balance values
        denominator = col.split('/')[1]
        balance_col = f'balance_{denominator}'
        metric_ntiles = pd.qcut(
            balance_features_df[balance_col],
            q=ntile_count,
            labels=False,
            duplicates='drop'
        )

        # Rank within appropriate denominator-based groups
        performance_features_df[f'{col}/ntile_rank'] = (
            series.groupby(metric_ntiles)
            .rank(method='average', pct=True)
            .fillna(0)
        )

    return performance_features_df
