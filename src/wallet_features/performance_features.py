"""
Calculates wallet-level financial performance metrics
"""
import logging
import pandas as pd
import numpy as np
import scipy

# local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()


def calculate_profits_features(wallet_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates candidate profit profits_features for return calculations.

    Params:
    - wallet_features_df (DataFrame): Required columns:
        - crypto_net_gain: Total gain including unrealized
        - net_crypto_investment: Net realized gain
        - total_crypto_buys: Sum of buy transactions
        - total_crypto_sells: Sum of sell transactions

    Returns:
    - profits_df (DataFrame): Profit profits_features with columns:
        - total_gain: Total gain including unrealized P&L
        - realized_gain: Net cash flows from trading
        - unrealized_gain: Mark-to-market gains not yet realized

    """
    profits_features_df = pd.DataFrame(index=wallet_features_df.index)

    # Primary gain profits_features
    profits_features_df['crypto_net_gain'] = wallet_features_df['crypto_net_gain']
    profits_features_df['net_crypto_investment'] = wallet_features_df['net_crypto_investment']

    # DISABLED FEATURES
    # -----------------------------------------------------
    # profits_features_df['unrealized_gain'] = (
    #     wallet_features_df['crypto_net_gain'] -
    #     wallet_features_df['net_crypto_investment']
    # )

    # # Volume-based profits_features
    # profits_features_df['buy_volume'] = wallet_features_df['total_crypto_buys']
    # profits_features_df['sell_volume'] = wallet_features_df['total_crypto_sells']
    # profits_features_df['total_volume'] = wallet_features_df['total_volume']
    # profits_features_df['net_volume'] = profits_features_df['buy_volume'] - profits_features_df['sell_volume']

    # Verify no nulls produced
    null_check = profits_features_df.isnull().sum()
    if null_check.any():
        raise ValueError(f"Null values found in columns: {null_check[null_check > 0].index.tolist()}")

    return profits_features_df



def calculate_balance_features(trading_features_df: pd.DataFrame) -> pd.DataFrame:
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
    balance_features_df['time_weighted_balance'] = trading_features_df['time_weighted_balance']
    balance_features_df['active_time_weighted_balance'] = trading_features_df['active_time_weighted_balance']

    # DISABLED FEATURES
    # -----------------------------------------------------
    # # Hybrid features combining size and activity
    # balance_features_df['activity_weighted_balance'] = (
    #     trading_features_df['time_weighted_balance'] *
    #     trading_features_df['activity_density']
    # )

    # balance_features_df['transaction_weighted_balance'] = (
    #     trading_features_df['time_weighted_balance'] *
    #     np.log1p(trading_features_df['transaction_days'])
    # )

    # balance_features_df['velocity_weighted_balance'] = (
    #     trading_features_df['time_weighted_balance'] *
    #     trading_features_df['volume_vs_twb_ratio'].clip(0, 10)  # Cap extreme velocity
    # )

    # balance_features_df['sustained_exposure'] = (
    #     trading_features_df['time_weighted_balance'] *
    #     np.sqrt(trading_features_df['transaction_days'])
    # )

    # balance_features_df['turnover_exposure'] = (
    #     trading_features_df['total_volume'] *
    #     (trading_features_df['time_weighted_balance'] /
    #      (trading_features_df['max_investment']
    # )

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
            ratio_name = f'performance_{p_feature}_v_{b_feature}'

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
        performance_features_df[f'{col}_base'] = series

        # Rank of ratio
        performance_features_df[f'{col}_rank'] = series.rank(method='average', pct=True)

        # Signed log of ratio
        performance_features_df[f'{col}_log'] = np.sign(series) * np.log1p(series.abs())

        # Winsorized ratio
        performance_features_df[f'{col}_winsorized'] = u.winsorize(series,returns_winsorization)

        # Ntile rank of ratio
        # Extract denominator and create ntils of balance values
        denominator = col.split('_v_')[1]
        balance_col = f'balance_{denominator}'
        metric_ntiles = pd.qcut(
            balance_features_df[balance_col],
            q=ntile_count,
            labels=False,
            duplicates='drop'
        )

        # Rank within appropriate denominator-based groups
        performance_features_df[f'{col}_ntile_rank'] = (
            series.groupby(metric_ntiles)
            .rank(method='average', pct=True)
            .fillna(0)
        )

    return performance_features_df



def calculate_performance_features(trading_features_df):
    """
    Calculates a set of profit numerators, investment denominators, ratio combinations,
    and transformations of ratios to create a matrix of performance scores.
    """
    trading_features_df = trading_features_df.copy()

    # Features reflecting the profit/return rate/cash inflows/inflow
    profits_features_df = calculate_profits_features(trading_features_df)
    profits_features_df = profits_features_df.add_prefix('profits_')

    # Features reflecting the balance/investment/outlays
    balance_features_df = calculate_balance_features(trading_features_df)
    balance_features_df = balance_features_df.add_prefix('balance_')

    # Compute ratios
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




# # IN PROGRESS: needs price data for every coin-date pair to be computed accurately
# # --------------------------------------------------------------------------------
# # the balances of each coin-wallet pair needs to be imputed for every date that the wallet
# # has a transaction in order for this calculation to work correctly against production data.

# def calculate_time_weighted_returns(profits_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Calculates time-weighted returns (TWR) using actual holding period in days.

#     Params:
#     - profits_df (DataFrame): Daily profits data

#     Returns:
#     - twr_df (DataFrame): TWR metrics keyed on wallet_address
#     """
#     profits_df = profits_df.sort_values(['wallet_address', 'coin_id', 'date'])

#     # Calculate holding period returns
#     profits_df['pre_transfer_balance'] = profits_df['usd_balance'] - profits_df['usd_net_transfers']
#     profits_df['prev_balance'] = profits_df.groupby(['wallet_address', 'coin_id'])['usd_balance'].shift()
#     profits_df['days_held'] = profits_df.groupby(['wallet_address', 'coin_id'])['date'].diff().dt.days

#     # Calculate period returns and weights
#     profits_df['period_return'] = np.where(
#         profits_df['usd_net_transfers'] != 0,
#         profits_df['pre_transfer_balance'] / profits_df['prev_balance'],
#         profits_df['usd_balance'] / profits_df['prev_balance']
#     )
#     profits_df['period_return'] = profits_df['period_return'].replace([np.inf, -np.inf], 1).fillna(1)

#     # Weight by holding period duration
#     profits_df['weighted_return'] = (profits_df['period_return'] - 1) * profits_df['days_held']

#     # Get total days for each wallet
#     total_days = profits_df.groupby('wallet_address')['date'].agg(lambda x: (x.max() - x.min()).days)

#     # Calculate TWR using total days held
#     def safe_twr(weighted_returns, wallet):
#         if len(weighted_returns) == 0 or weighted_returns.isna().all():
#             return 0
#         days = max(total_days[wallet], 1)  # Get days for this wallet, minimum 1
#         return weighted_returns.sum() / days

#     # Compute TWR and days_held using vectorized operations
#     twr_df = profits_df.groupby('wallet_address').agg(
#         time_weighted_return=('weighted_return',
#                               lambda x: safe_twr(x, profits_df.loc[x.index, 'wallet_address'].iloc[0])),
#         days_held=('date', lambda x: max((x.max() - x.min()).days, 1))
#     )

#     # Annualize returns
#     twr_df['annualized_twr'] = ((1 + twr_df['time_weighted_return']) ** (365 / twr_df['days_held'])) - 1
#     twr_df = twr_df.replace([np.inf, -np.inf], np.nan)

#     # Winsorize output
#     returns_winsorization = wallets_config['modeling']['returns_winsorization']
#     if returns_winsorization > 0:
#         twr_df['time_weighted_return'] = u.winsorize(twr_df['time_weighted_return'],returns_winsorization)
#         twr_df['annualized_twr'] = u.winsorize(twr_df['annualized_twr'],returns_winsorization)

#     return twr_df
