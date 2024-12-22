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


def calculate_profitability_features(wallet_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates candidate profit profitability_features for return calculations.

    Params:
    - wallet_features_df (DataFrame): Required columns:
        - crypto_net_gain: Total gain including unrealized
        - net_crypto_investment: Net realized gain
        - total_crypto_buys: Sum of buy transactions
        - total_crypto_sells: Sum of sell transactions

    Returns:
    - profitability_df (DataFrame): Profit profitability_features with columns:
        - total_gain: Total gain including unrealized P&L
        - realized_gain: Net cash flows from trading
        - unrealized_gain: Mark-to-market gains not yet realized
        - buy_volume: Total buy transactions
        - sell_volume: Total sell transactions
        - total_volume: Total transaction volume
        - net_volume: Net transaction flow
    """
    profitability_features_df = pd.DataFrame(index=wallet_features_df.index)

    # Primary gain profitability_features
    profitability_features_df['total_gain'] = wallet_features_df['crypto_net_gain']
    profitability_features_df['realized_gain'] = wallet_features_df['net_crypto_investment']
    profitability_features_df['unrealized_gain'] = (
        wallet_features_df['crypto_net_gain'] -
        wallet_features_df['net_crypto_investment']
    )

    # Volume-based profitability_features
    profitability_features_df['buy_volume'] = wallet_features_df['total_crypto_buys']
    profitability_features_df['sell_volume'] = wallet_features_df['total_crypto_sells']
    profitability_features_df['total_volume'] = wallet_features_df['total_volume']
    profitability_features_df['net_volume'] = profitability_features_df['buy_volume'] - profitability_features_df['sell_volume']

    # Verify no nulls produced
    null_check = profitability_features_df.isnull().sum()
    if null_check.any():
        raise ValueError(f"Null values found in columns: {null_check[null_check > 0].index.tolist()}")

    # Prefix to identify which columns should be tried as numerators
    profitability_features_df = profitability_features_df.add_prefix('profitability_')

    return profitability_features_df



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
    epsilon = 1e-10

    # Basic size features
    balance_features_df['max_investment'] = trading_features_df['max_investment']
    balance_features_df['time_weighted_balance'] = trading_features_df['time_weighted_balance']
    balance_features_df['active_time_weighted_balance'] = trading_features_df['active_time_weighted_balance']

    # Hybrid features combining size and activity
    balance_features_df['activity_weighted_balance'] = (
        trading_features_df['time_weighted_balance'] *
        trading_features_df['activity_density']
    )

    balance_features_df['transaction_weighted_balance'] = (
        trading_features_df['time_weighted_balance'] *
        np.log1p(trading_features_df['transaction_days'])
    )

    balance_features_df['velocity_weighted_balance'] = (
        trading_features_df['time_weighted_balance'] *
        trading_features_df['volume_vs_twb_ratio'].clip(0, 10)  # Cap extreme velocity
    )

    # Risk exposure features
    balance_features_df['peak_exposure'] = np.maximum(
        trading_features_df['total_crypto_buys'],
        trading_features_df['total_crypto_sells']
    )

    balance_features_df['sustained_exposure'] = (
        trading_features_df['time_weighted_balance'] *
        np.sqrt(trading_features_df['transaction_days'])
    )

    balance_features_df['turnover_exposure'] = (
        trading_features_df['total_volume'] *
        (trading_features_df['time_weighted_balance'] /
         (trading_features_df['max_investment'] + epsilon))
    )

    # Verify no nulls produced
    null_check = balance_features_df.isnull().sum()
    if null_check.any():
        raise ValueError(f"Null values found in columns: {null_check[null_check > 0].index.tolist()}")

    return balance_features_df



def calculate_wallet_intelligence_metrics(wallet_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates composite intelligence metrics capturing wallet's predictive capabilities

    Params:
    - wallet_features_df (DataFrame): Required columns from various feature sets:
        Timing: buyer_number, new_coin_buy_counts
        Trading: activity_density, volume_vs_twb_ratio
        Market Cap: volume_wtd_market_cap
        Performance: returns metrics

    Returns:
    - intelligence_df (DataFrame): Composite intelligence metrics including:
        - early_mover_score: Early adoption + successful exits
        - active_trader_score: Trading frequency weighted by success
        - market_navigation_score: Timing of entries/exits
        - capital_efficiency_score: Return per unit of deployed capital
    """
    metrics_df = pd.DataFrame(index=wallet_features_df.index)
    epsilon = 1e-10

    # Early Mover Intelligence
    # Combines early adoption with exit timing success
    metrics_df['early_mover_score'] = (
        (1 / (wallet_features_df['avg_buyer_number'] + epsilon)) *
        np.clip(wallet_features_df['volume_vs_twb_ratio'], 0, 10)
    )

    # Active Trading Intelligence
    # Weight activity by return success to distinguish active winners
    metrics_df['active_trader_score'] = (
        wallet_features_df['activity_density'] *
        np.clip(wallet_features_df['returns'], -1, 3) *
        np.log1p(wallet_features_df['unique_coins_traded'])
    )

    # Market Navigation Intelligence
    # How well they time entries/exits relative to market cap
    metrics_df['market_navigation_score'] = (
        wallet_features_df['volume_wtd_market_cap'] /
        (wallet_features_df['end_portfolio_wtd_market_cap'] + epsilon)
    )

    # Capital Efficiency Intelligence
    # Return generation per unit of deployed capital
    metrics_df['capital_efficiency_score'] = (
        wallet_features_df['total_volume'] /
        (wallet_features_df['active_time_weighted_balance'] + epsilon) *
        np.clip(wallet_features_df['returns'], 0, None)
    )

    # Normalize all scores to 0-1 range
    for col in metrics_df.columns:
        metrics_df[col] = (
            (metrics_df[col] - metrics_df[col].min()) /
            (metrics_df[col].max() - metrics_df[col].min() + epsilon)
        )

    # Create composite score weighted toward early adoption and capital efficiency
    metrics_df['composite_intelligence_score'] = (
        0.35 * metrics_df['early_mover_score'] +
        0.15 * metrics_df['active_trader_score'] +
        0.15 * metrics_df['market_navigation_score'] +
        0.35 * metrics_df['capital_efficiency_score']
    )

    return metrics_df.round(6)



# # def calculate_performance_metrics(wallet_features_df: pd.DataFrame) -> pd.DataFrame:
# #     """
# #     Orchestrates calculation of all performance metrics.

# #     Params:
# #     - wallet_features_df (DataFrame): Must include columns:
# #         max_investment, time_weighted_balance, crypto_net_gain, net_crypto_investment

# #     Returns:
# #     - metrics_df (DataFrame): All performance metrics calculated on both bases
# #     """
# #     # Calculate basic metrics first
# #     basic_metrics = calculate_basic_performance_metrics(wallet_features_df)

# #     # Use basic metrics to calculate enhanced scores
# #     enhanced_metrics = calculate_enhanced_performance_metrics(basic_metrics)

# #     # Combine metrics
# #     return pd.concat([basic_metrics, enhanced_metrics], axis=1)


# def calculate_basic_performance_metrics(wallet_features_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Calculates standard return metrics using max_investment and both TWB versions.

#     Params:
#     - wallet_features_df (DataFrame): Input metrics with required columns:
#         max_investment, time_weighted_balance, active_time_weighted_balance,
#         crypto_net_gain, net_crypto_investment

#     Returns:
#     - basic_metrics_df (DataFrame): Standard performance metrics
#     """
#     metrics_df = wallet_features_df[['max_investment', 'time_weighted_balance',
#                                    'active_time_weighted_balance', 'crypto_net_gain',
#                                    'net_crypto_investment']].copy().round(6)
#     returns_winsorization = 0.02
#     epsilon = 1e-10

#     # Calculate returns on max_investment
#     metrics_df['max_investment_return'] = np.where(
#         abs(metrics_df['max_investment']) == 0, 0,
#         metrics_df['crypto_net_gain'] / metrics_df['max_investment']
#     )

#     metrics_df['max_investment_realized_return'] = np.where(
#         abs(metrics_df['max_investment']) == 0, 0,
#         metrics_df['net_crypto_investment'] / metrics_df['max_investment']
#     )

#     # Calculate returns on total TWB
#     metrics_df['twb_return'] = np.where(
#         abs(metrics_df['time_weighted_balance']) == 0, 0,
#         metrics_df['crypto_net_gain'] / metrics_df['time_weighted_balance']
#     )

#     metrics_df['twb_realized_return'] = np.where(
#         abs(metrics_df['time_weighted_balance']) == 0, 0,
#         metrics_df['net_crypto_investment'] / metrics_df['time_weighted_balance']
#     )

#     # Calculate returns on active TWB
#     metrics_df['active_twb_return'] = np.where(
#         abs(metrics_df['active_time_weighted_balance']) == 0, 0,
#         metrics_df['crypto_net_gain'] / metrics_df['active_time_weighted_balance']
#     )

#     metrics_df['active_twb_realized_return'] = np.where(
#         abs(metrics_df['active_time_weighted_balance']) == 0, 0,
#         metrics_df['net_crypto_investment'] / metrics_df['active_time_weighted_balance']
#     )

#     # Store unwinsorized versions
#     metrics_df['max_investment_return_unwinsorized'] = metrics_df['max_investment_return']
#     metrics_df['twb_return_unwinsorized'] = metrics_df['twb_return']
#     metrics_df['active_twb_return_unwinsorized'] = metrics_df['active_twb_return']

#     # Apply winsorization
#     if returns_winsorization > 0:
#         for col in ['max_investment_return', 'twb_return', 'active_twb_return',
#                    'max_investment_realized_return', 'twb_realized_return',
#                    'active_twb_realized_return']:
#             metrics_df[col] = u.winsorize(metrics_df[col], returns_winsorization)

#     # Calculate normalized versions
#     for prefix in ['max_investment', 'twb', 'active_twb']:
#         metrics_df[f'norm_{prefix}_return'] = (
#             (metrics_df[f'{prefix}_return'] - metrics_df[f'{prefix}_return'].min()) /
#             (metrics_df[f'{prefix}_return'].max() - metrics_df[f'{prefix}_return'].min() + epsilon)
#         )

#     # Normalize investment bases
#     for col, prefix in [('max_investment', 'max_investment'),
#                        ('time_weighted_balance', 'twb'),
#                        ('active_time_weighted_balance', 'active_twb')]:
#         log_value = np.log10(metrics_df[col] + epsilon)
#         metrics_df[f'norm_{prefix}_invested'] = (
#             (log_value - log_value.min()) /
#             (log_value.max() - log_value.min() + epsilon)
#         )

#     return metrics_df



# # @u.timing_decorator
# # def calculate_performance_features(wallet_features_df):
# #     """
# #     Generates various target variables for modeling wallet performance.

# #     Parameters:
# #     - wallet_features_df: pandas DataFrame with columns ['crypto_net_gain', 'max_investment']

# #     Returns:
# #     - DataFrame with additional target variables
# #     """
# #     metrics_df = wallet_features_df[['max_investment','crypto_net_gain','net_crypto_investment']].copy().round(6)
# #     returns_winsorization = wallets_config['modeling']['returns_winsorization']
# #     epsilon = 1e-10

# #     # Calculate base return, including unrealized price change impacts
# #     metrics_df['return'] = np.where(abs(metrics_df['max_investment']) == 0,0,
# #                                     metrics_df['crypto_net_gain'] / metrics_df['max_investment'])

# #     # Calculate realized return, based on actual cash flows only
# #     metrics_df['realized_return'] = np.where(abs(metrics_df['max_investment']) == 0,0,
# #                                     metrics_df['net_crypto_investment'] / metrics_df['max_investment'])

# #     # Apply winsorization
# #     if returns_winsorization > 0:
# #         metrics_df['return_unwinsorized'] = metrics_df['return']
# #         metrics_df['return'] = u.winsorize(metrics_df['return'],returns_winsorization)

# #     # Normalize returns
# #     metrics_df['norm_return'] = (metrics_df['return'] - metrics_df['return'].min()) / \
# #         (metrics_df['return'].max() - metrics_df['return'].min())

# #     # Normalize logged investments
# #     log_invested = np.log10(metrics_df['max_investment'] + epsilon)
# #     metrics_df['norm_invested'] = (log_invested - log_invested.min()) / \
# #         (log_invested.max() - log_invested.min())

# #     # Performance score
# #     return_weight = wallets_config['modeling']['target_var_params']['performance_score_return_weight']
# #     metrics_df['performance_score'] = (return_weight * metrics_df['norm_return'] +
# #                                      (1-return_weight) * metrics_df['norm_invested'])

# #     # Size-adjusted rank
# #     # Create mask for zero values
# #     zero_mask = metrics_df['max_investment'] == 0

# #     # Create quartiles series initialized with 'q0' for zero values
# #     quartiles = pd.Series('q0', index=metrics_df.index)

# #     # Calculate quartiles for non-zero values
# #     non_zero_quartiles = pd.qcut(metrics_df['max_investment'][~zero_mask],
# #                                 q=4,
# #                                 labels=['q1', 'q2', 'q3', 'q4'])

# #     # Assign the quartiles to non-zero values
# #     quartiles[~zero_mask] = non_zero_quartiles

# #     # Calculate size-adjusted rank within each quartile
# #     metrics_df['size_adjusted_rank'] = metrics_df.groupby(quartiles)['return'].rank(pct=True)


# #     # Clean up intermediate columns
# #     cols_to_drop = ['norm_return', 'norm_invested', 'norm_gain', 'net_crypto_investment']
# #     metrics_df = metrics_df.drop(columns=[c for c in cols_to_drop
# #                                         if c in metrics_df.columns])

# #     # Verify all input wallets exist in output
# #     missing_wallets = set(wallet_features_df.index) - set(metrics_df.index)
# #     if missing_wallets:
# #         raise ValueError(f"Found {len(missing_wallets)} wallets in input that are missing from output")

# #     return metrics_df.round(6)



# # TODO: NEEDS EVERY COIN-DATE TO HAVE PRICE DATA TO BE CALCULATED
# # --------------------------
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
