import logging
from typing import List
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

# local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Locate the config directory
current_dir = Path(__file__).parent
config_directory = current_dir / '..' / '..' / 'config'

# Load wallets_config at the module level
wallets_config = WalletsConfig()
wallets_coin_config = yaml.safe_load((config_directory / 'wallets_coin_config.yaml').read_text(encoding='utf-8'))  # pylint:disable=line-too-long


def prepare_balance_data(modeling_profits_df: pd.DataFrame,
                        wallet_scores_df: pd.DataFrame,
                        balance_date: str) -> pd.DataFrame:
    """
    Prepares wallet balance and score data for feature calculation.

    Params:
    - modeling_profits_df (DataFrame): Profits data with coin_id, wallet_address, date, usd_balance
    - wallet_scores_df (DataFrame): Wallet scores data indexed by wallet_address with score column
    - balance_date (str): Date for balance analysis

    Returns:
    - analysis_df (DataFrame): Combined wallet balances and scores for the specified date,
                             filtered to positive balances
    """
    # Convert date and filter to balance date
    balance_date = pd.to_datetime(balance_date)
    modeling_balances_df = modeling_profits_df[
        modeling_profits_df['date'] == balance_date
    ].copy()

    # Select required columns and filter to positive balances
    modeling_balances_df = modeling_balances_df[
        ['coin_id', 'wallet_address', 'usd_balance']
    ]
    modeling_balances_df = modeling_balances_df[
        modeling_balances_df['usd_balance'] > 0
    ]

    # Merge wallet scores with balance data
    analysis_df = modeling_balances_df.merge(
        wallet_scores_df[['score']],
        left_on='wallet_address',
        right_index=True,
        how='left'
    )

    return analysis_df



def safe_weighted_average(scores: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculate weighted average, handling zero weights safely.

    Params:
    - scores (ndarray): Array of score values
    - weights (ndarray): Array of weights (e.g. balances)

    Returns:
    - float: Weighted average score, or simple mean if weights sum to 0
    """
    if np.sum(weights) == 0:
        return np.mean(scores) if len(scores) > 0 else 0
    return np.sum(scores * weights) / np.sum(weights)



def calculate_weighted_balance_scores(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate balance-weighted average scores for each coin.

    Params:
    - analysis_df (DataFrame): Combined balance and score data

    Returns:
    - weighted_scores_df (DataFrame): Balance-weighted mean scores by coin_id
    """
    return analysis_df.groupby('coin_id', observed=True).apply(
        lambda x: pd.Series({
            'top_100pct/balance_wtd_mean_score': safe_weighted_average(
                x['score'].values,
                x['usd_balance'].values
            )
        })
    )



def calculate_quantile_metrics(analysis_df: pd.DataFrame,
                            quantile: float) -> pd.DataFrame:
    """
    Calculate metrics for a specific score quantile cohort.

    Params:
    - analysis_df (DataFrame): Combined balance and score data
    - quantile (float, optional): Score quantile threshold (0.0-1.0)
                                If None, calculates for all wallets

    Returns:
    - metrics_df (DataFrame): Standard metrics with automatically generated prefixes:
        - {q_prefix}/balance: Total USD balance
        - {q_prefix}/count: Number of wallets
        - {q_prefix}/mean_score: Mean wallet score
        Where q_prefix is either 'all_wallets' or 'top_Xpct' based on quantile
    """
    # Determine cohort and prefix
    threshold = analysis_df['score'].quantile(1 - quantile)
    df = analysis_df[analysis_df['score'] >= threshold]
    prefix = f'top_{int(quantile * 100)}pct'

    # Calculate metrics
    metrics_df = df.groupby('coin_id', observed=True).agg(
        balance=('usd_balance', 'sum'),
        count=('wallet_address', 'count'),
        mean_score=('score', 'mean')
    )

    # Rename columns with prefix
    metrics_df = metrics_df.rename(columns={
        'balance': f'{prefix}/balance',
        'count': f'{prefix}/count',
        'mean_score': f'{prefix}/mean_score'
    })

    return metrics_df


def calculate_coin_wallet_balance_features(
                profits_df: pd.DataFrame,
                wallet_scores_df: pd.DataFrame,
                balance_date: str,
                all_coin_ids: List[str]) -> pd.DataFrame:
    """
    Calculates coin-level metrics based on wallet behavior and scores as of a specific date.

    Params:
    - profits_df (DataFrame): Profits data with coin_id, wallet_address, date, usd_balance columns
    - wallet_scores_df (DataFrame): Wallet scores data indexed by wallet_address with score column
    - balance_date (str): The date one which balance distributions should be analyzed
    - all_coin_ids (List[str]): Complete list of coins to include in output

    Returns:
    - coin_wallet_balance_features_df (DataFrame): Coin-level features including:
        - balance_wtd_mean_score: Wallet scores weighted by USD balance
        - all_cohort_wallets_*: Metrics across all wallets (balance, count, mean score)
        - top_wallets_*: Metrics for high-scoring wallets only
        - *_pct: Relative metrics comparing top wallets to all wallets
        Indexed by coin_id.

    Raises:
    - ValueError: If any coin_ids are missing from final features dataframe

    """
    # Combine wallet-coin pair period balances and scores
    analysis_df = prepare_balance_data(profits_df, wallet_scores_df, balance_date)

    # Initialize with complete coin list
    coin_wallet_features_df = pd.DataFrame(index=all_coin_ids)
    coin_wallet_features_df.index.name = 'coin_id'

    # Calculate balance weighted average scores
    weighted_scores_df = calculate_weighted_balance_scores(analysis_df)
    coin_wallet_features_df = coin_wallet_features_df.join(
        weighted_scores_df,
        how='left'
    )

    # Retrieve quantiles and perform data quality checks
    quantiles = wallets_coin_config['features']['top_wallets_quantiles']

    if len(quantiles) != len(set(quantiles)):
        raise ValueError("Quantiles must be unique")
    if not all(0 < q <= 1 for q in quantiles):
        raise ValueError("Quantiles must be in range (0, 1]")

    # Add 1.0 and sort in descending order
    quantiles = sorted(quantiles + [1.0], reverse=True)

    # Calculate metrics for each quantile cohort
    for quantile in quantiles:

        # Calculate balances and counts
        metrics_df = calculate_quantile_metrics(analysis_df, quantile)
        prefix = f'top_{int(quantile * 100)}pct'

        # Join to df with all features
        coin_wallet_features_df = coin_wallet_features_df.join(
            metrics_df,
            how='left'
        )

        # Add relative metrics for quantiles < 1.0
        if quantile < 1.0:
            coin_wallet_features_df[f'{prefix}/balance_pct'] = (
                coin_wallet_features_df[f'{prefix}/balance'] / coin_wallet_features_df['top_100pct/balance']
            ).fillna(0)
            coin_wallet_features_df[f'{prefix}/count_pct'] = (
                coin_wallet_features_df[f'{prefix}/count'] / coin_wallet_features_df['top_100pct/count']
            ).fillna(0)

            # Define and apply fill values for this quantile's metrics
            fill_values = {
                f'{prefix}/balance': 0,
                f'{prefix}/count': 0,
                f'{prefix}/balance_pct': 0,
                f'{prefix}/count_pct': 0,
            }
            coin_wallet_features_df = coin_wallet_features_df.fillna(fill_values)

    # Validation
    missing_coins = set(all_coin_ids) - set(coin_wallet_features_df.index)
    if missing_coins:
        raise ValueError(f"Found {len(missing_coins)} coin_ids in all coins list "
                         "missing from analysis_df")

    return coin_wallet_features_df
