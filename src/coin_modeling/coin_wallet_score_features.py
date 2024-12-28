import logging
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


def calculate_coin_balance_features(modeling_profits_df: pd.DataFrame,
                                    wallet_scores_df: pd.DataFrame,
                                    balance_date: str
                                         ) -> pd.DataFrame:
    """
    Calculates coin-level metrics based on wallet behavior and scores as of a specific date.

    Params:
    - modeling_profits_df (DataFrame): Profits data with coin_id, wallet_address, date, usd_balance columns
    - wallet_scores_df (DataFrame): Wallet scores data indexed by wallet_address with score column
    - balance_date (str): The date one which balance distributions should be analyzed

    Returns:
    - coin_wallet_features_df (DataFrame): Coin-level features including:
        - balance_wtd_mean_score: Wallet scores weighted by USD balance
        - all_cohort_wallets_*: Metrics across all wallets (balance, count, mean score)
        - top_wallets_*: Metrics for high-scoring wallets only
        - *_pct: Relative metrics comparing top wallets to all wallets
        Indexed by coin_id.

    Raises:
    - ValueError: If any coin_ids are missing from final features dataframe

    """
    # 1. Combine wallet-coin pair period balances and scores
    # ----------------------------------------------------------
    # identify balances as of the balance_date
    balance_date = pd.to_datetime(balance_date)
    modeling_end_balances_df = modeling_profits_df[modeling_profits_df['date']==balance_date].copy()
    modeling_end_balances_df = modeling_end_balances_df[['coin_id','wallet_address','usd_balance']]
    modeling_end_balances_df = modeling_end_balances_df[modeling_end_balances_df['usd_balance']>0]

    # Merge wallet scores with balance data
    analysis_df = modeling_end_balances_df.merge(
        wallet_scores_df[['score']],
        left_on='wallet_address',
        right_index=True,
        how='left'
    )

    # 2. Generate coin-level metrics from wallet behavior
    # ---------------------------------------------------
    # Initialize output dataframe with an index for every coin_id with an end balance
    coin_wallet_features_df = pd.DataFrame(index=analysis_df['coin_id'].unique())
    coin_wallet_features_df.index.name = 'coin_id'


    # Balance-weighted average score
    def safe_weighted_average(scores, weights):
        """Calculate weighted average, handling zero weights safely"""
        if np.sum(weights) == 0:
            return np.mean(scores) if len(scores) > 0 else 0
        return np.sum(scores * weights) / np.sum(weights)

    weighted_scores_df = (analysis_df.groupby('coin_id', observed=True)
                        .apply(
                            lambda x: pd.Series({
                                'all_cohort_wallets/balance_wtd_mean_score': safe_weighted_average(
                                x['score'].values,
                                x['usd_balance'].values)
                            })
                        ,include_groups=False))
    coin_wallet_features_df = coin_wallet_features_df.join(weighted_scores_df,how='inner')


    # Calculate total metrics for all wallets in profits_df
    all_wallets_metrics = analysis_df.groupby('coin_id',observed=True).agg(
        balance=('usd_balance', 'sum'),
        count=('wallet_address', 'count'),
        mean_score=('score', 'mean')
    ).rename(columns={
        'balance': 'all_cohort_wallets/balance',
        'count': 'all_cohort_wallets/count',
        'mean_score': 'all_cohort_wallets/mean_score'
    })
    coin_wallet_features_df = coin_wallet_features_df.join(all_wallets_metrics,how='inner')


    # Top wallet concentration metrics
    top_wallets_cutoff = wallets_coin_config['features']['top_wallets_cutoff']
    high_score_threshold = wallet_scores_df['score'].quantile(1 - top_wallets_cutoff)
    top_wallet_metrics = analysis_df[analysis_df['score'] >= high_score_threshold].groupby('coin_id',observed=True).agg(
        balance=('usd_balance', 'sum'),
        count=('wallet_address', 'count'),
        mean_score=('score', 'mean')
    ).rename(columns={
        'balance': 'top_wallets/balance',
        'count': 'top_wallets/count',
        'mean_score': 'top_wallets/mean_score'
    })
    fill_values = {
        'top_wallets/balance': 0,
        'top_wallets/count': 0,
        'top_wallets/mean_score': -1  # indicates there are no scores rather than a low avg
    }
    coin_wallet_features_df = coin_wallet_features_df.join(
        top_wallet_metrics,
        how='left'
    ).fillna(fill_values)


    # Calculate relative percentages
    coin_wallet_features_df['top_wallets/balance_pct'] = (coin_wallet_features_df['top_wallets/balance']
                                                        / coin_wallet_features_df['all_cohort_wallets/balance'])

    coin_wallet_features_df['top_wallets/count_pct'] = (coin_wallet_features_df['top_wallets/count']
                                                        / coin_wallet_features_df['all_cohort_wallets/count'])


    # Validation: check if any coin_ids missing from final features
    missing_coins = set(analysis_df['coin_id']) - set(coin_wallet_features_df.index)
    if missing_coins:
        raise ValueError(f"Found {len(missing_coins)} coin_ids in analysis_df that are missing from features")

    return coin_wallet_features_df
