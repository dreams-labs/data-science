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


def calculate_coin_end_balance_features(modeling_profits_df: pd.DataFrame, wallet_scores_df: pd.DataFrame
                                         ) -> pd.DataFrame:
    """
    Calculates coin-level metrics based on wallet behavior and scores at the end of the modeling period.

    Params:
    - modeling_profits_df (DataFrame): Profits data with coin_id, wallet_address, date, usd_balance columns
    - wallet_scores_df (DataFrame): Wallet scores data indexed by wallet_address with score column
    - wallets_config (dict): Config containing modeling_period_end date
    - wallets_coin_config (dict): Config containing top_wallets_cutoff threshold

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
    # 1. Combine wallet-coin pair period end balances and scores
    # ----------------------------------------------------------
    # identify balances at the end of the modeling period
    modeling_end_date = pd.to_datetime(wallets_config['training_data']['modeling_period_end'])
    modeling_end_balances_df = modeling_profits_df[modeling_profits_df['date']==modeling_end_date].copy()
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
                                'balance_wtd_mean_score': safe_weighted_average(
                                x['score'].values,
                                x['usd_balance'].values)
                            })
                        ,include_groups=False))
    coin_wallet_features_df = coin_wallet_features_df.join(weighted_scores_df,how='inner')


    # Calculate total metrics for all wallets in profits_df
    all_wallets_metrics = analysis_df.groupby('coin_id',observed=True).agg(
        all_cohort_wallets_balance=('usd_balance', 'sum'),
        all_cohort_wallets_count=('wallet_address', 'count'),
        all_cohort_wallets_mean_score=('score', 'mean')
    )
    coin_wallet_features_df = coin_wallet_features_df.join(all_wallets_metrics,how='inner')


    # Top wallet concentration metrics
    top_wallets_cutoff = wallets_coin_config['features']['top_wallets_cutoff']
    high_score_threshold = wallet_scores_df['score'].quantile(1 - top_wallets_cutoff)
    top_wallet_metrics = analysis_df[analysis_df['score'] >= high_score_threshold].groupby('coin_id',observed=True).agg(
        top_wallets_balance=('usd_balance', 'sum'),
        top_wallets_count=('wallet_address', 'count'),
        top_wallets_mean_score=('score', 'mean')
    )
    fill_values = {
        'top_wallets_balance': 0,
        'top_wallets_count': 0,
        'top_wallets_mean_score': -1  # indicates there are no scores rather than a low avg
    }
    coin_wallet_features_df = coin_wallet_features_df.join(
        top_wallet_metrics,
        how='left'
    ).fillna(fill_values)


    # Calculate relative percentages
    coin_wallet_features_df['top_wallets_balance_pct'] = (coin_wallet_features_df['top_wallets_balance']
                                                        / coin_wallet_features_df['all_cohort_wallets_balance'])

    coin_wallet_features_df['top_wallets_count_pct'] = (coin_wallet_features_df['top_wallets_count']
                                                        / coin_wallet_features_df['all_cohort_wallets_count'])


    # Validation: check if any coin_ids missing from final features
    missing_coins = set(analysis_df['coin_id']) - set(coin_wallet_features_df.index)
    if missing_coins:
        raise ValueError(f"Found {len(missing_coins)} coin_ids in analysis_df that are missing from features")

    return coin_wallet_features_df
