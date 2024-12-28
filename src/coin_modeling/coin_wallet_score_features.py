import logging
import pandas as pd
import numpy as np


# local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig
import utils as u


# pylint:disable=invalid-name  # X doesn't conform to snake case

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()



def calculate_coin_metrics_from_wallet_scores(validation_profits_df, wallet_scores_df, validation_market_data_df):
    """
    Consolidates wallet scores and metrics to coin-level metrics, creating a comprehensive
    metrics system that considers wallet quality, balance distribution, and activity levels.

    Parameters
    ----------
    validation_profits_df : pandas.DataFrame
        profits_df covering the full course of the validation period, containing columns:
        - wallet_address: Unique identifier for each wallet
        - coin_id: Identifier for each cryptocurrency
        - usd_balance: Balance in USD for each wallet-coin combination

    wallet_scores_df : pandas.DataFrame
        DataFrame indexed by wallet_address containing wallet quality scores with columns:
        - score: Quality score for each wallet (0 to 1)

    Returns
    -------
    coin_wallet_metrics_df : pandas.DataFrame
        DataFrame indexed by coin_id containing the following metrics:
        - weighted_avg_score: Average wallet score weighted by USD balance
        - top_wallet_balance: Total USD balance held by top 20% scoring wallets
        - top_wallet_count: Number of wallets in the top 20% by score
        - total_balance: Total USD balance across all wallets
        - total_wallets: Total number of unique wallets
        - mean_score: Simple average of wallet scores
        - score_std: Standard deviation of wallet scores
        - score_count: Number of scored wallets
        - top_wallet_balance_pct: Percentage of total balance held by top wallets
        - top_wallet_count_pct: Percentage of wallets that are top scored
        - composite_score: Combined metric (40% weighted score, 30% balance concentration, 30% wallet concentration)
        - avg_wallet_balance: Average USD balance per wallet
        - score_confidence: Confidence metric based on number of scored wallets

    Notes
    -----
    - Filters out coins with fewer than 5 wallets or less than $10,000 total balance
    - Handles negative balances by clipping to 0
    - Fills missing scores with 0
    - Returns results sorted by composite_score in descending order
    """

    # 1. Combine and filter metrics to create base analysis df
    # --------------------------------------------------------
    # identify balances at start of validation period
    validation_start_date = pd.to_datetime(wallets_config['training_data']['validation_period_start'])
    validation_start_df = validation_profits_df[validation_profits_df['date']==validation_start_date].copy()
    validation_start_df = validation_start_df[['coin_id','wallet_address','usd_balance']]
    validation_start_df = validation_start_df[validation_start_df['usd_balance']>0]

    # Merge wallet scores with balance data
    analysis_df = validation_start_df.merge(
        wallet_scores_df[['score']],
        left_on='wallet_address',
        right_index=True,
        how='left'
    )

    # Ensure no negative balances and fill any NA scores
    analysis_df['usd_balance'] = analysis_df['usd_balance'].clip(lower=0)
    analysis_df['score'] = analysis_df['score'].fillna(0)

    # 2. Generate coin-level metrics from wallet behavior
    # ---------------------------------------------------
    # Calculate weighted average score differently
    def safe_weighted_average(scores, weights):
        """Calculate weighted average, handling zero weights safely"""
        if np.sum(weights) == 0:
            return np.mean(scores) if len(scores) > 0 else 0
        return np.sum(scores * weights) / np.sum(weights)

    weighted_scores = analysis_df.groupby('coin_id').apply(
        lambda x: safe_weighted_average(x['score'].values, x['usd_balance'].values)
    ,include_groups=False).reset_index()
    weighted_scores.columns = ['coin_id', 'weighted_avg_score']

    # Top wallet concentration
    top_wallets_cutoff = wallets_config['coin_validation_analysis']['top_wallets_cutoff']
    high_score_threshold = wallet_scores_df['score'].quantile(1 - top_wallets_cutoff)
    top_wallet_metrics = analysis_df[analysis_df['score'] >= high_score_threshold].groupby('coin_id').agg({
        'usd_balance': 'sum',
        'wallet_address': 'count'
    }).reset_index()
    top_wallet_metrics.columns = ['coin_id', 'top_wallet_balance', 'top_wallet_count']

    # Calculate total metrics
    total_metrics = analysis_df.groupby('coin_id').agg({
        'usd_balance': 'sum',
        'wallet_address': 'count',
        'score': ['mean', 'std']
    }).reset_index()
    total_metrics.columns = ['coin_id', 'total_balance', 'total_wallets', 'mean_score', 'score_std']

    # Combine metrics
    coin_wallet_metrics_df = pd.merge(weighted_scores, top_wallet_metrics, on='coin_id', how='left')
    coin_wallet_metrics_df = pd.merge(coin_wallet_metrics_df, total_metrics, on='coin_id', how='left')

    # Set index
    coin_wallet_metrics_df=coin_wallet_metrics_df.set_index('coin_id')

    # Fill NaN values
    fill_columns = ['top_wallet_balance', 'top_wallet_count', 'score_std']
    coin_wallet_metrics_df[fill_columns] = coin_wallet_metrics_df[fill_columns].fillna(0)

    # Calculate percentages safely
    coin_wallet_metrics_df['top_wallet_balance_pct'] = np.where(
        coin_wallet_metrics_df['total_balance'] > 0,
        coin_wallet_metrics_df['top_wallet_balance'] / coin_wallet_metrics_df['total_balance'],
        0
    )

    coin_wallet_metrics_df['top_wallet_count_pct'] = np.where(
        coin_wallet_metrics_df['total_wallets'] > 0,
        coin_wallet_metrics_df['top_wallet_count'] / coin_wallet_metrics_df['total_wallets'],
        0
    )

    # Create composite score
    coin_wallet_metrics_df['composite_score'] = (
        coin_wallet_metrics_df['weighted_avg_score'] * 0.4 +
        coin_wallet_metrics_df['top_wallet_balance_pct'] * 0.3 +
        coin_wallet_metrics_df['top_wallet_count_pct'] * 0.3
    )

    # Additional metrics
    coin_wallet_metrics_df['avg_wallet_balance'] = (coin_wallet_metrics_df['total_balance']
                                                    / coin_wallet_metrics_df['total_wallets'])
    coin_wallet_metrics_df['score_confidence'] = 1 - (
        1 / np.sqrt(coin_wallet_metrics_df['total_wallets'] + 1))  # Added +1 to avoid division by zero


    # Append the market cap at the end of the modeling period
    modeling_end_market_cap_df = validation_market_data_df[validation_market_data_df['date']
                                            ==wallets_config['training_data']['validation_starting_balance_date']]

    modeling_end_market_cap_df = modeling_end_market_cap_df.set_index('coin_id')
    modeling_end_market_cap_df = modeling_end_market_cap_df[['market_cap','market_cap_imputed','market_cap_filled']]
    coin_wallet_metrics_df = coin_wallet_metrics_df.join(modeling_end_market_cap_df, how='inner')


    # 3. Apply filters based on wallets_config
    # ----------------------------------------
    # Log initial count
    initial_count = len(coin_wallet_metrics_df)
    logger.info("Starting coin count: %d", initial_count)

    # Filter for minimum activity
    min_wallets = wallets_config['coin_validation_analysis']['min_wallets']
    min_balance = wallets_config['coin_validation_analysis']['min_balance']

    # Apply wallet threshold and log
    wallets_filtered_df = coin_wallet_metrics_df[coin_wallet_metrics_df['total_wallets'] >= min_wallets]
    wallets_removed = initial_count - len(wallets_filtered_df)
    logger.info(
        "Removed %d coins (%.1f%%) with fewer than %d wallets",
        wallets_removed,
        (wallets_removed/initial_count)*100,
        min_wallets
    )

    # Apply balance threshold and log
    coin_wallet_metrics_df = wallets_filtered_df[wallets_filtered_df['total_balance'] >= min_balance]
    balance_removed = len(wallets_filtered_df) - len(coin_wallet_metrics_df)
    logger.info(
        "Removed %d coins (%.1f%%) with balance below %d",
        balance_removed,
        (balance_removed/initial_count)*100,
        min_balance
    )

    # Log final count
    logger.info(
        "Final coin count after all filters: %d (%.1f%% of initial)",
        len(coin_wallet_metrics_df),
        (len(coin_wallet_metrics_df)/initial_count)*100
    )

    return coin_wallet_metrics_df
