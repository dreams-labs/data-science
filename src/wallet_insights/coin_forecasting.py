"""
Orchestrates groups of functions to generate wallet model pipeline
"""

import logging
import pandas as pd
import numpy as np
from dreams_core import core as dc

# Local module imports
import wallet_features.wallet_coin_date_features as wcdf
from wallet_modeling.wallets_config_manager import WalletsConfig

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()



def calculate_coin_metrics_from_wallet_scores(validation_profits_df, wallet_scores_df):
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
    ).reset_index()
    weighted_scores.columns = ['coin_id', 'weighted_avg_score']

    # Top wallet concentration
    high_score_threshold = wallet_scores_df['score'].quantile(0.8)
    top_wallet_metrics = analysis_df[analysis_df['score'] >= high_score_threshold].groupby('coin_id').agg({
        'usd_balance': 'sum',
        'wallet_address': 'count'
    }).reset_index()
    top_wallet_metrics.columns = ['coin_id', 'top_wallet_balance', 'top_wallet_count']

    # Calculate total metrics
    total_metrics = analysis_df.groupby('coin_id').agg({
        'usd_balance': 'sum',
        'wallet_address': 'count',
        'score': ['mean', 'std', 'count']
    }).reset_index()
    total_metrics.columns = ['coin_id', 'total_balance', 'total_wallets',
                           'mean_score', 'score_std', 'score_count']

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
        1 / np.sqrt(coin_wallet_metrics_df['score_count'] + 1))  # Added +1 to avoid division by zero


    # 3. Apply filters based on wallets_config
    # ----------------------------------------
    # Log initial count
    initial_count = len(coin_wallet_metrics_df)
    logger.info("Starting coin count: %d", initial_count)

    # Filter for minimum activity
    min_wallets = wallets_config['coin_forecasting']['min_wallets']
    min_balance = wallets_config['coin_forecasting']['min_balance']

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



def calculate_coin_performance(market_data_df, start_date, end_date):
    """
    Calculates coin performance metrics over a specified period.

    Params:
    - market_data_df (df): dataframe containing market data that includes the relevant dates
    - start_date (str): 'YYYY-MM-DD' string that defines coin starting price/market cap values
    - start_date (str): 'YYYY-MM-DD' string that defines coin end price values

    Returns:
    - coin_performance_df (df): df indexed on coin_id that includes columns:
        starting_price: price on start_date
        ending_price: price on end_date
        coin_return: price change between start and end date
        market_cap: unmodified market_cap
        market_cap_filled: market cap with 100% coverage
    """
    # Convert dates to datetime
    start_date = pd.to_datetime(wallets_config['training_data']['validation_period_start'])
    end_date = pd.to_datetime(wallets_config['training_data']['validation_period_end'])

    # Get all required data for start date in one operation
    start_data = market_data_df[market_data_df['date'] == start_date].set_index('coin_id')[['price', 'market_cap']]
    end_data = market_data_df[market_data_df['date'] == end_date].set_index('coin_id')[['price']]

    # Fill market cap
    market_data_filled_df = wcdf.force_fill_market_cap(market_data_df)
    start_market_cap_filled = market_data_filled_df[
        market_data_filled_df['date'] == start_date
    ].set_index('coin_id')['market_cap_filled']

    # Create consolidated dataframe
    coin_performance_df = pd.DataFrame({
        'starting_price': start_data['price'],
        'ending_price': end_data['price'],
        'market_cap': start_data['market_cap'],
        'market_cap_filled': start_market_cap_filled
    })

    # Remove coins with zero starting price
    coin_performance_df = coin_performance_df[coin_performance_df['starting_price'] > 0]

    # Calculate returns
    coin_performance_df['coin_return'] = (coin_performance_df['ending_price']
                                        / coin_performance_df['starting_price']) - 1

    # Drop price columns
    coin_performance_df = coin_performance_df.drop(['starting_price','ending_price'], axis=1)

    return coin_performance_df



def validate_coin_performance(coin_performance_df, top_n, max_market_cap, min_market_cap):
    """
    For each metric in the dataframe, analyze return performance of top n coins sorted by that metric,
    as well as performance across all coins.

    Params:
    - coin_performance_df (df): dataframe indexed on coin_id with aggregated wallet metrics
    - top_n (int): assess the returns of the top n coins, sorted by each of the metrics
    - max_market_cap (int): coins above this market cap will not be included in forecasts
    - min_market_cap (int): coins below this market cap will not be included in forecasts

    Returns:
    - metric_top_n_returns_df (df): dataframe showing return metrics for the top_n coins when sorted by
        each wallet aggregation column, plus an "all_coins" row showing metrics across all coins
    """
    # Log initial count
    initial_count = len(coin_performance_df)
    logger.info("Starting coin count: %d", initial_count)

    # Count coins above and below thresholds before filtering
    above_max = len(coin_performance_df[coin_performance_df['market_cap_filled'] > max_market_cap])
    below_min = len(coin_performance_df[coin_performance_df['market_cap_filled'] < min_market_cap])

    logger.info("Found %d coins (%.1f%%) above maximum market cap %s",
        above_max,(above_max/initial_count)*100,dc.human_format(max_market_cap))

    logger.info("Found %d coins (%.1f%%) below minimum market cap %s",
        below_min,(below_min/initial_count)*100,dc.human_format(min_market_cap))

    # Filter based on market cap thresholds
    coin_performance_df = coin_performance_df[
        (coin_performance_df['market_cap_filled'] <= max_market_cap)
        & (coin_performance_df['market_cap_filled'] >= min_market_cap)
    ].copy()

    # Log final results
    filtered_count = len(coin_performance_df)
    logger.info(
        "Final coin count after market cap filter: %d (%.1f%% of initial)",
        filtered_count,
        (filtered_count/initial_count)*100
    )
    # Skip these columns as they're not useful ranking metrics
    skip_columns = ['coin_return', 'coin_id', 'market_cap', 'market_cap_filled']

    metric_top_n_returns = {}

    # Calculate performance for each metric
    for column in coin_performance_df.columns:
        if column not in skip_columns:
            # Sort by metric and get top n coins
            top_n_df = coin_performance_df.sort_values(column, ascending=False).head(top_n)

            # Calculate average return
            avg_return = top_n_df['coin_return'].mean()
            median_return = top_n_df['coin_return'].median()
            min_return = top_n_df['coin_return'].min()
            max_return = top_n_df['coin_return'].max()

            metric_top_n_returns[column] = {
                'mean_return': avg_return,
                'median_return': median_return,
                'min_return': min_return,
                'max_return': max_return
            }

    # Add metrics for all coins
    metric_top_n_returns['all_coins'] = {
        'mean_return': coin_performance_df['coin_return'].mean(),
        'median_return': coin_performance_df['coin_return'].median(),
        'min_return': coin_performance_df['coin_return'].min(),
        'max_return': coin_performance_df['coin_return'].max()
    }

    # Convert to dataframe
    metric_top_n_returns_df = pd.DataFrame(metric_top_n_returns).T

    # Sort by mean return
    metric_top_n_returns_df = metric_top_n_returns_df.sort_values('mean_return', ascending=False)

    return metric_top_n_returns_df
