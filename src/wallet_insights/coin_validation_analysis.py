import logging
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from dreams_core import core as dc

# Local module imports
import wallet_features.performance_features as wpf
import wallet_features.trading_features as wtf
import wallet_features.market_cap_features as wmc
import wallet_insights.wallet_model_evaluation as wime
from wallet_modeling.wallets_config_manager import WalletsConfig

# pylint: disable=unused-variable  # messy stats functions in visualizations


# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()



def calculate_validation_metrics(X_test, y_pred, validation_profits_df, n_buckets=20, method='score_buckets'):
    """
    Params:
    - X_test (DataFrame): test set features with wallet addresses as index
    - y_pred (array): predicted scores from model
    - validation_profits_df (DataFrame): validation period profit data
    - n_buckets (int): number of buckets to split scores into
    - method (str): 'score_buckets' or 'ntiles' for score grouping method

    Returns:
    - wallet_performance_df (DataFrame): wallet-level validation metrics
    - bucketed_wallet_performance_df (DataFrame): grouped metrics by score bucket
    """
    if method not in ['score_buckets', 'ntiles']:
        raise ValueError("method must be either 'score_buckets' or 'ntiles'")

    # Calculate validation period wallet metrics
    validation_profits_df = wtf.add_cash_flow_transfers_logic(validation_profits_df)
    wallet_trading_features_df = wtf.calculate_wallet_trading_features(validation_profits_df)
    validation_wallets_df = wpf.calculate_performance_features(wallet_trading_features_df)

    # Attach validation period performance to modeling period scores
    wallet_performance_df = pd.DataFrame()
    wallet_performance_df['wallet_address'] = X_test.index.values
    wallet_performance_df['score'] = y_pred

    if method == 'score_buckets':
        # Original bucketing logic
        wallet_performance_df['score_rounded'] = (np.ceil(wallet_performance_df['score'].astype(np.float64)
                                                        *n_buckets)/n_buckets).round(2)
    else:
        # Ntile bucketing
        wallet_performance_df['score_rounded'] = pd.qcut(wallet_performance_df['score'],
                                                        n_buckets,
                                                        labels=False,
                                                        duplicates='drop') / (n_buckets-1)
        wallet_performance_df['score_rounded'] = wallet_performance_df['score_rounded'].round(2)

    wallet_performance_df = wallet_performance_df.set_index('wallet_address')
    wallet_performance_df = wallet_performance_df.join(validation_wallets_df, how='left')

    # Rest remains identical
    bucketed_performance_df = wallet_performance_df.groupby('score_rounded').agg(
        wallets=('score', 'count'),
        mean_invested=('max_investment', 'mean'),
        mean_total_net_flows=('total_net_flows', 'mean'),
        median_invested=('max_investment', 'median'),
        median_total_net_flows=('total_net_flows', 'median'),
    )

    bucketed_performance_df['mean_return'] = (bucketed_performance_df['mean_total_net_flows']
                                                    / bucketed_performance_df['mean_invested'])
    bucketed_performance_df['median_return'] = (bucketed_performance_df['median_total_net_flows']
                                                        / bucketed_performance_df['median_invested'])

    return wallet_performance_df, bucketed_performance_df



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
    total_metrics.columns = ['coin_id', 'total_balance', 'total_wallets',
                           'mean_score', 'score_std']

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
    market_data_filled_df = wmc.force_fill_market_cap(market_data_df)
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


# pylint:disable=dangerous-default-value
def analyze_market_cap_segments(
    df: pd.DataFrame,
    market_cap_ranges: List[Tuple[float, float]] = [
        (0.1e6, 1e6),
        # (1e6, 15e6),
        (1e6, 35e6),
        (35e6, 100e6),
        (100e6, 1e10)
    ],
    top_n: int = 10
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Analyze metric performance across different market cap segments.

    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe containing coin metrics and performance data
    market_cap_ranges : List[Tuple[float, float]]
        List of (min, max) market cap boundaries for each segment
    top_n : int
        Number of top coins to analyze for each metric

    Returns:
    --------
    segment_results : Dict[str, pd.DataFrame]
        Dictionary mapping segment names to performance DataFrames
    summary_df : pd.DataFrame
        Summary DataFrame comparing metric performance across segments
    """
    if len(df) == 0:
        raise ValueError("Input DataFrame is empty")

    # Create segment labels
    segment_labels = [
        f"Segment {i+1} ({dc.human_format(low)}-{dc.human_format(high)})"
        for i, (low, high) in enumerate(market_cap_ranges)
    ]

    # Initialize results
    segment_results = {}
    summary_data = []

    # Analyze each segment
    for i, (low, high) in enumerate(market_cap_ranges):
        # Filter data for this segment
        segment_df = df[
            (df['market_cap_filled'] >= low) &
            (df['market_cap_filled'] < high)
        ].copy()

        if len(segment_df) == 0:
            logging.warning(f"No coins found in segment {i+1} ({dc.human_format(low)}-{dc.human_format(high)})")
            continue

        # Run validation for this segment
        segment_results_df = validate_coin_performance(
            df,
            top_n=min(top_n, len(segment_df)),  # Ensure top_n doesn't exceed segment size
            max_market_cap=float(high),
            min_market_cap=float(low)
        )

        # Store results
        segment_name = segment_labels[i]
        segment_results[segment_name] = segment_results_df

        # Add to summary data
        for metric in segment_results_df.index:
            summary_data.append({
                'Segment': segment_name,
                'Metric': metric,
                'Mean Return': segment_results_df.loc[metric, 'mean_return'],
                'Median Return': segment_results_df.loc[metric, 'median_return'],
                'Min Return': segment_results_df.loc[metric, 'min_return'],
                'Max Return': segment_results_df.loc[metric, 'max_return'],
                'Market Cap Range': f"{dc.human_format(low)}-{dc.human_format(high)}",
                'Segment Size': len(segment_df)
            })

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Add segment performance rankings
    summary_df['Segment Rank'] = summary_df.groupby('Segment')['Mean Return'].rank(ascending=False)

    # Add overall metric consistency score (average rank across segments)
    metric_consistency = summary_df.groupby('Metric')['Segment Rank'].mean()
    summary_df = summary_df.merge(
        metric_consistency.rename('Overall Consistency').reset_index(),
        on='Metric'
    )

    return segment_results, summary_df


def print_segment_analysis(
    segment_results: Dict[str, pd.DataFrame],
    summary_df: pd.DataFrame
) -> None:
    """Print formatted analysis of segment results."""
    # Print segment-by-segment analysis
    print("=== Market Cap Segment Analysis ===\n")

    for segment, results_df in segment_results.items():
        print(f"\n{segment}")
        print("-" * len(segment))

        # Print segment size
        segment_size = summary_df[summary_df['Segment'] == segment]['Segment Size'].iloc[0]
        print(f"Number of coins: {segment_size}")

        # Get top 5 performing metrics for this segment
        top_metrics = results_df.head().index.tolist()
        print("\nTop performing metrics:")
        for i, metric in enumerate(top_metrics, 1):
            mean_return = results_df.loc[metric, 'mean_return'] * 100
            print(f"{i}. {metric}: {mean_return:.1f}%")

    # Print overall metric consistency
    print("\n=== Overall Metric Consistency ===")
    print("(Lower score indicates more consistent performance across segments)\n")

    consistency_summary = summary_df.groupby('Metric')['Overall Consistency'].first()
    consistency_summary = consistency_summary.sort_values()

    for metric, score in consistency_summary.head().items():
        print(f"{metric}: {score:.2f}")


def plot_segment_heatmap(
    summary_df: pd.DataFrame,
    figsize=(12, 8),
    cmap='YlOrRd',
    return_fig=False
):
    """
    Create a heatmap visualization of metric performance across market cap segments.

    Parameters:
    -----------
    segment_results : Dict[str, pd.DataFrame]
        Output from analyze_market_cap_segments
    summary_df : pd.DataFrame
        Summary DataFrame from analyze_market_cap_segments
    figsize : tuple
        Figure size (width, height)
    cmap : str
        Colormap to use for the heatmap
    return_fig : bool
        If True, return the figure object instead of displaying

    Returns:
    --------
    fig : matplotlib.figure.Figure, optional
        The figure object if return_fig is True
    """
    # Pivot the summary data for the heatmap
    heatmap_data = summary_df.pivot(
        index='Metric',
        columns='Segment',
        values='Mean Return'
    ) * 100  # Convert to percentage

    # Sort metrics by overall performance
    metric_means = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.loc[metric_means.sort_values(ascending=False).index]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap=cmap,
        center=0,
        ax=ax,
        cbar_kws={'label': 'Mean Return (%)'}
    )

    # Customize appearance
    plt.title('Metric Performance by Market Cap Segment', pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Add sample sizes
    segment_sizes = summary_df.groupby('Segment')['Segment Size'].first()
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels([f'n={n}' for n in segment_sizes], rotation=45, ha='left')

    # Adjust layout
    plt.tight_layout()

    if return_fig:
        return fig
    plt.show()



def plot_metric_consistency(
    summary_df: pd.DataFrame,
    figsize=(10, 6),
    return_fig=False
):
    """
    Create a secondary visualization showing metric consistency across segments.

    Parameters:
    -----------
    summary_df : pd.DataFrame
        Summary DataFrame from analyze_market_cap_segments
    figsize : tuple
        Figure size (width, height)
    return_fig : bool
        If True, return the figure object instead of displaying

    Returns:
    --------
    fig : matplotlib.figure.Figure, optional
        The figure object if return_fig is True
    """
    # Get consistency scores
    consistency_data = summary_df.groupby('Metric')['Overall Consistency'].first()
    consistency_data = consistency_data.sort_values()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create bar plot
    consistency_data.plot(
        kind='barh',
        ax=ax,
        color='skyblue'
    )

    # Customize appearance
    plt.title('Metric Consistency Across Segments', pad=20)
    ax.set_xlabel('Average Rank (lower is more consistent)')
    ax.invert_yaxis()  # Put best performers at top

    plt.tight_layout()

    if return_fig:
        return fig
    plt.show()


# Convert the data into a pandas DataFrame
def analyze_coin_metrics(df):
    """
    Analyze relationships between coin metrics and returns
    """
    # Calculate correlations with coin_return
    metrics_of_interest = [
        'weighted_avg_score',
        'composite_score',
        'score_confidence',
        'top_wallet_balance_pct',
        'top_wallet_count_pct',
        'total_wallets',
        'avg_wallet_balance',
        'market_cap'
    ]

    # Calculate correlations
    correlations = {}
    for metric in metrics_of_interest:
        correlation = df[metric].corr(df['coin_return'])
        correlations[metric] = correlation

    # Sort correlations by absolute value
    correlations_sorted = {k: v for k, v in sorted(correlations.items(),
                                                 key=lambda x: abs(x[1]),
                                                 reverse=True)}

    # Calculate basic statistics for coins with positive vs negative returns
    positive_returns = df[df['coin_return'] > 0]
    negative_returns = df[df['coin_return'] <= 0]

    comparison_stats = {}
    for metric in metrics_of_interest:
        pos_mean = positive_returns[metric].mean()
        neg_mean = negative_returns[metric].mean()
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(positive_returns[metric],
                                        negative_returns[metric])

        comparison_stats[metric] = {
            'positive_mean': pos_mean,
            'negative_mean': neg_mean,
            'difference': pos_mean - neg_mean,
            'p_value': p_value
        }

    # Identify potential success indicators
    success_indicators = {
        metric: metric_stats for metric, metric_stats in comparison_stats.items()
        if (abs(metric_stats['difference']) > 0.1 * metric_stats['negative_mean'] and
            metric_stats['p_value'] < 0.05)
    }

    return {
        'correlations': correlations_sorted,
        'comparison_stats': comparison_stats,
        'success_indicators': success_indicators
    }

# Create summary statistics
def print_analysis_results(results):
    """
    Print formatted analysis results
    """
    print("\n=== Correlation Analysis ===")
    print("\nCorrelations with coin return (sorted by strength):")
    for metric, corr in results['correlations'].items():
        print(f"{metric:25} : {corr:0.4f}")

    print("\n=== Positive vs Negative Returns Analysis ===")
    print("\nMetrics comparison for positive vs negative returns:")
    for metric, metric_stats in results['comparison_stats'].items():
        print(f"\n{metric}:")
        print(f"  Positive returns mean: {metric_stats['positive_mean']:0.4f}")
        print(f"  Negative returns mean: {metric_stats['negative_mean']:0.4f}")
        print(f"  Difference: {metric_stats['difference']:0.4f}")
        print(f"  P-value: {metric_stats['p_value']:0.4f}")

    print("\n=== Strong Success Indicators ===")
    print("\nMetrics showing significant difference between positive and negative returns:")
    for metric, metric_stats in results['success_indicators'].items():
        print(f"\n{metric}:")
        print(f"  Mean difference: {metric_stats['difference']:0.4f}")
        print(f"  P-value: {metric_stats['p_value']:0.4f}")

# Create visualizations
def create_visualizations(df):
    """
    Create visualization plots for the analysis
    """
    plt.figure(figsize=(15, 10))

    # Create correlation heatmap
    metrics = ['weighted_avg_score', 'composite_score', 'score_confidence',
              'top_wallet_balance_pct', 'top_wallet_count_pct', 'coin_return']
    correlation_matrix = df[metrics].corr()

    plt.subplot(2, 2, 1)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')

    # Create scatter plot of weighted_avg_score vs returns
    plt.subplot(2, 2, 2)
    plt.scatter(df['weighted_avg_score'], df['coin_return'], alpha=0.5)
    plt.xlabel('Weighted Average Score')
    plt.ylabel('Coin Return')
    plt.title('Weighted Score vs Returns')

    # Create scatter plot of composite_score vs returns
    plt.subplot(2, 2, 3)
    plt.scatter(df['composite_score'], df['coin_return'], alpha=0.5)
    plt.xlabel('Composite Score')
    plt.ylabel('Coin Return')
    plt.title('Composite Score vs Returns')

    # Create box plot of score distributions for positive/negative returns
    plt.subplot(2, 2, 4)
    df['return_category'] = df['coin_return'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
    sns.boxplot(x='return_category', y='weighted_avg_score', data=df)
    plt.title('Score Distribution by Return Category')

    plt.tight_layout()
    plt.show()


def analyze_top_coins_wallet_metrics(df: pd.DataFrame,
                                   percentile: int,
                                   method: str = 'median') -> pd.DataFrame:
    """
    Creates a formatted df comparing wallet metrics between top performing coins.

    Params:
    - df (DataFrame): DataFrame with coin metrics and returns
    - percentile (int): Percentile threshold for top performers (must be >50)
    - method (str): Aggregation method - 'mean' or 'median'

    Returns:
    - DataFrame: Analysis results formatted for styling
    """
    if percentile <= 50:
        raise ValueError("Percentile must be greater than 50")

    # Define column names and thresholds
    ntile_column = f'top_{(100 - percentile):.0f}_pct_coins'
    middle_column = f'next_{(percentile - 50):.0f}_pct_coins'
    bottom_column = 'bottom_50_pct_coins'

    top_threshold = np.percentile(df['coin_return'], percentile)
    mid_threshold = np.percentile(df['coin_return'], 50)

    # Split coins into three groups
    top_coins = df[df['coin_return'] >= top_threshold]
    middle_coins = df[(df['coin_return'] < top_threshold) & (df['coin_return'] >= mid_threshold)]
    bottom_coins = df[df['coin_return'] < mid_threshold]

    agg_func = np.mean if method == 'mean' else np.median

    results = {
        'number_of_coins': {
            ntile_column: len(top_coins),
            middle_column: len(middle_coins),
            bottom_column: len(bottom_coins),
            'all_coins': len(df)
        },
        'pct_of_coins': {
            ntile_column: len(top_coins) / len(df) * 100,
            middle_column: len(middle_coins) / len(df) * 100,
            bottom_column: len(bottom_coins) / len(df) * 100,
            'all_coins': 100.0
        }
    }

    # Wallet metrics to analyze
    wallet_metrics = [
        'weighted_avg_score',
        'composite_score',
        'top_wallet_balance_pct',
        'top_wallet_count_pct',
        'score_confidence',
        'total_balance',
        'avg_wallet_balance',
        'total_wallets',
    ]

    # Calculate wallet metrics for ntiles
    for metric in wallet_metrics:
        results[metric] = {
            ntile_column: agg_func(top_coins[metric]),
            middle_column: agg_func(middle_coins[metric]),
            bottom_column: agg_func(bottom_coins[metric]),
            'all_coins': agg_func(df[metric])
        }


    # Generate return metric names
    return_metric_names = []
    for metric in ['coin_return']:
        for stat in ['mean', 'median']:
            metric_name = f'{metric}_{stat}'
            return_metric_names.append(metric_name)
            results[metric_name] = {
                ntile_column: (np.mean if stat == 'mean' else np.median)(top_coins[metric]) * 100,
                middle_column: (np.mean if stat == 'mean' else np.median)(middle_coins[metric]) * 100,
                bottom_column: (np.mean if stat == 'mean' else np.median)(bottom_coins[metric]) * 100,
                'all_coins': (np.mean if stat == 'mean' else np.median)(df[metric])
            }

    results_df = pd.DataFrame(results).T
    size_metrics = ['number_of_coins', 'pct_of_coins']
    ordered_rows = size_metrics + return_metric_names + wallet_metrics

    return results_df.reindex(ordered_rows)


def create_top_coins_wallet_metrics_report(df: pd.DataFrame,
                            percentile: int = 75,
                            method: str = 'median') -> pd.DataFrame.style:
    """
    Creates a styled performance analysis report showing summary metrics of the
    coin-level wallet metrics.

    Params:
    - df (DataFrame): DataFrame with coin metrics and returns
    - percentile (int): Percentile threshold for top performers
    - how the metrics should be summarized (string): e.g. 'median','mean'

    Returns:
    - styled_df (DataFrame.style): Styled analysis results
    """
    # Generate results DataFrame
    results_df = analyze_top_coins_wallet_metrics(df, percentile, method)

    # Apply consistent styling
    styled_df = wime.style_rows(results_df)

    return styled_df
