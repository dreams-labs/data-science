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
import wallet_insights.wallet_model_evaluation as wime
from wallet_modeling.wallets_config_manager import WalletsConfig

# pylint: disable=unused-variable  # messy stats functions in visualizations
# pylint: disable=invalid-name  # X_test isn't camelcase


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
        mean_crypto_net_gain=('crypto_net_gain', 'mean'),
        median_invested=('max_investment', 'median'),
        median_crypto_net_gain=('crypto_net_gain', 'median'),
    )

    bucketed_performance_df['mean_return'] = (bucketed_performance_df['mean_crypto_net_gain']
                                                    / bucketed_performance_df['mean_invested'])
    bucketed_performance_df['median_return'] = (bucketed_performance_df['median_crypto_net_gain']
                                                        / bucketed_performance_df['median_invested'])

    return wallet_performance_df, bucketed_performance_df



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
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Get all required data for start date in one operation
    start_data = market_data_df[market_data_df['date'] == start_date].set_index('coin_id')[['price']]
    end_data = market_data_df[market_data_df['date'] == end_date].set_index('coin_id')[['price']]

    # Create consolidated dataframe
    coin_performance_df = pd.DataFrame({
        'starting_price': start_data['price'],
        'ending_price': end_data['price']
    })

    # Calculate returns
    coin_performance_df['coin_return'] = np.where(coin_performance_df['starting_price']==0, np.nan,
                                                  ((coin_performance_df['ending_price']
                                                    / coin_performance_df['starting_price']) - 1)
    )

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
                                   wallet_metrics: List,
                                   method: str = 'median') -> pd.DataFrame:
    """
    Creates a formatted df comparing wallet metrics between top performing coins.

    Params:
    - df (DataFrame): DataFrame with coin metrics and returns
    - percentile (int): Percentile threshold for top performers (must be >50)
    - wallet_metrics (list): List of the metrics to generate analysis for
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
            ntile_column: len(top_coins) / len(df),
            middle_column: len(middle_coins) / len(df),
            bottom_column: len(bottom_coins) / len(df),
            'all_coins': 100.0
        }
    }

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
                ntile_column: (np.mean if stat == 'mean' else np.median)(top_coins[metric]),
                middle_column: (np.mean if stat == 'mean' else np.median)(middle_coins[metric]),
                bottom_column: (np.mean if stat == 'mean' else np.median)(bottom_coins[metric]),
                'all_coins': (np.mean if stat == 'mean' else np.median)(df[metric])
            }

    results_df = pd.DataFrame(results).T
    size_metrics = ['number_of_coins', 'pct_of_coins']
    ordered_rows = size_metrics + return_metric_names + wallet_metrics

    return results_df.reindex(ordered_rows)


def create_top_coins_wallet_metrics_report(df: pd.DataFrame,
                                           percentile,
                                           wallet_metrics: List,
                                           method: str = 'median') -> pd.DataFrame.style:
    """
    Creates a styled performance analysis report showing summary metrics of the
    coin-level wallet metrics.

    Params:
    - df (DataFrame): DataFrame with coin metrics and returns
    - percentile (int): Percentile threshold for top performers
    - wallet_metrics (list): List of the metrics to generate analysis for
    - how the metrics should be summarized (string): e.g. 'median','mean'

    Returns:
    - styled_df (DataFrame.style): Styled analysis results
    """
    # Generate results DataFrame
    results_df = analyze_top_coins_wallet_metrics(df, percentile, wallet_metrics, method)

    # Apply consistent styling
    styled_df = wime.style_rows(results_df)

    return styled_df


def analyze_metric_segments(
    df: pd.DataFrame,
    wallet_metrics: List[str],
    n_quantiles: int,
    return_column: str
) -> pd.DataFrame:
    """
    Analyzes both mean returns and metric values for different quantile buckets.

    Params:
    - df (DataFrame): Input data with metrics and returns
    - wallet_metrics (List[str]): Metrics to analyze (higher values = higher scores)
    - n_quantiles (int): Number of quantile buckets (e.g. 4 for quartiles, 10 for deciles)
    - return_column (str): The name of the column with coin return values

    Returns:
    - DataFrame: Multi-level columns with metrics and returns for each segment
    """
    # Generate quantile breakpoints
    quantiles = np.linspace(0, 1, n_quantiles+1)

    # Generate segment names
    segments = [
        f"q{int(q1*100):02d}_q{int(q2*100):02d}"
        for q1, q2 in zip(quantiles[:-1], quantiles[1:])
    ]


    results = {}

    # Baseline row (aggregate of all coins)
    metrics = ['metric_mean', 'return_mean', 'n_coins']
    segment_size = len(df) // n_quantiles

    for metric in metrics:
        for segment in segments:
            if metric == 'metric_mean':
                value = 0
            elif metric == 'return_mean':
                value = df[return_column].mean()
            else: # n_coins
                value = segment_size

            results[('BASELINE', segment, metric)] = value

    # Perfect ordering row
    sorted_returns = np.sort(df[return_column].values)
    metrics = ['metric_mean', 'return_mean', 'n_coins']  # Define order for third key

    for metric in metrics:  # Outer loop on metric to group them together
        for i, segment in enumerate(segments):
            start_idx = i * len(df) // n_quantiles
            end_idx = (i + 1) * len(df) // n_quantiles if i < n_quantiles-1 else len(df)

            if metric == 'metric_mean':
                value = np.nan
            elif metric == 'return_mean':
                value = sorted_returns[start_idx:end_idx].mean()
            else:  # n_coins
                value = end_idx - start_idx

            results[('PERFECT', segment, metric)] = value


    for metric in wallet_metrics:
        # Calculate threshold values for this metric
        thresholds = np.quantile(df[metric], quantiles)

        # Initialize metric results
        metric_stats = {'metric_mean': [], 'return_mean': [], 'n_coins': []}

        # Calculate stats for each bucket
        for i in range(len(quantiles)-1):
            if i == len(quantiles)-2:  # Last segment includes highest value
                mask = (df[metric] >= thresholds[i])
            else:
                mask = (df[metric] >= thresholds[i]) & (df[metric] < thresholds[i+1])

            segment_data = df[mask]

            metric_stats['metric_mean'].append(segment_data[metric].mean())
            metric_stats['return_mean'].append(segment_data[return_column].mean())
            metric_stats['n_coins'].append(len(segment_data))

        # Create multi-level columns for this metric
        for stat_name, values in metric_stats.items():
            for segment, value in zip(segments, values):
                results[(metric, segment, stat_name)] = value

    # Convert to DataFrame with multi-level columns
    results_df = pd.DataFrame(results, index=[0]).T
    results_df = results_df.unstack(level=[1, 2])
    results_df.columns = results_df.columns.droplevel(0)

    return results_df


def style_metric_segments(df: pd.DataFrame) -> pd.DataFrame.style:
    """
    Apply conditional formatting with different color scales for metrics and returns.

    Params:
    - df (DataFrame): Multi-level column DataFrame from analyze_metric_segments

    Returns:
    - styled_df (DataFrame.style): DataFrame with separate metric/return formatting
    """
    def color_by_type(row, metric_type, color):
        # Filter to columns of given metric type
        type_cols = [col for col in row.index if metric_type in col[1]]

        # Get values for just this metric type
        type_values = row[type_cols].astype(float)

        # Skip if no valid values
        valid_vals = type_values.dropna()
        if len(valid_vals) == 0:
            return pd.Series([''] * len(row), index=row.index)

        # Normalize within this metric type
        min_val = valid_vals.min()
        max_val = valid_vals.max()
        if min_val == max_val:
            return pd.Series([''] * len(row), index=row.index)

        norm = type_values.apply(lambda x: (x - min_val) / (max_val - min_val) if pd.notna(x) else np.nan)

        # Create color series for all columns (defaulting to empty string)
        colors = pd.Series([''] * len(row), index=row.index)

        # Set colors only for this metric type's columns
        for col in type_cols:
            val = norm[col]
            if pd.notna(val):
                colors[col] = f'background-color: rgba{color}, {val:.2f})'

        return colors

    def row_style(row):
        # Combine colors from different metrics
        metric_colors = color_by_type(row, 'metric_mean', '(0, 100, 255')  # Blue
        return_colors = color_by_type(row, 'return_mean', '(60, 179, 113')  # Green
        count_colors = color_by_type(row, 'n_coins', '(128, 128, 128')  # Gray

        # Combine all colors (non-empty strings will overwrite empty ones)
        final_colors = pd.Series([''] * len(row), index=row.index)
        for colors in [metric_colors, return_colors, count_colors]:
            final_colors[colors != ''] = colors[colors != '']

        return final_colors

    # Create style object
    styled = df.style.apply(row_style, axis=1)

    # Format numbers
    format_dict = {}
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32]:
            format_dict[col] = '{:.3f}'.format
        elif df[col].dtype in [np.int64, np.int32]:
            format_dict[col] = '{:,.0f}'.format

    # Add just before return styled.format(format_dict):
    metric_cols = [col for col in df.columns if 'metric_mean' in col[1]]


    # Get BASELINE and PERFECT rows
    special_rows = df.loc[['BASELINE', 'PERFECT']]
    # Get and sort remaining rows
    other_rows = df.drop(['BASELINE', 'PERFECT']).sort_values(by=metric_cols[-1], ascending=True)
    # Concatenate in desired order
    df_sorted = pd.concat([special_rows, other_rows])

    styled = df_sorted.style.apply(row_style, axis=1)



    return styled.format(format_dict)

