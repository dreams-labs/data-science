"""
Functions to flatten wallet-coin features to coin-only
"""
import logging
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

# Local module imports
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)



# -----------------------------------
#       Main Interface Function
# -----------------------------------

@u.timing_decorator
def flatten_cw_to_coin_segment_features(
    cw_metrics_df: pd.DataFrame,
    wallet_segmentation_df: pd.DataFrame,
    training_coin_cohort: list,
    score_distribution_cols: list,
    n_threads: Optional[int] = 6
) -> pd.DataFrame:
    """
    Flatten coin-wallet metrics into coin-level features across segments.

    Params:
    - cw_metrics_df (DataFrame): Trading metrics for wallets in the wallet modeling cohort.
        Indexed by [coin_id,wallet_address] with trading and balance metrics.
    - wallet_segmentation_df (DataFrame): Segment assignments for wallets in the wallet modeling cohort.
        Indexed by wallet_address with segment assignments.
    - training_coin_cohort (list): list of coin_ids to include
    - score_distribution_cols (list): Which scores should have distribution metrics.
    - n_threads (Optional[int]): number of threads for parallel processing (defaults to ThreadPoolExecutor default)

    Returns:
    - coin_wallet_features_df (DataFrame): indexed by coin_id, joined features for each
        metric × segment family.
    """
    # Data quality checks
    if cw_metrics_df.empty:
        raise ValueError(f"Provided cw_metrics_df is empty. Columns: {list(cw_metrics_df.columns)}")
    # Matching index check
    if not np.array_equal(
        cw_metrics_df.index.get_level_values('wallet_address').sort_values(),  # ignores coin_id multiindex level
        wallet_segmentation_df.index.sort_values()
    ):
        raise ValueError("Wallet cohort in cw_metrics_df doesn't match cohort in wallet_segmentation_df.")
    if not cw_metrics_df.columns.is_unique:
        raise ValueError("Duplicate columns found in cw_metrics_df.")
    if not wallet_segmentation_df.columns.is_unique:
        raise ValueError("Duplicate columns found in wallet_segmentation_df.")


    # start with an empty coin-level df
    coin_wallet_features_df = pd.DataFrame(index=training_coin_cohort)
    coin_wallet_features_df.index.name = 'coin_id'

    # identify which segmentation columns to use
    segmentation_families = wallet_segmentation_df.columns[
        ~wallet_segmentation_df.columns.str.startswith('scores|')
    ]

    # Pre‑join segmentation data once to avoid repeating the join inside
    joined_metrics_df = cw_metrics_df.join(wallet_segmentation_df, how='inner')

    # Function to process a single metric column in parallel
    def process_metric(metric_column):
        # local DataFrame for this metric
        metric_df = pd.DataFrame(index=training_coin_cohort)
        metric_df.index.name = 'coin_id'
        # loop through segment families for this metric
        for segment_family in segmentation_families:
            seg_df = flatten_cw_to_coin_features(
                joined_metrics_df,
                metric_column,
                wallet_segmentation_df,
                segment_family,
                training_coin_cohort,
                score_distribution_cols
            )
            metric_df = metric_df.join(seg_df, how='inner')
        return metric_column, metric_df

    # Multithread using n_threads from the param value
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(process_metric, metric_column): metric_column
                   for metric_column in cw_metrics_df.columns}
        for future in as_completed(futures):
            metric_column, metric_features_df = future.result()
            if not metric_features_df.columns.is_unique:
                raise ValueError("Duplicate columns found in metric_df.")
            coin_wallet_features_df = coin_wallet_features_df.join(metric_features_df, how='inner')
            if not metric_features_df.columns.is_unique:
                raise ValueError("Duplicate columns found in metric_features_df.")
            logger.info("Flattened metric: %s", metric_column)

    logger.info(
        "All wallet-based features flattened. Final shape: %s",
        coin_wallet_features_df.shape
    )
    return coin_wallet_features_df


def flatten_cw_to_coin_features(
    analysis_df: pd.DataFrame,
    metric_column: str,
    wallet_segmentation_df: pd.DataFrame,
    segment_family: str,
    all_coin_ids: List[str],
    score_distribution_cols: list,
    usd_materiality: float = 20.0
) -> pd.DataFrame:
    """Generate coin-level features from wallet-level metric.

    Params:
    - wallet_metric_df (DataFrame): MultiIndexed (coin_id, wallet_address) metrics
    - metric_column (str): Name of metric column without date suffix
    - wallet_segmentation_df (DataFrame): Segment labels and scores
    - segment_family (str): Column name for segment labels
    - all_coin_ids (List[str]): Complete list of coins
    - score_distribution_cols (list): Which scores should have distribution metrics.
    - usd_materiality (float): USD threshold for including wallets in distribution metrics

    Returns:
    - DataFrame: Coin-level features with segment metrics
    """
    # Get score columns
    score_columns = [f"scores|{dist}" for dist in score_distribution_cols]

    # Initialize results with MultiIndex aware groupby
    totals_df = analysis_df.groupby(level='coin_id', observed=True).agg({
        f'{metric_column}': 'sum',
        segment_family: 'count'
    }).rename(columns={
        f'{metric_column}': f'{segment_family}/total|{metric_column}|aggregations/aggregations/sum',
        segment_family: f'{segment_family}/total|{metric_column}|aggregations/aggregations/count'
    })

    result_df = pd.DataFrame(index=pd.Index(all_coin_ids, name='coin_id'))

    # Binary score columns should only make metrics for positive predictions
    if segment_family.startswith('score_binary'):
        segments_to_process = ['1']
    else:
        segments_to_process = wallet_segmentation_df[segment_family].unique()

    for segment_value in segments_to_process:

        # Computes basic aggregations
        aggregation_metrics_df = calculate_aggregation_metrics(
            analysis_df, segment_family, segment_value,
            metric_column, totals_df
        )

        result_df = result_df.join(aggregation_metrics_df,how='left')\
            .fillna({col: 0 for col in aggregation_metrics_df.columns})

        # #FeatureRemoval Not Predictive
        # # Computes weighted balance scores
        # score_metrics_df = calculate_score_weighted_metrics(
        #     analysis_df, segment_family, segment_value,
        #     metric_column, score_columns
        # )
        # result_df = result_df.join(score_metrics_df,how='left') # leave nulls as null


        # Add new distribution metrics
        score_dist_metrics_df = calculate_score_distribution_metrics(
            analysis_df, segment_family, segment_value,
            metric_column, score_columns, usd_materiality
        )
        result_df = result_df.join(score_dist_metrics_df,how='left') # leave nulls as null

    # Validation
    missing_coins = set(all_coin_ids) - set(result_df.index)
    if missing_coins:
        raise ValueError(f"Found {len(missing_coins)} coin_ids missing from analysis")
    duplicate_cols = result_df.columns[result_df.columns.duplicated()].tolist()
    if not result_df.columns.is_unique:
        raise ValueError(f"Found duplicate columns after flattening cw features: {duplicate_cols}")

    return result_df




# ------------------------------
#         Helper Functions
# ------------------------------

def calculate_aggregation_metrics(
    analysis_df: pd.DataFrame,
    segment_family: str,
    segment_value: str,
    metric_column: str,
    totals_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate metrics for a single segment.

    Params:
    - analysis_df (DataFrame): MultiIndexed (coin_id, wallet_address) data
    - segment_family (str): Name of segmentation column
    - segment_value (str): Current segment value
    - metric_column (str): Full metric column name
    - totals_df (DataFrame): Pre-calculated totals for percentages

    Returns:
    - DataFrame: Metrics for segment
    """
    segment_mask = analysis_df[segment_family] == segment_value
    segment_data = analysis_df[segment_mask]

    if segment_data.empty:
        # Create empty metrics with correct columns and index structure
        columns = [
            f'{segment_family}/{segment_value}|{metric_column}|aggregations/aggregations/sum',
            f'{segment_family}/{segment_value}|{metric_column}|aggregations/aggregations/count',
            f'{segment_family}/{segment_value}|{metric_column}|aggregations/aggregations/sum_pct',
            f'{segment_family}/{segment_value}|{metric_column}|aggregations/aggregations/count_pct'
        ]
        return pd.DataFrame(
            0.0,
            index=totals_df.index,  # Same index as totals_df
            columns=columns
        )

    metrics = segment_data.groupby(level='coin_id', observed=True).agg({
        metric_column: 'sum',
        segment_family: 'count'
    }).rename(columns={
        metric_column: f'{segment_family}/{segment_value}|{metric_column}|aggregations/aggregations/sum',
        segment_family: f'{segment_family}/{segment_value}|{metric_column}|aggregations/aggregations/count'
    })

    # Calculate percentages using Series division with fill_value=np.nan
    sum_col = f'{segment_family}/{segment_value}|{metric_column}|aggregations/aggregations/sum'
    count_col = f'{segment_family}/{segment_value}|{metric_column}|aggregations/aggregations/count'

    total_sum = totals_df[f'{segment_family}/total|{metric_column}|aggregations/aggregations/sum']
    total_count = totals_df[f'{segment_family}/total|{metric_column}|aggregations/aggregations/count']

    # Handle division with explicit zero check
    metrics[f'{sum_col}_pct'] = (metrics[sum_col] / total_sum).replace([np.inf, -np.inf], np.nan)
    metrics[f'{count_col}_pct'] = (metrics[count_col] / total_count).replace([np.inf, -np.inf], np.nan)

    return metrics


def calculate_score_weighted_metrics(
    analysis_df: pd.DataFrame,
    segment_family: str,
    segment_value: str,
    metric_column: str,
    score_columns: List[str]
) -> pd.DataFrame:
    """Calculate weighted score metrics for all score columns within a segment.

    Params:
    - analysis_df (DataFrame): MultiIndexed (coin_id, wallet_address) data
    - segment_family (str): Name of segmentation column
    - segment_value (str): Current segment value
    - metric_column (str): Full metric column name
    - score_columns (List[str]): List of score columns to process

    Returns:
    - DataFrame: Weighted score metrics for segment
    """
    segment_data = analysis_df[analysis_df[segment_family] == segment_value]

    # Vectorized score calculation
    score_sums = (segment_data[score_columns]
                 .mul(segment_data[metric_column], axis=0)
                 .groupby(level='coin_id', observed=True)
                 .sum())

    # Get metric totals using index-aware groupby
    weight_sums = segment_data.groupby(level='coin_id', observed=True)[metric_column].sum()

    # Handle division with explicit inf replacement
    weighted_scores = score_sums.div(weight_sums, axis=0).replace([np.inf, -np.inf], np.nan)

    # Create renamed columns dict
    renamed_cols = {
        col: f'{segment_family}/{segment_value}|{metric_column}|score_wtd/{col.split("|")[1]}'
        for col in score_columns
    }
    weighted_scores.rename(columns=renamed_cols, inplace=True)

    return weighted_scores


def calculate_score_distribution_metrics(
    analysis_df: pd.DataFrame,
    segment_family: str,
    segment_value: str,
    metric_column: str,
    score_columns: List[str],
    usd_materiality: float = 20.0
) -> pd.DataFrame:
    """
    Calculate distribution metrics (median, percentiles, std, skewness, kurtosis)
    for multiple score columns without inline lambdas for better performance.

    Params:
    - analysis_df (DataFrame): MultiIndexed (coin_id, wallet_address) data
    - segment_family (str): Name of segmentation column
    - segment_value (str): Current segment value
    - metric_column (str): Full metric column name
    - score_columns (List[str]): List of score columns to process
    - usd_materiality (float): USD threshold for including wallets

    Returns:
    - DataFrame: Distribution metrics for each score column
    """
    # filter for segment & minimum USD
    seg_data = analysis_df[
        (analysis_df[segment_family] == segment_value) &
        (analysis_df[metric_column] >= usd_materiality)
    ].copy()

    # Handle empty case - create consistent column structure
    if seg_data.empty:
        # Generate all expected column names
        suffixes = ['median', 'p002', 'p01', 'p05', 'p10', 'p90', 'p95', 'p99', 'p998', 'std', 'skew', 'kurt']
        columns = []
        for score_col in score_columns:
            score_name = score_col.split('|')[1]  # "scores|xxx" -> "xxx"
            for suffix in suffixes:
                columns.append(f'{segment_family}/{segment_value}|{metric_column}|score_dist/{score_name}/{suffix}')

        # Return empty DataFrame with correct structure
        # Use analysis_df's coin_id index to maintain consistency
        coin_ids = analysis_df.index.get_level_values('coin_id').unique()
        return pd.DataFrame(
            0.0,
            index=pd.Index(coin_ids, name='coin_id'),
            columns=columns
        )

    # percentiles
    p002_df =   seg_data.groupby(level='coin_id', observed=True)[score_columns].quantile(0.002)
    p01_df =    seg_data.groupby(level='coin_id', observed=True)[score_columns].quantile(0.01)
    p05_df =    seg_data.groupby(level='coin_id', observed=True)[score_columns].quantile(0.05)
    p10_df =    seg_data.groupby(level='coin_id', observed=True)[score_columns].quantile(0.1)
    median_df = seg_data.groupby(level='coin_id', observed=True)[score_columns].median()
    p90_df =    seg_data.groupby(level='coin_id', observed=True)[score_columns].quantile(0.9)
    p95_df =    seg_data.groupby(level='coin_id', observed=True)[score_columns].quantile(0.95)
    p99_df =    seg_data.groupby(level='coin_id', observed=True)[score_columns].quantile(0.99)
    p998_df =   seg_data.groupby(level='coin_id', observed=True)[score_columns].quantile(0.998)

    # std
    std_df = seg_data.groupby(level='coin_id', observed=True)[score_columns].std()

    # skewness - vectorized calculation
    skew_df = seg_data.groupby(level='coin_id', observed=True)[score_columns].apply(
        lambda x: x.skew()
    )

    # kurtosis - vectorized calculation
    kurt_df = seg_data.groupby(level='coin_id', observed=True)[score_columns].apply(
        lambda x: x.kurt()
    )

    # rename columns and combine
    def rename_cols(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        """
        Add naming convention for each metric type.
        """
        new_cols = {}
        for c in df.columns:
            score_name = c.split('|')[1]  # e.g. "score|xxx" -> "xxx"
            new_cols[c] = f'{segment_family}/{segment_value}|{metric_column}|score_dist/{score_name}/{suffix}'
        return df.rename(columns=new_cols)

    p002_df =   rename_cols(p002_df, 'p002')
    p01_df =    rename_cols(p01_df, 'p01')
    p05_df =    rename_cols(p05_df, 'p05')
    p10_df =    rename_cols(p10_df, 'p10')
    median_df = rename_cols(median_df, 'median')
    p90_df =    rename_cols(p90_df, 'p90')
    p95_df =    rename_cols(p95_df, 'p95')
    p99_df =    rename_cols(p99_df, 'p99')
    p998_df =   rename_cols(p998_df, 'p998')
    std_df =    rename_cols(std_df, 'std')
    skew_df =   rename_cols(skew_df, 'skew')
    kurt_df =   rename_cols(kurt_df, 'kurt')

    # merge all results horizontally
    metrics_df = pd.concat(
        [
            median_df,
            p002_df,
            p01_df,
            p05_df,
            p10_df,
            p90_df,
            p95_df,
            p99_df,
            p998_df,
            std_df,
            skew_df,
            kurt_df
        ],
        axis=1
    ).fillna(0)

    return metrics_df
