"""
Functions to flatten wallet-coin features to coin-only
"""
import logging
from typing import List
import pandas as pd
import numpy as np

# Local module imports
# import coin_wallet_features.wallet_balance_features as cwb

# Set up logger at the module level
logger = logging.getLogger(__name__)



# -----------------------------------
#       Main Interface Function
# -----------------------------------

def flatten_cw_to_coin_segment_features(
    cw_metrics_df: pd.DataFrame,
    wallet_segmentation_df: pd.DataFrame,
    training_coin_cohort: list
) -> pd.DataFrame:
    """
    Flatten coin-wallet metrics into coin-level features across segments.

    Params:
    - cw_metrics_df (DataFrame): indexed by (coin_id, wallet_address) with 'balances/...'
        and 'trading/...' columns.
    - wallet_segmentation_df (DataFrame): indexed by wallet_address with segment assignments.
    - training_coin_cohort (list): list of coin_ids to include

    Returns:
    - coin_wallet_features_df (DataFrame): indexed by coin_id, joined features for each
        metric × segment family.
    """
    # start with an empty coin-level df
    coin_wallet_features_df = pd.DataFrame(index=training_coin_cohort)
    coin_wallet_features_df.index.name = 'coin_id'

    # identify which segmentation columns to use
    segmentation_families = wallet_segmentation_df.columns[
        ~wallet_segmentation_df.columns.str.startswith('scores|')
    ]

    # loop through each metric × segment and join
    total_metrics = len(cw_metrics_df.columns)
    for i, metric_column in enumerate(cw_metrics_df.columns, start=1):
        for segment_family in segmentation_families:
            # generate coin-level features for this metric & segment
            segment_df = flatten_cw_to_coin_features(
                cw_metrics_df,
                metric_column,
                wallet_segmentation_df,
                segment_family,
                training_coin_cohort
            )
            coin_wallet_features_df = coin_wallet_features_df.join(segment_df, how='inner')
        logger.info("Flattened metric %s/%s: %s", i, total_metrics, metric_column)

    logger.info(
        "All wallet-based features flattened. Final shape: %s",
        coin_wallet_features_df.shape
    )
    return coin_wallet_features_df



def flatten_cw_to_coin_features(
    wallet_metric_df: pd.DataFrame,
    metric_column: str,
    wallet_segmentation_df: pd.DataFrame,
    segment_family: str,
    all_coin_ids: List[str],
    usd_materiality: float = 20.0
) -> pd.DataFrame:
    """Generate coin-level features from wallet-level metric.

    Params:
    - wallet_metric_df (DataFrame): MultiIndexed (coin_id, wallet_address) metrics
    - metric_column (str): Name of metric column without date suffix
    - wallet_segmentation_df (DataFrame): Segment labels and scores
    - segment_family (str): Column name for segment labels
    - all_coin_ids (List[str]): Complete list of coins
    - usd_materiality (float): USD threshold for including wallets in distribution metrics

    Returns:
    - DataFrame: Coin-level features with segment metrics
    """
    # Get score columns
    score_columns = [col for col in wallet_segmentation_df.columns
                    if col.startswith('scores|')]

    # Join segmentation data using index
    analysis_df = wallet_metric_df.join(
        wallet_segmentation_df[[segment_family] + score_columns],
        how='left'
    )

    # Initialize results with MultiIndex aware groupby
    totals_df = analysis_df.groupby(level='coin_id', observed=True).agg({
        f'{metric_column}': 'sum',
        segment_family: 'count'
    }).rename(columns={
        f'{metric_column}': f'{segment_family}/total|{metric_column}|aggregations/sum',
        segment_family: f'{segment_family}/total|{metric_column}|aggregations/count'
    })

    result_df = pd.DataFrame(index=pd.Index(all_coin_ids, name='coin_id'))

    for segment_value in wallet_segmentation_df[segment_family].unique():
        # Computes basic aggregations
        aggregation_metrics_df = calculate_aggregation_metrics(
            analysis_df, segment_family, segment_value,
            metric_column, totals_df
        )
        result_df = result_df.join(aggregation_metrics_df,how='left')\
            .fillna({col: 0 for col in aggregation_metrics_df.columns})


        # Computes weighted balance scores
        score_metrics_df = calculate_score_weighted_metrics(
            analysis_df, segment_family, segment_value,
            metric_column, score_columns
        )
        result_df = result_df.join(score_metrics_df,how='left') # leave nulls as null


        # Checks if the column is includes material usd values
        if wallet_metric_df[metric_column].abs().mean() > (10 * usd_materiality):

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

    metrics = segment_data.groupby(level='coin_id', observed=True).agg({
        metric_column: 'sum',
        segment_family: 'count'
    }).rename(columns={
        metric_column: f'{segment_family}/{segment_value}|{metric_column}|aggregations/sum',
        segment_family: f'{segment_family}/{segment_value}|{metric_column}|aggregations/count'
    })

    # Calculate percentages using Series division with fill_value=np.nan
    sum_col = f'{segment_family}/{segment_value}|{metric_column}|aggregations/sum'
    count_col = f'{segment_family}/{segment_value}|{metric_column}|aggregations/count'

    total_sum = totals_df[f'{segment_family}/total|{metric_column}|aggregations/sum']
    total_count = totals_df[f'{segment_family}/total|{metric_column}|aggregations/count']

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
    Calculate distribution metrics (median, p10, p90, std) for multiple score columns
    without inline lambdas, potentially improving performance for large data.

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

    # 1) median
    median_df = seg_data.groupby(level='coin_id', observed=True)[score_columns].median()

    # 2) p10
    p10_df = seg_data.groupby(level='coin_id', observed=True)[score_columns].quantile(0.1)

    # 3) p90
    p90_df = seg_data.groupby(level='coin_id', observed=True)[score_columns].quantile(0.9)

    # 4) std
    std_df = seg_data.groupby(level='coin_id', observed=True)[score_columns].std()

    # rename columns and combine
    def rename_cols(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        """
        Add naming convention for each metric type.
        """
        new_cols = {}
        for c in df.columns:
            score_name = c.split('|')[1]  # e.g. "score|xxx" -> "xxx"
            new_cols[c] = f'{segment_family}/{segment_value}|{metric_column}|score_dist/{score_name}_{suffix}'
        return df.rename(columns=new_cols)

    median_df = rename_cols(median_df, 'median')
    p10_df = rename_cols(p10_df, 'p10')
    p90_df = rename_cols(p90_df, 'p90')
    std_df = rename_cols(std_df, 'std')

    # merge all results horizontally
    metrics_df = median_df.join([p10_df, p90_df, std_df], how='outer').fillna(0)

    return metrics_df
