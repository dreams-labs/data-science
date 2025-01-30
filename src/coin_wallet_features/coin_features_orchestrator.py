"""
Calculates metrics aggregated at the coin level
"""
import logging
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import yaml

# Local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig
# import coin_wallet_features.wallet_balance_features as cwb
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()
wallets_metrics_config = u.load_config('../config/wallets_metrics_config.yaml')
wallets_features_config = yaml.safe_load(Path('../config/wallets_features_config.yaml').read_text(encoding='utf-8'))



def load_wallet_scores(wallet_scores: list, wallet_scores_path: str) -> pd.DataFrame:
    """
    Params:
    - wallet_scores (list): List of score names to merge
    - wallet_scores_path (str): Base path for score parquet files

    Returns:
    - wallet_scores_df (DataFrame):
        wallet_address (index): contains all wallet addresses included in any score
        score|{score_name} (float): the predicted score
        residual|{score_name} (float): the residual of the score
    """
    wallet_scores_df = pd.DataFrame()

    for score_name in wallet_scores:
        score_df = pd.read_parquet(f"{wallet_scores_path}/{score_name}.parquet")
        feature_cols = []

        # Add scores column
        score_df[f'scores|{score_name}_score'] = score_df[f'score|{score_name}']
        feature_cols.append(f'scores|{score_name}_score')

        # Add residuals column
        score_df[f'scores|{score_name}_residual'] = (
            score_df[f'score|{score_name}'] - score_df[f'actual|{score_name}']
        )
        feature_cols.append(f'scores|{score_name}_residual')

        # Add confidence if provided
        if f'confidence|{score_name}' in score_df.columns:
            score_df[f'scores|{score_name}_confidence'] = score_df[f'confidence|{score_name}']
            feature_cols.append(f'scores|{score_name}_confidence')

        # Full outer join with existing results
        wallet_scores_df = (
            score_df[feature_cols] if wallet_scores_df.empty
            else wallet_scores_df.join(score_df[feature_cols], how='outer')
        )

    return wallet_scores_df


# ------------------------------------------------------- #
# Functions to flatten wallet-coin features to coin-only
# ------------------------------------------------------- #


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
        on='wallet_address',
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
