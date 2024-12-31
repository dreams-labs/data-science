import logging
from typing import List
from pathlib import Path
import yaml
import pandas as pd

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



def calculate_segment_metrics(
    analysis_df: pd.DataFrame,
    segment_family: str,
    segment_value: str,
    metric_column: str,
    totals_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate metrics for a single segment.

    Params:
    - analysis_df (DataFrame): Merged metric and segment data
    - segment_family (str): Name of segmentation column
    - segment_value (str): Current segment value
    - metric_column (str): Full metric column name (e.g. 'usd_balance/241120')
    - totals_df (DataFrame): Pre-calculated totals for percentages

    Returns:
    - DataFrame: Metrics for segment
    """
    segment_data = analysis_df[analysis_df[segment_family] == segment_value]

    metrics = segment_data.groupby('coin_id', observed=True).agg({
        metric_column: 'sum',
        'wallet_address': 'count'
    }).rename(columns={
        metric_column: f'{segment_family}/{segment_value}|{metric_column}|sum',
        'wallet_address': f'{segment_family}/{segment_value}|{metric_column}|count'
    })

    # Calculate percentages using the passed metric column name
    metrics[f'{segment_family}/{segment_value}|{metric_column}|sum_pct'] = (
        metrics[f'{segment_family}/{segment_value}|{metric_column}|sum'] /
        totals_df[f'{segment_family}/total|{metric_column}|sum']
    ).fillna(0)

    metrics[f'{segment_family}/{segment_value}|count_pct'] = (
        metrics[f'{segment_family}/{segment_value}|{metric_column}|count'] /
        totals_df[f'{segment_family}/total|{metric_column}|count']
    ).fillna(0)

    return metrics


def calculate_score_weighted_metrics(
    analysis_df: pd.DataFrame,
    segment_name: str,
    segment_value: str,
    metric_column: str,
    score_columns: List[str]
) -> pd.DataFrame:
    """Calculate weighted score metrics for all score columns within a segment.

    Params:
    - analysis_df (DataFrame): Merged metric and segment data
    - segment_name (str): Name of segmentation column
    - segment_value (str): Current segment value
    - metric_column (str): Full metric column name (e.g. 'usd_balance/241120')
    - score_columns (List[str]): List of score columns to process

    Returns:
    - DataFrame: Weighted score metrics for segment
    """
    segment_data = analysis_df[analysis_df[segment_name] == segment_value].copy()

    # Multiply scores by metric in single pass
    score_sums = segment_data[score_columns].mul(segment_data[metric_column], axis=0)
    score_sums = score_sums.groupby(segment_data['coin_id'], observed=True).sum()

    # Get metric totals per coin_id
    weight_sums = segment_data.groupby('coin_id', observed=True)[metric_column].sum()

    # Compute weighted averages
    weighted_scores = score_sums.div(weight_sums, axis=0)

    # Rename using passed metric column
    renamed_cols = {}
    for col in score_columns:
        score_name = col.split('|')[1]
        renamed_cols[col] = f'{segment_name}/{segment_value}|{metric_column}|score_wtd/{score_name}'
    weighted_scores.rename(columns=renamed_cols, inplace=True)

    return weighted_scores


def generate_segmented_features(
    wallet_metric_df: pd.DataFrame,
    metric_column: str,
    wallet_segmentation_df: pd.DataFrame,
    segment_family: str,
    all_coin_ids: List[str]
    ) -> pd.DataFrame:
    """Generate coin-level features from wallet-level metric.

    Params:
    - wallet_metric_df (DataFrame): Output from calculate_wallet_metric
    - metric_column (str): Name of metric column without date suffix
    - wallet_segmentation_df (DataFrame): Segment labels and scores
    - segment_family (str): Column name for segment labels
    - all_coin_ids (List[str]): Complete list of coins

    Returns:
    - DataFrame: Coin-level features with segment metrics
    """

    # Get score columns
    score_columns = [col for col in wallet_segmentation_df.columns
                    if col.startswith('scores|')]

    # Merge metric data with segmentation
    analysis_df = wallet_metric_df.merge(
        wallet_segmentation_df[[segment_family] + score_columns],
        left_on='wallet_address',
        right_index=True,
        how='left'
    )


    # Initialize results DataFrame
    result_df = pd.DataFrame(index=all_coin_ids)
    result_df.index.name = 'coin_id'

    # Calculate totals for the metric
    totals_df = analysis_df.groupby('coin_id', observed=True).agg({
        f'{metric_column}': 'sum',
        'wallet_address': 'count'
    }).rename(columns={
        f'{metric_column}': f'{segment_family}/total|{metric_column}|sum',
        'wallet_address': f'{segment_family}/total|{metric_column}|count'
    })

    # Process each segment value
    for segment_value in wallet_segmentation_df[segment_family].unique():
        segment_metrics = calculate_segment_metrics(
            analysis_df, segment_family, segment_value,
            metric_column, totals_df
        )

        score_metrics = calculate_score_weighted_metrics(
            analysis_df, segment_family, segment_value,
            metric_column, score_columns
        )

        combined_metrics = segment_metrics.join(score_metrics, how='outer')
        result_df = result_df.join(combined_metrics, how='left')

    result_df = result_df.fillna(0)

    # Validation
    missing_coins = set(all_coin_ids) - set(result_df.index)
    if missing_coins:
        raise ValueError(f"Found {len(missing_coins)} coin_ids missing from analysis")

    return result_df
