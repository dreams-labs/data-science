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
    segment_name: str,
    segment_value: str,
    balance_date_str: str,
    totals_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate balance and count metrics for a single segment.

    Params:
    - analysis_df (DataFrame): Merged balance and segment data
    - segment_name (str): Name of segmentation column
    - segment_value (str): Current segment value being processed
    - balance_date_str (str): Formatted date string
    - totals_df (DataFrame): Pre-calculated totals for percentages

    Returns:
    - DataFrame: Balance and count metrics for segment
    """
    segment_data = analysis_df[analysis_df[segment_name] == segment_value]

    metrics = segment_data.groupby('coin_id', observed=True).agg({
        'usd_balance': 'sum',
        'wallet_address': 'count'
    }).rename(columns={
        'usd_balance': f'{segment_name}/{segment_value}|balance/{balance_date_str}|balance',
        'wallet_address': f'{segment_name}/{segment_value}|balance/{balance_date_str}|count'
    })

    # Calculate percentages
    metrics[f'{segment_name}/{segment_value}|balance/{balance_date_str}|balance_pct'] = (
        metrics[f'{segment_name}/{segment_value}|balance/{balance_date_str}|balance'] /
        totals_df[f'{segment_name}/total/balance']
    ).fillna(0)

    metrics[f'{segment_name}/{segment_value}|balance/{balance_date_str}|count_pct'] = (
        metrics[f'{segment_name}/{segment_value}|balance/{balance_date_str}|count'] /
        totals_df[f'{segment_name}/total/count']
    ).fillna(0)

    return metrics



def calculate_score_weighted_metrics(
    analysis_df: pd.DataFrame,
    segment_name: str,
    segment_value: str,
    balance_date_str: str,
    score_columns: List[str]
) -> pd.DataFrame:
    """
    Calculate weighted score metrics for all score columns within a segment.

    Params:
    - analysis_df (DataFrame): Merged balance and segment data
    - segment_name (str): Name of segmentation column
    - segment_value (str): Current segment value being processed
    - balance_date_str (str): Formatted date string
    - score_columns (List[str]): List of score columns to process

    Returns:
    - DataFrame: Weighted score metrics for segment
    """
    # filter data for current segment
    segment_data = analysis_df[analysis_df[segment_name] == segment_value].copy()

    # multiply each score column by usd_balance in a single pass
    score_sums = segment_data[score_columns].mul(segment_data['usd_balance'], axis=0)
    score_sums = score_sums.groupby(segment_data['coin_id'],observed=True).sum()  # sum by coin_id

    # get total balances per coin_id
    weight_sums = segment_data.groupby('coin_id',observed=True)['usd_balance'].sum()

    # compute weighted averages for all columns in one step
    weighted_scores = score_sums.div(weight_sums, axis=0)

    # rename columns to match your desired naming convention
    renamed_cols = {}
    for col in score_columns:
        score_name = col.split('|')[1]
        renamed_cols[col] = f'{segment_name}/{segment_value}|balance/{balance_date_str}|score_wtd_balance/{score_name}'
    weighted_scores.rename(columns=renamed_cols, inplace=True)

    return weighted_scores



def calculate_segment_wallet_balance_features(
    profits_df: pd.DataFrame,
    wallet_segmentation_df: pd.DataFrame,
    segment_column: str,
    balance_date: str,
    all_coin_ids: List[str]
) -> pd.DataFrame:
    """
    Main function orchestrating segment and score calculations.

    Params:
    - profits_df (DataFrame): Profits data
    - wallet_segmentation_df (DataFrame): Segment labels and scores
    - segment_column (str): Column name for segment labels
    - balance_date (str): Date for analysis
    - all_coin_ids (List[str]): Complete list of coins

    Returns:
    - DataFrame: Combined segment and score metrics
    """
    # Initial setup
    balance_date = pd.to_datetime(balance_date)
    balance_date_str = balance_date.strftime('%y%m%d')
    balances_df = profits_df[profits_df['date'] == balance_date].copy()

    # Get score columns
    score_columns = [col for col in wallet_segmentation_df.columns if col.startswith('scores|')]

    # Prepare analysis DataFrame
    analysis_df = balances_df[['coin_id', 'wallet_address', 'usd_balance']].merge(
        wallet_segmentation_df[[segment_column] + score_columns],
        left_on='wallet_address',
        right_index=True,
        how='left'
    )

    # Initialize results
    result_df = pd.DataFrame(index=all_coin_ids)
    result_df.index.name = 'coin_id'

    # Calculate totals once
    totals_df = analysis_df.groupby('coin_id', observed=True).agg({
        'usd_balance': 'sum',
        'wallet_address': 'count'
    }).rename(columns={
        'usd_balance': f'{segment_column}/total/balance',
        'wallet_address': f'{segment_column}/total/count'
    })

    # Process each segment
    for segment in wallet_segmentation_df[segment_column].unique():
        # Get segment metrics
        segment_metrics = calculate_segment_metrics(
            analysis_df, segment_column, segment,
            balance_date_str, totals_df
        )

        # Get score metrics
        score_metrics = calculate_score_weighted_metrics(
            analysis_df, segment_column, segment,
            balance_date_str, score_columns
        )

        # Combine metrics
        combined_metrics = segment_metrics.join(score_metrics, how='outer')
        result_df = result_df.join(combined_metrics, how='left')

    result_df = result_df.fillna(0)

    # Validation
    missing_coins = set(all_coin_ids) - set(result_df.index)
    if missing_coins:
        raise ValueError(f"Found {len(missing_coins)} coin_ids missing from analysis")

    return result_df
