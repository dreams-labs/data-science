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



def calculate_segment_wallet_balance_features(
    profits_df: pd.DataFrame,
    wallet_segmentation_df: pd.DataFrame,
    segment_column: str,
    balance_date: str,
    all_coin_ids: List[str]) -> pd.DataFrame:
    """
    Calculates coin-level metrics for each wallet segment.

    Params:
    - profits_df (DataFrame): Profits data with coin_id, wallet_address, date, usd_balance
    - wallet_segments (Series): Segment labels indexed by wallet_address
    - balance_date (str): Date for balance analysis
    - all_coin_ids (List[str]): Complete list of coins to include in output

    Returns:
    - DataFrame: Coin-level features for each segment, including:
        {segments_name}/{segment_label}|balance/{YYMMDD}/balance: Total USD balance
        {segments_name}/{segment_label}|balance/{YYMMDD}/count: Number of wallets
        {segments_name}/{segment_label}|balance/{YYMMDD}/balance_pct: Segment balance / total balance
        {segments_name}/{segment_label}|balance/{YYMMDD}/count_pct: Segment wallet count / total wallet count
        Indexed by coin_id
    """
    balance_date = pd.to_datetime(balance_date)
    balance_date_str = balance_date.strftime('%y%m%d')
    balances_df = profits_df[profits_df['date'] == balance_date].copy()

    # Get segments name for prefix, defaulting to 'segment' if unnamed
    segments_series = wallet_segmentation_df[segment_column]
    segments_name = segment_column

    analysis_df = balances_df[['coin_id', 'wallet_address', 'usd_balance']].merge(
        segments_series.rename(f'{segments_name}'),
        left_on='wallet_address',
        right_index=True,
        how='left'
    )

    result_df = pd.DataFrame(index=all_coin_ids)
    result_df.index.name = 'coin_id'

    totals = analysis_df.groupby('coin_id', observed=True).agg({
        'usd_balance': 'sum',
        'wallet_address': 'count'
    }).rename(columns={
        'usd_balance': f'{segments_name}/total/balance',
        'wallet_address': f'{segments_name}/total/count'
    })

    for segment in segments_series.unique():
        segment_metrics = analysis_df[analysis_df[segments_name] == segment].groupby(
            'coin_id',
            observed=True
        ).agg({
            'usd_balance': 'sum',
            'wallet_address': 'count'
        }).rename(columns={
            'usd_balance': f'{segments_name}/{segment}|balance/{balance_date_str}/balance',
            'wallet_address': f'{segments_name}/{segment}|balance/{balance_date_str}/count'
        })

        segment_metrics[f'{segments_name}/{segment}|balance/{balance_date_str}/balance_pct'] = (
            segment_metrics[f'{segments_name}/{segment}|balance/{balance_date_str}/balance'] /
            totals[f'{segments_name}/total/balance']
        ).fillna(0)

        segment_metrics[f'{segments_name}/{segment}|balance/{balance_date_str}/count_pct'] = (
            segment_metrics[f'{segments_name}/{segment}|balance/{balance_date_str}/count'] /
            totals[f'{segments_name}/total/count']
        ).fillna(0)

        result_df = result_df.join(segment_metrics, how='left')

    result_df = result_df.fillna(0)

    missing_coins = set(all_coin_ids) - set(result_df.index)
    if missing_coins:
        raise ValueError(f"Found {len(missing_coins)} coin_ids missing from analysis")


    return result_df
