import logging
from pathlib import Path
from datetime import datetime,timedelta
import json
import pandas as pd

# Local module imports
import coin_insights.coin_validation_analysis as civa
import utils as u

# pylint: disable=unused-variable  # messy stats functions in visualizations
# pylint: disable=invalid-name  # X_test isn't camelcase

# Set up logger at the module level
logger = logging.getLogger(__name__)


def analyze_investment_performance_by_cycle(
        coin_scores_df: pd.DataFrame,
        score_threshold: float
    ) -> pd.DataFrame:
    """
    Compute performance metrics by coin_epoch_start_date with threshold filtering.

    Params:
    - coin_scores_df (DataFrame): coin performance data with multi-index (coin_id, coin_epoch_start_date)
        output by orchestrate_coin_investment_cycles()
    - score_threshold (float): minimum score for filtered analysis

    Returns:
    - epoch_analysis_df (DataFrame): performance metrics by epoch
    """
    # Reset index to work with coin_epoch_start_date as column
    df_reset = coin_scores_df.reset_index()

    # winsorize returns
    df_reset['coin_return_wins'] = u.winsorize(df_reset['coin_return'],0.005)

    # Create mask for above-threshold coins
    above_threshold = df_reset['score'] >= score_threshold

    # Compute overall metrics by epoch
    overall_metrics = df_reset.groupby('coin_epoch_start_date')['coin_return_wins'].agg([
        ('mean_return_all', 'mean'),
        ('median_return_all', 'median'),
        ('count_all', 'count')
    ])

    # Compute threshold-filtered metrics by epoch
    threshold_metrics = df_reset[above_threshold].groupby('coin_epoch_start_date')['coin_return_wins'].agg([
        ('mean_return_above_threshold', 'mean'),
        ('median_return_above_threshold', 'median'),
        ('count_above_threshold', 'count')
    ])

    # Combine results
    epoch_analysis_df = overall_metrics.join(threshold_metrics, how='left')
    epoch_analysis_df = epoch_analysis_df.reset_index()

    return epoch_analysis_df



def calculate_lifetime_performance_by_thresholds(
        coin_scores_df: pd.DataFrame,
        score_thresholds: list
    ) -> pd.DataFrame:
    """
    Calculate lifetime compound returns across multiple score thresholds.

    Params:
    - coin_scores_df (DataFrame): coin performance data with multi-index (coin_id, coin_epoch_start_date)
    - score_thresholds (list): list of score thresholds to analyze

    Returns:
    - lifetime_performance_df (DataFrame): compound returns by threshold
    """
    results = []

    for threshold in score_thresholds:
        # Get epoch-level performance for this threshold
        epoch_metrics = analyze_investment_performance_by_cycle(coin_scores_df, threshold)

        # Calculate compound returns using mean returns
        # Adding 1 to convert returns to multipliers, then compound
        compound_all = (1 + epoch_metrics['mean_return_all']).prod() - 1
        compound_threshold = (1 + epoch_metrics['mean_return_above_threshold'].fillna(0)).prod() - 1

        # Calculate total coins across all epochs
        total_coins_threshold = epoch_metrics['count_above_threshold'].fillna(0).sum()

        results.append({
            'score_threshold': threshold,
            'lifetime_return_all': compound_all,
            'lifetime_return_above_threshold': compound_threshold,
            'total_coins_above_threshold': total_coins_threshold
        })

    lifetime_performance_df = pd.DataFrame(results)
    return lifetime_performance_df



def generate_coin_metrics(
   base_folder: str,
   model: str,
   complete_hybrid_cw_id_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate coin-level metrics by aggregating wallet scores across epochs.

    Params:
    - base_folder (str): Base directory to search for score files
    - model (str): Model name used in file paths and column names
    - complete_hybrid_cw_id_df (DataFrame): Mapping between coin_id and hybrid_cw_id

    Returns:
    - coin_metrics_df (DataFrame): Aggregated coin metrics with returns and macro indicators
    """
    all_scores_files = sorted(list(Path(base_folder).glob(f'*/scores/{model}.parquet')))
    if len(all_scores_files) == 0:
        raise ValueError("Couldn't find any matching scores files.")

    epoch_metrics_dfs = []

    for file in all_scores_files:
        # Load file and extract epoch date
        scores_df = pd.read_parquet(file)
        if len(scores_df.index.get_level_values('epoch_start_date').unique()) != 1:
            raise ValueError(f"Only one coin epoch should be present in file '{file}'.")

        # Convert hybrid_cw_ids to coin_ids
        coin_scores_df = (
            scores_df.reset_index()
            .merge(
                complete_hybrid_cw_id_df[['coin_id','hybrid_cw_id']],
                how='inner',
                left_on='wallet_address',
                right_on='hybrid_cw_id'
            )[['coin_id', 'epoch_start_date', f'score|{model}']]
            .set_index(['coin_id','epoch_start_date'])
        )

        # Vectorized grouping without lambdas
        score_col = f'score|{model}'
        series = coin_scores_df[score_col]
        grouped = series.groupby(level=['coin_id','epoch_start_date'], observed=True)

        # Basic stats
        epoch_metrics_df = grouped.agg(
            count_scores='count',
            avg_scores='mean',
            median_scores='median',
            stdev_scores='std'
        )

        # Threshold-based metrics   # pylint:disable=line-too-long
        epoch_metrics_df['count_above_80'] = (series > 0.80).groupby(level=['coin_id','epoch_start_date'], observed=True).sum()
        epoch_metrics_df['pct_above_80']   = (series > 0.80).groupby(level=['coin_id','epoch_start_date'], observed=True).mean() * 100
        epoch_metrics_df['count_above_90'] = (series > 0.90).groupby(level=['coin_id','epoch_start_date'], observed=True).sum()
        epoch_metrics_df['pct_above_90']   = (series > 0.90).groupby(level=['coin_id','epoch_start_date'], observed=True).mean() * 100
        # epoch_metrics_df['count_below_10'] = (series < 0.10).groupby(level=['coin_id','epoch_start_date'], observed=True).sum()
        # epoch_metrics_df['pct_below_10']   = (series < 0.10).groupby(level=['coin_id','epoch_start_date'], observed=True).mean() * 100
        # epoch_metrics_df['count_below_20'] = (series < 0.20).groupby(level=['coin_id','epoch_start_date'], observed=True).sum()
        # epoch_metrics_df['pct_below_20']   = (series < 0.20).groupby(level=['coin_id','epoch_start_date'], observed=True).mean() * 100

        # Quantiles
        epoch_metrics_df['p99_score'] = grouped.quantile(0.99)
        epoch_metrics_df['p95_score'] = grouped.quantile(0.95)
        epoch_metrics_df['p90_score'] = grouped.quantile(0.9)
        epoch_metrics_df['p10_score'] = grouped.quantile(0.1)
        epoch_metrics_df['p05_score'] = grouped.quantile(0.05)
        epoch_metrics_df['p01_score'] = grouped.quantile(0.01)

        # Round for readability
        epoch_metrics_df = epoch_metrics_df.round(4)
        epoch_metrics_dfs.append(epoch_metrics_df)

    coin_metrics_df = pd.concat(epoch_metrics_dfs)
    logger.info(f"Successfully generated coin metrics for model '{model}'")

    return coin_metrics_df


def calculate_epoch_coin_returns(
    coin_metrics_df: pd.DataFrame,
    complete_market_data_df: pd.DataFrame,
    wallets_config: dict,
    base_folder: str,
    macro_col: str
) -> pd.DataFrame:
    """
    Calculate coin returns across epochs with macro indicators and winsorization.

    Params:
    - coin_metrics_df (DataFrame): MultiIndex with 'epoch_start_date' level
    - complete_market_data_df (DataFrame): market data for performance calculations
    - wallets_config (dict): config containing training_data.modeling_period_duration
    - base_folder (str): path for finding wallet_model_ids.json files
    - macro_col (str): key for macro averages in models dict

    Returns:
    - result_df (DataFrame): coin returns with macro indicators, winsorized and joined
    """
    returns_dfs = []

    for file in sorted(list(Path(base_folder).glob('*/wallet_model_ids.json'))):

        start_date = datetime.strptime(file.parent.name,'%y%m%d')
        end_date = start_date + timedelta(days=wallets_config['training_data']['modeling_period_duration'])

        # Calculate coin returns
        returns_df = civa.calculate_coin_performance(
            complete_market_data_df,
            start_date,
            end_date
        )
        returns_df['epoch_start_date'] = start_date
        returns_df = returns_df.reset_index().set_index(['coin_id','epoch_start_date'])

        # Append indicators
        json_path = Path(base_folder) / start_date.strftime('%y%m%d') / 'wallet_model_ids.json'
        with open(json_path, 'r', encoding='utf-8') as f:
            models_dict = json.load(f)
        try:
            macro_metric = models_dict[list(models_dict.keys())[0]]['macro_averages'][macro_col]
        except KeyError:
            logger.warning(f"No macro metrics found in '{file}'")
            continue
        returns_df['macro_indicator'] = macro_metric

        returns_dfs.append(returns_df)

    # Join and winsorize
    all_returns_df = pd.concat(returns_dfs)
    all_returns_df['coin_return_wins'] = u.winsorize(all_returns_df['coin_return'], 0.01)
    result_df = all_returns_df.join(coin_metrics_df, how='inner')

    return result_df
