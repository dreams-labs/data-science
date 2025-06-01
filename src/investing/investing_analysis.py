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

        start_date = datetime.strptime(file.parent.name,'%Y%m%d')
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
        json_path = Path(base_folder) / start_date.strftime('%Y%m%d') / 'wallet_model_ids.json'
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
