import logging
from typing import List,Dict
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
)
from dreams_core import core as dc

# local module imports
import wallet_features.clustering_features as clf
from wallet_modeling.wallets_config_manager import WalletsConfig

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()


# ---------------------------------------------
#             Cluster Analysis Tools
# ---------------------------------------------

def analyze_cluster_metrics(modeling_df: pd.DataFrame,
                         cluster_counts: List[int],
                         comparison_metrics: List[str],
                         agg_method: str = 'median') -> Dict[int, pd.DataFrame]:
    """
    Calculate aggregate metrics for each cluster grouping.

    Params:
    - modeling_df (DataFrame): DataFrame with cluster assignments and metrics
    - cluster_counts (List[int]): List of k values [e.g. 2, 4]
    - comparison_metrics (List[str]): Metrics to analyze
    - agg_method (str): Aggregation method - either 'median' or 'mean'

    Returns:
    - Dict[int, DataFrame]: Aggregated metrics for each k value's clusters
    """
    if agg_method not in ['median', 'mean']:
        raise ValueError("agg_method must be either 'median' or 'mean'")

    results = {}

    for k in cluster_counts:
        cluster_col = f'k{k}_cluster'
        cluster_sizes = modeling_df[cluster_col].value_counts()

        # Create initial DataFrame with size metrics
        cluster_stats = pd.DataFrame({
            'cluster_size': cluster_sizes,
            'cluster_pct': (cluster_sizes / len(modeling_df) * 100).round(2)
        })

        # Add the aggregated metrics
        metric_aggs = modeling_df.groupby(cluster_col)[comparison_metrics].agg(agg_method)
        cluster_stats = pd.concat([cluster_stats, metric_aggs], axis=1)

        results[k] = cluster_stats

    return results


def analyze_cluster_performance(modeling_df: pd.DataFrame,
                                cluster_counts: List[int],
                                y_true: pd.Series,
                                y_pred: pd.Series) -> Dict[int, pd.DataFrame]:
    """
    Calculate model performance metrics for each cluster in test set.

    Params:
    - modeling_df (DataFrame): DataFrame with cluster assignments
    - cluster_counts (List[int]): List of k values [e.g. 2, 4]
    - y_true (Series): True target values (test set)
    - y_pred (Series): Model predictions (test set)

    Returns:
    - Dict[int, DataFrame]: Performance metrics for each k value's clusters
    """
    # Clusters with fewer than this amount of wallets will have NaNs instead of metrics.
    cluster_required_sample_size = 50

    # Filter modeling_df to only include test set wallets
    test_wallets = y_true.index
    test_df = modeling_df.loc[test_wallets]

    # Convert to numpy arrays with matching indices
    y_true_arr = y_true.values
    y_pred_arr = y_pred

    results = {}

    for k in cluster_counts:
        cluster_col = f'k{k}_cluster'
        metrics_by_cluster = []

        for cluster in range(k):
            # Get mask for current cluster
            cluster_mask = test_df[cluster_col] == cluster
            n_samples = cluster_mask.sum()

            # Initialize base metrics dictionary
            cluster_metrics = {
                'test_set_samples': n_samples
            }

            # If cluster is too small, log warning and use NaN metrics
            if n_samples < cluster_required_sample_size:
                logger.warning(
                    f"Cluster {cluster} (k={k}) has only {n_samples} test set samples. "
                    f"Test set performance metrics will be reported as NaN."
                )
                cluster_metrics.update({
                    'r2': np.nan,
                    'rmse': np.nan,
                    'mae': np.nan,
                    'explained_variance': np.nan
                })

            # Otherwise calculate metrics for the cluster
            else:
                cluster_metrics.update({
                    'r2': r2_score(y_true_arr[cluster_mask], y_pred_arr[cluster_mask]),
                    'rmse': np.sqrt(mean_squared_error(y_true_arr[cluster_mask], y_pred_arr[cluster_mask])),
                    'mae': mean_absolute_error(y_true_arr[cluster_mask], y_pred_arr[cluster_mask]),
                    'explained_variance': explained_variance_score(y_true_arr[cluster_mask], y_pred_arr[cluster_mask])
                })

            metrics_by_cluster.append(cluster_metrics)

        results[k] = pd.DataFrame(metrics_by_cluster)

    return results


def style_rows(df: pd.DataFrame) -> pd.DataFrame.style:
    """
    Apply row-wise conditional formatting with blue gradient and human-readable numbers.

    Params:
    - df (DataFrame): input DataFrame to style

    Returns:
    - styled_df (DataFrame.style): DataFrame with formatting and human-readable numbers
    """
    def row_style(row):
        # Skip non-numeric rows
        if not np.issubdtype(row.dtype, np.number):
            return [''] * len(row)

        # Handle rows with NaN values
        valid_vals = row.dropna()
        if len(valid_vals) == 0:
            return [''] * len(row)

        # Normalize values between 0 and 1 for each row
        min_val = valid_vals.min()
        max_val = valid_vals.max()
        if min_val == max_val:
            return ['background-color: rgba(0, 0, 255, 0)'] * len(row)

        # Handle NaN explicitly during normalization
        norm = row.apply(lambda x: (x - min_val) / (max_val - min_val) if pd.notna(x) else np.nan)
        colors = [f'background-color: rgba(0, 0, 255, {x:.2f})' if not np.isnan(x) else '' for x in norm]
        return colors

    # Create the style object with background colors
    styled = df.style.apply(row_style, axis=1)

    # Add number formatting for numeric columns with more than 2 unique values
    format_dict = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].nunique() > 2:  # Skip binary columns
            format_dict[col] = lambda x: dc.human_format(x) if pd.notna(x) else ''

    return styled.format(format_dict)


def create_cluster_report(modeling_df: pd.DataFrame,
                          model_results: Dict,
                          n: int,
                          comparison_metrics: List[str],
                          agg_method: str = 'median') -> pd.DataFrame.style:
    """
    Generates a formatted dataframe report on the clusters' sizes, performance, and aggregated
    values for the comparison_metrics columns.

    Params:
    - modeling_df (DataFrame): the df with all the features input to the model
    - model_results (dict): output dict with results generated by wm.WalletModel.run_experiment()
    - n (int): cluster set to analyze
    - comparison_metrics (list): the metrics from modeling_df to provide aggregated values for
    - agg_method (str): Aggregation method - either 'median' or 'mean'

    Returns:
    - styled_df (pandas.io.formats.style.Styler): a pretty df with metrics

    """
    # Create df that includes base metrics, all cluster columns, and param comparison metrics
    base_metrics = [
        'trading|max_investment|all_windows',
        'trading|crypto_net_gain|all_windows',
        'mktcap|end_portfolio_wtd_market_cap|all_windows',
        'performance|crypto_net_gain/max_investment/base|all_windows',
        'performance|crypto_net_gain/active_twb/base|all_windows',
    ]
    cluster_cols = [col for col in modeling_df.columns if col.startswith('cluster|')]
    cluster_analysis_df = modeling_df[list(set(cluster_cols + base_metrics + comparison_metrics))].copy()

    # Assign wallets to categorical clusters based on the distance values
    cluster_assignments_df = clf.assign_clusters_from_distances(cluster_analysis_df,
                                                         wallets_config['features']['clustering_n_clusters'])
    cluster_analysis_df = cluster_analysis_df.join(cluster_assignments_df,how='inner')

    # Generate metrics for clusters
    cluster_profiles = analyze_cluster_metrics(
        cluster_analysis_df,
        wallets_config['features']['clustering_n_clusters'],
        list(set(base_metrics + comparison_metrics)),
        agg_method=agg_method
    )

    # Assess model performance in the test set of each cluster
    cluster_performance = analyze_cluster_performance(
        cluster_analysis_df,
        wallets_config['features']['clustering_n_clusters'],
        model_results['y_test'],  # True values
        model_results['y_pred']   # Predictions
    )

    # Join metrics with performance and reorder rows
    cluster_results_df = cluster_profiles[n].join(cluster_performance[n])

    # Convert capital F Float columns to normal float columns to fix styling failures
    float_cols = cluster_results_df.select_dtypes(['Float64','Float32']).columns
    type_map = {col: 'float64' for col in float_cols}

    # Pivot the df
    cluster_results_pivot = (
        cluster_results_df.copy()
        .astype(type_map)
        .T
    )

    # Define the desired row order
    size_metrics = ['cluster_size', 'cluster_pct', 'test_set_samples']
    perf_metrics = ['r2', 'rmse', 'mae', 'explained_variance']
    remaining_metrics = [col for col in cluster_results_pivot.index
                        if col not in size_metrics + perf_metrics + base_metrics]

    # Reorder the rows
    ordered_rows = (size_metrics + base_metrics + perf_metrics + remaining_metrics)
    cluster_results_pivot = cluster_results_pivot.reindex(ordered_rows)

    styled_df = style_rows(cluster_results_pivot)

    return styled_df,cluster_results_df
