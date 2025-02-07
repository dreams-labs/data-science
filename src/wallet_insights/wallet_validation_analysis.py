import logging
from typing import List, Dict
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    mean_absolute_percentage_error,
)

# Local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig
import wallet_insights.model_evaluation as wime

# plint:disable=invalid-name  # X isn't snake case

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()


# ----------------------------------------
#      Validation Period Predictions
# ----------------------------------------

def load_and_predict(model_id: str, new_data: pd.DataFrame, base_path: str) -> pd.Series:
    """
    Load a saved pipeline by UUID and generate predictions for new data.

    Params:
    - model_id (str): UUID of the saved model
    - new_data (pd.DataFrame): New feature data for predictions
    - base_path (str): Base path where model artifacts are stored

    Returns:
    - pd.Series: Predictions indexed by new_data index
    """
    pipeline_path = Path(base_path) / 'wallet_models' / f"wallet_model_{model_id}.pkl"

    if not pipeline_path.exists():
        raise FileNotFoundError(f"No pipeline found at {pipeline_path}")

    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)

    predictions = pd.Series(
        pipeline.predict(new_data),
        index=new_data.index
    )

    return predictions



def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Calculate core regression metrics for overlapping ids between y_true and y_pred.

    Params:
    - y_true (Series): Actual values with ids as index.
    - y_pred (Series): Predicted values with ids as index.

    Returns:
    - dict: Core performance metrics computed on overlapping ids.
    """
    # Identify common ids between y_true and y_pred
    common_idx = y_true.index.intersection(y_pred.index)
    if common_idx.empty:
        raise ValueError("No overlapping ids between y_true and y_pred")

    # Filter to only overlapping ids
    y_true_common = y_true.loc[common_idx]
    y_pred_common = y_pred.loc[common_idx]

    # Compute metrics using common indices
    metrics = {
        'r2': r2_score(y_true_common, y_pred_common),
        'rmse': np.sqrt(mean_squared_error(y_true_common, y_pred_common)),
        'mae': mean_absolute_error(y_true_common, y_pred_common),
        'explained_variance': explained_variance_score(y_true_common, y_pred_common)
    }

    # Residuals analysis
    residuals = y_true_common - y_pred_common
    metrics.update({
        'residuals_mean': residuals.mean(),
        'residuals_std': residuals.std(),
        'prediction_interval_95': 1.96 * residuals.std()
    })

    return metrics



# ----------------------------------------
#       Other Analytics Functions
# ----------------------------------------

def analyze_cohort_performance(performance_df: pd.DataFrame,
                             cohort_dict: Dict[str, List[str]],
                             comparison_metrics: List[str],
                             min_activity_threshold: float = 100) -> pd.DataFrame:
    """
    Analyzes cohort performance in a format similar to cluster analysis.

    Params:
    - performance_df (DataFrame): DataFrame containing wallet performance metrics
    - cohort_dict (Dict[str, List[str]]): Dictionary mapping cohort names to lists of wallet addresses
    - comparison_metrics (List[str]): Metrics to analyze
    - min_activity_threshold (float): Minimum USD volume to consider wallet active

    Returns:
    - DataFrame: Pivoted results with cohorts as columns, metrics as rows
    """
    # Validate inputs
    if not isinstance(performance_df, pd.DataFrame):
        raise ValueError("performance_df must be a pandas DataFrame")

    # Add activity flag
    performance_df['is_active'] = performance_df['total_volume'] > min_activity_threshold

    # Combine base metrics with comparison metrics and ensure they exist
    all_metrics = comparison_metrics
    missing_metrics = [m for m in all_metrics if m not in performance_df.columns]
    if missing_metrics:
        raise ValueError(f"Missing metrics in performance_df: {missing_metrics}")

    # Initialize results dictionary
    results = {}

    # Calculate metrics for the general population (excluding inactive)
    pop_df = performance_df[performance_df['is_active']]
    pop_size = len(pop_df)

    # For each cohort
    for cohort_name, wallet_list in cohort_dict.items():
        # Get cohort data
        cohort_df = performance_df[
            (performance_df.index.isin(wallet_list)) &
            (performance_df['is_active'])
        ]
        cohort_size = len(cohort_df)

        # Calculate basic size metrics
        size_metrics = {
            'cohort_size': cohort_size,
            'cohort_pct': np.round((cohort_size / pop_size * 100), 2)
        }

        # Calculate median metrics
        metric_medians = cohort_df[all_metrics].median()

        # Calculate model performance metrics if available
        if all(col in performance_df.columns for col in ['predicted', 'true_value']):
            perf_metrics = {
                'r2': r2_score(cohort_df['true_value'], cohort_df['predicted']),
                'rmse': np.sqrt(mean_squared_error(cohort_df['true_value'], cohort_df['predicted'])),
                'mae': mean_absolute_error(cohort_df['true_value'], cohort_df['predicted']),
                'mape': mean_absolute_percentage_error(cohort_df['true_value'], cohort_df['predicted']),
                'explained_variance': explained_variance_score(cohort_df['true_value'], cohort_df['predicted'])
            }
        else:
            perf_metrics = {}

        # Combine all metrics
        results[cohort_name] = {**size_metrics, **metric_medians.to_dict(), **perf_metrics}

    # Convert to DataFrame and transpose
    results_df = pd.DataFrame(results).T

    # Define row ordering similar to cluster report
    size_metrics = ['cohort_size', 'cohort_pct']
    perf_metrics = ['r2', 'rmse', 'mae', 'mape', 'explained_variance']
    remaining_metrics = [col for col in results_df.columns
                        if col not in size_metrics + perf_metrics]

    # Order rows
    ordered_rows = size_metrics + perf_metrics + remaining_metrics
    ordered_rows = [col for col in ordered_rows if col in results_df.columns]
    results_df = results_df.reindex(columns=ordered_rows)

    # Transpose to match cluster report format
    results_df = results_df.T

    return results_df


def create_cohort_report(performance_df: pd.DataFrame,
                        cohort_dict: Dict[str, List[str]],
                        comparison_metrics: List[str],
                        min_activity_threshold: float = 100) -> pd.DataFrame.style:
    """
    Creates a styled cohort analysis report similar to cluster report.

    Params:
    - performance_df (DataFrame): DataFrame containing wallet performance metrics
    - cohort_dict (Dict[str, List[str]]): Dictionary mapping cohort names to lists of wallet addresses
    - comparison_metrics (List[str]): Metrics to analyze
    - min_activity_threshold (float): Minimum USD volume to consider wallet active

    Returns:
    - styled_df (DataFrame.style): Styled DataFrame with cohort analysis
    """
    # Generate results DataFrame
    results_df = analyze_cohort_performance(
        performance_df,
        cohort_dict,
        comparison_metrics,
        min_activity_threshold
    )

    # Apply styling
    styled_df = wime.style_rows(results_df)

    return styled_df


def create_prediction_quantiles(
    model_scores: pd.Series,
    num_quantiles: int = 5
) -> Dict[str, List[str]]:
    """
    Splits wallets into equal-sized groups based on their prediction scores.

    Params:
    - model_scores (Series): Prediction scores (0-1) indexed by wallet_address
    - num_quantiles (int): Number of equal-sized groups to create

    Returns:
    - Dict mapping quantile names to lists of wallet addresses, including:
        - quantile_1 through quantile_N: Wallets in each quantile (lowest to highest)
        - all_wallets: All wallet addresses
    """
    labels = [f'quantile_{i+1}' for i in range(num_quantiles)]
    wallet_quantiles = pd.qcut(model_scores, num_quantiles, labels=labels)

    quantile_groups = {
        label: model_scores[wallet_quantiles == label].index.tolist()
        for label in labels
    }
    quantile_groups['all_wallets'] = model_scores.index.tolist()

    return quantile_groups


def create_quantile_report(
    wallet_metrics: pd.DataFrame,
    model_scores: pd.Series,
    metrics_to_compare: List[str],
    num_quantiles: int = 5,
    min_wallet_volume_usd: float = 100
) -> pd.DataFrame.style:
    """
    Creates a styled report comparing metrics across prediction score quantiles.

    Params:
    - wallet_metrics (DataFrame): Performance metrics for each wallet
    - model_scores (Series): Prediction scores (0-1) indexed by wallet_address
    - metrics_to_compare (List[str]): Metrics to analyze
    - num_quantiles (int): Number of equal-sized quantiles to create
    - min_wallet_volume_usd (float): Minimum USD volume to include wallet

    Returns:
    - styled_df (DataFrame.style): Styled comparison table
    """
    # Ensure indices match before filtering
    common_wallets = wallet_metrics.index.intersection(model_scores.index)
    wallet_metrics = wallet_metrics.loc[common_wallets]
    model_scores = model_scores[common_wallets]

    # Filter for minimum activity
    active_wallets = wallet_metrics[
        wallet_metrics['total_volume'] >= min_wallet_volume_usd
    ].index

    # Filter scores for active wallets
    active_scores = model_scores[active_wallets]

    # Create quantile groups from active wallets
    wallet_groups = create_prediction_quantiles(active_scores, num_quantiles)

    results = []
    for group_name, wallet_list in wallet_groups.items():
        group_data = {'group': group_name}

        # Add size metrics
        group_data['wallets_count'] = len(wallet_list)
        group_data['pct_of_wallets'] = (
            len(wallet_list) / len(active_wallets) * 100
        )

        # Add average prediction score
        group_data['avg_pred_score'] = active_scores[wallet_list].mean()

        # Add performance metrics
        for metric in metrics_to_compare:
            group_data[metric] = wallet_metrics.loc[
                wallet_list, metric
            ].mean()

        results.append(group_data)

    # Convert to DataFrame and transpose
    results_df = pd.DataFrame(results).set_index('group').T
    styled_df = wime.style_rows(results_df)

    return styled_df


def analyze_wallet_model_importance(feature_importances):
    """
    Splits the wallet model feature importance df into component columns.

    Params:
    - feature_importance (dict of lists of floats): Output of wallet_evaluator.metrics['importances'].
        Includes lists ['feature'] and ['importance']

    Returns:
    - feature_details_df (df): df with columns for all column components along with importance
    """
    # Convert lists to df
    feature_importance_df = pd.DataFrame(feature_importances)

    # Split on pipe delimiters
    split_df = feature_importance_df['feature'].str.split('|', expand=True)
    split_df.columns = ['feature_category','feature_details','training_segment']

    # Split nested components
    features_df = split_df['feature_details'].str.split('/', expand=True)
    features_df.columns = ['feature_name', 'feature_comparison', 'feature_aggregation']

    segments_df = split_df['training_segment'].str.split('/', expand=True)
    if len(segments_df.columns) == 1:
        segments_df.columns = ['training_segment']
    elif len(segments_df.columns) == 2:
        segments_df.columns = ['training_segment','record_type']
    else:
        raise ValueError("Unknown segments data components found")

    # Combine all components
    feature_details_df = pd.concat([
        split_df['feature_category'],
        features_df,
        segments_df,
        feature_importance_df['importance']
    ], axis=1)

    return feature_details_df



def calculate_tree_confidence(model, wallet_training_data_df: pd.DataFrame) -> pd.Series:
    """
    Calculate confidence scores using tree prediction variance.

    Params:
    - model (XGBRegressor): Trained XGBoost model
    - wallet_training_data_df (DataFrame): Training data for the full wallet cohort

    Returns:
    - confidence_scores (Series): Confidence score per prediction

    Adding to scores DataFrame:
        modeling_wallet_scores_df[f'confidence|{score_name}'] = confidence_scores
    """
    # Get per-tree predictions
    tree_preds = np.array([
        model.predict(wallet_training_data_df, iteration_range=(i, i+1), output_margin=True)
        for i in range(model.best_iteration)
    ]).T

    # Calculate confidence using prediction variance
    confidence_scores = pd.Series(
        1 - np.std(tree_preds, axis=1) / np.mean(tree_preds, axis=1),
        index=wallet_training_data_df.index
    )

    return confidence_scores
