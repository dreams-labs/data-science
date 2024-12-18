import logging
from typing import List, Dict
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
import wallet_insights.wallet_model_evaluation as wime

# plint:disable=invalid-name  # X isn't snake case

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()



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


def create_prediction_cohorts(y_pred: pd.Series,
                            n_bands: int = 5) -> Dict[str, List[str]]:
    """
    Creates cohorts based on prediction score ranges.

    Params:
    - y_pred (Series): Predicted values, indexed by wallet_address
    - n_bands (int): Number of bands to split predictions into

    Returns:
    - Dict mapping band names to lists of wallet addresses
    """
    # Calculate band boundaries
    quantiles = np.linspace(0, 1, n_bands + 1)
    boundaries = np.quantile(y_pred, quantiles)

    # Initialize cohorts
    cohorts = {}

    # Create each band
    for i in range(n_bands):
        band_name = f"pred_{i+1}"
        if i == 0:
            mask = (y_pred >= boundaries[i]) & (y_pred <= boundaries[i+1])
        else:
            mask = (y_pred > boundaries[i]) & (y_pred <= boundaries[i+1])
        cohorts[band_name] = y_pred[mask].index.tolist()

    # Add full population
    cohorts['population'] = y_pred.index.tolist()

    return cohorts


def analyze_prediction_bands(validation_performance_df: pd.DataFrame,
                           y_pred: pd.Series,
                           metrics: List[str],
                           n_bands: int = 5,
                           min_activity_threshold: float = 100) -> pd.DataFrame.style:
    """
    Analyzes performance across prediction score bands.

    Params:
    - validation_performance_df (DataFrame): Validation period performance metrics
    - y_pred (Series): Model predictions
    - metrics (List[str]): Metrics to analyze
    - n_bands (int): Number of prediction bands to create
    - min_activity_threshold (float): Minimum activity threshold

    Returns:
    - styled_df (DataFrame.style): Styled analysis results
    """
    # Create prediction-based cohorts
    pred_cohorts = create_prediction_cohorts(y_pred, n_bands)

    # Add mean prediction to results
    validation_performance_df = validation_performance_df.copy()
    validation_performance_df['predicted_score'] = y_pred

    # Add to base metrics
    metrics = ['predicted_score'] + metrics

    # Generate report
    styled_df = create_cohort_report(
        validation_performance_df,
        pred_cohorts,
        metrics,
        min_activity_threshold
    )

    return styled_df
