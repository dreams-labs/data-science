"""
Calculates metrics aggregated at the wallet level
"""
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    mean_absolute_percentage_error
)

# local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()


def evaluate_regression_model(y_true, y_pred, model=None, feature_names=None, show_plot=True):
    """
    Comprehensive evaluation of regression model performance.

    Parameters:
    -----------
    y_true : array-like
        Actual target values
    y_pred : array-like
        Predicted target values
    model : sklearn estimator, optional
        The fitted model object
    feature_names : list, optional
        List of feature names for feature importance plot
    show_plot : bool, optional (default=True)
        Whether to generate and display the visualization plots

    Returns:
    --------
    dict
        Dictionary containing various performance metrics
    """
    metrics = {}

    # Basic metrics
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100
    metrics['r2'] = r2_score(y_true, y_pred)
    metrics['explained_variance'] = explained_variance_score(y_true, y_pred)

    # Additional statistical metrics
    residuals = y_true - y_pred
    metrics['residuals_mean'] = np.mean(residuals)
    metrics['residuals_std'] = np.std(residuals)

    # Calculate prediction intervals (assuming normal distribution of residuals)
    z_score = 1.96  # 95% confidence interval
    prediction_interval = z_score * metrics['residuals_std']
    metrics['prediction_interval_95'] = prediction_interval

    if show_plot:
        try:
            # Generate visualizations
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Actual vs Predicted Plot
            axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
            axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('Actual Values')
            axes[0, 0].set_ylabel('Predicted Values')
            axes[0, 0].set_title('Actual vs Predicted Values')

            # Residuals Plot
            axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('Predicted Values')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residuals vs Predicted Values')

            # Residuals Distribution
            sns.histplot(residuals, kde=True, ax=axes[1, 0])
            axes[1, 0].set_title('Distribution of Residuals')

            # Feature Importance Plot (if applicable)
            if model is not None and hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                if feature_names is None:
                    feature_names = [f'Feature {i}' for i in range(len(importances))]

                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)  # Changed to descending order

                # Store in dictionary
                metrics['importances'] = importance_df

                # Using barplot instead of barh
                sns.barplot(data=importance_df.head(20), x='Importance', y='Feature', ax=axes[1, 1])
                axes[1, 1].set_title('Top 25 Feature Importances')
            else:
                axes[1, 1].text(0.5, 0.5, 'Feature Importance Not Available',
                               ha='center', va='center')

            plt.tight_layout()
            metrics['figures'] = fig
        except Exception as e:  # pylint:disable=broad-exception-caught
            print(f"Warning: Error generating plots: {str(e)}")
            metrics['figures'] = None
    else:
        metrics['figures'] = None

    # Generate summary report
    metrics['summary_report'] = f"""
    Model Performance Summary:
    -------------------------
    R² Score: {metrics['r2']:.3f}
    RMSE: {metrics['rmse']:.3f}
    MAE: {metrics['mae']:.3f}
    MAPE: {metrics['mape']:.1f}%

    Residuals Analysis:
    ------------------
    Mean of Residuals: {metrics['residuals_mean']:.3f}
    Standard Deviation of Residuals: {metrics['residuals_std']:.3f}
    95% Prediction Interval: ±{metrics['prediction_interval_95']:.3f}
    """

    return metrics
