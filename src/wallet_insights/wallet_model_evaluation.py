"""
Calculates metrics aggregated at the wallet level
"""
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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


class RegressionEvaluator:
    """
    A utility class for evaluating and visualizing regression model performance.

    Methods:
        get_summary_report(): Returns a formatted text summary of model performance
        plot_evaluation(plot_type='all'): Creates visualization plots of model performance

    Internal Methods:
        __init__(y_true, y_pred, model=None, feature_names=None): Initialize with actual and predicted values
        _calculate_metrics(): Computes regression performance metrics like RMSE, MAE, R2
        _plot_actual_vs_predicted(ax): Plots actual vs predicted values
        _plot_residuals(ax): Plots residuals vs predicted values
        _plot_residuals_distribution(ax): Plots histogram of residuals
        _plot_feature_importance(ax): Plots feature importance if available from model
    """
    def __init__(self, y_true, y_pred, model=None, feature_names=None):
        """
        Initialize the evaluator with actual and predicted values.

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
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.model = model
        self.feature_names = feature_names
        self.metrics = None
        self._calculate_metrics()

    def _calculate_metrics(self):
        """Calculate all regression metrics."""
        self.metrics = {}

        # Basic metrics
        self.metrics['mse'] = mean_squared_error(self.y_true, self.y_pred)
        self.metrics['rmse'] = np.sqrt(self.metrics['mse'])
        self.metrics['mae'] = mean_absolute_error(self.y_true, self.y_pred)
        self.metrics['mape'] = mean_absolute_percentage_error(self.y_true, self.y_pred) * 100
        self.metrics['r2'] = r2_score(self.y_true, self.y_pred)
        self.metrics['explained_variance'] = explained_variance_score(self.y_true, self.y_pred)

        # Additional statistical metrics
        self.residuals = self.y_true - self.y_pred
        self.metrics['residuals_mean'] = np.mean(self.residuals)
        self.metrics['residuals_std'] = np.std(self.residuals)

        # Calculate prediction intervals
        z_score = 1.96  # 95% confidence interval
        self.metrics['prediction_interval_95'] = z_score * self.metrics['residuals_std']

        # Calculate and sort feature importances if available
        if self.model is not None and hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            if self.feature_names is None:
                self.feature_names = [f'Feature {i}' for i in range(len(importances))]

            # Create list of (feature, importance) pairs and sort by importance
            feature_importance_pairs = list(zip(self.feature_names, importances))
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

            # Unzip the sorted pairs
            sorted_features, sorted_values = zip(*feature_importance_pairs)

            # Store sorted values
            self.metrics['importances'] = {
                'feature': list(sorted_features),  # Convert tuple to list
                'importance': list(sorted_values)      # Convert tuple to list
            }


    def get_summary_report(self):
        """Generate and return a formatted text summary of the model's performance."""
        summary = [
            "Model Performance Summary",
            "=" * 25,
            f"R² Score:                    {self.metrics['r2']:.3f}",
            f"RMSE:                        {self.metrics['rmse']:.3f}",
            f"MAE:                         {self.metrics['mae']:.3f}",
            f"MAPE:                        {self.metrics['mape']:.1f}%",
            "",
            "Residuals Analysis",
            "=" * 25,
            f"Mean of Residuals:           {self.metrics['residuals_mean']:.3f}",
            f"Standard Dev of Residuals:   {self.metrics['residuals_std']:.3f}",
            f"95% Prediction Interval:     ±{self.metrics['prediction_interval_95']:.3f}"
        ]

        print("\n".join(summary))


    def plot_evaluation(self, plot_type='all', display=True):
        """
        Generate and display specific evaluation plots.

        Parameters:
        -----------
        plot_type : str
            Type of plot to display: 'actual_vs_predicted', 'residuals',
            'residuals_dist', 'feature_importance', or 'all'
        display : bool, default=True
            If True, displays the plot using plt.show(). If False, returns the figure.
        """
        # Dark mode charts
        plt.rcParams['figure.facecolor'] = '#181818'
        plt.rcParams['axes.facecolor'] = '#181818'
        plt.rcParams['text.color'] = '#afc6ba'
        plt.rcParams['axes.labelcolor'] = '#afc6ba'
        plt.rcParams['xtick.color'] = '#afc6ba'
        plt.rcParams['ytick.color'] = '#afc6ba'
        plt.rcParams['axes.titlecolor'] = '#afc6ba'

        # Create custom colormap that starts from background color
        self.custom_cmap = mcolors.LinearSegmentedColormap.from_list(  # pylint:disable=attribute-defined-outside-init
            'custom_blues', ['#181818', '#145a8d', '#69c4ff']
        )

        if plot_type == 'all':
            fig = plt.figure(figsize=(15, 12))
            gs = plt.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])

            self._plot_actual_vs_predicted(ax1)
            self._plot_residuals(ax2)
            self._plot_residuals_distribution(ax3)
            self._plot_feature_importance(ax4)
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            if plot_type == 'actual_vs_predicted':
                self._plot_actual_vs_predicted(ax)
            elif plot_type == 'residuals':
                self._plot_residuals(ax)
            elif plot_type == 'residuals_dist':
                self._plot_residuals_distribution(ax)
            elif plot_type == 'feature_importance':
                self._plot_feature_importance(ax)

        plt.tight_layout()

        if display:
            plt.show()
            return None
        return fig

    def _plot_actual_vs_predicted(self, ax):
        """Plot actual vs predicted values using hexbin without colorbar."""
        # Set the same range for both axes to ensure hexagons are not stretched
        extent = [
            min(self.y_true.min(), self.y_pred.min()),
            max(self.y_true.max(), self.y_pred.max()),
            min(self.y_true.min(), self.y_pred.min()),
            max(self.y_true.max(), self.y_pred.max())
        ]

        # Create hexbin plot with defined extent
        ax.hexbin(self.y_true, self.y_pred,
                gridsize=50,
                cmap=self.custom_cmap,
                mincnt=1,
                bins='log',
                extent=extent)

        # Add diagonal reference line
        ax.plot([self.y_true.min(), self.y_true.max()],
                [self.y_true.min(), self.y_true.max()],
                'r--', lw=2)

        ax.set_xlim(extent[:2])  # Set x-axis limits
        ax.set_ylim(extent[2:])  # Set y-axis limits

        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Actual vs Predicted Values')

    def _plot_residuals(self, ax):
        """Plot residuals vs predicted values using hexbin without colorbar."""
        # Create hexbin plot
        ax.hexbin(self.y_pred, self.residuals,
                gridsize=50,
                cmap=self.custom_cmap,
                mincnt=1,
                bins='log')

        # Add horizontal reference line at y=0
        ax.axhline(y=0, color='r', linestyle='--')

        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Predicted Values')

    def _plot_residuals_distribution(self, ax):
        """Plot distribution of residuals."""
        sns.histplot(self.residuals, kde=True, ax=ax)
        ax.set_title('Distribution of Residuals')

    def _plot_feature_importance(self, ax):
        """Plot feature importance if available."""
        if 'importances' in self.metrics:
            # Create DataFrame with lowercase column names
            df = pd.DataFrame(self.metrics['importances']).head(20)

            # Plot with uppercase axis labels
            sns.barplot(
                data=df,
                x='importance',  # lowercase to match DataFrame
                y='feature',     # lowercase to match DataFrame
                ax=ax
            )

            # Set uppercase axis labels
            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
            ax.set_title('Top 20 Feature Importances')
        else:
            ax.text(0.5, 0.5, 'Feature Importance Not Available',
                ha='center', va='center')
