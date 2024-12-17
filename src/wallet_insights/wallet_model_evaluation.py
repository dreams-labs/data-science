"""
Calculates metrics aggregated at the wallet level
"""
import logging
from typing import List,Dict
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
from dreams_core import core as dc

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
        summary_report(): Returns a formatted text summary of model performance
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
        self.metrics['mape'] = mean_absolute_percentage_error(self.y_true, self.y_pred)
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


    def summary_report(self):
        """
        Generate and return a formatted text summary of the model's performance.

        Returns:
        - str: Formatted summary of model metrics
        """
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

        return "\n".join(summary)



    def importance_summary(self):
        """
        Generate and return a df showing total importance by feature category and the best performing
        feature in each category.

        Returns:
        - importance_summary_df (df): formatted df showing importance metrics for each feature category
        """

        feature_importance_df = pd.DataFrame(self.metrics['importances'])

        # Extract prefix before first underscore
        feature_importance_df['prefix'] = feature_importance_df['feature'].str.split('_').str[0]

        # Calculate total_importance by summing importance for each prefix
        importance_summary_df = feature_importance_df.groupby('prefix').agg(
            total_importance=('importance', 'sum'),
        )

        # Take the highest importance feature from each prefix
        highest_importances_df = (feature_importance_df
                                .sort_values(by='importance', ascending=False)
                                .groupby('prefix')
                                .first())
        highest_importances_df.columns = ['best_feature','best_importance']

        # Join the total importances with the highest importances
        importance_summary_df = (importance_summary_df
                            .join(highest_importances_df)
                            .sort_values(by='total_importance', ascending=False))

        # Format output
        importance_summary_df = (importance_summary_df
                                .rename(columns={
                                    'total_importance': 'Total Importance',
                                    'best_feature': 'Best Feature',
                                    'best_importance': 'Best Importance'
                                })
                                .style.format({
                                    'Total Importance': '{:.3f}',
                                    'Best Importance': '{:.3f}'
                                }))

        return importance_summary_df



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
        """Plot feature importance if available with color-coded feature prefixes."""
        if 'importances' in self.metrics:
            # Create DataFrame with feature prefixes
            df = pd.DataFrame(self.metrics['importances']).head(20)
            df['prefix'] = df['feature'].str.split('_').str[0]

            # Create color palette for prefixes
            unique_prefixes = df['prefix'].unique()
            palette = dict(zip(unique_prefixes, sns.color_palette("husl", len(unique_prefixes))))

            # Plot using hue instead of explicit colors
            sns.barplot(
                data=df,
                x='importance',
                y='feature',
                ax=ax,
                hue='prefix',
                palette=palette
            )

            # Adjust legend position
            ax.legend(title='Feature Type', bbox_to_anchor=(1.05, 1), loc='upper left')

            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
            ax.set_title('Top 20 Feature Importances')
        else:
            ax.text(0.5, 0.5, 'Feature Importance Not Available',
                ha='center', va='center')



# ---------------------------------------------
#             Cluster Analysis Tools
# ---------------------------------------------

def assign_clusters_from_distances(modeling_df: pd.DataFrame, cluster_counts: List[int]) -> pd.DataFrame:
    """
    Assign clusters based on minimum distances for each k in cluster_counts.

    Params:
    - modeling_df (DataFrame): DataFrame with distance features, indexed by wallet_address
    - cluster_counts (List[int]): List of k values to process [e.g. 2, 4]

    Returns:
    - modeling_df (DataFrame): Original df with new cluster assignment columns
    """
    for k in cluster_counts:
        # Get distance columns for this k value
        distance_cols = [f'cluster_k{k}_distance_to_cluster_{i}' for i in range(k)]

        # Assign cluster based on minimum distance
        modeling_df[f'k{k}_cluster'] = (
            modeling_df[distance_cols]
            .idxmin(axis=1)
            .str[-1]
            .astype(int)
        )

    return modeling_df


def analyze_cluster_metrics(modeling_df: pd.DataFrame,
                         cluster_counts: List[int],
                         comparison_metrics: List[str]) -> Dict[int, pd.DataFrame]:
    """
    Calculate median metrics for each cluster grouping.

    Params:
    - modeling_df (DataFrame): DataFrame with cluster assignments and metrics
    - cluster_counts (List[int]): List of k values [e.g. 2, 4]
    - comparison_metrics (List[str]): Metrics to analyze

    Returns:
    - Dict[int, DataFrame]: Median metrics for each k value's clusters
    """
    results = {}

    for k in cluster_counts:
        cluster_col = f'k{k}_cluster'

        # Calculate medians for each metric by cluster
        medians = modeling_df.groupby(cluster_col)[comparison_metrics].median()

        # Add cluster size info
        cluster_sizes = modeling_df[cluster_col].value_counts()
        medians['cluster_size'] = cluster_sizes
        medians['cluster_pct'] = (cluster_sizes / len(modeling_df) * 100).round(2)

        results[k] = medians

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
    cluster_required_sample_size = 2

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

            # If a cluster has fewer than the required samples, log a warning and return NaN metrics
            if n_samples < cluster_required_sample_size:
                logger.warning(
                    f"Cluster {cluster} (k={k}) has only {n_samples} samples - insufficient for reliable metrics. "
                    f"Metrics will be reported as NaN."
                )
                cluster_metrics = {
                    'pct_total': 0.0,
                    'r2': np.nan,
                    'rmse': np.nan,
                    'mae': np.nan,
                    'mape': np.nan,
                    'explained_variance': np.nan,
                    'n_samples': n_samples
                }

            # Otherwise calculate metrics for the cluster
            else:
                cluster_metrics = {
                    'pct_total': (cluster_mask.sum() / len(test_wallets)),
                    'r2': r2_score(y_true_arr[cluster_mask], y_pred_arr[cluster_mask]),
                    'rmse': np.sqrt(mean_squared_error(y_true_arr[cluster_mask], y_pred_arr[cluster_mask])),
                    'mae': mean_absolute_error(y_true_arr[cluster_mask], y_pred_arr[cluster_mask]),
                    'mape': mean_absolute_percentage_error(y_true_arr[cluster_mask], y_pred_arr[cluster_mask]),
                    'explained_variance': explained_variance_score(y_true_arr[cluster_mask], y_pred_arr[cluster_mask])
                }
                metrics_by_cluster.append(cluster_metrics)

        results[k] = pd.DataFrame(metrics_by_cluster)

    return results


def format_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply human_format to all numeric columns in dataframe.

    Params:
    - df (DataFrame): Input dataframe with numeric columns

    Returns:
    - DataFrame: Copy of input with formatted numeric columns
    """
    # Create copy to avoid modifying original
    formatted_df = df.copy()

    # Only apply to numeric columns that aren't just 0/1 categories
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if df[col].nunique() > 2:  # Skip binary columns
            formatted_df[col] = df[col].apply(dc.human_format)

    return formatted_df
