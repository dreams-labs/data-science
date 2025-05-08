import logging
from typing import List
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    auc,
    log_loss,
    RocCurveDisplay,
)
from dreams_core import core as dc
import utils as u

# pylint:disable=invalid-name  # X_test isn't camelcase
# pylint:disable=too-many-lines  # graphs gotta live somewhere

# Set up logger at the module level
logger = logging.getLogger(__name__)


class RegressorEvaluator:
    """
    A utility class for evaluating and visualizing regression model performance.

    Methods:
        summary_report(): Returns a formatted text summary of model performance
        plot_evaluation(plot_type='all'): Creates visualization plots of model performance
    """
    def __init__(self, wallet_model_results: dict):
        """
        Params:
        - wallet_model_results (dict): output of wallet_model.construct_wallet_model

        Required keys:
        'y_train','y_test','y_pred',
        'training_cohort_pred','training_cohort_actuals',
        'pipeline','X_test'
        """
        # core arrays
        self.y_test  = wallet_model_results['y_test']
        self.y_pred  = wallet_model_results['y_pred']
        self.y_train = wallet_model_results['y_train']
        self.training_cohort_pred     = wallet_model_results.get('training_cohort_pred')
        self.training_cohort_actuals  = wallet_model_results.get('training_cohort_actuals')

        # validation (if present)
        self.X_validation      = wallet_model_results.get('X_validation')
        self.y_validation      = wallet_model_results.get('y_validation')
        self.y_validation_pred = wallet_model_results.get('y_validation_pred')
        self.validation_wallet_features_df = wallet_model_results.get('validation_wallet_features_df')

        # model + features
        self.modeling_config = wallet_model_results['modeling_config']
        pipeline = wallet_model_results['pipeline']
        self.model = pipeline.named_steps['estimator']
        self.feature_names = (
            pipeline[:-1].transform(wallet_model_results['X_train']).columns.tolist()
            if hasattr(pipeline[:-1], 'transform') else None
        )

        # raw X_test for cohort methods
        self.X_test = wallet_model_results['X_test']

        # init metrics & styling
        self.metrics = {}
        self._calculate_metrics()
        self._setup_plot_style()

        # model id
        self.model_id = wallet_model_results['model_id']


    def _calculate_metrics(self):
        """Calculate core regression metrics and optional cohort metrics."""
        # Calculate residuals
        self.residuals = self.y_test - self.y_pred

        # Core test set metrics
        self.metrics['mse'] = mean_squared_error(self.y_test, self.y_pred)
        self.metrics['rmse'] = np.sqrt(self.metrics['mse'])
        self.metrics['mae'] = mean_absolute_error(self.y_test, self.y_pred)
        self.metrics['r2'] = r2_score(self.y_test, self.y_pred)
        self.metrics['explained_variance'] = explained_variance_score(self.y_test, self.y_pred)

        # Residuals analysis
        self.metrics['residuals_mean'] = np.mean(self.residuals)
        self.metrics['residuals_std'] = np.std(self.residuals)
        self.metrics['prediction_interval_95'] = 1.96 * self.metrics['residuals_std']

        # Sample size tracking
        self.metrics['test_samples'] = len(self.y_pred)
        if self.y_train is not None:
            self.metrics['train_samples'] = len(self.y_train)

        # Training cohort metrics if available
        if (self.training_cohort_pred is not None and
            self.training_cohort_actuals is not None):
            self.metrics['training_cohort'] = {
                'mse': mean_squared_error(
                    self.training_cohort_actuals,
                    self.training_cohort_pred
                ),
                'rmse': np.sqrt(mean_squared_error(
                    self.training_cohort_actuals,
                    self.training_cohort_pred
                )),
                'mae': mean_absolute_error(
                    self.training_cohort_actuals,
                    self.training_cohort_pred
                ),
                'r2': r2_score(
                    self.training_cohort_actuals,
                    self.training_cohort_pred
                )
            }
            self.metrics['total_cohort_samples'] = len(self.training_cohort_actuals)

            # Add cohort comparison metrics
            self._calculate_cohort_metrics()

        # Feature importance if available
        if self.model is not None and hasattr(self.model, 'feature_importances_'):
            self._calculate_feature_importance()

        # Validation set performance if applicable
        if self.y_validation is not None and self.y_validation_pred is not None:
            self.metrics['validation_metrics'] = {
                'r2': r2_score(self.y_validation, self.y_validation_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_validation, self.y_validation_pred)),
                'mae': mean_absolute_error(self.y_validation, self.y_validation_pred)
            }

    def _calculate_cohort_metrics(self):
        """Calculate comparison metrics between modeling and training cohorts."""
        self.metrics['cohort_comparison'] = {
            'modeling_mean': np.mean(self.y_pred),
            'modeling_std': np.std(self.y_pred),
            'modeling_q25': np.percentile(self.y_pred, 25),
            'modeling_q75': np.percentile(self.y_pred, 75),
            'training_mean_pred': np.mean(self.training_cohort_pred),
            'training_std_pred': np.std(self.training_cohort_pred),
            'training_q25_pred': np.percentile(self.training_cohort_pred, 25),
            'training_q75_pred': np.percentile(self.training_cohort_pred, 75),
            'training_mean_actual': np.mean(self.training_cohort_actuals),
            'training_std_actual': np.std(self.training_cohort_actuals),
            'training_q25_actual': np.percentile(self.training_cohort_actuals, 25),
            'training_q75_actual': np.percentile(self.training_cohort_actuals, 75)
        }

    def _calculate_feature_importance(self):
        """Calculate and sort feature importance metrics."""
        importances = self.model.feature_importances_
        if self.feature_names is None:
            self.feature_names = [f'Feature {i}' for i in range(len(importances))]

        # Sort feature importances
        feature_importance_pairs = list(zip(self.feature_names, importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        sorted_features, sorted_values = zip(*feature_importance_pairs)

        self.metrics['importances'] = {
            'feature': list(sorted_features),
            'importance': list(sorted_values)
        }

    def _setup_plot_style(self):
        """Configure matplotlib style for dark mode plots."""
        plt.rcParams['figure.facecolor'] = '#181818'
        plt.rcParams['axes.facecolor'] = '#181818'
        plt.rcParams['text.color'] = '#afc6ba'
        plt.rcParams['axes.labelcolor'] = '#afc6ba'
        plt.rcParams['xtick.color'] = '#afc6ba'
        plt.rcParams['ytick.color'] = '#afc6ba'
        plt.rcParams['axes.titlecolor'] = '#afc6ba'

        # Increase default font sizes for all plots
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['xtick.labelsize'] = 16
        plt.rcParams['ytick.labelsize'] = 13

        self.custom_cmap = mcolors.LinearSegmentedColormap.from_list(
            'custom_blues', ['#1b2530', '#145a8d', '#ddeeff']
        )


    def _get_summary_header(self) -> list[str]:
        """
        Build header lines including title, target, ID, samples and feature counts.
        """
        # Include class threshold if it's a classification model
        if self.modeling_config['model_type'] == 'classification':
            class_threshold_str = (f"{self.modeling_config.get('target_var_min_threshold', '')} to "
                                   f"{self.modeling_config.get('target_var_max_threshold', '')}")
        else:
            class_threshold_str = ''

        header = [
            "Model Performance Summary",
            f"Target: {self.modeling_config['target_variable']} {class_threshold_str}",
            f"ID: {self.model_id}",
            "=" * 35,
        ]
        # feature counts
        n_features = len(self.feature_names) if self.feature_names is not None else 0
        n_per_window = sum(1 for f in self.feature_names if "|w2" in f)
        if n_per_window > 0:
            window_n_features_str = f"Features per Window:      {n_per_window:,d}\n"
        else:
            window_n_features_str = '\n'

        if "total_cohort_samples" in self.metrics:
            header.extend([
                f"Training Cohort:          {self.metrics['total_cohort_samples']:,d}",
                f"Modeling Cohort Train:    {self.metrics['train_samples']:,d}",
                f"Modeling Cohort Test:     {self.metrics['test_samples']:,d}",
                ""
            ])
        else:
            header.extend([
                f"Test Samples:             {self.metrics['test_samples']:,d}",
                f"Number of Features:       {n_features:,d}",
                window_n_features_str,
            ])
        return header


    def summary_report(self):
        """Generate formatted summary of model performance."""
        # now just grab the header + samples
        summary = self._get_summary_header()

        # Add core regression metrics
        summary.extend([
            "Core Metrics",
            "-" * 35,
            f"R² Score:                 {self.metrics['r2']:.3f}",
            f"RMSE:                     {self.metrics['rmse']:.3f}",
            f"MAE:                      {self.metrics['mae']:.3f}",
            ""
        ])

        # Validation metrics
        if "validation_metrics" in self.metrics:
            vm = self.metrics["validation_metrics"]
            summary.extend([
                "Validation Set Metrics",
                "-" * 35,
                f"R² Score:                 {vm['r2']:.3f}",
                f"RMSE:                     {vm['rmse']:.3f}",
                f"MAE:                      {vm['mae']:.3f}",
                ""
            ])

        # Training-cohort metrics
        if "training_cohort" in self.metrics:
            tc = self.metrics["training_cohort"]
            summary.extend([
                "Inactive Wallets Cohort Metrics",
                "-" * 35,
                f"R² Score:                 {tc['r2']:.3f}",
                f"RMSE:                     {tc['rmse']:.3f}",
                f"MAE:                      {tc['mae']:.3f}",
                ""
            ])

        # Residuals
        summary.extend([
            "Residuals Analysis",
            "-" * 35,
            f"Mean of Residuals:        {self.metrics['residuals_mean']:.3f}",
            f"Std of Residuals:         {self.metrics['residuals_std']:.3f}",
            f"95% Prediction Interval:  ±{self.metrics['prediction_interval_95']:.3f}"
        ])

        report = "\n".join(summary)
        logger.info("\n%s", report)


    def importance_summary(self, levels=0):
        """
        Generate feature importance summary with configurable grouping levels.

        Params:
        - levels: Depth of feature name splitting (0=category, 1=subcategory, 2=component)

        Returns:
        - DataFrame with importance metrics by feature group
        """
        if 'importances' not in self.metrics:
            return pd.DataFrame()

        feature_importance_df = pd.DataFrame(self.metrics['importances'])

        # Split feature names based on level
        level_0_split = feature_importance_df['feature'].str.split('|', expand=True)
        level_1_split = level_0_split[1].str.split('/', expand=True)

        # Assign prefix based on requested level
        if levels == 0:
            feature_importance_df['prefix'] = level_0_split[0]
        elif levels == 1:
            feature_importance_df['prefix'] = level_0_split[0] + '|' + level_1_split[0]
        elif levels == 2:
            feature_importance_df['prefix'] = (level_0_split[0] + '|' +
                                            level_1_split[0] + '/' + level_1_split[1])

        # Calculate aggregated metrics
        importance_summary_df = feature_importance_df.groupby('prefix').agg(
            total_importance=('importance', 'sum'),
            total_features=('importance', 'count')
        )

        # Get highest importance feature for each prefix
        highest_importances_df = (feature_importance_df
                                .sort_values(by='importance', ascending=False)
                                .groupby('prefix')
                                .first())
        highest_importances_df.columns = ['best_feature', 'best_importance']

        # Combine and format results
        importance_summary_df = (importance_summary_df
                               .join(highest_importances_df)
                               .sort_values(by='total_importance', ascending=False)
                               .rename(columns={
                                   'total_importance': 'Total Importance',
                                   'total_features': 'Total Features',
                                   'best_feature': 'Best Feature',
                                   'best_importance': 'Best Importance'
                               })
                               .style.format({
                                   'Total Importance': '{:.3f}',
                                   'Best Importance': '{:.3f}'
                               }))

        return importance_summary_df


    def _plot_actual_vs_predicted(self, ax):
        """Plot actual vs predicted values using hexbin."""
        extent = [
            min(self.y_test.min(), self.y_pred.min()),
            max(self.y_test.max(), self.y_pred.max()),
            min(self.y_test.min(), self.y_pred.min()),
            max(self.y_test.max(), self.y_pred.max())
        ]

        ax.hexbin(self.y_test, self.y_pred,
                gridsize=50,
                cmap=self.custom_cmap,
                mincnt=1,
                bins=10**np.linspace(-1, 2, 20),
                extent=extent)

        ax.plot([self.y_test.min(), self.y_test.max()],
                [self.y_test.min(), self.y_test.max()],
                'r--', lw=2)

        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Actual vs Predicted Values')

    def _plot_residuals(self, ax):
        """Plot residuals vs predicted values using hexbin."""
        ax.hexbin(self.y_pred, self.residuals,
                gridsize=50,
                cmap=self.custom_cmap,
                mincnt=1,
                bins=10**np.linspace(.5, 6, 50))

        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Predicted Values')

    def _plot_feature_importance(self, ax, levels=0):
        """
        Plot prefix-level aggregated feature importance using the existing
        importance_summary function.

        Params:
        - ax: Matplotlib axis to draw on.
        - levels: Prefix splitting depth (0=top-level, 1=next, etc.)
        """
        # Call the already implemented importance_summary method
        summary_styler = self.importance_summary(levels=levels)
        # Extract the underlying DataFrame (if a Styler is returned)
        if hasattr(summary_styler, "data"):
            summary_df = summary_styler.data.copy()
        else:
            summary_df = summary_styler.copy()

        # Reset the index so that the prefix becomes a column
        summary_df = summary_df.reset_index().rename(columns={'prefix': 'Prefix'})

        if summary_df.empty:
            ax.text(0.5, 0.5, 'Feature Importance Not Available',
                    ha='center', va='center')
            return

        sns.barplot(
            data=summary_df,
            x='Total Importance',
            y='Prefix',
            ax=ax,
            color='#145a8d'
        )
        ax.set_title(f'Feature Importance by Prefix (levels={levels})')
        ax.set_xlabel('Total Importance')
        ax.set_ylabel('Prefix')

    def _plot_score_distribution(self, ax):
        """Basic distribution plot of actual vs predicted values."""
        sns.kdeplot(data=self.y_test, ax=ax, label='Actual', color='#69c4ff')
        sns.kdeplot(data=self.y_pred, ax=ax, label='Predicted', color='#ff6969')

        ax.axvline(np.mean(self.y_test), color='#69c4ff', linestyle='--', alpha=0.5)
        ax.axvline(np.mean(self.y_pred), color='#ff6969', linestyle='--', alpha=0.5)

        ax.set_title('Score Distribution')
        ax.set_xlabel('Values')
        ax.set_ylabel('Density')
        ax.legend()

    def _plot_cohort_comparison(self, ax):
        """Enhanced distribution plot comparing training and modeling cohorts."""
        if not hasattr(self, 'training_cohort_pred'):
            raise ValueError("Cohort comparison requires training cohort data")

        sns.kdeplot(data=self.y_pred, ax=ax,
                   label='Modeling Cohort (pred)', color='#69c4ff')
        sns.kdeplot(data=self.training_cohort_pred, ax=ax,
                   label='Inactive Wallets (pred)', color='#ff6969')
        sns.kdeplot(data=self.training_cohort_actuals, ax=ax,
                   label='Actual Values', color='#69ff69')

        # Add mean lines
        ax.axvline(np.mean(self.y_pred), color='#69c4ff', linestyle='--', alpha=0.5)
        ax.axvline(np.mean(self.training_cohort_pred),
                  color='#ff6969', linestyle='--', alpha=0.5)
        ax.axvline(np.mean(self.training_cohort_actuals),
                  color='#69ff69', linestyle='--', alpha=0.5)

        # Add quartile markers
        for q in [25, 75]:
            ax.axvline(np.percentile(self.y_pred, q),
                      color='#69c4ff', linestyle=':', alpha=0.3)
            ax.axvline(np.percentile(self.training_cohort_pred, q),
                      color='#ff6969', linestyle=':', alpha=0.3)
            ax.axvline(np.percentile(self.training_cohort_actuals, q),
                      color='#69ff69', linestyle=':', alpha=0.3)

        ax.set_title('Score Distribution by Cohort')
        ax.set_xlabel('Values')
        ax.set_ylabel('Density')
        ax.legend()


    def _plot_return_vs_rank(self, ax, n_buckets: int = 100):
        """
        Create a line chart showing mean returns by prediction-rank bucket.
        Bucket 1 = top-score wallets (left), bucket n = lowest (right).
        Y-axis shows mean actual return in each bucket (validation set).
        """
        # need validation preds + raw returns
        if self.y_validation_pred is None or self.validation_wallet_features_df is None:
            ax.text(0.5, 0.5, "Validation data not available",
                    ha="center", va="center")
            return

        target_var = self.modeling_config["target_variable"]

        # Create a DataFrame with predictions and target values
        # Handle both NumPy arrays and pandas Series/DataFrames
        if hasattr(self.y_validation_pred, 'index'):  # pandas Series/DataFrame
            # Use pandas index for alignment
            returns = self.validation_wallet_features_df[target_var].reindex(
                self.y_validation_pred.index
            )
            df = pd.DataFrame({"pred": self.y_validation_pred, "ret": returns}).dropna()
        else:  # NumPy array
            # Since we don't have index, use positions - assumes validation_wallet_features_df is aligned
            returns = self.validation_wallet_features_df[target_var].values
            df = pd.DataFrame({
                "pred": self.y_validation_pred,
                "ret": returns
            }).dropna()

        # Equal-count buckets on predictions
        try:
            df["bucket_raw"] = pd.qcut(df["pred"], n_buckets, labels=False, duplicates="drop")
        except ValueError:  # very few unique preds → fall back
            n_unique = df["pred"].nunique()
            nb = max(2, min(n_buckets, n_unique))
            df["bucket_raw"] = pd.qcut(df["pred"], nb, labels=False, duplicates="drop")
            n_buckets = nb

        # Re-index so bucket 1 = highest scores
        df["bucket"] = n_buckets - df["bucket_raw"]

        bucket_mean = df.groupby("bucket")["ret"].mean().reindex(range(1, n_buckets + 1))
        bucket_median = df.groupby("bucket")["ret"].median().reindex(range(1, n_buckets + 1))
        overall_mean = df["ret"].mean()
        overall_median = df["ret"].median()

        # Calculate y-axis limits with buffer
        min_val = min(bucket_mean.min(), overall_mean, overall_median)
        max_val = max(bucket_mean.max(), overall_mean, overall_median)
        y_range = max_val - min_val

        # Add buffer (20% on each side)
        buffer = y_range * 0.2
        y_min = min_val - buffer
        y_max = max_val + buffer

        # Plot as line chart with markers
        ax.plot(bucket_mean.index, bucket_mean.values,
                marker='o', markersize=5, linewidth=2, color="#145a8d")
        # Plot as line chart with markers
        ax.plot(bucket_median.index, bucket_median.values,
                marker='o', markersize=5, linewidth=2, color="#5D3FD3")

        # Add overall mean reference line
        ax.axhline(overall_mean, linestyle="--", color="#afc6ba",
                linewidth=1, label="Overall mean")
        # Add overall mean reference line
        ax.axhline(overall_median, linestyle="--", color="#5D3FD3",
                linewidth=1, label="Overall median")

        # Set axis limits with buffer
        ax.set_ylim(y_min, y_max)

        # Add grid for better readability
        ax.grid(True, linestyle=":", alpha=0.3)

        # Labels and title
        ax.set_xlabel("Prediction-rank bucket (1 = top scores)")
        ax.set_ylabel(f"Mean {target_var} during validation")
        ax.set_title("Return vs Rank – Validation")
        ax.legend()


    def _plot_combined_score_return(self, ax, n_buckets: int = 25):
        """
        Combined visualization showing both:
        1. Score distribution (KDE plot on left Y-axis)
        2. Average return by score bucket (line plot on right Y-axis)

        X-axis shows the actual prediction scores rather than percentiles.
        """
        # Check if validation data is available
        if self.y_validation_pred is None or self.validation_wallet_features_df is None:
            # Fall back to just showing score distribution without returns
            self._plot_score_distribution(ax)
            ax.text(0.5, 0.1, "Return data not available (validation set missing)",
                    ha="center", va="center", alpha=0.7, transform=ax.transAxes)
            return

        # Create a twin axis for the return line
        ax_ret = ax.twinx()

        # 1. Plot score distribution on primary Y-axis (left)
        sns.kdeplot(data=self.y_test, ax=ax, label='Actual', color='#69c4ff')
        sns.kdeplot(data=self.y_pred, ax=ax, label='Predicted', color='#ff6969')

        ax.axvline(np.mean(self.y_test), color='#69c4ff', linestyle='--', alpha=0.5)
        ax.axvline(np.mean(self.y_pred), color='#ff6969', linestyle='--', alpha=0.5)

        ax.set_title('Score Distribution & Returns by Score')
        ax.set_xlabel('Prediction Score')
        ax.set_ylabel('Density', color='#afc6ba')
        ax.tick_params(axis='y', colors='#afc6ba')
        ax.legend(loc='upper left')

        # 2. Prepare return data by score buckets
        target_var = self.modeling_config["target_variable"]

        # Extract validation data for returns
        if hasattr(self.y_validation_pred, 'index'):  # pandas Series/DataFrame
            returns = self.validation_wallet_features_df[target_var].reindex(
                self.y_validation_pred.index
            )
            df = pd.DataFrame({"pred": self.y_validation_pred, "ret": returns}).dropna()
        else:  # NumPy array
            returns = self.validation_wallet_features_df[target_var].values
            df = pd.DataFrame({
                "pred": self.y_validation_pred,
                "ret": returns,
                "ret_wins": u.winsorize(returns, 0.005)
            }).dropna()

        # Create score buckets based on actual score values, not percentiles
        score_min = df["pred"].min()
        score_max = df["pred"].max()
        bucket_edges = np.linspace(score_min, score_max, n_buckets + 1)

        # Calculate mean return for each score bucket
        buckets = []
        for i in range(len(bucket_edges) - 1):
            low = bucket_edges[i]
            high = bucket_edges[i + 1]
            mask = (df["pred"] >= low) & (df["pred"] <= high)

            if mask.sum() > 0:  # Only include if there are samples
                mean_return = df.loc[mask, "ret"].mean()
                median_return = df.loc[mask, "ret"].median()
                wins_return = df.loc[mask, "ret_wins"].mean()
                buckets.append({
                    "score_mid": (low + high) / 2,
                    "mean_return": mean_return,
                    "median_return": median_return,
                    "wins_return": wins_return,
                    "count": mask.sum()
                })

        bucket_df = pd.DataFrame(buckets)

        if not bucket_df.empty:
            # 3. Plot return line on secondary Y-axis (right)
            # Use a single fixed marker size instead of variable sizes
            # ax_ret.plot(bucket_df["score_mid"], bucket_df["wins_return"],
            #         marker='o', markersize=6, linewidth=2,
            #         color='#ffe000', label='Winsorized Return')
            ax_ret.plot(bucket_df["score_mid"], bucket_df["median_return"],
                    marker='o', markersize=6, linewidth=2,
                    color='#8000ff', label='Median Return')
            ax_ret.plot(bucket_df["score_mid"], bucket_df["mean_return"],
                    marker='o', markersize=6, linewidth=2,
                    color='#22DD22', label='Mean Return')

            # Add horizontal line for overall mean return
            overall_mean = df["ret"].mean()
            ax_ret.axhline(overall_mean, linestyle=":", color='#22DD22',
                        linewidth=1, label='Overall Mean Return')

            # Set Y-axis label and color for return axis
            ax_ret.set_ylabel(f'Mean {target_var}', color='#22DD22')
            ax_ret.tick_params(axis='y', colors='#22DD22')
            ax_ret.legend(loc='upper right')

            # Add some buffer to the return y-axis
            y_min, y_max = ax_ret.get_ylim()
            y_range = y_max - y_min
            buffer = y_range * 0.2
            ax_ret.set_ylim(y_min - buffer, y_max + buffer)

        # Add grid for better readability
        ax.grid(True, linestyle=":", alpha=0.3)


    def plot_coin_evaluation(self, plot_type='all', display=True):
        """Generate evaluation plots for coin models."""
        if plot_type == 'all':
            fig = plt.figure(figsize=(15, 12))
            gs = plt.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

            ax1 = fig.add_subplot(gs[0, 0])  # Actual vs Predicted
            ax2 = fig.add_subplot(gs[0, 1])  # Residuals
            ax3 = fig.add_subplot(gs[1, 0])  # Score Distribution
            ax4 = fig.add_subplot(gs[1, 1])  # Feature Importance

            self._plot_return_vs_rank(ax1)
            self._plot_actual_vs_predicted(ax2)
            self._plot_score_distribution(ax3)
            self._plot_feature_importance(ax4)
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            if plot_type == 'actual_vs_predicted':
                self._plot_actual_vs_predicted(ax)
            elif plot_type == 'residuals':
                self._plot_residuals(ax)
            elif plot_type == 'score_distribution':
                self._plot_score_distribution(ax)
            elif plot_type == 'feature_importance':
                self._plot_feature_importance(ax)

        plt.tight_layout()
        if display:
            plt.show()
            return None
        return fig


    def plot_wallet_evaluation(self, plot_type='all', display=True, levels=0):
        """
        Generate evaluation plots for wallet models with cohort analysis.

        Params:
        - plot_type: 'all' or one of ['actual_vs_predicted', 'residuals', 'combined_score_return',
        'feature_importance', 'prefix_importance', 'cohort_comparison']
        - display: If True, show plots directly; if False, return the figure.
        - levels: Prefix grouping depth (used only if plot_type=='prefix_importance')
        """
        if not hasattr(self, 'training_cohort_pred'):
            raise ValueError("Wallet evaluation requires training cohort data")

        if plot_type == 'all':
            # 2x2 layout for multiple evaluation plots
            fig = plt.figure(figsize=(15, 12))
            gs = plt.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

            ax1 = fig.add_subplot(gs[0, 0])  # Actual vs Predicted
            ax2 = fig.add_subplot(gs[0, 1])  # Residuals
            ax3 = fig.add_subplot(gs[1, 0])  # Combined Score & Return
            ax4 = fig.add_subplot(gs[1, 1])  # Prefix-based Importance

            self._plot_actual_vs_predicted(ax1)
            self._plot_residuals(ax2)
            self._plot_combined_score_return(ax3, n_buckets=50)  # New combined method
            self._plot_feature_importance(ax4, levels=levels)
        else:
            # Single plot mode
            fig, ax = plt.subplots(figsize=(8, 6))
            if plot_type == 'actual_vs_predicted':
                self._plot_actual_vs_predicted(ax)
            elif plot_type == 'residuals':
                self._plot_residuals(ax)
            elif plot_type == 'combined_score_return':  # New option
                self._plot_combined_score_return(ax, n_buckets=25)
            elif plot_type == 'cohort_comparison':  # Keep old option for backward compatibility
                self._plot_cohort_comparison(ax)
            elif plot_type == 'feature_importance':
                self._plot_feature_importance(ax)
            elif plot_type == 'prefix_importance':
                self._plot_feature_importance(ax, levels=levels)
            elif plot_type == 'return_vs_rank':
                self._plot_return_vs_rank(ax, n_buckets=25)
        plt.tight_layout()
        if display:
            plt.show()
            return None
        return fig


    def identify_predictive_populations(
        self,
        segmentation_features: List[str],
        min_pop_pct: float = 0.05,
        max_segments: int = 10,
        n_bins: int = 5
    ) -> pd.DataFrame:
        """
        Params:
        - segmentation_features (List[str]): numeric features to segment on
        - min_pop_pct (float): min segment size as fraction of total
        - max_segments (int): max number of segments to return
        - n_bins (int): number of quantile bins to cut into

        Returns:
        - DataFrame of top segments by error lift
        """
        # require validation data
        if self.X_validation is None or self.y_validation_pred is None or self.y_validation is None:
            raise ValueError("Validation data not set on this evaluator")

        # build DataFrame from validation features + preds
        df = self.X_validation.copy()
        df['pred'], df['actual'] = self.y_validation_pred, self.y_validation


        df['err'] = (df['actual'] - df['pred']).abs()
        df['sq_err'] = df['err'] ** 2
        overall_mean_err = df['err'].mean()
        overall_median_err = df['err'].median()
        overall_rmse = np.sqrt(df['sq_err'].mean())
        overall_r2 = r2_score(df['actual'], df['pred'])

        df['high_perf'] = df['err'] < overall_median_err
        logger.warning('1')
        contrast_sets = []
        for feat in segmentation_features:
            if feat not in df:
                continue

            bin_col = f"{feat}_bin"
            # try cutting into n_bins, even if fewer unique values
            try:
                df[bin_col] = pd.qcut(df[feat], n_bins, labels=False, duplicates='drop')
            except ValueError:
                # e.g. all values identical
                df[bin_col] = 0  # single bin
            logger.warning('2')
            for b in df[bin_col].dropna().unique():
                mask = df[bin_col] == b
                support, size = mask.mean(), mask.sum()
                if support < min_pop_pct or size < 30:
                    continue

                seg = df[mask]
                mean_err = seg['err'].mean()
                seg_r2 = r2_score(seg['actual'], seg['pred'])

                lift = (overall_r2 - seg_r2) / overall_r2

                ct = pd.crosstab(mask, df['high_perf'])
                if ct.shape == (2,2) and ct.values.min() >= 5:
                    p = chi2_contingency(ct)[1]
                    # if p < 0.05 and abs(lift) > 0.1:
                    vals = seg[feat]
                    contrast_sets.append({
                        'Feature': feat.replace('|all_windows',''),
                        'Quantile': int(b),
                        'Range': f"{dc.human_format(vals.min())}-{dc.human_format(vals.max())}",
                        'Wallets': size,
                        'Pop. Pct': f"{support:.3f}",
                        'R2': f"{seg_r2:.3f}",
                        'R2 Overall': f"{overall_r2:.3f}",
                        'R2 vs Overall': f"{(seg_r2 - overall_r2):.3f}",

                        'Mean Error': f"{mean_err:.3f}",
                        'Mean Err Overall': f"{overall_mean_err:.3f}",
                        'ME vs Overall': f"{mean_err-overall_mean_err:.3f}",
                        'RMSE': f"{np.sqrt(seg['sq_err'].mean()):.3f}",
                        'RMSE Overall': f"{overall_rmse:.3f}",
                        'RMSE vs Overall': f"{np.sqrt(seg['sq_err'].mean()) - overall_rmse:.3f}",
                        'P-Value': f"{p:.2f}",
                        'abs_error_lift': abs(lift),
                    })

        logger.warning('3')
        if not contrast_sets:
            return pd.DataFrame()

        out = (
            pd.DataFrame(contrast_sets)
            .sort_values('abs_error_lift', ascending=False)
            .head(max_segments)
            .drop('abs_error_lift', axis=1)
        )
        # ensure numeric dtypes
        for col in ['Pop. Pct','Mean Error','ME vs Overall','RMSE','RMSE vs Overall']:
            out[col] = pd.to_numeric(out[col])
        return out







# -----------------------------------
#          Classifier Class
# -----------------------------------

class ClassifierEvaluator(RegressorEvaluator):
    """
    Same interface as RegressorEvaluator but for classification models.
    """
    def __init__(self, wallet_model_results: dict):

        # Extract probability predictions
        self.y_pred_proba = wallet_model_results['y_pred_proba']
        self.y_validation_pred_proba = wallet_model_results.get('y_validation_pred_proba')
        self.y_validation_pred = wallet_model_results.get('y_validation_pred')
        self.y_pred_threshold = wallet_model_results['modeling_config']['y_pred_threshold']


        # super() creates metrics and
        super().__init__(wallet_model_results)


    def _calculate_metrics(self):
        """
        Calculate core classification metrics for test and validation sets.
        """
        # Sample size tracking
        self.metrics['test_samples'] = len(self.y_pred)
        if self.y_train is not None:
            self.metrics['train_samples'] = len(self.y_train)

        # Test set metrics
        self.metrics['accuracy'] = accuracy_score(self.y_test, self.y_pred)
        self.metrics['precision'] = precision_score(self.y_test, self.y_pred, zero_division=0)
        self.metrics['recall'] = recall_score(self.y_test, self.y_pred, zero_division=0)
        self.metrics['f1'] = f1_score(self.y_test, self.y_pred, zero_division=0)
        try:
            self.metrics['roc_auc'] = roc_auc_score(self.y_test, self.y_pred_proba)
            self.metrics['log_loss'] = log_loss(self.y_test, self.y_pred_proba)
        except ValueError:
            logger.warning("Only one class found in classifier predictions.")
            self.metrics['roc_auc'] = np.nan
            self.metrics['log_loss'] = np.nan

        # Validation set metrics if available
        if getattr(self, 'y_validation', None) is not None and hasattr(self, 'y_validation_pred_proba'):
            self.metrics['positive_predictions'] = self.y_validation_pred.sum()
            self.metrics['positive_pct'] = (self.y_validation_pred.sum()/len(self.y_validation_pred))*100
            self.metrics['val_accuracy'] = accuracy_score(self.y_validation, self.y_validation_pred)
            self.metrics['val_precision'] = precision_score(self.y_validation, self.y_validation_pred, zero_division=0)
            self.metrics['val_recall'] = recall_score(self.y_validation, self.y_validation_pred, zero_division=0)
            self.metrics['val_f1'] = f1_score(self.y_validation, self.y_validation_pred, zero_division=0)
            try:
                self.metrics['val_roc_auc'] = roc_auc_score(self.y_validation, self.y_validation_pred_proba)
            except ValueError:
                logger.warning("ROC AUC score failed to generate. Only like class is likely predicted.")
                self.metrics['val_roc_auc'] = np.nan

        # Validation return-based metrics
        if (getattr(self, 'y_validation_pred_proba', None) is not None
            and hasattr(self, 'validation_wallet_features_df')):
            target = self.modeling_config['target_variable']
            returns = self.validation_wallet_features_df[target].reindex(self.y_validation_pred_proba.index)
            df_val = pd.DataFrame({
                'pred' :self.y_validation_pred,
                'proba': self.y_validation_pred_proba,
                'ret': returns,
                'ret_wins': u.winsorize(returns,0.005)
            }).dropna()
            pct1 = np.percentile(df_val['proba'], 99)
            pct5 = np.percentile(df_val['proba'], 95)
            self.metrics['positive_pred_return'] = df_val.loc[df_val['pred'] == 1, 'ret'].mean()
            self.metrics['positive_pred_wins_return'] = df_val.loc[df_val['pred'] == 1, 'ret_wins'].mean()
            self.metrics['val_wins_return_top1'] = df_val.loc[df_val['proba'] >= pct1, 'ret_wins'].mean()
            self.metrics['val_wins_return_top5'] = df_val.loc[df_val['proba'] >= pct5, 'ret_wins'].mean()
            self.metrics['val_wins_return_overall'] = df_val['ret_wins'].mean()

        # Feature importance if available
        if self.model is not None and hasattr(self.model, 'feature_importances_'):
            self._calculate_feature_importance()


    def summary_report(self):
        """
        Generate formatted summary of classification model performance.
        """
        # Header and sample info
        summary = self._get_summary_header()

        # Classification metrics
        summary.extend([
                "Classification Metrics",
                "-" * 35,
                f"ROC AUC:                    {self.metrics['roc_auc']:.3f}",
                f"Log Loss:                   {self.metrics['log_loss']:.3f}",
                f"Accuracy:                   {self.metrics['accuracy']:.3f}",
                f"Precision:                  {self.metrics['precision']:.3f}",
                f"Recall:                     {self.metrics['recall']:.3f}",
                f"F1 Score:                   {self.metrics['f1']:.3f}",
                ""
        ])

        # Validation return metrics
        if 'val_roc_auc' in self.metrics:
            summary.extend([
                "Validation Metrics",
                "-" * 35,
                f"Val ROC AUC:                {self.metrics['val_roc_auc']:.3f}",
                f"Val Accuracy:               {self.metrics['val_accuracy']:.3f}",
                f"Val Precision:              {self.metrics['val_precision']:.3f}",
                f"Val Recall:                 {self.metrics['val_recall']:.3f}",
                f"Val F1 Score:               {self.metrics['val_f1']:.3f}",
                "",
                "Validation Return Metrics",
                "-" * 35,
                f"Positive Threshold:         {self.y_pred_threshold:.2f}",
                f"Positive Predictions:       {self.metrics['positive_predictions']:.0f}"
                    f"/{len(self.y_validation_pred)} ({self.metrics['positive_pct']:.2f}%)",
                f"Positive Mean Outcome:      {self.metrics['positive_pred_return']:.3f}",
                f"Positive W-Mean Outcome:    {self.metrics['positive_pred_wins_return']:.3f}",
                f"Top 1% W-Mean Outcome:      {self.metrics['val_wins_return_top1']:.3f}",
                f"Top 5% W-Mean Outcome:      {self.metrics['val_wins_return_top5']:.3f}",
                f"Overall W-Mean Outcome:     {self.metrics['val_wins_return_overall']:.3f}",
                ""
            ])

        report = "\n".join(summary)
        logger.info("\n%s", report)


    def plot_wallet_evaluation(
        self,
        plot_type: str = "all",
        display: bool = True,
        levels: int = 0,
    ):
        """
        Skeleton wallet-level evaluation plot for classification models.

        Current layout (2 × 2):
        • Chart 1  – placeholder for ROC / PR curves
        • Chart 2  – placeholder for calibration or return-vs-rank
        • Chart 3  – placeholder for cohort comparison
        • Chart 4  – feature-importance bar (re-uses parent helper)

        Parameters
        ----------
        plot_type : str, default "all"
            Only 'all' is supported in this skeleton.
        display : bool, default True
            If True, shows the figure; otherwise returns the Matplotlib figure.
        levels : int, default 0
            Grouping depth passed to `_plot_feature_importance`.
        """
        if plot_type != "all":
            raise NotImplementedError("This skeleton only supports plot_type='all'.")

        # --- build 2×2 canvas
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        self._plot_roc_curves(ax1)
        self._plot_auc_pr_curves(ax2)
        self._plot_return_vs_rank_classifier(ax3, n_buckets=20)
        self._plot_feature_importance(ax4, levels=levels)


        plt.tight_layout()
        if display:
            plt.show()
            return None
        return fig


    # ------------------------------------------------------------------
    #                Chart-building helpers
    # ------------------------------------------------------------------
    def _plot_roc_curves(self, ax):
        """
        ROC curves for test & validation.
        Validation line: thick green (#22DD22)
        Random-guess diagonal: grey dashed.
        """
        # --- Test ROC (thin, default blue)
        RocCurveDisplay.from_predictions(
            self.y_test,
            self.y_pred_proba,
            ax=ax,
            name="Test",
            linewidth=1.5,
        )

        # --- Validation ROC (thick green) ------------------------------------
        if getattr(self, "y_validation_pred_proba", None) is not None \
        and getattr(self, "y_validation", None) is not None:
            RocCurveDisplay.from_predictions(
                self.y_validation,
                self.y_validation_pred_proba,
                ax=ax,
                name="Validation",
                linewidth=2.5,
                color="#22DD22",
            )

        # --- 45° reference line ---------------------------------------------
        ax.plot(
            [0, 1], [0, 1],
            linestyle="--",
            linewidth=1,
            color="#afc6ba",
            label="Random"
        )

        ax.set_title("ROC Curve – Test vs Validation")
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.legend()


    def _plot_auc_pr_curves(self, ax):
        """
        Precision-Recall curves with AUC-PR values for test & validation sets.

        Validation line: thick green (#22DD22)
        Baseline: horizontal dashed line showing the positive class prevalence.
        """
        # Calculate baseline (positive class prevalence)
        baseline = self.y_test.mean()

        # --- Test PR curve with AUC-PR
        precision, recall, _ = precision_recall_curve(
            self.y_test,
            self.y_pred_proba,
            pos_label=1
        )
        test_auc_pr = auc(recall, precision)  # Calculate AUC-PR

        # Plot test curve
        ax.plot(
            recall[1:], precision[1:],  # Skip first point to avoid misleading spike
            linewidth=1.5,
            color="#1f77b4",  # Default blue
            label=f"Test (AUC-PR: {test_auc_pr:.3f})"
        )

        # --- Validation PR curve with AUC-PR (if available)
        if getattr(self, "y_validation_pred_proba", None) is not None \
        and getattr(self, "y_validation", None) is not None:
            # Calculate validation baseline
            val_baseline = self.y_validation.mean()

            # Compute precision-recall pairs and AUC-PR for validation
            val_precision, val_recall, _ = precision_recall_curve(
                self.y_validation,
                self.y_validation_pred_proba,
                pos_label=1
            )
            val_auc_pr = auc(val_recall, val_precision)

            # Plot validation curve
            ax.plot(
                val_recall[1:], val_precision[1:],  # Skip first point
                linewidth=2.5,
                color="#22DD22",  # Green
                label=f"Validation (AUC-PR: {val_auc_pr:.3f})"
            )

            # Update baseline to show validation baseline if available
            baseline = val_baseline

        # --- Baseline reference line (positive class prevalence)
        ax.axhline(
            baseline,
            linestyle="--",
            linewidth=1,
            color="#afc6ba",  # Light grey-green
            label=f"Baseline ({baseline:.3f})"
        )

        # Formatting
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve with AUC-PR')
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.legend(loc="lower left")


    def _plot_return_vs_rank_classifier(self, ax, n_buckets: int = 10):
        """
        Plot histogram of prediction probabilities and returns by probability bins.
        X-axis is actual prediction score.
        Primary Y-axis: count histogram of wallets per score bin.
        Secondary Y-axis: mean return per score bin.
        """
        # Check for validation data
        if self.y_validation_pred_proba is None or self.validation_wallet_features_df is None:
            ax.text(0.5, 0.5, "Validation data not available",
                    ha="center", va="center")
            return

        target_var = self.modeling_config["target_variable"]
        returns = self.validation_wallet_features_df[target_var].reindex(
            self.y_validation_pred_proba.index
        )
        returns_winsorized = u.winsorize(returns, 0.01)

        df = pd.DataFrame({
            "proba": self.y_validation_pred_proba,
            "ret": returns,
            "ret_win": returns_winsorized
        }).dropna()

        # Define score bins
        try:
            score_min, score_max = df["proba"].min(), df["proba"].max()
            bin_edges = np.linspace(score_min, score_max, n_buckets + 1)
            df["score_bin"] = pd.cut(df["proba"], bins=bin_edges, include_lowest=True)
        except ValueError:
            ax.text(0.5, 0.5, 'Insufficient score spread to generate bins.',
                    ha='center', va='center')
            return


        # Compute counts and mean returns per bin
        bin_counts = df.groupby("score_bin", observed=True).size()
        bin_mean_ret = df.groupby("score_bin", observed=True)["ret"].mean()
        bin_median_ret = df.groupby("score_bin", observed=True)["ret"].median()
        bin_winsorized_ret = df.groupby("score_bin", observed=True)["ret_win"].mean()

        # Drop bins with zero count
        valid_bins = bin_counts[bin_counts > 0]
        valid_centers = [
            interval.left + (interval.right - interval.left) / 2
            for interval in valid_bins.index
        ]
        valid_counts = valid_bins.values
        valid_mean_ret = bin_mean_ret.reindex(valid_bins.index).values
        valid_median_ret = bin_median_ret.reindex(valid_bins.index).values
        valid_winsorized_ret = bin_winsorized_ret.reindex(valid_bins.index).values
        width = bin_edges[1] - bin_edges[0]

        # Primary axis: histogram of counts
        ax.bar(valid_centers, valid_counts, width=width, alpha=0.6, label="Count")

        # Use log scale for wallet counts
        ax.set_yscale('log')
        ax.figure.canvas.draw()

        # Secondary axis: mean return line
        ax2 = ax.twinx()
        # Compute threshold on absolute returns to ensure a positive linthresh
        abs_returns = np.abs(df["ret"])
        linthresh = np.percentile(abs_returns, 95)
        # Fallback to a small positive value if the threshold isn't positive
        if linthresh <= 0:
            max_abs = abs_returns.max()
            linthresh = max_abs * 0.05 if max_abs > 0 else 1.0
        ax2.set_yscale("symlog", linthresh=linthresh)  # comment to toggle log/linear y2 axis
        ax2.plot(
            valid_centers,
            valid_median_ret,
            marker='o',
            linestyle='-',
            linewidth=2,
            label="Median Return",
            color="#8000ff"
        )
        ax2.plot(
            valid_centers,
            valid_winsorized_ret,
            marker='o',
            linestyle='-',
            linewidth=2,
            label="Winsorized Return",
            color="#ffe000"
        )
        # Annotate lowest and highest winsorized return
        low_interval = bin_winsorized_ret.idxmin()
        high_interval = bin_winsorized_ret.idxmax()
        x_low = (low_interval.left + low_interval.right) / 2
        x_high = (high_interval.left + high_interval.right) / 2
        y_low = bin_winsorized_ret.loc[low_interval]
        y_high = bin_winsorized_ret.loc[high_interval]
        ax2.annotate(f"{y_low:.2f}", xy=(x_low, y_low),
                     xytext=(0, -10), textcoords="offset points",
                     ha="center", va="top")
        ax2.annotate(f"{y_high:.2f}", xy=(x_high, y_high),
                     xytext=(0, 10), textcoords="offset points",
                     ha="center", va="bottom")
        ax2.plot(
            valid_centers,
            valid_mean_ret,
            marker='o',
            linestyle='-',
            linewidth=2,
            label="Mean Return",
            color="#22DD22"
        )

        # Overall mean return line
        overall_mean = df["ret"].mean()
        ax2.axhline(
            overall_mean,
            linestyle="--",
            color="#afc6ba",
            linewidth=1,
            label="Overall mean return"
        )

        # Labels and title
        ax.set_xlabel("Prediction Score")
        ax.set_ylabel("Number of Wallets")
        ax2.set_ylabel(f"Mean {target_var} during validation")
        ax.set_title("Prediction Score Distribution and Returns")
        ax.grid(True, linestyle=":", alpha=0.3)

        # Combine legends from both axes
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper left")
