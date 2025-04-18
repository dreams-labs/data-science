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
)
from dreams_core import core as dc

# pylint:disable=invalid-name  # X_test isn't camelcase

# Set up logger at the module level
logger = logging.getLogger(__name__)


class RegressionEvaluator:
    """
    A utility class for evaluating and visualizing regression model performance.

    Methods:
        summary_report(): Returns a formatted text summary of model performance
        plot_evaluation(plot_type='all'): Creates visualization plots of model performance

    Internal Methods:
        __init__(y_test, y_pred, model=None, feature_names=None): Initialize with actual and predicted values
        _calculate_metrics(): Computes regression performance metrics like RMSE, MAE, R2
        _plot_actual_vs_predicted(ax): Plots actual vs predicted values
        _plot_residuals(ax): Plots residuals vs predicted values
        _plot_residuals_distribution(ax): Plots histogram of residuals
        _plot_feature_importance(ax): Plots feature importance if available from model
    """
    def __init__(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        model=None,
        feature_names=None,
        y_train: np.ndarray = None,
        training_cohort_pred: np.ndarray = None,
        training_cohort_actuals: np.ndarray = None,
        y_validation: np.ndarray = None,
        y_validation_pred: np.ndarray = None,
        X_test: np.ndarray = None,
    ):
        """
        Initialize evaluator with prediction data and optional training cohort data.

        Params:
        - y_test: Test set actual values
        - y_pred: Test set predicted values
        - model: Optional fitted model for feature importance
        - feature_names: Optional feature names for importance plots
        - y_train: Optional training set values
        - training_cohort_pred: Optional full training cohort predictions
        - training_cohort_actuals: Optional full training cohort actual values
        """
        # Core prediction data
        self.y_test = np.array(y_test)
        self.y_pred = np.array(y_pred)

        # Optional model data for feature importance
        self.model = model
        self.feature_names = feature_names

        # Optional training/cohort data
        self.y_train = np.array(y_train) if y_train is not None else None
        self.training_cohort_pred = (np.array(training_cohort_pred)
                                   if training_cohort_pred is not None else None)
        self.training_cohort_actuals = (np.array(training_cohort_actuals)
                                      if training_cohort_actuals is not None else None)

        # Optional validation set
        self.y_validation = np.array(y_validation) if y_validation is not None else None
        self.y_validation_pred = np.array(y_validation_pred) if y_validation_pred is not None else None

        # Optional training data
        self.X_test = X_test

        # Initialize storage
        self.metrics = {}
        self.residuals = None
        self.custom_cmap = None

        # Calculate base metrics
        self._calculate_metrics()

        # Set up plot styling
        self._setup_plot_style()


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

        self.custom_cmap = mcolors.LinearSegmentedColormap.from_list(
            'custom_blues', ['#1b2530', '#145a8d', '#ddeeff']
        )



    def summary_report(self):
        """Generate formatted summary of model performance."""
        summary = [
            "Model Performance Summary",
            "=" * 35,
        ]

        # Add sample sizes and feature count
        n_features = len(self.feature_names) if self.feature_names is not None else 0
        n_all_windows = sum(1 for feature in self.feature_names if '|all_windows' in feature)

        if hasattr(self.metrics, 'total_cohort_samples'):
            summary.extend([
                f"Training Cohort:          {self.metrics['total_cohort_samples']:,d}",
                f"Modeling Cohort Train:    {self.metrics['train_samples']:,d}",
                f"Modeling Cohort Test:     {self.metrics['test_samples']:,d}",
                ""
            ])
        else:
            summary.extend([
                f"Test Samples:             {self.metrics['test_samples']:,d}",
                f"Number of Features:       {n_features:,d}",
                f"Features per Window:      {n_all_windows:,d}",
                ""
            ])

        # Add core metrics
        summary.extend([
            "Core Metrics",
            "-" * 35,
            f"R² Score:                 {self.metrics['r2']:.3f}",
            f"RMSE:                     {self.metrics['rmse']:.3f}",
            f"MAE:                      {self.metrics['mae']:.3f}",
            ""
        ])

        # Add validation metrics if we computed them
        if 'validation_metrics' in self.metrics:
            vm = self.metrics['validation_metrics']
            summary.extend([
                "Validation Set Metrics",
                "-" * 35,
                f"R² Score:                 {vm['r2']:.3f}",
                f"RMSE:                     {vm['rmse']:.3f}",
                f"MAE:                      {vm['mae']:.3f}",
                ""
            ])

        # Add training cohort metrics if available
        if 'training_cohort' in self.metrics:
            summary.extend([
                "Inactive Wallets Cohort Metrics",
                "-" * 35,
                f"R² Score:                 {self.metrics['training_cohort']['r2']:.3f}",
                f"RMSE:                     {self.metrics['training_cohort']['rmse']:.3f}",
                f"MAE:                      {self.metrics['training_cohort']['mae']:.3f}",
                ""
            ])

        # Add residuals analysis
        summary.extend([
            "Residuals Analysis",
            "-" * 35,
            f"Mean of Residuals:        {self.metrics['residuals_mean']:.3f}",
            f"Standard Dev of Residuals:{self.metrics['residuals_std']:.3f}",
            f"95% Prediction Interval:  ±{self.metrics['prediction_interval_95']:.3f}"
        ])

        return "\n".join(summary)


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

    def plot_coin_evaluation(self, plot_type='all', display=True):
        """Generate evaluation plots for coin models."""
        if plot_type == 'all':
            fig = plt.figure(figsize=(15, 12))
            gs = plt.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

            ax1 = fig.add_subplot(gs[0, 0])  # Actual vs Predicted
            ax2 = fig.add_subplot(gs[0, 1])  # Residuals
            ax3 = fig.add_subplot(gs[1, 0])  # Score Distribution
            ax4 = fig.add_subplot(gs[1, 1])  # Feature Importance

            self._plot_actual_vs_predicted(ax1)
            self._plot_residuals(ax2)
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
        - plot_type: 'all' or one of ['actual_vs_predicted', 'residuals', 'cohort_comparison',
          'feature_importance', 'prefix_importance']
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
            ax3 = fig.add_subplot(gs[1, 0])  # Cohort Comparison
            ax4 = fig.add_subplot(gs[1, 1])  # Prefix-based Importance

            self._plot_actual_vs_predicted(ax1)
            self._plot_residuals(ax2)
            self._plot_cohort_comparison(ax3)
            # Re-use the importance_summary logic via _plot_prefix_importance
            self._plot_feature_importance(ax4, levels=levels)
        else:
            # Single plot mode
            fig, ax = plt.subplots(figsize=(8, 6))
            if plot_type == 'actual_vs_predicted':
                self._plot_actual_vs_predicted(ax)
            elif plot_type == 'residuals':
                self._plot_residuals(ax)
            elif plot_type == 'cohort_comparison':
                self._plot_cohort_comparison(ax)
            elif plot_type == 'feature_importance':
                self._plot_feature_importance(ax)
            elif plot_type == 'prefix_importance':
                self._plot_feature_importance(ax, levels=levels)
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
        df = self.X_test.copy()
        df['pred'], df['actual'] = self.y_pred, self.y_test
        df['err'] = (df['actual'] - df['pred']).abs()
        df['sq_err'] = df['err'] ** 2
        overall_mean_err = df['err'].mean()
        overall_median_err = df['err'].median()
        overall_rmse = np.sqrt(df['sq_err'].mean())

        df['high_perf'] = df['err'] < overall_median_err

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
            for b in df[bin_col].dropna().unique():
                mask = df[bin_col] == b
                support, size = mask.mean(), mask.sum()
                if support < min_pop_pct or size < 30:
                    continue

                seg = df[mask]
                mean_err = seg['err'].mean()
                lift = (overall_mean_err - mean_err) / overall_mean_err

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
                        'Mean Error': f"{mean_err:.3f}",
                        'ME vs Overall': f"{mean_err-overall_mean_err:.3f}",
                        'RMSE': f"{np.sqrt(seg['sq_err'].mean()):.3f}",
                        'RMSE vs Overall': f"{np.sqrt(seg['sq_err'].mean()) - overall_rmse:.3f}",
                        'P-Value': f"{p:.2f}",
                        'abs_error_lift': abs(lift),
                    })

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
