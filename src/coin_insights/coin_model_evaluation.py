import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import colors as mcolors


class CoinRegressionEvaluator:
    """Utility class for basic regression model evaluation."""

    def __init__(self, y_test, y_pred, model=None, feature_names=None):
        """
        Params:
        - y_test (array-like): Test set actual values
        - y_pred (array-like): Test set predicted values
        - model (sklearn estimator): Optional fitted model for feature importance
        - feature_names (list): Optional feature names for importance plots
        """
        self.y_test = np.array(y_test)
        self.y_pred = np.array(y_pred)
        self.model = model
        self.feature_names = feature_names
        self.metrics = None
        self.residuals = None
        self.custom_cmap = None
        self._calculate_metrics()

    def _calculate_metrics(self):
        """Calculate core regression performance metrics."""
        self.metrics = {}
        self.residuals = self.y_test - self.y_pred

        # Core metrics
        self.metrics['mse'] = mean_squared_error(self.y_test, self.y_pred)
        self.metrics['rmse'] = np.sqrt(self.metrics['mse'])
        self.metrics['mae'] = mean_absolute_error(self.y_test, self.y_pred)
        self.metrics['r2'] = r2_score(self.y_test, self.y_pred)

        # Residuals stats
        self.metrics['residuals_mean'] = np.mean(self.residuals)
        self.metrics['residuals_std'] = np.std(self.residuals)
        self.metrics['prediction_interval_95'] = 1.96 * self.metrics['residuals_std']

        # Feature importances if available
        if self.model is not None and hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            if self.feature_names is None:
                self.feature_names = [f'Feature {i}' for i in range(len(importances))]

            feature_importance_pairs = sorted(
                zip(self.feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )
            self.metrics['importances'] = {
                'feature': [f[0] for f in feature_importance_pairs],
                'importance': [f[1] for f in feature_importance_pairs]
            }

    def summary_report(self):
        """Generate formatted summary of model performance."""
        summary = [
            "Model Performance Summary",
            "=" * 50,
            f"R² Score:                 {self.metrics['r2']:.3f}",
            f"RMSE:                     {self.metrics['rmse']:.3f}",
            f"MAE:                      {self.metrics['mae']:.3f}",
            "",
            "Residuals Analysis",
            "-" * 35,
            f"Mean of Residuals:        {self.metrics['residuals_mean']:.3f}",
            f"Standard Dev of Residuals:{self.metrics['residuals_std']:.3f}",
            f"95% Prediction Interval:  ±{self.metrics['prediction_interval_95']:.3f}"
        ]
        return "\n".join(summary)

    def plot_evaluation(self):
        """Generate evaluation plots."""
        # Match original styling
        plt.rcParams['figure.facecolor'] = '#181818'
        plt.rcParams['axes.facecolor'] = '#181818'
        plt.rcParams['text.color'] = '#afc6ba'
        plt.rcParams['axes.labelcolor'] = '#afc6ba'
        plt.rcParams['xtick.color'] = '#afc6ba'
        plt.rcParams['ytick.color'] = '#afc6ba'
        plt.rcParams['axes.titlecolor'] = '#afc6ba'

        # Use original colormap
        self.custom_cmap = mcolors.LinearSegmentedColormap.from_list(
            'custom_blues', ['#1b2530', '#145a8d', '#69c4ff']
        )

        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        self._plot_actual_vs_predicted(ax1)
        self._plot_residuals(ax2)
        self._plot_score_distribution(ax3)
        self._plot_feature_importance(ax4)

        plt.tight_layout()


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
                mincnt=0,
                bins='log',
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
                bins='log')

        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Predicted Values')

    def _plot_score_distribution(self, ax):
        """Plot density distribution of actual and predicted values."""
        sns.kdeplot(data=self.y_test, ax=ax, label='Actual', color='#69c4ff')
        sns.kdeplot(data=self.y_pred, ax=ax, label='Predicted', color='#ff6969')

        ax.axvline(np.mean(self.y_test), color='#69c4ff', linestyle='--', alpha=0.5)
        ax.axvline(np.mean(self.y_pred), color='#ff6969', linestyle='--', alpha=0.5)

        ax.set_title('Score Distribution')
        ax.set_xlabel('Values')
        ax.set_ylabel('Density')
        ax.legend()

    def _plot_feature_importance(self, ax):
        """Plot feature importance with inset legend."""
        if 'importances' in self.metrics:
            # Create DataFrame with feature prefixes
            df = pd.DataFrame(self.metrics['importances']).head(20)
            df['prefix'] = df['feature'].str.split('|').str[0]

            # Create color palette for prefixes
            unique_prefixes = df['prefix'].unique()
            palette = dict(zip(unique_prefixes,
                            sns.color_palette("husl", len(unique_prefixes))))

            sns.barplot(
                data=df,
                x='importance',
                y='feature',
                ax=ax,
                hue='prefix',
                palette=palette
            )

            # ax.legend(title='Feature Type', loc='lower right',
            #         bbox_to_anchor=(0.98, 0.02))

            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
            ax.set_title('Top 20 Feature Importances')
        else:
            ax.text(0.5, 0.5, 'Feature Importance Not Available',
                    ha='center', va='center')
