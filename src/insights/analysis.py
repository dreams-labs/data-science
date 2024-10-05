"""
functions used to analyze model and experiment performance
"""
# pylint: disable=C0103 # X_train violates camelcase
# pylint: disable=E0401 # can't find utils import
# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# project files
import modeling as m


def generate_profitability_curves(predictions, returns, winsorization_cutoff=0):
    """
    Generates charts showing the model's performance vs the optimal profitability curves
    based on cumulative averages or sums.
    """
    if len(predictions) != len(returns):
        raise ValueError("Predictions and returns must have the same length")

    # Winsorize the returns (apply caps to the top n % of values)
    returns_winsorized = m.winsorize(returns, winsorization_cutoff)

    # Merge datasets
    df = pd.DataFrame({
        'predictions': predictions,
        'returns': returns_winsorized,
    })

    # Sort by actual returns to obtain optimal performance
    df_sorted = df.sort_values('returns', ascending=False)
    cumulative_best_returns = np.cumsum(df_sorted['returns'])
    cumulative_best_avg_returns = df_sorted['returns'].expanding().mean()

    # Sort by model score to obtain modeled performance
    df_sorted = df.sort_values('predictions', ascending=False)
    cumulative_model_returns = np.cumsum(df_sorted['returns'])
    cumulative_model_avg_returns = df_sorted['returns'].expanding().mean()

    # Calculate average return across all data
    average_return = np.mean(returns_winsorized)

    # Create subplots for side-by-side plots
    _, axes = plt.subplots(1, 2, figsize=(14, 6))

    # First plot: Cumulative Returns Performance
    axes[0].plot(cumulative_best_returns.values, label='Optimal Performance')
    axes[0].plot(cumulative_model_returns.values, label='Model Performance')
    axes[0].set_title('Cumulative Returns Performance')
    axes[0].set_ylabel('Cumulative Returns')
    axes[0].set_xlabel('Rank Number')
    axes[0].legend()

    # Second plot: Average Returns Performance
    axes[1].plot(cumulative_best_avg_returns.values, label='Optimal Avg Performance')
    axes[1].plot(cumulative_model_avg_returns.values, label='Model Avg Performance')
    axes[1].set_title('Average Returns Performance')
    axes[1].set_ylabel('Average Returns')
    axes[1].set_xlabel('Rank Number')

    # Plot the horizontal dotted line for the average return
    axes[1].axhline(y=average_return, color='gray', linestyle='--', label='Average Return')

    # Add annotation for the average return
    axes[1].annotate('Average Return', xy=(0, average_return),
                    xytext=(-10, average_return + 0.01),
                    ha='left', fontsize=10, color='gray')

    axes[1].legend()

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()