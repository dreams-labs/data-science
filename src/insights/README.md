# data-science/insights

This directory contains insights about the now-deprecated Coin Flow model. The current models are the wallet_modeling/wallet_model.py and coin_modeling/coin_model.py. This README was prepared with the assistance of AI.

## Overview

The insights directory provides comprehensive model evaluation capabilities, combining statistical metrics with domain-specific profitability analysis. It supports both classification and regression models with automated experiment tracking and visualization.

## Key Modules

**analysis.py**
- `generate_profitability_curves()` - Creates side-by-side plots comparing model performance against optimal performance using cumulative and average returns
- `generate_profitability_curves_by_time_window()` - Generates time-window-specific profitability analysis with separate charts for each modeling period

**modeling.py**
- `train_model()` - Trains RandomForest and GradientBoosting models with k-fold cross-validation, saves model artifacts with UUID tracking
- `evaluate_model()` - Calculates comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC, MSE, R², profitability metrics) and saves predictions
- `calculate_profitability_auc()` - Computes area under the profitability curve, measuring how well model rankings capture actual returns
- `calculate_downside_profitability_auc()` - Evaluates model's ability to avoid poor investments by inverting predictions and returns
- `log_trial_results()` - Aggregates training logs, performance metrics, feature importance, and predictions into comprehensive experiment records

## Integration

Integrates with the broader pipeline through:
- **feature_engineering**: Receives preprocessed training data and target variables from data splitting functions
- **training_data**: Uses price data for return calculations in profitability metrics
- **utils**: Leverages timing decorators and configuration loading utilities

## Usage Patterns

The module follows a standard ML evaluation workflow:
1. Train models using `train_model()` with automatic artifact saving
2. Evaluate performance using `evaluate_model()` with both statistical and profitability metrics
3. Generate visual analysis using profitability curve functions
4. Track experiments using `log_trial_results()` for comprehensive result storage

All functions support both classification (moon/crater prediction) and regression (return prediction) tasks, with automatic detection based on model type.