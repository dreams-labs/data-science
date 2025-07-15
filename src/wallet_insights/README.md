# data-science/wallet_insights

Provides comprehensive model evaluation, validation analysis, and performance reporting tools for wallet prediction models. This directory focuses on post-training analysis, validation workflows, and artifact management for the wallet-level XGBoost models.

## Key Modules

### model_evaluation.py
**RegressorEvaluator** and **ClassifierEvaluator** classes for comprehensive model performance analysis
- Calculate core regression/classification metrics (R², RMSE, ROC AUC, precision/recall)
- Generate visualization plots including actual vs predicted, residuals, feature importance
- Support validation set analysis with return-based metrics for real-world performance assessment
- Provide cluster-based performance segmentation and predictive population identification

### wallet_cluster_analysis.py
Functions for analyzing wallet behavior through clustering and segmentation
- **analyze_cluster_metrics()** - Calculate aggregate metrics for each cluster grouping
- **analyze_cluster_performance()** - Assess model performance within different clusters
- **create_cluster_report()** - Generate formatted reports comparing cluster characteristics
- **style_rows()** - Apply conditional formatting with human-readable numbers

### wallet_validation_analysis.py
Validation and prediction workflows for model performance assessment
- **load_and_predict()** - Load saved models and generate predictions on new data
- **compute_validation_coin_returns()** - Calculate coin-level returns for validation analysis
- **evaluate_predictions()** - Calculate core performance metrics for overlapping predictions
- **analyze_cohort_performance()** - Compare performance across different wallet cohorts

### wallet_model_reporting.py
Model artifact management and reporting infrastructure
- **generate_and_save_wallet_model_artifacts()** - Comprehensive model artifact saving with evaluation
- **generate_and_upload_wallet_cw_scores()** - Generate and upload wallet scores to BigQuery
- **save_model_artifacts()** - Save model pipelines, reports, and metadata with consistent UUIDs
- **load_model_report()** - Load and retrieve saved model artifacts and configurations

## Integration

Integrates with base_modeling for core model evaluation patterns and wallet_modeling for model artifacts and training data. Uses pandas_gbq for BigQuery uploads and cloudpickle for model serialization. Provides visualization capabilities through matplotlib and seaborn.

## Usage

Typically used after model training to evaluate performance, generate validation reports, and save model artifacts. The evaluation classes provide both programmatic metrics and visualization tools for model assessment and comparison.