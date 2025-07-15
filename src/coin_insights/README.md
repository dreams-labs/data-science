# data-science/coin_insights

## Overview
Provides analysis and reporting tools for coin model evaluation. Handles model performance assessment, feature importance analysis, market cap segmentation, and artifact management for coin prediction models.

## Key Modules

### coin_validation_analysis.py
- **analyze_coin_model_importance()**: Breaks down feature importance into component categories
- **calculate_coin_performance()**: Computes coin returns over specified periods
- **validate_coin_performance()**: Analyzes return performance of top-n coins by various metrics
- **analyze_market_cap_segments()**: Evaluates model performance across different market cap ranges
- **analyze_top_coins_wallet_metrics()**: Compares wallet metrics between top and bottom performing coins
- Includes visualization functions for heatmaps, consistency analysis, and segment comparisons

### coin_model_reporting.py
- **generate_and_save_coin_model_artifacts()**: Creates model evaluation metrics and saves all artifacts
- **generate_and_upload_coin_scores()**: Generates wallet scores and uploads to BigQuery tables
- **plot_wallet_model_comparison()**: Visualizes model performance across different epochs with macro indicators
- **aggregate_feature_importance()**: Aggregates feature importance statistics across multiple model runs
- **save_coin_model_artifacts()**: Handles model persistence and metadata storage

## Integration
- Uses `wallet_insights.model_evaluation` for core evaluation infrastructure
- Connects to `coin_modeling` for model results and pipeline management
- Integrates with `coin_features` for feature analysis and importance breakdown
- Works with BigQuery for score storage and retrieval

## Usage Patterns
1. Generate model artifacts and evaluation metrics after training
2. Analyze feature importance and model performance across segments
3. Compare models across different time periods and market conditions
4. Upload scores to database for downstream consumption
5. Create visualizations for model validation and comparison