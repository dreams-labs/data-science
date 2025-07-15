# data-science/feature_engineering

## Overview
Comprehensive feature engineering pipeline that primarily relates to the Coin Flow model. transforms raw cryptocurrency market and transaction data into model-ready features. Handles multi-window time series analysis, target variable creation, and preprocessing for both wallet-level and coin-level machine learning models.

## Key Modules

### `coin_flow_features_orchestrator.py`
Main orchestration class for the complete feature engineering pipeline:
- `CoinFlowFeaturesOrchestrator` - Manages multi-window feature generation across all datasets
- `generate_all_time_windows_model_inputs()` - Executes full pipeline from raw data to model inputs
- `prepare_configs()` - Handles configuration management for different time windows
- Coordinates market data, wallet cohorts, and macro trends feature generation

### `target_variables.py`
Target variable creation for supervised learning models:
- `create_target_variables()` - Main interface for target generation
- `calculate_coin_returns()` - Computes price performance over modeling periods
- `calculate_mooncrater_targets()` - Creates binary classification targets for extreme movements
- `create_target_variables_for_all_time_windows()` - Multi-window target variable generation

### `flattening.py`
Transforms time series data into feature vectors for machine learning:
- `flatten_coin_date_df()` - Converts coin-date time series to coin-level features
- `calculate_rolling_window_features()` - Generates rolling window aggregations and comparisons
- `calculate_aggregation()` - Statistical aggregations (mean, std, percentiles, etc.)
- Handles complex indicator promotion and bucketing transformations

### `data_splitting.py`
Manages train/test/validation splits with temporal considerations:
- `perform_train_test_validation_future_splits()` - Main splitting orchestrator
- `split_future_set()` - Separates future time windows for realistic validation
- `split_validation_set()` - Creates validation splits by coin_id to prevent leakage
- Includes comprehensive data quality validation

### `preprocessing.py`
Feature scaling and preprocessing pipeline:
- `DataPreprocessor` - Main preprocessing class with configurable steps
- `ScalingProcessor` - Handles feature scaling based on metrics configuration
- `preprocess_sets_X_y()` - Applies consistent preprocessing across all data splits
- Supports categorical encoding, feature dropping, and variance-based filtering

### `feature_generation.py`
Window-specific feature generation functions:
- `generate_window_time_series_features()` - Market data and technical indicators
- `generate_window_macro_trends_features()` - Economic indicator features
- `generate_window_wallet_cohort_features()` - Wallet behavior pattern features
- Manages feature naming conventions and file persistence

## Integration
- Primary consumer of `coin_wallet_metrics` for technical indicators and cohort analysis
- Uses `training_data` modules for raw data retrieval and profits imputation
- Integrates with `utils` for configuration management and performance optimization
- Outputs directly consumed by modeling workflows in other directories

## Usage Patterns
Designed for batch processing of multiple time windows simultaneously. The orchestrator handles memory-efficient processing of large datasets (~200M+ rows) while maintaining temporal integrity. Features are automatically saved and can be incrementally generated across different time periods for backtesting and model validation.