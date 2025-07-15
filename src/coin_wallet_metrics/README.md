# data-science/coin_wallet_metrics

## Overview
Provides wallet cohort analysis and technical indicator calculations for the Coin Flow model. This module processes large-scale transaction data to identify wallet behavior patterns and generate trading signals through technical indicators applied to time series data.

## Key Modules

### `indicators.py`
Technical indicator calculation engine for time series analysis:
- `generate_time_series_indicators()` - Applies configured indicators to time series datasets
- `calculate_sma()`, `calculate_ema()`, `calculate_rsi()` - Standard technical indicators
- `calculate_bollinger_bands()`, `calculate_mfi()` - Advanced volatility and momentum indicators
- `identify_crossovers()`, `generalized_obv()` - Signal detection and volume analysis
- Supports both single-series and multi-coin analysis with proper grouping

### `coin_wallet_metrics.py`
Wallet cohort classification and trading metrics generation:
- `classify_wallet_cohort()` - Segments wallets into behavioral cohorts based on profit thresholds
- `generate_buysell_metrics_df()` - Aggregates trading behavior metrics across wallet cohorts
- `split_dataframe_by_coverage()` - Handles data quality by separating complete vs partial coverage
- `apply_period_boundaries()` - Manages temporal data filtering for training periods
- Processes coin metadata and macro trend features for modeling

## Integration
- Consumed by `feature_engineering` modules for model feature generation
- Uses `profits_df` data structures from `training_data` pipeline
- Integrates with `utils` for performance optimization and data validation
- Outputs feed into both wallet-level and coin-level modeling workflows

## Usage Patterns
Designed for batch processing of large datasets (~200M+ rows) with emphasis on vectorized operations and memory efficiency. Typically used in feature engineering pipelines where wallet behavior classification and technical indicators are combined to create predictive features for cryptocurrency investment models.
