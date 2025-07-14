# data-science/coin_features

## Overview
Builds coin-level features for predictive modeling by aggregating wallet behavior, market data, and blockchain metadata. Takes wallet trading patterns and transforms them into coin-level signals through segmentation and statistical aggregation.

Used by both the wallet_modeling and coin_modeling sequences, as any coin-level features that are appendeded to the wallet model's hybridized coin-wallet IDs are generated through this directory.

## Key Modules

### coin_features_orchestrator.py
- **CoinFeaturesOrchestrator**: Main pipeline coordinator that merges all feature sources
- Handles wallet metrics, time series data, metadata, and optional Coin Flow features
- Manages target variable calculation and feature validation

### wallet_metrics.py
- **compute_coin_wallet_metrics()**: Calculates trading and balance metrics for each coin-wallet pair
- Extracts wallet training features and computes starting/ending balances
- Creates the base data for wallet behavior aggregation

### wallet_metrics_flattening.py
- **flatten_cw_to_coin_segment_features()**: Converts coin-wallet metrics to coin-level features
- Aggregates across wallet segments with parallel processing
- Calculates distribution stats (percentiles, skewness, kurtosis) for wallet scores

### wallet_segmentation.py
- **build_wallet_segmentation()**: Groups wallets by scores, quantiles, and clusters
- Loads wallet scores and creates categorical segments for analysis
- Supports binary segments, defined cohorts, and cluster assignments

### coin_time_series.py
- **generate_macro_features()**: Flattens macroeconomic indicators into single-row features
- **generate_market_features()**: Creates market data features with consistent naming
- Converts time series into cross-joinable feature sets

### coin_metadata.py
- **retrieve_metadata_df()**: Gets blockchain metadata without survivorship bias
- Returns supply metrics, blockchain categories, and descriptive flags
- Avoids external sources like Coingecko to prevent data leakage

### coin_trends.py
- **generate_coin_trends_features()**: Builds holder and price trend features
- Tracks metrics like holders in profit and days since all-time high
- Includes validation and comprehensive trend analysis

## Integration
- Takes wallet training data from `wallet_modeling` for feature transformation
- Connects to `coin_modeling` for target variables and model inputs
- Uses `training_data` for raw market and blockchain data
- Works with optional `coin_flow` features when available

## Usage Patterns
1. Compute coin-wallet metrics from profits and wallet data
2. Segment wallets into analysis groups
3. Flatten wallet metrics into coin-level features by segment
4. Add time series features from macro and market data
5. Include metadata and trend features
6. Merge into final coin training dataset