# data-science/coin_modeling

## Overview
The coin_modeling directory implements the coin-level predictive modeling pipeline. It orchestrates the generation of coin training data across multiple time periods, manages coin-specific configurations, and provides the core modeling infrastructure for predicting coin price movements based on aggregated wallet behavior and market data.

## Key Modules

### coin_model.py
- **CoinModel**: Extends BaseModel with coin-specific data preparation and modeling logic
- Handles multi-epoch coin training data, validation datasets, and asymmetric loss configurations
- Integrates custom grid search parameters and validation scoring for coin predictions

### coin_epochs_orchestrator.py
- **CoinEpochsOrchestrator**: Orchestrates training data generation across multiple time periods (epochs)
- Manages complete dataset loading, wallet model training, and coin feature generation
- Coordinates parallel processing of multiple epochs and handles data persistence across time windows

### coin_config_manager.py
- **WalletsCoinConfig**: Singleton class for loading and managing coin modeling configuration from YAML files
- Provides validation for score distribution features and epoch overlap checking
- Handles configuration reloading and dict-style access patterns

## Integration
- Extends `base_modeling.BaseModel` for core modeling infrastructure
- Integrates with `coin_features` for feature engineering and `wallet_modeling` for underlying wallet data
- Uses `training_data` for raw data retrieval and processing
- Coordinates with `coin_insights` for model evaluation and reporting

## Usage Patterns
The coin modeling pipeline follows a multi-stage approach:
1. Load complete datasets spanning all required time periods via CoinEpochsOrchestrator
2. Generate wallet training data and models for each epoch offset
3. Transform wallet-level features into coin-level aggregated features
4. Train coin models using CoinModel with optional grid search and validation
5. Score coins and evaluate performance across time periods
