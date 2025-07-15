# data-science/wallet_modeling

This is the primary repo for WalletModel construction, from training data preparation, feature engineering, and modeling. The scores from the WalletModel can then be aggregated to the coin-level and used as features for the coin_modeling/coin_model.py.

This sequence makes heavy use of orchestration functions that coordinate key steps of the modeling pipeline. This also allows for concurrent feature generation by calling multiple orchestrators to generate data for individual time offsets relative to the base modeling period.

## Key Modules

**wallet_model.py**
- `WalletModel` class - Core modeling engine extending BaseModel for wallet-specific prediction
- Handles data preparation, cohort selection, epoch assignment, and asymmetric loss
- Supports both classification and regression with hybridized wallet-coin pair modeling
- Includes S3 export capability for Sagemaker workflows

**wallets_config_manager.py**
- `WalletsConfig` class - Singleton configuration manager for wallet modeling parameters
- `load_all_wallets_configs()` function - Reloads both wallet and wallet-coin configurations
- `add_derived_values()` function - Calculates period boundaries and validation dates
- Handles dataset switching (prod/dev) and folder path management

**wallet_training_data.py**
- `WalletTrainingData` class - Wrapper for training data preparation functions
- Raw dataset retrieval with wallet cohort filtering and market data cleaning
- Training window splitting and wallet threshold application
- BigQuery cohort upload for complex feature generation

**wallet_model_orchestrator.py**
- `WalletModelOrchestrator` class - Trains multiple wallet scoring models with different parameter configurations
- Loads existing models or trains new ones based on score parameters
- Generates predictions and stores them as parquet files for downstream coin modeling
- Handles model evaluation and performance plotting

**wallet_training_data_orchestrator.py**
- `WalletTrainingDataOrchestrator` class - Orchestrates wallet feature generation and data preparation
- Coordinates raw data retrieval, cleaning, and feature calculation across multiple time windows
- Manages hybrid wallet-coin ID mapping and dehybridization cycles
- Concurrent processing of training and modeling features

**wallet_epochs_orchestrator.py**
- `WalletEpochsOrchestrator` class - Builds and caches wallet-epoch features across multiple time periods
- Transforms raw datasets into monthly feature snapshots for training/validation periods
- Handles epoch configuration generation with date offsetting
- Multithreaded epoch processing with parquet caching and coverage validation

**s3_exporter.py**
- `export_s3_training_data()` function - Exports exact training data splits for S3 upload
- Handles dev mode sampling and metadata generation
- Supports external training workflows outside the main pipeline

## Integration

Inherits from `base_modeling.BaseModel` for core XGBoost functionality and extends it with wallet-specific data transformations. Integrates with `wallet_features` modules for feature generation and uses `coin_modeling` components for downstream analysis.

## Usage

Primary entry point is `WalletEpochsOrchestrator.generate_epochs_training_data()` which returns training and validation datasets ready for `WalletModel.construct_wallet_model()`. The `WalletModelOrchestrator` coordinates multiple model training for different scoring objectives.
