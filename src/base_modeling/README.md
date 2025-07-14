# data-science/base_modeling

Core modeling framework providing reusable components for XGBoost-based prediction models.

## Overview

This directory contains the foundational classes and utilities used by both `wallet_modeling` and `coin_modeling`. It handles the complex orchestration of feature preprocessing, target variable transformations, and model training while maintaining compatibility with scikit-learn's ecosystem.

## Key Modules

### `base_model.py`
**BaseModel** - Abstract modeling engine that orchestrates the complete ML workflow:
- Data preparation and train/eval/test splitting
- Pipeline construction with custom transformations
- Grid search with early stopping and custom metrics

Extended by `WalletModel` and `CoinModel` for domain-specific implementations.

### `pipeline.py`
**MetaPipeline** - Coordinates separate X (features) and y (targets) transformations:
- **TargetVarSelector** - Extracts target columns, applies thresholds, handles asymmetric loss
- **FeatureSelector** - Removes low-variance and highly-correlated features
- **DropColumnPatterns** - Pattern-based feature removal with wildcard support

### `scorers.py`
Custom scoring functions that handle y-transformations during grid search:
- `custom_r2_scorer`, `custom_neg_rmse_scorer` - Transform targets before evaluation
- `validation_*_scorer` family - Use separate validation datasets for realistic performance assessment
- `validation_top_percentile_returns_scorer` - Business-specific metric for investment performance

### `feature_selection.py`
Utilities for automated feature removal:
- Variance-based filtering with protected feature lists
- Correlation-based removal with pattern matching
- Wildcard column matching with `fnmatch` support

## Integration with Domain Models

### Wallet Modeling
- `WalletModel` extends `BaseModel` for wallet behavior prediction
- Handles multi-epoch training data and cohort filtering
- Supports both regression (return prediction) and classification (win/loss prediction)

### Coin Modeling
- `CoinModel` extends `BaseModel` for cryptocurrency price forecasting
- Integrates wallet-derived features with market/macro indicators
- Specialized target variable generation for price movement classification

## Key Features

- **Unified Interface**: Same API for regression, binary classification, and asymmetric 3-class problems
- **Custom Scorers**: Proper evaluation during grid search despite complex target transformations
- **Early Stopping**: Uses separate eval sets (not CV folds) for realistic overfitting prevention
- **Memory Efficient**: Pattern-based feature dropping and optional S3 export mode
- **Extensible**: Clean inheritance pattern for domain-specific model classes

## Usage Pattern

```python
# Domain models inherit from BaseModel
class WalletModel(BaseModel):
   def construct_wallet_model(self, X, y):
       # Domain-specific data prep
       X, y = self._prepare_data(X, y)
       # BaseModel handles the rest
       return super().construct_base_model(X, y)