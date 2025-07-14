# data-science/wallet_investing

Validates whether the WalletModels are truly predictive across macroeconomic conditions.

## Example Sequence

#### Generate training data and train models:
- Predict wallet performance using training data as of 5/1/24
- Predict wallet performance using training data as of 6/1/24
- Predict wallet performance using training data as of 7/1/24

#### Review Model Performance
Compare model performance vs actual behavior as of those periods. If the model is highly predictive in May but incurs huge losses in June/July, the model is not truly predictive across macroeconomic conditions. A strong model would identify top performers during positive macro conditions, and output low scores across the board in poor macro conditions. We are looking for consistent accuracy over time, rather than perfect identification of the best coin in every period.

## Key Modules

**wallet_model_investing_orchestrator.py**
- `WalletModelInvestingOrchestrator` class - Generates wallet features for investment epochs and scores wallet-coin pairs using pre-trained models
- Inherits from CoinEpochsOrchestrator for epoch management while focusing on wallet-level prediction
- Converts wallet-coin scores into coin-level buy signals using aggregation rules and trading thresholds
- Supports both backtesting and live investment signal generation workflows

**wallet_temporal_searcher.py**
- `TemporalGridSearcher` class - Multi-temporal grid search orchestrator for assessing feature stability across different investing cycles
- Generates training data across multiple dates with configurable offsets and executes parallel grid search experiments
- Consolidates results to identify temporally stable parameter combinations and builds comparative models
- Provides comprehensive stability analysis using coefficient of variation metrics and feature importance aggregation

**wallet_investing_performance.py**
- `compute_epoch_buy_metrics()` - Analyzes buy and overall performance metrics for each investment epoch
- Calculates median/mean returns, best performance, and winsorized metrics for bought vs overall coin performance

## Integration

Uses pre-trained models from `wallet_modeling.WalletModel` and extends `coin_modeling.CoinEpochsOrchestrator` for epoch configuration. Integrates with `wallet_insights` for model evaluation and `coin_insights` for performance analysis.

## Usage

Primary workflow: `WalletModelInvestingOrchestrator.score_all_investment_cycles()` followed by `determine_epoch_buys()` to generate trading signals. The `TemporalGridSearcher` provides multi-date model validation for robust parameter selection before deployment.
