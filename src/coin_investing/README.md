# data-science/coin_investing

## Overview
Implements investment strategy evaluation and multi-temporal model testing for coin predictions. Handles backtesting across investment cycles, grid search stability analysis, and performance measurement for coin trading strategies. We can only assess whether the CoinModel is truly predictive by assessing its performance across many months.

## Key Modules

### coin_investing_orchestrator.py
- **CoinInvestingOrchestrator**: Orchestrates wallet model scoring across multiple investment epochs
- Inherits from CoinEpochsOrchestrator to reuse data loading and pipeline infrastructure
- Processes investment cycles by training models and scoring subsequent periods
- Handles concurrent processing and performance calculation for investment backtesting

### coin_temporal_searcher.py
- **CoinTemporalGridSearcher**: Runs grid search experiments across multiple time periods
- Generates coin training data for all specified modeling dates with caching
- Analyzes parameter stability and consistency across different market conditions
- Consolidates results and identifies stable parameter combinations over time

### coin_investing_analysis.py
- **analyze_investment_performance_by_cycle()**: Computes performance metrics by investment cycle with threshold filtering
- **calculate_lifetime_performance_by_thresholds()**: Calculates compound returns across multiple score thresholds
- **generate_coin_metrics()**: Aggregates wallet scores into coin-level investment metrics
- **calculate_epoch_coin_returns()**: Calculates coin returns with macro indicators and winsorization

## Integration
- Extends `coin_modeling.CoinEpochsOrchestrator` for data orchestration infrastructure
- Uses `coin_insights` for model evaluation and artifact management
- Connects to `coin_features` for feature generation across time periods
- Integrates with `wallet_modeling` for underlying wallet data and model training

## Usage Patterns
1. Set up investment cycles with specific time offsets and holding periods
2. Train coin models for each cycle and score subsequent investment periods
3. Calculate actual returns and compare with model predictions
4. Run temporal grid search to find stable parameters across market conditions
5. Analyze compound returns and threshold performance for strategy optimization