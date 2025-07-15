# data-science/src/config_models

Pydantic-based configuration validation system for the wallet modeling pipeline. Provides type-safe, validated configuration objects that prevent runtime errors from malformed YAML files and enforce business rules across the modeling workflow.

## Overview

This directory contains Pydantic models that validate and structure configuration data loaded from YAML files. Each model corresponds to a specific configuration file in the pipeline, ensuring data integrity and providing clear interfaces for configuration access throughout the system.

## Key Modules

**modeling_config.py**
- `ModelingConfig` - Validates modeling pipeline configuration including preprocessing, target variables, model settings, and evaluation metrics
- `ModelType`, `TargetColumn`, `EvaluationMetric` enums - Constrain valid model types and evaluation approaches
- `PreprocessingConfig`, `ModelingSettings`, `EvaluationConfig` - Modular configuration sections for different pipeline stages

**config.py**
- `MainConfig` - Validates core system configuration including training periods, datasets, and data cleaning parameters
- `TrainingDataConfig` - Defines temporal boundaries and dataset selection for model training
- `DatasetsConfig` - Configures wallet cohorts, time series data, coin facts, and macro trends
- `DataCleaningConfig` - Sets filtering thresholds and data quality parameters

**experiments_config.py**
- `ExperimentsConfig` - Validates hyperparameter search and experiment metadata configuration
- `MetadataConfig` - Experiment naming, search methods, and evaluation parameters
- `VariableOverrides` - Allows selective overriding of base configurations for experiments

**metrics_config.py**
- `MetricsConfig` - Validates complex metrics calculation configuration with modular aggregation system
- `Metric`, `AggregationType`, `IndicatorType` - Defines available metric calculations and transformations
- `WalletCohort`, `TimeSeriesValueColumn` - Specialized metric configurations for different data types
- Modular metrics system supporting aggregations, rolling windows, technical indicators, and scaling

## Integration

These configuration models are imported throughout the wallet modeling system to:
- Validate YAML configuration files at startup
- Provide type hints and autocompletion in IDEs
- Enforce business rules and valid parameter ranges
- Enable configuration inheritance and overrides for experiments

## Usage Patterns

Configuration models follow the `NoExtrasBaseModel` pattern that prevents unexpected fields and provides detailed validation error messages. Models use Pydantic's validation features including field constraints, custom validators, and automatic type conversion.