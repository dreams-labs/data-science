"""
Validation logic for items in config.yaml
"""
from datetime import date
from typing import Dict, Optional, Literal
from pydantic import BaseModel, Field

# Custom base model to disable extra fields in all sections
class NoExtrasBaseModel(BaseModel):
    """Custom BaseModel to apply config settings globally."""
    model_config = {
        "extra": "forbid",  # Prevent extra fields that are not defined
        "str_max_length": 2000,    # Increase the max length of error message string representations
    }

# ============================================================================
#                      config.yaml Main Configuration
# ============================================================================
class MainConfig(NoExtrasBaseModel):
    """
    Top level structure of the main config.yaml file
    """
    training_data: 'TrainingDataConfig' = Field(...)
    datasets: 'DatasetsConfig' = Field(...)
    data_cleaning: 'DataCleaningConfig' = Field(...)


# ----------------------------------------------------------------------------
# Training Data Section
# ----------------------------------------------------------------------------

class TrainingDataConfig(NoExtrasBaseModel):
    """
    These variables relate to how the training period is defined
    """
    training_period_duration: int = Field(..., gt=0)
    training_period_start: date = Field(...)
    training_period_end: date = Field(...)
    modeling_period_duration: int = Field(..., gt=0)
    modeling_period_start: date = Field(...)
    modeling_period_end: date = Field(...)
    additional_windows: int = Field(..., ge=0)
    time_window_frequency: int = Field(..., gt=0)
    earliest_window_start: date = Field(...)
    earliest_cohort_lookback_start: date = Field(...)
    dataset: Literal['prod', 'dev'] = Field(...)


# ----------------------------------------------------------------------------
# Datasets Section
# ----------------------------------------------------------------------------

class DatasetsConfig(NoExtrasBaseModel):
    """
    These items represent categories of datasets that will be converted to features and used
    by the model. Each category must contain at least one item.
    """
    wallet_cohorts: Optional[Dict[str, 'WalletCohortConfig']] = None
    time_series: Optional[Dict[str, Dict[str, 'TimeSeriesDataConfig']]] = None
    coin_facts: Optional[Dict[str, 'CoinFactsConfig']] = None
    macro_trends: Optional[Dict[str, 'MacroTrendsConfig']] = None

# Wallet Cohorts Configuration
# ---------------------------
class WalletCohortConfig(NoExtrasBaseModel):
    """
    This data category is a series of metrics generated based on the behavior of a cohort
    of wallets, which are defined using the variables within.
    """
    sameness_threshold: float = Field(..., ge=0, le=1)
    lookback_period: int = Field(..., ge=0)
    wallet_minimum_inflows: float = Field(..., ge=0)
    wallet_maximum_inflows: float = Field(..., gt=0)
    coin_profits_win_threshold: float = Field(...)
    coin_return_win_threshold: float = Field(...)
    wallet_min_coin_wins: int = Field(..., ge=0)

# Time Series Data Configuration
# ------------------------------
class TimeSeriesDataConfig(NoExtrasBaseModel):
    """
    This data category includes any dataset keyed on both coin_id and date.
    """
    sameness_threshold: float = Field(..., ge=0, le=1)

# Coin Facts Configuration
# ------------------------
class CoinFactsConfig(NoExtrasBaseModel):
    """
    This data category includes items only keyed on coin_id and not date, meaning that they
    do not generally change over time. Examples include a token's category, blockchain,
    fee structure, etc.
    """
    sameness_threshold: float = Field(..., ge=0, le=1)
    chain_threshold: Optional[int] = Field(None, ge=0)

# Macro Trends Configuration
# ------------------------
class MacroTrendsConfig(NoExtrasBaseModel):
    """
    This data category includes items only keyed on coin_id and not date, meaning that they
    do not generally change over time. Examples include a token's category, blockchain,
    fee structure, etc.
    """
    sameness_threshold: float = Field(..., ge=0, le=1)


# ----------------------------------------------------------------------------
# Data Cleaning Section
# ----------------------------------------------------------------------------

class DataCleaningConfig(NoExtrasBaseModel):
    """
    Variables used to clean and filter raw data before training data is built
    """
    max_wallet_inflows: float = Field(..., gt=0)
    max_gap_days: int = Field(..., gt=0)
    min_daily_volume: float = Field(..., gt=0)
    min_wallet_inflows: float = Field(..., gt=0)
    maximum_market_cap_share: float = Field(..., gt=0)
    exclude_coins_without_transfers: bool = Field(False)
    price_coverage_warning_min_coin_increase: int = Field(..., gt=0)  # if this many coin prices has recently become imputed...
    price_coverage_warning_min_pct_increase: float = Field(..., gt=0)  # ...and the % of imputed coin prices has increased by this value, issue a warning
    transfers_coverage_warning_min_coin_increase: int = Field(..., gt=0)  # if this many coins have lost transfers data...
    transfers_coverage_warning_min_pct_increase: float = Field(..., gt=0)  # ...and the % of coin without transfers data has increased by this value, issue a warning



# ============================================================================
# Model Rebuilding
# ============================================================================
# Ensures all classes are fully reflected in structure regardless of the order they were defined
MainConfig.model_rebuild()
