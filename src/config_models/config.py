"""
Validation logic for items in config.yaml
"""
from datetime import date
from typing import Dict, Optional
from pydantic import BaseModel, Field

# pylint: disable=C0115  # no docstring for class Config
# pylint: disable=R0903  # too few methods for class Config


# ____________________________________________________________________________
# ----------------------------------------------------------------------------
#                      config.yaml Main Configuration
# ----------------------------------------------------------------------------
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
class MainConfig(BaseModel):
    """
    Top level structure of the main config.yaml file
    """
    training_data: 'TrainingDataConfig' = Field(...)
    datasets: 'DatasetsConfig' = Field(...)
    data_cleaning: 'DataCleaningConfig' = Field(...)

    model_config = {
        "extra": "forbid",  # Prevent extra fields that are not defined
        "str_max_length": 2000  # Increase the max length of error message string representations
    }

# ============================================================================
# Training Data Configuration
# ============================================================================

class TrainingDataConfig(BaseModel):
    """
    These variables relate to how the training period is defined
    """
    training_period_duration: int = Field(..., gt=0)
    training_period_start: date = Field(...)
    training_period_end: date = Field(...)
    modeling_period_duration: int = Field(..., gt=0)
    modeling_period_start: date = Field(...)
    modeling_period_end: date = Field(...)

# ============================================================================
# Datasets Configuration
# ============================================================================

class DatasetsConfig(BaseModel):
    """
    These items represent categories of datasets that will be converted to features and used
    by the model. Each category must contain at least one item.
    """
    wallet_cohorts: Dict[str, 'WalletCohortConfig'] = Field(..., min_length=1)
    time_series: Dict[str, Dict[str, 'TimeSeriesDataConfig']] = Field(..., min_length=1)
    coin_facts: Dict[str, 'CoinFactsConfig'] = Field(..., min_length=1)

    model_config = {
        "extra": "forbid",  # Prevent extra fields that are not defined
    }

# ----------------------------------------------------------------------------
# Wallet Cohort Configuration
# ----------------------------------------------------------------------------

class WalletCohortConfig(BaseModel):
    """
    This data category is a series of metrics generated based on the behavior of a cohort
    of wallets, which are defined using the variables within.
    """
    description: str = Field(...)
    fill_method: str = Field(...)
    sameness_threshold: float = Field(..., ge=0, le=1)
    wallet_minimum_inflows: float = Field(..., ge=0)
    wallet_maximum_inflows: float = Field(..., gt=0)
    coin_profits_win_threshold: float = Field(...)
    coin_return_win_threshold: float = Field(...)
    wallet_min_coin_wins: int = Field(..., ge=0)

# ----------------------------------------------------------------------------
# Time Series Data Configuration
# ----------------------------------------------------------------------------

class TimeSeriesDataConfig(BaseModel):
    """
    This data category includes any dataset keyed on both coin_id and date.
    """
    description: str = Field(...)
    fill_method: str = Field(...)
    sameness_threshold: float = Field(..., ge=0, le=1)

# ----------------------------------------------------------------------------
# Coin Facts Configuration
# ----------------------------------------------------------------------------

class CoinFactsConfig(BaseModel):
    """
    This data category includes items only keyed on coin_id and not date, meaning that they
    do not generally change over time. Examples include a token's category, blockchain,
    fee structure, etc.
    """
    description: str = Field(...)
    fill_method: str = Field(...)
    sameness_threshold: float = Field(..., ge=0, le=1)
    chain_threshold: Optional[int] = Field(None, ge=0)

# ============================================================================
# Data Cleaning Configuration
# ============================================================================

class DataCleaningConfig(BaseModel):
    """
    Variables used to clean and filter raw data before training data is built
    """
    profitability_filter: float = Field(..., gt=0)
    inflows_filter: float = Field(..., gt=0)
    max_gap_days: int = Field(..., gt=0)

# ============================================================================
# Model Rebuilding
# ============================================================================

MainConfig.model_rebuild()
