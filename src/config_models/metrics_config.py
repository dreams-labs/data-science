"""
Validation logic for items in metrics_config.yaml
"""
from enum import Enum
from typing import Dict, Optional, Literal
from pydantic import BaseModel, Field, model_validator

# pylint: disable=C0115  # no docstring for class Config
# pylint: disable=R0903  # too few methods for class Config


# pylint: disable=W0611
# ----------------------------------------------------------------------------
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#                    metrics_config.yaml Main Configuration
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ----------------------------------------------------------------------------

class MetricsConfig(BaseModel):
    """
    Top level structure of the main metrics_config.yaml file
    """
    wallet_cohorts: Dict[str, 'WalletCohort'] = Field(default_factory=dict)

    model_config = {
        "extra": "forbid"
    }

    @model_validator(mode='after')
    def check_non_empty_cohorts(self) -> 'MetricsConfig':
        if not self.wallet_cohorts:
            raise ValueError("wallet_cohorts must contain at least one cohort")
        return self

# ============================================================================
# Wallet Cohorts
# ============================================================================

class WalletCohortMetricType(str, Enum):
    TOTAL_BALANCE = "total_balance"
    TOTAL_HOLDERS = "total_holders"
    TOTAL_BOUGHT = "total_bought"
    TOTAL_SOLD = "total_sold"
    TOTAL_NET_TRANSFERS = "total_net_transfers"
    TOTAL_VOLUME = "total_volume"
    BUYERS_NEW = "buyers_new"
    BUYERS_REPEAT = "buyers_repeat"
    SELLERS_NEW = "sellers_new"
    SELLERS_REPEAT = "sellers_repeat"


class WalletCohort(BaseModel):
    """
    This represents a given wallet cohort, e.g. ['sharks'],['under_15k'], etc
    """
    metrics: Dict[WalletCohortMetricType, 'Metric'] = Field(default_factory=dict)

    @model_validator(mode='after')
    def check_non_empty(self) -> 'WalletCohort':
        if not self.metrics:
            raise ValueError("WalletCohort must contain at least one metric")
        return self

class Metric(BaseModel):
    aggregations: Optional['Aggregations'] = None

class Aggregations(BaseModel):
    sum: Optional['ScalingConfig'] = None

class ScalingConfig(BaseModel):
    scaling: Literal["log", "standard", "none"] = "none"


# ----------------------------------------------------------------------------
# Wallet Cohort Configuration
# ----------------------------------------------------------------------------
