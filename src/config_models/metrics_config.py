"""
Validation logic for items in metrics_config.yaml
"""
from enum import Enum
from typing import Dict, Optional, Literal
from pydantic import BaseModel, Field

# pylint: disable=C0115  # no docstring for class Config
# pylint: disable=R0903  # too few methods for class Config

# ----------------------------------------------------------------------------
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#                    metrics_config.yaml Main Configuration
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ----------------------------------------------------------------------------

class MetricsConfig(BaseModel):
    """
    Top level structure of the main metrics_config.yaml file
    """
    wallet_cohorts: Optional['WalletCohorts'] = None


    class Config:
        extra = "forbid"


# ============================================================================
# Wallet Cohorts Configurations
# ============================================================================

class WalletCohorts(BaseModel):
    """
    The top level wallet cohorts category in the config, e.g. metrics_config['wallet_cohorts]
    """
    cohorts: Optional[Dict[str, 'WalletCohort']] = Field(default_factory=dict)



# ----------------------------------------------------------------------------
# Wallet Cohort Configuration
# ----------------------------------------------------------------------------


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
    metrics: Dict[WalletCohortMetricType, 'Metric'] = Field(default_factory=dict)

    class Config:
        extra = "forbid"

    def __init__(self, **data):
        super().__init__(**data)
        self._validate_unique_metrics()

    def _validate_unique_metrics(self):
        if len(self.metrics) != len(set(self.metrics.keys())):
            raise ValueError("Duplicate metric types are not allowed in a wallet cohort")



class Metric(BaseModel):
    aggregations: Optional['Aggregations'] = None

class Aggregations(BaseModel):
    sum: Optional['ScalingConfig'] = None

class ScalingConfig(BaseModel):
    scaling: Literal["log", "standard", "none"] = "none"

