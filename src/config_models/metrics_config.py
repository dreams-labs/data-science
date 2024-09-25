"""
Validation logic for items in metrics_config.yaml
"""
# pylint: disable=W0611
from enum import Enum
from typing import Dict, Optional, Literal, Any, Annotated
from pydantic import BaseModel, RootModel, Field, model_validator

# pylint: disable=C0115  # no docstring for class Config
# pylint: disable=R0903  # too few methods for class Config


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Modular Metrics System Components
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Metric(BaseModel):
    """
    This config file defines how a time series should be flattened into a single row by defining
    the columns that will show up in the row, e.g. sum of buyers_new in the sharks cohort. These
    are the three ways that metrics can be generated from time series data:

    * Aggregations are calculations that flatten an entire series, e.g. sum, last, max, etc.

    * RollingMetrics split the time series into x lookback_periods of y window_duration days. The
        lookback periods can then be flattened into Aggregations or compared with each other
        using Comparisons.

    * Indicator metrics are transformations that result in a new time series, e.g. SMA, EMA, RSI.
        Indicator metrics can flattened through Aggregations or RollingMetrics.

    Finally, all of these types of metrics can be scaled using ScalingConfig after they've been
    calculated. Scaling is applied to the dataframe containing every coin_id's values for the
    metric and done as part of preprocessing.
    """
    aggregations: Optional['Aggregations'] = None
    rolling: Optional['RollingMetrics'] = None
    indicators: Optional[Dict[str, 'Indicators']] = None


# ----------------------------------------------------------------------------
# Modular Metrics: Aggregations
# ----------------------------------------------------------------------------
class AggregationType(str, Enum):
    """
    Lists the aggregation functions that can be applied
    """
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    STD = "std"
    MAX = "max"
    MIN = "min"
    FIRST = "first"
    LAST = "last"

# Use RootModel for dynamic aggregation types
class Aggregations(RootModel[Dict[AggregationType, 'ScalingConfig']]):
    """
    RootModel: This how to define a class that acts as a wrapper around a dictionary. The
    Aggregations class will now dynamically accept any valid AggregationType as keys and
    their corresponding ScalingConfig values.
    """
    pass  # pylint: disable=W0107


# ----------------------------------------------------------------------------
# Modular Metrics: RollingMetrics and Comparisons
# ----------------------------------------------------------------------------
class ComparisonType(str, Enum):
    """
    Lists the functions that can be used to compare rolling metric periods.
    """
    CHANGE = "change"
    PCT_CHANGE = "pct_change"


class RollingMetrics(BaseModel):
    """
    Rolling metrics generates a column for each {lookback_period} that last for
    {window_duration} days which can then be flattened through Aggregations or Comparisons.
    """
    window_duration: Annotated[int, Field(gt=0)]  # Must be an integer > 0
    lookback_periods: Annotated[int, Field(gt=0)]  # Must be an integer > 0
    aggregations: Optional['Aggregations'] = None
    comparisons: Optional[Dict['ComparisonType', 'ScalingConfig']] = None


class Comparisons(BaseModel):
    """
    Comparisons are performed between the first and last value in any given lookback_period.

    See feature_engineering.calculate_comparisons().
    """
    comparison_type: ComparisonType
    scaling: Optional['ScalingConfig'] = None


# ----------------------------------------------------------------------------
# Modular Metrics: Indicators
# ----------------------------------------------------------------------------
class IndicatorType(str, Enum):
    """
    Lists the technical analysis indicators that can be created
    """
    SMA = "sma"
    EMA = "ema"

class Indicators(BaseModel):
    parameters: Dict[str, Any]  # Flexible to handle unique parameters
    aggregations: Optional['Aggregations'] = None
    rolling: Optional['RollingMetrics'] = None


# ----------------------------------------------------------------------------
# Modular Metrics: ScalingConfig
# ----------------------------------------------------------------------------
class ScalingConfig(BaseModel):
    scaling: Literal["log", "standard", "none"] = "none"





# \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
#                    metrics_config.yaml Main Configuration
# /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\

class MetricsConfig(BaseModel):
    """
    Top level structure of the main metrics_config.yaml file.

    Wallet cohorts conform to the structure of a dict keyed on the cohort name and containing
    a dict with keys that match WalletCohortMetricType.
    """
    wallet_cohorts: Dict[str, Dict['WalletCohortMetricType', 'Metric']]  # Direct mapping

    model_config = {
        "extra": "forbid"
    }

# ============================================================================
# Wallet Cohort Metrics
# ============================================================================

class WalletCohortMetricType(str, Enum):
    """
    A list of all valid names of wallet cohort buysell metrics.
    """
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


