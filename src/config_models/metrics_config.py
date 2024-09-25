"""
Validation logic for items in metrics_config.yaml
"""
# pylint: disable=W0611
from enum import Enum
from typing import Dict, Optional, Literal, Any, Annotated, List, Union
from pydantic import BaseModel, RootModel, Field, model_validator, conlist

# pylint: disable=C0115  # no docstring for class Config
# pylint: disable=R0903  # too few methods for class Config
# pylint: disable=W0107  # unneccesary pass in the RootModel classes
# pylint: disable=E0213  # we are defining classes and not class instancees so we don't need "self"


# ____________________________________________________________________________
# ----------------------------------------------------------------------------
#                   metrics_config.yaml Main Configuration
# ----------------------------------------------------------------------------
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

class MetricsConfig(BaseModel):
    """
    Top level structure of the main metrics_config.yaml file.

    Wallet cohorts conform to the structure of a dict keyed on the cohort name and containing
    a dict with keys that match WalletCohortMetricType.
    """
    wallet_cohorts: Optional[Dict[str, 'WalletCohort']] = Field(default=None)
    time_series: Optional[Dict[str, 'TimeSeriesValueColumn']] = Field(default=None)

    model_config = {
        "extra": "forbid",  # Prevent extra fields that are not defined
        "str_max_length": 2000  # Increase the max length of error message string representations
    }

# ============================================================================
# Wallet Cohort Metrics
# ============================================================================

class WalletCohortMetric(str, Enum):
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


class WalletCohort(RootModel[Dict['WalletCohortMetric', 'Metric']]):
    """
    Represents a cohort that contains valid cohort metrics (e.g. total_bought, buyers_new, etc.)
    and their associated modular metrics flattening definitions.
    """
    pass


# ============================================================================
# Time Series Metrics
# ============================================================================

class TimeSeriesValueColumn(RootModel[Dict[str, 'Metric']]):
    """
    Represents a dataset that contains a value_column such as price, volume, etc. and their
    corresponding metrics flattening definitions.

    RootModel is used to define a class that acts as a wrapper around a dictionary.
    """
    pass



# ____________________________________________________________________________
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Modular Metrics Flattening System
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

class Metric(BaseModel):
    """
    The Metric config class defines how a time series should be flattened into a single row by
    specifying the columns that will show up in the row, e.g. sum of buyers_new in the sharks
    cohort. These are the three ways that metrics can be generated from time series data:

    * Aggregations are calculations that flatten an entire series, e.g. sum, last, max, etc. The
        Bucket

    * RollingMetrics split the time series into x lookback_periods of y window_duration days. The
        lookback periods can then be flattened into Aggregations or compared with each other
        using Comparisons.

    * Indicator metrics are transformations that result in a new time series, e.g. SMA, EMA, RSI.
        Indicator metrics can flattened through Aggregations or RollingMetrics.

    Finally, all of these types of metrics can be scaled using ScalingConfig after they've been
    calculated. Scaling is applied to the dataframe containing every coin_id's values for the
    metric and done as part of preprocessing.
    """
    aggregations: Optional[Dict['AggregationType', 'AggregationConfig']] = Field(default=None)
    rolling: Optional['RollingMetrics'] = Field(default=None)
    indicators: Optional[Dict[str, 'Indicators']] = Field(default=None)

    @model_validator(mode='after')
    def remove_empty_fields(cls, values):
        """
        Remove all empty dictionaries, None values, or empty lists from the nested structure.
        """
        values_dict = values.dict(exclude_none=True)  # Exclude None values
        cleaned_dict = remove_empty_dicts(values_dict)  # Recursively remove empty dictionaries
        return cleaned_dict

def remove_empty_dicts(data):
    """
    Recursively remove all empty dictionaries from a nested structure.
    """
    if isinstance(data, dict):
        return {k: remove_empty_dicts(v) for k, v in data.items() if v not in [None, {}, []]}
    elif isinstance(data, list):
        return [remove_empty_dicts(item) for item in data if item not in [None, {}, []]]
    else:
        return data


# Modular Metrics: Aggregations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class AggregationType(str, Enum):
    """
    List of aggregation functions that can be applied
    """
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    STD = "std"
    MAX = "max"
    MIN = "min"
    FIRST = "first"
    LAST = "last"

class AggregationConfig(BaseModel):
    """
    Defines the configuration for each aggregation type.
    An aggregation can have a scaling field or a buckets field.
    """
    scaling: Optional[str] = Field(default=None)
    buckets: Optional['BucketsList'] = Field(default=None)



# Modular Metrics: Buckets
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
class BucketsList(RootModel[List[Dict[str, Union[float, str]]]]):
    """
    Represents a collection of bucket tiers, ensuring that one bucket contains 'remainder'.
    The RootModel now directly supports a list of dictionaries, where the key is the label
    and the value is the threshold.
    """

    @model_validator(mode='after')
    def validate_remainder_bucket(cls, values):
        """confirms that a bucket has the 'remainder' value """
        remainder_found = any("remainder" in bucket.values() for bucket in values.root)
        if not remainder_found:
            raise ValueError("At least one bucket must have the 'remainder' value.")
        return values


# Modular Metrics: RollingMetrics and Comparisons
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class ComparisonType(str, Enum):
    """
    List of functions that can be used to compare rolling metric periods.
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
    aggregations: Optional['AggregationConfig'] = Field(default=None)
    comparisons: Optional[Dict[ComparisonType, Optional['ScalingConfig']]] = Field(default=None)

    @model_validator(mode='after')
    def validate_comparisons_scaling(cls, values):
        """
        Validate that all ComparisonType entries in comparisons have a valid 'scaling' config.
        """
        comparisons = values.comparisons  # Access attribute directly
        if comparisons:
            for comparison_type, scaling_config in comparisons.items():
                if not scaling_config or not scaling_config.scaling:
                    raise ValueError(
                        f"Comparison '{comparison_type}' requires a valid 'scaling' configuration."
                        )
        return values



class Comparisons(BaseModel):
    """
    Comparisons are performed between the first and last value in any given lookback_period.

    See feature_engineering.calculate_comparisons().
    """
    comparison_type: ComparisonType
    scaling: Optional['ScalingConfig'] = Field(default=None)



# Modular Metrics: Indicators
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
class IndicatorType(str, Enum):
    """
    List of technical analysis indicators that can be created
    """
    SMA = "sma"
    EMA = "ema"

class Indicators(BaseModel):
    parameters: Dict[str, Any]  # Flexible to handle unique parameters
    aggregations: Optional['AggregationConfig'] = Field(default=None)
    rolling: Optional['RollingMetrics'] = Field(default=None)


# Modular Metrics: ScalingConfig
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class ScalingType(str, Enum):
    """
    List of scaling methods that can be applied
    """
    STANDARD = "standard"
    MINMAX = "minmax"
    LOG = "log"
    NONE = "none"

class ScalingConfig(BaseModel):
    """
    Configuration for applying scaling to metrics.
    """
    scaling: ScalingType  # Make scaling required for any ComparisonType



# ============================================================================
# Model Rebuilding
# ============================================================================

MetricsConfig.model_rebuild()
