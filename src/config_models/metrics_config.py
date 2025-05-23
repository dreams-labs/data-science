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

# Custom base model to disable extra fields in all sections
class NoExtrasBaseModel(BaseModel):
    """Custom BaseModel to apply config settings globally."""
    model_config = {
        "extra": "forbid",  # Prevent extra fields that are not defined
        "str_max_length": 2000,    # Increase the max length of error message string representations
    }

# ____________________________________________________________________________
# ----------------------------------------------------------------------------
#                   metrics_config.yaml Main Configuration
# ----------------------------------------------------------------------------
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾


class MetricsConfig(NoExtrasBaseModel):
    """
    Top level structure of the main metrics_config.yaml file.

    Wallet cohorts conform to the structure of a dict keyed on the cohort name and containing
    a dict with keys that match WalletCohortMetricType.
    """
    wallet_cohorts: Optional[Dict[str, 'WalletCohort']] = Field(default=None)
    time_series: Optional[Dict[str, 'TimeSeriesValueColumn']] = Field(default=None)
    macro_trends: Optional[Dict['MacroTrendMetric', 'Metric']] = Field(default=None)


# Wallet Cohort Metrics
# ---------------------
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


# Time Series Metrics
# -------------------
class TimeSeriesValueColumn(RootModel[Dict[str, 'Metric']]):
    """
    Represents a dataset that contains a value_column such as price, volume, etc. and their
    corresponding metrics flattening definitions.

    RootModel is used to define a class that acts as a wrapper around a dictionary.
    """
    pass


# Macro Trends Metrics
# -------------------
class MacroTrendMetric(str, Enum):
    """
    A list of all valid names of macro trend metrics.
    """
    BTC_PRICE = "btc_price"
    BTC_CDD_TERMINAL_ADJUSTED_90DMA = "btc_cdd_terminal_adjusted_90dma"
    BTC_FEAR_AND_GREED = "btc_fear_and_greed"
    BTC_MVRV_Z_SCORE = "btc_mvrv_z_score"
    BTC_VDD_MULTIPLE = "btc_vdd_multiple"
    GLOBAL_MARKET_CAP = "global_market_cap"
    GLOBAL_VOLUME = "global_volume"
    GTRENDS_ALTCOIN_WORLDWIDE = "gtrends_altcoin_worldwide"
    GTRENDS_CRYPTOCURRENCY_WORLDWIDE = "gtrends_cryptocurrency_worldwide"
    GTRENDS_SOLANA_US = "gtrends_solana_us"
    GTRENDS_CRYPTOCURRENCY_US = "gtrends_cryptocurrency_us"
    GTRENDS_BITCOIN_US = "gtrends_bitcoin_us"
    GTRENDS_SOLANA_WORLDWIDE = "gtrends_solana_worldwide"
    GTRENDS_COINBASE_US = "gtrends_coinbase_us"
    GTRENDS_BITCOIN_WORLDWIDE = "gtrends_bitcoin_worldwide"
    GTRENDS_ETHEREUM_WORLDWIDE = "gtrends_ethereum_worldwide"
    GTRENDS_ETHEREUM_US = "gtrends_ethereum_us"
    GTRENDS_ALTCOIN_US = "gtrends_altcoin_us"
    GTRENDS_COINBASE_WORLDWIDE = "gtrends_coinbase_worldwide"
    GTRENDS_MEMECOIN_WORLDWIDE = "gtrends_memecoin_worldwide"
    GTRENDS_MEMECOIN_US = "gtrends_memecoin_us"




# ____________________________________________________________________________
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Modular Metrics Flattening System
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    """
    The Metric config defines how a time series should be flattened into a single row by
    specifying the columns that will show up in the row, e.g. sum of buyers_new in the sharks
    cohort. These are the three ways that metrics can be generated from time series data:

    * Aggregations are calculations that flatten an entire series, e.g. sum, last, max, etc. The
        Bucket

    * RollingMetrics split the time series into x lookback_periods of y window_duration days. The
        lookback periods can then be flattened into Aggregations or compared with each other
        using Comparisons.

    * Indicator metrics are transformations that result in a new time series, e.g. SMA, EMA, RSI.
        Indicator metrics can be flattened through Aggregations or RollingMetrics.

    Finally, all of these types of metrics can be scaled using ScalingConfig after they've been
    calculated. Scaling is applied to the dataframe containing every coin_id's values for the
    metric and done as part of preprocessing.
    """


class BaseMetric(NoExtrasBaseModel):
    """
    Base metrics template defining the fields and datatypes for metrics at all
    levels of the config
    """
    aggregations: Optional[Dict['AggregationType', 'AggregationConfig']] = Field(default=None)
    rolling: Optional['RollingMetrics'] = Field(default=None)

    @model_validator(mode='after')
    def remove_empty_fields(cls, values):
        """
        Remove all empty dictionaries, None values, or empty lists from the nested structure.
        """
        values_dict = values.model_dump(exclude_none=True)  # Exclude None values
        cleaned_dict = remove_empty_dicts(values_dict)  # Recursively remove empty dictionaries
        return cleaned_dict


class Metric(BaseMetric):
    """
    The Metric config class represents metrics calculated from the dataset base columns.
    They can generate new time series or aggregation fields through indicators or rolling.
    """
    indicators: Optional[Dict['IndicatorType', 'IndicatorMetric']] = Field(default=None)


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
    NONE = "none" # generates no features but allows column to be used for ratios

class AggregationConfig(NoExtrasBaseModel):
    """
    Defines the configuration for each aggregation type.
    An aggregation can have a scaling field or a buckets field.
    """
    scaling: Optional['ScalingType'] = Field(default=None)
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

class RollingMetrics(NoExtrasBaseModel):
    """
    Rolling metrics generates a column for each {lookback_period} that last for
    {window_duration} days which can then be flattened through Aggregations or Comparisons.

    Periods with lower numbers are more recent, i.e. '*_period_1' represents the most recent period.
    """
    window_duration: Annotated[int, Field(gt=0)]  # Must be an integer > 0
    lookback_periods: Annotated[int, Field(gt=0)]  # Must be an integer > 0
    aggregations: Optional[Dict['AggregationType', 'AggregationConfig']] = Field(default=None)
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

class Comparisons(NoExtrasBaseModel):
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
    RSI = "rsi"
    BOLLINGER_BANDS_UPPER = "bollinger_bands_upper"
    BOLLINGER_BANDS_LOWER = "bollinger_bands_lower"

class IndicatorParams(NoExtrasBaseModel):
    """
    This is class that defines all parameters that can be used by all indicators. If a parameter isn't
    applicable to the specific indicator, it will be ignored.
    """
    window: List[int] # used in all indicators
    num_std: Optional[float] = None  # used in bollinger_bands

class IndicatorMetric(BaseMetric):
    """
    This includes all fields from the Metric parent class as well as IndicatorParameters.
    """
    parameters: 'IndicatorParams'


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

class ScalingConfig(NoExtrasBaseModel):
    """
    Configuration for applying scaling to metrics.
    """
    scaling: ScalingType  # Make scaling required for any ComparisonType


# ============================================================================
# Model Rebuilding
# ============================================================================
# Ensures all classes are fully reflected in structure regardless of the order they were defined
MetricsConfig.model_rebuild()
Metric.model_rebuild()
IndicatorMetric.model_rebuild()
