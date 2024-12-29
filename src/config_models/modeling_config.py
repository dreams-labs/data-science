"""
Validation logic for items in modeling_config.yaml
"""
from enum import Enum
from typing import Optional, List, Dict
from typing_extensions import Annotated
from pydantic import BaseModel, Field


# Custom base model to disable extra fields in all sections
class NoExtrasBaseModel(BaseModel):
    """Custom BaseModel to apply config settings globally."""
    model_config = {
        "extra": "forbid",  # Prevent extra fields that are not defined
        "str_max_length": 2000,    # Increase the max length of error message string representations
        "protected_namespaces": ()  # Ignores protected spaces for params starting with  "model_"
    }

# ============================================================================
#                   modeling_config.yaml Main Configuration
# ============================================================================

class ModelingConfig(NoExtrasBaseModel):
    """Top-level structure of the main modeling_config.yaml file."""
    preprocessing: 'PreprocessingConfig' = Field(...)
    target_variables: 'TargetVariablesConfig'
    modeling: 'ModelingSettings'  # Modeling section is now its own object
    evaluation: 'EvaluationConfig' = Field(...)


# Preprocessing section
# ---------------------

class FillMethod(str, Enum):
    """
    These dicatates what to do if the dataset doesn't have rows for every coin_id in other
    datasets.
    """
    RETAIN_NULLS = "retain_nulls"               # any missing rows are left unchanged
    FILL_ZEROS = "fill_zeros"                   # any missing rows are filled with 0
    DROP_RECORDS = "drop_records"               # any missing rows are dropped from the training set
    EXTEND_COIN_IDS = "extend_coin_ids"         # used for macro series; copies time_window features to all coins
    EXTEND_TIME_WINDOWS = "extend_time_windows" # used for metadata series; copies coin features to all time windows

class PreprocessingConfig(NoExtrasBaseModel):
    """
    Configuration for preprocessing steps.
    """
    drop_features: Optional[List[str]] = Field(default=None)
    fill_methods: Dict[str, 'FillMethod'] = Field(...)
    data_partitioning: 'DataPartitioning'

class DataPartitioning(NoExtrasBaseModel):
    """
    Defines the train/test/validation/future split shares
    """
    test_set_share: Annotated[float, Field(gt=0, lt=1)] = Field(...)
    validation_set_share: Annotated[float, Field(ge=0, lt=1)] = Field(...)
    future_set_time_windows: Annotated[int, Field(ge=0)] = Field(...)


# Target Variables section
# ------------------------

class TargetVariablesConfig(BaseModel):
    """
    Configuration for target variables.
    """
    moon_threshold: Optional[Annotated[float, Field(ge=0)]] = Field(default=0.3)
    moon_minimum_percent: Optional[Annotated[float, Field(ge=0, le=1)]] = Field(default=0.1)
    crater_threshold: Optional[Annotated[float, Field(le=0)]] = Field(default=-0.3)
    crater_minimum_percent: Optional[Annotated[float, Field(ge=0, le=1)]] = Field(default=0.1)


# Modeling section
# ----------------

class ModelType(str, Enum):
    """The model to train"""
    RANDOMFORESTCLASSIFIER = "RandomForestClassifier"
    RANDOMFORESTREGRESSOR = "RandomForestRegressor"
    GRADIENTBOOSTINGREGRESSOR = "GradientBoostingRegressor"
    GRADIENTBOOSTINGCLASSIFIER = "GradientBoostingClassifier"


class TargetColumn(str, Enum):
    """Enum for target_column values."""
    IS_MOON = "is_moon"
    IS_CRATER = "is_crater"
    RETURNS = "returns"


class ModelingSettings(NoExtrasBaseModel):
    """Configuration settings related to modeling."""
    target_column: TargetColumn = Field(default=TargetColumn.IS_MOON)
    modeling_folder: str = Field(default="../modeling")
    config_folder: str = Field(default="../config")
    random_seed: Annotated[int, Field(ge=0)] = Field(default=45)
    model_type: ModelType
    model_params: Optional['ModelParams'] = None

class ModelParams(BaseModel):
    """Parameters for the model."""
    n_estimators: Optional[Annotated[int, Field(ge=1)]] = Field(default=None)


# Evaluation section
# ---------------------

class EvaluationMetric(str, Enum):
    """
    Evaluation Metrics
    """
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    LOG_LOSS = "log_loss"
    CONFUSION_MATRIX = "confusion_matrix"
    PROFITABILITY_AUC = "profitability_auc"
    DOWNSIDE_PROFITABILITY_AUC = "downside_profitability_auc"
    MSE = "mse"
    RMSE = "rmse"
    MAE = "mae"
    R2 = "r2"
    EXPLAINED_VARIANCE = "explained_variance"
    MAX_ERROR = "max_error"


class EvaluationConfig(NoExtrasBaseModel):
    """Configuration for model evaluation step."""
    metrics: Dict[EvaluationMetric, Optional[dict]]
    winsorization_cutoff: Optional[float]

# ============================================================================
# Model Rebuilding
# ============================================================================
# Ensures all classes are fully reflected in structure regardless of the order they were defined
ModelingConfig.model_rebuild()
