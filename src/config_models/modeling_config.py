"""
Validation logic for items in modeling_config.yaml
"""
from enum import Enum
from typing import Optional, List
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
    preprocessing: Optional['PreprocessingConfig'] = Field(default=None)
    target_variables: 'TargetVariablesConfig'
    modeling: 'ModelingSettings'  # Modeling section is now its own object


# Preprocessing section
# ---------------------
class PreprocessingConfig(NoExtrasBaseModel):
    """Configuration for preprocessing step."""
    drop_features: Optional[List[str]] = Field(default=None)


# Target Variables section
# ------------------------
class TargetVariablesConfig(NoExtrasBaseModel):
    """Configuration for target variables."""
    moon_threshold: Annotated[float, Field(ge=0, le=1)] = Field(default=0.3)
    moon_minimum_percent: Annotated[float, Field(ge=0, le=1)] = Field(default=0.1)
    crater_threshold: Annotated[float, Field(ge=-1, le=0)] = Field(default=-0.3)
    crater_minimum_percent: Annotated[float, Field(ge=0, le=1)] = Field(default=0.1)


# Modeling section
# ----------------
class TargetColumn(str, Enum):
    """Enum for target_column values."""
    IS_MOON = "is_moon"
    IS_CRATER = "is_crater"


class ModelingSettings(NoExtrasBaseModel):
    """Configuration settings related to modeling."""
    target_column: TargetColumn = Field(default=TargetColumn.IS_MOON)
    modeling_folder: str = Field(default="../modeling")
    config_folder: str = Field(default="../config")
    train_test_split: Annotated[float, Field(gt=0, lt=1)] = Field(default=0.25)
    random_state: Annotated[int, Field(ge=0)] = Field(default=45)
    model_params: 'ModelParams'

class ModelParams(NoExtrasBaseModel):
    """Parameters for the model."""
    n_estimators: Annotated[int, Field(ge=1)] = Field(default=100)
    random_state: Annotated[int, Field(ge=0)] = Field(default=45)


# ============================================================================
# Model Rebuilding
# ============================================================================
# Ensures all classes are fully reflected in structure regardless of the order they were defined
ModelingConfig.model_rebuild()
