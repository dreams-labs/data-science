"""
Validation logic for items in experiments_config.yaml
"""
from enum import Enum
from typing import Optional, Dict
from typing_extensions import Annotated
from pydantic import BaseModel, Field

# Custom base model to disable extra fields in all sections
class NoExtrasBaseModel(BaseModel):
    """Custom BaseModel to apply config settings globally."""
    model_config = {
        "extra": "forbid",  # Prevent extra fields that are not defined
        "str_max_length": 2000,    # Increase the max length of error message string representations
    }

# ============================================================================
#                   experiments_config.yaml Main Configuration
# ============================================================================
class ExperimentsConfig(NoExtrasBaseModel):
    """Top-level structure of the main experiments_config.yaml file."""
    metadata: 'MetadataConfig'
    variable_overrides: Optional['VariableOverrides'] = Field(default=None)


# Metadata Section
# ----------------
class SearchMethod(str, Enum):
    """Enum for target_column values."""
    GRID = "grid"
    RANDOM = "random"


class MetadataConfig(NoExtrasBaseModel):
    """Configuration for the metadata section."""
    experiment_name: str = Field(default="experiment")
    search_method: SearchMethod = Field(default=SearchMethod.RANDOM)
    description: Optional[str] = Field(default=None)
    # metrics_to_compare: List[str]  # List of metric names to compare
    # threshold: Annotated[float, Field(gt=0, lt=1)] = Field(default=0.5)  # Between 0.0 and 1.0
    max_evals: Annotated[int, Field(ge=1)] = Field(default=45)  # Integer 1 or higher


# Variable Overrides Section
# --------------------------
class VariableOverrides(NoExtrasBaseModel):
    """Explicitly allowed overrides in the variable_overrides section."""
    config: Optional[Dict] = Field(default=None)
    modeling_config: Optional[Dict] = Field(default=None)
    metrics_config: Optional[Dict] = Field(default=None)



# ============================================================================
# Model Rebuilding
# ============================================================================
# Ensures all classes are fully reflected in structure regardless of the order they were defined
ExperimentsConfig.model_rebuild()
