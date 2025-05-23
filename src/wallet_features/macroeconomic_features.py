import logging
from datetime import timedelta
import pandas as pd

# Local module imports
import feature_engineering.flattening as flt
import utils as u

# set up logger at the module level
logger = logging.getLogger(__name__)



# --------------------------------------
#        Features Main Interface
# --------------------------------------

@u.timing_decorator
def calculate_macro_features(
        training_macro_indicators_df: pd.DataFrame,
        df_metrics_config: dict
    ) -> pd.DataFrame:
    """
    Generates a single row containing all variables specified in the df_metrics_config. The config
    file structure is defined and documented in src/config_models/metrics_config.py.
    """
    # Calculate all specified metrics
    macro_features_dict = flt.flatten_date_features(training_macro_indicators_df,df_metrics_config)

    # Convert to DataFrame
    macro_features_df = pd.DataFrame([macro_features_dict])

    return macro_features_df
