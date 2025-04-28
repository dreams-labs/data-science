import logging
from typing import Dict, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline


# Local module imports
from wallet_modeling.wallet_model import BaseModel

# pylint:disable=invalid-name  # X_test isn't camelcase
# pylint: disable=W0201  # Attribute defined outside __init__, false positive due to inheritance

# Set up logger at the module level
logger = logging.getLogger(__name__)


class CoinModel(BaseModel):
    """
    Coin-specific model implementation.
    Extends BaseModel with coin-specific data preparation.
    """

    def __init__(self, modeling_config: dict):  # pylint:disable=useless-parent-delegation
        """
        Initialize WalletModel with configuration and wallet features DataFrame.

        Params:
        - modeling_config (dict): Configuration dictionary for modeling parameters.
        """
        # Initialize BaseModel with the given configuration
        super().__init__(modeling_config)


    def _prepare_data(self, feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Params:
        - feature_df (DataFrame): Pre-joined dataframe with features and target

        Returns:
        - X (DataFrame): feature data for modeling
        - y (Series): target variable for modeling
        """
        # Store full dataset
        self.training_data_df = feature_df.copy()

        # Separate target variable
        target_var = self.modeling_config['target_variable']
        X = feature_df.drop([target_var], axis=1)
        y = feature_df[target_var]

        # Let BaseModel handle the splits
        self._split_data(X, y)

        return X, y


    def construct_coin_model(self, feature_df: pd.DataFrame,
                          return_data: bool = True) -> Dict[str, Union[Pipeline, pd.DataFrame, np.ndarray]]:
        """
        Run coin-specific modeling experiment.

        Params:
        - feature_df (DataFrame): Pre-joined feature and target data
        - return_data (bool): Whether to return train/test splits and predictions

        Returns:
        - result (dict): Contains fitted pipeline and optionally train/test data
        """
        self._prepare_data(feature_df)
        return super().construct_base_model(return_data)
