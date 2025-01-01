import logging
from typing import Dict, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

        # Drop specified columns and separate target
        target_var = self.modeling_config['target_variable']
        drop_cols = self.modeling_config.get('drop_columns', [])

        X = feature_df.drop([target_var] + drop_cols, axis=1)
        y = feature_df[target_var]

        # Create train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.modeling_config['train_test_split'],
            random_state=self.modeling_config['model_params']['random_state']
        )

        return X, y

    def run_coin_experiment(self, feature_df: pd.DataFrame,
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
        return super().run_base_experiment(return_data)
