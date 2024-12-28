import logging
from typing import Dict, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# Set up logger at the module level
logger = logging.getLogger(__name__)

# pylint:disable=invalid-name  # X_test isn't camelcase


class CoinModel:
    """
    Simplified model for coin return predictions. Assumes pre-joined feature dataset.
    """

    def __init__(self, wallets_coin_config: Dict):
        """
        Params:
        - wallets_coin_config (dict): wallets_coin_config.yaml
        """
        self.wallets_coin_config = wallets_coin_config
        self.pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.training_data = None

    def _prepare_data(self, feature_df: pd.DataFrame) -> None:
        """
        Params:
        - feature_df (DataFrame): Pre-joined dataframe with features and target
        """
        self.training_data = feature_df.copy()

        # Separate features and target
        target_var = self.wallets_coin_config['coin_modeling']['target_variable']
        X = feature_df.drop(target_var, axis=1)
        y = feature_df[target_var]

        # Split train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def _build_pipeline(self) -> None:
        """Build scikit-learn pipeline with scaling and XGBoost model"""
        # Get columns to drop
        drop_cols = self.wallets_coin_config['coin_modeling']['drop_columns']

        # Define feature columns
        feature_cols = [col for col in self.X_train.columns
                       if col not in (drop_cols or [])]

        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('features', StandardScaler(), feature_cols)
            ],
            remainder='drop'
        )

        # Create model
        model = XGBRegressor(**self.wallets_coin_config['coin_modeling']['model_params'])

        # Build pipeline
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

    def run_experiment(self, feature_df: pd.DataFrame,
                    return_data: bool = True) -> Dict[str, Union[Pipeline, pd.DataFrame, np.ndarray]]:
        """
        Params:
        - feature_df (DataFrame): Pre-joined feature and target data
        - return_data (bool): Whether to return data splits and predictions

        Returns:
        - result (dict): Contains fitted pipeline and optionally train/test data
        """
        self._prepare_data(feature_df)
        self._build_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)

        result = {'pipeline': self.pipeline}

        if return_data:
            # Test predictions
            self.y_pred = pd.Series(
                self.pipeline.predict(self.X_test),
                index=self.X_test.index
            )

            result.update({
                'X_train': self.X_train,
                'X_test': self.X_test,
                'y_train': self.y_train,
                'y_test': self.y_test,
                'y_pred': self.y_pred,
            })

        return result
