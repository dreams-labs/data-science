"""
Calculates metrics aggregated at the wallet level
"""
# pylint:disable=invalid-name  # X_test isn't camelcase
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


class WalletModel:
    """
    A class for running wallet model experiments with a single model.
    Encapsulates data preparation, training, prediction, and result management.
    """

    def __init__(self, wallets_config):
        """
        Params:
        - wallets_config (dict): configuration dictionary for modeling parameters.
        """
        self.wallets_config = wallets_config
        self.pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    def prepare_data(self, modeling_df: pd.DataFrame) -> None:
        """
        Params:
        - modeling_df (DataFrame): full modeling DataFrame including features and target.

        Returns:
        - None
        """
        # Make a copy to avoid mutating the original DataFrame
        df = modeling_df.copy()

        # Drop configured columns if they exist
        drop_cols = self.wallets_config['modeling']['drop_columns']
        if drop_cols:
            existing_columns = [c for c in drop_cols if c in df.columns]
            if existing_columns:
                df = df.drop(columns=existing_columns)

        # Separate target variable
        target_var = self.wallets_config['modeling']['target_variable']
        X = df.drop(target_var, axis=1)
        y = df[target_var]

        # Split into train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def build_pipeline(self) -> None:
        """
        Build the pipeline with a numeric scaler and a model.
        """
        # Use all columns as numeric for now
        numeric_features = self.X_train.columns.tolist()

        # Simple numeric preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features)
            ]
        )

        # Define the model
        if self.wallets_config['modeling']['model_type']=='xgb':
            model = XGBRegressor(**self.wallets_config['modeling']['model_params'])
        else:
            raise ValueError("Invalid model type found in wallets_config['modeling']['model_type'].")

        # Create pipeline
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

    def fit(self) -> None:
        """
        Fit the pipeline on training data.
        """
        # Train pipeline
        self.pipeline.fit(self.X_train, self.y_train)

    def predict(self) -> np.ndarray:
        """
        Make predictions on the test set.

        Returns:
        - predictions (ndarray): predicted values for the test set.
        """
        # Store predictions for later use
        self.y_pred = self.pipeline.predict(self.X_test)
        return self.y_pred

    def run_experiment(self, modeling_df: pd.DataFrame, return_data: bool = True
                       ) -> Dict[str, Union[Pipeline, pd.DataFrame, np.ndarray]]:
        """
        Params:
        - modeling_df (DataFrame): input modeling data.
        - return_data (bool): whether to return train/test splits and predictions.

        Returns:
        - result (dict): contains pipeline and optionally data splits and predictions.
        """
        # Run full experiment: prep data, build pipeline, fit model
        self.prepare_data(modeling_df)
        self.build_pipeline()
        self.fit()

        # Always return the pipeline
        result = {'pipeline': self.pipeline}

        # Optionally return detailed data
        if return_data:
            y_pred = self.predict()
            result.update({
                'X_train': self.X_train,
                'X_test': self.X_test,
                'y_train': self.y_train,
                'y_test': self.y_test,
                'y_pred': y_pred
            })

        return result
