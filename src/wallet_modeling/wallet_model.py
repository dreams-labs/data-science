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
        self.training_data_df = None  # Store full training cohort dataset for later scoring


    def _prepare_data(self, training_data_df: pd.DataFrame, modeling_cohort_target_var_df: pd.Series) -> None:
        """
        Params:
        - training_data_df (DataFrame): full training cohort feature data
        - modeling_cohort_target_var_df (Series): target variable for modeling cohort only

        Returns:
        - None
        """
        # Store full training cohort for later scoring
        self.training_data_df = training_data_df.copy()

        # Create modeling_df by joining features to modeling cohort targets
        modeling_df = training_data_df.join(modeling_cohort_target_var_df, how='inner')

        # Separate target variable
        target_var = self.wallets_config['modeling']['target_variable']
        X = modeling_df.drop(target_var, axis=1)
        y = modeling_df[target_var]

        # Split into train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )


    def _build_pipeline(self) -> None:
        """
        Build the pipeline with column dropping, numeric scaling, and model.
        """
        # Configure column dropping
        drop_cols = self.wallets_config['modeling']['drop_columns']

        # Get feature columns (all columns except those to be dropped)
        feature_cols = [col for col in self.X_train.columns
                    if col not in (drop_cols or [])]

        # Create preprocessor with two steps:
        # 1. Drop unwanted columns
        # 2. Scale remaining features
        preprocessor = ColumnTransformer(
            transformers=[
                ('features', StandardScaler(), feature_cols)
            ],
            remainder='drop'  # This will drop any columns not explicitly included
        )

        # Define the model
        if self.wallets_config['modeling']['model_type'] == 'xgb':
            model = XGBRegressor(**self.wallets_config['modeling']['model_params'])
        else:
            raise ValueError("Invalid model type found in wallets_config['modeling']['model_type'].")

        # Create pipeline
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

    def _fit(self) -> None:
        """
        Fit the pipeline on training data.
        """
        # Train pipeline
        self.pipeline.fit(self.X_train, self.y_train)

    def _predict(self) -> pd.Series:
        """
        Make predictions on the test set.

        Returns:
        - predictions (Series): predicted values for the test set, indexed like y_test
        """
        # Get raw predictions from pipeline
        raw_predictions = self.pipeline.predict(self.X_test)

        # Convert to Series with same index as test data
        self.y_pred = pd.Series(raw_predictions, index=self.X_test.index)
        return self.y_pred

    def _predict_training_cohort(self) -> pd.Series:
        """
        Make predictions on the full training cohort.

        Returns:
        - predictions (Series): predicted values for full training cohort
        """
        if self.training_data_df is None:
            raise ValueError("No training cohort data found. Run prepare_data first.")

        # Validate indexes are wallet addresses and matching
        if not self.training_data_df.index.name == 'wallet_address':
            raise ValueError("training_data_df index must be wallet_address")

        predictions = pd.Series(
            self.pipeline.predict(self.training_data_df),
            index=self.training_data_df.index
        )

        # Verify no wallets were dropped during prediction
        if not predictions.index.equals(self.training_data_df.index):
            raise ValueError("Prediction dropped some wallet addresses")

        return predictions

    def run_experiment(self, training_data_df: pd.DataFrame, modeling_cohort_target_var_df: pd.Series,
                      return_data: bool = True) -> Dict[str, Union[Pipeline, pd.DataFrame, np.ndarray]]:
        """
        Params:
        - training_data_df (DataFrame): full training cohort feature data
        - modeling_cohort_target_var_df (Series): target variable for modeling cohort only
        - return_data (bool): whether to return train/test splits and predictions

        Returns:
        - result (dict): contains pipeline and optionally data splits and predictions
        """
        # Run full experiment: prep data, build pipeline, fit model
        self._prepare_data(training_data_df, modeling_cohort_target_var_df)
        self._build_pipeline()
        self._fit()

        # Always return the pipeline
        result = {'pipeline': self.pipeline}

        # Optionally return detailed data
        if return_data:
            y_pred = self._predict()
            training_cohort_pred = self._predict_training_cohort()
            result.update({
                'X_train': self.X_train,
                'X_test': self.X_test,
                'y_train': self.y_train,
                'y_test': self.y_test,
                'y_pred': y_pred,
                'training_cohort_pred': training_cohort_pred
            })

        return result
