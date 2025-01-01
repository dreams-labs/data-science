# pylint:disable=invalid-name  # X_test isn't camelcase
import logging
from typing import Dict, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
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


    def _prepare_data(self, training_data_df: pd.DataFrame, modeling_cohort_target_var_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Params:
        - training_data_df (DataFrame): full training cohort feature data
        - modeling_cohort_target_var_df (DataFrame): Contains in_modeling_cohort flag and target variable
                                                for full training cohort

        Returns:
        - X (DataFrame): feature data for modeling cohort
        - y (Series): target variable for modeling cohort
        """
        # Store full training cohort for later scoring
        self.training_data_df = training_data_df.copy()

        # Validate indexes are wallet addresses and matching
        if not (training_data_df.index.name == 'wallet_address' and
                modeling_cohort_target_var_df.index.name == 'wallet_address'):
            raise ValueError("Both dataframes must have wallet_address as index")

        # Join target data to features
        modeling_df = training_data_df.join(
            modeling_cohort_target_var_df,
            how='left'
        )

        # Filter to modeling cohort for training
        modeling_cohort_mask = modeling_df['in_modeling_cohort'] == 1
        modeling_df = modeling_df[modeling_cohort_mask]

        # Separate target variable
        target_var = self.wallets_config['modeling']['target_variable']
        X = modeling_df.drop([target_var, 'in_modeling_cohort'], axis=1)
        y = modeling_df[target_var]

        # Create train/test split for final model
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return X, y


    def _build_pipeline(self) -> None:
        """
        Build the pipeline with column dropping, numeric scaling, and model.
        """

        # Define the model
        if self.wallets_config['modeling']['model_type'] == 'xgb':
            model = XGBRegressor(**self.wallets_config['modeling']['model_params'])
        else:
            raise ValueError("Invalid model type found in wallets_config['modeling']['model_type'].")

        # Create pipeline
        self.pipeline = Pipeline([
            ('regressor', model)
        ])


    def _run_grid_search(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Perform grid search with cross validation using configured parameters.

        Params:
        - X (DataFrame): feature data for modeling cohort
        - y (Series): target variable for modeling cohort

        Returns:
        - cv_results (dict): Mean and std of CV scores
        """
        if not self.wallets_config['modeling'].get('grid_search_params', {}).get('enabled'):
            return {}

        # Get CV parameters from config
        grid_search_params = self.wallets_config['modeling']['grid_search_params']
        n_splits = grid_search_params.get('n_splits', 5)

        param_grid = grid_search_params['param_grid']

        # Create base model without early stopping params
        cv_model_params = self.wallets_config['modeling']['model_params'].copy()
        cv_model_params.pop('early_stopping_rounds', None)
        cv_model_params.pop('eval_metric', None)
        cv_model_params.pop('verbose', None)

        # Create pipeline for grid search
        cv_pipeline = Pipeline([
            ('regressor', XGBRegressor(**cv_model_params))
        ])

        # Perform grid search
        grid_search = GridSearchCV(
            cv_pipeline,
            param_grid,
            cv=n_splits,
            scoring='neg_root_mean_squared_error',
            n_jobs=cv_model_params.get('n_jobs', -1),
            verbose=grid_search_params.get('verbose_level', 0)
        )

        grid_search.fit(X, y)

        return {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,  # Convert back to positive RMSE
            'cv_results': grid_search.cv_results_
        }

    def _fit(self) -> None:
        """
        Fit the pipeline on training data with early stopping using test set.
        """
        # Create eval set for monitoring validation performance
        eval_set = [(self.X_test, self.y_test)]

        # Pass eval_set to XGBoost through the pipeline
        # The regressor__ prefix routes the parameter to the XGBoost step
        self.pipeline.fit(
            self.X_train,
            self.y_train,
            regressor__eval_set=eval_set
        )


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
        - actuals (Series): actual target values for full training cohort
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


    def run_experiment(self, training_data_df: pd.DataFrame, modeling_cohort_target_var_df: pd.DataFrame,
                    return_data: bool = True) -> Dict[str, Union[Pipeline, pd.DataFrame, np.ndarray]]:
        """
        Params:
        - training_data_df (DataFrame): full training cohort feature data
        - modeling_cohort_target_var_df (DataFrame): Contains in_modeling_cohort flag and target variable
        - return_data (bool): whether to return train/test splits and predictions

        Returns:
        - result (dict): contains pipeline, cv results if enabled, and optionally data splits and predictions
        """
        # Validate matching indexes and lengths
        if not training_data_df.index.equals(modeling_cohort_target_var_df.index):
            raise ValueError(
                "training_data_df and modeling_cohort_target_var_df must have identical indexes. "
                f"Found lengths {len(training_data_df)} and {len(modeling_cohort_target_var_df)}"
            )

        # Store data for CV and final training
        self.training_data_df = training_data_df.copy()

        # Prepare data
        X, y = self._prepare_data(training_data_df, modeling_cohort_target_var_df)

        # Run cross-validation if enabled and get best params
        cv_results = self._run_grid_search(X, y)

        # Update model params with best params from CV if available
        if cv_results.get('best_params'):
            # Remove 'regressor__' prefix from param names
            best_params = {
                k.replace('regressor__', ''): v
                for k, v in cv_results['best_params'].items()
            }
            self.wallets_config['modeling']['model_params'].update(best_params)
            logger.info(f"Updated model params with CV best params: {best_params}")

        # Build and fit final model
        self._build_pipeline()
        self._fit()

        # Build results dict
        result = {
            'pipeline': self.pipeline,
            'cv_results': cv_results
        }

        # Optionally return detailed data
        if return_data:
            y_pred = self._predict()
            training_cohort_pred = self._predict_training_cohort()
            target_var = self.wallets_config['modeling']['target_variable']
            full_cohort_actuals = modeling_cohort_target_var_df[target_var]

            result.update({
                'X_train': self.X_train,
                'X_test': self.X_test,
                'y_train': self.y_train,
                'y_test': self.y_test,
                'y_pred': y_pred,
                'training_cohort_pred': training_cohort_pred,
                'training_cohort_actuals': full_cohort_actuals
            })

        return result
