# pylint:disable=invalid-name  # X_test isn't camelcase
import logging
from typing import Dict, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# Set up logger at the module level
logger = logging.getLogger(__name__)


class BaseModel:
    """
    Base class for XGBoost-based prediction models.
    Handles core pipeline functionality, data preparation, and model training.
    """

    def __init__(self, modeling_config):
        """
        Params:
        - wallets_config (dict): configuration dictionary for modeling parameters.
        """
        self.modeling_config = modeling_config
        self.pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.training_data_df = None

        # Grid Search Objects
        self.random_search = None
        self.cv_results_ = None
        self.best_params_ = None
        self.best_score_ = None


    def _build_pipeline(self) -> None:
        """
        Build basic XGBoost pipeline. Override for custom preprocessing.
        """
        model = XGBRegressor(**self.modeling_config['model_params'])
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
        if not self.modeling_config.get('grid_search_params', {}).get('enabled'):
            return {}

        # Get CV parameters from config
        grid_search_params = self.modeling_config['grid_search_params']

        # Copy model params but remove the ones that don't apply to skl cross validation
        cv_model_params = self.modeling_config['model_params'].copy()
        for param in ['early_stopping_rounds', 'eval_metric', 'verbose']:
            cv_model_params.pop(param, None)

        # Create pipeline for grid search
        cv_pipeline = Pipeline([
            ('regressor', XGBRegressor(**cv_model_params))
        ])

        # Store the search object in the instance
        self.random_search = RandomizedSearchCV(
            cv_pipeline,
            grid_search_params['param_grid'],
            n_iter = grid_search_params['n_iter'],
            cv = grid_search_params['n_splits'],
            scoring = grid_search_params['scoring'],
            verbose = grid_search_params.get('verbose_level', 0),
            n_jobs = cv_model_params.get('n_jobs', -1),
            random_state = cv_model_params.get('random_state', 42),
        )

        # Store intermediate results in the instance
        self.cv_results_ = []
        self.best_params_ = None
        self.best_score_ = float('inf')

        try:
            self.random_search.fit(X, y)
            # Store final results
            self.cv_results_ = self.random_search.cv_results_
            self.best_params_ = self.random_search.best_params_
            self.best_score_ = -self.random_search.best_score_
        except KeyboardInterrupt:
            logger.info("Grid search interrupted. Saving partial results...")
            if hasattr(self.random_search, 'cv_results_'):
                self.cv_results_ = self.random_search.cv_results_
                self.best_params_ = self.random_search.best_params_
                self.best_score_ = -self.random_search.best_score_


        self.random_search.fit(X, y)

        return {
            'best_params': self.random_search.best_params_,
            'best_score': -self.random_search.best_score_,
            'cv_results': self.random_search.cv_results_
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


    def run_base_experiment(self, return_data: bool = True) -> Dict[str, Union[Pipeline, pd.DataFrame, np.ndarray]]:
        """
        Core experiment runner with parameter tuning and model fitting.

        Params:
        - return_data (bool): Whether to return train/test splits and predictions

        Returns:
        - result (dict): Contains fitted pipeline and optionally train/test data
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data must be prepared before running experiment")

        cv_results = self._run_grid_search(self.X_train, self.y_train)

        if cv_results.get('best_params'):
            best_params = {
                k.replace('regressor__', ''): v
                for k, v in cv_results['best_params'].items()
            }
            self.modeling_config['model_params'].update(best_params)
            logger.info(f"Updated model params with CV best params: {best_params}")

        self._build_pipeline()
        self._fit()

        result = {
            'pipeline': self.pipeline,
            'cv_results': cv_results
        }

        if return_data:
            self.y_pred = self._predict()  # Store prediction in instance
            result.update({
                'X_train': self.X_train,
                'X_test': self.X_test,
                'y_train': self.y_train,
                'y_test': self.y_test,
                'y_pred': self.y_pred,
            })

        return result


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


class WalletModel(BaseModel):
    """
    Wallet-specific model implementation.
    Extends BaseModel with wallet-specific data preparation and grid search.
    """

    def _prepare_data(self, training_data_df: pd.DataFrame,
                      modeling_cohort_target_var_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
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
        target_var = self.modeling_config['target_variable']
        X = modeling_df.drop([target_var, 'in_modeling_cohort'], axis=1)
        y = modeling_df[target_var]

        # Create train/test split for final model
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.modeling_config['train_test_split'],
            random_state=self.modeling_config['model_params']['random_state']
        )

        return X, y


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


    def run_wallet_experiment(self, training_data_df: pd.DataFrame,
                            modeling_cohort_target_var_df: pd.DataFrame,
                            return_data: bool = True) -> Dict[str, Union[Pipeline, pd.DataFrame, np.ndarray]]:
        """
        Run wallet-specific modeling experiment.

        Params:
        - training_data_df (DataFrame): full training cohort feature data
        - modeling_cohort_target_var_df (DataFrame): Contains modeling cohort flag and target
        - return_data (bool): Whether to return train/test splits and predictions

        Returns:
        - result (dict): Contains fitted pipeline, predictions, and optional train/test data
        """
        # Validate indexes match
        if not training_data_df.index.equals(modeling_cohort_target_var_df.index):
            raise ValueError(
                "training_data_df and modeling_cohort_target_var_df must have identical indexes. "
                f"Found lengths {len(training_data_df)} and {len(modeling_cohort_target_var_df)}"
            )

        # Run base experiment
        self._prepare_data(training_data_df, modeling_cohort_target_var_df)
        result = super().run_base_experiment(return_data)

        # Add wallet-specific predictions if requested
        if return_data:
            training_cohort_pred = self._predict_training_cohort()
            target_var = self.modeling_config['target_variable']
            full_cohort_actuals = modeling_cohort_target_var_df[target_var]

            result.update({
                'training_cohort_pred': training_cohort_pred,
                'training_cohort_actuals': full_cohort_actuals
            })

        return result
