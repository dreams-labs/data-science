# pylint:disable=invalid-name  # X_test isn't camelcase
import logging
from typing import Dict, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
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
        self.random_search = None


    def _convert_min_child_pct_to_weight(self, X: pd.DataFrame, base_pct: float = 0.01) -> int:
        """
        Calculate min_child_weight based on dataset size.

        Params:
        - X (DataFrame): feature data to determine size from
        - base_pct (float): baseline percentage for min_child_weight calculation

        Returns:
        - min_child_weight (int): calculated minimum child weight
        """
        n_samples = X.shape[0]
        return max(1, int(n_samples * base_pct))

    def _convert_min_child_weight_to_pct(self, X: pd.DataFrame, min_child_weight: int) -> float:
        """
        Convert min_child_weight back to percentage based on dataset size.

        Params:
        - X (DataFrame): feature data used for size calculation
        - min_child_weight (int): minimum child weight value

        Returns:
        - pct (float): corresponding percentage of dataset size
        """
        n_samples = X.shape[0]
        return min_child_weight / n_samples



    def _build_pipeline(self) -> None:
        """
        Build basic XGBoost pipeline. Override for custom preprocessing.
        """
        model_params = self.modeling_config['model_params'].copy()

        # Update min_child_weight if percentage is specified
        if model_params.get('min_child_weight_pct'):
            model_params['min_child_weight'] = self._convert_min_child_pct_to_weight(
                self.X_train,
                model_params.pop('min_child_weight_pct')
            )

        model = XGBRegressor(**model_params)
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

        # Get base model params and handle child weight percentages
        cv_model_params = self.modeling_config['model_params'].copy()
        if cv_model_params.get('min_child_weight_pct'):
            cv_model_params['min_child_weight'] = self._convert_min_child_pct_to_weight(
                self.X_train,
                cv_model_params.pop('min_child_weight_pct')
            )

        # Get grid search params and handle child weight percentages
        grid_search_params = self.modeling_config['grid_search_params']
        if 'regressor__min_child_weight_pct' in grid_search_params['param_grid']:
            pct_grid = grid_search_params['param_grid'].pop('regressor__min_child_weight_pct')
            grid_search_params['param_grid']['regressor__min_child_weight'] = [
                self._convert_min_child_pct_to_weight(X, pct) for pct in pct_grid
            ]

        # Remove params that don't apply to the model
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

        self.random_search.fit(X, y)

        return {
            'best_params': {
                k: (self._convert_min_child_weight_to_pct(X, v) if k == 'regressor__min_child_weight' else v)
                for k, v in self.random_search.best_params_.items()
            },
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
