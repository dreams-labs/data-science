import time
import logging
from typing import Dict, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor

# Local modules
import base_modeling.feature_selection as fs
import utils as u


# pylint:disable=invalid-name  # X_test isn't camelcase

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
        # Key Params
        self.modeling_config = modeling_config
        self.training_data_df = None

        # Pipeline Steps
        self.pipeline = None
        self.columns_to_drop = None
        self.random_search = None

        # Model Datasets
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

        # Utils
        self.start_time = time.time()
        self.training_time = None



    # -----------------------------------
    #     Primary Modeling Interface
    # -----------------------------------

    def construct_base_model(self, return_data: bool = True) -> Dict[str, Union[Pipeline, pd.DataFrame, np.ndarray]]:
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

            # Return the search results without building a model if configured to
            if not self.modeling_config.get('grid_search_params', {}).get('build_post_search_model'):
                return cv_results

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



    # -----------------------------------
    #      Modeling Helper Methods
    # -----------------------------------

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

        # Pipeline Begins
        self.pipeline = Pipeline([
            ('drop_columns', DropColumnPatterns(
                drop_patterns=self.modeling_config['feature_selection']['drop_patterns']
            )),
            ('regressor', XGBRegressor(**model_params))
        ])


    def _fit(self) -> None:
        """
        Fit the pipeline on training data with early stopping using test set.
        """
        # Get pipeline steps
        transformer = self.pipeline[:-1]
        regressor = self.pipeline[-1]

        # Fit transformer and transform both train and test
        X_train_transformed = transformer.fit_transform(self.X_train)
        X_test_transformed = transformer.transform(self.X_test)

        # Create eval set with transformed data
        eval_set = [(X_train_transformed, self.y_train),
                    (X_test_transformed, self.y_test)]

        # Fit final regressor with transformed data
        logger.info(f"Training model using data with shape: {X_train_transformed.shape}...")
        regressor.fit(
            X_train_transformed,
            self.y_train,
            eval_set=eval_set
        )

        self.training_time = time.time() - self.start_time
        logger.info("Training completed after %.2f seconds.", self.training_time)


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




    # -----------------------------------
    #         Grid Search Methods
    # -----------------------------------

    def _run_grid_search(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Perform grid search with cross validation using configured parameters.

        Params:
        - X (DataFrame): feature data for modeling cohort
        - y (Series): target variable for modeling cohort

        Returns:
        - cv_results (dict): Mean and std of CV scores
        """
        # If grid search is disabled, return nothing
        if not self.modeling_config.get('grid_search_params', {}).get('enabled'):
            logger.info("Constructing production model with base params...")
            return {}
        logger.info("Initiating grid search...")
        u.notify('gadget')

        # 1. Retrieve base params
        # -----------------------
        cv_model_params = self.modeling_config['model_params'].copy()

        # Handle 'min_child_weight_pct' to 'min_child_weight' conversion
        if cv_model_params.get('min_child_weight_pct'):
            cv_model_params['min_child_weight'] = self._convert_min_child_pct_to_weight(
                self.X_train,
                cv_model_params.pop('min_child_weight_pct')
            )

        # Store base column drop patterns
        base_drop_patterns = self.modeling_config['feature_selection']['drop_patterns']


        # 2. Prepare grid search params
        # -----------------------------
        grid_search_params = self.modeling_config['grid_search_params']

        # Handle 'min_child_weight_pct' to 'min_child_weight' conversion
        if 'regressor__min_child_weight_pct' in grid_search_params['param_grid']:
            pct_grid = grid_search_params['param_grid'].pop('regressor__min_child_weight_pct')
            grid_search_params['param_grid']['regressor__min_child_weight'] = [
                self._convert_min_child_pct_to_weight(X, pct) for pct in pct_grid
            ]

        # Add base drop patterns in addition to the grid search params
        grid_search_params['param_grid']['drop_columns__drop_patterns'] = [
            base_drop_patterns + grid_pattern
            for grid_pattern in grid_search_params['param_grid']['drop_columns__drop_patterns']
        ]

        # Remove params that don't apply to the model
        for param in ['early_stopping_rounds', 'eval_metric', 'verbose']:
            cv_model_params.pop(param, None)


        # 3. Search
        # ---------
        # Create pipeline for grid search
        cv_pipeline = Pipeline([
            ('drop_columns', DropColumnPatterns()),
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

        # Log best results
        logger.info("Grid search complete. Best score: %f",
                    -self.random_search.best_score_)
        u.notify('synth_magic')

        return {
            'best_params': {
                k: (self._convert_min_child_weight_to_pct(X, v) if k == 'regressor__min_child_weight' else v)
                for k, v in self.random_search.best_params_.items()
            },
            'best_score': -self.random_search.best_score_,
            'cv_results': self.random_search.cv_results_
        }

    def generate_search_report(self, output_raw_data=False) -> pd.DataFrame:
        """
        Generate a report of the random search results.

        Returns:
        - report_df (DataFrame): A DataFrame with columns for 'param', 'param_value',
                                'avg_score', and 'total_builds'.
        """
        if not self.random_search:
            logger.error("Random search has not been run.")
            return None
        elif not hasattr(self.random_search, 'cv_results_'):
            logger.error("cv_results_ is unavailable.")
            return None

        # Extract cv_results from the random search
        cv_results = self.random_search.cv_results_

        # Convert cv_results to a DataFrame
        results_df = pd.DataFrame(cv_results)

        if output_raw_data is True:
            return results_df

        # Prepare a list to hold rows for the report
        report_data = []

        # Loop through each parameter in the param grid
        for param in [col for col in results_df.columns if col.startswith("param_")]:
            param_name = param.replace("param_", "")
            for _, row in results_df.iterrows():
                report_data.append({
                    'param': param_name,
                    'param_value': str(row[param]),
                    'avg_score': row['mean_test_score'],
                    'total_builds': len(results_df)
                })

        # Create a DataFrame for the report
        report_df = pd.DataFrame(report_data)

        return (report_df
                    .groupby(['param','param_value'])[['avg_score']]
                    .mean('avg_score')
                    .sort_values(by='avg_score', ascending=False)
                )




    # -----------------------------------
    #           Utility Methods
    # -----------------------------------

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




# -----------------------------------
#           Pipeline Steps
# -----------------------------------

class DropColumnPatterns(BaseEstimator, TransformerMixin):
    """
    Pipeline step that drops columns based on the patterns provided in the config, including
    support for * wildcards.

    Valid format example: 'training_clusters|k2_cluster/*|trading/*'
    """
    def __init__(self, drop_patterns=None):
        """
        Transformer for dropping columns based on patterns.

        Params:
        - drop_patterns (list): List of patterns to match columns for dropping.
        """
        self.drop_patterns = drop_patterns
        self.columns_to_drop = None  # Persist calculated columns to drop

    # pylint:disable=unused-argument  # y param is needed to match pipeline structure
    def fit(self, X, y=None):
        """
        Identify columns to drop based on the given patterns.

        Params:
        - X (DataFrame): Input training data.
        - y (Series): Target variable (ignored but required for pipeline format).

        Returns:
        - self: Fitted transformer.
        """
        # Only update columns_to_drop if drop_patterns is explicitly set
        if self.drop_patterns is not None:
            all_columns = X.columns.tolist()
            self.columns_to_drop = fs.identify_matching_columns(
                self.drop_patterns, all_columns
            )
            logger.info(f"Identified {len(self.columns_to_drop)} columns to drop")
        else:
            # If no patterns are provided, log and leave columns_to_drop unchanged
            logger.info("No drop_patterns provided. Keeping columns_to_drop unchanged.")

        return self

    def transform(self, X):
        """
        Drop the identified columns from the input data.

        Params:
        - X (DataFrame): Input data.

        Returns:
        - DataFrame: Data with specified columns dropped.
        """
        if not self.columns_to_drop:
            logger.info("No columns to drop. Returning data unchanged.")
            return X

        # Filter columns that exist in the current dataset
        dropped_columns = [col for col in self.columns_to_drop if col in X.columns]
        logger.info(f"Dropping {len(dropped_columns)} columns based on name pattern params.")

        # Drop columns safely
        return X.drop(columns=dropped_columns, errors='ignore')
