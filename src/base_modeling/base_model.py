import time
import logging
from typing import Dict, Union, List
from itertools import chain,combinations
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
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_validate = None
        self.X_validate = None
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
            ('feature_selector', FeatureSelector(
                variance_threshold=self.modeling_config['feature_selection'].get('variance_threshold'),
                correlation_threshold=self.modeling_config['feature_selection'].get('correlation_threshold'),
                protected_features=self.modeling_config['feature_selection']['protected_features']
            )),
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
        u.notify('startup')
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

        # 1. Retrieve and modify base params
        # -----------------------
        base_model_params = self.modeling_config['model_params'].copy()

        # Remove params that don't apply to the grid search
        for param in ['early_stopping_rounds', 'eval_metric', 'verbose']:
            base_model_params.pop(param, None)

        # Handle 'min_child_weight_pct' in base params
        if base_model_params.get('min_child_weight_pct'):
            base_model_params['min_child_weight'] = self._convert_min_child_pct_to_weight(
                self.X_train,
                base_model_params.pop('min_child_weight_pct')
            )


        # 2. Prepare grid search param options
        # ------------------------------------
        grid_search_params = self.modeling_config['grid_search_params']

        # Handle 'min_child_weight_pct' in grid search params
        if 'regressor__min_child_weight_pct' in grid_search_params['param_grid']:
            pct_grid = grid_search_params['param_grid'].pop('regressor__min_child_weight_pct')
            grid_search_params['param_grid']['regressor__min_child_weight'] = [
                self._convert_min_child_pct_to_weight(X, pct) for pct in pct_grid
            ]

        # Combine drop patterns in grid search params with base drop patterns
        if 'drop_columns__drop_patterns' in grid_search_params['param_grid']:

            # if adding features in groups of n, use helper function
            if self.modeling_config['grid_search_params'].get('drop_patterns_include_n_features'):
                drop_pattern_combinations = self._create_drop_pattern_combinations()

            # if removing features one by one, generate combinations
            else:
                base_drop_patterns = self.modeling_config['feature_selection']['drop_patterns']
                drop_pattern_combinations = [
                    base_drop_patterns + grid_pattern
                    for grid_pattern in grid_search_params['param_grid']['drop_columns__drop_patterns']
                ]

            # Override config with the merged drop_patterns
            grid_search_params['param_grid']['drop_columns__drop_patterns'] = drop_pattern_combinations


        # 3. Search
        # ---------
        # Create pipeline for grid search
        cv_pipeline = Pipeline([
            ('drop_columns', DropColumnPatterns()),
            ('regressor', XGBRegressor(**base_model_params))
        ])

        # Store the search object in the instance
        self.random_search = RandomizedSearchCV(
            cv_pipeline,
            grid_search_params['param_grid'],
            n_iter = grid_search_params['n_iter'],
            cv = grid_search_params['n_splits'],
            scoring = grid_search_params['scoring'],
            verbose = grid_search_params.get('verbose_level', 0),
            n_jobs = base_model_params.get('n_jobs', -1),
            random_state = base_model_params.get('random_state', 42),
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


    def _create_drop_pattern_combinations(self) -> List[List[str]]:
        """
        Generates lists of drop features that contain
            1. all base drop patterns, plus
            2. all grid search drop patterns except for n features, defined through
                drop_patterns_include_n_features
        """
        max_additions = self.modeling_config['grid_search_params'].get('drop_patterns_include_n_features')
        grid_patterns = self.modeling_config['grid_search_params']['param_grid']['drop_columns__drop_patterns']
        base_patterns = self.modeling_config['feature_selection']['drop_patterns']

        # Flatten all grid drop patterns into a single list
        grid_patterns_list = list(chain.from_iterable(grid_patterns)) + ['feature_retainer']

        # Create combinations of grid drop patterns that retain max_additions features
        grid_pattern_combinations = [list(combo)
                                    for combo in combinations(grid_patterns_list,
                                                            len(grid_patterns_list) - max_additions)]

        # Combine each grid_pattern_combination with the base drop patterns
        search_drop_combinations = [
            base_patterns + grid_pattern_combination
            for grid_pattern_combination in grid_pattern_combinations
        ]

        return search_drop_combinations


    def generate_search_report(self, output_raw_data=False) -> pd.DataFrame:
        """
        Generate a report of the random search results, excluding constant parameters.

        Params:
        - output_raw_data (bool): If True, returns raw cv_results_

        Returns:
        - report_df (DataFrame): DataFrame with variable parameters and their scores
        """
        if not self.random_search:
            logger.error("Random search has not been run.")
            return None
        elif not hasattr(self.random_search, 'cv_results_'):
            logger.error("cv_results_ is unavailable.")
            return None

        results_df = pd.DataFrame(self.random_search.cv_results_)

        if output_raw_data:
            return results_df

        report_data = []
        param_cols = [col for col in results_df.columns if col.startswith("param_")]

        # Only include parameters with multiple unique values
        variable_params = [
            col for col in param_cols
            if results_df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x).nunique() > 1
        ]

        for param in variable_params:
            param_name = param.replace("param_", "")
            for _, row in results_df.iterrows():
                report_data.append({
                    'param': param_name,
                    'param_value': str(row[param]),
                    'avg_score': row['mean_test_score'],
                    'total_builds': len(results_df)
                })

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


    def fit(self, X, y=None): # pylint:disable=unused-argument  # y param needed for pipeline structure
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



class FeatureSelector(BaseEstimator, TransformerMixin):
    """Pipeline step for feature selection based on variance and correlation"""

    def __init__(self, variance_threshold: float, correlation_threshold: float,
                 protected_features: List[str]):
        """
        Params:
        - variance_threshold (float): Minimum variance threshold for features
        - correlation_threshold (float): Maximum correlation threshold between features
        - protected_features (List[str]): Features to retain regardless of thresholds
        """
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.protected_features = protected_features
        self.selected_features = None


    def fit(self, X: pd.DataFrame, y=None):  # pylint:disable=unused-argument  # y param needed for pipeline structure
        """Identify features to keep based on variance and correlation thresholds"""
        # Remove low variance features
        post_variance_df = fs.remove_low_variance_features(
            X,
            self.variance_threshold,
            self.protected_features
        )

        # Remove correlated features
        post_correlation_df = fs.remove_correlated_features(
            post_variance_df,
            self.correlation_threshold,
            self.protected_features
        )

        self.selected_features = post_correlation_df.columns.tolist()
        return self


    def transform(self, X: pd.DataFrame):
        """Apply feature selection"""
        return X[self.selected_features]
