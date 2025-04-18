import time
import logging
from typing import Dict, Union, List, Any
from itertools import chain,combinations
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
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
        self.X_eval = None
        self.y_eval = None
        self.X_test = None
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

        self.pipeline = self._get_base_pipeline()
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
    #      Pipeline/Modeling Methods
    # -----------------------------------

    def _get_base_pipeline(self) -> Pipeline:
        """
        Construct and return the base modeling pipeline (feature selection, dropping columns, and the regressor).
        """
        model_params = self.modeling_config['model_params'].copy()

        # Update min_child_weight if percentage is specified
        if model_params.get('min_child_weight_pct'):
            model_params['min_child_weight'] = self._convert_min_child_pct_to_weight(
                self.X_train,
                model_params.pop('min_child_weight_pct')
            )

        base_pipeline = Pipeline([
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

        return base_pipeline


    # Modify _split_data method to do two splits:
    def _split_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Create train/eval/test splits relative to total population.

        Params:
        - X (DataFrame): feature data
        - y (Series): target variable

        Test split happens first, then eval split is calculated relative to original population.
        For example: test_size=0.2, eval_size=0.1 means:
        - 20% goes to test
        - 10% of total (12.5% of remaining) goes to eval
        - 70% goes to train
        """
        # First split off test set
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y,
            test_size=self.modeling_config['test_size'],
            random_state=self.modeling_config['model_params'].get('random_state', 42)
        )

        # Calculate eval split size relative to original population
        remaining_fraction = 1 - self.modeling_config['test_size']
        relative_eval_size = self.modeling_config['eval_size'] / remaining_fraction

        # Split remaining data into train/eval
        self.X_train, self.X_eval, self.y_train, self.y_eval = train_test_split(
            X_temp, y_temp,
            test_size=relative_eval_size,
            random_state=self.modeling_config['model_params'].get('random_state', 42)
        )


    def _fit(self) -> None:
        """
        Fit the pipeline on training data with early stopping using test set.
        """
        # Get pipeline steps
        transformer = self.pipeline[:-1]
        regressor = self.pipeline[-1]

        # Fit transformer and transform X_train and X_eval for early stopping
        X_train_transformed = transformer.fit_transform(self.X_train)
        X_eval_transformed = transformer.transform(self.X_eval)
        eval_set = [(X_train_transformed, self.y_train),
                    (X_eval_transformed, self.y_eval)]


        # Check if phase training is enabled
        phase_config = self.modeling_config.get('phase_training', {})
        if not phase_config.get('enabled'):
            # Original single-phase training
            logger.info(f"Training model using data with shape: {X_train_transformed.shape}...")
            u.notify('startup')
            regressor.fit(
                X_train_transformed,
                self.y_train,
                eval_set=eval_set
            )
        else:
            # Multi-phase training
            logger.info(f"Beginning phase training with {len(phase_config['phases'])} phases...")
            u.notify('startup')

            # Store original parameters
            base_params = regressor.get_params()

            for i, phase in enumerate(phase_config['phases'], 1):
                # Update all parameters specified in this phase
                current_params = base_params.copy()
                current_params.update(phase['params'])

                # Ensure eval_metric is string
                current_params['eval_metric'] = 'rmse'

                # Log all parameter overrides
                param_changes = {k: v for k, v in phase['params'].items()}
                logger.info(f"Phase {i}: Training with parameters: {param_changes}")

                regressor.set_params(**current_params)
                regressor.fit(
                    X_train_transformed,
                    self.y_train,
                    eval_set=eval_set,
                    xgb_model=None if i == 1 else regressor
                )

        self.training_time = time.time() - self.start_time
        logger.milestone("Training completed after %.2f seconds.", self.training_time)


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

    def _run_grid_search(self, X: pd.DataFrame, y: pd.Series, pipeline=None) -> Dict[str, float]:
        """
        Perform grid search with cross validation using configured parameters.

        Params:
        - X (DataFrame): Feature data
        - y (Series): Target variable
        - pipeline (Pipeline, optional): Custom pipeline to use for grid search. If None,
          a basic pipeline will be created.

        Returns:
        - Dict: Results containing best parameters, score and CV results
        """
        # If grid search is disabled, return nothing
        if not self.modeling_config.get('grid_search_params', {}).get('enabled'):
            logger.info("Constructing production model with base params...")
            return {}
        logger.info("Initiating grid search...")
        u.notify('gadget')

        # Get prepared grid search params
        gs_config = self._prepare_grid_search_params(X)

        # Validate that the param_grid has at least 2 configurations
        total_configurations = 1
        for param_values in gs_config['param_grid'].values():
            if isinstance(param_values, list):
                total_configurations *= len(param_values)
        if total_configurations < 2:
            raise ValueError("Grid search requires at least 2 different configurations. "
                             "Current param_grid generates only 1.")

        # Use provided pipeline or create default pipeline
        if pipeline is None:
            cv_pipeline = self._get_model_pipeline(gs_config['base_model_params'])
        else:
            cv_pipeline = pipeline

        # Store the search object in the instance
        self.random_search = RandomizedSearchCV(
            cv_pipeline,
            gs_config['param_grid'],
            **gs_config['search_config']
        )

        # Fit without passing verbose flag (XGB verbosity controlled via model_params)
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


    def _prepare_grid_search_params(self, X: pd.DataFrame, base_params_override=None) -> dict:
        """
        Prepare grid search parameters that can be used by both BaseModel and WalletModel.

        Params:
        - X (DataFrame): Data used for parameter calculations
        - base_params_override (dict, optional): Override base params if needed

        Returns:
        - dict: Prepared grid search configuration
        """
        # 1. Retrieve and modify base params
        base_model_params = base_params_override or self.modeling_config['model_params'].copy()

        # Remove params that don't apply to the grid search
        for param in ['early_stopping_rounds', 'eval_metric', 'verbose']:
            base_model_params.pop(param, None)

        # Handle 'min_child_weight_pct' in base params
        if base_model_params.get('min_child_weight_pct'):
            base_model_params['min_child_weight'] = self._convert_min_child_pct_to_weight(
                X,
                base_model_params.pop('min_child_weight_pct')
            )

        # 2. Prepare grid search param options
        grid_search_params = self.modeling_config['grid_search_params']
        param_grid = grid_search_params['param_grid'].copy()

        # Handle 'min_child_weight_pct' in grid search params
        if 'regressor__min_child_weight_pct' in param_grid:
            pct_grid = param_grid.pop('regressor__min_child_weight_pct')
            param_grid['regressor__min_child_weight'] = [
                self._convert_min_child_pct_to_weight(X, pct) for pct in pct_grid
            ]

        # Process drop patterns logic
        if 'drop_columns__drop_patterns' in param_grid:
            if self.modeling_config['grid_search_params'].get('drop_patterns_include_n_features'):
                drop_pattern_combinations = self._create_drop_pattern_combinations()
            else:
                base_drop_patterns = self.modeling_config['feature_selection']['drop_patterns']
                drop_pattern_combinations = [
                    base_drop_patterns + grid_pattern
                    for grid_pattern in param_grid['drop_columns__drop_patterns']
                ]
            param_grid['drop_columns__drop_patterns'] = drop_pattern_combinations

        return {
            'base_model_params': base_model_params,
            'param_grid': param_grid,
            'search_config': {
                'n_iter': grid_search_params['n_iter'],
                'cv': grid_search_params['n_splits'],
                'scoring': grid_search_params['scoring'],
                'verbose': grid_search_params.get('verbose_level', 0),
                'n_jobs': base_model_params.get('n_jobs', -1),
                'random_state': base_model_params.get('random_state', 42),
            }
        }



    def _get_model_pipeline(self, base_model_params: Dict[str, Any]) -> Pipeline:
        """
        Create a standard model pipeline with drop columns transformer and XGBoost regressor.

        Params:
        - base_model_params (Dict, optional): Parameters for the XGBoost regressor.
        If None, default parameters will be used.

        Returns:
        - Pipeline: Scikit-learn pipeline with preprocessing and model components
        """
        pipeline = Pipeline([
            ('drop_columns', DropColumnPatterns()),
            ('regressor', XGBRegressor(**base_model_params))
        ])

        return pipeline


    def save_pipeline(self, filepath: str) -> None:
        """
        Save the fitted pipeline to disk using pickle.

        Params:
        - filepath (str): Path where pipeline should be saved, should end in .pkl
        """
        if self.pipeline is None:
            raise ValueError("Pipeline must be fitted before saving")

        # Save pipeline which contains both model and column transformers
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)

        logger.info(f"Pipeline saved to {filepath}")


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

        # First get columns that remain after base patterns
        all_columns = self.X_train.columns.tolist()
        base_drops = fs.identify_matching_columns(base_patterns, all_columns)
        remaining_columns = [col for col in all_columns if col not in base_drops]

        # Validate grid patterns against remaining columns
        for pattern_list in grid_patterns:
            for pattern in pattern_list:
                if pattern != 'feature_retainer':
                    matching_cols = fs.identify_matching_columns([pattern], remaining_columns)
                    if not matching_cols:
                        raise ValueError(
                            f"Grid search drop pattern '{pattern}' matches no columns after base drops"
                        )

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

        # Extract base patterns to remove them from display
        base_patterns = set(self.modeling_config['feature_selection']['drop_patterns'])

        report_data = []
        param_cols = [col for col in results_df.columns if col.startswith("param_")]
        variable_params = [
            col for col in param_cols
            if results_df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x).nunique() > 1
        ]

        for param in variable_params:
            param_name = param.replace("param_", "")
            for _, row in results_df.iterrows():
                param_value = row[param]

                # Clean up drop patterns display
                if param_name == 'drop_columns__drop_patterns' and isinstance(param_value, list):
                    # Only show patterns that aren't in base config
                    unique_patterns = [p for p in param_value if p not in base_patterns and p != 'feature_retainer']
                    param_value = unique_patterns if unique_patterns else ['feature_retainer']

                report_data.append({
                    'param': param_name,
                    'param_value': str(param_value),
                    'avg_score': row['mean_test_score'],
                    'total_builds': len(results_df)
                })

        return (pd.DataFrame(report_data)
                .groupby(['param','param_value'])[['avg_score']]
                .mean('avg_score')
                .sort_values(by='avg_score', ascending=False))




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

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Pipeline step for feature selection based on variance and correlation"""
    __module__ = 'base_modeling.base_model'  # Add this line

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
            logger.debug("No columns to drop. Returning data unchanged.")
            return X

        # Filter columns that exist in the current dataset
        dropped_columns = [col for col in self.columns_to_drop if col in X.columns]
        logger.debug(f"Dropping {len(dropped_columns)} columns based on name pattern params.")

        # Drop columns safely
        return X.drop(columns=dropped_columns, errors='ignore')
