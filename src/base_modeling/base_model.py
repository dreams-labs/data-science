"""
Core modeling framework for XGBoost-based prediction with custom pipeline management.

This BaseModel contains shared methods called by classes:
    - WalletModel (wallet_modeling/wallet_model.py)
    - CoinModel (coin_modeling/coin_model.py)

The BaseModel class orchestrates the entire modeling workflow:
1. Data preparation and train/eval/test splitting
2. Pipeline construction using components from pipeline.py (MetaPipeline, TargetVarSelector, etc.)
3. Grid search with custom scorers from scorers.py
4. Model training with optional multi-phase training and asymmetric loss

Key integrations:
- **pipeline.py**: Provides MetaPipeline for simultaneous X/y transformations, TargetVarSelector
  for multi-column target handling, FeatureSelector and DropColumnPatterns for feature reduction
- **scorers.py**: Custom scoring functions that handle y transformations during grid search,
  enabling proper evaluation of models with complex target preprocessing
- **feature_selection.py**: Utilities for variance/correlation-based feature removal used by
  the pipeline components

The class handles both regression and classification (binary and asymmetric 3-class), with
extensive configuration through modeling_config dict. Designed to be extended by domain-specific
model classes (WalletModel, CoinModel) that add their own data loading and feature engineering.

Grid search intelligently scales from single parameter tests to complex multi-pattern feature
selection experiments, always maintaining compatibility with early stopping and custom metrics.
"""
# Multithreading configurations for grid search
import os
os.environ['OMP_NUM_THREADS'] = '12'
os.environ['XGBOOST_OMP_NUM_THREADS'] = '12'
os.environ['MKL_NUM_THREADS'] = '12'

# pylint:disable=wrong-import-position
import time
import logging
import uuid
from typing import Dict, Union, List, Any
from itertools import chain,combinations
import cloudpickle
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor, XGBClassifier, XGBRanker

# Local modules
import base_modeling.feature_selection as fs
import base_modeling.pipeline as bmp
import base_modeling.scorers as sco
import utils as u
from utils import ConfigError

# pylint:disable=invalid-name  # X_test isn't camelcase
# pylint:disable=unused-argument  # X and y params are always needed for pipeline structure


# Set up logger at the module level
logger = logging.getLogger(__name__)


class BaseModel:
    """
    Base class for XGBoost-based prediction models.
    Handles core pipeline functionality, data preparation, and model training.

    Technical Architecture:
    - Uses MetaPipeline to coordinate separate X and y transformations
    - Integrates custom scorers that transform y before evaluation during grid search
    - Supports regression, binary classification, and 3-class asymmetric classification
    - Implements early stopping with separate eval set (not cross-validation folds)

    Key Methods:
    - construct_base_model(): Main entry point, orchestrates grid search and training
    - _get_base_pipeline(): Builds sklearn Pipeline with feature selection + XGBoost
    - _get_meta_pipeline(): Wraps base pipeline in MetaPipeline with y transformations
    - _run_grid_search(): Executes RandomizedSearchCV with custom scorers
    - _fit(): Handles actual training including multi-phase training support

    Configuration:
    All behavior controlled through modeling_config dict including:
    - Model type and parameters
    - Feature selection thresholds
    - Grid search parameters and scorer selection
    - Train/eval/test split ratios

    Usage Pattern:
    BaseModel provides the modeling engine and wallet-specific features are built on top
    such as methods for loading wallet/coin data and creating features, while BaseModel's
    construct_base_model() used to actually train the model. Think of BaseModel as the
    reusable modeling toolkit that domain-specific models build upon.

    See module docstring for overall workflow and business logic details.
    """
    def __init__(self, modeling_config: dict):
        """
        Params:
        - modeling_config (dict): configuration dictionary for modeling parameters.
        """
        # Generate ID
        self.model_id = str(uuid.uuid4())

        # Key Params
        self.modeling_config = modeling_config
        self.training_data_df = None

        # Pipeline Steps
        self.pipeline = None
        self.y_pipeline = None
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
        self.asymmetric_loss_enabled = modeling_config['asymmetric_loss'].get('enabled',False)

        # Validate presence of grid_search_params
        if 'grid_search_params' not in self.modeling_config:
            raise ConfigError("Missing 'grid_search_params' in modeling_config")

        # Convert drop patterns to a list of lists
        param_grid = self.modeling_config.get('grid_search_params', {}).get('param_grid', {})
        if 'drop_columns__drop_patterns' in param_grid \
        and self.modeling_config['grid_search_params'].get('enabled',False):
            grid_patterns = param_grid['drop_columns__drop_patterns']
            if grid_patterns and all(isinstance(p, str) for p in grid_patterns):
                grid_patterns = [[p] for p in grid_patterns]
            elif grid_patterns == []:
                pass
            else:
                raise ValueError(
                    f"drop_columns__drop_patterns must be a list of strings, got {type(grid_patterns)}"
                )
            param_grid['drop_columns__drop_patterns'] = grid_patterns
            self.modeling_config['grid_search_params']['param_grid']['drop_columns__drop_patterns'] = grid_patterns

        # Assign scorer based on model type
        if self.modeling_config['model_type'] == 'regression':
            scorer = modeling_config['grid_search_params'].get('regressor_scoring')
        elif self.modeling_config['model_type'] == 'classification':
            scorer = modeling_config['grid_search_params'].get('classifier_scoring')
        else:
            raise ValueError(f"Invalid model type '{self.modeling_config['model_type']}' found in modeling config.")
        self.modeling_config['grid_search_params']['scorer'] = scorer



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

        # Asymmetric loss is slow so broadcast a warning
        if self.modeling_config['asymmetric_loss'].get('enabled',False):
            logger.warning("Beginning extended training with asymmetric loss target variables...")


        cv_results = self._run_grid_search(self.X_train, self.y_train, self.pipeline)

        if cv_results.get('best_params'):
            best_params = {
                k.replace('estimator__', ''): v
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
            'model_id': self.model_id,
            'modeling_config': self.modeling_config,
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
    #          Modeling Methods
    # -----------------------------------

    @u.timing_decorator
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

        # Predict probabilities
        if self.modeling_config['model_type'] == 'classification':
            raw_probs = self.pipeline.predict_proba(self.X_test)

            # pick the “positive” class index (typically 1)
            pos_idx = list(self.pipeline.named_steps['regressor'].classes_).index(1)

            # build a Series of just the positive‐class probability
            self.y_pred = pd.Series(
                raw_probs[:, pos_idx],
                index=self.X_test.index,
                name='probability_of_1'
            )

        return self.y_pred




    # -----------------------------------
    #         Grid Search Methods
    # -----------------------------------

    # pylint:disable=no-member  # some objects are from WalletModel or CoinModel
    def _run_grid_search(self, X: pd.DataFrame, y: pd.Series, pipeline) -> Dict[str, float]:
        """
        Run grid search while always providing an eval set for early stopping.
        """
        if not self.modeling_config.get('grid_search_params', {}).get('enabled'):
            logger.info("Constructing production model with base params...")
            return {}

        logger.info("Initiating grid search with eval set...")
        u.notify('gadget')

        # Set up grid search options as ['param_grid']
        gs_config = self._prepare_grid_search_params(X)

        # Assign custom scorers if applicable
        scoring_param = gs_config['search_config'].get('scoring')

        if scoring_param == 'custom_r2_scorer':
            gs_config['search_config']['scoring'] = sco.custom_r2_scorer

        elif scoring_param == 'custom_neg_rmse_scorer':
            gs_config['search_config']['scoring'] = sco.custom_neg_rmse_scorer

        elif scoring_param == 'validation_r2_scorer':
            # Ensure validation data is available
            if self.X_validation is None or self.validation_target_vars_df is None:
                raise ValueError("Validation data required for validation_r2_scorer")
            gs_config['search_config']['scoring'] = sco.validation_r2_scorer(self)

        elif scoring_param == 'validation_auc_scorer':
            # Ensure validation data is available
            if self.X_validation is None or self.validation_target_vars_df is None:
                raise ValueError("Validation data required for validation_auc_scorer")
            gs_config['search_config']['scoring'] = sco.validation_auc_scorer(self)

        elif scoring_param == 'validation_top_percentile_returns_scorer':
            # read your desired top_pct from config, e.g. self.modeling_config['grid_search_params']['top_pct']
            threshold = self.modeling_config['grid_search_params']['percentile_threshold']
            gs_config['search_config']['scoring'] = sco.validation_top_percentile_returns_scorer(self, threshold)

        elif scoring_param == 'validation_top_scores_returns_scorer':
            # read your desired top_pct from config, e.g. self.modeling_config['grid_search_params']['top_pct']
            gs_config['search_config']['scoring'] = sco.validation_top_scores_returns_scorer(self)

        elif scoring_param == 'neg_log_loss':
            # if self.modeling_config['asymmetric_loss'].get('enabled',False):
                # raise ConfigError("Cannot use neg_log_loss for an asymmetric loss target.")
            # Use custom neg_log_loss scorer that handles y transformation
            gs_config['search_config']['scoring'] = sco.custom_neg_log_loss_scorer

        else:
            raise ValueError(f"Invalid scoring metric '{scoring_param}' found in grid_search_params.")

        # Generate pipeline
        cv_pipeline = pipeline if pipeline is not None else self._get_model_pipeline(gs_config['base_model_params'])

        # Random search with pipeline
        self.random_search = RandomizedSearchCV(
            cv_pipeline,
            gs_config['param_grid'],
            **gs_config['search_config'],
            refit=False,   # skip that extra full-dataset fit
        )

        # Always pass the eval_set for early stopping
        self.random_search.fit(
            X, y,
            eval_set=(self.X_eval, self.y_eval),
            verbose_estimators=self.modeling_config['grid_search_params'].get('verbose_estimators',False)
        )

        logger.info("Grid search complete. Best score: %f", -self.random_search.best_score_)
        u.notify('synth_magic')

        return {
            'best_params': {
                k: (self._convert_min_child_weight_to_pct(X, v) if k == 'estimator__min_child_weight' else v)
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

            # Define drop pattern combinations as selections of columns
            if (
                self.modeling_config['grid_search_params'].get('enabled') and
                self.modeling_config['grid_search_params'].get('drop_patterns_include_n_features')
            ):
                drop_pattern_combinations = self._create_drop_pattern_combinations()
            else:
                # Add feature_retainer to get score if nothing is dropped
                param_grid['drop_columns__drop_patterns'] = (param_grid['drop_columns__drop_patterns']
                                                              + [['feature_retainer']])

                base_drop_patterns = self.modeling_config['feature_selection']['drop_patterns']
                drop_pattern_combinations = [
                    base_drop_patterns + grid_pattern
                    for grid_pattern in param_grid['drop_columns__drop_patterns']
                ]
            param_grid['drop_columns__drop_patterns'] = drop_pattern_combinations

        if self.modeling_config['model_type'] == 'regression':
            scorer = grid_search_params['regressor_scoring']
        else:
            scorer = grid_search_params['classifier_scoring']


        return {
            'base_model_params': base_model_params,
            'param_grid': param_grid,
            'search_config': {
                'n_iter': grid_search_params['n_iter'],
                'cv': grid_search_params['n_splits'],
                'scoring': scorer,
                'verbose': grid_search_params.get('verbose_level', 0),
                'n_jobs': grid_search_params.get('n_jobs',
                              base_model_params.get('n_jobs', -1)),
                'pre_dispatch': grid_search_params.get('pre_dispatch', 4),  # <-- limit pre‑dispatch
                'random_state': base_model_params.get('random_state', 42),
            }
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
                if param_name == 'model_pipeline__drop_columns__drop_patterns' and isinstance(param_value, list):
                    # Only show patterns that aren't in base config
                    unique_patterns = [p for p in param_value if p not in base_patterns]

                    if isinstance(self.modeling_config['grid_search_params']
                                  .get('drop_patterns_include_n_features'),int):

                        # Get param grid drop patterns
                        base_drop_patterns = (self.modeling_config['grid_search_params']
                                              ['param_grid']['drop_columns__drop_patterns'])

                        # Convert to single list, rather than list of lists
                        base_drop_patterns = [item for sublist in base_drop_patterns for item in sublist]

                        # Identify which features were retained in the search
                        retained_drop_patterns = set(base_drop_patterns) - set(unique_patterns)
                        if len(retained_drop_patterns) == 0:
                            retained_drop_patterns = 'None'
                        param_value = f"Retained features: {retained_drop_patterns}"
                    else:
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
                .round(3)
                .sort_values(by='avg_score', ascending=False))


    def generate_individual_results_report(self) -> pd.DataFrame:
        """
        Generate a report showing performance of each individual parameter combination tested.

        Returns:
        - individual_df (DataFrame): Each row represents one combination with its score and rank
        """
        if not self.random_search:
            logger.error("Random search has not been run.")
            return None
        elif not hasattr(self.random_search, 'cv_results_'):
            logger.error("cv_results_ is unavailable.")
            return None

        results_df = pd.DataFrame(self.random_search.cv_results_)
        base_patterns = set(self.modeling_config['feature_selection']['drop_patterns'])

        # Extract parameter columns and identify variable ones
        param_cols = [col for col in results_df.columns if col.startswith("param_")]
        variable_params = [
            col for col in param_cols
            if results_df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x).nunique() > 1
        ]

        individual_results = []
        for _, row in results_df.iterrows():
            result_entry = {
                'mean_test_score': round(row['mean_test_score'], 3),
                'std_test_score': round(row['std_test_score'], 3),
                'median_test_score': round(np.median([
                    row[f'split{i}_test_score'] for i in range(self.modeling_config['grid_search_params']['n_splits'])
                ]), 3),
                'rank_test_score': row['rank_test_score']
            }

            # Add only variable parameters as separate columns
            for param_col in variable_params:
                param_name = param_col.replace("param_", "")
                param_value = row[param_col]

                # Clean up drop patterns display
                if param_name == 'model_pipeline__drop_columns__drop_patterns' and isinstance(param_value, list):
                    unique_patterns = [p for p in param_value if p not in base_patterns]
                    if not unique_patterns:
                        param_value = 'feature_retainer'
                    else:
                        param_value = str(unique_patterns)

                result_entry[param_name] = param_value

            individual_results.append(result_entry)

        individual_df = pd.DataFrame(individual_results)

        # Remove pipeline prefixes from column names
        individual_df.columns = [
        (col
         .replace('y_pipeline__', '')
         .replace('model_pipeline__', '')
         )
        for col in individual_df.columns
        ]

        # Sort by performance (best score first)
        individual_df = individual_df.sort_values('mean_test_score', ascending=False).reset_index(drop=True)

        return individual_df

    # -----------------------------------
    #          Pipeline Methods
    # -----------------------------------

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
            ('drop_columns', bmp.DropColumnPatterns()),
            ('estimator', XGBRegressor(**base_model_params))
        ])

        return pipeline


    def _get_base_pipeline(self) -> Pipeline:
        """
        Construct and return the base modeling pipeline (feature selection, dropping columns, and the regressor).
        """
        model_params = self.modeling_config['model_params'].copy()

        # Select model type and eval method
        if self.modeling_config['model_type']=='classification':
            model = XGBClassifier
            model_params['eval_metric'] = 'logloss'  # Binary classification metric

            # Set appropriate eval_metric based on asymmetric loss configuration
            if self.modeling_config.get('asymmetric_loss', {}).get('enabled'):
                model_params['eval_metric'] = 'mlogloss'  # Multi-class metric - OVERRIDE
                model_params['objective'] = 'multi:softprob'  # 3-class classification
                model_params['num_class'] = 3

        elif self.modeling_config['model_type']=='regression':
            model = XGBRegressor
            model_params.setdefault('eval_metric', 'rmse')
        elif self.modeling_config['model_type']=='ranker':
            model = XGBRanker
            model_params.setdefault('eval_metric', 'ndcg')
        else:
            raise ValueError(f"Invalid model type '{self.modeling_config['model_type']}' found in config.")

        # Update min_child_weight if percentage is specified
        if model_params.get('min_child_weight_pct'):
            model_params['min_child_weight'] = self._convert_min_child_pct_to_weight(
                self.X_train,
                model_params.pop('min_child_weight_pct')
            )

        base_pipeline = Pipeline([
            ('feature_selector', bmp.FeatureSelector(
                variance_threshold=self.modeling_config['feature_selection'].get('variance_threshold'),
                correlation_threshold=self.modeling_config['feature_selection'].get('correlation_threshold'),
                protected_features=self.modeling_config['feature_selection']['protected_features']
            )),
            ('drop_columns', bmp.DropColumnPatterns(
                drop_patterns=self.modeling_config['feature_selection']['drop_patterns']
            )),
            ('estimator', model(**model_params))
        ])

        return base_pipeline


    @u.timing_decorator
    def _get_meta_pipeline(self) -> Pipeline:
        """
        Return a single Pipeline that first applies y transformations,
        then the usual feature+estimator steps. Step names remain
        exactly ['target_selector', 'feature_selector', 'drop_columns', 'estimator'].
        """
        # Get the steps from the y_pipeline
        y_steps = self._get_y_pipeline()
        # Get the steps from the base pipeline (feature_selector, drop_columns, estimator)
        model_steps = self._get_base_pipeline()

        # Concatenate them into one pipeline
        combined_steps = bmp.MetaPipeline(y_steps, model_steps)

        return combined_steps


    def _get_y_pipeline(self) -> Pipeline:
        """
        Build the wallet-specific pipeline by prepending the wallet cohort selection
        to the base pipeline steps.
        Validates that classification thresholds are provided and numeric (inf allowed).
        """
        # Determine target variable
        target_var = self.modeling_config.get(
            'model_params', {}
        ).get(
            'target_selector__target_variable',
            self.modeling_config['target_variable']
        )

        # Initialize thresholds
        target_var_min_threshold = None
        target_var_max_threshold = None

        # For classification, require and validate thresholds
        if (self.modeling_config['model_type'] == 'classification'
            and not self.modeling_config.get('asymmetric_loss',{}).get('enabled',False)
            ):
            # Retrieve raw values
            raw_min = (
                self.modeling_config.get('model_params', {}
                ).get('target_selector__target_var_min_threshold',
                    self.modeling_config.get('target_var_min_threshold'))
            )
            raw_max = (
                self.modeling_config.get('model_params', {}
                ).get('target_selector__target_var_max_threshold',
                    self.modeling_config.get('target_var_max_threshold'))
            )
            # Ensure thresholds are provided
            if raw_min is None:
                raise ConfigError("target_var_min_threshold must be set for classification models")
            if raw_max is None:
                raise ConfigError("target_var_max_threshold must be set for classification models")
            # Convert to float (allows inf)
            try:
                target_var_min_threshold = float(raw_min)  # supports float('inf')
                target_var_max_threshold = float(raw_max)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"Thresholds must be numeric values (int, float or inf), got: {raw_min!r}, {raw_max!r}"
                ) from exc

        # Get asymmetric loss parameters - check model_params first for grid search values
        base_asymmetric_config = self.modeling_config.get('asymmetric_loss')

        # Build pipeline
        y_pipeline = Pipeline([
            ('target_selector', bmp.TargetVarSelector(
                target_var,
                target_var_min_threshold,
                target_var_max_threshold,
                asymmetric_config=base_asymmetric_config,
            ))
        ])

        return y_pipeline


    def save_pipeline(self, filepath: str) -> None:
        """
        Save the fitted pipeline to disk using cloudpickle.

        Params:
        - filepath (str): Path where pipeline should be saved, should end in .pkl
        """
        if self.pipeline is None:
            raise ValueError("Pipeline must be fitted before saving")

        # Save pipeline which contains both model and column transformers
        with open(filepath, 'wb') as f:
            cloudpickle.dump(self.pipeline, f)

        logger.info(f"Pipeline saved to {filepath}")




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
