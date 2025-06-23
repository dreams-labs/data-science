"""
Custom pipeline components for XGBoost modeling with separate X and y transformations.

This module provides pipeline steps that integrate with sklearn's Pipeline framework
to handle our specific data transformation needs:

1. **MetaPipeline**: Orchestrates both feature (X) and target (y) transformations
   - Applies y_pipeline to transform raw targets (potentially multi-column DataFrames)
   - Applies model_pipeline to transform features and fit the estimator
   - Handles classification (binary and asymmetric 3-class) and regression
   - Exposes underlying estimator attributes for sklearn compatibility

2. **TargetVarSelector**: Transforms target variables
   - Extracts specific columns from multi-column target DataFrames
   - Converts continuous targets to binary classification (using thresholds)
   - Handles asymmetric loss with 3 classes (big loss, neutral, big win)

3. **FeatureSelector**: Removes low-quality features
   - Drops low variance features (near-constant)
   - Removes highly correlated features
   - Protects specified features from removal

4. **DropColumnPatterns**: Drops columns by pattern matching
   - Supports wildcards (*) for flexible pattern matching
   - Used in grid search to test different feature subsets
   - Preserves protected columns

These components work together to create a flexible modeling pipeline that can
handle complex transformations while remaining compatible with sklearn's grid search.
"""

# pylint:disable=wrong-import-position
import logging
from typing import List
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# Local modules
import base_modeling.feature_selection as fs

# pylint:disable=invalid-name  # X_test isn't camelcase
# pylint:disable=unused-argument  # X and y params are always needed for pipeline structure


# Set up logger at the module level
logger = logging.getLogger(__name__)



# -----------------------------------
#           Pipeline Steps
# -----------------------------------

class MetaPipeline(BaseEstimator, TransformerMixin):
    """Meta-pipeline that applies x and y transformations then fits a regressor."""

    def __init__(self, y_pipeline: Pipeline, model_pipeline: Pipeline):
        """Initialize MetaPipeline with y_pipelin and model_pipeline."""
        self.y_pipeline = y_pipeline
        self.model_pipeline = model_pipeline
        self.estimator = None
        self.x_transformer_ = None  # will store the transformer sub-pipeline for later use

        # Create a named_steps attribute that mimics sklearn Pipeline interface
        self.named_steps = {}
        # Add steps from model_pipeline to named_steps
        for name, step in y_pipeline.named_steps.items():
            self.named_steps[name] = step
            self.name = step
        for name, step in model_pipeline.named_steps.items():
            self.named_steps[name] = step
            self.name = step

    def fit(
            self, X, y,
            eval_set=None,
            verbose_estimators=False,
            modeling_config=None,
            sample_weight=None):
        """Fit the MetaPipeline on raw X and y data, using an optional eval_set for early stopping."""
        # First, transform y using the y_pipeline
        y_trans = self.y_pipeline.fit_transform(y)

        # Create a transformer sub-pipeline (all steps except the final estimator)
        transformer = Pipeline(self.model_pipeline.steps[:-1])

        # Transform training data
        X_trans = transformer.fit_transform(X, y_trans)

        # Extract the regressor from the pipeline
        regressor_name, self.estimator = self.model_pipeline.steps[-1]

        # Generate sample weights AFTER y_pipeline has been fitted with grid search params
        target_selector = self.y_pipeline.named_steps['target_selector']
        effective_asymmetric_config = target_selector.get_effective_asymmetric_config()

        if effective_asymmetric_config and effective_asymmetric_config.get('enabled'):
            sample_weight = self._generate_asymmetric_sample_weights(y_trans, effective_asymmetric_config)


        # If evaluation set is provided, transform it and use for early stopping
        transformed_eval_set = None
        if eval_set is not None:
            X_eval, y_eval = eval_set
            y_eval_trans = self.y_pipeline.transform(y_eval)
            X_eval_trans = transformer.transform(X_eval)
            transformed_eval_set = [(X_trans, y_trans), (X_eval_trans, y_eval_trans)]

            # Define all fit params
            fit_params = {
                'eval_set': transformed_eval_set,
                'verbose': verbose_estimators
            }
            if sample_weight is not None:
                fit_params['sample_weight'] = sample_weight


            # Fit with early stopping using the transformed eval set
            self.estimator.fit(
                X_trans,
                y_trans,
                **fit_params
            )
        else:
            # Regular fit without early stopping if no eval set provided
            self.estimator.fit(X_trans, y_trans)

        # Store the transformer sub-pipeline for use during prediction
        self.x_transformer_ = transformer

        # Update named_steps with the fitted regressor
        self.named_steps[regressor_name] = self.estimator

    def predict_proba(self, X):
        """
        Transform input features and delegate to the underlying estimator's predict_proba.
        For multi-class asymmetric loss, reshape to binary format focusing on positive class.
        """
        X_trans = self.x_transformer_.transform(X)
        probas = self.estimator.predict_proba(X_trans)

        # Check if multi-class (3 classes for asymmetric loss)
        if probas.shape[1] == 3:
            # Convert to binary: class 2 vs (classes 0&1)
            positive_probs = probas[:, 2]  # class 2 probabilities
            negative_probs = probas[:, 0] + probas[:, 1]  # combined classes 0&1
            return np.column_stack([negative_probs, positive_probs])

        # Return as-is for binary classification
        return probas

    def predict(self, X):
        """
        Predict using the fitted regressor on transformed X.
        For multi-class classification, convert to binary by treating class 2 as positive.
        """
        X_trans = self.x_transformer_.transform(X)
        predictions = self.estimator.predict(X_trans)

        # Check if we have multi-class predictions (0, 1, 2 for asymmetric loss)
        unique_preds = np.unique(predictions)
        if len(unique_preds) > 2 and max(unique_preds) == 2:
            # Convert multi-class to binary: class 2 → 1, classes 0&1 → 0
            return (predictions == 2).astype(int)

        # Return as-is for binary classification or regression
        return predictions

    def score(self, X, y):
        """Return the regressor's score on transformed X and y."""
        X_trans = self.x_transformer_.transform(X)
        y_trans = self.y_pipeline.transform(y)
        return self.estimator.score(X_trans, y_trans)

    # Add methods to make it behave more like a sklearn Pipeline
    def __getitem__(self, key):
        """Support indexing like a regular pipeline."""
        if isinstance(key, slice):
            # If it's a slice, return a new Pipeline with the sliced steps
            return Pipeline(self.model_pipeline.steps[key])
        else:
            # Otherwise return the specific step
            return self.model_pipeline.steps[key][1]

    def _generate_asymmetric_sample_weights(self, y_transformed, asymmetric_config):
        """Generate sample weights for asymmetric loss"""
        weights = np.ones(len(y_transformed))
        weights[y_transformed == 0] = asymmetric_config.get('loss_penalty_weight', 1.0)
        weights[y_transformed == 2] = asymmetric_config.get('win_reward_weight', 1.0)
        return weights



class TargetVarSelector(BaseEstimator, TransformerMixin):
    """
    Transformer that extracts the target variable from the input.

    If y is a DataFrame, returns the column specified by target_variable as a Series.
    If y is already a Series, returns it unchanged.

    Optionally, if min/max thresholds are provided, values between (inclusive) are labelled
     positive (1), others negative (0).

    This centralizes the target extraction logic so that grid search can update the target
     variable parameter without interference from pre-extraction.
    """
    def __init__(
            self,
            target_variable: str,
            target_var_min_threshold: float | None = None,
            target_var_max_threshold: float | None = None,
            asymmetric_config: dict = None,

            # Add individual asymmetric parameters for grid search
            asymmetric_enabled: bool = None,
            asymmetric_big_loss_threshold: float = None,
            asymmetric_big_win_threshold: float = None,
            asymmetric_loss_penalty_weight: float = None,
            asymmetric_win_reward_weight: float = None,
    ):
        self.target_variable = target_variable
        self.target_var_min_threshold = target_var_min_threshold
        self.target_var_max_threshold = target_var_max_threshold
        self.asymmetric_config = asymmetric_config

        # Individual asymmetric parameters for grid search compatibility
        self.asymmetric_enabled = asymmetric_enabled
        self.asymmetric_big_loss_threshold = asymmetric_big_loss_threshold
        self.asymmetric_big_win_threshold = asymmetric_big_win_threshold
        self.asymmetric_loss_penalty_weight = asymmetric_loss_penalty_weight
        self.asymmetric_win_reward_weight = asymmetric_win_reward_weight

    def fit(self, y, X=None):
        """
        Validate that the target column exists in y if it is a DataFrame.
        """
        if isinstance(y, pd.DataFrame):
            if self.target_variable not in y.columns:
                raise ValueError(
                    f"Target variable '{self.target_variable}' not found in columns: {y.columns.tolist()}"
                )
        return self

    def transform(self, y, X=None):
        """
        Extract the target column and apply transformations.
        """
        # Extract target variable
        result = y[self.target_variable]

        # Get effective asymmetric config (handles grid search overrides)
        effective_asymmetric_config = self.get_effective_asymmetric_config()

        # Asymmetric loss configuration if configured
        if isinstance(effective_asymmetric_config, dict) and effective_asymmetric_config.get('enabled'):
            loss_thresh = effective_asymmetric_config['big_loss_threshold']
            win_thresh = effective_asymmetric_config['big_win_threshold']
            result = np.where(result < loss_thresh, 0,
                            np.where(result >= win_thresh, 2, 1))

        # Convert to boolean if a min or max threshold is provided
        elif (self.target_var_min_threshold is not None) or (self.target_var_max_threshold is not None):
            lower = -np.inf if self.target_var_min_threshold is None else self.target_var_min_threshold
            upper = np.inf if self.target_var_max_threshold is None else self.target_var_max_threshold
            result = ((result >= lower) & (result <= upper)).astype(int)

        return result

    def get_effective_asymmetric_config(self):
        """
        Merge asymmetric_config with individual parameters, prioritizing individual params.
        This allows grid search to override specific asymmetric parameters.
        """
        if self.asymmetric_config is None and self.asymmetric_enabled is None:
            return None

        # Start with base config
        effective_config = (self.asymmetric_config or {}).copy()

        # Override with individual parameters if they're set
        if self.asymmetric_enabled is not None:
            effective_config['enabled'] = self.asymmetric_enabled
        if self.asymmetric_big_loss_threshold is not None:
            effective_config['big_loss_threshold'] = self.asymmetric_big_loss_threshold
        if self.asymmetric_big_win_threshold is not None:
            effective_config['big_win_threshold'] = self.asymmetric_big_win_threshold
        if self.asymmetric_loss_penalty_weight is not None:
            effective_config['loss_penalty_weight'] = self.asymmetric_loss_penalty_weight
        if self.asymmetric_win_reward_weight is not None:
            effective_config['win_reward_weight'] = self.asymmetric_win_reward_weight

        return effective_config



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



class DropColumnPatterns(BaseEstimator, TransformerMixin):
    """
    Pipeline step that drops columns based on the patterns provided in the config, including
    support for * wildcards.

    Valid format examples: 'cluster|*', 'trading|crypto_net_gain|all_windows'
    """
    def __init__(self, drop_patterns=None, protected_columns=None):
        """
        Transformer for dropping columns based on patterns.

        Params:
        - drop_patterns (list): List of patterns to match columns for dropping.
        - protected_columns (list): List of columns to exclude from dropping.
        """
        self.drop_patterns = drop_patterns
        self.protected_columns = protected_columns
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
                self.drop_patterns, all_columns, self.protected_columns
            )
            logger.info(f"Identified {len(self.columns_to_drop)} columns to drop...")
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
