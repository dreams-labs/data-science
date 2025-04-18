import logging
from typing import Dict, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error, r2_score

# Local modules
from base_modeling.base_model import BaseModel
import wallet_insights.wallet_validation_analysis as wiva
import utils as u

# pylint:disable=invalid-name  # X_test isn't camelcase
# pylint:disable=unused-argument  # y param needed for pipeline structure
# pylint:disable=W0201  # Attribute defined outside __init__, false positive due to inheritance
# pylint:disable=access-member-before-definition  # init params from BaseModel are tripping this


# Set up logger at the module level
logger = logging.getLogger(__name__)


# WalletModel Constructor
class WalletModel(BaseModel):
    """
    Wallet-specific model implementation.
    Extends BaseModel with wallet-specific data preparation and grid search.
    """

    def __init__(self, modeling_config: dict):
        """
        Initialize WalletModel with configuration and wallet features DataFrame.

        Params:
        - modeling_config (dict): Configuration dictionary for modeling parameters.
        """
        # Initialize BaseModel with the given configuration
        super().__init__(modeling_config)

        # y data for modeling period; includes Modeling cohort and target variables
        self.modeling_wallet_features_df = None

        # Target variable pipeline
        self.y_pipeline = None

        # Validation set objects
        self.X_validation = None
        self.validation_wallet_features_df = None

    # -----------------------------------
    #         Primary Interface
    # -----------------------------------

    def construct_wallet_model(
            self,
            training_data_df: pd.DataFrame,
            modeling_wallet_features_df: pd.DataFrame,
            validation_data_df: pd.DataFrame = None,
            validation_wallet_features_df: pd.DataFrame = None,
            return_data: bool = True
        ) -> Dict[str, Union[Pipeline, pd.DataFrame, np.ndarray]]:
        """
        Run wallet-specific modeling experiment.

        Params:
        - training_data_df (DataFrame): full training cohort feature data
        - modeling_wallet_features_df (DataFrame): Contains modeling cohort flag and target
        - validation_data_df (DataFrame, optional): Feature data for external validation
        - validation_wallet_features_df (DataFrame, optional): Target data for external validation
        - return_data (bool): Whether to return train/test splits and predictions

        Returns:
        - result (dict): Contains fitted pipeline, predictions, and optional train/test data
        """
        logger.info("Preparing training data for model construction...")

        # Validate indices match and store DataFrames
        u.assert_matching_indices(training_data_df, modeling_wallet_features_df)
        self.training_data_df = training_data_df
        self.modeling_wallet_features_df = modeling_wallet_features_df

        # Store validation data if provided
        if validation_data_df is not None and validation_wallet_features_df is not None:
            u.assert_matching_indices(validation_data_df, validation_wallet_features_df)
            self.X_validation = validation_data_df
            self.validation_wallet_features_df = validation_wallet_features_df
            logger.info(f"Validation data set with {len(validation_data_df)} records loaded.")

        # Prepare data
        X, y = self._prepare_data(training_data_df, modeling_wallet_features_df)

        # Split data
        self._split_data(X, y)

        # Build meta pipeline
        meta_pipeline = self._get_meta_pipeline()

        # Run grid search if enabled
        if self.modeling_config.get('grid_search_params', {}).get('enabled'):
            cv_results = self._run_grid_search(self.X_train, self.y_train, pipeline=meta_pipeline)

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

        meta_pipeline.fit(self.X_train, self.y_train, eval_set=(self.X_eval, self.y_eval))
        self.pipeline = meta_pipeline
        self.y_pipeline = meta_pipeline.y_pipeline  # <-- add this

        # Update target variables to be 1D Series using the y_pipeline
        self.y_train = self.y_pipeline.transform(self.y_train)
        self.y_test = self.y_pipeline.transform(self.y_test)
        self.y_eval = self.y_pipeline.transform(self.y_eval)

        # Prepare result dictionary
        result = {
            'pipeline': self.pipeline,
        }

        # Add train/test data if requested
        if return_data:
            # Make predictions on test set
            self.y_pred = meta_pipeline.predict(self.X_test)

            result.update({
                'X_train': self.X_train,
                'X_test': self.X_test,
                'y_train': self.y_train,
                'y_test': self.y_test,
                'y_pred': self.y_pred
            })

            # Optionally add predictions for full training cohort
            training_cohort_pred = self._predict_training_cohort()
            target_var = self.modeling_config['target_variable']
            full_cohort_actuals = modeling_wallet_features_df[target_var]

            result.update({
                'training_cohort_pred': training_cohort_pred,
                'training_cohort_actuals': full_cohort_actuals
            })

        u.notify('notify')
        return result


    # -----------------------------------
    #           Helper Methods
    # -----------------------------------

    def _prepare_data(
            self,
            training_data_df: pd.DataFrame,
            modeling_wallet_features_df: pd.DataFrame
        ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare wallet-specific data for modeling. Returns features and target only.

        Params:
        - training_data_df (DataFrame): full training cohort feature data
        - modeling_wallet_features_df (DataFrame): Contains in_modeling_cohort flag and target variable

        Returns:
        - X (DataFrame): feature data for modeling cohort
        - y (Series): target variable for modeling cohort
        """
        # Store full training cohort for later scoring
        self.training_data_df = training_data_df.copy()

        # Filter to modeling cohort
        cohort_mask = modeling_wallet_features_df['in_modeling_cohort'] == 1

        # Define X
        X = training_data_df[cohort_mask].copy()

        # Define y
        y = modeling_wallet_features_df[cohort_mask].copy()

        return X, y


    def _get_meta_pipeline(self) -> Pipeline:
        """
        Return a single Pipeline that first applies y transformations,
        then the usual feature+regressor steps. Step names remain
        exactly ['target_selector', 'feature_selector', 'drop_columns', 'regressor'].
        """
        # Get the steps from the y_pipeline
        y_steps = self._get_y_pipeline()
        # Get the steps from the base pipeline (feature_selector, drop_columns, regressor)
        model_steps = self._get_base_pipeline()

        # Concatenate them into one pipeline
        combined_steps = MetaPipeline(y_steps, model_steps)

        return combined_steps


    def _get_y_pipeline(self) -> None:
        """
        Build the wallet-specific pipeline by prepending the wallet cohort selection
        to the base pipeline steps.
        """
        target_var = self.modeling_config.get('model_params', {}).get(
            'target_selector__target_variable',
            self.modeling_config['target_variable']
        )

        y_pipeline = Pipeline([
            ('target_selector', TargetVarSelector(target_variable=target_var))
        ])

        return y_pipeline


    def _prepare_grid_search_params(self, X: pd.DataFrame, base_params_override=None) -> dict:
        """
        Override to prepend 'model_pipeline__' to all keys in param_grid.
        """
        gs_config = super()._prepare_grid_search_params(X, base_params_override)
        # Prepend "model_pipeline__" to each key if it isn’t already prefixed
        gs_config['param_grid'] = {
            f"model_pipeline__{k}" if not k.startswith("model_pipeline__") else k: v
            for k, v in gs_config['param_grid'].items()
        }

        # Add target variable options into the grid search.
        if 'target_selector__target_variable' in self.modeling_config['grid_search_params']['param_grid_y']:
            target_variables = self.modeling_config['grid_search_params']['param_grid_y']['target_selector__target_variable']  # pylint:disable=line-too-long
            gs_config['param_grid']['y_pipeline__target_selector__target_variable'] = target_variables

        return gs_config


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
            gs_config['search_config']['scoring'] = custom_r2_scorer
        elif scoring_param == 'custom_neg_rmse_scorer':
            gs_config['search_config']['scoring'] = custom_neg_rmse_scorer
        elif scoring_param == 'validation_r2_scorer':
            # Ensure validation data is available
            if self.X_validation is None or self.validation_wallet_features_df is None:
                raise ValueError("Validation data required for validation_r2_scorer")

            # Create the custom scorer with access to validation data
            gs_config['search_config']['scoring'] = validation_r2_scorer(self)
        else:
            raise ValueError(f"Invalid scoring metric '{scoring_param}' found in grid_search_params.")

        # Generate pipeline
        cv_pipeline = pipeline if pipeline is not None else self._get_model_pipeline(gs_config['base_model_params'])

        # Random search with pipeline
        self.random_search = RandomizedSearchCV(
            cv_pipeline,
            gs_config['param_grid'],
            **gs_config['search_config']
        )

        # Always pass the eval_set for early stopping
        self.random_search.fit(X, y, eval_set=(self.X_eval, self.y_eval))

        logger.info("Grid search complete. Best score: %f", -self.random_search.best_score_)
        u.notify('synth_magic')

        return {
            'best_params': {
                k: (self._convert_min_child_weight_to_pct(X, v) if k == 'regressor__min_child_weight' else v)
                for k, v in self.random_search.best_params_.items()
            },
            'best_score': -self.random_search.best_score_,
            'cv_results': self.random_search.cv_results_
        }


    def _predict_training_cohort(self) -> pd.Series:
        """
        Make predictions on the full training cohort.

        Returns:
        - predictions (Series): predicted values for full training cohort
        - actuals (Series): actual target values for full training cohort
        """
        if self.training_data_df is None:
            raise ValueError("No training cohort data found. Run prepare_data first.")

        predictions = pd.Series(
            self.pipeline.predict(self.training_data_df),
            index=self.training_data_df.index
        )

        # Verify no wallets were dropped during prediction
        if not predictions.index.equals(self.training_data_df.index):
            raise ValueError("Prediction dropped some wallet addresses")

        return predictions



# -----------------------------------
#           Pipeline Steps
# -----------------------------------

class TargetVarSelector(BaseEstimator, TransformerMixin):
    """
    Transformer that extracts the target variable from the input.

    If y is a DataFrame, returns the column specified by target_variable as a Series.
    If y is already a Series, returns it unchanged.

    This centralizes the target extraction logic so that grid search can update the target
    variable parameter without interference from pre-extraction.
    """
    def __init__(self, target_variable: str):
        self.target_variable = target_variable

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
        If y is a DataFrame, extract the target column specified by target_variable.
        Return a 1D Series even if the extraction yields a single-column DataFrame.
        If y is already a Series, return it unchanged.
        """
        if isinstance(y, pd.DataFrame):
            result = y[self.target_variable]
            # Ensure result is a Series (squeeze if it is still a DataFrame)
            if isinstance(result, pd.DataFrame):
                result = result.squeeze()
            return result
        return y


class MetaPipeline(BaseEstimator, TransformerMixin):
    """Meta-pipeline that applies x and y transformations then fits a regressor."""

    def __init__(self, y_pipeline: Pipeline, model_pipeline: Pipeline):
        """Initialize MetaPipeline with y_pipelin and model_pipeline."""
        self.y_pipeline = y_pipeline
        self.model_pipeline = model_pipeline
        self.regressor = None
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

    def fit(self, X, y, eval_set=None):
        """Fit the MetaPipeline on raw X and y data, using an optional eval_set for early stopping."""
        # First, transform y using the y_pipeline
        y_trans = self.y_pipeline.fit_transform(y)

        # Create a transformer sub-pipeline (all steps except the final estimator)
        transformer = Pipeline(self.model_pipeline.steps[:-1])

        # Transform training data
        X_trans = transformer.fit_transform(X, y_trans)

        # Extract the regressor from the pipeline
        regressor_name, self.regressor = self.model_pipeline.steps[-1]

        # If evaluation set is provided, transform it and use for early stopping
        if eval_set is not None:
            X_eval, y_eval = eval_set
            y_eval_trans = self.y_pipeline.transform(y_eval)
            X_eval_trans = transformer.transform(X_eval)

            # Create eval_set in the format expected by XGBoost
            transformed_eval_set = [(X_trans, y_trans), (X_eval_trans, y_eval_trans)]

            # Fit with early stopping using the transformed eval set
            self.regressor.fit(
                X_trans,
                y_trans,
                eval_set=transformed_eval_set,
                verbose=False
            )
        else:
            # Regular fit without early stopping if no eval set provided
            self.regressor.fit(X_trans, y_trans)

        # Store the transformer sub-pipeline for use during prediction
        self.x_transformer_ = transformer

        # Update named_steps with the fitted regressor
        self.named_steps[regressor_name] = self.regressor

        return self

    def predict(self, X):
        """Predict using the fitted regressor on transformed X."""
        X_trans = self.x_transformer_.transform(X)
        return self.regressor.predict(X_trans)

    def score(self, X, y):
        """Return the regressor's score on transformed X and y."""
        X_trans = self.x_transformer_.transform(X)
        y_trans = self.y_pipeline.transform(y)
        return self.regressor.score(X_trans, y_trans)

    # Add methods to make it behave more like a sklearn Pipeline
    def __getitem__(self, key):
        """Support indexing like a regular pipeline."""
        if isinstance(key, slice):
            # If it's a slice, return a new Pipeline with the sliced steps
            return Pipeline(self.model_pipeline.steps[key])
        else:
            # Otherwise return the specific step
            return self.model_pipeline.steps[key][1]




# -----------------------------------
#          Scorer Functions
# -----------------------------------

def custom_neg_rmse_scorer(estimator, X, y):
    """
    Custom scorer that transforms y before computing RMSE.
    Applies the estimator's y_pipeline to extract the proper target.
    Returns negative RMSE for grid search scoring.
    """
    y_trans = estimator.y_pipeline.transform(y)
    y_pred = estimator.predict(X)
    rmse = root_mean_squared_error(y_trans, y_pred)
    return -rmse


def custom_r2_scorer(estimator, X, y):
    """
    Custom scorer for R² that first applies the pipeline's y transformation.

    Parameters:
      estimator: The fitted MetaPipeline, which includes a y_pipeline.
      X (DataFrame or array): Feature data.
      y (DataFrame or array): The raw target data.

    Returns:
      R² score computed on the transformed target and predictions.
    """
    y_trans = estimator.y_pipeline.transform(y)
    y_pred = estimator.predict(X)
    return r2_score(y_trans, y_pred)


def validation_r2_scorer(wallet_model):
    """
    Factory function that returns a custom scorer using validation data.

    Params:
    - wallet_model: WalletModel instance containing validation data

    Returns:
    - scorer function compatible with scikit-learn
    """
    def scorer(estimator, X=None, y=None):
        """Score using the validation data instead of provided X and y"""
        if wallet_model.X_validation is None or wallet_model.validation_wallet_features_df is None:
            raise ValueError("Validation data not set in wallet_model")

        # Transform y using the pipeline
        y_trans = estimator.y_pipeline.transform(wallet_model.validation_wallet_features_df)

        # Get predictions
        y_pred = estimator.predict(wallet_model.X_validation)

        # Calculate and return R2 score
        return r2_score(y_trans, y_pred)

    return scorer
