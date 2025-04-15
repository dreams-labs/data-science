import logging
from typing import Dict, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


# Local modules
from base_modeling.base_model import BaseModel
import utils as u

# pylint:disable=invalid-name  # X_test isn't camelcase
# pylint:disable=unused-argument  # y param needed for pipeline structure
# pylint:disable=W0201  # Attribute defined outside __init__, false positive due to inheritance

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

        # Modeling cohort and target variables
        self.modeling_wallet_features_df = None

        # Target variable pipeline
        self.y_pipeline = None



    # -----------------------------------
    #         Primary Interface
    # -----------------------------------

    def construct_wallet_model(
            self,
            training_data_df: pd.DataFrame,
            modeling_wallet_features_df: pd.DataFrame,
            return_data: bool = True
        ) -> Dict[str, Union[Pipeline, pd.DataFrame, np.ndarray]]:
        """
        Run wallet-specific modeling experiment.

        Params:
        - training_data_df (DataFrame): full training cohort feature data
        - modeling_wallet_features_df (DataFrame): Contains modeling cohort flag and target
        - return_data (bool): Whether to return train/test splits and predictions

        Returns:
        - result (dict): Contains fitted pipeline, predictions, and optional train/test data
        """
        logger.info("Preparing training data for model construction...")

        # Validate indices match and store DataFrames
        u.assert_matching_indices(training_data_df,modeling_wallet_features_df)
        self.training_data_df = training_data_df
        self.modeling_wallet_features_df = modeling_wallet_features_df

        # Prepare data
        X, y = self._prepare_data(training_data_df, modeling_wallet_features_df)

        # Build our MetaPipeline that jointly transforms X and y
        meta_pipeline = self._get_meta_pipeline()

        # Split data using your _split_data method
        self._split_data(X, y)

        # Extract target variable series
        target_var = self.modeling_config['target_variable']
        self.y_train = self.y_train[target_var] if isinstance(self.y_train, pd.DataFrame) else self.y_train
        self.y_test = self.y_test[target_var] if isinstance(self.y_test, pd.DataFrame) else self.y_test
        self.y_eval = self.y_eval[target_var] if isinstance(self.y_eval, pd.DataFrame) else self.y_eval

        # Run grid search if enabled - NOW AFTER TARGET EXTRACTION
        if self.modeling_config.get('grid_search_params', {}).get('enabled'):
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

        # Rest of the method remains the same
        meta_pipeline = self._get_meta_pipeline()
        meta_pipeline.fit(self.X_train, self.y_train, eval_set=(self.X_eval, self.y_eval))
        self.pipeline = meta_pipeline


        # Make predictions etc. as before
        self.y_pred = meta_pipeline.predict(self.X_test)
        result = {
            'pipeline': self.pipeline,
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'y_pred': self.y_pred,
        }

        # If grid search was run and no final model is requested
        if self.modeling_config.get('grid_search_params', {}).get('enabled') is True and \
        self.modeling_config.get('grid_search_params', {}).get('build_post_search_model') is False:
            return result

        # Optionally store predictions for the full training cohort
        if return_data:
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


    def _get_meta_pipeline(self) -> 'MetaPipeline':
        """
        Build and return a MetaPipeline that wraps both the x_pipeline (base pipeline)
        and the y_pipeline.
        """
        # x_pipeline: your existing base pipeline steps
        x_pipeline = self._get_base_pipeline()

        # y_pipeline: here we simply select the target column,
        # but additional steps can be added in the future.
        y_pipeline = Pipeline([
            ('target_selector', TargetVarSelector(target_variable=self.modeling_config['target_variable']))
        ])

        return MetaPipeline(x_pipeline=x_pipeline, y_pipeline=y_pipeline)


    def _get_y_pipeline(self) -> None:
        """
        Build the wallet-specific pipeline by prepending the wallet cohort selection
        to the base pipeline steps.
        """
        y_pipeline = Pipeline([
            ('target_selector', TargetVarSelector(target_variable=self.modeling_config['target_variable']))
        ])

        return y_pipeline


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
    Transformer that selects the target variable column from a DataFrame.
    """
    def __init__(self, target_variable: str):
        self.target_variable = target_variable

    def fit(self, y, X=None):
        """No fitting necessary; returns self."""
        return self

    def transform(self, y, X=None):
        """
        If y is a DataFrame, select and return the column specified by target_variable.
        Otherwise, return y unchanged.
        """
        if isinstance(y, pd.DataFrame):
            return y[self.target_variable]
        return y


class CombinedCohortSelector(BaseEstimator, TransformerMixin):
    """
    Transformer that filters both X and y based on a cohort DataFrame containing
    the 'in_modeling_cohort' flag. Assumes X and the cohort DataFrame share the same index.
    """
    def __init__(self, cohort_df: pd.DataFrame):
        """
        Parameters:
        - cohort_df (pd.DataFrame): DataFrame with 'in_modeling_cohort' flag; must be indexed like X.
        """
        self.cohort_df = cohort_df

    def fit(self, X: pd.DataFrame, y=None):
        """No fitting required; returns self."""
        return self

    def transform(self, X: pd.DataFrame, y=None):
        """
        Confirm X’s indices are in cohort_df, join on the index, filter rows where
        in_modeling_cohort == 1, and return X (and y) with the flag dropped.
        """
        # Confirm that every index in X exists in cohort_df
        if not X.index.isin(self.cohort_df.index).all():
            raise ValueError("Not all index values in X have a match in the cohort DataFrame.")

        # Perform an inner join of X with the 'in_modeling_cohort' column
        X_joined = X.join(self.cohort_df[['in_modeling_cohort']], how='inner')

        # Filter rows where in_modeling_cohort is 1
        X_filtered = X_joined[X_joined['in_modeling_cohort'] == 1].copy()

        # Drop the in_modeling_cohort column before returning
        X_filtered.drop(columns=['in_modeling_cohort'], inplace=True, errors='ignore')

        # If y is provided, align y with the filtered X by selecting matching indices
        if y is not None:
            y_filtered = y.loc[X_filtered.index].copy()
            return X_filtered, y_filtered

        return X_filtered


class MetaPipeline(BaseEstimator, TransformerMixin):
    """Meta-pipeline that applies x and y transformations then fits a regressor."""

    def __init__(self, x_pipeline: Pipeline, y_pipeline: Pipeline):
        """Initialize MetaPipeline with x_pipeline and y_pipeline."""
        self.x_pipeline = x_pipeline
        self.y_pipeline = y_pipeline
        self.regressor = None
        self.x_transformer_ = None  # will store the transformer sub-pipeline for later use

        # Create a named_steps attribute that mimics sklearn Pipeline interface
        self.named_steps = {}
        # Add steps from x_pipeline to named_steps
        for name, step in x_pipeline.named_steps.items():
            self.named_steps[name] = step

    def fit(self, X, y, eval_set=None):
        """Fit the MetaPipeline on raw X and y data, using an optional eval_set for early stopping."""
        # First, transform y using the y_pipeline
        y_trans = self.y_pipeline.fit_transform(y)

        # Create a transformer sub-pipeline (all steps except the final estimator)
        transformer = Pipeline(self.x_pipeline.steps[:-1])

        # Transform training data
        X_trans = transformer.fit_transform(X, y_trans)

        # Extract the regressor from the pipeline
        regressor_name, self.regressor = self.x_pipeline.steps[-1]

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
                eval_set=transformed_eval_set
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
            return Pipeline(self.x_pipeline.steps[key])
        else:
            # Otherwise return the specific step
            return self.x_pipeline.steps[key][1]
