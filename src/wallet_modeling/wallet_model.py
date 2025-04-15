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
# pylint: disable=W0201  # Attribute defined outside __init__, false positive due to inheritance

# Set up logger at the module level
logger = logging.getLogger(__name__)


# WalletModel Constructor
class WalletModel(BaseModel):
    """
    Wallet-specific model implementation.
    Extends BaseModel with wallet-specific data preparation and grid search.
    """

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
        X = training_data_df[cohort_mask].copy().copy()

        # Define y for modeling cohort
        target_var = self.modeling_config['target_variable']
        y = modeling_wallet_features_df[target_var][cohort_mask]

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

        predictions = pd.Series(
            self.pipeline.predict(self.training_data_df),
            index=self.training_data_df.index
        )

        # Verify no wallets were dropped during prediction
        if not predictions.index.equals(self.training_data_df.index):
            raise ValueError("Prediction dropped some wallet addresses")

        return predictions


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

        # Validate indices match
        u.assert_matching_indices(training_data_df,modeling_wallet_features_df)

        # Prepare data (just X, y now)
        X, y = self._prepare_data(training_data_df, modeling_wallet_features_df)

        # Do the actual train/test split in BaseModel
        self._split_data(X, y)

        # Build wallet-specific pipeline (which calls _get_base_pipeline() internally)
        # self._build_wallet_pipeline()
        self.pipeline = self._get_wallet_pipeline()

        # Fit the wallet pipeline (using the same _fit() method as in BaseModel)
        self._fit()
        result = {
            'pipeline': self.pipeline,
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'y_pred': self._predict()
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
    #          Pipeline Methods
    # -----------------------------------

    def _get_wallet_pipeline(self) -> None:
        """
        Build the wallet-specific pipeline by prepending the wallet cohort selection
        to the base pipeline steps.
        """
        # wallet_cohort_selector = WalletCohortSelector(
        #     target_variable=self.modeling_config['target_variable']
        # )
        base_pipeline = self._get_base_pipeline()

        # Combine wallet cohort selector with all the base steps
        # self.pipeline = Pipeline([('wallet_cohort_selector', wallet_cohort_selector)]
                                #   + base_pipeline.steps)
        return(Pipeline(base_pipeline.steps))


class WalletCohortSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to filter training data for wallets in the modeling cohort
    and extract the target variable.
    """
    def __init__(self, target_variable: str):
        """Initialize with target variable name."""
        self.target_variable = target_variable

    def fit(self, X: pd.DataFrame, y=None):
        """No fitting needed; return self."""
        return self

    def transform(self, X: pd.DataFrame):
        """
        Filter rows where 'in_modeling_cohort' equals 1 and split X and y.
        Assumes X contains both feature data and the 'in_modeling_cohort' flag,
        as well as the target column.
        """
        # Filter rows for the modeling cohort
        cohort_mask = X['in_modeling_cohort'] == 1
        X_cohort = X[cohort_mask].copy()
        y_cohort = X_cohort.pop(self.target_variable)
        # Optionally, you might also want to drop the cohort flag column:
        # X_cohort = X_cohort.drop(columns=['in_modeling_cohort'])
        # Return as a tuple (this may require custom handling)
        return X_cohort, y_cohort
