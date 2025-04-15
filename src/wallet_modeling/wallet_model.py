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
        self.y_pipeline = Pipeline([
            ('identity_y', IdentityYTransformer())
        ])

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

        # Define y for modeling cohort
        target_var = self.modeling_config['target_variable']
        y = modeling_wallet_features_df[target_var][cohort_mask]

        return X, y

    def _get_wallet_pipeline(self) -> None:
        """
        Build the wallet-specific pipeline by prepending the wallet cohort selection
        to the base pipeline steps.
        """
        # Create a combined selector that will filter X and y using the cohort DataFrame
        combined_selector = CombinedCohortSelector(cohort_df=self.modeling_wallet_features_df)
        base_pipeline = self._get_base_pipeline()
        # Concatenate the selector with the base pipeline steps
        # return Pipeline([('combined_selector', combined_selector)] + base_pipeline.steps)
        return Pipeline(base_pipeline.steps)


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

        # Validate indices match and store DataFrames
        u.assert_matching_indices(training_data_df,modeling_wallet_features_df)
        self.training_data_df = training_data_df
        self.modeling_wallet_features_df = modeling_wallet_features_df

        # Prepare data (just X, y now)
        X, y = self._prepare_data(training_data_df, modeling_wallet_features_df)

        # Apply the y pipeline (currently an identity transformer)
        # (In the future, you can add additional steps here.)
        self.y_pipeline = Pipeline([
            ('identity_y', IdentityYTransformer())
        ])
        y = self.y_pipeline.fit_transform(y)

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
#           Pipeline Steps
# -----------------------------------

class IdentityYTransformer(BaseEstimator, TransformerMixin):
    """
    A simple transformer that passes through the target variable y without modification.
    """
    def fit(self, y, X=None):
        """Fit method does nothing and returns self."""
        return self

    def transform(self, y, X=None):
        """Return y unchanged."""
        return y


# class WalletXTransformer(BaseEstimator, TransformerMixin):
#     """
#     Transformer that filters rows in X by joining with a cohort dataframe (which contains
#     the 'in_modeling_cohort' flag) and then drops that flag column.
#     """
#     def __init__(self, cohort_df: pd.DataFrame):
#         """
#         Parameters:
#         - cohort_df (pd.DataFrame): DataFrame that must contain the 'in_modeling_cohort' column,
#           indexed in the same way as X.
#         """
#         self.cohort_df = cohort_df

#     def fit(self, X: pd.DataFrame, y=None):
#         """No fitting necessary; returns self."""
#         return self

#     def transform(self, X: pd.DataFrame, y=None):
#         """
#         Join X with the cohort_df (using the index), filter rows where in_modeling_cohort == 1,
#         then drop the 'in_modeling_cohort' column.
#         """
#         u.assert_matching_indices(X,self.cohort_df)

#         # Filter to only records in the modeling cohort
#         cohort_mask = self.cohort_df['in_modeling_cohort'] == 1
#         df_filtered = X[cohort_mask].copy()

#         return df_filtered


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
        """Confirm X’s indices are in cohort_df, join on the index, filter rows where in_modeling_cohort == 1, and return X (and y) with the flag dropped."""
        # Confirm that every index in X exists in cohort_df
        if not X.index.isin(self.cohort_df.index).all():
            raise ValueError("Not all index values in X have a corresponding match in the cohort DataFrame.")

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
