import logging
from typing import Dict, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

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

    def _prepare_data(self, training_data_df: pd.DataFrame,
                      modeling_cohort_target_var_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Params:
        - training_data_df (DataFrame): full training cohort feature data
        - modeling_cohort_target_var_df (DataFrame): Contains in_modeling_cohort flag and target variable
                                                for full training cohort

        Returns:
        - X (DataFrame): feature data for modeling cohort
        - y (Series): target variable for modeling cohort
        """
        # Store full training cohort for later scoring
        self.training_data_df = training_data_df.copy()

        # Validate indexes are wallet addresses and matching
        if not (training_data_df.index.name == 'wallet_address' and
                modeling_cohort_target_var_df.index.name == 'wallet_address'):
            raise ValueError("Both dataframes must have wallet_address as index")

        # Join target data to features
        modeling_df = training_data_df.join(
            modeling_cohort_target_var_df,
            how='left'
        )

        # Filter to modeling cohort for training
        modeling_cohort_mask = modeling_df['in_modeling_cohort'] == 1
        modeling_df = modeling_df[modeling_cohort_mask]

        # Separate target variable
        target_var = self.modeling_config['target_variable']
        X = modeling_df.drop([target_var, 'in_modeling_cohort'], axis=1)
        y = modeling_df[target_var]

        # Create train/test split for final model
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.modeling_config['train_test_split'],
            random_state=self.modeling_config['model_params']['random_state']
        )

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

        # Validate indexes are wallet addresses and matching
        if not self.training_data_df.index.name == 'wallet_address':
            raise ValueError("training_data_df index must be wallet_address")

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

    def construct_wallet_model(self, training_data_df: pd.DataFrame,
                            modeling_cohort_target_var_df: pd.DataFrame,
                            return_data: bool = True) -> Dict[str, Union[Pipeline, pd.DataFrame, np.ndarray]]:
        """
        Run wallet-specific modeling experiment.

        Params:
        - training_data_df (DataFrame): full training cohort feature data
        - modeling_cohort_target_var_df (DataFrame): Contains modeling cohort flag and target
        - return_data (bool): Whether to return train/test splits and predictions

        Returns:
        - result (dict): Contains fitted pipeline, predictions, and optional train/test data
        """
        logger.info("Beginning model construction...")

        # Validate all training data has targets and remove excess target rows
        if not training_data_df.index.isin(modeling_cohort_target_var_df.index).all():
            raise ValueError(
                "Some training data points are missing target values. "
                f"Found {(~training_data_df.index.isin(modeling_cohort_target_var_df.index)).sum()} "
                "training rows without targets."
            )

        # Filter target df to only include rows with training data
        modeling_cohort_target_var_df = modeling_cohort_target_var_df[
            modeling_cohort_target_var_df.index.isin(training_data_df.index)
        ]

        # Run base experiment
        self._prepare_data(training_data_df, modeling_cohort_target_var_df)
        result = super().construct_base_model(return_data)

        # If grid search is run and doesn't request a model, don't make predictions
        if self.modeling_config.get('grid_search_params', {}).get('enabled') is True and \
        self.modeling_config.get('grid_search_params', {}).get('build_post_search_model') is False:
            return result

        # Add wallet-specific predictions if requested
        if return_data:
            training_cohort_pred = self._predict_training_cohort()
            target_var = self.modeling_config['target_variable']
            full_cohort_actuals = modeling_cohort_target_var_df[target_var]

            result.update({
                'training_cohort_pred': training_cohort_pred,
                'training_cohort_actuals': full_cohort_actuals
            })

        u.notify('notify')

        return result
