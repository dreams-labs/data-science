import logging
from typing import Dict, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

# Local module imports
from base_modeling.base_model import BaseModel
import base_modeling.scorers as sco
import utils as u

# pylint:disable=invalid-name  # X_test isn't camelcase
# pylint:disable=attribute-defined-outside-init  # false positive due to inheritance
# pylint:disable=access-member-before-definition  # init params from BaseModel are tripping this


# Set up logger at the module level
logger = logging.getLogger(__name__)


class CoinModel(BaseModel):
    """
    Coin-specific model implementation.
    Extends BaseModel with coin-specific data preparation.
    """

    def __init__(self, modeling_config: dict):  # pylint:disable=useless-parent-delegation
        """
        Initialize WalletModel with configuration and wallet features DataFrame.

        Params:
        - modeling_config (dict): Configuration dictionary for modeling parameters.
        """
        # Initialize BaseModel with the given configuration
        super().__init__(modeling_config)

        self.modeling_coin_df = None
        self.X_validation = None
        self.validation_coin_df = None


    # -----------------------------------
    #         Primary Interface
    # -----------------------------------

    @u.timing_decorator(logging.MILESTONE)  # pylint: disable=no-member
    def construct_coin_model(
            self,
            training_df: pd.DataFrame,
            modeling_coin_df: pd.DataFrame,
            validation_df:   pd.DataFrame = None,
            validation_coin_df: pd.DataFrame = None,
            return_data: bool = True
        ) -> Dict[str, Union[Pipeline, pd.DataFrame, np.ndarray]]:
        """
        Run coin-specific modeling experiment.

        Params:
        - training_df (DataFrame): full training cohort feature data
        - modeling_coin_df (DataFrame): Contains modeling cohort flag and target
        - validation_df (DataFrame, optional): Feature data for external validation
        - validation_coin_df (DataFrame, optional): Target data for external validation
        - return_data (bool): Whether to return train/test splits and predictions

        Returns:
        - result (dict): Contains fitted pipeline, predictions, and optional train/test data
        """
        logger.info("Preparing training data for coin model construction...")
        u.notify('intro_2')

        # Store validation data if provided
        if validation_df is not None and validation_coin_df is not None:

            # Prepare and store validation datasets
            self.X_validation, self.validation_wallet_features_df = self._prepare_data(
                validation_df,
                validation_coin_df
            )

            # Confirm aligned indices
            u.assert_matching_indices(validation_df, validation_coin_df)

            logger.info(f"Validation data with {len(self.X_validation)} records loaded.")

        # Prepare and split data
        X, y = self._prepare_data(training_df, modeling_coin_df)
        self._split_data(X, y)

        # Build meta pipeline
        meta_pipeline = self._get_meta_pipeline()

        # Run grid search if enabled
        if self.modeling_config.get('grid_search_params', {}).get('enabled'):

            # Use validation_auc_scorer if configured
            if (self.modeling_config['model_type']=='classification'
                and self.modeling_config['grid_search_params']['classifier_scoring']
                == 'coin_validation_auc_scorer'):

                self.modeling_config['grid_search_params']['scoring'] = \
                    sco.coin_validation_auc_scorer(self)

            # Grid search
            cv_results = self._run_grid_search(self.X_train, self.y_train, pipeline=meta_pipeline)
            if cv_results.get('best_params'):
                best_params = {
                    k.replace('estimator__', ''): v
                    for k, v in cv_results['best_params'].items()
                }
                self.modeling_config['model_params'].update(best_params)
                logger.info(f"Updated model params with CV best params: {best_params}")
                if not self.modeling_config['grid_search_params'].get('build_post_search_model'):
                    return cv_results

        # Fit pipeline
        meta_pipeline.fit(
            self.X_train,
            self.y_train,
            eval_set=(self.X_eval, self.y_eval),
            verbose_estimators=self.modeling_config.get('verbose_estimators', False)
        )
        self.pipeline = meta_pipeline
        self.y_pipeline = meta_pipeline.y_pipeline

        # Transform y sets
        self.y_train = self.y_pipeline.transform(self.y_train)
        self.y_test = self.y_pipeline.transform(self.y_test)
        self.y_eval = self.y_pipeline.transform(self.y_eval)

        if self.y_train.nunique() == 1:
            raise ValueError(f"All values in y_train classification target var were {str(self.y_train[0])}. "
                                "Adjust thresholds to ensure both 1s and 0s.")
        if self.y_test.nunique() == 1:
            raise ValueError(f"All values in y_test classification target var were {str(self.y_test[0])}. "
                                "Adjust thresholds to ensure both 1s and 0s.")
        if self.y_eval.nunique() == 1:
            raise ValueError(f"All values in y_eval classification target var were {str(self.y_eval[0])}. "
                                "Adjust thresholds to ensure both 1s and 0s.")



        # Build result dict
        result = {
            'modeling_config': self.modeling_config,
            'model_type': self.modeling_config['model_type'],
            'model_id': self.model_id,
            'pipeline': self.pipeline,
        }

        # Validation predictions
        if self.X_validation is not None:
            result['X_validation'] = self.X_validation
            result['validation_wallet_features_df'] = self.validation_wallet_features_df
            result['y_validation'] = self.y_pipeline.transform(self.validation_wallet_features_df)

            # Classification predictions
            if self.modeling_config['model_type'] == 'classification':
                # Generate probabilities for the positive class
                X_val_trans = meta_pipeline.x_transformer_.transform(self.X_validation)
                probas = meta_pipeline.estimator.predict_proba(X_val_trans)
                pos_idx = list(meta_pipeline.estimator.classes_).index(1)
                proba_series = pd.Series(probas[:, pos_idx], index=self.X_validation.index)
                result['y_validation_pred_proba'] = proba_series

                # Apply configurable threshold for class prediction
                threshold = self.modeling_config.get('y_pred_threshold', 0.5)
                result['y_validation_pred'] = (proba_series >= threshold).astype(int)
            # Regression predictions
            else:
                result['y_validation_pred'] = meta_pipeline.predict(self.X_validation)


        # Test set predictions and data
        if return_data:
            self.y_pred = meta_pipeline.predict(self.X_test)
            result.update({
                'X_train': self.X_train,
                'X_test': self.X_test,
                'y_train': self.y_train,
                'y_test': self.y_test,
                'y_pred': self.y_pred
            })
            if self.modeling_config['model_type'] == 'classification':
                X_test_trans = self.pipeline.x_transformer_.transform(self.X_test)
                probas = self.pipeline.estimator.predict_proba(X_test_trans)
                pos_idx = list(self.pipeline.estimator.classes_).index(1)
                result['y_pred_proba'] = pd.Series(probas[:, pos_idx], index=self.X_test.index)

        u.notify('notify_coin_model')
        return result



    # -----------------------------------
    #           Helper Methods
    # -----------------------------------

    def _prepare_data(
        self,
        training_data_df: pd.DataFrame,
        target_vars_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        # Convert index to str (from categorical) and sort
        training_data_df.index = training_data_df.index.astype(str)
        training_data_df = training_data_df.sort_index()
        target_vars_df.index = target_vars_df.index.astype(str)
        target_vars_df = target_vars_df.sort_index()

        u.assert_matching_indices(training_data_df, target_vars_df)

        # Filter based on holdings
        logger.info("Starting coins: %s", len(training_data_df))

        coin_training_data_df = training_data_df[
            (training_data_df['all_wallets|all/all|balances/usd_balance_ending|aggregations/aggregations/count']
                >= self.modeling_config['min_cohort_wallets'])
            & (training_data_df['all_wallets|all/all|balances/usd_balance_ending|aggregations/aggregations/sum']
                 >= self.modeling_config['min_cohort_balance'])
        ]
        logger.info("Coins after balance filters: %s", len(coin_training_data_df))

        # # Filter based on market cap
        # min_market_cap = wallets_coin_config['coin_modeling']['min_market_cap']
        # max_market_cap = wallets_coin_config['coin_modeling']['max_market_cap']

        # coin_training_data_df = coin_training_data_df[
        #     (coin_training_data_df['time_series|market_data|market_cap_last'].isna())
        #     | (
        #         (coin_training_data_df['time_series|market_data|market_cap_last'] >= min_market_cap)
        #         & (coin_training_data_df['time_series|market_data|market_cap_last'] <= max_market_cap)
        #     )
        # ]
        # logger.info("Coins after market cap filters: %s", len(coin_training_data_df))


        X = training_data_df.copy()
        y = target_vars_df.copy()

        return X, y
