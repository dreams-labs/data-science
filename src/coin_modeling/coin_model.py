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

            # Confirm no overlap between training and validation data
            if len(set(training_df.index.get_level_values('coin_epoch_start_date')).intersection(
            set(validation_df.index.get_level_values('coin_epoch_start_date')))) > 0:
                raise ValueError("Overlap found between training data and validation data epochs. "
                                "This will cause data leakage of the validation target variables.")
            # Confirm columns match
            if not np.array_equal(set(training_df.columns),set(validation_df.columns)):
                raise ValueError("Columns in training_df do not match columns in validation_df.")
            if not np.array_equal(set(modeling_coin_df.columns),set(validation_coin_df.columns)):
                raise ValueError("Columns in training target vars do not match columns in validation target vars.")

            # Prepare and store validation datasets
            self.X_validation, self.validation_target_vars_df = self._prepare_data(
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
                logger.debug(f"Updated model params with CV best params: {best_params}")
                if not self.modeling_config['grid_search_params'].get('build_post_search_model'):
                    return cv_results

        # Fit pipeline
        logger.info(f"Training coin model using data with shape: {self.X_train.shape}...")
        u.notify('startup')
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

        # Convert multiclass to binary for asymmetric loss (add this block)
        if self.modeling_config.get('asymmetric_loss', {}).get('enabled'):
            self.y_train = pd.Series((self.y_train == 2).astype(int), index=self.X_train.index)
            self.y_test = pd.Series((self.y_test == 2).astype(int), index=self.X_test.index)
            self.y_eval = pd.Series((self.y_eval == 2).astype(int), index=self.X_eval.index)


        if self.y_train.nunique() == 1:
            logger.warning(f"All values in y_train classification target var were {str(self.y_train[0])}. "
                                "Adjust thresholds to ensure both 1s and 0s.")
        if self.y_test.nunique() == 1:
            logger.warning(f"All values in y_test classification target var were {str(self.y_test[0])}. "
                                "Adjust thresholds to ensure both 1s and 0s.")
        if self.y_eval.nunique() == 1:
            logger.warning(f"All values in y_eval classification target var were {str(self.y_eval[0])}. "
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
            result['validation_target_vars_df'] = self.validation_target_vars_df
            result['y_validation'] = self.y_pipeline.transform(self.validation_target_vars_df)

            # Convert validation targets to binary for asymmetric loss
            if self.modeling_config.get('asymmetric_loss', {}).get('enabled'):
                result['y_validation'] = pd.Series(
                    (result['y_validation'] == 2).astype(int),
                    index=self.X_validation.index
                )

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

        u.notify('notify')

        return result



    # -----------------------------------
    #           Helper Methods
    # -----------------------------------

    def _prepare_data(
        self,
        training_data_df: pd.DataFrame,
        target_vars_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        def _convert_index_to_str(idx: pd.Index) -> pd.Index:
            """Convert only the 'coin_id' level to str, preserve other levels."""
            if isinstance(idx, pd.MultiIndex):
                lvl_names = idx.names
                lvl_vals = []
                for name in lvl_names:
                    vals = idx.get_level_values(name)
                    if name == 'coin_id':
                        vals = vals.astype(str)
                    lvl_vals.append(vals)
                return pd.MultiIndex.from_arrays(lvl_vals, names=lvl_names)
            else:
                return idx.astype(str)

        # Normalize indices: only cast coin_id to str, preserve other levels
        training_data_df.index = _convert_index_to_str(training_data_df.index)
        training_data_df = training_data_df.sort_index()
        target_vars_df.index = _convert_index_to_str(target_vars_df.index)
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

        # Filter based on market cap
        if self.modeling_config['market_cap_column'] == 'market_cap':
            market_cap_col = 'market_data|market_cap/last'
        elif self.modeling_config['market_cap_column'] == 'market_cap_filled':
            market_cap_col = 'market_data|market_cap/filled_last'
        else:
            raise ValueError(f"Invalid value '{self.modeling_config['market_cap_column']}' found in "
                             f"wallets_coin_config['coin_modeling']['market_cap_column']. The value must "
                             "be 'market_cap' or 'market_cap_filled'.")

        min_market_cap = self.modeling_config['min_market_cap']
        max_market_cap = self.modeling_config['max_market_cap']

        coin_training_data_df = coin_training_data_df[
            (coin_training_data_df[market_cap_col].isna())
            | (
                (coin_training_data_df[market_cap_col] >= min_market_cap)
                & (coin_training_data_df[market_cap_col] <= max_market_cap)
            )
        ]
        logger.info("Coins after market cap filters: %s", len(coin_training_data_df))

        # Align target_vars_df index with filtered coin_training_data_df
        target_vars_df = target_vars_df.reindex(coin_training_data_df.index)
        u.assert_matching_indices(coin_training_data_df, target_vars_df)

        X = coin_training_data_df.copy()
        y = target_vars_df.copy()

        return X, y

    def _prepare_grid_search_params(self, X: pd.DataFrame, base_params_override=None) -> dict:
        """
        Override to prepend 'model_pipeline__' to all keys in param_grid for CoinModel.
        """
        gs_config = super()._prepare_grid_search_params(X, base_params_override)
        gs_config['param_grid'] = {
            f"model_pipeline__{k}" if not k.startswith("model_pipeline__") else k: v
            for k, v in gs_config['param_grid'].items()
        }

        # Add target variable options into the grid search (same as WalletModel)
        param_grid_y = self.modeling_config.get('grid_search_params', {}).get('param_grid_y') or {}
        if 'target_selector__target_variable' in param_grid_y:
            target_variables = param_grid_y['target_selector__target_variable']
            gs_config['param_grid']['y_pipeline__target_selector__target_variable'] = target_variables

        # Add target variable min/max threshold options
        if 'target_selector__target_var_min_threshold' in param_grid_y:
            min_thresholds = param_grid_y['target_selector__target_var_min_threshold']
            gs_config['param_grid']['y_pipeline__target_selector__target_var_min_threshold'] = min_thresholds

        if 'target_selector__target_var_max_threshold' in param_grid_y:
            max_thresholds = param_grid_y['target_selector__target_var_max_threshold']
            gs_config['param_grid']['y_pipeline__target_selector__target_var_max_threshold'] = max_thresholds

        return gs_config
