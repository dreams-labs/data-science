import logging
from typing import Dict, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

# Local modules
from base_modeling.base_model import BaseModel
import base_modeling.scorers as sco
import utils as u

# pylint:disable=invalid-name  # X_test isn't camelcase
# pylint:disable=attribute-defined-outside-init  # false positive due to inheritance
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

        # Validation set objects
        self.X_validation = None
        self.validation_wallet_features_df = None


    # -----------------------------------
    #         Primary Interface
    # -----------------------------------

    @u.timing_decorator(logging.MILESTONE)  # pylint: disable=no-member
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
        u.notify('intro_2')

        # Validate indices match and store DataFrames
        u.assert_matching_indices(training_data_df, modeling_wallet_features_df)
        self.training_data_df = training_data_df
        self.modeling_wallet_features_df = modeling_wallet_features_df

        # Store validation data if provided
        if validation_data_df is not None and validation_wallet_features_df is not None:
            u.assert_matching_indices(validation_data_df, validation_wallet_features_df)
            self.X_validation = validation_data_df
            self.validation_wallet_features_df = validation_wallet_features_df
            logger.info(f"Validation data with {len(validation_data_df)} records loaded.")

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
                    k.replace('estimator__', ''): v
                    for k, v in cv_results['best_params'].items()
                }
                self.modeling_config['model_params'].update(best_params)
                logger.info(f"Updated model params with CV best params: {best_params}")

                # Return the search results without building a model if configured to
                if not self.modeling_config.get('grid_search_params', {}).get('build_post_search_model'):
                    return cv_results

        # Log training start and play startup notification
        logger.info(f"Training wallet model using data with shape: {self.X_train.shape}...")
        u.notify('startup')
        meta_pipeline.fit(
            self.X_train,
            self.y_train,
            eval_set=(self.X_eval, self.y_eval),
            verbose_estimators=self.modeling_config.get('verbose_estimators', False)
        )
        # Log training completion
        logger.info("Wallet model training completed.")
        self.pipeline = meta_pipeline
        self.y_pipeline = meta_pipeline.y_pipeline

        # Update target variables to be 1D Series using the y_pipeline
        self.y_train = self.y_pipeline.transform(self.y_train)
        self.y_test = self.y_pipeline.transform(self.y_test)
        self.y_eval = self.y_pipeline.transform(self.y_eval)

        # Prepare result dictionary
        result = {
            'modeling_config': self.modeling_config,
            'model_type': self.modeling_config['model_type'],
            'model_id': self.model_id,
            'pipeline': self.pipeline,
        }

        # Store validation datasets and predictions if applicable
        if self.X_validation is not None:
            result['X_validation'] = self.X_validation
            result['validation_wallet_features_df'] = self.validation_wallet_features_df
            result['y_validation'] = self.y_pipeline.transform(self.validation_wallet_features_df)

            if self.modeling_config['model_type'] == 'classification':
                # get positive-class probabilities directly from pipeline
                proba = meta_pipeline.predict_proba(self.X_validation)
                pos_idx = list(meta_pipeline.named_steps['estimator'].classes_).index(1)
                proba_series = pd.Series(proba[:, pos_idx], index=self.X_validation.index)

                # Apply configurable threshold for class prediction
                threshold = self.modeling_config.get('y_pred_threshold', 0.5)
                result['y_validation_pred_proba'] = proba_series
                result['y_validation_pred']       = (proba_series >= threshold).astype(int)
            else:
                result['y_validation_pred'] = meta_pipeline.predict(self.X_validation)

        # Add train/test data if requested
        if return_data:
            # Make predictions on test set
            if self.modeling_config['model_type'] == 'classification':
                # get positive-class probabilities and apply threshold
                proba = meta_pipeline.predict_proba(self.X_test)
                pos_idx = list(meta_pipeline.named_steps['estimator'].classes_).index(1)
                proba_series = pd.Series(proba[:, pos_idx], index=self.X_test.index)

                # Apply configurable threshold for class prediction
                threshold = self.modeling_config.get('y_pred_threshold', 0.5)
                self.y_pred = (proba_series >= threshold).astype(int)
            else:
                self.y_pred = meta_pipeline.predict(self.X_test)


            result.update({
                'X_train': self.X_train,
                'X_test': self.X_test,
                'y_train': self.y_train,
                'y_test': self.y_test,
                'y_pred': self.y_pred
            })

            # Include prediction probabilities for classification models
            if self.modeling_config['model_type'] == 'classification':
                # Transform test features for probability prediction
                X_test_trans = self.pipeline.x_transformer_.transform(self.X_test)
                probas = self.pipeline.estimator.predict_proba(X_test_trans)
                pos_idx = list(self.pipeline.estimator.classes_).index(1)
                result['y_pred_proba'] = pd.Series(probas[:, pos_idx], index=self.X_test.index)

            # Optionally add predictions for full training cohort
            training_cohort_pred = self._predict_training_cohort()
            target_var = self.modeling_config['target_variable']
            full_cohort_actuals = modeling_wallet_features_df[target_var]
            training_cohort_actuals = full_cohort_actuals.loc[training_cohort_pred.index]

            # update the result with all three series
            result.update({
                'training_cohort_pred':     training_cohort_pred,      # your preds for the training cohort
                'training_cohort_actuals':  training_cohort_actuals,   # matching actuals
                'full_cohort_actuals':      full_cohort_actuals        # unfiltered full‐cohort actuals
            })

        u.notify('notify')
        return result


    # -----------------------------------
    #           Helper Methods
    # -----------------------------------

    @u.timing_decorator
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
        if self.modeling_config['assign_epochs_to_addresses']:
            # Select 1 epoch for each wallet address
            training_data_df, modeling_wallet_features_df = self._assign_epoch_to_wallets(
                training_data_df,
                modeling_wallet_features_df,
                random_state = self.modeling_config['model_params']['random_state']
            )

        # Store full training cohort for later scoring
        self.training_data_df = training_data_df.copy()

        # Identify modeling cohort
        cohort_mask = (
            (modeling_wallet_features_df['max_investment'] >= self.modeling_config['modeling_min_investment']) &
            (modeling_wallet_features_df['unique_coins_traded'] >= self.modeling_config['modeling_min_coins_traded'])
        )
        logger.milestone("Defined modeling cohort as %.1f%% (%s/%s) wallets.",
            cohort_mask.sum()/len(modeling_wallet_features_df)*100,
            cohort_mask.sum(),
            len(modeling_wallet_features_df)
        )

        # Define X and y
        X = training_data_df[cohort_mask].copy()
        y = modeling_wallet_features_df[cohort_mask].copy()

        return X, y


    def _assign_epoch_to_wallets(
        self,
        training_data_df: pd.DataFrame,
        modeling_wallet_features_df: pd.DataFrame,
        random_state: int = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Selects a single epoch for any given wallet address. This avoids data leakage
        when the target variable for one time window is exposed as a feature for another.

        Params:
        - training_data_df (DataFrame): multiindexed by ['wallet_address','epoch_start_date']
        - modeling_wallet_features_df  (DataFrame): same index as training_data_df

        Returns:
        - sampled_training_data_df: one random epoch per wallet
        - sampled_modeling_wallet_features_df:   matching rows from modeling_wallet_features_df
        """
        wallet_addresses = training_data_df.index.get_level_values('wallet_address').to_numpy()
        random_generator = np.random.default_rng(random_state)
        random_scores = random_generator.random(len(wallet_addresses))

        # Sort by wallets then random scores to group addresses while preserving randomization
        sorted_indices = np.lexsort((random_scores, wallet_addresses))
        _, unique_wallet_positions = np.unique(wallet_addresses[sorted_indices], return_index=True)
        # Select exactly one random epoch per wallet address
        selected_positions = sorted_indices[unique_wallet_positions]
        selected_multiindices = training_data_df.index[selected_positions]

        # Create df of sampled records
        sampled_training_data_df = training_data_df.loc[selected_multiindices]
        sampled_modeling_wallet_features_df = modeling_wallet_features_df.loc[selected_multiindices]
        u.assert_matching_indices(sampled_training_data_df,sampled_modeling_wallet_features_df)

        return sampled_training_data_df,sampled_modeling_wallet_features_df


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
        param_grid_y = self.modeling_config.get('grid_search_params', {}).get('param_grid_y') or {}
        if 'target_selector__target_variable' in param_grid_y:
            target_variables = param_grid_y['target_selector__target_variable']
            gs_config['param_grid']['y_pipeline__target_selector__target_variable'] = target_variables

        # Add target variable classification threshold options
        if 'target_selector__target_var_class_threshold' in param_grid_y:
            thresholds = param_grid_y['target_selector__target_var_class_threshold']
            gs_config['param_grid']['y_pipeline__target_selector__target_var_class_threshold'] = thresholds

        # Confirm there are multiple configurations
        if not any(isinstance(value, list) and len(value) > 1 for value in gs_config['param_grid'].values()):
            raise ValueError("Grid search param grid only contains one scenario. "
                             "Add more scenarios to run grid search.")

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
            gs_config['search_config']['scoring'] = sco.custom_r2_scorer

        elif scoring_param == 'custom_neg_rmse_scorer':
            gs_config['search_config']['scoring'] = sco.custom_neg_rmse_scorer

        elif scoring_param == 'validation_r2_scorer':
            # Ensure validation data is available
            if self.X_validation is None or self.validation_wallet_features_df is None:
                raise ValueError("Validation data required for validation_r2_scorer")
            gs_config['search_config']['scoring'] = sco.validation_r2_scorer(self)

        elif scoring_param == 'validation_auc_scorer':
            # Ensure validation data is available
            if self.X_validation is None or self.validation_wallet_features_df is None:
                raise ValueError("Validation data required for validation_auc_scorer")
            gs_config['search_config']['scoring'] = sco.validation_auc_scorer(self)

        elif scoring_param == 'validation_top_return_scorer':
            # read your desired top_pct from config, e.g. self.modeling_config['grid_search_params']['top_pct']
            top_pct = self.modeling_config['grid_search_params'].get('top_pct', 0.05)
            gs_config['search_config']['scoring'] = sco.validation_top_return_scorer(self, top_pct)

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
