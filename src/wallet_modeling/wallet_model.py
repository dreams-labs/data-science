"""
wallet_model.py
===============

Business logic
--------------
Goal - Train machine learning models to predict crypto wallet behavior
 and performance, enabling identification of skilled traders for downstream
 investment and analysis workflows.

Inputs
- Wide, multi-indexed feature matrix keyed on wallet_address / epoch_start_date
- Target-variable DataFrame with modeling-cohort flags + returns
- YAML configs (training bounds, feature drops, asymmetric-loss rules, etc.)

Flow
1.  Data validation + cohort selection (_prepare_data)
2.  Optional epoch de-duplication per wallet (_assign_epoch_to_wallets)
3.  Train/test/eval split with asymmetric-loss handling
4.  Meta-pipeline build (X transformer → estimator)
    Optional grid-search via scikit-learn
5.  Model training and validation with early stopping
6.  Escape hatch - if export_s3_training_data.enabled, skip training and
    export the exact X/y parquet splits (plus metadata JSON) for offline runs.

Downstream orchestrators
------------------------
- RegressorEvaluator/ClassifierEvaluator - for metrics and visualization
- wallet_temporal_searcher - multi-date grid search & model comparison
- WalletModelingOrchestrator - creates multiple models for different epochs
- Investment workflows - use trained models for live scoring and signals

Utilities
---------
- Grid-search param expansion with safety checks on asymmetric-loss conflicts
- S3 export capabilities for external training workflows
- Full-cohort prediction helpers for downstream scoring applications
- Support for both binary classification and regression with asymmetric loss

External deps: BaseModel, utils (logging, timing, config helpers).
"""
import logging
from typing import Dict, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve

# Local modules
from base_modeling.base_model import BaseModel
import wallet_modeling.s3_exporter as se
import utils as u

# pylint:disable=invalid-name  # X_test isn't camelcase
# pylint:disable=attribute-defined-outside-init  # false positive due to inheritance
# pylint:disable=access-member-before-definition  # init params from BaseModel are tripping this


# Set up logger at the module level
logger = logging.getLogger(__name__)


# WalletModel Constructor
class WalletModel(BaseModel):
    """
    Wallet-specific modeling engine (extends **BaseModel**).

    Technical overview
    ------------------
    * **Meta-pipeline** – builds an sklearn.Pipeline that wraps:
        1.  x_transformer_ (feature scaling / drops)
        2.  estimator      (XGBoost-style tree model)
        3.  y_pipeline     (target selector, optional asymmetric-loss mapper)
    * **Key public method**
        - construct_wallet_model(...) – prepares data, runs optional
          grid-search, fits the pipeline, and returns a results dict.
    * **Escape modes**
        - export_s3_training_data.enabled → bypass training, dump parquet + JSON.
    * **Important helpers**
        - _prepare_data                – cohort mask + X / y extraction
        - _assign_epoch_to_wallets     – one epoch per wallet (anti-leakage)
        - _prepare_grid_search_params  – prefixes params + validates combos
        - _export_s3_training_data     – robust parquet export, dev sampling,
                                           NaN/Inf-safe metadata writer
        - _predict_training_cohort     – full-cohort inference for later scoring
    * **Attributes after fit**
        - pipeline, model_id, train/test/eval splits, y_pred, etc.

    Orchestrator usage
    ------------------
    Typically invoked by:
    * wallet_temporal_searcher  – iterates models across dates
    * WalletInvestingOrchestrator – loads the fitted pipeline to
      generate live investing signals

    Supports both **classification** and **regression**; asymmetric-loss logic
    adds a third “big loss” class internally and re-maps to binary later.
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
        self.wallet_target_vars_df = None

        # Validation set objects
        self.X_validation = None
        self.validation_target_vars_df = None


    # -----------------------------------
    #         Primary Interface
    # -----------------------------------

    @u.timing_decorator(logging.MILESTONE)  # pylint: disable=no-member
    def construct_wallet_model(
            self,
            training_data_df: pd.DataFrame,
            wallet_target_vars_df: pd.DataFrame,
            validation_data_df: pd.DataFrame = None,
            validation_target_vars_df: pd.DataFrame = None,
            return_data: bool = True
        ) -> Dict[str, Union[Pipeline, pd.DataFrame, np.ndarray]]:
        """
        Run wallet-specific modeling experiment.

        Params:
        - training_data_df (DataFrame): full training cohort feature data
        - wallet_target_vars_df (DataFrame): Contains modeling cohort flag and target
        - validation_data_df (DataFrame, optional): Feature data for external validation
        - validation_target_vars_df (DataFrame, optional): Target data for external validation
        - return_data (bool): Whether to return train/test splits and predictions

        Returns:
        - result (dict): Contains fitted pipeline, predictions, and optional train/test data

        Data Split Usage Summary
        -----------------------
        X_train/y_train: Primary training data for model fitting
        X_eval/y_eval: Early stopping validation during XGBoost training (prevents overfitting)
        X_test/y_test: Hold-out test set for final model evaluation (traditional ML validation)
        X_validation/y_validation: Future time period data for realistic performance assessment

        Key Interactions:
        The Test set ML metrics (accuracy, R², etc.) are based on data from the same period
         as the Train set.
        The Validation set metrics are based on data from the future period just after the
         base y_train period ends. The Validation set represents actual future data the model
         would see in production, and Validation metrics are primary indicators for model performance
         in a real world scenario.
        """
        logger.info("Preparing training data for model construction...")
        u.notify('intro_2')

        # Sort columns to ensure feature windows align
        training_data_df = training_data_df.sort_index(axis=1)

        # Validate indices match and store DataFrames
        u.assert_matching_indices(training_data_df, wallet_target_vars_df)
        self.training_data_df = training_data_df
        self.wallet_target_vars_df = wallet_target_vars_df

        # Store validation data if provided
        if validation_data_df is not None and validation_target_vars_df is not None:
            validation_data_df = validation_data_df.sort_index(axis=1)
            u.assert_matching_indices(validation_data_df, validation_target_vars_df)
            self.X_validation = validation_data_df
            self.validation_target_vars_df = validation_target_vars_df

            # Confirm columns match after sorting
            u.validate_column_consistency(self.training_data_df, validation_data_df)

            logger.info(f"Validation data with {len(validation_data_df)} records loaded.")


        # Prepare data for training cohort
        X, y = self._prepare_data(training_data_df, wallet_target_vars_df)

        # Validation cohort – same transformation
        if self.X_validation is not None:
            self.X_validation, self.validation_target_vars_df = self._prepare_data(
            validation_data_df,
            validation_target_vars_df
        )

        # Split data
        self._split_data(X, y)

        # Build meta pipeline
        meta_pipeline = self._get_meta_pipeline()

        # ESCAPE: Skip model construction and export training datasets to S3 if configured
        export_config = self.modeling_config.get('export_s3_training_data', {})
        if export_config.get('enabled', False):
            logger.warning(
                "export_s3_training_data enabled — skipping model construction and exporting datasets..."
            )
            return se.export_s3_training_data(
                export_config=export_config,
                model_id=self.model_id,
                target_var=self.modeling_config['target_variable'],
                X_train=self.X_train,
                X_test=self.X_test,
                X_eval=self.X_eval,
                y_train=self.y_train,
                y_test=self.y_test,
                y_eval=self.y_eval,
                meta_pipeline=meta_pipeline,
                asymmetric_loss_enabled=self.asymmetric_loss_enabled,
                X_validation=self.X_validation,
                validation_target_vars_df=self.validation_target_vars_df
            )

        # Asymmetric loss is slow so broadcast a warning
        if self.modeling_config['asymmetric_loss'].get('enabled',False):
            logger.warning("Beginning extended training with asymmetric loss target variables...")

        # Run grid search if enabled
        if self.modeling_config.get('grid_search_params', {}).get('enabled'):
            cv_results = self._run_grid_search(self.X_train, self.y_train, pipeline=meta_pipeline)

            if cv_results.get('best_params'):
                best_params = {
                    k.replace('estimator__', ''): v
                    for k, v in cv_results['best_params'].items()
                }
                self.modeling_config['model_params'].update(best_params)
                logger.debug(f"Updated model params with CV best params: {best_params}")

                # Return the search results without building a model if configured to
                if not self.modeling_config.get('grid_search_params', {}).get('build_post_search_model'):
                    return cv_results

        # Log training start and play startup notification
        logger.milestone(f"Training wallet model using data with shape: {self.X_train.shape}...")
        u.notify('startup')

        meta_pipeline.fit(
            self.X_train,
            self.y_train,
            eval_set=(self.X_eval, self.y_eval),
            verbose_estimators=self.modeling_config.get('verbose_estimators', False),
            modeling_config=self.modeling_config  # Pass config for asymmetric loss
        )

        # Log training completion
        logger.info("Wallet model training completed.")
        self.pipeline = meta_pipeline
        self.y_pipeline = meta_pipeline.y_pipeline

        # Prepare result dictionary
        result = {
            'modeling_config': self.modeling_config,
            'model_type': self.modeling_config['model_type'],
            'model_id': self.model_id,
            'pipeline': self.pipeline,
        }

        # Update target variables to be 1D Series using the y_pipeline
        self.y_train = self.y_pipeline.transform(self.y_train)
        self.y_test  = self.y_pipeline.transform(self.y_test)
        self.y_eval  = self.y_pipeline.transform(self.y_eval)

        # Convert multi-class labels to binary for asymmetric loss
        if self.asymmetric_loss_enabled:
            self.y_train = pd.Series((self.y_train == 2).astype(int), index=self.X_train.index)
            self.y_test  = pd.Series((self.y_test == 2).astype(int), index=self.X_test.index)
            self.y_eval  = pd.Series((self.y_eval == 2).astype(int), index=self.X_eval.index)

        # Store validation datasets and predictions if applicable
        if self.X_validation is None or self.X_validation.empty:
            logger.warning("Validation dataset is empty – skipping validation metrics.")
            self.X_validation = None
            self.validation_target_vars_df = None

        if self.X_validation is not None:
            result['X_validation'] = self.X_validation
            result['validation_target_vars_df'] = self.validation_target_vars_df
            result['y_validation'] = self.y_pipeline.transform(self.validation_target_vars_df)

            # Also convert validation multiclass labels if present
            if self.asymmetric_loss_enabled:
                result['y_validation'] = pd.Series((result['y_validation'] == 2).astype(int),
                                                   index=self.X_validation.index)

            if self.modeling_config['model_type'] == 'classification':
                # get positive-class probabilities directly from pipeline
                proba = meta_pipeline.predict_proba(self.X_validation)
                # For binary classification and asymmetric loss the positive class is always index 1
                pos_idx = 1
                proba_series = pd.Series(proba[:, pos_idx], index=self.X_validation.index)

                # Resolve F-beta string threshold if present
                raw_thr = self.modeling_config['y_pred_threshold']
                if isinstance(raw_thr, str) and raw_thr.startswith('f'):
                    beta = float(raw_thr[1:])
                    # Compute precision-recall curve on validation set
                    true_vals = result['y_validation']
                    precisions, recalls, ths = precision_recall_curve(true_vals, proba_series)
                    ths = np.append(ths, 1.0)  # pad to align with precision/recall arrays
                    beta_sq = beta ** 2
                    f_scores = (1 + beta_sq) * precisions * recalls / (beta_sq * precisions + recalls + 1e-9)
                    best_idx = np.nanargmax(f_scores)
                    thr_val = ths[best_idx] if best_idx < len(ths) else ths[-1]
                    # overwrite config and use numeric threshold
                    self.modeling_config['y_pred_threshold'] = thr_val
                    threshold = thr_val
                else:
                    threshold = raw_thr
                result['y_validation_pred_proba'] = proba_series
                result['y_validation_pred'] = (proba_series >= threshold).astype(int)
            else:
                result['y_validation_pred'] = meta_pipeline.predict(self.X_validation)

            # Add train/test data if requested
            if return_data:
                # Make predictions on test set
                if self.modeling_config['model_type'] == 'classification':
                    # Get positive-class probabilities via MetaPipeline (includes transformation)
                    proba = meta_pipeline.predict_proba(self.X_test)
                    # For binary classification and asymmetric loss the positive class is always index 1
                    pos_idx = 1
                    proba_series = pd.Series(proba[:, pos_idx], index=self.X_test.index)

                    # Apply configurable threshold for class prediction
                    threshold = self.modeling_config['y_pred_threshold']
                    self.y_pred = (proba_series >= threshold).astype(int)

                    # Store the same probabilities we just calculated
                    result['y_pred_proba'] = proba_series
                else:
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
            full_cohort_actuals = wallet_target_vars_df[target_var]
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
            wallet_target_vars_df: pd.DataFrame
        ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare wallet-specific data for modeling. Returns features and target only.

        Params:
        - training_data_df (DataFrame): full training cohort feature data
        - wallet_target_vars_df (DataFrame): Contains in_modeling_cohort flag and target variable

        Returns:
        - X (DataFrame): feature data for modeling cohort
        - y (Series): target variable for modeling cohort
        """
        if self.modeling_config['assign_epochs_to_addresses']:
            # Select 1 epoch for each wallet address
            training_data_df, wallet_target_vars_df = self._assign_epoch_to_wallets(
                training_data_df,
                wallet_target_vars_df,
                random_state = self.modeling_config['model_params'].get('random_state', 42)
            )

        # Identify modeling cohort   # pylint:disable=line-too-long
        if 'cw_crypto_inflows' in wallet_target_vars_df.columns:
            cohort_mask = (
                # Target vars filters
                (wallet_target_vars_df['crypto_inflows'] >= self.modeling_config['modeling_min_crypto_inflows']) &
                (wallet_target_vars_df['unique_coins_traded'] >= self.modeling_config['modeling_min_coins_traded']) &
                (wallet_target_vars_df['cw_crypto_inflows'] >= self.modeling_config['cw_modeling_min_crypto_inflows']) &
                (wallet_target_vars_df['cw_unique_coins_traded'] >= self.modeling_config['cw_modeling_min_coins_traded']) &
                # Training data filters
                (training_data_df['cw_mktcap|end_portfolio_wtd_market_cap/market_cap_filled|w1']
                    >= self.modeling_config['cw_modeling_min_market_cap']) &
                (training_data_df['cw_mktcap|end_portfolio_wtd_market_cap/market_cap_filled|w1']
                    <= self.modeling_config['cw_modeling_max_market_cap'])
            )
        else:
            cohort_mask = (
                (wallet_target_vars_df['crypto_inflows'] >= self.modeling_config['modeling_min_crypto_inflows']) &
                (wallet_target_vars_df['unique_coins_traded'] >= self.modeling_config['modeling_min_coins_traded'])
            )
        logger.milestone("Defined modeling cohort as %.1f%% (%s/%s) wallets.",
            cohort_mask.sum()/len(wallet_target_vars_df)*100,
            cohort_mask.sum(),
            len(wallet_target_vars_df)
        )

        # Define X and y
        X = training_data_df[cohort_mask].copy()
        y = wallet_target_vars_df[cohort_mask].copy()
        u.assert_matching_indices(X,y)

        return X, y


    def _assign_epoch_to_wallets(
        self,
        training_data_df: pd.DataFrame,
        wallet_target_vars_df: pd.DataFrame,
        random_state: int = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Selects a single epoch for any given wallet address. This avoids data leakage
        when the target variable for one time window is exposed as a feature for another.

        Params:
        - training_data_df (DataFrame): multiindexed by ['wallet_address','epoch_start_date']
        - wallet_target_vars_df  (DataFrame): same index as training_data_df

        Returns:
        - sampled_training_data_df: one random epoch per wallet
        - sampled_wallet_target_vars_df:   matching rows from wallet_target_vars_df
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
        sampled_wallet_target_vars_df = wallet_target_vars_df.loc[selected_multiindices]
        u.assert_matching_indices(sampled_training_data_df,sampled_wallet_target_vars_df)

        return sampled_training_data_df,sampled_wallet_target_vars_df


    def _prepare_grid_search_params(self, X: pd.DataFrame, base_params_override=None) -> dict:
        """
        Override to prepend 'model_pipeline__' to all keys in param_grid.
        """
        gs_config = super()._prepare_grid_search_params(X, base_params_override)
        # Prepend "model_pipeline__" to each key if it isn't already prefixed
        gs_config['param_grid'] = {
            f"model_pipeline__{k}" if not k.startswith("model_pipeline__") else k: v
            for k, v in gs_config['param_grid'].items()
        }

        # Add target variable options into the grid search.
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

        # Validate asymmetric loss compatibility with threshold grid search
        asymmetric_enabled = self.asymmetric_loss_enabled
        threshold_params = [
            'target_selector__target_variable',
            'target_selector__target_var_min_threshold',
            'target_selector__target_var_max_threshold'
        ]
        conflicting_params = [param for param in threshold_params if param in param_grid_y]
        if asymmetric_enabled and conflicting_params:
            raise u.ConfigError(
                f"Cannot perform grid search with asymmetric loss enabled on target threshold parameters "
                f"{conflicting_params} Asymmetric loss overrides threshold-based "
                f"target transformation. Either disable asymmetric loss or remove threshold "
                f"parameters from param_grid_y."
            )


        # Add asymmetric loss parameters
        asymmetric_params = [
            'target_selector__asymmetric_enabled',
            'target_selector__asymmetric_big_loss_threshold',
            'target_selector__asymmetric_big_win_threshold',
            'target_selector__asymmetric_loss_penalty_weight',
            'target_selector__asymmetric_win_reward_weight'
        ]

        for param in asymmetric_params:
            if param in param_grid_y:
                prefixed_param = f"y_pipeline__{param}"
                gs_config['param_grid'][prefixed_param] = param_grid_y[param]

        # Validate asymmetric loss compatibility
        asymmetric_enabled = self.asymmetric_loss_enabled
        threshold_params = [
            'target_selector__target_variable',
            'target_selector__target_var_min_threshold',
            'target_selector__target_var_max_threshold'
        ]

        conflicting_params = [param for param in threshold_params if param in param_grid_y]

        # Check if asymmetric is enabled in base config OR being grid searched
        asymmetric_in_grid = 'target_selector__asymmetric_enabled' in param_grid_y
        asymmetric_params_in_grid = any(param in param_grid_y for param in asymmetric_params[1:])  # Skip enabled flag

        if (asymmetric_enabled or asymmetric_in_grid or asymmetric_params_in_grid) and conflicting_params:
            raise u.ConfigError(
                f"Cannot perform grid search on target threshold parameters {conflicting_params} "
                f"when asymmetric loss is enabled or being grid searched. "
                f"Remove either asymmetric parameters or threshold parameters from param_grid_y."
            )


        # Confirm there are multiple configurations
        if not any(isinstance(value, list) and len(value) > 1 for value in gs_config['param_grid'].values()):
            raise ValueError("Grid search param grid only contains one scenario. "
                             "Add more scenarios to run grid search.")

        return gs_config


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
