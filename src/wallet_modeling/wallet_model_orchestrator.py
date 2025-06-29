"""
wallet_model_orchestrator.py
============================

Purpose
-------
Drive the end-to-end **“one date ⇒ one model”** loop:

1. Ask **WalletEpochsOrchestrator** for the ready-made feature / target
   snapshots for each modeling date.
2. Feed those snapshots into **WalletModel** to train or grid-search.
3. Collect scores, best params, and fitted pipelines for later use
   (e.g. wallet-investing backtests).

Why it exists
-------------
Keeps *data prep* and *model fit* concerns separate:

* WalletEpochsOrchestrator ⇢ heavy aggregation & cohort filters
* WalletModelOrchestrator  ⇢ modeling experiments & result curation
"""
import logging
import copy
import json
import math
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import cloudpickle

# Local modules
from wallet_modeling.wallet_model import WalletModel
import wallet_insights.wallet_model_reporting as wimr
import wallet_insights.wallet_validation_analysis as wiva
import wallet_insights.model_evaluation as wime
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)


class WalletModelOrchestrator:
    """
    Thin controller that loops over modeling dates.

    Workflow
    --------
    1. `load_all_training_data()`
       → caches `(X, y)` pairs from **WalletEpochsOrchestrator**.
    2. `run_multi_temporal_grid_search()`
       → optional CV search; stores `grid_search_results`.
    3. `run_multi_temporal_model_comparison()`
       → fits one WalletModel per date (no search) and records metrics.

    Key interactions
    ----------------
    * **WalletEpochsOrchestrator** supplies data; this class never touches raw
      SQL or parquet directly.
    * Hands each `(X, y)` to **WalletModel.construct_wallet_model()**.
    * Honors the model’s *export-only* flag—if a date requests
      `export_s3_training_data`, it skips training for that date.

    Artifacts
    ---------
    * `training_data_cache`   – {yyMMdd: (X, y)}
    * `grid_search_results`   – CV summaries per date
    * `fitted_models`         – {yyMMdd: sklearn pipeline}

    Lightweight by design: no heavy computation beyond the model fits
    themselves.
    """
    def __init__(
        self,
        wallets_config: dict,
        wallets_metrics_config: dict,
        wallets_features_config: dict,
        wallets_epochs_config: dict,
        wallets_coin_config: dict,
    ):
        """
        Initialize the wallet model trainer.

        Params:
        - wallets_config: Main configuration object
        - wallets_metrics_config: Metrics configuration
        - wallets_features_config: Features configuration
        - wallets_epochs_config: Epochs configuration
        - wallets_coin_config: Coin-specific configuration
        """
        self.wallets_config = wallets_config
        self.wallets_metrics_config = wallets_metrics_config
        self.wallets_features_config = wallets_features_config
        self.wallets_epochs_config = wallets_epochs_config
        self.wallets_coin_config = wallets_coin_config

        # Extract parameters from configs
        self.score_params = wallets_coin_config['wallet_scores']['score_params']
        self.base_path = self.wallets_config['training_data']['model_artifacts_folder']

        # Dict to store model IDs
        self.models_dict = None



    # -----------------------------------
    #         Primary Interface
    # -----------------------------------

    def train_wallet_models(
        self,
        wallet_training_data_df: pd.DataFrame,
        wallet_target_vars_df: pd.DataFrame,
        validation_training_data_df: pd.DataFrame,
        validation_target_vars_df: pd.DataFrame,
    ) -> dict:
        """
        Train multiple wallet scoring models with different parameter configurations.

        New models will be trained if no models exist yet, or if the param
         wallets_coin_config['training_data']['toggle_rebuild_wallet_models'] is set to True.

        Existing models will be used if 'wallet_model_ids.json' contains their metadata
         and toggle_rebuild_wallet_models==False.

        Params:
        - wallet_training_data_df: Training data DataFrame
        - wallet_target_vars_df: Features for modeling
        - validation_training_data_df: Validation training data
        - validation_target_vars_df: Validation features

        Returns:
        - models_dict: Dictionary mapping score names to model IDs
        """
        ambient_player_wallets = u.AmbientPlayer()

        # Load existing models if any
        models_json_path = Path(self.wallets_coin_config['training_data']['parquet_folder']) / "wallet_model_ids.json"
        if models_json_path.exists():
            with open(models_json_path, 'r', encoding='utf-8') as f:
                models_json_dict = json.load(f)
        else:
            models_json_dict = {}
        evaluators = []

        i = 0

        for score_name in self.score_params:
            # Create a deep copy of the configuration to avoid modifying the original
            score_wallets_config = copy.deepcopy(self.wallets_config)

            # Override score name and model params in base config
            score_wallets_config['modeling']['score_name'] = score_name
            for param_name in self.score_params[score_name]:
                score_wallets_config['modeling'][param_name] = self.score_params[score_name][param_name]

            # Don't output the scores from every tree
            score_wallets_config['modeling']['verbose_estimators'] = False

            # Evaluate using an existing model if configured and available
            # ------------------------------------------------------------
            if score_name in models_json_dict:
                if not self.wallets_coin_config['training_data']['toggle_rebuild_wallet_models']:
                    model_id, evaluator = self._load_and_evaluate(
                        score_wallets_config,
                        wallet_training_data_df,
                        wallet_target_vars_df,
                        score_name = score_name,
                    )
                    evaluators.append((score_name, evaluator))
                    logger.info(f"Loaded pretrained model for score '{score_name}'.")
                    continue

                # Announce overwrite if applicable
                else:
                    logger.warning("Overwriting existing models due to 'toggle_rebuild_wallet_models'.")

            # Train new model if we didn't load existing
            # ------------------------------------------
            ambient_player_wallets.start('ship_power_room_loop')
            model_id, evaluator = self._train_and_evaluate(
                score_wallets_config,
                wallet_training_data_df,
                wallet_target_vars_df,
                validation_training_data_df,
                validation_target_vars_df
            )
            # Persist metrics for newly trained model
            models_json_dict[score_name] = self._store_model_metrics(model_id, evaluator)
            models_json_dict[score_name]['metrics'] = evaluator.metrics
            evaluators.append((score_name, evaluator))

            i+=1
            logger.milestone(f"Finished training model {i}/{len(self.score_params)}"
                            f": {score_name}.")

            # Resolve and persist the final classification cutoff
            # -------------------------------------------------
            # Persist *numeric* threshold into self.score_params for downstream scoring
            y_threshold_numeric = score_wallets_config['modeling'].get('y_pred_threshold')
            self.score_params[score_name]['y_pred_threshold'] = y_threshold_numeric

        ambient_player_wallets.stop()
        # u.notify('lovelyboot')

        # Store and save models_dict
        self.models_dict = models_json_dict
        save_location = f"{self.wallets_coin_config['training_data']['parquet_folder']}/wallet_model_ids.json"
        with open(save_location, 'w', encoding='utf-8') as f:
            json.dump(models_json_dict, f, indent=4, default=u.numpy_type_converter)
            logger.milestone(f"Saved wallet_model_ids.json to {save_location}.")

        logger.milestone(f"Prepared all {len(self.score_params)} wallet models.")

        if self.wallets_coin_config['training_data']['toggle_graph_wallet_model_performance']:
            self._plot_score_summaries(evaluators)

        return models_json_dict



    def _store_model_metrics(self, model_id: str, evaluator) -> dict:
        """
        Extract and store model performance metrics and macroeconomic features.

        Params:
        - model_id: Unique identifier for the trained model
        - evaluator: Model evaluator object containing predictions and validation data

        Returns:
        - dict: Model metrics including ID, return metrics, and macro averages
        """
        # If there isn't a validation set, return just the model_id. This happens when the
        #  wallet model is built for a current coin model without target variables, as coin
        #  target variables are generated from the same period as wallet validation data.
        if evaluator.X_validation is None:
            return {'model_id': model_id,}

        # Extract return metrics for analysis
        n_buckets = 20
        if evaluator.modeling_config.get('model_type') == 'classification':
            # Use evaluator's bucket computation method for classification
            bucket_df = self._compute_return_vs_rank_df(evaluator, n_buckets=n_buckets)
        else:
            # Equal-width buckets for regression
            bucket_df = self._compute_combined_score_return_df(evaluator, n_buckets=n_buckets)

        # Convert DataFrame to serializable dict
        return_metrics = bucket_df.to_dict(orient='list')

        # Extract macroeconomic features for analysis
        macro_cols = [col for col in evaluator.X_validation.columns if col.startswith('macro|')]
        macro_metrics = (evaluator.X_validation[macro_cols].reset_index()
                         .groupby('epoch_start_date')
                         .mean()
                         .mean())  # Second mean() collapses to single row
        # Convert macro_metrics to serializable dict
        macro_averages = macro_metrics.to_dict()

        return {
            'model_id': model_id,
            'return_metrics': return_metrics,
            'macro_averages': macro_averages
        }



    def predict_and_store(self, models_dict, training_data_df):
        """
        Generate predictions for each model and store them as parquet files.

        Params:
        - models_dict: Dictionary mapping score names to model IDs
        - training_data_df: Training data DataFrame containing epoch information

        Returns:
        - None: Files are saved to the temp_path directory
        """
        # Ensure there is exactly one epoch_start_date in the data, as each coin model
        #  epoch will have a single coin modeling period.
        epoch_dates = training_data_df.reset_index()['epoch_start_date'].unique()
        if len(epoch_dates) > 1:
            raise ValueError(f"Expected a single epoch_start_date, but found multiple: {epoch_dates}")

        # Process each model in the dictionary
        for score_name in models_dict.keys():
            model_id = models_dict[score_name]['model_id']

            # Load model and generate predictions
            y_pred = wiva.load_and_predict(
                model_id,
                training_data_df,
                self.base_path
            )
            wallet_scores_df = pd.DataFrame({f'score|{score_name}': y_pred})

            # Calculate binary predictions if applicable
            y_pred_threshold = self.score_params[score_name].get('y_pred_threshold')
            if y_pred_threshold is not None:
                y_pred_binary = (y_pred >= y_pred_threshold).astype(int)
                wallet_scores_df[f'binary|{score_name}'] = y_pred_binary

            # Identify scores folder
            scores_folder = self.wallets_coin_config['training_data']['coins_wallet_scores_folder']
            Path(scores_folder).mkdir(parents=True, exist_ok=True)

            # Save predictions to parquet file
            output_path = f"{scores_folder}/{score_name}.parquet"
            u.to_parquet_safe(wallet_scores_df, output_path, index=True)

            logger.info(f"Saved predictions for {score_name} to {output_path}")

        logger.info("Finished scoring wallets with all models.")



    # -----------------------------------
    #           Helper Methods
    # -----------------------------------

    def _train_and_evaluate(
        self,
        score_wallets_config,
        wallet_training_data_df,
        wallet_target_vars_df,
        validation_training_data_df,
        validation_target_vars_df
    ):
        """
        Train and evaluate a single model with given configuration.

        Params:
        - score_wallets_config: Configuration specific to this score
        - wallet_training_data_df: Training data DataFrame
        - wallet_target_vars_df: Features for modeling
        - validation_training_data_df: Validation training data
        - validation_target_vars_df: Validation features

        Returns:
        - (model_id, wallet_evaluator): ID of the trained model and evaluator object
        """
        # Construct model
        wallet_model = WalletModel(score_wallets_config['modeling'])
        wallet_model_results = wallet_model.construct_wallet_model(
            wallet_training_data_df, wallet_target_vars_df,
            validation_training_data_df, validation_target_vars_df
        )

        # Generate and save all model artifacts
        model_id, wallet_evaluator, _ = wimr.generate_and_save_wallet_model_artifacts(
            model_results=wallet_model_results,
            base_path=self.base_path,
            configs={
                'wallets_config': self.wallets_config,
                'wallets_metrics_config': self.wallets_metrics_config,
                'wallets_features_config': self.wallets_features_config,
                'wallets_epochs_config': self.wallets_epochs_config,
                'wallets_coin_config': self.wallets_coin_config
            }
        )

        # Display model evaluation summary
        wallet_evaluator.summary_report()

        return model_id, wallet_evaluator



    def _plot_score_summaries(self, evaluators: list[tuple[str, any]]) -> None:
        """
        Plot score summaries for each trained model in a 2-column layout.
        """
        # Temporarily reduce base font sizes
        for key in [
            'font.size',
            'axes.titlesize',
            'axes.labelsize',
            'xtick.labelsize',
            'ytick.labelsize'
        ]:
            plt.rcParams[key] = plt.rcParams[key] - 3

        n = len(evaluators)
        cols = 2
        rows = math.ceil(n / cols)
        _, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for ax, (score_name, evaluator) in zip(axes_flat, evaluators):
            if evaluator.modeling_config.get('model_type') == 'classification':
                evaluator._plot_return_vs_rank_classifier(ax, n_buckets=20)  # pylint:disable=protected-access
            else:
                evaluator._plot_combined_score_return(ax)  # pylint:disable=protected-access
            ax.set_title(score_name)

        # Hide any unused subplots
        for ax in axes_flat[n:]:
            ax.axis('off')

        plt.tight_layout()
        plt.show()



    def _compute_return_vs_rank_df(self, evaluator, n_buckets: int) -> pd.DataFrame:
        """
        Compute return vs rank DataFrame using the same binning logic as charts.
        For classifiers, delegates to the evaluator's compute_score_buckets method.
        """
        # For classifiers, delegate to evaluator's bucket computation method
        if evaluator.modeling_config.get('model_type') == 'classification':
            bucket_df = evaluator.compute_score_buckets(n_buckets)

            if bucket_df.empty:
                # Return empty buckets filled with None
                return pd.DataFrame({
                    "bucket": [None] * n_buckets,
                    "mean_return": [None] * n_buckets,
                    "wins_return": [None] * n_buckets,
                    "score_mid": [None] * n_buckets
                })

            # Sort buckets by score (lowest to highest)
            bucket_df = bucket_df.sort_values('score_mid', ascending=True).reset_index(drop=True)

            # Forward fill to ensure we have n_buckets entries
            mean_values = []
            wins_values = []
            score_values = []

            for i in range(len(bucket_df)):
                row = bucket_df.iloc[i]
                mean_values.append(row['mean_return'])
                wins_values.append(row['wins_return'])
                score_values.append(row['score_mid'])

            # Pad with None if we have fewer buckets than requested
            while len(mean_values) < n_buckets:
                mean_values.append(None)
                wins_values.append(None)
                score_values.append(None)

            return pd.DataFrame({
                "bucket": [round(score, 3) if score is not None else None for score in score_values],
                "mean_return": mean_values,
                "wins_return": wins_values,
                "score_mid": score_values
            })

        # Original regression logic for non-classification models
        y_pred = evaluator.y_validation_pred
        returns = evaluator.validation_target_vars_df[evaluator.modeling_config["target_variable"]]
        df = pd.DataFrame({"pred": y_pred, "ret": returns.reindex(y_pred.index)}).dropna()

        # Check if we have enough unique values for n_buckets
        unique_preds = df["pred"].nunique()
        actual_buckets = min(n_buckets, unique_preds)

        if actual_buckets < 2:
            # If only 1 unique prediction, create single bucket
            df["bucket"] = 1
            df["score_mid"] = df["pred"]
        else:
            # Use qcut with duplicates='drop'
            df["bucket_raw"] = pd.qcut(df["pred"], actual_buckets, labels=False, duplicates="drop")
            # Keep natural ordering (no reversal)
            df["bucket"] = df["bucket_raw"] + 1
            # Calculate score midpoints for each bucket
            bucket_score_mid = df.groupby("bucket")["pred"].mean()
            df["score_mid"] = df["bucket"].map(bucket_score_mid)

        # Calculate metrics only for buckets that exist
        bucket_mean = df.groupby("bucket")["ret"].mean()
        bucket_wins = df.groupby("bucket")["ret"].apply(
            lambda x: u.winsorize(x.values, evaluator.modeling_config.get("returns_winsorization", 0.005)).mean()
        )
        bucket_scores = df.groupby("bucket")["score_mid"].first()

        # Create output preserving natural score order
        bucket_indices = sorted(bucket_mean.index) if len(bucket_mean) > 0 else []

        for bucket_idx in bucket_indices:
            mean_values.append(bucket_mean[bucket_idx])
            wins_values.append(bucket_wins[bucket_idx])
            score_values.append(bucket_scores[bucket_idx])

        # Pad with None if we have fewer buckets than requested
        while len(mean_values) < n_buckets:
            mean_values.append(None)
            wins_values.append(None)
            score_values.append(None)

        result_df = pd.DataFrame({
            "bucket": [round(score, 3) if score is not None else None for score in score_values],
            "mean_return": mean_values,
            "wins_return": wins_values,
            "score_mid": score_values
        })

        return result_df


    def _compute_combined_score_return_df(self, evaluator, n_buckets: int) -> pd.DataFrame:
        # Replicate ModelEvaluator._plot_combined_score_return bucket calc
        y_pred = evaluator.y_validation_pred
        returns = evaluator.validation_target_vars_df[evaluator.modeling_config["target_variable"]]
        # Align numpy predictions to returns index
        if not isinstance(y_pred, pd.Series):
            y_pred = pd.Series(y_pred, index=returns.index)
        df = pd.DataFrame({"pred": y_pred, "ret": returns}).dropna()
        wins_thr = evaluator.modeling_config.get("returns_winsorization", 0.005)
        df["ret_wins"] = u.winsorize(df["ret"].values, wins_thr)
        edges = np.linspace(df["pred"].min(), df["pred"].max(), n_buckets+1)
        buckets = []
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (df["pred"]>=low)&(df["pred"]<=high)
            if mask.sum()>0:
                buckets.append({
                    "score_mid": (low+high)/2,
                    "mean_return": df.loc[mask, "ret"].mean(),
                    "median_return": df.loc[mask, "ret"].median(),
                    "wins_return": df.loc[mask, "ret_wins"].mean(),
                })
        return pd.DataFrame(buckets)



    def _load_and_evaluate(
        self,
        score_wallets_config: dict,
        wallet_training_data_df: pd.DataFrame,
        wallet_target_vars_df: pd.DataFrame,
        model_id: str = None,
        score_name: str = None
    ) -> tuple[str, any]:
        """
        Load an existing saved model by score_name and evaluate on provided data.

        Note that these dfs are passed into the evaluator in the X_validation and y_validation
         params. This is because we cannot calculate any train/test set metrics when we aren't
         training a model. Because we are evaluating the model on scored wallet performance
         versus true future values, the graphs and metrics should be generated using the logic
         from the y_validation/y_validation_pred calculations.

        If model_id is not provided, it can be looked up using score_name. If neither are provided,
         raises a ValueError.

        Params:
        - score_wallets_config (dict): wallets_config.yaml with overrides for the specific score
        - validation_training_data_df (df): Contains wallet training data for the validation period
        - validation_target_vars_df (df): Contains target variables for the validation period
        - model_id (str) [Optional]: uuid of the model to load. If None, score_name is used to look it up.
        - score_name (str) [Optional]: from wallets_coin_config['wallet_scores']['score_params'].keys()

        Returns:
        - model_id (str): ID of the loaded model
        - evaluator (custom class): Evaluator from model_evaluation.py with metrics and predictions
        """
        # Data quality checks
        if wallet_training_data_df.empty or wallet_target_vars_df.empty:
            raise ValueError("Training data or target vars cannot be empty.",
                             "There is valid business logic that would only provide training_data_df " \
                             "but the logic for empty target_vars_df would need to be built out.")
        u.assert_matching_indices(wallet_training_data_df,wallet_target_vars_df)
        if (model_id is not None and score_name is not None):
            raise ValueError("Only one of 'model_id' or 'score_name' can be used as the model identifier.")

        # 1) Look up the model_id if not provided
        # ---------------------------------------
        if model_id is None:
            if score_name is None:
                raise ValueError("Either 'model_id_ or 'score_name' must be provided to look up "
                                 "an existing model")
            models_json_path = (Path(self.wallets_coin_config['training_data']['parquet_folder'])
                                / "wallet_model_ids.json")
            with open(models_json_path, 'r', encoding='utf-8') as f:
                models_dict = json.load(f)
            model_info = models_dict.get(score_name)
            if model_info is None:
                raise KeyError(f"No existing model found for '{score_name}'")
            model_id = model_info['model_id']


        # 2) Unpickle the pipeline
        # ------------------------
        pipeline_path = Path(self.base_path) / 'wallet_models' / f"wallet_model_{model_id}.pkl"
        if not pipeline_path.exists():
            logger.warning(
                f"Pipeline file missing for model {score_name} ({model_id}). "
                f"Falling back to rebuilding the model."
            )
            # Fall back to training a new model
            return self._train_and_evaluate(
                score_wallets_config,
                wallet_training_data_df,
                wallet_target_vars_df,
                None,  # validation_training_data_df
                None   # validation_target_vars_df
            )
        with open(pipeline_path, 'rb') as f:
            pipeline = cloudpickle.load(f)


        # 3) Build a minimal result dict
        # ------------------------------
        modeling_cfg = score_wallets_config['modeling']
        model_type = modeling_cfg['model_type']
        result = {
            'model_type': model_type,
            'model_id': model_id,
            'pipeline': pipeline,
            'modeling_config': modeling_cfg,

            # Pretrained models don't have train/test sets
            'X_train': None,
            'y_train': None,
            'X_test': None,
            'y_test': None,
            'y_pred': None,
            'training_cohort_pred': None,
            'training_cohort_actuals': None,
            'full_cohort_actuals': None
        }


        # 4) Attach validation set + preds
        # --------------------------------------------
        # Retain X_validation and make predictions
        result['X_validation'] = wallet_training_data_df
        result['validation_target_vars_df'] = wallet_target_vars_df
        y_val_series = wallet_target_vars_df[modeling_cfg['target_variable']]
        val_preds = wiva.load_and_predict(
            model_id,
            wallet_training_data_df,
            self.base_path
        )

        # Store regressor outcomes
        if model_type == 'regression':
            # y_true
            result['y_validation'] = y_val_series
            # y_pred
            result['y_validation_pred'] = val_preds

        elif model_type == 'classification':
            # Ensure thresholds are provided
            raw_min = modeling_cfg.get('target_var_min_threshold')
            raw_max = modeling_cfg.get('target_var_max_threshold')
            if raw_min is None:
                raise u.ConfigError("target_var_min_threshold must be set for classification models")
            if raw_max is None:
                raise u.ConfigError("target_var_max_threshold must be set for classification models")
            # Convert to float (allows inf) or raise configuration error
            try:
                min_thr = float(raw_min)
                max_thr = float(raw_max)
            except (TypeError, ValueError) as exc:
                raise u.ConfigError(
                    f"target_var thresholds must be numeric (int, float or inf), got: {raw_min!r}, {raw_max!r}"
                ) from exc
            # y_true based on numeric thresholds
            result['y_validation'] = (
                (y_val_series >= min_thr) &
                (y_val_series <= max_thr)
            ).astype(int)
            n_positive_val = result['y_validation'].sum()
            n_total_val = len(result['y_validation'])
            logger.info(f"Model {score_name}: y_validation true positives={n_positive_val}/{n_total_val} "
                        f"({100*n_positive_val/n_total_val:.2f}%)")

            # y_pred
            # define y_threshold
            y_threshold = modeling_cfg.get('y_pred_threshold')
            if isinstance(y_threshold, str):
                # define y_threshold if threshold was passed as an F-beta value
                precisions, recalls, ths = precision_recall_curve(
                    result['y_test'], val_preds
                )
                ths = np.append(ths, 1.0)
                beta = float(y_threshold[1:])
                y_threshold = wime.ClassifierEvaluator.add_fbeta_metrics(
                    precisions, recalls, ths, beta
                )
            modeling_cfg['y_pred_threshold'] = y_threshold

            # define predictions
            result['y_validation_pred_proba'] = val_preds
            result['y_validation_pred'] = (val_preds >= y_threshold).astype(int)
            n_positive_val_pred = result['y_validation_pred'].sum()
            n_total_val_pred = len(result['y_validation_pred'])
            logger.info(f"Model {score_name}: y_validation predicted positive={n_positive_val_pred}/{n_total_val_pred} "
                        f"({100*n_positive_val_pred/n_total_val_pred:.2f}%)")
        else:
            raise ValueError(f"Invalid model type '{model_type}' found in config.")


        # 5) Instantiate evaluator
        # ------------------------
        if model_type == 'classification':
            evaluator = wime.ClassifierEvaluator(result)
        else:
            evaluator = wime.RegressorEvaluator(result)


        return model_id, evaluator
