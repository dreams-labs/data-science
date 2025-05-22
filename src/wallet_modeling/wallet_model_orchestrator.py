import logging
import copy
import json
import math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Local modules
from wallet_modeling.wallet_model import WalletModel
import wallet_insights.wallet_model_reporting as wimr
import wallet_insights.wallet_validation_analysis as wiva
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)


class WalletModelOrchestrator:
    """
    Trainer for wallet scoring models that handles multiple parameter configurations.
    """

    def __init__(
        self,
        wallets_config,
        wallets_metrics_config,
        wallets_features_config,
        wallets_epochs_config,
        wallets_coin_config,
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
        wallet_training_data_df,
        wallet_target_vars_df,
        validation_training_data_df,
        validation_target_vars_df
    ) -> dict:
        """
        Train multiple wallet scoring models with different parameter configurations.

        Params:
        - wallet_training_data_df: Training data DataFrame
        - wallet_target_vars_df: Features for modeling
        - validation_training_data_df: Validation training data
        - validation_target_vars_df: Validation features

        Returns:
        - models_dict: Dictionary mapping score names to model IDs
        """
        models_dict = {}
        evaluators = []

        for score_name in self.score_params:
            # Create a deep copy of the configuration to avoid modifying the original
            score_wallets_config = copy.deepcopy(self.wallets_config)

            # Override score name and model params in base config
            score_wallets_config['modeling']['score_name'] = score_name
            for param_name in self.score_params[score_name]:
                score_wallets_config['modeling'][param_name] = self.score_params[score_name][param_name]

            # Don't output the scores from every tree
            score_wallets_config['modeling']['verbose_estimators'] = False

            # Train and evaluate the model
            model_id, evaluator = self._train_and_evaluate(
                score_wallets_config,
                wallet_training_data_df,
                wallet_target_vars_df,
                validation_training_data_df,
                validation_target_vars_df
            )
            evaluators.append((score_name, evaluator))

            # -------------------------------------------------
            # Resolve and persist the final classification cutoff
            # -------------------------------------------------
            resolved_thr = score_wallets_config['modeling'].get('y_pred_threshold')

            # If the threshold is still an F‑beta string (e.g. "f1", "f0.25"),
            # convert it to the numeric value contained in the evaluator.
            if isinstance(resolved_thr, str) and resolved_thr.startswith('f'):
                beta_str = resolved_thr[1:]
                thr_key = f"f{beta_str}_thr"
                # ensure the evaluator has calculated this F‑beta metric
                if thr_key not in evaluator.metrics:
                    evaluator.add_fbeta_metrics()
                resolved_thr = evaluator.metrics[thr_key]
                # update the modeling config so any future use sees the numeric value
                score_wallets_config['modeling']['y_pred_threshold'] = resolved_thr
                logger.info(
                    "Set y_pred_threshold for '%s' to F%s threshold: %.4f",
                    score_name,
                    beta_str,
                    resolved_thr,
                )

            # Persist *numeric* threshold into self.score_params for downstream scoring
            # (predict_and_store relies on this dict).
            if resolved_thr is not None:
                self.score_params[score_name]['y_pred_threshold'] = float(resolved_thr)
            else:
                # keep None explicitly so predict_and_store skips binary prediction
                self.score_params[score_name]['y_pred_threshold'] = None

            # Store model metrics
            models_dict[score_name] = self._store_model_metrics(model_id, evaluator)

            logger.milestone(f"Finished training model {len(models_dict)}/{len(self.score_params)}"
                             f": {score_name}.")

        # Store and save models_dict
        self.models_dict = models_dict
        save_location = f"{self.wallets_coin_config['training_data']['parquet_folder']}/wallet_model_ids.json"
        with open(save_location, 'w', encoding='utf-8') as f:
            json.dump(models_dict, f, indent=4, default=u.numpy_type_converter)

        logger.info(f"Finished traning all {len(self.score_params)} models.")

        self._plot_score_summaries(evaluators)

        return models_dict



    def _store_model_metrics(self, model_id: str, evaluator) -> dict:
        """
        Extract and store model performance metrics and macroeconomic features.

        Params:
        - model_id: Unique identifier for the trained model
        - evaluator: Model evaluator object containing predictions and validation data

        Returns:
        - dict: Model metrics including ID, return metrics, and macro averages
        """
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
                         .mean())
        # Convert macro_metrics to serializable dict
        macro_averages = macro_metrics.to_dict(orient='list')

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

        # Extract epoch start date from training data
        epoch_start = epoch_dates[0].strftime('%Y%m%d')

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
            output_path = f"{scores_folder}/{score_name}|{epoch_start}.parquet"
            wallet_scores_df.to_parquet(output_path, index=True)

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
                    "bucket": range(1, n_buckets + 1),
                    "mean_return": [None] * n_buckets,
                    "wins_return": [None] * n_buckets
                })

            # Convert score-based buckets to rank-based format for storage
            # Rank buckets from 1 (highest score) to n_buckets (lowest score)
            bucket_df = bucket_df.sort_values('score_mid', ascending=False).reset_index(drop=True)
            bucket_df['bucket'] = range(1, len(bucket_df) + 1)

            # Forward fill to ensure we have n_buckets entries
            mean_values = []
            wins_values = []
            last_mean = None
            last_wins = None

            for i in range(1, n_buckets + 1):
                if i <= len(bucket_df):
                    row = bucket_df.iloc[i-1]
                    last_mean = row['mean_return']
                    last_wins = row['wins_return']
                mean_values.append(last_mean)
                wins_values.append(last_wins)

            return pd.DataFrame({
                "bucket": range(1, n_buckets + 1),
                "mean_return": mean_values,
                "wins_return": wins_values
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
        else:
            # Use qcut with duplicates='drop' and track actual buckets created
            df["bucket_raw"] = pd.qcut(df["pred"], actual_buckets, labels=False, duplicates="drop")
            # Reverse ranking so highest predictions get bucket 1
            df["bucket"] = actual_buckets - df["bucket_raw"]
            bucket_indices = sorted(df["bucket"].unique())

        # Calculate metrics only for buckets that exist
        bucket_mean = df.groupby("bucket")["ret"].mean()
        bucket_wins = df.groupby("bucket")["ret"].apply(
            lambda x: u.winsorize(x.values, evaluator.modeling_config.get("returns_winsorization", 0.005)).mean()
        )

        # Create output with full range 1 to n_buckets, forward filling missing values
        mean_values = []
        wins_values = []
        last_mean = None
        last_wins = None

        for i in range(1, n_buckets + 1):
            if i in bucket_mean.index:
                last_mean = bucket_mean[i]
                last_wins = bucket_wins[i]
            mean_values.append(last_mean)
            wins_values.append(last_wins)

        result_df = pd.DataFrame({
            "bucket": range(1, n_buckets + 1),
            "mean_return": mean_values,
            "wins_return": wins_values
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
