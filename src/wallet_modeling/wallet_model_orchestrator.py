import logging
import copy
import json
import math
from pathlib import Path
import pandas as pd
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
        modeling_wallet_features_df,
        validation_training_data_df,
        validation_wallet_features_df
    ) -> dict:
        """
        Train multiple wallet scoring models with different parameter configurations.

        Params:
        - wallet_training_data_df: Training data DataFrame
        - modeling_wallet_features_df: Features for modeling
        - validation_training_data_df: Validation training data
        - validation_wallet_features_df: Validation features

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
                modeling_wallet_features_df,
                validation_training_data_df,
                validation_wallet_features_df
            )
            evaluators.append((score_name, evaluator))

            # Store model ID in dictionary for later use
            models_dict[score_name] = model_id

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
            model_id = models_dict[score_name]

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
        modeling_wallet_features_df,
        validation_training_data_df,
        validation_wallet_features_df
    ):
        """
        Train and evaluate a single model with given configuration.

        Params:
        - score_wallets_config: Configuration specific to this score
        - wallet_training_data_df: Training data DataFrame
        - modeling_wallet_features_df: Features for modeling
        - validation_training_data_df: Validation training data
        - validation_wallet_features_df: Validation features

        Returns:
        - (model_id, wallet_evaluator): ID of the trained model and evaluator object
        """
        # Construct model
        wallet_model = WalletModel(score_wallets_config['modeling'])
        wallet_model_results = wallet_model.construct_wallet_model(
            wallet_training_data_df, modeling_wallet_features_df,
            validation_training_data_df, validation_wallet_features_df
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
                evaluator._plot_score_distribution(ax)  # pylint:disable=protected-access
            ax.set_title(score_name)

        # Hide any unused subplots
        for ax in axes_flat[n:]:
            ax.axis('off')

        plt.tight_layout()
        plt.show()
