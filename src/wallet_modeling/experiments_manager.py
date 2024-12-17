"""
Calculates metrics aggregated at the wallet level
"""
import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Local module imports
import wallet_modeling.wallet_modeling_orchestrator as wmo
import wallet_modeling.wallet_model as wm
import wallet_features.performance_features as wpf

# Set up logger at the module level
logger = logging.getLogger(__name__)


class ExperimentsManager:
    """
    Manages multiple wallet modeling experiments, handles data preparation,
    and tracks experiment results.
    """

    def __init__(self, config: dict, training_data_df: pd.DataFrame):
        """
        Params:
        - config (dict): configuration dictionary containing modeling parameters
        - training_data_df (DataFrame): pre-computed training features
        """
        self.config = config
        self.training_data_df = training_data_df
        self.experiments: Dict[str, Dict] = {}
        self.model_results: Dict[str, Dict] = {}

    def prepare_modeling_data(self, modeling_profits_df: pd.DataFrame) -> pd.DataFrame:
        """
        Params:
        - modeling_profits_df (DataFrame): raw profits data from the modeling period only

        Returns:
        - modeling_df (DataFrame): prepared modeling DataFrame
        """
        # Create modeling dataset using existing pipeline
        modeling_wallets_df = wmo.filter_modeling_period_wallets(modeling_profits_df)
        target_vars_df = wpf.calculate_performance_features(modeling_wallets_df)

        modeling_df = self.training_data_df.join(
            target_vars_df[self.config['modeling']['target_variable']],
            how='inner'
        )
        return modeling_df

    def run_experiment(self,
                      experiment_name: str,
                      modeling_df: pd.DataFrame,
                      experiment_config: Optional[dict] = None) -> Dict:
        """
        Params:
        - experiment_name (str): unique identifier for this experiment
        - modeling_df (DataFrame): prepared modeling data
        - experiment_config (dict, optional): override default config for this experiment

        Returns:
        - results (dict): experiment results including model and metrics
        """
        # Use experiment-specific config if provided
        config = experiment_config or self.config

        # Initialize and run experiment
        experiment = wm.WalletModel(config)
        results = experiment.run_experiment(modeling_df)

        # Calculate metrics
        y_pred = results['y_pred']
        y_test = results['y_test']
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }

        # Store results
        self.experiments[experiment_name] = experiment
        self.model_results[experiment_name] = {
            'results': results,
            'metrics': metrics,
            'config': config.config
        }

        return self.model_results[experiment_name]

    def compare_experiments(self) -> pd.DataFrame:
        """
        Returns:
        - comparison_df (DataFrame): metrics comparison across experiments
        """
        metrics_data = []
        for exp_name, exp_data in self.model_results.items():
            metrics = exp_data['metrics']
            metrics['experiment'] = exp_name
            metrics_data.append(metrics)

        return pd.DataFrame(metrics_data).set_index('experiment')
