import logging
import itertools
from typing import Dict, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Local module imports
import wallet_modeling.wallet_modeling_orchestrator as wmo
import wallet_modeling.wallet_model as wm
import wallet_features.performance_features as wpf

logger = logging.getLogger(__name__)

class ExperimentsManager:
    """
    Manages multiple wallet modeling experiments, handles data preparation,
    and tracks experiment results.
    """

    def __init__(self,
                 config: dict,
                 training_data_df: pd.DataFrame,
                 metrics_config: Optional[Dict] = None):
        """
        Params:
        - config (dict): configuration dictionary containing modeling parameters
        - training_data_df (DataFrame): pre-computed training features
        - metrics_config (dict, optional): mapping of metric names to scoring functions
        """
        self.config = config
        self.training_data_df = training_data_df

        # Default metrics if none provided
        self.metrics_config = metrics_config or {
            'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score
        }

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

        target_var = self.config['modeling']['target_variable']
        modeling_df = self.training_data_df.join(
            target_vars_df[target_var],
            how=self.config['modeling'].get('join_type', 'inner')
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
        logger.info(f"Initializing model for experiment: {experiment_name}")
        experiment = wm.WalletModel(config)
        results = experiment.run_experiment(modeling_df)

        # Calculate all configured metrics
        y_pred = results['y_pred']
        y_test = results['y_test']
        metrics = {
            name: metric_fn(y_test, y_pred)
            for name, metric_fn in self.metrics_config.items()
        }

        logger.info(f"Experiment {experiment_name} complete. Metrics: {metrics}")

        # Store results
        self.experiments[experiment_name] = experiment
        self.model_results[experiment_name] = {
            'results': results,
            'metrics': metrics,
            'config': config
        }

        return self.model_results[experiment_name]


    def _get_variation_paths(self, d: Dict, path: Optional[list] = None) -> list:
        """
        Helper function to get all paths to lists in nested dict.

        Params:
        - d (dict): dictionary to search
        - path (list, optional): current path in recursion

        Returns:
        - paths (list): list of tuples (path, values)
        """
        path = path or []
        paths = []

        for k, v in d.items():
            current = path + [k]
            if isinstance(v, list):
                paths.append((current, v))
            elif isinstance(v, dict):
                paths.extend(self._get_variation_paths(v, current))
        return paths

    def run_experiment_sequence(self,
                            modeling_df: pd.DataFrame,
                            sequence_config: Dict) -> pd.DataFrame:
        """
        Run a sequence of experiments defined in config.
        """
        # Calculate total experiments
        n_experiments = 1 if sequence_config.get('run_baseline', True) else 0

        # Count parameter combinations
        param_vars = sequence_config.get('parameter_variations', {})
        if param_vars:
            variation_paths = self._get_variation_paths(param_vars)
            param_values = [values for _, values in variation_paths]
            n_combinations = len(list(itertools.product(*param_values)))
            n_experiments += n_combinations

        logger.info(f"Beginning experiment sequence with {n_experiments} total experiments")

        # Run baseline first if specified
        current_exp = 1
        if sequence_config.get('run_baseline', True):
            logger.info(f"Running experiment {current_exp}/{n_experiments}: baseline")
            self.run_experiment('baseline', modeling_df)
            current_exp += 1

        # Run parameter variations
        if param_vars:
            logger.info("Starting parameter variation experiments")
            self._run_parameter_variations(modeling_df, param_vars, current_exp, n_experiments)

        logger.info("Experiment sequence complete. Generating comparison.")
        return self.compare_experiments()

    def _run_parameter_variations(self,
                                modeling_df: pd.DataFrame,
                                param_variations: Dict,
                                current_exp: int,
                                total_experiments: int) -> None:
        """Helper method to run parameter sweep experiments"""
        import itertools
        from copy import deepcopy

        # Get all parameter paths and their values
        variation_paths = self._get_variation_paths(param_variations)
        param_names = ['/'.join(path) for path, _ in variation_paths]
        param_values = [values for _, values in variation_paths]

        # Get all combinations
        param_combinations = list(itertools.product(*param_values))

        # Run experiment for each combination
        for params in param_combinations:
            config = deepcopy(self.config)
            param_dict = dict(zip(param_names, params))

            # Update nested config values
            param_desc = []
            for path_str, value in param_dict.items():
                path = path_str.split('/')
                current = config
                for key in path[:-1]:
                    current = current[key]
                current[path[-1]] = value
                param_desc.append(f"{path[-1]}={value}")

            # Create experiment name and log
            exp_name = 'params_' + '_'.join(f"{p.replace('/', '_')}{v}"
                                        for p, v in param_dict.items())
            logger.info(f"Running experiment {current_exp}/{total_experiments}: {', '.join(param_desc)}")

            # Run experiment
            self.run_experiment(exp_name, modeling_df, config)
            current_exp += 1

    def _run_target_variations(self,
                             modeling_df: pd.DataFrame,
                             target_variations: list) -> None:
        """Helper method to run target variable variations"""
        for target in target_variations:
            config = self.config.copy()
            config['modeling']['target_variable'] = target
            self.run_experiment(f'target_{target}', modeling_df, config)

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
