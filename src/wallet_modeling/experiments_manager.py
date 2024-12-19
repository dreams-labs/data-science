import logging
import itertools
from copy import deepcopy
from typing import Dict, Optional
import pandas as pd

# Local module imports
import wallet_modeling.wallet_modeling_orchestrator as wmo
import wallet_modeling.wallet_model as wm
import wallet_modeling.model_reporting as wmr
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
                validation_profits_df: pd.DataFrame,  # Add validation data
                base_path: str = '../wallet_modeling'):  # Add base path for artifacts
        """
        Params:
        - config (dict): configuration dictionary containing modeling parameters
        - training_data_df (DataFrame): pre-computed training features
        - validation_profits_df (DataFrame): profits data for validation period
        - base_path (str): path for saving model artifacts
        """
        self.config = config
        self.training_data_df = training_data_df
        self.validation_profits_df = validation_profits_df
        self.base_path = base_path
        self.experiments: Dict[str, Dict] = {}
        self.model_results: Dict[str, Dict] = {}

    def prepare_modeling_data(self, modeling_profits_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        Params:
        - modeling_profits_df (DataFrame): raw profits data from the modeling period only
        - config (Dict): experiment-specific config to use

        Returns:
        - modeling_df (DataFrame): prepared modeling DataFrame with all target variables
        """
        logger.info("Preparing modeling data")

        # Create modeling dataset using existing pipeline
        modeling_wallets_df = wmo.filter_modeling_period_wallets(modeling_profits_df)
        target_vars_df = wpf.calculate_performance_features(modeling_wallets_df)

        # Use experiment-specific target variable
        target_var = config['modeling']['target_variable']
        logger.info(f"Using target variable: {target_var}")

        # Join with target variable
        modeling_df = self.training_data_df.join(
            target_vars_df[target_var],
            how=config['modeling'].get('join_type', 'inner')
        )

        return modeling_df

    def run_experiment(self,
                    experiment_name: str,
                    modeling_profits_df: pd.DataFrame,
                    experiment_config: Optional[dict] = None) -> Dict:
        """
        Params:
        - experiment_name (str): unique identifier for this experiment
        - modeling_profits_df (DataFrame): raw profits data from modeling period
        - experiment_config (dict, optional): override default config for this experiment

        Returns:
        - results (dict): experiment results including model and metrics
        """
        # Use experiment-specific config if provided
        config = experiment_config or self.config

        # Generate modeling data with this experiment's config
        logger.info(f"Preparing data for experiment: {experiment_name}")
        modeling_df = self.prepare_modeling_data(modeling_profits_df, config)

        # Initialize and run experiment
        logger.info(f"Initializing model for experiment: {experiment_name}")
        experiment = wm.WalletModel(config)
        results = experiment.run_experiment(modeling_df)

        # Save model artifacts
        logger.info(f"Saving artifacts for experiment: {experiment_name}")
        model_id, evaluator, wallet_scores_df = wmr.generate_and_save_model_artifacts(
            model_results=results,
            validation_profits_df=self.validation_profits_df,
            base_path=self.base_path
        )

        # Calculate metrics using evaluator
        metrics = evaluator.metrics
        logger.info(f"Experiment {experiment_name} complete. Metrics: {metrics}")

        # Store results with additional data
        self.experiments[experiment_name] = experiment
        self.model_results[experiment_name] = {
            'results': results,
            'metrics': metrics,
            'model_id': model_id,
            'evaluator': evaluator,
            'wallet_scores': wallet_scores_df,

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
