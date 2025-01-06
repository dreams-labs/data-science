import logging
from typing import Dict
import pandas as pd
from sklearn.metrics import r2_score


# Local modules
from wallet_modeling.wallet_model import WalletModel
import utils as u

# pylint:disable=invalid-name  # X_test isn't camelcase
# pylint: disable=W0201  # Attribute defined outside __init__, false positive due to inheritance

# Set up logger at the module level
logger = logging.getLogger(__name__)


# WalletModel Experiments Orchestrator
class WalletExperimentsOrchestrator:
    """
    Orchestrates wallet model experiments to find optimal parameters.
    Currently focused on target variable selection.
    """
    def __init__(self, config_base: Dict, config_experiment: Dict):

        self.config_base = config_base
        self.config_experiment = config_experiment
        self.model_outcomes = []

        # alidate that the config structures match
        self._validate_configs()


    # -----------------------------------
    #         Primary Interface
    # -----------------------------------

    @u.timing_decorator
    def orchestrate_wallet_experiment(
        self,
        training_data_df: pd.DataFrame,
        modeling_cohort_target_var_df: pd.DataFrame
    ) -> Dict:
        """
        Orchestrates modeling workflow across multiple experiment configurations.

        Params:
            training_data_df (DataFrame): Training dataset
            modeling_cohort_target_var_df (DataFrame): Target variable dataset

        Returns:
            Dict: Summary of experiment results and best model
        """
        # Generate experiment configurations
        experiment_configs = self._generate_experiment_configs()

        # Run experiments
        for config in experiment_configs:
            model_outcome = self._construct_experiment_model(
                config,
                training_data_df,
                modeling_cohort_target_var_df
            )
            self.model_outcomes.append(model_outcome)

        # Analyze and select best model
        return self._analyze_experiment_outcome()




    # -----------------------------------
    #           Helper Methods
    # -----------------------------------


    def _generate_experiment_configs(self) -> list:
        """
        Generates list of experiment configs based on base config and experiment overrides.

        Returns:
            List[Dict]: List of complete configs with experiment parameters
        """
        experiment_configs = []

        # Extract target variables from experiment config
        target_vars = self.config_experiment['modeling']['target_variable']

        for target_var in target_vars:
            # Create deep copy to avoid modifying base config
            experiment_config = self.config_base.copy()

            # Override target variable
            experiment_config['modeling']['target_variable'] = target_var

            experiment_configs.append(experiment_config)

        logger.info(f"Generated {len(experiment_configs)} experiment configurations")
        return experiment_configs



    def _construct_experiment_model(
        self,
        experiment_config: Dict,
        training_data_df: pd.DataFrame,
        modeling_cohort_target_var_df: pd.DataFrame
    ) -> Dict:
        """
        Constructs single wallet model with experiment configuration.

        Params:
            experiment_config (Dict): Configuration for this experiment iteration
            training_data_df (DataFrame): Training data
            modeling_cohort_target_var_df (DataFrame): Target variable data

        Returns:
            Dict: Model outputs including metrics and parameters
        """
        try:
            # Initialize model with experiment config
            wallet_model = WalletModel(experiment_config)

            # Construct and get results
            model_output = wallet_model.construct_wallet_model(
                training_data_df,
                modeling_cohort_target_var_df
            )

            # Add experiment metadata
            model_output['experiment_params'] = {
                'target_variable': experiment_config['modeling']['target_variable']
            }

            logger.info(f"Successfully constructed model with target: {experiment_config['modeling']['target_variable']}")
            return model_output

        except Exception as e:
            logger.error(f"Failed to construct model: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'experiment_params': experiment_config['modeling']
            }


    def _analyze_experiment_outcome(self) -> Dict:
        """
        Basic analysis printing R2 scores for each model experiment.

        Returns:
            Dict: Simple summary of experiment results
        """
        # Filter out failed experiments
        successful_experiments = [
            outcome for outcome in self.model_outcomes
            if outcome.get('status') != 'failed'
        ]

        if not successful_experiments:
            logger.error("No successful experiments to analyze")
            return {'status': 'failed', 'error': 'No successful experiments'}

        # Print results for each experiment
        print("\nExperiment Results:")
        print("-" * 50)

        for outcome in successful_experiments:
            r2 = r2_score(outcome['y_test'], outcome['y_pred'])
            target = outcome['experiment_params']['target_variable']
            print(f"Target: {target}")
            print(f"R2 Score: {r2:.4f}\n")

        return {
            'status': 'success',
            'total_experiments': len(self.model_outcomes),
            'successful_experiments': len(successful_experiments)
        }


    # -----------------------------------
    #           Utility Methods
    # -----------------------------------

    def _validate_configs(self):
        """Validates experiment config keys are valid by checking against base config"""
        invalid_keys = set(self.config_experiment.keys()) - set(self.config_base.keys())
        if invalid_keys:
            raise ValueError(f"Invalid keys in experiment config: {invalid_keys}. "
                           f"All keys must exist in base config.")
