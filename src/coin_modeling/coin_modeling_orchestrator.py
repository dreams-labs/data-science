"""
Orchestrates groups of functions to generate wallet model pipeline
"""
from typing import Dict,Tuple
import logging
import pandas as pd

# Local module imports
import wallet_insights.model_evaluation as wime
import wallet_modeling.wallet_model_reporting as wmr


# Set up logger at the module level
logger = logging.getLogger(__name__)



def generate_and_save_coin_model_artifacts(
    model_results: Dict,
    base_path: str,
    configs: Dict[str, Dict]
) -> Tuple[str, object, pd.DataFrame]:
    """
    Wrapper to generate evaluations and save model artifacts.

    Params:
    - model_results (Dict): Output from model.run_experiment containing pipeline and data
    - base_path (str): Base path for saving artifacts
    - configs (Dict[str, Dict]): Dictionary of named config objects (e.g. {'wallets_config': config_dict})

    Returns:
    - str: model_id used for artifacts
    - object: model evaluator
    - DataFrame: score results
    """
    model = model_results['pipeline'].named_steps['regressor']
    evaluator = wime.WalletRegressionEvaluator(
        y_train=model_results['y_train'],
        y_true=model_results['y_test'],
        y_pred=model_results['y_pred'],
        model=model,
        feature_names=model_results['X_train'].columns.tolist()
    )

    evaluation = {
        **evaluator.metrics,
        'summary_report': evaluator.summary_report(),
        'cohort_sizes': {
            'total_rows': len(model_results['X_train']) + len(model_results['X_test'])
        }
    }

    scores_df = pd.DataFrame({
        'score': model_results['y_pred'],
        'actual': model_results['y_test']
    })

    model_id = wmr.save_model_artifacts(
        model_results={
            **model_results,
            'training_data': {
                'n_samples': len(model_results['y_train']) + len(model_results['y_test']),
                'n_features': len(model_results['X_train'].columns)
            }
        },
        evaluation_dict=evaluation,
        configs=configs,
        base_path=base_path
    )

    return model_id, evaluator, scores_df
