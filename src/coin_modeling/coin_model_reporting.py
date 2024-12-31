"""
Functions for generating and storing model training reports and associated data
"""
from typing import Dict,Tuple
import json
from pathlib import Path
import logging
import pandas as pd
from dreams_core.googlecloud import GoogleCloud as dgc
import wallet_insights.wallet_model_evaluation as wime
from wallet_modeling.wallet_model_reporting import wmr


logger = logging.getLogger(__name__)

def get_training_cohort_addresses():
    """
    Retrieves wallet addresses for the full training cohort from BigQuery.

    Returns:
    - pd.DataFrame: DataFrame containing wallet_id and wallet_address mappings
    """
    wallet_query = """
    select wc.wallet_id, wi.wallet_address
    from `temp.wallet_modeling_training_cohort` wc  # Update table name
    join `reference.wallet_ids` wi on wi.wallet_id = wc.wallet_id
    """
    wallet_addresses_df = dgc().run_sql(wallet_query)
    logger.debug(f"Retrieved training cohort of {len(wallet_addresses_df)} wallet addresses")

    return wallet_addresses_df



def load_model_artifacts(model_id, base_path):
    """
    Loads all artifacts associated with a specific model ID

    Parameters:
    - model_id (str): UUID of the model to load
    - base_path (str): Base path where model artifacts are stored

    Returns:
    - dict: Dictionary containing:
        - report: Model report dictionary
        - wallet_scores: DataFrame of wallet-level scores
    """
    base_dir = Path(base_path)

    # Load model report
    report_path = base_dir / 'model_reports' / f"model_report_{model_id}.json"
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    # Load wallet scores
    wallet_scores_path = base_dir / 'wallet_scores' / f"wallet_scores_{model_id}.csv"
    wallet_scores = pd.read_csv(wallet_scores_path)

    return {
        'report': report,
        'wallet_scores': wallet_scores,
    }



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