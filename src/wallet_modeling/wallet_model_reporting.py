"""
Functions for generating and storing model training reports and associated data
"""
from typing import Dict,Tuple
import json
import uuid
from datetime import datetime
from pathlib import Path
import logging
import pickle
import pandas as pd
import numpy as np
from dreams_core.googlecloud import GoogleCloud as dgc
import wallet_insights.model_evaluation as wime


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



def save_model_artifacts(model_results, evaluation_dict, configs, base_path):
    """
    Saves all model-related artifacts with a consistent UUID across files.

    Parameters:
    - model_results (dict): Output from train_xgb_model containing pipeline and training data
    - evaluation_dict (dict): Model evaluation metrics and analysis
    - configs (dict): Dictionary containing configuration objects
    - base_path (str): Base path for saving all model artifacts

    Returns:
    - str: The UUID used for this model's artifacts
    """
    def numpy_type_converter(obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Validate required directories exist
    base_dir = Path(base_path)
    required_dirs = ['model_reports', 'wallet_scores', 'wallet_models']
    missing_dirs = [dir_name for dir_name in required_dirs
                    if not (base_dir / dir_name).exists()]
    if missing_dirs:
        raise FileNotFoundError(
            f"Required directories {missing_dirs} not found in {base_dir}. "
            "Please create them before saving model artifacts."
        )

    # 1. Generate model metadata
    model_id = str(uuid.uuid4())
    model_time = datetime.now()
    filename_timestamp = model_time.strftime('%Y%m%d_%Hh%Mm%Ss')
    model_r2 = evaluation_dict['r2']
    model_report_filename = f"model_report_{filename_timestamp}_{model_r2:.3f}_{model_id}.json"


    # 2. Save model pipeline
    pipeline_path = base_dir / 'wallet_models' / f"wallet_model_{model_id}.pkl"
    with open(pipeline_path, 'wb') as f:
        pickle.dump(model_results['pipeline'], f)
    logger.info(f"Saved pipeline to {pipeline_path}")


    # 3. Save model report
    report = {
        'model_id': model_id,
        'model_type': 'wallet',
        'timestamp': model_time.isoformat(),
        'training_data': model_results['training_data'],
        'configurations': configs,
        'evaluation': evaluation_dict
    }

    report_path = base_dir / 'model_reports' / model_report_filename
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=numpy_type_converter)
    logger.info(f"Saved model report to {report_path}")


    # 4. Save wallet scores
    wallet_addresses = get_training_cohort_addresses()
    wallet_scores_df = pd.DataFrame({
        'wallet_id': model_results['training_cohort_pred'].index,
        'score': model_results['training_cohort_pred'],
        'in_modeling_cohort': model_results['training_cohort_pred'].index.isin(model_results['y_test'].index)
    })
    wallet_scores_df = wallet_scores_df.merge(
        wallet_addresses,
        on='wallet_id',
        how='left'
    )
    wallet_scores_path = base_dir / 'wallet_scores' / f"wallet_scores_{model_id}.csv"
    wallet_scores_df.to_csv(wallet_scores_path, index=True)
    logger.info(f"Saved wallet scores and addresses to {wallet_scores_path}")

    return model_id



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



def generate_and_save_wallet_model_artifacts(
    model_results: Dict,
    base_path: str,
    configs: Dict[str, Dict]
) -> Tuple[str, object, pd.DataFrame]:
    """
    Wrapper function to generate evaluations, metrics, and save all model artifacts.
    Uses WalletRegressionEvaluator for model evaluation.

    Parameters:
    - model_results (dict): Output from WalletModel.run_experiment containing:
        - pipeline: The trained model pipeline
        - X_train, X_test: Feature data
        - y_train, y_test: Target data (modeling cohort)
        - y_pred: Model predictions on modeling cohort test set
        - training_cohort_pred: Predictions for full training cohort
    - base_path (str): Base path for saving artifacts
    - configs (dict of dicts): configs to store

    Returns:
    - dict: Dictionary containing:
        - model_id: UUID of the saved artifacts
        - evaluation: Model evaluation metrics
        - wallet_scores: DataFrame of wallet-level predictions
    """
    # 1. Generate model evaluation metrics using WalletRegressionEvaluator
    model = model_results['pipeline'].named_steps['regressor']
    evaluator = wime.RegressionEvaluator(
        y_train=model_results['y_train'],
        y_test=model_results['y_test'],
        y_pred=model_results['y_pred'],
        training_cohort_pred=model_results['training_cohort_pred'],
        training_cohort_actuals=model_results['training_cohort_actuals'],
        model=model,
        feature_names=model_results['pipeline'][:-1].transform(model_results['X_train']).columns.tolist()
    )

    # Create evaluation dictionary with the same structure as before
    evaluation = {
        **evaluator.metrics,
        'summary_report': evaluator.summary_report(),
        'cohort_sizes': {
            'training_cohort': len(model_results['training_cohort_pred']),
            'modeling_cohort': len(model_results['y_pred']),
        }
    }

    # Create wallet scores DataFrame with both cohorts
    wallet_scores_df = pd.DataFrame({
        'score': model_results['training_cohort_pred'],
        'actual': model_results['training_cohort_actuals'],
        'in_modeling_cohort': model_results['training_cohort_pred'].index.isin(model_results['y_test'].index)
    })

    # 5. Save all artifacts
    model_id = save_model_artifacts(
        model_results={
            **model_results,
            'training_data': {
                'n_samples': len(model_results['training_cohort_pred']),
                'n_features': len(model_results['X_train'].columns)
            }
        },
        evaluation_dict=evaluation,
        configs=configs,
        base_path=base_path
    )

    return model_id, evaluator, wallet_scores_df
