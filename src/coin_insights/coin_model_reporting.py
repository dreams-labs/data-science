"""
Functions for generating and storing model training reports and associated data
"""
from typing import Dict,Tuple
from pathlib import Path
import logging
from datetime import datetime
import json
import pandas as pd
import numpy as np
import wallet_insights.model_evaluation as wime
import joblib

# Local modules
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)


# -----------------------------------
#       Main Interface Function
# -----------------------------------

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
    - configs (Dict[str, Dict]): Dictionary of named config objects

    Returns:
    - str: model_id used for artifacts
    - object: model evaluator
    - DataFrame: score results
    """
    # 1. Generate model evaluation metrics using WalletRegressorEvaluator
    if model_results['model_type'] == 'regression':
        evaluator = wime.RegressorEvaluator(model_results)
    elif model_results['model_type'] == 'classification':
        evaluator = wime.ClassifierEvaluator(model_results)
    else:
        raise ValueError(f"Invalid model type {model_results['model_type']} found in results.")


    evaluation = {
        **evaluator.metrics,
        'summary_report': evaluator.summary_report(),
        'cohort_sizes': {
            'total_rows': len(model_results['X_train']) + len(model_results['X_test'])
        }
    }

    coin_scores_df = pd.DataFrame({
        'score': model_results['y_pred'],
        'actual': model_results['y_test']
    })

    model_results_artifacts={
        **model_results,
        'training_data': {
            'n_samples': len(model_results['y_train']) + len(model_results['y_test']),
            'n_features': len(model_results['X_train'].columns)
        },
    }

    # 5. Save all artifacts
    model_id = save_coin_model_artifacts(
        model_results=model_results_artifacts,
        evaluation_dict=evaluation,
        pipeline=model_results['pipeline'],
        configs=configs,
        base_path=base_path
    )

    return model_id, evaluator, coin_scores_df



# ---------------------------------
#         Helper Functions
# ---------------------------------

def save_coin_model_artifacts(model_results, evaluation_dict, pipeline, configs, base_path):
    """
    Saves all model-related artifacts with a consistent UUID across files.

    Parameters:
    - model_results (dict): Output from train_xgb_model containing pipeline and training data
    - evaluation_dict (dict): Model evaluation metrics and analysis
    - model (xgboost model): Raw XGB model
    - configs (dict): Dictionary containing configuration objects
    - base_path (str): Base path for saving all model artifacts

    Returns:
    - str: The UUID used for this model's artifacts
    """
    # Validate required directories exist
    base_dir = Path(base_path)
    required_dirs = ['model_reports', 'coin_models', 'coin_scores']
    missing_dirs = [dir_name for dir_name in required_dirs
                    if not (base_dir / dir_name).exists()]
    if missing_dirs:
        raise FileNotFoundError(
            f"Required directories {missing_dirs} not found in {base_dir}. "
            "Please create them before saving model artifacts."
        )

    # Generate additional metadata for the filename
    model_id = model_results['model_id']
    model_time = datetime.now()
    filename_timestamp = model_time.strftime('%Y%m%d_%Hh%Mm%Ss')

    if model_results['model_type'] == 'regression':
        model_r2 = evaluation_dict['r2']
        validation_r2 = evaluation_dict.get('validation_metrics', {}).get('r2', np.nan)
        model_report_filename = (
            f"model_report_{filename_timestamp}__"
            f"mr{model_r2:.3f}__"
            f"{f'vr{validation_r2:.3f}' if not np.isnan(validation_r2) else 'vr___'}.json"
        )
        base_dir = Path(base_path)
    elif model_results['model_type'] == 'classification':
        model_auc = evaluation_dict['roc_auc']
        validation_auc = evaluation_dict.get('val_roc_auc', np.nan)
        model_report_filename = (
            f"model_report_{filename_timestamp}__"
            f"mauc{model_auc:.3f}__"
            f"{f'vauc{validation_auc:.3f}' if not np.isnan(validation_auc) else 'vauc___'}"
            f"|{model_id}.json"
        )
    else:
        raise ValueError(f"Invalid model type {model_results['model_type']} found in results object.")

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory {base_dir} does not exist")

    # Create necessary directories
    for dir_name in ['model_reports', 'coin_scores']:
        (base_dir / dir_name).mkdir(parents=True, exist_ok=True)

    # 1. Save model report
    report = {
        'model_id': model_id,
        'model_type': 'coin',
        'timestamp': model_time.isoformat(),
        'training_data': model_results['training_data'],
        'configurations': configs,
        'evaluation': evaluation_dict
    }
    report_path = base_dir / 'model_reports' / model_report_filename
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=u.numpy_type_converter)
    logger.info(f"Saved model report to {report_path}")

    # Save full transformation+estimator pipeline
    models_dir = base_dir / 'coin_models'
    models_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = models_dir / f'coin_model_pipeline_{model_id}.pkl'
    joblib.dump(pipeline, pipeline_path)
    logger.info(f"Saved coin model pipeline to {pipeline_path}")

    # Save scores
    coin_scores_df = pd.DataFrame(model_results['y_pred'])
    coin_scores_df.columns = ['y_pred']
    coin_scores_path = base_dir / 'coin_scores' / f"coin_scores_{model_id}.csv"
    coin_scores_df.to_csv(coin_scores_path, index=True)
    logger.info(f"Saved coin scores and addresses to {coin_scores_path}")

    return model_id
