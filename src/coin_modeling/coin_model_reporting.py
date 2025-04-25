"""
Functions for generating and storing model training reports and associated data
"""
from typing import Dict,Tuple
from pathlib import Path
import logging
from datetime import datetime
import uuid
import json
import pandas as pd
import numpy as np
import wallet_insights.model_evaluation as wime

logger = logging.getLogger(__name__)


def save_coin_model_artifacts(model_results, evaluation_dict, configs, base_path):
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

    # Generate single UUID for all artifacts
    model_id = str(uuid.uuid4())

    # Generate additional metadata for the filename
    model_time = datetime.now()
    filename_timestamp = model_time.strftime('%Y%m%d_%Hh%Mm%Ss')
    model_r2 = evaluation_dict['r2']
    model_report_filename = f"coin_model_report_{filename_timestamp}_{model_r2:.3f}_{model_id}.json"
    base_dir = Path(base_path)

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
        json.dump(report, f, indent=2, default=numpy_type_converter)
    logger.info(f"Saved model report to {report_path}")

    # Save scores
    coin_scores_df = pd.DataFrame(model_results['y_pred'])
    coin_scores_df.columns = ['y_pred']
    coin_scores_path = base_dir / 'coin_scores' / f"coin_scores_{model_id}.csv"
    coin_scores_df.to_csv(coin_scores_path, index=True)
    logger.info(f"Saved coin scores and addresses to {coin_scores_path}")

    return model_id



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
    evaluator = wime.RegressorEvaluator(
        y_train=model_results['y_train'],
        y_test=model_results['y_test'],
        y_pred=model_results['y_pred'],
        model=model_results['pipeline'].named_steps['regressor'],
        feature_names=model_results['pipeline'][:-1].transform(model_results['X_train']).columns.tolist()
    )

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
        configs=configs,
        base_path=base_path
    )

    return model_id, evaluator, coin_scores_df
