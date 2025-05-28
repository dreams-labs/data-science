"""
Functions for generating and storing model training reports and associated data
"""
from typing import Dict,Tuple
from pathlib import Path
import logging
import uuid
from datetime import datetime,timedelta
import json
import math
import pandas as pd
import numpy as np
import joblib
import pandas_gbq
import matplotlib.pyplot as plt
import wallet_insights.model_evaluation as wime
import wallet_insights.wallet_model_reporting as wimr
import coin_modeling.coin_epochs_orchestrator as ceo

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



def generate_and_upload_coin_scores(
    wallets_coin_config: dict,
    training_data_df: pd.DataFrame,
    model_id: str,
    score_name: str,
    score_notes: str,
    project_id: str = 'western-verve-411004'
) -> None:
    """
    Generate wallet scores for coin-wallet pairs using the specified model and upload
     to normalized BigQuery tables.

    Params:
    - wallets_coin_config (dict, optional): Configuration dictionary.
    - training_data_df (DataFrame): Multiwindow wallet training data.
    - model_id (str): Unique ID of the model to use for prediction.
    - score_name (str): Name of the score being generated.
    - score_notes (str): Additional notes about the scoring process.
    - project_id (str, optional): GCP project ID for BigQuery upload.
    """
    # Validate epochs
    epochs = sorted(list(training_data_df.index.get_level_values('coin_epoch_start_date').unique()))
    if len(epochs) > 1:
        raise ValueError("Training data contains more than one epoch. Predictions should only "
                            f"include a single epoch. Epochs found: {epochs}")

    # Generate a unique score_run_id to link metadata with scores
    score_run_id = str(uuid.uuid4())

    # Load and predict
    y_pred = ceo.CoinEpochsOrchestrator.score_coin_training_data(
        wallets_coin_config,
        model_id,
        wallets_coin_config['training_data']['model_artifacts_folder'],
        training_data_df,
    )

    # Create base df
    coin_scores_df = y_pred.reset_index()

    # Add score_run_id to link with metadata
    coin_scores_df['score_run_id'] = score_run_id

    # Create metadata dataframe with a single row
    report = wimr.load_model_report(
        model_id,
        wallets_coin_config['training_data']['model_artifacts_folder']
    )
    report_model_cfg = report['configurations']['wallets_coin_config']['coin_modeling']
    epoch_duration = report['configurations']['wallets_config']['training_data']['modeling_period_duration']
    epoch_end_date = (epochs[0] + timedelta(days=epoch_duration))
    if report_model_cfg['model_type'] == 'regression':
        target_threshold = np.nan
    else:
        target_threshold = report_model_cfg['target_var_min_threshold']

    score_metadata_df = pd.DataFrame({
        'score_run_id': [score_run_id],
        'epoch_end_date': [epoch_end_date],
        'model_id': [report['model_id']],
        'score_name': [score_name],
        'scored_at': [report['timestamp']],
        'model_type': [report_model_cfg['model_type']],
        'target_var': [report_model_cfg['target_variable']],
        'target_var_threshold': [target_threshold],
        'notes': [score_notes],
        'updated_at': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    })

    # Upload to BigQuery
    pandas_gbq.to_gbq(
        dataframe=coin_scores_df,
        destination_table='scores.coin_scores',
        project_id=project_id,
        if_exists='append'
    )
    pandas_gbq.to_gbq(
        dataframe=score_metadata_df,
        destination_table='scores.coin_scores_metadata',
        project_id=project_id,
        if_exists='append'
    )

    logger.milestone(f"Successfully uploaded score '{score_name}' with {len(coin_scores_df)} records.")
    u.notify('ui_sound_on')



def plot_wallet_model_comparison(
    wallets_coin_config: dict,
    metric: str = 'wins_return',
    figsize: tuple = (12, 20),
    cols: int = 2,
) -> None:
    """
    Plot comparison of wallet model return metrics across different epochs.
    Creates separate subplot for each model in a flexible column layout.

    Params:
    - wallets_coin_config: Configuration containing parquet folder paths
    - metric: 'wins_return' or 'mean_return'
    - figsize: Figure dimensions
    - cols: how many columns of charts to make
    """
    base_folder = Path(wallets_coin_config['training_data']['parquet_folder'])

    # Find all wallet_model_ids.json files
    json_files = list(base_folder.glob('*/wallet_model_ids.json'))

    if not json_files:
        raise FileNotFoundError(f"No wallet_model_ids.json files found in {base_folder}")

    # Get all unique model names from actual JSON files
    all_model_names = set()
    for json_path in json_files:
        with open(json_path, 'r', encoding='utf-8') as f:
            models_dict = json.load(f)
            all_model_names.update(models_dict.keys())

    model_names = sorted(list(all_model_names))

    if not model_names:
        raise ValueError("No models found in JSON files")

    # Load and combine all return metrics
    combined_data = []

    for json_path in json_files:
        epoch_date = json_path.parent.name

        with open(json_path, 'r', encoding='utf-8') as f:
            models_dict = json.load(f)

        for model_name, model_data in models_dict.items():
            if 'return_metrics' not in model_data:
                continue

            return_metrics = model_data['return_metrics']

            # Create records for each bucket
            for i, bucket in enumerate(return_metrics['bucket']):
                combined_data.append({
                    'epoch_date': epoch_date,
                    'model_name': model_name,
                    'bucket': bucket,
                    'mean_return': return_metrics['mean_return'][i],
                    'wins_return': return_metrics['wins_return'][i]
                })

    if not combined_data:
        raise ValueError("No return metrics data found")

    # Convert to DataFrame
    df = pd.DataFrame(combined_data)

    # Calculate subplot layout
    n_models = len(model_names)
    rows = math.ceil(n_models / cols)

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Handle axes indexing for different subplot configurations
    if n_models == 1:
        axes_flat = [axes]
    elif rows == 1:
        axes_flat = axes if cols > 1 else [axes]
    else:
        axes_flat = axes.flatten()

    # Plot each model in its own subplot
    for i, model_name in enumerate(model_names):
        ax = axes_flat[i]
        model_data = df[df['model_name'] == model_name]

        if model_data.empty:
            ax.text(0.5, 0.5, f'No data for {model_name}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model_name)
            continue

        # Plot separate line for each epoch
        for epoch_date, group in model_data.groupby('epoch_date'):
            ax.plot(group['bucket'], group[metric],
                   marker='o', linewidth=2, label=f'Epoch {epoch_date}')

        ax.set_xlabel('Prediction Rank Bucket (1 = highest scores)')
        ax.set_ylabel(f'{metric.replace("_", " ").title()}')
        ax.set_title(model_name)
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Hide any unused subplots
    for i in range(n_models, len(axes_flat)):
        axes_flat[i].axis('off')

    plt.suptitle(f'Wallet Model {metric.replace("_", " ").title()} Comparison by Epoch',
                 fontsize=16, y=0.995)
    plt.tight_layout()
    plt.show()




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
            f"|{model_id}.json"
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



def get_all_wallet_model_paths(wallets_coin_config: dict) -> list:
    """
    Get all wallet_model_ids.json file paths for analysis.

    Returns:
    - List of Path objects to wallet_model_ids.json files
    """
    base_folder = Path(wallets_coin_config['training_data']['parquet_folder'])
    return list(base_folder.glob('*/wallet_model_ids.json'))
