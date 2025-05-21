"""
Functions for generating and storing model training reports and associated data
"""
from typing import Dict, Tuple, Any
import json
import uuid
from datetime import datetime,timedelta
from pathlib import Path
import logging
import yaml
import cloudpickle
import pandas as pd
import numpy as np
from dreams_core.googlecloud import GoogleCloud as dgc
import pandas_gbq

# Local modules
import wallet_modeling.wallet_training_data_orchestrator as wtdo
import wallet_insights.model_evaluation as wime
import wallet_insights.wallet_validation_analysis as wiva
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)



# ------------------------------------
#       Main Interface Functions
# ------------------------------------

def generate_and_save_wallet_model_artifacts(
    model_results: Dict,
    base_path: str,
    configs: Dict[str, Dict],
    save_scores: bool = False
) -> Tuple[str, object, pd.DataFrame]:
    """
    Wrapper function to generate evaluations, metrics, and save all model artifacts.
    Uses WalletRegressorEvaluator for model evaluation.

    Parameters:
    - model_results (dict): Output from WalletModel.run_experiment containing:
        - pipeline: The trained model pipeline
        - X_train, X_test: Feature data
        - y_train, y_test: Target data (modeling cohort)
        - y_pred: Model predictions on modeling cohort test set
        - training_cohort_pred: Predictions for full training cohort
    - base_path (str): Base path for saving artifacts
    - configs (dict of dicts): configs to store
    - save_scores (bool): whether to save the wallet-level scores

    Returns:
    - dict: Dictionary containing:
        - model_id: UUID of the saved artifacts
        - evaluation: Model evaluation metrics
        - wallet_scores: DataFrame of wallet-level predictions
    """
    # 1. Generate model evaluation metrics using WalletRegressorEvaluator
    if model_results['model_type'] == 'regression':
        evaluator = wime.RegressorEvaluator(model_results)
    elif model_results['model_type'] == 'classification':
        evaluator = wime.ClassifierEvaluator(model_results)
    else:
        raise ValueError(f"Invalid model type {model_results['model_type']} found in results.")

    # Create evaluation dictionary with the same structure as before
    evaluation = {
        **evaluator.metrics,
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
        base_path=base_path,
        save_scores=save_scores
    )

    return model_id, evaluator, wallet_scores_df



def generate_and_upload_wallet_cw_scores(
    wallets_config: dict,
    training_data_df: pd.DataFrame,
    complete_hybrid_cw_id_df: pd.DataFrame,
    model_id: str,
    score_name: str,
    score_notes: str,
    project_id: str = 'western-verve-411004'
) -> None:
    """
    Generate wallet scores for coin-wallet pairs using the specified model and upload
     to normalized BigQuery tables.

    Params:
    - wallets_config (dict, optional): Configuration dictionary.
    - training_data_df (DataFrame): Multiwindow wallet training data.
    - complete_hybrid_cw_id_df (DataFrame, optional): Hybrid wallet mapping data.
    - model_id (str): Unique ID of the model to use for prediction.
    - score_name (str): Name of the score being generated.
    - score_notes (str): Additional notes about the scoring process.
    - project_id (str, optional): GCP project ID for BigQuery upload.
    """
    # Validate epochs
    epochs = sorted(list(training_data_df.index.get_level_values('epoch_start_date').unique()))
    if len(epochs) > 1:
        raise ValueError("Training data contains more than one epoch. Predictions should only "
                         f"include a single epoch. Epochs found: {epochs}")

    # Generate a unique score_run_id to link metadata with scores
    score_run_id = str(uuid.uuid4())

    # Load and predict
    y_pred = wiva.load_and_predict(
        model_id,
        training_data_df,
        wallets_config['training_data']['model_artifacts_folder']
    )
    wallet_scores_df = pd.DataFrame({
        'score': y_pred
    })

    # Create base df
    wallet_scores_df = wtdo.dehybridize_wallet_address(wallet_scores_df, complete_hybrid_cw_id_df)
    wallet_scores_df = wallet_scores_df.reset_index()

    # Add score_run_id to link with metadata
    wallet_scores_df['score_run_id'] = score_run_id

    # Create metadata dataframe with a single row
    report = load_model_report(model_id, wallets_config['training_data']['model_artifacts_folder'])
    report_model_cfg = report['configurations']['wallets_config']['modeling']
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
        dataframe=wallet_scores_df,
        destination_table='scores.wallet_cw_scores',
        project_id=project_id,
        if_exists='append'
    )
    pandas_gbq.to_gbq(
        dataframe=score_metadata_df,
        destination_table='scores.wallet_cw_scores_metadata',
        project_id=project_id,
        if_exists='append'
    )

    logger.milestone(f"Successfully uploaded score '{score_name}' with {len(wallet_scores_df)} records.")
    u.notify('ui_sound_on')




# ---------------------------------
#         Helper Functions
# ---------------------------------

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



def save_model_artifacts(model_results, evaluation_dict, configs, base_path, save_scores):
    """
    Saves all model-related artifacts with a consistent UUID across files.

    Parameters:
    - model_results (dict): Output from train_xgb_model containing pipeline and training data
    - evaluation_dict (dict): Model evaluation metrics and analysis
    - configs (dict): Dictionary containing configuration objects
    - base_path (str): Base path for saving all model artifacts
    - save_scores (bool): whether to save the wallet-level scores

    Returns:
    - str: The UUID used for this model's artifacts
    """
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
    model_id = model_results['model_id']
    model_time = datetime.now()
    filename_timestamp = model_time.strftime('%Y%m%d_%Hh%Mm%Ss')

    if model_results['model_type'] == 'regression':
        model_r2 = evaluation_dict['r2']
        validation_r2 = evaluation_dict.get('validation_metrics', {}).get('r2', np.nan)
        model_report_filename = (
            f"model_report_{filename_timestamp}__"
            f"mr{model_r2:.3f}__"
            f"{f'vr{validation_r2:.3f}' if not np.isnan(validation_r2) else 'vr___'}"
            f"|{model_id}.json"
        )
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


    # 2. Save model pipeline
    pipeline_path = base_dir / 'wallet_models' / f"wallet_model_{model_id}.pkl"
    with open(pipeline_path, 'wb') as f:
        cloudpickle.dump(model_results['pipeline'], f)
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
        json.dump(report, f, indent=2, default=u.numpy_type_converter)
    logger.info(f"Saved model report to {report_path}")


    # 4. Save wallet scores if configured to
    if save_scores is True:
        wallet_scores_df = pd.DataFrame({
            'wallet_id': model_results['training_cohort_pred'].index,
            'score': model_results['training_cohort_pred'],
            'in_modeling_cohort': model_results['training_cohort_pred'].index.isin(model_results['y_test'].index)
        })

        # Retrieve non-id wallet_address values
        wallet_addresses = get_training_cohort_addresses()
        wallet_scores_df = (
            wallet_scores_df
            .reset_index(level='wallet_address')
            .merge(
                wallet_addresses,
                left_on='wallet_address',
                right_on='wallet_id',
                how='left',
                suffixes=('_orig', '')  # keep right's wallet_address as is
            )
            .set_index('wallet_address', append=True)  # use wallet_address from wallet_addresses
            .drop(['wallet_id', 'wallet_address_orig'], axis=1)  # drop unneeded columns
        )

        wallet_scores_path = base_dir / 'wallet_scores' / f"wallet_scores_{model_id}.csv"
        wallet_scores_df.to_csv(wallet_scores_path, index=True)
        logger.info(f"Saved wallet scores and addresses to {wallet_scores_path}")

    return model_id




# ---------------------------------
#         Utility Functions
# ---------------------------------

def load_model_report(model_id: str, base_path: str, configs_output_path: str = None):
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

    # Load model report by matching filename suffix "|{model_id}.json"
    reports_dir = base_dir / 'model_reports'
    matching_reports = list(reports_dir.glob(f"*|{model_id}.json"))
    if not matching_reports:
        raise FileNotFoundError(f"No model report found for model_id {model_id} in {reports_dir}")
    report_path = matching_reports[0]
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    # Optionally dump each configuration to YAML
    if configs_output_path:
        configs_dir = Path(configs_output_path)
        configs_dir.mkdir(parents=True, exist_ok=True)
        for config_name, config_obj in report.get('configurations', {}).items():
            yaml_path = configs_dir / f"{config_name}.yaml"
            yaml_path = configs_dir / f"{config_name}.yaml"
            with open(yaml_path, 'w', encoding='utf-8') as yaml_file:
                yaml.safe_dump(config_obj, yaml_file)

    return report



def compare_configs(config_1: Dict[str, Any], config_2: Dict[str, Any]) -> None:
    """
    Compare two configuration dicts and log differences.
    """
    all_keys = set(config_1.keys()) | set(config_2.keys())
    for key in sorted(all_keys):
        val1 = config_1.get(key)
        val2 = config_2.get(key)
        if val1 != val2:
            logger.info(
                "Config key '%s' differs: %s vs %s",
                key, val1, val2
            )
