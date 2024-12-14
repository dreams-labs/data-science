"""
Functions for generating and storing model training reports and associated data
"""
import json
import uuid
from datetime import datetime
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import yaml
import utils as u
from dreams_core.googlecloud import GoogleCloud as dgc
import wallet_insights.wallet_model_evaluation as wime
import wallet_insights.coin_forecasting as wicf
from wallet_modeling.wallets_config_manager import WalletsConfig


logger = logging.getLogger(__name__)

def get_wallet_addresses():
    """
    Retrieves wallet addresses for the current modeling cohort from BigQuery.

    Returns:
    - pd.DataFrame: DataFrame containing wallet_id and wallet_address mappings
    """
    wallet_query = """
    select wc.wallet_id, wi.wallet_address
    from `temp.wallet_modeling_cohort` wc
    join `reference.wallet_ids` wi on wi.wallet_id = wc.wallet_id
    """
    wallet_addresses_df = dgc().run_sql(wallet_query)

    logger.debug(f"Retrieved {len(wallet_addresses_df)} wallet addresses")
    return wallet_addresses_df



def save_model_artifacts(model_results, evaluation_dict, configs, coin_validation_df, base_path):
    """
    Saves all model-related artifacts with a consistent UUID across files.

    Parameters:
    - model_results (dict): Output from train_xgb_model containing pipeline and training data
    - evaluation_dict (dict): Model evaluation metrics and analysis
    - configs (dict): Dictionary containing configuration objects:
        - wallets_config: Main wallet modeling configuration
        - wallets_metrics_config: Metrics calculation configuration
        - wallets_features_config: Feature engineering configuration
    - coin_validation_df (pd.DataFrame): DataFrame containing coin-level metrics and validation results
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
    base_dir = Path(base_path)

    # Create necessary directories
    for dir_name in ['model_reports', 'wallet_scores', 'coin_metrics']:
        (base_dir / dir_name).mkdir(parents=True, exist_ok=True)

    # 1. Save model report
    report = {
        'model_id': model_id,
        'timestamp': datetime.now().isoformat(),
        'training_data': {
            'n_samples': model_results['X'].shape[0] if 'X' in model_results else None,
            'n_features': model_results['X'].shape[1] if 'X' in model_results else None
        },
        'configurations': configs,
        'evaluation': evaluation_dict
    }

    report_path = base_dir / 'model_reports' / f"model_report_{model_id}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=numpy_type_converter)
    logger.info(f"Saved model report to {report_path}")

    # 2. Get wallet addresses and save wallet scores
    wallet_addresses = get_wallet_addresses()
    wallet_scores_df = pd.DataFrame({
        'wallet_id': model_results['y_test'].index,
        'score': model_results['y_pred']
    })
    wallet_scores_df = wallet_scores_df.merge(
        wallet_addresses,
        on='wallet_id',
        how='left'
    )
    wallet_scores_path = base_dir / 'wallet_scores' / f"wallet_scores_{model_id}.csv"
    wallet_scores_df.to_csv(wallet_scores_path, index=False)
    logger.info(f"Saved wallet scores and addresses to {wallet_scores_path}")

    # 3. Save coin metrics
    coin_metrics_path = base_dir / 'coin_metrics' / f"coin_metrics_{model_id}.csv"
    coin_validation_df.to_csv(coin_metrics_path, index=True)
    logger.info(f"Saved coin metrics to {coin_metrics_path}")

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
        - coin_metrics: DataFrame of coin-level metrics
    """
    base_dir = Path(base_path)

    # Load model report
    report_path = base_dir / 'model_reports' / f"model_report_{model_id}.json"
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    # Load wallet scores
    wallet_scores_path = base_dir / 'wallet_scores' / f"wallet_scores_{model_id}.csv"
    wallet_scores = pd.read_csv(wallet_scores_path)

    # Load coin metrics
    coin_metrics_path = base_dir / 'coin_metrics' / f"coin_metrics_{model_id}.csv"
    coin_metrics = pd.read_csv(coin_metrics_path, index_col=0)

    return {
        'report': report,
        'wallet_scores': wallet_scores,
        'coin_metrics': coin_metrics
    }



def generate_and_save_model_artifacts(model_results, validation_profits_df, base_path):
    """
    Wrapper function to generate evaluations, metrics, and save all model artifacts.
    Uses RegressionEvaluator for model evaluation.

    Parameters:
    - model_results (dict): Output from train_xgb_model containing:
        - pipeline: The trained model pipeline
        - X, X_train, X_test: Feature data
        - y_train, y_test: Target data
        - y_pred: Model predictions
    - validation_profits_df (pd.DataFrame): Profits DataFrame for validation period
    - base_path (str): Base path for saving artifacts

    Returns:
    - dict: Dictionary containing:
        - model_id: UUID of the saved artifacts
        - evaluation: Model evaluation metrics
        - wallet_scores: DataFrame of wallet-level predictions
        - coin_validation: DataFrame of coin-level metrics
    """
    # 1. Generate model evaluation metrics using RegressionEvaluator
    model = model_results['pipeline'].named_steps['regressor']
    evaluator = wime.RegressionEvaluator(
        y_true=model_results['y_test'],
        y_pred=model_results['y_pred'],
        model=model,
        feature_names=model_results['X_train'].columns.tolist()
    )

    # Create evaluation dictionary with the same structure as before
    evaluation = {
        **evaluator.metrics,  # Include all basic metrics
        'summary_report': evaluator.get_summary_report()
    }

    # 2. Create wallet scores DataFrame
    wallet_scores_df = pd.DataFrame({
        'score': model_results['y_pred']
    }, index=model_results['y_test'].index)

    # 3. Calculate coin-level metrics
    coin_validation_df = wicf.calculate_coin_metrics_from_wallet_scores(
        validation_profits_df,
        wallet_scores_df
    )

    # 4. Load configurations
    wallets_config = WalletsConfig()
    wallets_metrics_config = u.load_config('../config/wallets_metrics_config.yaml')
    wallets_features_config = yaml.safe_load(
        Path('../config/wallets_features_config.yaml').read_text(encoding='utf-8')
    )

    configs = {
        'wallets_config': wallets_config.config,
        'wallets_metrics_config': wallets_metrics_config,
        'wallets_features_config': wallets_features_config
    }

    # 5. Save all artifacts
    model_id = save_model_artifacts(
        model_results=model_results,
        evaluation_dict=evaluation,
        configs=configs,
        coin_validation_df=coin_validation_df,
        base_path=base_path
    )

    return model_id, evaluator, wallet_scores_df, coin_validation_df
