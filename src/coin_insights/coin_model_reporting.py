"""
Functions for generating and storing model training reports and associated data
"""
from typing import Dict,Tuple
from pathlib import Path
import logging
import uuid
from datetime import datetime,timedelta
import json
import pandas as pd
import numpy as np
import joblib
import pandas_gbq
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
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
    cols: int = 2,
    macro_comparison: str = None,
) -> None:
    """
    Plot comparison of wallet model return metrics across different epochs.
    Creates paired subplots for each model showing wins_return and mean_return side by side.
    Each model gets two charts per row: wins_return (left) and mean_return (right).
    Validates that return data contains exactly 20 items before plotting.

    Params:
    - wallets_coin_config: Configuration containing parquet folder paths
    - cols: how many columns of charts to make (kept for backward compatibility but forced to 2)
    - macro_comparison: macro key for color coding (e.g. 'btc_mvrv_z_score_last|w4')
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

    # Load and combine all return metrics with validation for both metrics
    combined_data = []
    macro_values = []

    for json_path in json_files:
        epoch_date = json_path.parent.name

        with open(json_path, 'r', encoding='utf-8') as f:
            models_dict = json.load(f)

        for model_name, model_data in models_dict.items():
            if 'return_metrics' not in model_data:
                continue

            return_metrics = model_data['return_metrics']

            # Validate that both metric data contains exactly 20 items
            wins_valid = 'wins_return' in return_metrics and len(return_metrics['wins_return']) == 20
            mean_valid = 'mean_return' in return_metrics and len(return_metrics['mean_return']) == 20

            if not (wins_valid and mean_valid):
                logger.warning(f"Skipping {model_name} epoch {epoch_date}: Missing or invalid return metrics data")
                continue

            # Extract macro value for color coding if specified
            macro_value = None
            if macro_comparison and 'macro_averages' in model_data:
                macro_key = f'macro|{macro_comparison}'
                macro_value = model_data['macro_averages'].get(macro_key)

            # Create records for each position (1-20) for both metrics
            for i in range(20):
                combined_data.append({
                    'epoch_date': epoch_date,
                    'model_name': model_name,
                    'position': i + 1,  # 1-indexed positions
                    'wins_return': return_metrics['wins_return'][i],
                    'mean_return': return_metrics['mean_return'][i],
                    'macro_value': macro_value
                })

            # Collect macro values for color scale calculation
            if macro_value is not None:
                macro_values.append(macro_value)

    if not combined_data:
        logger.warning("No valid return metrics data found after validation")
        return

    # Convert to DataFrame
    df = pd.DataFrame(combined_data)

    # Calculate color mapping if macro_comparison is specified
    color_map = {}
    if macro_comparison and macro_values:
        macro_values = np.array(macro_values)
        min_val, max_val = np.min(macro_values), np.max(macro_values)
        median_val = np.median(macro_values)

        # Create color mapping for each unique macro value
        unique_macro_values = np.unique(macro_values)
        for val in unique_macro_values:
            if val < median_val:
                # Scale from red to light grey
                norm_val = (val - min_val) / (median_val - min_val) if median_val != min_val else 0
                color_map[val] = plt.cm.Reds(0.3 + 0.7 * (1 - norm_val))  # pylint:disable=no-member
            elif val > median_val:
                # Scale from light grey to blue
                norm_val = (val - median_val) / (max_val - median_val) if max_val != median_val else 0
                color_map[val] = plt.cm.Blues(0.3 + 0.7 * norm_val)  # pylint:disable=no-member
            else:
                # Median value - light grey
                color_map[val] = '#D3D3D3'

    # Get models that have valid data
    valid_model_names = sorted(df['model_name'].unique())
    n_models = len(valid_model_names)

    if n_models == 0:
        logger.warning("No models with valid data to plot")
        return

    # Force 2 columns (wins_return, mean_return) per row per model
    rows = n_models
    cols = 2

    # Create subplots with 15x15 size per chart
    fig_width = cols * 8
    fig_height = rows * 7
    _, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    # Handle axes indexing for different subplot configurations
    if n_models == 1:
        axes_flat = [axes] if cols == 1 else axes
    else:
        axes_flat = axes

    # Plot each model with paired charts (wins_return left, mean_return right)
    for i, model_name in enumerate(valid_model_names):
        model_data = df[df['model_name'] == model_name]

        # Get axes for this model's pair of charts
        if n_models == 1:
            ax_wins = axes_flat[0] if cols > 1 else axes_flat
            ax_mean = axes_flat[1] if cols > 1 else axes_flat
        else:
            ax_wins = axes_flat[i, 0]
            ax_mean = axes_flat[i, 1]

        # Plot both metrics for this model
        for metric_name, ax in [('wins_return', ax_wins), ('mean_return', ax_mean)]:
            # Plot separate line for each epoch
            for epoch_date, group in model_data.groupby('epoch_date'):
                if macro_comparison and not group['macro_value'].isna().all():
                    # Use macro-based color
                    macro_val = group['macro_value'].iloc[0]
                    line_color = color_map.get(macro_val, '#808080')
                    label = f'Epoch {epoch_date} ({macro_comparison}={macro_val:.3f})'
                else:
                    # Default color scheme
                    line_color = None
                    label = f'Epoch {epoch_date}'

                ax.plot(group['position'], group[metric_name],
                       marker='o', linewidth=2, label=label, color=line_color, alpha=0.3)

            # Calculate and plot overall average across all epochs for this model
            overall_average = model_data.groupby('position')[metric_name].mean()
            ax.plot(overall_average.index, overall_average.values,
                   color='lime', linewidth=5, label='Overall Average', zorder=10)

            # Calculate macro-conditional averages if macro_comparison is specified
            label_positions = [1, 5, 10, 15, 20]
            if macro_comparison and not model_data['macro_value'].isna().all():
                # Get median of macro values for this model
                macro_median = model_data['macro_value'].median()

                # Split data into below/above median (excluding exact median values)
                below_median_data = model_data[model_data['macro_value'] < macro_median]
                above_median_data = model_data[model_data['macro_value'] > macro_median]

                # Plot below median average (light red)
                if not below_median_data.empty:
                    below_average = below_median_data.groupby('position')[metric_name].mean()
                    ax.plot(below_average.index, below_average.values,
                           color='lightcoral', linewidth=5, label='Below Median Macro', zorder=9)

                    # Add text labels for below median line
                    for pos in label_positions:
                        if pos in below_average.index:
                            value = below_average[pos]
                            ax.text(pos, value, f'{value:.3f}',
                                   color='lightcoral', fontweight='bold', fontsize=10,
                                   ha='center', va='top', zorder=11,
                                   path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])

                # Plot above median average (light blue)
                if not above_median_data.empty:
                    above_average = above_median_data.groupby('position')[metric_name].mean()
                    ax.plot(above_average.index, above_average.values,
                           color='lightblue', linewidth=5, label='Above Median Macro', zorder=9)

                    # Add text labels for above median line
                    for pos in label_positions:
                        if pos in above_average.index:
                            value = above_average[pos]
                            ax.text(pos, value, f'{value:.3f}',
                                   color='lightblue', fontweight='bold', fontsize=10,
                                   ha='center', va='top', zorder=11,
                                   path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])

            # Add text labels for specific positions on the bright green line
            for pos in label_positions:
                if pos in overall_average.index:
                    value = overall_average[pos]
                    ax.text(pos, value, f'{value:.3f}',
                           color='lime', fontweight='bold', fontsize=10,
                           ha='center', va='bottom', zorder=11,
                           path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])

            ax.set_xlabel('Prediction Rank Position (1 = highest scores)')
            ax.set_ylabel(f'{metric_name.replace("_", " ").title()}')
            ax.set_title(f'{model_name} - {metric_name.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
            ax.invert_xaxis()

            # Only show legend if no macro comparison (to avoid clutter)
            if not macro_comparison:
                ax.legend()

    # Update title to include macro comparison info
    title = 'Wallet Model Return Metrics Comparison by Epoch (Wins Return | Mean Return)'
    if macro_comparison:
        title += f' (Color-coded by {macro_comparison})'

    plt.suptitle(title, fontsize=16, y=0.995)
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
