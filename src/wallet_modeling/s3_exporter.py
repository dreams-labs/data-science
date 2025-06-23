"""
s3_exporter.py
======================

Standalone export utilities for storing wallet model training data for
S3 upload. Data is generally stored in the sagemaker repo at directory
'sagemaker/s3_wallet_loading_queue'.
"""
import logging
import json
from typing import Dict, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np

# Local modules
import utils as u

logger = logging.getLogger(__name__)

# pylint:disable=invalid-name  # X_test isn't camelcase


def export_s3_training_data(
    export_config: dict,
    model_id: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_eval: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    y_eval: pd.Series,
    meta_pipeline: Any,
    asymmetric_loss_enabled: bool,
    X_validation: Optional[pd.DataFrame] = None,
    validation_target_vars_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Export the exact training and target data that will be fed to the model as parquet files.
    This captures data after all filtering, transformations, and cohort selection.

    Files are exported with date-based naming using the modeling_period_start date:
    Final filepath: {parent_folder}/{batch_folder}/{filename}_{YYMMDD}.parquet
    Example: ../sagemaker/s3_wallet_loading_queue/dda858_batch/x_train_240315.parquet

    Exported files:
    - x_train_{YYMMDD}.parquet, y_train_{YYMMDD}.parquet
    - x_eval_{YYMMDD}.parquet, y_eval_{YYMMDD}.parquet
    - x_test_{YYMMDD}.parquet, y_test_{YYMMDD}.parquet
    - x_validation_{YYMMDD}.parquet, y_validation_{YYMMDD}.parquet
    - export_metadata_{YYMMDD}.json

    Params:
    - export_config (dict): Configuration containing parent_folder and batch_folder
    - model_id (str): Unique identifier for the model
    - X_train, X_test, X_eval (DataFrame): Feature datasets
    - y_train, y_test, y_eval (Series): Target datasets
    - meta_pipeline: Pipeline object with y_pipeline for target transformation
    - asymmetric_loss_enabled (bool): Whether to apply asymmetric loss binary conversion
    - X_validation (DataFrame, optional): Validation feature data
    - validation_target_vars_df (DataFrame, optional): Validation target data

    Returns:
    - result (dict): Export completion status and metadata

    Raises:
    - ConfigError: If parent_folder doesn't exist or folder names contain spaces
    """
    # 1. Validate data
    # Validate configuration
    parent_folder = export_config.get('parent_folder', '')
    batch_folder = export_config.get('batch_folder', '')
    if export_config.get('dev_mode', False):
        batch_folder = f"{batch_folder}_dev"

    # Check for spaces in folder names
    if ' ' in parent_folder:
        raise u.ConfigError(f"parent_folder cannot contain spaces: '{parent_folder}'")
    if ' ' in batch_folder:
        raise u.ConfigError(f"batch_folder cannot contain spaces: '{batch_folder}'")

    # Validate parent folder exists
    parent_path = Path(parent_folder)
    if not parent_path.exists():
        raise u.ConfigError(f"parent_folder does not exist: '{parent_folder}'")
    if not parent_path.is_dir():
        raise u.ConfigError(f"parent_folder is not a directory: '{parent_folder}'")

    # Require both X and y validation data
    if X_validation is not None:
        if X_validation.empty:
            raise ValueError("X_validation is empty - cannot export validation data without features")
        if validation_target_vars_df is None or validation_target_vars_df.empty:
            raise ValueError("validation_target_vars_df is empty - cannot export validation data without targets")

    # 2. Export files
    # Create export directory
    export_folder = parent_path / batch_folder
    export_folder.mkdir(parents=True, exist_ok=True)

    # Generate date suffix by extracting the latest modeling_epoch_start
    modeling_date = X_train.index.get_level_values("epoch_start_date").max()
    date_suffix = pd.to_datetime(modeling_date, '%Y-%m-%d').strftime('%y%m%d')

    # Transform targets using the same y_pipeline that would be used in training
    y_train_transformed = meta_pipeline.y_pipeline.transform(y_train)
    y_eval_transformed = meta_pipeline.y_pipeline.transform(y_eval)
    y_test_transformed = meta_pipeline.y_pipeline.transform(y_test)

    # Initialize validation transform as None
    y_val_transformed = None
    if validation_target_vars_df is not None:
        y_val_transformed = meta_pipeline.y_pipeline.transform(validation_target_vars_df)

    # Convert multi-class labels to binary for asymmetric loss
    if asymmetric_loss_enabled:
        y_train_transformed = pd.Series((y_train_transformed == 2).astype(int), index=X_train.index)
        y_test_transformed = pd.Series((y_test_transformed == 2).astype(int), index=X_test.index)
        y_eval_transformed = pd.Series((y_eval_transformed == 2).astype(int), index=X_eval.index)
        if y_val_transformed is not None:
            y_val_transformed = pd.Series((y_val_transformed == 2).astype(int), index=X_validation.index)

    # Prepare datasets for export with standardized lowercase names
    datasets = {
        'x_train': X_train,
        'x_test': X_test,
        'x_eval': X_eval,
        'y_train': y_train_transformed,
        'y_eval': y_eval_transformed,
        'y_test': y_test_transformed,
    }

    # Add validation data if present
    if X_validation is not None and y_val_transformed is not None:
        datasets['x_val'] = X_validation
        datasets['y_val'] = y_val_transformed

    # Apply dev_mode sampling if enabled
    dev_mode = export_config.get('dev_mode', False)
    exported_files = {}

    for name, data in datasets.items():
        # Convert numpy ndarrays to Series for unified handling
        if isinstance(data, np.ndarray):
            raise ValueError("Data must be a DataFrame or Series with IDs as " \
                             "the index.")
        # --- Validate dataset presence and non-emptiness ---
        if data is None:
            raise ValueError(
                f"{name} dataset is None – cannot export S3 training data "
                f"(model_id={model_id}, date_suffix={date_suffix})"
            )
        if isinstance(data, (pd.DataFrame, pd.Series)) and data.empty:
            raise ValueError(
                f"{name} dataset is empty – cannot export S3 training data "
                f"(shape={data.shape}, model_id={model_id}, date_suffix={date_suffix})"
            )

        # --- Optional down-sampling for dev mode ---
        if dev_mode and len(data) > 1000:
            original_len = len(data)
            data = data.head(1000)
            logger.info(f"[DevMode] Sampled {name} from {original_len} to {len(data)} rows")

        file_path = export_folder / f"{name}_{date_suffix}.parquet"

        # Convert Series to DataFrame for parquet export
        data_to_export = data.to_frame() if isinstance(data, pd.Series) else data

        data_to_export.to_parquet(file_path, index=True)

        exported_files[name] = {
            'path': str(file_path),
            'shape': data.shape,
        }
        logger.info(f"Exported {name} with shape {data.shape} to {file_path}")

    # Export metadata about the modeling configuration
    metadata = {
        'model_id': model_id,
        'export_timestamp': pd.Timestamp.now().isoformat(),
        'exported_files': exported_files,
        'pipeline_steps': [step[0] for step in meta_pipeline.model_pipeline.steps],
        'y_pipeline_steps': [step[0] for step in meta_pipeline.y_pipeline.steps]
    }

    # Export metadata as JSON (handling numpy, datetime, and infinite values)
    metadata_path = export_folder / f"export_metadata_{date_suffix}.json"
    # write JSON with utf-8 encoding, using numpy_type_converter for special types
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
            default=u.numpy_type_converter
        )

    logger.milestone(f"Successfully exported S3 training data to {export_folder}")

    return {
        'export_folder': str(export_folder),
        'exported_files': exported_files,
        'metadata_path': str(metadata_path)
    }
