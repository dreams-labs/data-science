"""
functions used to build coin-level features from training data
"""
# pylint: disable=C0301 # line over 100 chars
# pylint: disable=C0303 # trailing whitespace
# pylint: disable=C0103 # X_train violates camelcase

import os
import json
import uuid
import joblib
from datetime import datetime
import pandas as pd
import numpy as np
import dreams_core.core as dc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, log_loss
import matplotlib.pyplot as plt
import seaborn as sns

# set up logger at the module level
logger = dc.setup_logger()


def split_model_input(model_input_df, target_column, test_size=0.2, random_state=42):
    """
    Splits the input DataFrame into training and test sets, separating features and target.
    
    Args:
    - model_input_df (pd.DataFrame): The full input DataFrame containing features and target.
    - target_column (str): The name of the target variable.
    - test_size (float): The proportion of data to include in the test set.
    - random_state (int): Random state for reproducibility.
    
    Returns:
    - X_train (pd.DataFrame): Training features.
    - X_test (pd.DataFrame): Test features.
    - y_train (pd.Series): Training target.
    - y_test (pd.Series): Test target.

    Raises:
    - ValueError: If the 'coin_id' column is not present in the DataFrame.
    - ValueError: If features contain missing values.
    - ValueError: If the target column contains missing values.
    - ValueError: If the dataset is too small to perform a meaningful split.
    - ValueError: If the target is heavily imbalanced (over 95% in one class).
    - ValueError: If any features are non-numeric and require encoding.
    - ValueError: If y_train or y_test contains only one unique class, which is unsuitable for model training.
    """
    # Check for 'coin_id' column
    if 'coin_id' not in model_input_df.columns:
        raise ValueError("'coin_id' column is required in the DataFrame.")
    
    # Separate the features and the target
    X = model_input_df.drop(columns=[target_column, 'coin_id'])  # Drop target and coin_id from features
    y = model_input_df[target_column]  # The target column

    # Check for missing values in features or target
    if X.isnull().values.any():
        raise ValueError("Features contain missing values. Please handle missing data before splitting.")
    if y.isnull().values.any():
        raise ValueError("Target column contains missing values. Please handle missing data before splitting.")

    # Check if dataset is too small
    if len(X) < 10:
        raise ValueError("Dataset is too small to perform a meaningful split. Need at least 10 data points.")
    
    # Check for imbalanced target (for classification problems)
    if y.value_counts(normalize=True).max() > 0.95:
        raise ValueError("Target is heavily imbalanced. Consider rebalancing or using specialized techniques.")

    # Check for non-numeric features
    if not all(np.issubdtype(dtype, np.number) for dtype in X.dtypes):
        raise ValueError("Features contain non-numeric data. Consider encoding categorical features.")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Check if y_train or y_test contains only one unique value
    if len(np.unique(y_train)) <= 1 or len(np.unique(y_test)) <= 1:
        raise ValueError("y_train or y_test contains only one class, which is not suitable for model training.")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, modeling_folder, model_params=None):
    """
    Trains a model on the training data and saves the model, logs, and feature importance.
    Uses a UUID to uniquely identify the model files.
    
    Args:
    - X_train (pd.DataFrame): The training features.
    - y_train (pd.Series): The training target.
    - modeling_folder (str): The base folder for saving models, logs, and feature importance.
    - model_params (dict): Parameters to pass to the model (optional).

    Returns:
    - model (sklearn model): The trained model.
    """
    # Generate a UUID for this model instance
    model_id = str(uuid.uuid4())

    # Initialize model with default params if none provided
    if model_params is None:
        model_params = {"n_estimators": 100, "random_state": 42}

    # Initialize the model
    model = RandomForestClassifier(**model_params)

    # Train the model
    model.fit(X_train, y_train)

    # Prepare subfolder paths
    model_output_path = os.path.join(modeling_folder, "models")
    logs_path = os.path.join(modeling_folder, "logs")
    feature_importance_path = os.path.join(modeling_folder, "outputs", "feature_importance")

    # Ensure the required subfolders exist
    os.makedirs(model_output_path, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(feature_importance_path, exist_ok=True)

    # Step 1: Save the model with UUID
    model_filename = os.path.join(model_output_path, f"model_{model_id}.pkl")
    joblib.dump(model, model_filename)

    # Step 2: Log the model parameters and UUID in JSON format
    log_data = {
        "Model ID": model_id,
        "Model parameters": model_params,
    }
    log_filename = os.path.join(logs_path, f"log_{model_id}.json")
    with open(log_filename, 'w') as log_file:
        json.dump(log_data, log_file, indent=4)

    # Step 3: Save feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        feature_importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        })
        feature_importances_filename = os.path.join(feature_importance_path, f"feature_importance_{model_id}.csv")
        feature_importances.to_csv(feature_importances_filename, index=False)

    return model, model_id



def evaluate_model(model, X_test, y_test, model_id, modeling_folder):
    """
    Evaluates a trained model on the test set and outputs key metrics.
    
    Args:
    - model (sklearn model): The trained model.
    - X_test (pd.DataFrame): The test features.
    - y_test (pd.Series): The true labels for the test set.
    - model_id (str): The unique ID of the model being evaluated.
    - modeling_folder (str): The base folder for saving outputs.

    Returns:
    - metrics_dict (dict): A dictionary of calculated evaluation metrics.
    """
    # Construct the performance metrics folder path
    evaluation_folder = os.path.join(modeling_folder, "outputs", "performance_metrics")
    
    # Ensure the evaluation folder exists
    if not os.path.exists(evaluation_folder):
        raise FileNotFoundError(f"The evaluation folder '{evaluation_folder}' does not exist.")

    # Predict the probabilities and the labels
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics_dict = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_prob),
        "log_loss": log_loss(y_test, y_pred_prob)
    }

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Save confusion matrix as a heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix for Model {model_id}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    conf_matrix_filename = os.path.join(evaluation_folder, f"confusion_matrix_{model_id}.png")
    plt.savefig(conf_matrix_filename)
    plt.close()

    # Save metrics to a CSV
    metrics_df = pd.DataFrame([metrics_dict])
    metrics_filename = os.path.join(evaluation_folder, f"metrics_{model_id}.csv")
    metrics_df.to_csv(metrics_filename, index=False)

    return metrics_dict



def log_experiment_results(modeling_folder, model_id):
    """
    Logs the results of a modeling experiment by pulling data from saved files 
    and storing the combined results in the experiment tracking folder.
    
    Args:
    - modeling_folder (str): The base folder where models, logs, and outputs are saved.
    - model_id (str): The unique ID of the model being evaluated.
    
    Returns:
    - experiment_log (dict): A dictionary with the logged experiment details.
    """
    # Define folder paths
    logs_path = os.path.join(modeling_folder, "logs")
    performance_metrics_path = os.path.join(modeling_folder, "outputs", "performance_metrics")
    feature_importance_path = os.path.join(modeling_folder, "outputs", "feature_importance")
    experiment_tracking_path = os.path.join(modeling_folder, "experiment_tracking")
    
    # Step 1: Ensure the experiment_tracking folder exists
    if not os.path.exists(experiment_tracking_path):
        raise FileNotFoundError(f"The folder '{experiment_tracking_path}' does not exist.")

    # Initialize an empty dictionary to collect all the results
    experiment_log = {}

    # Step 2: Read the model training log in JSON format
    log_filename = os.path.join(logs_path, f"log_{model_id}.json")
    if os.path.exists(log_filename):
        with open(log_filename, 'r') as log_file:
            log_data = json.load(log_file)
            experiment_log.update(log_data)
    else:
        raise FileNotFoundError(f"Training log not found for model {model_id}.")

    # Step 3: Read the performance metrics
    metrics_filename = os.path.join(performance_metrics_path, f"metrics_{model_id}.csv")
    if os.path.exists(metrics_filename):
        metrics_df = pd.read_csv(metrics_filename)
        experiment_log.update(metrics_df.iloc[0].to_dict())
    else:
        raise FileNotFoundError(f"Performance metrics not found for model {model_id}.")

    # Step 4: Read the feature importance if available
    feature_importance_filename = os.path.join(feature_importance_path, f"feature_importance_{model_id}.csv")
    if os.path.exists(feature_importance_filename):
        experiment_log["feature_importance"] = feature_importance_filename
    else:
        experiment_log["feature_importance"] = "N/A"  # Optional, in case feature importance isn't available

    # Step 5: Save the experiment log as a JSON file
    experiment_log_filename = os.path.join(experiment_tracking_path, f"experiment_log_{model_id}.json")
    with open(experiment_log_filename, 'w') as experiment_log_file:
        json.dump(experiment_log, experiment_log_file, indent=4)

    return experiment_log
