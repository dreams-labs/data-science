"""
functions used to build coin-level features from training data
"""
# pylint: disable=C0103 # X_train violates camelcase
import os
import json
import uuid
import joblib
import pandas as pd
import numpy as np
import dreams_core.core as dc
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, log_loss
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, max_error
from sklearn.model_selection import KFold, cross_val_score


# project files
import utils as u  # pylint: disable=E0401 # can't find utils import

# set up logger at the module level
logger = dc.setup_logger()


@u.timing_decorator
def train_model(X_train, y_train, modeling_config):
    """
    Trains a model (classifier or regressor) on the training data and saves the model, logs, and feature importance.
    Uses a UUID to uniquely identify the model files.

    Args:
    - X_train (pd.DataFrame): The training features.
    - y_train (pd.Series): The training target.
    - modeling_config (dict): modeling_config.yaml

    Returns:
    - model (sklearn model): The trained model.
    - model_id (str): The UUID of the trained model.
    """
    # Generate a UUID for this model instance
    model_id = str(uuid.uuid4())
    modeling_folder = modeling_config['modeling']['modeling_folder']

    # Initialize model with default params if none provided
    model_params = modeling_config["modeling"].get("model_params", None)
    if model_params is None:
        model_params = {"n_estimators": 100}

    # Initialize the model
    model_type = modeling_config["modeling"]["model_type"]
    if model_type == "RandomForestClassifier":
        model = RandomForestClassifier(**model_params)
    elif model_type == "RandomForestRegressor":
        model = RandomForestRegressor(**model_params)
    elif model_type == "GradientBoostingRegressor":
        model = GradientBoostingRegressor(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # K-Fold cross-validation
    n_splits = modeling_config["modeling"].get("n_splits", 5)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf)


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
        "Model type": model_type,
        "Model parameters": model_params,
    }
    log_filename = os.path.join(logs_path, f"log_{model_id}.json")
    with open(log_filename, "w", encoding="utf-8") as log_file:
        json.dump(log_data, log_file, indent=4)

    # Step 3: Save feature importance (if available)
    if hasattr(model, "feature_importances_"):
        feature_importances = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_
        })
        feature_importances_filename = os.path.join(
            feature_importance_path, f"feature_importance_{model_id}.csv")
        feature_importances.to_csv(feature_importances_filename, index=False)

    return model, model_id, cv_scores


def evaluate_model(model, X_test, y_test, model_id, returns_test, modeling_config):
    """
    Evaluates a trained model (classification or regression) on the test set and outputs key metrics and stores predictions.

    Args:
    - model (sklearn model): The trained model.
    - X_test (pd.DataFrame): The test features with 'coin_id' as the index.
    - y_test (pd.Series): The true labels/values with 'coin_id' as the index.
    - model_id (str): The unique ID of the model being evaluated
    - returns_test (pd.DataFrame): The actual returns of each coin in the test set
    - modeling_config (str): modeling_config.yaml

    Returns:
    - metrics_dict (dict): A dictionary of calculated evaluation metrics.
    """
    modeling_folder = modeling_config['modeling']['modeling_folder']

    # Construct the performance metrics folder path
    evaluation_folder = os.path.join(modeling_folder, "outputs", "performance_metrics")
    predictions_folder = os.path.join(modeling_folder, "outputs", "predictions")

    # Ensure the evaluation and predictions folders exist
    if not os.path.exists(evaluation_folder):
        raise FileNotFoundError(f"The evaluation folder '{evaluation_folder}' does not exist.")
    if not os.path.exists(predictions_folder):
        raise FileNotFoundError(f"The predictions folder '{predictions_folder}' does not exist.")

    # Check if the model is a classifier or regressor
    is_classifier = hasattr(model, "predict_proba")

    # Predict the values
    if is_classifier:
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
        predictions_df = pd.DataFrame({
            "y_pred_prob": y_pred_prob,
            "y_pred": y_pred
        }, index=X_test.index)
    else:
        y_pred = model.predict(X_test)
        y_pred_prob = None
        predictions_df = pd.DataFrame({
            "y_pred": y_pred
        }, index=X_test.index)

    # Save predictions to CSV with 'coin_id' as the index
    predictions_filename = os.path.join(predictions_folder, f"predictions_{model_id}.csv")
    predictions_df.to_csv(predictions_filename, index=True)

    # Calculate requested metrics
    metrics_request = modeling_config['evaluation']['metrics']
    metrics_dict = {}

    if is_classifier:
        if "accuracy" in metrics_request:
            metrics_dict["accuracy"] = accuracy_score(y_test, y_pred)
        if "precision" in metrics_request:
            metrics_dict["precision"] = precision_score(y_test, y_pred)
        if "recall" in metrics_request:
            metrics_dict["recall"] = recall_score(y_test, y_pred)
        if "f1_score" in metrics_request:
            metrics_dict["f1_score"] = f1_score(y_test, y_pred)
        if "roc_auc" in metrics_request:
            metrics_dict["roc_auc"] = roc_auc_score(y_test, y_pred_prob)
        if "log_loss" in metrics_request:
            metrics_dict["log_loss"] = log_loss(y_test, y_pred_prob)
        if "confusion_matrix" in metrics_request:
            metrics_dict["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        if "profitability_auc" in metrics_request:
            metrics_dict["profitability_auc"] = calculate_profitability_auc(
                                                        y_pred_prob,
                                                        returns_test['returns'],
                                                        metrics_request["profitability_auc"]["top_percentage_filter"],
                                                        modeling_config["evaluation"]["winsorization_cutoff"]
                                                        )
        if "downside_profitability_auc" in metrics_request:
            metrics_dict["downside_profitability_auc"] = calculate_downside_profitability_auc(
                                                        y_pred_prob,
                                                        returns_test['returns'],
                                                        metrics_request["profitability_auc"]["top_percentage_filter"],
                                                        modeling_config["evaluation"]["winsorization_cutoff"]
                                                        )
    else:  # Regression metrics
        if "mse" in metrics_request:
            metrics_dict["mse"] = mean_squared_error(y_test, y_pred)
        if "rmse" in metrics_request:
            metrics_dict["rmse"] = np.sqrt(mean_squared_error(y_test, y_pred))
        if "mae" in metrics_request:
            metrics_dict["mae"] = mean_absolute_error(y_test, y_pred)
        if "r2" in metrics_request:
            metrics_dict["r2"] = r2_score(y_test, y_pred)
        if "explained_variance" in metrics_request:
            metrics_dict["explained_variance"] = explained_variance_score(y_test, y_pred)
        if "max_error" in metrics_request:
            metrics_dict["max_error"] = max_error(y_test, y_pred)
        if "profitability_auc" in metrics_request:
            metrics_dict["profitability_auc"] = calculate_profitability_auc(
                                                        y_pred,
                                                        returns_test['returns'],
                                                        metrics_request["profitability_auc"]["top_percentage_filter"],
                                                        modeling_config["evaluation"]["winsorization_cutoff"]
                                                        )
        if "downside_profitability_auc" in metrics_request:
            metrics_dict["downside_profitability_auc"] = calculate_downside_profitability_auc(
                                                        y_pred,
                                                        returns_test['returns'],
                                                        metrics_request["profitability_auc"]["top_percentage_filter"],
                                                        modeling_config["evaluation"]["winsorization_cutoff"]
                                                        )

    # Save metrics to a CSV
    metrics_df = pd.DataFrame([metrics_dict])
    metrics_filename = os.path.join(evaluation_folder, f"metrics_{model_id}.csv")
    metrics_df.to_csv(metrics_filename, index=False)

    return metrics_dict, y_pred, y_pred_prob



def calculate_running_profitability_score(predictions, returns, winsorization_cutoff=None):
    """
    Calculates the running profitability score for the entire series.

    Args:
    - predictions (numpy.array or pandas.Series): The model's predictions (probabilities or values).
    - returns (numpy.array or pandas.Series): The coin's actual gains or losses during the period.
    - winsorization_cutoff (float): Returns in the top and bottom n% of the array will be capped at
        the value as of the cutoff point

    Returns:
    - running_profitability_scores: The model's cumulative profitability score through the array length,
        e.g. the first score is through 1 coin, the second score is through 2 coins, etc

    Raises:
    - ValueError: If predictions and returns have different lengths.
    """
    if len(predictions) != len(returns):
        raise ValueError("Predictions and returns must have the same length")

    if winsorization_cutoff:
        returns = winsorize(returns, winsorization_cutoff)

    # Create a DataFrame with predictions and returns
    df = pd.DataFrame({'predictions': predictions, 'returns': returns})

    # Calculate the cumulative profits of the model predictions
    df_sorted = df.sort_values('predictions', ascending=False)
    cumulative_model_returns = np.cumsum(df_sorted['returns'])

    # Calculate best possible returns for each number of picks
    best_possible_returns = np.sort(returns)[::-1]  # Sort returns in descending order
    cumulative_best_returns = np.cumsum(best_possible_returns)

    # Calculate running profitability scores
    running_profitability_scores = np.divide(
        cumulative_model_returns,
        cumulative_best_returns,
        out=np.zeros_like(cumulative_model_returns),
        where=cumulative_best_returns != 0  # if cumulative profit is 0, return 0 instead of a div0 error
    )

    return running_profitability_scores



def calculate_profitability_auc(predictions,
                                returns,
                                top_percentage_filter=1.0,
                                winsorization_cutoff=None):
    """
    Calculates the Profitability AUC (Area Under the Curve) metric for the top percentage of predictions.

    Args:
    - predictions (numpy.array or pandas.Series): The model's predictions (probabilities or values).
    - returns (numpy.array or pandas.Series): The actual performance values.
    - top_percentage_filter (float): The top percentage of predictions to consider, between 0 and 1.

    Returns:
    - profitability_auc (float): The Profitability AUC score for the filtered data, ranging from 0 to 1.
    """
    if not 0 < top_percentage_filter <= 1:
        raise ValueError("top_percentage_filter must be between 0 and 1")

    # Calculate the full range of profitability scores
    running_scores = calculate_running_profitability_score(predictions, returns, winsorization_cutoff)

    # Calculate how many scores to look at based on the percentage filter
    n_top = int(len(predictions) * top_percentage_filter)
    if n_top < 2:
        raise ValueError("Filtered dataset is too small for meaningful calculation")

    # Limit the scores for AUC calculation to the percentile input
    filtered_running_scores = running_scores[:n_top]

    # Create x-axis values (fraction of filtered predictions)
    x = np.linspace(0, 1, len(filtered_running_scores))

    # Calculate the area under the curve using NumPy's trapezoidal rule
    auc = np.trapezoid(filtered_running_scores, x)

    return auc



def calculate_downside_profitability_auc(predictions,
                                         returns_test,
                                         top_percentage_filter=1.0,
                                         winsorization_cutoff=None):
    """
    Calculates the Profitability AUC (Area Under the Curve) metric for the bottom percentage of
    predictions by inverting returns and predictions.
    """
    # make negative returns the highest values
    returns = returns_test * -1

    # find the inverse of model predictions
    predictions = 1 - predictions

    # calculate the normal AUC on inverted numbers
    downside_auc = calculate_profitability_auc(predictions,
                                returns,
                                top_percentage_filter,
                                winsorization_cutoff)

    return downside_auc


def winsorize(data, cutoff):
    """
    Applies winsorization to the input data.

    Args:
    - data (numpy.array or pandas.Series): The data to be winsorized.
    - cutoff (float): The percentile at which to winsorize (e.g., 0.05 for 5th and 95th percentiles).

    Returns:
    - numpy.array: The winsorized data.

    Raises:
    - ValueError: If cutoff is not between 0 and 0.5.
    """
    if not 0 < cutoff <= 0.5:
        raise ValueError("Cutoff must be between 0 and 0.5")

    lower_bound = np.percentile(data, cutoff * 100)
    upper_bound = np.percentile(data, (1 - cutoff) * 100)

    return np.clip(data, lower_bound, upper_bound)



def log_trial_results(modeling_config, model_id, experiment_id=None, trial_overrides=None):
    """
    Logs the results of a modeling trial by pulling data from saved files
    and storing the combined results in the experiment tracking folder.

    Args:
    - modeling_folder (str): The base folder where models, logs, and outputs are saved.
    - model_id (str): The unique ID of the model being evaluated.
    - experiment_id (str, optional): The unique ID of the experiment.
    - trial_overrides (dict, optional): The override parameters used in the specific trial.

    Returns:
    - trial_log (dict): A dictionary with the logged trial details.
    """
    # Define folder paths
    modeling_folder = modeling_config['modeling']['modeling_folder']
    logs_path = os.path.join(modeling_folder, "logs")
    performance_metrics_path = os.path.join(modeling_folder, "outputs", "performance_metrics")
    feature_importance_path = os.path.join(modeling_folder, "outputs", "feature_importance")
    predictions_path = os.path.join(modeling_folder, "outputs", "predictions")
    experiment_tracking_path = os.path.join(modeling_folder, "outputs", "experiment_tracking")

    # Step 1: Ensure the experiment_tracking folder exists
    if not os.path.exists(experiment_tracking_path):
        raise FileNotFoundError(f"The folder '{experiment_tracking_path}' does not exist.")

    # Initialize an empty dictionary to collect all the results
    trial_log = {
        "experiment_id": experiment_id,
        "model_id": model_id,
        "trial_overrides": trial_overrides,
        "metrics": {}  # Initialize the 'metrics' key for performance metrics
    }

    # Step 2: Read the model training log in JSON format
    log_filename = os.path.join(logs_path, f"log_{model_id}.json")
    if os.path.exists(log_filename):
        with open(log_filename, 'r', encoding='utf-8') as log_file:
            log_data = json.load(log_file)
            trial_log.update(log_data)
    else:
        raise FileNotFoundError(f"Training log not found for model {model_id}.")

    # Step 3: Read the performance metrics and store them under 'metrics' key
    metrics_filename = os.path.join(performance_metrics_path, f"metrics_{model_id}.csv")
    if os.path.exists(metrics_filename):
        metrics_df = pd.read_csv(metrics_filename)
        trial_log['metrics'] = metrics_df.iloc[0].to_dict()
    else:
        raise FileNotFoundError(f"Performance metrics not found for model {model_id}.")

    # Step 4: Read the feature importance if available and store as a dict
    feature_importance_filename = os.path.join(feature_importance_path,
                                               f"feature_importance_{model_id}.csv")
    if os.path.exists(feature_importance_filename):
        feature_importance_df = pd.read_csv(feature_importance_filename)
        feature_importance_dict = dict(zip(feature_importance_df['feature'],
                                           feature_importance_df['importance']))
        trial_log["feature_importance"] = feature_importance_dict
    else:
        trial_log["feature_importance"] = "N/A"

    # Step 5: Read the predictions from CSV and store as a dict
    predictions_filename = os.path.join(predictions_path, f"predictions_{model_id}.csv")
    if os.path.exists(predictions_filename):
        predictions_df = pd.read_csv(predictions_filename)
        predictions_dict = predictions_df.to_dict(orient='list')
        trial_log["predictions"] = predictions_dict
    else:
        trial_log["predictions"] = "N/A"

    # Step 6: Add all configs to the trial log
    config_folder = modeling_config['modeling']['config_folder']
    config, metrics_config, modeling_config, _ = u.load_all_configs(config_folder)  # Reload all configs

    trial_log["configs"]["config"] = config
    trial_log["configs"]["metrics_config"] = metrics_config
    trial_log["configs"]["modeling_config"] = modeling_config

    # Step 7: Save the trial log as a JSON file
    trial_log_filename = os.path.join(experiment_tracking_path, f"trial_log_{model_id}.json")
    with open(trial_log_filename, 'w', encoding='utf-8') as trial_log_file:
        json.dump(trial_log, trial_log_file, indent=4)

    return trial_log_filename
