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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, log_loss
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, max_error


# project files
from utils import timing_decorator  # pylint: disable=E0401 # can't find utils import

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
    - X_train (pd.DataFrame): Training features with 'coin_id' as the index.
    - X_test (pd.DataFrame): Test features with 'coin_id' as the index.
    - y_train (pd.Series): Training target with 'coin_id' as the index.
    - y_test (pd.Series): Test target with 'coin_id' as the index.

    Raises:
    - ValueError: If the 'coin_id' column is not present in the DataFrame.
    - ValueError: If features contain missing values.
    - ValueError: If the target column contains missing values.
    - ValueError: If the dataset is too small to perform a meaningful split.
    - ValueError: If the target is heavily imbalanced (over 95% in one class).
    - ValueError: If any features are non-numeric and require encoding.
    - ValueError: If y_train or y_test contains only one unique class, which is unsuitable for
        model training.
    """
    # Check for 'coin_id' column
    if 'coin_id' not in model_input_df.columns:
        raise ValueError("'coin_id' column is required in the DataFrame.")

    # Separate the features and the target
    # Set 'coin_id' as index for X
    X = model_input_df.drop(columns=[target_column]).set_index('coin_id')
    # Extract target as Series, it will retain the index from model_input_df
    y = model_input_df[target_column]

    # Check for missing values in features or target
    if X.isnull().values.any():
        raise ValueError("Features contain missing values. Please handle missing data \
                         before splitting.")
    if y.isnull().values.any():
        raise ValueError("Target column contains missing values. Please handle missing \
                         data before splitting.")

    # Check if dataset is too small
    if len(X) < 10:
        raise ValueError("Dataset is too small to perform a meaningful split. Need at \
                         least 10 data points.")

    # Check for imbalanced target (for classification problems)
    if y.value_counts(normalize=True).max() > 0.95:
        raise ValueError("Target is heavily imbalanced. Consider rebalancing or using \
                         specialized techniques.")

    # Check for non-numeric features
    if not all(np.issubdtype(dtype, np.number) for dtype in X.dtypes):
        raise ValueError("Features contain non-numeric data. Consider encoding categorical \
                         features.")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Log the size and number of positives for y_train and y_test
    logger.info("y_train: %d/%d positives, y_test: %d/%d positives",
                y_train.sum(), len(y_train), y_test.sum(), len(y_test))

    # Check if y_train or y_test contains only one unique value
    if len(np.unique(y_train)) <= 1 or len(np.unique(y_test)) <= 1:
        raise ValueError("y_train or y_test contains only one class, which is not suitable \
                         for model training.")

    return X_train, X_test, y_train, y_test



@timing_decorator
def train_model(X_train, y_train, modeling_folder, modeling_config):
    """
    Trains a model (classifier or regressor) on the training data and saves the model, logs, and feature importance.
    Uses a UUID to uniquely identify the model files.

    Args:
    - X_train (pd.DataFrame): The training features.
    - y_train (pd.Series): The training target.
    - modeling_folder (str): The base folder for saving models, logs, and feature importance.
    - modeling_config (dict): modeling_config.yaml

    Returns:
    - model (sklearn model): The trained model.
    - model_id (str): The UUID of the trained model.
    """
    # Generate a UUID for this model instance
    model_id = str(uuid.uuid4())

    # Initialize model with default params if none provided
    model_params = modeling_config["modeling"].get("model_params", None)
    if model_params is None:
        model_params = {"n_estimators": 100, "random_state": 42}

    # Initialize the model
    model_type = modeling_config["modeling"]["model_type"]
    if model_type == "RandomForestClassifier":
        model = RandomForestClassifier(**model_params)
    elif model_type == "RandomForestRegressor":
        model = RandomForestRegressor(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

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

    return model, model_id


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
        y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
        y_pred = model.predict(X_test)
        predictions_df = pd.DataFrame({
            "y_pred_prob": y_pred_prob,
            "y_pred": y_pred
        }, index=X_test.index)
    else:
        y_pred = model.predict(X_test)
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
                                                        returns_test,
                                                        metrics_request["profitability_auc"]["top_percentage_filter"],
                                                        modeling_config["evaluation"]["winsorization_cutoff"]
                                                        )
        if "downside_profitability_auc" in metrics_request:
            metrics_dict["downside_profitability_auc"] = calculate_downside_profitability_auc(
                                                        y_pred_prob,
                                                        returns_test,
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
                                                        returns_test,
                                                        metrics_request["profitability_auc"]["top_percentage_filter"],
                                                        modeling_config["evaluation"]["winsorization_cutoff"]
                                                        )
        if "downside_profitability_auc" in metrics_request:
            metrics_dict["downside_profitability_auc"] = calculate_downside_profitability_auc(
                                                        y_pred,
                                                        returns_test,
                                                        metrics_request["profitability_auc"]["top_percentage_filter"],
                                                        modeling_config["evaluation"]["winsorization_cutoff"]
                                                        )

    # Save metrics to a CSV
    metrics_df = pd.DataFrame([metrics_dict])
    metrics_filename = os.path.join(evaluation_folder, f"metrics_{model_id}.csv")
    metrics_df.to_csv(metrics_filename, index=False)

    return metrics_dict



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



def log_trial_results(modeling_folder, model_id, experiment_id=None, trial_overrides=None):
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

    # Step 6: Save the trial log as a JSON file
    trial_log_filename = os.path.join(experiment_tracking_path, f"trial_log_{model_id}.json")
    with open(trial_log_filename, 'w', encoding='utf-8') as trial_log_file:
        json.dump(trial_log, trial_log_file, indent=4)

    return trial_log_filename
