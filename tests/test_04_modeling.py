"""
tests used to audit the files in the data-science/src folder
"""
# pylint: disable=C0301 # line over 100 chars
# pylint: disable=C0413 # import not at top of doc (due to local import)
# pylint: disable=C0303 # trailing whitespace
# pylint: disable=W1203 # fstrings in logs
# pylint: disable=W0612 # unused variables (due to test reusing functions with 2 outputs)
# pylint: disable=W0621 # redefining from outer scope triggering on pytest fixtures
# pylint: disable=E0401 # can't find import (due to local import)
# pylint: disable=C0103 # X_train violates camelcase
# pylint: disable=W0613  # unused arguments, some mocked fixtures are "unused" but needed to mock the rest of the data

import sys
import os
from unittest import mock
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pytest
from sklearn.ensemble import RandomForestClassifier
from dreams_core import core as dc

# pyright: reportMissingImports=false
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import modeling as m
from utils import load_config

load_dotenv()
logger = dc.setup_logger()




# ===================================================== #
#                                                       #
#                 U N I T   T E S T S                   #
#                                                       #
# ===================================================== #

# ---------------------------------- #
# split_model_input() unit tests
# ---------------------------------- #


def create_dataframe(features, target, coin_id=True):
    """
    Helper function to create a DataFrame with specified features,
    """
    data = pd.DataFrame(features)
    data['target'] = target
    if coin_id:
        data['coin_id'] = np.random.randint(1000, 9999, size=len(data))  # Random coin_id column
    return data

@pytest.mark.unit
def test_missing_coin_id():
    """
    Test if ValueError is raised when the 'coin_id' column is missing.
    """
    # Create a DataFrame without 'coin_id' column
    data = create_dataframe({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}, [0, 1, 0], coin_id=False)

    with pytest.raises(ValueError, match="'coin_id' column is required in the DataFrame"):
        m.split_model_input(data, 'target')

@pytest.mark.unit
def test_missing_values_in_features():
    """
    Test if ValueError is raised when features contain missing values.
    """
    # Create a DataFrame with missing values in features
    data = create_dataframe({'feature1': [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10], 'feature2': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}, [0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    with pytest.raises(ValueError, match="Features contain missing values"):
        m.split_model_input(data, 'target')

@pytest.mark.unit
def test_missing_values_in_target():
    """
    Test if ValueError is raised when the target contains missing values.
    """
    # Create a DataFrame with missing values in the target
    data = create_dataframe({'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'feature2': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}, [0, np.nan, 1, 0, 1, 0, 1, 0, 1, 0])

    with pytest.raises(ValueError, match="Target column contains missing values"):
        m.split_model_input(data, 'target')

@pytest.mark.unit
def test_dataset_too_small():
    """
    Test if ValueError is raised when the dataset has less than 10 rows.
    """
    # Create a DataFrame with less than 10 rows
    data = create_dataframe({'feature1': [1, 2], 'feature2': [4, 5]}, [0, 1])

    with pytest.raises(ValueError, match="Dataset is too small"):
        m.split_model_input(data, 'target')

@pytest.mark.unit
def test_imbalanced_target():
    """
    Test if ValueError is raised when the target is heavily imbalanced.
    """
    # Create a DataFrame with an imbalanced target and 10 rows
    data = create_dataframe({'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'feature2': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    with pytest.raises(ValueError, match="Target is heavily imbalanced"):
        m.split_model_input(data, 'target')

@pytest.mark.unit
def test_non_numeric_features():
    """
    Test if ValueError is raised when features contain non-numeric data.
    """
    # Create a DataFrame with non-numeric features and 10 rows
    data = create_dataframe({'feature1': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'], 'feature2': ['x', 'y', 'z', 'a', 'b', 'c', 'd', 'e', 'f', 'g']}, [0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    with pytest.raises(ValueError, match="Features contain non-numeric data"):
        m.split_model_input(data, 'target')

@pytest.mark.unit
def test_one_class_in_target():
    """
    Test if ValueError is raised when the target has only one class in y_train or y_test.
    """
    # Create a DataFrame where the target has only one class and 10 rows
    data = create_dataframe({'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'feature2': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}, [1, 0, 1, 1, 1, 1, 1, 1, 1, 1])

    with pytest.raises(ValueError, match="y_train or y_test contains only one class"):
        m.split_model_input(data, 'target')


# ---------------------------------- #
# train_model() unit tests
# ---------------------------------- #

# Fixtures for sample data
@pytest.fixture
def sample_data():
    """
    Fixture that returns sample training data (X_train and y_train).
    X_train is a DataFrame with two features.
    y_train is a Series with binary target labels.
    """
    X_train = pd.DataFrame({
        'feature1': range(10),
        'feature2': range(10, 20)
    })
    y_train = pd.Series([0, 1] * 5)
    return X_train, y_train

@pytest.fixture
def modeling_folder():
    """
    Fixture that returns the path to the test_modeling folder.
    This folder is assumed to be in the same directory as the test files.
    """
    return "tests/test_modeling"  # Relative path to the test_modeling folder

@pytest.fixture
def setup_modeling_folders(modeling_folder):
    """
    Fixture that ensures the necessary directory structure is created
    for the train_model function.
    """
    os.makedirs(os.path.join(modeling_folder, "outputs", "feature_importance"), exist_ok=True)
    os.makedirs(os.path.join(modeling_folder, "logs"), exist_ok=True)
    os.makedirs(os.path.join(modeling_folder, "models"), exist_ok=True)

@pytest.mark.unit
@mock.patch('joblib.dump')  # Only mock saving the model, not folder creation
@mock.patch('builtins.open', new_callable=mock.mock_open)  # Mock the open function for file writing
def test_basic_functionality(mock_open, mock_dump, sample_data, modeling_folder):
    """
    Test basic functionality of the train_model function.
    Ensures model is trained, files are saved, and a model ID is generated.
    """
    X_train, y_train = sample_data

    # Call the function to test
    model, model_id = m.train_model(X_train, y_train, modeling_folder)

    # Assert the model is a RandomForestClassifier instance
    assert isinstance(model, RandomForestClassifier)

    # Assert model ID is generated
    assert isinstance(model_id, str)

    # Assert files are saved
    mock_dump.assert_called_once()
    mock_open.assert_called()

@pytest.mark.unit
@mock.patch('joblib.dump')
@mock.patch('builtins.open', new_callable=mock.mock_open)
def test_custom_model_parameters(mock_open, mock_dump, sample_data, setup_modeling_folders, modeling_folder):
    """
    Test train_model with custom model parameters.
    Ensures the model is trained with specified parameters.
    """
    X_train, y_train = sample_data
    custom_params = {"n_estimators": 50, "random_state": 10}

    model, model_id = m.train_model(X_train, y_train, modeling_folder, model_params=custom_params)

    # Assert the model is trained with custom parameters
    assert model.n_estimators == 50
    assert model.random_state == 10

    # Assert model ID is generated
    assert isinstance(model_id, str)

    # Assert files are saved
    mock_dump.assert_called_once()
    mock_open.assert_called()

# Unit Test: Invalid Model Parameters
@pytest.mark.unit
@mock.patch('joblib.dump')
@mock.patch('builtins.open', new_callable=mock.mock_open)
def test_invalid_model_parameters(mock_open, mock_dump, sample_data, setup_modeling_folders, modeling_folder):
    """
    Test train_model with invalid model parameters.
    Ensures the function raises a TypeError when invalid parameters are passed.
    """
    X_train, y_train = sample_data
    invalid_params = {"invalid_param": 100}

    # Expect TypeError due to invalid parameter
    with pytest.raises(TypeError):
        m.train_model(X_train, y_train, modeling_folder, model_params=invalid_params)

# Unit Test: Feature Importance Validation
@pytest.mark.unit
@mock.patch('pandas.DataFrame.to_csv')  # Mock the to_csv function
@mock.patch('pandas.read_csv')  # Mock the read_csv function
@mock.patch('joblib.dump')
@mock.patch('builtins.open', new_callable=mock.mock_open)
def test_feature_importance_validation(mock_open, mock_dump, mock_read_csv, mock_to_csv, sample_data, setup_modeling_folders, modeling_folder):
    """
    Test the accuracy of feature importance saved by train_model.
    Ensures the saved feature importance matches the model's internal feature importance.
    """
    X_train, y_train = sample_data

    # Expected feature importance content
    expected_feature_importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': [0.7, 0.3]  # Simulated importances
    })

    # Mock the to_csv call, assume it writes correctly
    mock_to_csv.return_value = None

    # Mock the read_csv call to return the expected feature importance DataFrame
    mock_read_csv.return_value = expected_feature_importances

    # Call the function to test
    model, model_id = m.train_model(X_train, y_train, modeling_folder)

    # Mock reading the feature importance CSV file
    feature_importances_path = os.path.join(modeling_folder, "outputs", "feature_importance", f"feature_importance_{model_id}.csv")

    # Now check that the read content matches the expected importances
    feature_importances = pd.read_csv(feature_importances_path)

    assert (feature_importances['feature'] == X_train.columns).all()
    assert (feature_importances['importance'] == expected_feature_importances['importance']).all()

# ---------------------------------- #
# log_trial_results() unit tests
# ---------------------------------- #

@pytest.fixture
def model_id():
    """Fixture to provide a model ID for testing."""
    return "model123"

@pytest.fixture
def experiment_id():
    """Fixture to provide an experiment ID for testing."""
    return "exp456"

@pytest.fixture
def mock_modeling_folder(tmp_path):
    """Fixture to create a temporary directory structure that mimics the modeling folder."""
    folder = tmp_path / "modeling_folder"
    folder.mkdir()

    # Create subdirectories
    (folder / "logs").mkdir()
    (folder / "outputs").mkdir()
    (folder / "outputs" / "performance_metrics").mkdir()
    (folder / "outputs" / "feature_importance").mkdir()
    (folder / "outputs" / "predictions").mkdir()
    (folder / "outputs" / "experiment_tracking").mkdir()

    return folder

@pytest.fixture
def mock_log_file(mock_modeling_folder, model_id):
    """Fixture to create a mock log file for the given model_id."""
    log_path = mock_modeling_folder / "logs" / f"log_{model_id}.json"
    log_data = {"training_accuracy": 0.95}
    with open(log_path, 'w', encoding='utf-8') as log_file:
        json.dump(log_data, log_file)
    return log_path

@pytest.fixture
def mock_metrics_file(mock_modeling_folder, model_id):
    """Fixture to create a mock metrics CSV file for the given model_id."""
    metrics_path = mock_modeling_folder / "outputs" / "performance_metrics" / f"metrics_{model_id}.csv"
    metrics_data = pd.DataFrame({"accuracy": [0.95], "precision": [0.9], "recall": [0.85]})
    metrics_data.to_csv(metrics_path, index=False)
    return metrics_path

@pytest.fixture
def mock_feature_importance_file(mock_modeling_folder, model_id):
    """Fixture to create a mock feature importance CSV file for the given model_id."""
    feature_importance_path = mock_modeling_folder / "outputs" / "feature_importance" / f"feature_importance_{model_id}.csv"
    feature_importance_data = pd.DataFrame({"feature": ["f1", "f2"], "importance": [0.8, 0.2]})
    feature_importance_data.to_csv(feature_importance_path, index=False)
    return feature_importance_path

@pytest.fixture
def mock_predictions_file(mock_modeling_folder, model_id):
    """Fixture to create a mock predictions CSV file for the given model_id."""
    predictions_path = mock_modeling_folder / "outputs" / "predictions" / f"predictions_{model_id}.csv"
    predictions_data = pd.DataFrame({"id": [1, 2], "prediction": [0.9, 0.8]})
    predictions_data.to_csv(predictions_path, index=False)
    return predictions_path

@pytest.mark.unit
def test_log_trial_results_normal_case(mock_modeling_folder, mock_log_file, mock_metrics_file, mock_feature_importance_file, mock_predictions_file, model_id, experiment_id):
    """
    Test normal case where all necessary files exist, and ensure that
    the function correctly logs the trial results.
    """
    trial_overrides = {"learning_rate": 0.01}

    # Call the function
    trial_log_filename = m.log_trial_results(mock_modeling_folder, model_id, experiment_id, trial_overrides)

    # Open and read the generated trial log file
    with open(trial_log_filename, 'r', encoding='utf-8') as log_file:
        trial_log = json.load(log_file)

    # Assertions to check correctness of logged data
    assert trial_log["experiment_id"] == experiment_id
    assert trial_log["model_id"] == model_id
    assert trial_log["metrics"]["accuracy"] == 0.95
    assert trial_log["feature_importance"]["f1"] == 0.8
    assert trial_log["predictions"]["id"] == [1, 2]

@pytest.mark.unit
def test_log_trial_results_missing_experiment_tracking_folder(mock_modeling_folder, model_id, experiment_id):
    """
    Test case where the experiment_tracking folder does not exist,
    expecting FileNotFoundError.
    """
    trial_overrides = {"learning_rate": 0.01}

    # Remove the experiment_tracking folder
    experiment_tracking_path = mock_modeling_folder / "outputs" / "experiment_tracking"
    os.rmdir(experiment_tracking_path)

    with pytest.raises(FileNotFoundError):
        m.log_trial_results(mock_modeling_folder, model_id, experiment_id, trial_overrides)


@pytest.mark.unit
def test_log_trial_results_missing_log_file(mock_modeling_folder, mock_metrics_file, mock_feature_importance_file, mock_predictions_file, model_id, experiment_id):
    """
    Test case where the log file is missing, expecting FileNotFoundError.
    """
    trial_overrides = {"learning_rate": 0.01}

    # No log file created, expect a FileNotFoundError
    with pytest.raises(FileNotFoundError):
        m.log_trial_results(mock_modeling_folder, model_id, experiment_id, trial_overrides)


@pytest.mark.unit
def test_log_trial_results_missing_metrics_file(mock_modeling_folder, mock_log_file, mock_feature_importance_file, mock_predictions_file, model_id, experiment_id):
    """
    Test case where the performance metrics file is missing,
    expecting FileNotFoundError.
    """
    trial_overrides = {"learning_rate": 0.01}

    # Don't create the metrics file to simulate it being missing

    # Call the function and expect it to raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        m.log_trial_results(mock_modeling_folder, model_id, experiment_id, trial_overrides)


@pytest.mark.unit
def test_log_trial_results_missing_feature_importance_file(mock_modeling_folder, mock_log_file, mock_metrics_file, mock_predictions_file, model_id, experiment_id):
    """
    Test case where the feature importance file is missing,
    expecting 'feature_importance' to be 'N/A'.
    """
    trial_overrides = {"learning_rate": 0.01}

    # Don't create the feature importance file to simulate it being missing

    # Call the function
    trial_log_filename = m.log_trial_results(mock_modeling_folder, model_id, experiment_id, trial_overrides)

    # Check the trial log
    with open(trial_log_filename, 'r', encoding='utf-8') as log_file:
        trial_log = json.load(log_file)

    # Expect the feature importance to be "N/A"
    assert trial_log["feature_importance"] == "N/A"


@pytest.mark.unit
def test_log_trial_results_missing_predictions_file(mock_modeling_folder, mock_log_file, mock_metrics_file, mock_feature_importance_file, model_id, experiment_id):
    """
    Test case where the predictions file is missing, expecting 'predictions' to be 'N/A'.
    """
    trial_overrides = {"learning_rate": 0.01}

    # Don't create the predictions file to simulate it being missing

    # Call the function
    trial_log_filename = m.log_trial_results(mock_modeling_folder, model_id, experiment_id, trial_overrides)

    # Check the trial log
    with open(trial_log_filename, 'r', encoding='utf-8') as log_file:
        trial_log = json.load(log_file)

    # Expect the predictions to be "N/A"
    assert trial_log["predictions"] == "N/A"


@pytest.mark.unit
def test_log_trial_results_empty_metrics_file(mock_modeling_folder, mock_log_file, mock_feature_importance_file, mock_predictions_file, model_id, experiment_id):
    """
    Test case where the metrics CSV file is empty, expecting a pandas.errors.EmptyDataError.
    """
    trial_overrides = {"learning_rate": 0.01}

    # Create an empty metrics file
    metrics_path = mock_modeling_folder / "outputs" / "performance_metrics" / f"metrics_{model_id}.csv"
    pd.DataFrame().to_csv(metrics_path, index=False)

    # Expect an EmptyDataError when trying to read an empty CSV
    with pytest.raises(pd.errors.EmptyDataError):
        m.log_trial_results(mock_modeling_folder, model_id, experiment_id, trial_overrides)


@pytest.mark.unit
def test_log_trial_results_trial_overrides_handling(mock_modeling_folder, mock_log_file, mock_metrics_file, mock_feature_importance_file, mock_predictions_file, model_id, experiment_id):
    """
    Test that the trial overrides are properly logged in the trial log.
    """
    trial_overrides = {"learning_rate": 0.01}

    # Call the function
    trial_log_filename = m.log_trial_results(mock_modeling_folder, model_id, experiment_id, trial_overrides)

    # Check the trial log
    with open(trial_log_filename, 'r', encoding='utf-8') as log_file:
        trial_log = json.load(log_file)

    assert trial_log["trial_overrides"] == {"learning_rate": 0.01}


# ---------------------------------------------- #
# calculate_running_profitability_score() unit tests
# ---------------------------------------------- #

@pytest.mark.unit
def test_calculate_running_profitability_score():
    """
    Test the calculate_running_profitability_score function with a predefined set of
    predictions and returns. This test ensures that:

    1. The function correctly sorts predictions and corresponding returns.
    2. The cumulative model returns are accurately calculated.
    3. The best possible returns are sorted and accumulated.
    4. The running profitability scores are correctly computed by comparing the
       cumulative model returns to the best possible returns.

    The expected output is compared against manually calculated running profitability
    scores, accounting for possible floating-point precision issues.
    """
    # Test inputs
    predictions = [0.55, 0.07, 0.14, 0.02, 0.07, 0.64, 0.04, 0.00, 0.02, 0.39]
    returns = [0.46, -0.1, -0.09, -0.09, -0.01, 0.57, -0.1, -0.01, -0.02, 2.62]

    # Expected outputs
    expected_running_scores = [0.2176, 0.3223, 1.0, 0.9780, 0.9532, 0.9557, 0.9517, 0.9505, 0.9729, 1.0]

    # Call the function
    running_scores = m.calculate_running_profitability_score(predictions, returns)

    # Assert equality (with some tolerance for floating-point precision)
    np.testing.assert_almost_equal(running_scores, expected_running_scores, decimal=3)

@pytest.mark.unit
def test_calculate_running_profitability_score_with_negative_top_performance():
    """
    Test the calculate_running_profitability_score function with a predefined set of
    predictions and returns where the top-scoring prediction has a negative return.

    This test ensures that:
    1. The function correctly handles negative returns in top-scoring predictions.
    2. The cumulative model returns and best possible returns are calculated correctly.
    3. The running profitability scores are computed accurately, even when the top
       performer is negative.

    The expected output is compared against manually calculated running profitability
    scores, accounting for possible floating-point precision issues.
    """
    # Test inputs
    predictions = [0.55, 0.07, 0.14, 0.02, 0.07, 0.64, 0.04, 0.00, 0.02, 0.39]
    returns = [0.46, -0.1, -0.09, -0.09, -0.01, -0.57, -0.1, -0.01, -0.02, 2.62]

    # Expected outputs (running scores manually calculated)
    expected_running_scores = [-0.2176, -0.0357, 0.8176, 0.7909, 0.7632]  # Adjusted for this scenario

    # Call the function
    running_scores = m.calculate_running_profitability_score(predictions, returns)

    # Assert equality (with some tolerance for floating-point precision)
    np.testing.assert_almost_equal(running_scores[:5], expected_running_scores, decimal=3)


@pytest.mark.unit
def test_calculate_profitability_auc_top_20_percent():
    """
    Runs through the logical steps to calculate the percentiled AUC and confirms that the
    function results in the same value.
    """
    predictions = np.array([0.55, 0.07, 0.14, 0.02, 0.07, 0.64, 0.04, 0.00, 0.02, 0.39])
    performances = np.array([0.46, -0.1, -0.09, -0.09, -0.01, 0.57, -0.1, -0.01, -0.02, 2.62])
    top_percentage_filter = 0.2

    # Step 1: Sort predictions and performances
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_predictions = predictions[sorted_indices]
    sorted_performances = performances[sorted_indices]

    # Step 2: Determine the number of top predictions to consider
    n_top = int(len(predictions) * top_percentage_filter)
    assert n_top == 2

    # Step 3: Filter the top predictions and performances
    top_predictions = np.round(sorted_predictions[:n_top], 3)
    top_performances = np.round(sorted_performances[:n_top], 3)
    np.testing.assert_array_equal(top_predictions, np.array([0.640, 0.550]))
    np.testing.assert_array_equal(top_performances, np.array([0.570, 0.460]))

    # Step 4: Calculate running profitability scores
    cumulative_model_returns = np.round(np.cumsum(top_performances), 3)
    best_possible_returns = np.round(np.sort(performances)[::-1][:n_top], 3)
    cumulative_best_returns = np.round(np.cumsum(best_possible_returns), 3)
    running_profitability_scores = np.round(cumulative_model_returns / cumulative_best_returns, 3)
    np.testing.assert_allclose(cumulative_model_returns, np.array([0.570, 1.030]))
    np.testing.assert_allclose(best_possible_returns, np.array([2.620, 0.570]))
    np.testing.assert_allclose(cumulative_best_returns, np.array([2.620, 3.190]))
    np.testing.assert_allclose(running_profitability_scores, np.array([0.218, 0.323]))

    # Step 5: Calculate the area under the curve
    x = np.linspace(0, 1, n_top)
    expected_auc = np.trapezoid(running_profitability_scores, x)
    np.testing.assert_almost_equal(expected_auc, 0.27025, decimal=3)

    # Final step: Compare with the function output
    calculated_auc = m.calculate_profitability_auc(predictions, performances, top_percentage_filter)
    np.testing.assert_almost_equal(calculated_auc, expected_auc, decimal=3,
                                   err_msg="Calculated Profitability AUC doesn't match expected value")



# ======================================================== #
#                                                          #
#            I N T E G R A T I O N   T E S T S             #
#                                                          #
# ======================================================== #

# ---------------------------------- #
# set up config and module-level fixtures
# ---------------------------------- #

@pytest.fixture(scope="session")
def config():
    """
    Fixture to load the configuration from the YAML file.
    """
    return load_config('tests/test_config/test_config.yaml')

@pytest.fixture(scope="session")
def metrics_config():
    """
    Fixture to load the configuration from the YAML file.
    """
    return load_config('tests/test_config/test_metrics_config.yaml')

@pytest.fixture(scope="session")
def buysell_metrics_df():
    """
    Fixture to load the buysell_metrics_df from the fixtures folder.
    """
    buysell_metrics_df = pd.read_csv('tests/fixtures/buysell_metrics_df.csv')
    buysell_metrics_df['date'] = pd.to_datetime(buysell_metrics_df['date']).astype('datetime64[ns]')
    return buysell_metrics_df

# ---------------------------------- #
# flatten_coin_date_df() integration tests
# ---------------------------------- #
