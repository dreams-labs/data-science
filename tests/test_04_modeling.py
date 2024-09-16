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


import sys
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pytest
from unittest import mock
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from dreams_core import core as dc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import modeling as m # type: ignore[reportMissingImports]
from utils import load_config # type: ignore[reportMissingImports]

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

# Helper function to create a DataFrame with specified features, target, and coin_id
def create_dataframe(features, target, coin_id=True):
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
    return "test_modeling"  # Relative path to the test_modeling folder

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
def test_basic_functionality(mock_open, mock_dump, sample_data, setup_modeling_folders, modeling_folder):
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

