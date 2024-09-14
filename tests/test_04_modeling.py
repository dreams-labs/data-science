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
    return load_config('tests/test_config/test_config_metrics.yaml')

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

