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
from unittest import mock
import pandas as pd
from dotenv import load_dotenv
import pytest
from dreams_core import core as dc


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import insights as i # type: ignore[reportMissingImports]
from utils import load_config # type: ignore[reportMissingImports]

load_dotenv()
logger = dc.setup_logger()






# ===================================================== #
#                                                       #
#                 U N I T   T E S T S                   #
#                                                       #
# ===================================================== #

# ---------------------------------------------- #
# validate_experiments_yaml() unit tests
# ---------------------------------------------- #

def test_validate_experiments_yaml_success(tmpdir):
    """Test the success case where 2 variables from 2 different config files are retrieved correctly."""

    # Create the config folder and files
    config_folder = tmpdir.mkdir("config_folder")

    # Create experiment_config.yaml
    experiment_config = """
    config:
      training_period:
        - {"start": "2023-01-01", "end": "2023-06-30"}
      feature_set:
        - basic
    
    modeling_config:
      learning_rate:
        - 0.001
      batch_size:
        - 16
    """
    config_folder.join("experiments_config.yaml").write(experiment_config)

    # Create config.yaml
    config = """
    training_period:
      - {"start": "2023-01-01", "end": "2023-06-30"}
    feature_set:
      - basic
    """
    config_folder.join("config.yaml").write(config)

    # Create modeling_config.yaml
    modeling_config = """
    learning_rate:
      - 0.001
    batch_size:
      - 16
    """
    config_folder.join("modeling_config.yaml").write(modeling_config)

    # Run the function and verify no errors
    configurations = i.validate_experiments_yaml(str(config_folder))
    assert len(configurations) == 2  # Two sections: config and modeling_config


def test_validate_experiments_yaml_missing_file(tmpdir):
    """Test failure case where a referenced config file does not exist."""

    # Create the config folder and files
    config_folder = tmpdir.mkdir("config_folder")

    # Create experiments_config.yaml referencing a non-existent file
    experiment_config = """
    config:
      training_period:
        - {"start": "2023-01-01", "end": "2023-06-30"}
    
    config_missing:
      param_x:
        - value
    """
    config_folder.join("experiments_config.yaml").write(experiment_config)

    # Create config.yaml
    config = """
    training_period:
      - {"start": "2023-01-01", "end": "2023-06-30"}
    """
    config_folder.join("config.yaml").write(config)

    # Verify that it raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        i.validate_experiments_yaml(str(config_folder))


def test_validate_experiments_yaml_invalid_key(tmpdir):
    """Test failure case where a referenced metric or key in experiment_config.yaml doesn't exist in the config files."""

    # Create the config folder and files
    config_folder = tmpdir.mkdir("config_folder")

    # Create experiments_config.yaml
    experiment_config = """
    config:
      training_period:
        - {"start": "2023-01-01", "end": "2023-06-30"}
      invalid_key:
        - non_existent
    """
    config_folder.join("experiments_config.yaml").write(experiment_config)

    # Create config.yaml
    config = """
    training_period:
      - {"start": "2023-01-01", "end": "2023-06-30"}
    """
    config_folder.join("config.yaml").write(config)

    # Verify that it raises a ValueError for the invalid key
    with pytest.raises(ValueError, match="Key 'invalid_key' in experiments_config.yaml not found in config.yaml"):
        i.validate_experiments_yaml(str(config_folder))



# ---------------------------------------------- #
# validate_experiments_yaml() unit tests
# ---------------------------------------------- #

def test_prepare_configs_success(tmpdir):
    """
    Test the success case for prepare_configs with valid override parameters.
    """

    # Create a temporary config folder
    config_folder = tmpdir.mkdir("config_folder")

    # Create mock config files
    config_yaml = """
    data_cleaning:
      inflows_filter: 5000000
      profitability_filter: 10000000
    training_data:
      modeling_period_duration: 30
      shark_wallet_min_coins: 2
    """
    metrics_config_yaml = """
    metrics:
      buyers_new:
        aggregations:
          mean:
            scaling: None
    """
    modeling_config_yaml = """
    preprocessing:
      drop_features: ['total_sellers_sum']
    target_variables:
      moon_threshold: 0.5
    """

    # Write these config files to the temporary directory
    config_file = config_folder.join("config.yaml")
    config_file.write(config_yaml)

    metrics_file = config_folder.join("metrics_config.yaml")
    metrics_file.write(metrics_config_yaml)

    modeling_file = config_folder.join("modeling_config.yaml")
    modeling_file.write(modeling_config_yaml)

    # Valid override parameters
    override_params = {
        'config.data_cleaning.inflows_filter': 10000000,
        'config.training_data.modeling_period_duration': 14,
        'metrics_config.metrics.buyers_new.aggregations.mean.scaling': 'standard',
        'modeling_config.preprocessing.drop_features': ['buyers_new_median'],
        'modeling_config.target_variables.moon_threshold': 0.3
    }

    # Run the function
    config, metrics_config, modeling_config = i.prepare_configs(str(config_folder), override_params)

    # Assert the overrides were applied correctly
    assert config['data_cleaning']['inflows_filter'] == 10000000
    assert config['training_data']['modeling_period_duration'] == 14
    assert metrics_config['metrics']['buyers_new']['aggregations']['mean']['scaling'] == 'standard'
    assert modeling_config['preprocessing']['drop_features'] == ['buyers_new_median']
    assert modeling_config['target_variables']['moon_threshold'] == 0.3


def test_prepare_configs_failure(tmpdir):
    """
    Test the failure case for prepare_configs when an invalid key is used in override_params.
    """

    # Create a temporary config folder
    config_folder = tmpdir.mkdir("config_folder")

    # Create mock config files
    config_yaml = """
    data_cleaning:
      inflows_filter: 5000000
      profitability_filter: 10000000
    training_data:
      modeling_period_duration: 30
      shark_wallet_min_coins: 2
    """
    metrics_config_yaml = """
    metrics:
      buyers_new:
        aggregations:
          mean:
            scaling: None
    """
    modeling_config_yaml = """
    preprocessing:
      drop_features: ['total_sellers_sum']
    target_variables:
      moon_threshold: 0.5
    """

    # Write these config files to the temporary directory
    config_file = config_folder.join("config.yaml")
    config_file.write(config_yaml)

    metrics_file = config_folder.join("metrics_config.yaml")
    metrics_file.write(metrics_config_yaml)

    modeling_file = config_folder.join("modeling_config.yaml")
    modeling_file.write(modeling_config_yaml)

    # Invalid override parameters (this key doesn't exist in the config)
    override_params = {
        'config.data_cleaning.non_existent_filter': 99999  # This will raise an error
    }

    # Run the function and expect a KeyError
    with pytest.raises(KeyError, match="Key 'non_existent_filter' not found"):
        i.prepare_configs(str(config_folder), override_params)



# ---------------------------------------------- #
# rebuild_profits_df_if_necessary() unit tests
# ---------------------------------------------- #
import pytest
from unittest import mock
import pandas as pd
import os

# Mock DataFrame to simulate profits_df
@pytest.fixture
def mock_profits_df():
    return pd.DataFrame({
        'wallet_address': ['wallet1', 'wallet2'],
        'profitability': [1000, 2000]
    })

# Mock Configuration
@pytest.fixture
def mock_config():
    return {
        'training_data': {
            'training_period_start': '2023-01-01',
            'modeling_period_start': '2023-02-01',
            'modeling_period_end': '2023-03-01'
        },
        'data_cleaning': {
            'max_gap_days': 7
        }
    }
@mock.patch('insights.td.retrieve_transfers_data')
@mock.patch('insights.td.retrieve_prices_data')
@mock.patch('insights.td.fill_prices_gaps')
@mock.patch('insights.td.prepare_profits_data')
@mock.patch('insights.td.calculate_wallet_profitability')
@mock.patch('insights.td.clean_profits_df')
# Correctly patch functions from the module where they are called
def test_rebuild_profits_df_if_necessary(
    mock_clean_profits_df, mock_calculate_wallet_profitability, 
    mock_prepare_profits_data, mock_fill_prices_gaps, 
    mock_retrieve_prices_data, mock_retrieve_transfers_data, 
    mock_config, mock_profits_df, tmpdir):
    """
    Test the case where the config changes and profits_df is rebuilt by calling
    the production functions (mocked to avoid time-consuming operations).
    """
        
    # Set up the mock return values
    mock_retrieve_transfers_data.return_value = pd.DataFrame({'transfers': [1, 2, 3]})
    mock_retrieve_prices_data.return_value = pd.DataFrame({'prices': [100, 200, 300]})
    mock_fill_prices_gaps.return_value = (pd.DataFrame({'prices': [100, 200, 300]}), None)
    mock_prepare_profits_data.return_value = pd.DataFrame({'profits': [1000, 2000]})
    mock_calculate_wallet_profitability.return_value = mock_profits_df
    mock_clean_profits_df.return_value = (mock_profits_df, None)

    # Create a temporary folder for hash checking
    temp_dir = tmpdir.mkdir("modeling")
    temp_folder = os.path.join(temp_dir, "outputs/temp")
    os.makedirs(temp_folder)

    # Call the function to trigger the rebuild
    new_profits_df = i.rebuild_profits_df_if_necessary(mock_config, str(temp_dir), profits_df=None)

    # Assertions
    mock_retrieve_transfers_data.assert_called_once()
    mock_retrieve_prices_data.assert_called_once()
    mock_fill_prices_gaps.assert_called_once()
    mock_prepare_profits_data.assert_called_once()
    mock_calculate_wallet_profitability.assert_called_once()
    mock_clean_profits_df.assert_called_once()

    assert new_profits_df.equals(mock_profits_df)

# Test using pytest timeout decorator
@pytest.mark.timeout(1)  # Set timeout limit of 1 second
def test_return_cached_profits_df(mock_config, mock_profits_df, tmpdir):
    """
    Test the case where the config has not changed and the cached profits_df is returned from memory.
    """

    # Create a temporary folder for hash checking
    temp_dir = tmpdir.mkdir("modeling")
    temp_folder = os.path.join(temp_dir, "outputs/temp")
    os.makedirs(temp_folder)

    # Generate the correct hash for the mock_config using i.generate_config_hash
    correct_hash = i.generate_config_hash({**mock_config['training_data'], **mock_config['data_cleaning']})

    # Create a hash file that matches the mock_config
    with open(os.path.join(temp_folder, 'config_hash.txt'), 'w') as f:
        f.write(correct_hash)

    # Call the function and verify the profits_df is returned from memory without rebuilding
    try:
        returned_profits_df = i.rebuild_profits_df_if_necessary(mock_config, str(temp_dir), profits_df=mock_profits_df)
        assert returned_profits_df.equals(mock_profits_df)
    except Exception as e:
        pytest.fail(f"Test failed due to timeout: {str(e)}")

# ======================================================== #
#                                                          #
#            I N T E G R A T I O N   T E S T S             #
#                                                          #
# ======================================================== #

# # ---------------------------------- #
# # set up config and module-level fixtures
# # ---------------------------------- #

# @pytest.fixture(scope="session")
# def config():
#     """
#     Fixture to load the configuration from the YAML file.
#     """
#     return load_config('tests/test_config/test_config.yaml')

# @pytest.fixture(scope="session")
# def metrics_config():
#     """
#     Fixture to load the configuration from the YAML file.
#     """
#     return load_config('tests/test_config/test_metrics_config.yaml')

# @pytest.fixture(scope="session")
# def buysell_metrics_df():
#     """
#     Fixture to load the buysell_metrics_df from the fixtures folder.
#     """
#     buysell_metrics_df = pd.read_csv('tests/fixtures/buysell_metrics_df.csv')
#     buysell_metrics_df['date'] = pd.to_datetime(buysell_metrics_df['date']).astype('datetime64[ns]')
#     return buysell_metrics_df

# # ---------------------------------- #
# # flatten_coin_date_df() integration tests
# # ---------------------------------- #

