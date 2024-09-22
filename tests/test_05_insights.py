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
# pyright: reportMissingModuleSource=false

import sys
import os
from unittest import mock
import pandas as pd
from dotenv import load_dotenv
import pytest
from dreams_core import core as dc

# pyright: reportMissingImports=false
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from utils import load_config
import feature_engineering as fe
import coin_wallet_metrics as cwm
import insights as i

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

@pytest.mark.unit
def test_validate_experiments_yaml_success(tmpdir):
    """Unit Test: Success case where variables from 2 different config files are retrieved correctly."""

    # Create the config folder and files
    config_folder = tmpdir.mkdir("config_folder")

    # Create experiments_config.yaml with variable_overrides
    experiment_config = """
    variable_overrides:
      config:
        training_data:
          modeling_period_duration:
            - 14
            - 30
      modeling_config:
        learning_rate:
          - 0.001
        batch_size:
          - 16
    """
    config_folder.join("experiments_config.yaml").write(experiment_config)

    # Create config.yaml
    config = """
    training_data:
      modeling_period_duration:
        - 14
        - 30
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

    # Assert that both config and modeling_config sections are validated
    assert len(configurations) == 2  # Two sections: config and modeling_config

    # Check the overrides for 'config'
    config_overrides = dict(configurations)["config"]
    assert "training_data" in config_overrides
    assert config_overrides["training_data"]["modeling_period_duration"] == [14, 30]

    # Check the overrides for 'modeling_config'
    modeling_overrides = dict(configurations)["modeling_config"]
    assert "learning_rate" in modeling_overrides
    assert modeling_overrides["learning_rate"] == [0.001]
    assert modeling_overrides["batch_size"] == [16]


@pytest.mark.unit
def test_validate_experiments_yaml_missing_file(tmpdir):
    """Unit Test: Failure case where a referenced config file does not exist."""

    # Create the config folder and files
    config_folder = tmpdir.mkdir("config_folder")

    # Create experiments_config.yaml referencing a non-existent file in variable_overrides
    experiment_config = """
    variable_overrides:
      config:
        training_data:
          modeling_period_duration:
            - 14
      config_missing:
        param_x:
          - value
    """
    config_folder.join("experiments_config.yaml").write(experiment_config)

    # Create config.yaml
    config = """
    training_data:
      modeling_period_duration:
        - 14
    """
    config_folder.join("config.yaml").write(config)

    # Verify that it raises FileNotFoundError for the missing config file
    with pytest.raises(FileNotFoundError, match="config_missing.yaml not found in"):
        i.validate_experiments_yaml(str(config_folder))


@pytest.mark.unit
def test_validate_experiments_yaml_invalid_key(tmpdir):
    """Unit Test: Failure case where a referenced key in variable_overrides doesn't exist in the config files."""

    # Create the config folder and files
    config_folder = tmpdir.mkdir("config_folder")

    # Create experiments_config.yaml with an invalid key
    experiment_config = """
    variable_overrides:
      config:
        training_data:
          modeling_period_duration:
            - 14
        invalid_key:
            - non_existent
    """
    config_folder.join("experiments_config.yaml").write(experiment_config)

    # Create config.yaml
    config = """
    training_data:
      modeling_period_duration:
        - 14
    """
    config_folder.join("config.yaml").write(config)

    # Verify that it raises a ValueError for the invalid key
    with pytest.raises(ValueError, match="Key 'invalid_key' in variable_overrides not found in config.yaml"):
        i.validate_experiments_yaml(str(config_folder))


# ---------------------------------------------- #
# validate_experiments_yaml() unit tests
# ---------------------------------------------- #

@pytest.mark.unit
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
      wallet_min_coins: 2
    """
    metrics_config_yaml = """
    wallet_cohorts:
      sharks:
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
        'metrics_config.wallet_cohorts.sharks.buyers_new.aggregations.mean.scaling': 'standard',
        'modeling_config.preprocessing.drop_features': ['buyers_new_median'],
        'modeling_config.target_variables.moon_threshold': 0.3
    }

    # Run the function
    config, metrics_config, modeling_config = i.prepare_configs(str(config_folder), override_params)

    # Assert the overrides were applied correctly
    assert config['data_cleaning']['inflows_filter'] == 10000000
    assert config['training_data']['modeling_period_duration'] == 14
    assert metrics_config['wallet_cohorts']['sharks']['buyers_new']['aggregations']['mean']['scaling'] == 'standard'
    assert modeling_config['preprocessing']['drop_features'] == ['buyers_new_median']
    assert modeling_config['target_variables']['moon_threshold'] == 0.3


@pytest.mark.unit
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
      wallet_min_coins: 2
    """
    metrics_config_yaml = """
    wallet_cohorts:
      sharks:
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
# Mock DataFrame to simulate profits_df
@pytest.fixture
def mock_profits_df():  # pylint: disable=C0116 # docstring
    return pd.DataFrame({
        'wallet_address': ['wallet1', 'wallet2'],
        'profitability': [1000, 2000]
    })

# Mock DataFrame to simulate prices_df
@pytest.fixture
def mock_prices_df():  # pylint: disable=C0116 # docstring
    return pd.DataFrame({
        'prices': [100, 200, 300]
    })

# Mock Configuration
@pytest.fixture
def mock_config():  # pylint: disable=C0116 # docstring
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
@mock.patch('insights.td.prepare_profits_data')
@mock.patch('insights.td.calculate_wallet_profitability')
@mock.patch('insights.td.clean_profits_df')

# Correctly patch functions from the module where they are called
@pytest.mark.unit
def test_rebuild_profits_df_if_necessary(
    mock_clean_profits_df, mock_calculate_wallet_profitability,
    mock_prepare_profits_data, mock_retrieve_transfers_data,
    mock_config, mock_profits_df, mock_prices_df, tmpdir):
    """
    Test the case where the config changes and profits_df is rebuilt by calling
    the production functions (mocked to avoid time-consuming operations).
    """

    # Set up the mock return values
    mock_retrieve_transfers_data.return_value = pd.DataFrame({'transfers': [1, 2, 3]})
    mock_prepare_profits_data.return_value = pd.DataFrame({'profits': [1000, 2000]})
    mock_calculate_wallet_profitability.return_value = mock_profits_df
    mock_clean_profits_df.return_value = (mock_profits_df, None)

    # Create a temporary folder for hash checking
    temp_dir = tmpdir.mkdir("modeling")
    temp_folder = os.path.join(temp_dir, "outputs/temp")
    os.makedirs(temp_folder)

    # Call the function to trigger the rebuild with prices_df passed as an argument
    new_profits_df = i.rebuild_profits_df_if_necessary(mock_config, str(temp_dir), mock_prices_df, profits_df=None)

    # Assertions
    mock_retrieve_transfers_data.assert_called_once()
    mock_prepare_profits_data.assert_called_once_with(mock_retrieve_transfers_data.return_value, mock_prices_df)
    mock_calculate_wallet_profitability.assert_called_once()
    mock_clean_profits_df.assert_called_once()

    assert new_profits_df.equals(mock_profits_df)


# Test using pytest timeout decorator
@pytest.mark.timeout(1)  # Set timeout limit of 1 second
@pytest.mark.unit
def test_return_cached_profits_df(mock_config, mock_profits_df, mock_prices_df, tmpdir):
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
    with open(os.path.join(temp_folder, 'config_hash.txt'), 'w', encoding='utf-8') as f:
        f.write(correct_hash)

    # pylint: disable=W0718
    # Call the function and verify the profits_df is returned from memory without rebuilding
    try:
        # Now passing mock_prices_df as an argument
        returned_profits_df = i.rebuild_profits_df_if_necessary(mock_config, str(temp_dir), mock_prices_df, profits_df=mock_profits_df)
        assert returned_profits_df.equals(mock_profits_df)
    except Exception as e:
        pytest.fail(f"Test failed due to timeout: {str(e)}")


# ======================================================== #
#                                                          #
#            I N T E G R A T I O N   T E S T S             #
#                                                          #
# ======================================================== #

# ---------------------------------- #
# set up config and module-level fixtures
# ---------------------------------- #

@pytest.fixture(scope='session')
def config():
    """
    Loads the base config file from the tests/test_config directory using load_config.
    """
    config_path = 'tests/test_config/test_config.yaml'
    return load_config(config_path)

@pytest.fixture(scope='session')
def metrics_config():
    """
    Loads the metrics config file from the tests/test_config directory using load_config.
    """
    metrics_config_path = 'tests/test_config/test_metrics_config.yaml'
    return load_config(metrics_config_path)

@pytest.fixture(scope='session')
def modeling_config():
    """
    Loads the modeling config file from the tests/test_config directory using load_config
    and overrides drop_features and modeling_folder.
    """
    modeling_config_path = 'tests/test_config/test_modeling_config.yaml'
    modeling_config = load_config(modeling_config_path)

    # Override modeling_folder to point to tests/tests_modeling
    modeling_config['modeling']['modeling_folder'] = 'tests/test_modeling'

    return modeling_config


@pytest.fixture(scope='session')
def target_config():
    """
    Loads the target variables config file from the tests/test_config directory using load_config.
    """
    target_config_path = 'tests/test_config/test_target_config.yaml'
    return load_config(target_config_path)

@pytest.fixture(scope='session')
def prices_df():
    """
    Loads the prices_df from a saved CSV for integration testing.
    """
    logger.info("Loading prices_df from saved CSV...")
    prices_df = pd.read_csv('tests/fixtures/prices_df.csv')
    return prices_df


@pytest.fixture(scope='session')
def profits_df():
    """
    Loads the cleaned_profits_df from a saved CSV for integration testing.
    """
    logger.info("Loading cleaned_profits_df from saved CSV...")
    cleaned_profits_df = pd.read_csv('tests/fixtures/cleaned_profits_df.csv')
    return cleaned_profits_df

# ----------------------------------------------- #
# Integration test for build_configured_model_input()
# ----------------------------------------------- #


@pytest.mark.xfail
# this function will be refactored after additional functionality is added \
# to feature eng and modeling
@pytest.mark.integration
def test_build_configured_model_input(config, metrics_config, modeling_config, prices_df, profits_df):
    """
    Runs build_configured_model_input() with the provided config, metrics_config, modeling_config, prices_df,
    and cleaned_profits_df, and ensures that no columns are inadvertently lost between the flattened
    DataFrame and the final training feature set (X_train).
    """

    # Override preprocessing/drop_features to have no columns specified
    modeling_config['preprocessing']['drop_features'] = []

    # 1. Identify cohort of wallets (e.g., sharks) based on the cohort classification logic
    cohort_summary_df = cwm.classify_wallet_cohort(profits_df, config['datasets']['wallet_cohorts']['sharks'])

    # 2. Generate buysell metrics for wallets in the identified cohort
    cohort_wallets = cohort_summary_df[cohort_summary_df['in_cohort']]['wallet_address']
    buysell_metrics_df = cwm.generate_buysell_metrics_df(
        profits_df,
        config['training_data']['training_period_end'],
        cohort_wallets
    )

    # Retrieve the metrics configuration for the first df
    _, df_metrics_config = next(iter(metrics_config['wallet_cohorts'].items()))


    flattened_buysell_metrics_df = fe.flatten_coin_date_df(
        buysell_metrics_df,
        df_metrics_config,
        config['training_data']['training_period_end']
    )
    logger.debug(f"Shape of flattened_buysell_metrics_df: {flattened_buysell_metrics_df.shape}")

    # Run build_configured_model_input
    X_train, X_test, y_train, y_test = i.build_configured_model_input(
        profits_df, prices_df, config, df_metrics_config, modeling_config
    )

    # Assert that the column count in flattened_buysell_metrics_df is 1 more than X_train
    assert flattened_buysell_metrics_df.shape[1] == X_train.shape[1] + 1, \
        "Column count mismatch: flattened_buysell_metrics_df should have 1 more column than X_train"

    logger.debug(f"Shape of X_train: {X_train.shape}")
