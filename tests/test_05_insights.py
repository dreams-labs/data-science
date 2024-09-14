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

