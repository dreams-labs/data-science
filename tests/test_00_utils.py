"""
tests used to audit the files in the data-science/src folder
"""
# pylint: disable=C0301 # line over 100 chars
# pylint: disable=C0413 # import not at top of doc (due to local import)
# pylint: disable=E0401 # can't find import (due to local import)
# pylint: disable=W0621 # redefining from outer scope triggering on pytest fixtures
# pylint: disable=W0718 # catching too general Exception
# pylint: disable=W1203 # fstrings in logs
# pyright: reportMissingModuleSource=false

import sys
import os
import logging
import yaml
from dotenv import load_dotenv
import pytest
from pydantic import ValidationError
from dreams_core import core as dc

# pyright: reportMissingImports=false
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import utils as u

load_dotenv()
logger = dc.setup_logger()



# ===================================================== #
#                                                       #
#                 U N I T   T E S T S                   #
#                                                       #
# ===================================================== #

# ---------------------------------------------- #
# load_config(metrics_config.yaml) unit tests
# ---------------------------------------------- #

@pytest.fixture
def valid_config_data():
    """
    Fixture that provides valid configuration data as a string.
    """
    return """
    time_series:
      market_data:
        price:
          aggregations:
            max:
              scaling: "standard"
            last:
              buckets:
                - small: 0.01
                - medium: 1.0
                - large: "remainder"
          rolling:
            window_duration: 3
            lookback_periods: 2
            aggregations:
              max:
                scaling: "standard"
            comparisons:
              change:
                scaling: "standard"
          indicators:
            sma:
              parameters:
                period: 3
              rolling:
                window_duration: 3
                lookback_periods: 2
                comparisons:
                  pct_change:
                    scaling: "none"
            ema:
              parameters:
                period: 3
    """

@pytest.fixture
def temp_config_file(tmp_path, valid_config_data):
    """
    Fixture that writes the valid configuration data to a temporary file and returns the file path.
    """

    config_file = tmp_path / "metrics_config.yaml"
    config_file.write_text(valid_config_data)
    return str(config_file)


@pytest.mark.unit
def test_valid_configuration_loading(temp_config_file, valid_config_data):
    """
    Test that a valid configuration file is correctly loaded and parsed without errors.
    """

    # Load the configuration using the u.load_config function
    try:
        config = u.load_config(file_path=temp_config_file)
    except Exception as e:
        pytest.fail(f"Loading valid configuration raised an exception: {e}")

    # Parse the expected configuration from the valid_config_data
    expected_config = yaml.safe_load(valid_config_data)

    # Assert that the loaded configuration matches the expected configuration
    assert config == expected_config, "Loaded configuration does not match the expected configuration"

@pytest.fixture
def config_missing_required_field():
    """
    Fixture that provides configuration data missing a required field ('window_duration' under 'rolling').
    """
    return """
    time_series:
      market_data:
        price:
          rolling:
            lookback_periods: 2
            aggregations:
              max:
                scaling: "standard"
    """

@pytest.fixture
def temp_config_file_missing_field(tmp_path, config_missing_required_field):
    """
    Fixture that writes the invalid configuration data to a temporary file and returns the file path.
    """
    config_file = tmp_path / "metrics_config.yaml"
    config_file.write_text(config_missing_required_field)
    return str(config_file)


@pytest.mark.unit
def test_missing_required_field(temp_config_file_missing_field, caplog):
    """
    Test that a ValidationError is raised when a required field is missing from the configuration.
    """

    # Suppress error logs during the test by setting the logging level to CRITICAL
    with caplog.at_level(logging.CRITICAL):
        # Attempt to load the configuration using the u.load_config function
        with pytest.raises(ValidationError) as exc_info:
            u.load_config(file_path=temp_config_file_missing_field)

    # Extract the exception information
    errors = exc_info.value.errors()

    # Check that the error message indicates 'window_duration' is missing
    error_found = False
    for error in errors:
        if 'window_duration' in error['loc']:
            error_found = True
            assert error['type'] == 'missing'
            assert error['msg'] == 'Field required'
            break

    assert error_found, "ValidationError for missing 'window_duration' field was not raised as expected"

@pytest.fixture
def config_invalid_field_type():
    """
    Fixture that provides configuration data with an invalid field type ('window_duration' as a string).
    """
    return """
    time_series:
      market_data:
        price:
          rolling:
            window_duration: "three"  # Invalid type
            lookback_periods: 2
    """

@pytest.fixture
def temp_config_file_invalid_type(tmp_path, config_invalid_field_type):
    """
    Fixture that writes the invalid configuration data to a temporary file and returns the file path.
    """
    config_file = tmp_path / "metrics_config.yaml"
    config_file.write_text(config_invalid_field_type)
    return str(config_file)


@pytest.mark.unit
def test_invalid_field_type(temp_config_file_invalid_type, caplog):
    """
    Test that a ValidationError is raised when a field has an invalid data type.
    """

    # Suppress error logs during the test by setting the logging level to CRITICAL
    with caplog.at_level(logging.CRITICAL):
        # Attempt to load the configuration using the u.load_config function
        with pytest.raises(ValidationError) as exc_info:
            u.load_config(file_path=temp_config_file_invalid_type)

    # Extract the exception information
    errors = exc_info.value.errors()

    # Check that the error message indicates 'window_duration' has an invalid type
    error_found = False
    for error in errors:
        if 'window_duration' in error['loc']:
            error_found = True
            assert error['type'] == 'int_parsing'
            assert 'should be a valid integer' in error['msg']
            break

    assert error_found, "ValidationError for invalid 'window_duration' type was not raised as expected"

@pytest.fixture
def config_unrecognized_top_level_field():
    """
    Fixture that provides configuration data with an unrecognized field at the top level.
    """
    return """
    unknown_field: "unexpected"
    time_series:
      market_data:
        price:
          aggregations:
            max:
              scaling: "standard"
    """

@pytest.fixture
def temp_config_file_unrecognized_field(tmp_path, config_unrecognized_top_level_field):
    """
    Fixture that writes the invalid configuration data to a temporary file and returns the file path.
    """
    config_file = tmp_path / "metrics_config.yaml"
    config_file.write_text(config_unrecognized_top_level_field)
    return str(config_file)


@pytest.mark.unit
def test_unrecognized_top_level_field(temp_config_file_unrecognized_field, caplog):
    """
    Test that a ValidationError is raised when an unrecognized field is present at the top level of the configuration.
    """

    # Suppress error logs during the test by setting the logging level to CRITICAL
    with caplog.at_level(logging.CRITICAL):
        # Attempt to load the configuration using the u.load_config function
        with pytest.raises(ValidationError) as exc_info:
            u.load_config(file_path=temp_config_file_unrecognized_field)

    # Extract the exception information
    errors = exc_info.value.errors()

    # Check that the error message indicates 'unknown_field' is not permitted
    error_found = False
    for error in errors:
        if error['loc'] == ('unknown_field',):
            error_found = True
            assert error['type'] == 'extra_forbidden'
            assert error['msg'] == 'Extra inputs are not permitted'
            break

    assert error_found, "ValidationError for unrecognized 'unknown_field' at top level was not raised as expected"

@pytest.fixture
def config_unrecognized_nested_field():
    """
    Fixture that provides configuration data with an unrecognized field in a nested model.
    """
    return """
    time_series:
      market_data:
        price:
          aggregations:
            max:
              scaling: "standard"
              unknown_param: "invalid"  # Unrecognized field
    """

@pytest.fixture
def temp_config_file_unrecognized_nested_field(tmp_path, config_unrecognized_nested_field):
    """
    Fixture that writes the invalid configuration data to a temporary file and returns the file path.
    """
    config_file = tmp_path / "metrics_config.yaml"
    config_file.write_text(config_unrecognized_nested_field)
    return str(config_file)


@pytest.mark.unit
def test_unrecognized_nested_field(temp_config_file_unrecognized_nested_field, caplog):
    """
    Test that a ValidationError is raised when an unrecognized field is present in a nested model.
    """
    # Suppress error logs during the test by setting the logging level to CRITICAL
    with caplog.at_level(logging.CRITICAL):
        # Attempt to load the configuration using the u.load_config function
        with pytest.raises(ValidationError) as exc_info:
            u.load_config(file_path=temp_config_file_unrecognized_nested_field)

    # Extract the exception information
    errors = exc_info.value.errors()

    # Check that the error message indicates 'unknown_param' is not permitted under 'max' aggregation
    error_found = False
    for error in errors:
        if 'unknown_param' in error['loc']:
            error_found = True
            assert error['type'] == 'extra_forbidden'
            assert error['msg'] == 'Extra inputs are not permitted'
            break

    assert error_found, "ValidationError for unrecognized 'unknown_param' in nested model was not raised as expected"

@pytest.fixture
def config_missing_remainder_bucket():
    """
    Fixture that provides configuration data missing the 'remainder' bucket in the 'buckets' list.
    """
    return """
    time_series:
      market_data:
        price:
          aggregations:
            last:
              buckets:
                - small: 0.01
                - medium: 1.0
                # Missing 'remainder' bucket
    """

@pytest.fixture
def temp_config_file_missing_remainder(tmp_path, config_missing_remainder_bucket):
    """
    Fixture that writes the invalid configuration data to a temporary file and returns the file path.
    """
    config_file = tmp_path / "metrics_config.yaml"
    config_file.write_text(config_missing_remainder_bucket)
    return str(config_file)


@pytest.mark.unit
def test_missing_remainder_bucket(temp_config_file_missing_remainder, caplog):
    """
    Test that a ValueError is raised when the 'buckets' list does not include a 'remainder' value.
    """
    # Suppress error logs during the test by setting the logging level to CRITICAL
    with caplog.at_level(logging.CRITICAL):
        # Attempt to load the configuration using the u.load_config function
        with pytest.raises(ValueError) as exc_info:
            u.load_config(file_path=temp_config_file_missing_remainder)

    # Assert that the error message indicates that 'remainder' bucket is missing
    error_message = str(exc_info.value)
    assert "At least one bucket must have the 'remainder' value." in error_message, \
        "ValueError for missing 'remainder' in buckets was not raised as expected"

@pytest.fixture
def config_invalid_scaling_type():
    """
    Fixture that provides configuration data with an invalid scaling type.
    """
    return """
    time_series:
      market_data:
        price:
          aggregations:
            max:
              scaling: "invalid_scale"  # Invalid scaling type
    """

@pytest.fixture
def temp_config_file_invalid_scaling(tmp_path, config_invalid_scaling_type):
    """
    Fixture that writes the invalid configuration data to a temporary file and returns the file path.
    """
    config_file = tmp_path / "metrics_config.yaml"
    config_file.write_text(config_invalid_scaling_type)
    return str(config_file)


@pytest.mark.unit
def test_invalid_scaling_type(temp_config_file_invalid_scaling, caplog):
    """
    Test that a ValidationError is raised when an invalid scaling type is specified.
    """
    # Suppress error logs during the test by setting the logging level to CRITICAL
    with caplog.at_level(logging.CRITICAL):
        # Attempt to load the configuration using the u.load_config function
        with pytest.raises(ValidationError) as exc_info:
            u.load_config(file_path=temp_config_file_invalid_scaling)

    # Extract the exception information
    errors = exc_info.value.errors()

    # Check that the error message indicates invalid scaling type
    error_found = False
    for error in errors:
        if 'scaling' in error['loc']:
            error_found = True
            assert error['type'] == 'enum'
            assert "Input should be 'standard'," in error['msg']
            assert 'invalid_scale' in error['input']
            break

    assert error_found, "ValidationError for invalid 'scaling' type was not raised as expected"

@pytest.fixture
def config_negative_lookback_periods():
    """
    Fixture that provides configuration data with a negative 'lookback_periods' value.
    """
    return """
    time_series:
      market_data:
        price:
          rolling:
            window_duration: 3
            lookback_periods: -2  # Invalid negative value
    """

@pytest.fixture
def temp_config_file_negative_lookback(tmp_path, config_negative_lookback_periods):
    """
    Fixture that writes the invalid configuration data to a temporary file and returns the file path.
    """
    config_file = tmp_path / "metrics_config.yaml"
    config_file.write_text(config_negative_lookback_periods)
    return str(config_file)


@pytest.mark.unit
def test_negative_lookback_periods(temp_config_file_negative_lookback, caplog):
    """
    Test that a ValidationError is raised when 'lookback_periods' is negative.
    """
    # Suppress error logs during the test by setting the logging level to CRITICAL
    with caplog.at_level(logging.CRITICAL):
        # Attempt to load the configuration using the u.load_config function
        with pytest.raises(ValidationError) as exc_info:
            u.load_config(file_path=temp_config_file_negative_lookback)

    # Extract the exception information
    errors = exc_info.value.errors()

    # Check that the error message indicates 'lookback_periods' must be greater than 0
    error_found = False
    for error in errors:
        if 'lookback_periods' in error['loc']:
            error_found = True
            # Depending on Pydantic version, adjust the error type and message checks
            assert error['type'] in ('greater_than', 'value_error.number.not_gt')
            assert 'greater than 0' in error['msg']
            break

    assert error_found, "ValidationError for negative 'lookback_periods' was not raised as expected"

@pytest.fixture
def config_missing_scaling_in_comparisons():
    """
    Fixture providing config data where a comparison lacks a scaling configuration.
    """
    return """
    time_series:
      market_data:
        price:
          rolling:
            window_duration: 3
            lookback_periods: 2
            comparisons:
              change: {}  # Missing scaling
    """

@pytest.fixture
def temp_config_file_missing_scaling(tmp_path, config_missing_scaling_in_comparisons):
    """
    Fixture that writes the invalid config data to a temp file.
    """
    config_file = tmp_path / "metrics_config.yaml"
    config_file.write_text(config_missing_scaling_in_comparisons)
    return str(config_file)


@pytest.mark.unit
def test_missing_scaling_in_comparisons(temp_config_file_missing_scaling, caplog):
    """
    Test that a ValidationError is raised when scaling is missing in comparisons.
    """
    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(ValidationError) as exc_info:
            u.load_config(file_path=temp_config_file_missing_scaling)
    error_message = str(exc_info.value)
    # Assert that the error message indicates that 'scaling' field is required
    assert "Field required" in error_message
    assert "comparisons.change.scaling" in error_message

@pytest.fixture
def config_invalid_aggregation_type():
    """
    Fixture providing config data with an invalid aggregation type.
    """
    return """
    time_series:
      market_data:
        price:
          aggregations:
            average: {}  # Invalid aggregation type
    """

@pytest.fixture
def temp_config_file_invalid_aggregation(tmp_path, config_invalid_aggregation_type):
    """
    Fixture that writes the invalid config data to a temp file.
    """
    config_file = tmp_path / "metrics_config.yaml"
    config_file.write_text(config_invalid_aggregation_type)
    return str(config_file)


@pytest.mark.unit
def test_invalid_aggregation_type(temp_config_file_invalid_aggregation, caplog):
    """
    Test that a ValidationError is raised when an invalid aggregation type is specified.
    """
    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(ValidationError) as exc_info:
            u.load_config(file_path=temp_config_file_invalid_aggregation)
    error_message = str(exc_info.value)
    # Assert that the error message indicates invalid aggregation type
    assert "average" in error_message
    assert "aggregations" in error_message
    assert "Input should be" in error_message


@pytest.fixture
def config_missing_required_sections():
    """
    Fixture providing an empty configuration data, missing required sections.
    """
    return """
    # Empty configuration
    """


@pytest.fixture
def config_null_in_required_field():
    """
    Fixture providing config data where a required field is set to null.
    """
    return """
    time_series:
      market_data:
        price:
          rolling:
            window_duration: null  # Invalid null value
            lookback_periods: 2
    """

@pytest.fixture
def temp_config_file_null_in_field(tmp_path, config_null_in_required_field):
    """
    Fixture that writes the invalid config data to a temp file.
    """
    config_file = tmp_path / "metrics_config.yaml"
    config_file.write_text(config_null_in_required_field)
    return str(config_file)

@pytest.mark.unit
def test_null_in_required_field(temp_config_file_null_in_field, caplog):
    """
    Test that a ValidationError is raised when a required field is set to null.
    """
    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(ValidationError) as exc_info:
            u.load_config(file_path=temp_config_file_null_in_field)
    error_message = str(exc_info.value)
    # Assert that the error message indicates that 'window_duration' must not be null
    assert "input should be a valid integer" in error_message.lower()
    assert "window_duration" in error_message

@pytest.fixture
def config_with_comments_and_whitespace():
    """
    Fixture providing a valid configuration with comments and extra whitespace.
    """
    return """
    # This is a comment

    time_series:

      market_data:
        price:
          aggregations:
            max:
              scaling: "standard"  # Inline comment
            last:
              buckets:
                - small: 0.01
                - medium: 1.0
                - large: "remainder"
    """

@pytest.fixture
def temp_config_file_with_comments(tmp_path, config_with_comments_and_whitespace):
    """
    Fixture that writes the configuration data with comments to a temp file.
    """
    config_file = tmp_path / "metrics_config.yaml"
    config_file.write_text(config_with_comments_and_whitespace)
    return str(config_file)

@pytest.mark.unit
def test_configuration_with_comments(temp_config_file_with_comments):
    """
    Test that the configuration file with comments and extra whitespace is loaded successfully.
    """
    try:
        config = u.load_config(file_path=temp_config_file_with_comments)
    except Exception as e:
        pytest.fail(f"Loading configuration with comments raised an exception: {e}")
    # Assert that key sections are present
    assert 'time_series' in config
    assert 'market_data' in config['time_series']
    assert 'price' in config['time_series']['market_data']


# ======================================================== #
#                                                          #
#            I N T E G R A T I O N   T E S T S             #
#                                                          #
# ======================================================== #

