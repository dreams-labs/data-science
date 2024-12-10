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
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pytest
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

import pytest
import yaml
import logging

# Keep fixtures used by multiple tests
@pytest.fixture
def valid_config_data():
    """
    Fixture that provides valid configuration data as a string.
    Used by multiple tests.
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
                            window: [3]
                        rolling:
                            window_duration: 3
                            lookback_periods: 2
                            comparisons:
                                pct_change:
                                    scaling: "none"
                    ema:
                        parameters:
                            window: [3]
    """

@pytest.fixture
def temp_config_file(tmp_path, valid_config_data):
    """
    Fixture that writes the valid configuration data to a temporary file.
    Used by multiple tests.
    """
    config_file = tmp_path / "metrics_config.yaml"
    config_file.write_text(valid_config_data)
    return str(config_file)

@pytest.mark.unit
def test_valid_configuration_loading(temp_config_file, valid_config_data):
    """
    Test that a valid configuration file is correctly loaded and parsed without errors.
    """
    try:
        config = u.load_config(file_path=temp_config_file)
    except Exception as e:
        pytest.fail(f"Loading valid configuration raised an exception: {e}")

    expected_config = yaml.safe_load(valid_config_data)
    assert config == expected_config, "Loaded configuration does not match the expected configuration"

@pytest.mark.unit
def test_missing_required_field(tmp_path, caplog):
    """
    Test that a ValueError is raised when a required field is missing from the configuration.
    """
    config_data = """
    time_series:
      market_data:
        price:
          rolling:
            lookback_periods: 2
            aggregations:
              max:
                scaling: "standard"
    """
    config_file = tmp_path / "metrics_config.yaml"
    config_file.write_text(config_data)

    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(ValueError) as exc_info:
            u.load_config(file_path=str(config_file))

    error_message = str(exc_info.value)
    assert "Validation Error in metrics_config.yaml" in error_message
    assert "Issue: Field required" in error_message
    assert "Location: time_series.market_data.price.rolling" in error_message
    assert "Bad Field: window_duration" in error_message

@pytest.mark.unit
def test_invalid_field_type(tmp_path, caplog):
    """
    Test that a ValueError is raised when a field has an invalid data type.
    """
    config_data = """
    time_series:
      market_data:
        price:
          rolling:
            window_duration: "three"  # Invalid type
            lookback_periods: 2
    """
    config_file = tmp_path / "metrics_config.yaml"
    config_file.write_text(config_data)

    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(ValueError) as exc_info:
            u.load_config(file_path=str(config_file))

    error_message = str(exc_info.value)
    assert "Validation Error in metrics_config.yaml" in error_message
    assert "Issue: Input should be a valid integer, unable to parse string as an integer" in error_message
    assert "Location: time_series.market_data.price.rolling" in error_message
    assert "Bad Field: window_duration" in error_message

@pytest.mark.unit
def test_unrecognized_top_level_field(tmp_path, caplog):
    """
    Test that a ValueError is raised when an unrecognized field is present at the top level.
    """
    config_data = """
    unknown_field: "unexpected"
    time_series:
      market_data:
        price:
          aggregations:
            max:
              scaling: "standard"
    """
    config_file = tmp_path / "metrics_config.yaml"
    config_file.write_text(config_data)

    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(ValueError) as exc_info:
            u.load_config(file_path=str(config_file))

    error_message = str(exc_info.value)
    assert "Validation Error in metrics_config.yaml" in error_message
    assert "Issue: Extra inputs are not permitted" in error_message
    assert "Bad Field: unknown_field" in error_message

@pytest.mark.unit
def test_unrecognized_nested_field(tmp_path, caplog):
    """
    Test that a ValueError is raised when an unrecognized field is present in a nested model.
    """
    config_data = """
    time_series:
      market_data:
        price:
          aggregations:
            max:
              scaling: "standard"
              unknown_param: "invalid"  # Unrecognized field
    """
    config_file = tmp_path / "metrics_config.yaml"
    config_file.write_text(config_data)

    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(ValueError) as exc_info:
            u.load_config(file_path=str(config_file))

    error_message = str(exc_info.value)
    assert "Validation Error in metrics_config.yaml" in error_message
    assert "Issue: Extra inputs are not permitted" in error_message
    assert "Location: time_series.market_data.price.aggregations.max" in error_message
    assert "Bad Field: unknown_param" in error_message

@pytest.mark.unit
def test_missing_remainder_bucket(tmp_path, caplog):
    """
    Test that a ValueError is raised when the 'buckets' list does not include a 'remainder' value.
    """
    config_data = """
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
    config_file = tmp_path / "metrics_config.yaml"
    config_file.write_text(config_data)

    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(ValueError) as exc_info:
            u.load_config(file_path=str(config_file))

    error_message = str(exc_info.value)
    assert "At least one bucket must have the 'remainder' value." in error_message

@pytest.mark.unit
def test_invalid_scaling_type(tmp_path, caplog):
    """
    Test that a ValueError is raised when an invalid scaling type is specified.
    """
    config_data = """
    time_series:
      market_data:
        price:
          aggregations:
            max:
              scaling: "invalid_scale"  # Invalid scaling type
    """
    config_file = tmp_path / "metrics_config.yaml"
    config_file.write_text(config_data)

    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(ValueError) as exc_info:
            u.load_config(file_path=str(config_file))

    error_message = str(exc_info.value)
    assert "Validation Error in metrics_config.yaml" in error_message
    assert "Issue: Input should be 'standard', 'minmax', 'log' or 'none'" in error_message
    assert "Location: time_series.market_data.price.aggregations.max" in error_message
    assert "Bad Field: scaling" in error_message

@pytest.mark.unit
def test_negative_lookback_periods(tmp_path, caplog):
    """
    Test that a ValueError is raised when 'lookback_periods' is negative.
    """
    config_data = """
    time_series:
      market_data:
        price:
          rolling:
            window_duration: 3
            lookback_periods: -2  # Invalid negative value
    """
    config_file = tmp_path / "metrics_config.yaml"
    config_file.write_text(config_data)

    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(ValueError) as exc_info:
            u.load_config(file_path=str(config_file))

    error_message = str(exc_info.value)
    assert "Validation Error in metrics_config.yaml" in error_message
    assert "Issue: Input should be greater than 0" in error_message
    assert "Location: time_series.market_data.price.rolling" in error_message
    assert "Bad Field: lookback_periods" in error_message

@pytest.mark.unit
def test_missing_scaling_in_comparisons(tmp_path, caplog):
    """
    Test that a ValueError is raised when scaling is missing in comparisons.
    """
    config_data = """
    time_series:
      market_data:
        price:
          rolling:
            window_duration: 3
            lookback_periods: 2
            comparisons:
              change: {}  # Missing scaling
    """
    config_file = tmp_path / "metrics_config.yaml"
    config_file.write_text(config_data)

    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(ValueError) as exc_info:
            u.load_config(file_path=str(config_file))

    error_message = str(exc_info.value)
    assert "Validation Error in metrics_config.yaml" in error_message
    assert "Issue: Field required" in error_message
    assert "Location: time_series.market_data.price.rolling.comparisons.change" in error_message
    assert "Bad Field: scaling" in error_message

@pytest.mark.unit
def test_invalid_aggregation_type(tmp_path, caplog):
    """
    Test that a ValueError is raised when an invalid aggregation type is specified.
    """
    config_data = """
    time_series:
      market_data:
        price:
          aggregations:
            average: {}  # Invalid aggregation type
    """
    config_file = tmp_path / "metrics_config.yaml"
    config_file.write_text(config_data)

    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(ValueError) as exc_info:
            u.load_config(file_path=str(config_file))

    error_message = str(exc_info.value)
    assert "Validation Error in metrics_config.yaml" in error_message
    assert "Issue: Input should be" in error_message
    assert "Location: time_series.market_data.price.aggregations.average" in error_message

@pytest.mark.unit
def test_null_in_required_field(tmp_path, caplog):
    """
    Test that a ValidationError is raised when a required field is set to null.
    """
    config_data = """
    time_series:
      market_data:
        price:
          rolling:
            window_duration: null  # Invalid null value
            lookback_periods: 2
    """
    config_file = tmp_path / "metrics_config.yaml"
    config_file.write_text(config_data)

    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(ValueError) as exc_info:
            u.load_config(file_path=str(config_file))

    error_message = str(exc_info.value)
    assert "Validation Error in metrics_config.yaml" in error_message
    assert "Issue: Input should be a valid integer" in error_message
    assert "Location: time_series.market_data.price.rolling" in error_message

@pytest.mark.unit
def test_configuration_with_comments(tmp_path):
    """
    Test that the configuration file with comments and extra whitespace is loaded successfully.
    """
    config_data = """
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
    config_file = tmp_path / "metrics_config.yaml"
    config_file.write_text(config_data)

    try:
        config = u.load_config(file_path=str(config_file))
    except Exception as e:
        pytest.fail(f"Loading configuration with comments raised an exception: {e}")

    assert 'time_series' in config
    assert 'market_data' in config['time_series']
    assert 'price' in config['time_series']['market_data']

@pytest.mark.unit
def test_load_demo_config(tmp_path):
    """
    Test that the demo configuration file loads completely.
    """
    config_data = """
    time_series:
        market_data:
            price:
                aggregations:
                    max:
                        scaling: "standard"
                    last:
                        buckets:
                        - small: .01
                        - medium: 1.0
                        - large: "remainder"
                rolling:
                    aggregations:
                        max:
                            scaling: "standard"
                    comparisons:
                        change:
                            scaling: "standard"
                    window_duration: 3
                    lookback_periods: 2
                indicators:
                    sma:
                        parameters:
                            window: [3]
                        aggregations:
                            max:
                                scaling: "standard"
                        rolling:
                            aggregations:
                                mean:
                                    scaling: "standard"
                            comparisons:
                                pct_change:
                                    scaling: "none"
                            window_duration: 3
                            lookback_periods: 2
                    ema:
                        parameters:
                            window: [3]
                        aggregations:
                            mean:
                                scaling: "standard"
    """
    config_file = tmp_path / "metrics_config.yaml"
    config_file.write_text(config_data)

    config = u.load_config(file_path=str(config_file))

    def count_keys(d):
        count = 0
        for _, value in d.items():
            count += 1
            if isinstance(value, dict):
                count += count_keys(value)
        return count

    key_count = count_keys(config)
    assert key_count == 39



# ---------------------------------------------- #
# load_config(metrics_config.yaml) unit tests
# ---------------------------------------------- #

@pytest.mark.unit
def test_winsorize_basic_outliers():
    """
    Test winsorization of a series with clear outliers.

    Verifies:
    - Correct clipping at both tails
    - Preservation of middle values
    - Output is a new object
    - Length and index preservation
    """
    # Create sample data with clear outliers
    input_data = pd.Series([1, 2, 3, 3, 4, 4, 5, 5, 20, 100])

    # Store original object id to verify no modification
    original_id = id(input_data)

    # Apply winsorization
    result = u.winsorize(input_data, cutoff=0.1)

    # Expected result based on percentile calculations
    expected = pd.Series([2, 2, 3, 3, 4, 4, 5, 5, 20, 20])

    # Verify results
    assert id(result) != original_id, "Function should return a new object"
    assert len(result) == len(input_data), "Length should be preserved"
    assert result.index.equals(input_data.index), "Index should be preserved"
    assert np.allclose(result, expected, equal_nan=True), "Values should be winsorized correctly"

    # Verify middle values unchanged (indices 1 to 8)
    assert np.allclose(
        result[1:8],
        input_data[1:8],
        equal_nan=True
    ), "Middle values should remain unchanged"



@pytest.mark.unit
def test_winsorize_with_nans():
    """
    Test winsorization of a series containing NaN values.

    Verifies:
    - NaN values remain NaN in output
    - Correct bounds calculation excluding NaN values
    - Proper winsorization of non-NaN values
    """
    input_data = pd.Series([1, np.nan, 3, 3, 4, np.nan, 5, 5, 20, 100])
    # 8 valid values, so cutoff of 0.125 will winsorize at least one value on each end

    result = u.winsorize(input_data, cutoff=0.125)

    # Expected: NaNs preserved, lowest becomes 3, highest becomes 5
    expected = pd.Series([3, np.nan, 3, 3, 4, np.nan, 5, 5, 20, 20])

    assert np.allclose(result, expected, equal_nan=True), "Values should be winsorized correctly with NaNs preserved"
    assert result.isna().sum() == input_data.isna().sum(), "Number of NaN values should remain unchanged"





# ======================================================== #
#                                                          #
#            I N T E G R A T I O N   T E S T S             #
#                                                          #
# ======================================================== #
