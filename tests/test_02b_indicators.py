"""
tests used to audit the files in the data-science/src folder
"""
# pylint: disable=C0301 # line over 100 chars
# pylint: disable=C0303 # trailing whitespace
# pylint: disable=C0413 # import not at top of doc (due to local import)
# pylint: disable=C0116 # missing docstring
# pylint: disable=W1203 # fstrings in logs
# pylint: disable=W0612 # unused variables (due to test reusing functions with 2 outputs)
# pylint: disable=W0621 # redefining from outer scope triggering on pytest fixtures
# pylint: disable=E0401 # can't find import (due to local import)
# pyright: reportMissingModuleSource=false

import sys
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pytest
from dreams_core import core as dc

# pyright: reportMissingImports=false
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import coin_wallet_metrics.indicators as ind
from utils import load_config

load_dotenv()
logger = dc.setup_logger()





# ===================================================== #
#                                                       #
#                 U N I T   T E S T S                   #
#                                                       #
# ===================================================== #

# ------------------------------------------ #
# generate_time_series_indicators() unit tests
# ------------------------------------------ #

@pytest.fixture
def sample_time_series_df():
    """Fixture that provides a sample DataFrame for the time series with multiple coin_ids."""
    data = {
        'coin_id': [1, 1, 1, 2, 2, 2],
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01', '2023-01-02', '2023-01-03'],
        'price': [100, 110, 120, 200, 210, 220]
    }
    df = pd.DataFrame(data)
    return df

@pytest.fixture
def sample_metrics_config():
    """Fixture that provides a sample metrics configuration for time series analysis."""
    return {
        'time_series': {
            'prices': {
                'price': {
                    'metrics': {
                        'sma': {
                            'parameters': {
                                'period': 2
                            }
                        },
                        'ema': {
                            'parameters': {
                                'period': 2
                            }
                        }
                    }
                }
            }
        }
    }

@pytest.fixture
def sample_config():
    """Fixture that provides a sample general config for the date range."""
    return {
        'training_data': {
            'training_period_start': '2023-01-01',
            'training_period_end': '2023-01-03'
        }
    }

# @pytest.mark.unit
# def test_generate_time_series_indicators_basic_functionality(sample_time_series_df, sample_metrics_config, sample_config):
#     """
#     Test the basic functionality of generate_time_series_indicators to ensure that SMA and EMA
#     are calculated correctly for a simple DataFrame with multiple coin_ids, and that the date
#     range filtering works.
#     """
#     # Convert the date to datetime in the sample data
#     sample_time_series_df['date'] = pd.to_datetime(sample_time_series_df['date'])

#     # Run the generate_time_series_indicators function
#     result_df, _ = generate_time_series_indicators(
#         sample_time_series_df,
#         sample_config,
#         sample_metrics_config['time_series']['prices']['price']['metrics'],
#         value_column='price'
#     )

#     # Expected columns in the result
#     expected_columns = ['coin_id', 'date', 'price', 'price_sma', 'price_ema']

#     # Assert that the columns exist in the result
#     assert all(col in result_df.columns for col in expected_columns), "Missing expected columns in the result."

#     # Assert that SMA and EMA are calculated correctly
#     expected_sma_1 = [100.0, 105.0, 115.0]  # SMA for coin_id=1 with period=2
#     expected_ema_1 = [100.0, 106.666667, 115.555556]  # EMA for coin_id=1 with period=2

#     # Confirm that the SMA result matches the expected, with special logic to handle NaNs
#     for i, (expected, actual) in enumerate(zip(
#         expected_sma_1,
#         result_df[result_df['coin_id'] == 1]['price_sma'].tolist()
#     )):
#         if np.isnan(expected) and np.isnan(actual):
#             continue  # Both values are NaN, so this is considered equal
#         assert expected == actual, f"Mismatch at index {i}: expected {expected}, got {actual}"

#     # Confirm that the EMA result matches the expected
#     assert result_df[result_df['coin_id'] == 1]['price_ema'].tolist() == pytest.approx(
#         expected_ema_1,
#         abs=1e-2
#     ), "EMA calculation incorrect for coin_id=1"

#     # Check for another coin_id
#     expected_sma_2 = [200.0, 205.0, 215.0]  # SMA for coin_id=2 with period=2
#     expected_ema_2 = [200.0, 206.666667, 215.555556]  # EMA for coin_id=2 with period=2

#     # Confirm that the SMA result matches the expected, with special logic to handle NaNs
#     for i, (expected, actual) in enumerate(zip(
#         expected_sma_2,
#         result_df[result_df['coin_id'] == 2]['price_sma'].tolist()
#     )):
#         if np.isnan(expected) and np.isnan(actual):
#             continue  # Both values are NaN, so this is considered equal
#         assert expected == actual, f"Mismatch at index {i}: expected {expected}, got {actual}"

#     # Confirm that the EMA result matches the expected
#     assert result_df[result_df['coin_id'] == 2]['price_ema'].tolist() == pytest.approx(
#         expected_ema_2,
#         abs=1e-2
#     ), "EMA calculation incorrect for coin_id=2"

#     # Confirm that the output df has the same number of rows as the filtered input df
#     assert len(result_df) == len(sample_time_series_df), "Output row count does not match input row count"

# @pytest.mark.unit
# def test_generate_time_series_indicators_different_periods(sample_time_series_df, sample_config):
#     """
#     Test the functionality of generate_time_series_indicators with different periods for SMA and EMA.
#     """
#     # Adjust the sample_metrics_config for different periods
#     sample_metrics_config = {
#         'time_series': {
#             'prices': {
#                 'price': {
#                     'metrics': {
#                         'sma': {
#                             'parameters': {
#                                 'period': 3  # Different period for SMA
#                             }
#                         },
#                         'ema': {
#                             'parameters': {
#                                 'period': 2  # Different period for EMA
#                             }
#                         }
#                     }
#                 }
#             }
#         }
#     }

#     # Convert the date to datetime in the sample data
#     sample_time_series_df['date'] = pd.to_datetime(sample_time_series_df['date'])

#     # Run the generate_time_series_metrics function
#     result_df, _ = generate_time_series_indicators(
#         sample_time_series_df,
#         sample_config,
#         sample_metrics_config['time_series']['prices']['price']['metrics'],
#         value_column='price'
#     )

#     # Expected columns in the result
#     expected_columns = ['coin_id', 'date', 'price', 'price_sma', 'price_ema']

#     # Assert that the columns exist in the result
#     assert all(col in result_df.columns for col in expected_columns), "Missing expected columns in the result."

#     # Expected SMA and EMA values for coin_id=1
#     expected_sma_1 = [100.0, 105.0, 110.0]  # SMA for coin_id=1 with period=3
#     expected_ema_1 = [100.0, 106.666667, 115.555556]  # EMA for coin_id=1 with period=2

#     # Confirm that the SMA result matches the expected, with special logic to handle NaNs
#     for i, (expected, actual) in enumerate(zip(
#         expected_sma_1,
#         result_df[result_df['coin_id'] == 1]['price_sma'].tolist()
#     )):
#         if np.isnan(expected) and np.isnan(actual):
#             continue  # Both values are NaN, so this is considered equal
#         assert expected == actual, f"Mismatch at index {i}: expected {expected}, got {actual}"

#     # Confirm that the EMA result matches the expected
#     assert result_df[result_df['coin_id'] == 1]['price_ema'].tolist() == pytest.approx(
#         expected_ema_1,
#         abs=1e-2
#     ), "EMA calculation incorrect for coin_id=1"

#     # Expected SMA and EMA values for coin_id=2
#     expected_sma_2 = [200.0, 205.0, 210.0]  # SMA for coin_id=2 with period=3
#     expected_ema_2 = [200.0, 206.666667, 215.555556]  # EMA for coin_id=2 with period=2

#     # Confirm that the SMA result matches the expected, with special logic to handle NaNs
#     for i, (expected, actual) in enumerate(zip(
#         expected_sma_2,
#         result_df[result_df['coin_id'] == 2]['price_sma'].tolist()
#     )):
#         if np.isnan(expected) and np.isnan(actual):
#             continue  # Both values are NaN, so this is considered equal
#         assert expected == actual, f"Mismatch at index {i}: expected {expected}, got {actual}"

#     # Confirm that the EMA result matches the expected
#     assert result_df[result_df['coin_id'] == 2]['price_ema'].tolist() == pytest.approx(
#         expected_ema_2,
#         abs=1e-2
#     ), "EMA calculation incorrect for coin_id=2"

# @pytest.mark.unit
# def test_generate_time_series_indicators_value_column_does_not_exist(sample_time_series_df, sample_metrics_config, sample_config):
#     """
#     Test that generate_time_series_indicators raises a KeyError if the value_column does not exist in the time_series_df.
#     """
#     # Use a value_column that doesn't exist in the DataFrame
#     invalid_value_column = 'non_existent_column'

#     with pytest.raises(KeyError, match=f"Input DataFrame does not include column '{invalid_value_column}'"):
#         generate_time_series_indicators(
#             time_series_df=sample_time_series_df,
#             config=sample_config,
#             value_column_indicators_config=sample_metrics_config['time_series']['prices']['price']['metrics'],
#             value_column=invalid_value_column
#         )


# @pytest.mark.unit
# def test_generate_time_series_indicators_value_column_contains_nan(sample_time_series_df, sample_metrics_config, sample_config):
#     """
#     Test that generate_time_series_indicators raises a ValueError if the value_column contains null values.
#     """
#     # Introduce null values into the 'price' column
#     sample_time_series_df.loc[0, 'price'] = None

#     with pytest.raises(ValueError, match="contains null values"):
#         generate_time_series_indicators(
#             time_series_df=sample_time_series_df,
#             config=sample_config,
#             value_column_indicators_config=sample_metrics_config['time_series']['prices']['price']['metrics'],
#             value_column='price'
#         )




# -------------------------------------------- #
# identify_crossovers() unit tests
# -------------------------------------------- #

@pytest.fixture
def input_series_1():
    """
    Fixture for input series1: [1, 2, 3, 4, 5]
    """
    return pd.Series([1, 2, 3, 4, 5])

@pytest.fixture
def input_series_2():
    """
    Fixture for input series2: [3, 3, 3, 3, 3]
    """
    return pd.Series([3, 3, 3, 3, 3])

@pytest.mark.unit
def test_identify_crossovers_scenario_1(input_series_1, input_series_2):
    """
    Unit test for identify_crossovers function - Scenario 1.

    This test checks if the function correctly identifies an upward crossover
    when series1 crosses from below to above series2. It confirms that the function
    returns the correct crossover points only when they happen.
    """
    # Run the function
    result = ind.identify_crossovers(input_series_1, input_series_2)

    # Step-by-step explanation of expected values:
    # At index 0, series1 (1) is below series2 (3): no crossover => result[0] = 0
    # At index 1, series1 (2) is still below series2 (3): no crossover => result[1] = 0
    # At index 2, series1 (3) equals series2 (3): no crossover => result[2] = 0
    # At index 3, series1 (4) crosses above series2 (3): upward crossover => result[3] = 1
    # At index 4, series1 (5) is above series2 (3), but no new crossover: => result[4] = 0

    expected_result = pd.Series([0, 0, 0, 1, 0])

    # Compare the result using np.array_equal
    assert np.array_equal(result.values, expected_result.values), \
        f"Expected {expected_result.values} but got {result.values}"


@pytest.fixture
def input_series_1_scenario_9():
    """
    Fixture for input series1: [2, 2, 3, 4, 5] for Scenario 9.
    """
    return pd.Series([2, 2, 3, 4, 5])

@pytest.fixture
def input_series_2_scenario_9():
    """
    Fixture for input series2: [2, 2, 2, 2, 2] for Scenario 9.
    """
    return pd.Series([2, 2, 2, 2, 2])

@pytest.mark.unit
def test_identify_crossovers_scenario_9(input_series_1_scenario_9, input_series_2_scenario_9):
    """
    Unit test for identify_crossovers function - Scenario 9.

    This test checks if the function correctly identifies an upward crossover
    when series1 is tied with series2 at first and later crosses above.
    """

    # Run the function
    result = ind.identify_crossovers(input_series_1_scenario_9, input_series_2_scenario_9)

    # Step-by-step explanation of expected values:
    # At index 0, series1 (2) is equal to series2 (2): no crossover => result[0] = 0
    # At index 1, series1 (2) is still equal to series2 (2): no crossover => result[1] = 0
    # At index 2, series1 (3) crosses above series2 (2): upward crossover => result[2] = 1
    # At index 3, series1 (4) is still above series2 (2): no new crossover => result[3] = 0
    # At index 4, series1 (5) is still above series2 (2): no new crossover => result[4] = 0

    expected_result = pd.Series([0, 0, 1, 0, 0])

    # Compare the result using np.array_equal
    assert np.array_equal(result.values, expected_result.values), \
        f"Expected {expected_result.values} but got {result.values}"


@pytest.fixture
def input_series_1_scenario_11():
    """
    Fixture for input series1: [2, 3, 4, 5, 3, 2, 1] for Scenario 11.
    """
    return pd.Series([2, 3, 4, 5, 3, 2, 1])

@pytest.fixture
def input_series_2_scenario_11():
    """
    Fixture for input series2: [3, 3, 3, 3, 3, 3, 3] for Scenario 11.
    """
    return pd.Series([3, 3, 3, 3, 3, 3, 3])

@pytest.mark.unit
def test_identify_crossovers_scenario_11(input_series_1_scenario_11, input_series_2_scenario_11):
    """
    Unit test for identify_crossovers function - Scenario 11.

    This test checks if the function correctly identifies both upward and downward crossovers
    when series1 gradually crosses above and then below series2.
    """

    # Run the function
    result = ind.identify_crossovers(input_series_1_scenario_11, input_series_2_scenario_11)

    # Step-by-step explanation of expected values:
    # At index 0, series1 (2) is below series2 (3): no crossover => result[0] = 0
    # At index 1, series1 (3) equals series2 (3): no crossover => result[1] = 0
    # At index 2, series1 (4) crosses above series2 (3): upward crossover => result[2] = 1
    # At index 3, series1 (5) is still above series2 (3): no new crossover => result[3] = 0
    # At index 4, series1 (3) crosses below series2 (3): downward crossover => result[4] = -1
    # At index 5, series1 (2) is below series2 (3): no new crossover => result[5] = 0
    # At index 6, series1 (1) is still below series2 (3): no new crossover => result[6] = 0

    expected_result = pd.Series([0, 0, 1, 0, 0, -1, 0])

    # Compare the result using np.array_equal
    assert np.array_equal(result.values, expected_result.values), \
        f"Expected {expected_result.values} but got {result.values}"




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
def prices_df():
    """
    Fixture to load the prices_df from the fixtures folder.
    """
    return pd.read_csv('tests/fixtures/prices_df.csv')

@pytest.mark.integration
def test_generate_time_seriesindicators_no_nulls_and_row_count(prices_df, config, metrics_config):
    """
    Integration test for cwm.generate_time_series_indicators to confirm that:
    1. The returned DataFrame has no null values.
    2. The number of rows in the output matches the input prices_df.
    """
    # Define dataset key and column name
    value_column = 'price'

    # Identify coins that have complete data for the period
    coin_data_range = prices_df.groupby('coin_id')['date'].agg(['min', 'max'])

    # Full duration coins: Data spans the entire training to modeling period
    training_period_start = pd.to_datetime(config['training_data']['training_period_start'])
    training_period_end = pd.to_datetime(config['training_data']['training_period_end'])

    full_duration_days = (training_period_end - training_period_start).days + 1
    full_duration_coins = coin_data_range[
        (coin_data_range['min'] <= config['training_data']['training_period_start']) &
        (coin_data_range['max'] >= config['training_data']['training_period_end'])
    ].index

    expected_rows = len(full_duration_coins) * full_duration_days

    if 'indicators' in metrics_config['time_series']['market_data']['price'].keys():
        # Run the generate_time_series_indicators function
        full_metrics_df, _ = ind.generate_time_series_indicators(
            time_series_df=prices_df,
            config=config,
            value_column_indicators_config=metrics_config['time_series']['market_data']['price']['indicators'],
            value_column=value_column
        )

        # Check that the number of rows in the result matches the expected number of rows
        assert len(full_metrics_df) == expected_rows, "The number of rows in the output does not match the expected number of rows."

        # Check that there are no null values in the result
        assert not full_metrics_df.isnull().values.any(), "The output DataFrame contains null values."
