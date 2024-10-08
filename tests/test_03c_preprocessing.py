"""
tests used to audit the files in the data-science/src folder
"""
# pylint: disable=C0302 # over 1000 lines
# pylint: disable=C0413 # import not at top of doc (due to local import)
# pylint: disable=W0612 # unused variables (due to test reusing functions with 2 outputs)
# pylint: disable=W0621 # redefining from outer scope triggering on pytest fixtures
# pylint: disable=W1203 # fstrings in logs
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
import feature_engineering.preprocessing as prp

load_dotenv()
logger = dc.setup_logger()



# ===================================================== #
#                                                       #
#                 U N I T   T E S T S                   #
#                                                       #
# ===================================================== #

# ------------------------------------------ #
# preprocess_coin_df() unit tests
# ------------------------------------------ #
@pytest.fixture
def mock_modeling_config():
    """
    Returns a mock modeling configuration dictionary.
    The configuration includes preprocessing options such as features to drop.
    """
    return {
        'preprocessing': {
            'drop_features': ['feature_to_drop']
        }
    }

@pytest.fixture
def mock_metrics_config():
    """
    Returns a mock metrics configuration dictionary.
    This configuration includes settings for scaling different features.
    """
    return {
            'feature_1': {
                'aggregations': {
                    'sum': {'scaling': 'standard'}
                    ,'max': {}
                }
            }
    }

@pytest.fixture
def mock_input_df():
    """
    Creates a mock DataFrame and saves it as a CSV for testing.
    The CSV file is saved in the 'tests/test_modeling/outputs/flattened_outputs' directory.

    Returns:
    - input_path: Path to the CSV file.
    - df: Original mock DataFrame.
    """
    data = {
        'feature_1_sum': [1, 2, 3],
        'feature_to_drop': [10, 20, 30],
        'feature_3': [100, 200, 300]
    }
    df = pd.DataFrame(data)
    input_path = 'tests/test_modeling/outputs/flattened_outputs/mock_input.csv'
    df.to_csv(input_path, index=False)
    return input_path, df

@pytest.mark.unit
def test_preprocess_coin_df_drops_columns(mock_modeling_config, mock_metrics_config, mock_input_df):
    """
    Tests that the preprocess_coin_df function correctly drops the specified columns.

    Steps:
    - Preprocesses the mock DataFrame by dropping the 'feature_to_drop' column.
    - Asserts that the output CSV is created and the column was dropped.
    - Cleans up the test files after execution.
    """
    input_path, original_df = mock_input_df

    # Call the function
    output_df, output_path = prp.preprocess_coin_df(input_path, mock_modeling_config, mock_metrics_config)

    # Check that the output file exists
    assert os.path.exists(output_path), "Output CSV file was not created."

    # Check that the 'feature_to_drop' column is missing in the output DataFrame
    assert 'feature_to_drop' not in output_df.columns, "Column 'feature_to_drop' was not dropped."
    assert len(output_df.columns) == len(original_df.columns) - 1, "Unexpected number of columns after preprocessing."

    # Cleanup (remove the test files)
    os.remove(output_path)
    os.remove(input_path)

@pytest.mark.unit
def test_preprocess_coin_df_scaling(mock_modeling_config, mock_metrics_config, mock_input_df):
    """
    Tests that the preprocess_coin_df function correctly applies scaling to the specified features.

    Steps:
    - Preprocesses the mock DataFrame by applying standard scaling to 'feature_1'.
    - Asserts that the column is scaled correctly.
    - Cleans up the test files after execution.
    """
    input_path, original_df = mock_input_df

    # Declare empty dataset_config
    mock_dataset_config = {}

    # Call the function
    output_df, output_path = prp.preprocess_coin_df(
        input_path, mock_modeling_config, mock_dataset_config, mock_metrics_config
    )

    # Check that 'feature_1' is scaled (mean should be near 0 and std should be near 1)
    scaled_column = output_df['feature_1_sum']
    assert abs(scaled_column.mean()) < 1e-6, "Standard scaling not applied correctly to 'feature_1_sum'."
    assert abs(np.std(scaled_column) - 1) < 1e-6, "Standard scaling not applied correctly to 'feature_1_sum'."

    # Cleanup (remove the test files)
    os.remove(output_path)
    os.remove(input_path)



# ------------------------------------------ #
# create_training_data_df() unit tests
# ------------------------------------------ #

@pytest.fixture
def mock_input_files_value_columns(tmpdir):
    """
    Unit test data for scenario with many duplicate columns and similar filenames.
    """
    # Create the correct subdirectory structure in tmpdir
    preprocessed_output_dir = tmpdir.mkdir("outputs").mkdir("preprocessed_outputs")

    # Create mock filenames and corresponding DataFrames
    filenames = [
        'buysell_metrics_2024-09-13_14-44_model_period_2024-05-01_v0.1.csv',
        'buysell_metrics_2024-09-13_14-45_model_period_2024-05-01_v0.1.csv',
        'buysell_metrics_megasharks_2024-09-13_14-45_model_period_2024-05-01_v0.1.csv',
        'buysell_metrics_megasharks_2024-09-13_14-45_model_period_2024-05-01_v0.2.csv',
        'price_metrics_2024-09-13_14-45_model_period_2024-05-01_v0.1.csv'
    ]

    # Create mock DataFrames for each file
    df1 = pd.DataFrame({'coin_id': [1, 2], 'buyers_new': [100, 200]})
    df2 = pd.DataFrame({'coin_id': [1, 2], 'buyers_new': [150, 250]})
    df3 = pd.DataFrame({'coin_id': [1, 2], 'buyers_new': [110, 210]})
    df4 = pd.DataFrame({'coin_id': [1, 2], 'buyers_new': [120, 220]})
    df5 = pd.DataFrame({'coin_id': [1, 2], 'buyers_new': [130, 230]})

    # Save each DataFrame as a CSV to the correct directory
    for i, df in enumerate([df1, df2, df3, df4, df5]):
        df.to_csv(os.path.join(preprocessed_output_dir, filenames[i]), index=False)

    # Create a tuple list with filenames and 'fill_zeros' strategy
    input_files = [(filenames[i], 'fill_zeros') for i in range(len(filenames))]

    return tmpdir, input_files

@pytest.mark.unit
def test_create_training_data_df(mock_input_files_value_columns):
    """
    Test column renaming logic for clarity when merging multiple files with similar filenames.
    """
    tmpdir, input_files = mock_input_files_value_columns

    # Call the function using tmpdir as the modeling_folder
    merged_df, _ = prp.create_training_data_df(tmpdir, input_files)

    # Check if the columns have the correct suffixes
    expected_columns = [
        'coin_id',
        'buyers_new_buysell_metrics_2024-09-13_14-44',
        'buyers_new_buysell_metrics_2024-09-13_14-45',
        'buyers_new_buysell_metrics_megasharks_2024-09-13_14-45',
        'buyers_new_buysell_metrics_megasharks_2024-09-13_14-45_2',
        'buyers_new_price_metrics'
    ]

    assert list(merged_df.columns) == expected_columns, \
        f"Expected columns: {expected_columns}, but got: {list(merged_df.columns)}"

@pytest.fixture
def mock_input_files(tmpdir):
    """
    Valid input filenames that will be combined with invalid files.
    """
    # Create the correct subdirectory structure in tmpdir
    preprocessed_output_dir = tmpdir.mkdir("outputs").mkdir("preprocessed_outputs")

    # Create mock filenames and corresponding DataFrames with dummy date values
    filenames = [
        'file1_2024-09-13_14-44.csv',
        'file2_2024-09-13_14-45.csv',
        'file3_2024-09-13_14-46.csv'
    ]

    # Create mock DataFrames
    df1 = pd.DataFrame({'coin_id': [1, 2], 'buyers_new': [100, 200]})
    df2 = pd.DataFrame({'coin_id': [1, 2], 'buyers_new': [150, 250]})
    df3 = pd.DataFrame({'coin_id': [1, 2], 'buyers_new': [110, 210]})

    # Save each DataFrame as a CSV to the correct directory
    for i, df in enumerate([df1, df2, df3]):
        df.to_csv(os.path.join(preprocessed_output_dir, filenames[i]), index=False)

    # Create a tuple list with filenames and 'fill_zeros' strategy
    input_files = [(filenames[i], 'fill_zeros') for i in range(len(filenames))]

    return tmpdir, input_files


@pytest.mark.unit
def test_file_not_found(mock_input_files):
    """
    Confirms the error message when an input file does not exist.
    """
    tmpdir, filenames = mock_input_files

    # Simulate one of the files not existing
    filenames = [('file4_nonexistent_2024-09-13_14-47.csv', 'fill_zeros')]

    with pytest.raises(ValueError, match="No DataFrames to merge."):
        prp.create_training_data_df(tmpdir, filenames)


@pytest.mark.unit
def test_missing_coin_id(mock_input_files):
    """
    Confirms the error message when an input file does not have a coin_id column.
    """
    tmpdir, filenames = mock_input_files

    # Create a DataFrame missing the 'coin_id' column
    df_missing_coin_id = pd.DataFrame({'buyers_new': [100, 200]})
    preprocessed_output_dir = os.path.join(tmpdir, 'outputs', 'preprocessed_outputs')
    df_missing_coin_id.to_csv(os.path.join(preprocessed_output_dir,
                                           'file_missing_coin_id_2024-09-13_14-47.csv'),
                                           index=False)

    filenames.append(('file_missing_coin_id_2024-09-13_14-47.csv', 'fill_zeros'))

    with pytest.raises(ValueError, match="coin_id column is missing in file_missing_coin_id_2024-09-13_14-47.csv"):
        prp.create_training_data_df(tmpdir, filenames)


@pytest.mark.unit
def test_duplicate_coin_id(mock_input_files):
    """
    Confirms the error message when an input file has duplicate coin_id rows.
    """
    tmpdir, filenames = mock_input_files
    preprocessed_output_dir = os.path.join(tmpdir, 'outputs', 'preprocessed_outputs')
    bad_file = 'file_duplicate_coin_id_2024-09-13_14-47.csv'

    # Create a DataFrame with duplicate 'coin_id' values
    df_duplicate_coin_id = pd.DataFrame({'coin_id': [1, 1], 'buyers_new': [100, 200]})
    df_duplicate_coin_id.to_csv(os.path.join(preprocessed_output_dir, bad_file), index=False)

    filenames.append((bad_file, 'fill_zeros'))

    with pytest.raises(ValueError, match=f"Duplicate coin_ids found in file: {bad_file}"):
        prp.create_training_data_df(tmpdir, filenames)


# ------------------------------------------ #
# merge_and_fill_training_data() unit tests
# ------------------------------------------ #

@pytest.mark.unit
def test_merge_and_fill_training_data_same_coin_ids():
    """
    Test the merge_and_fill_training_data function with two DataFrames
    that both have coin_id values 1, 2, 3 and the 'fill_zeros' strategy.
    """
    # Create mock DataFrames with the same coin_ids
    df1 = pd.DataFrame({
        'coin_id': [1, 2, 3],
        'metric_1': [100, 200, 300]
    })
    df2 = pd.DataFrame({
        'coin_id': [1, 2, 3],
        'metric_2': [400, 500, 600]
    })

    # fill_zeros happy path
    # ---------------------
    # List of tuples (df, fill_strategy, filename), where 'filename' is a placeholder for logging
    input_dfs = [
        (df1, 'fill_zeros', 'file1'),
        (df2, 'fill_zeros', 'file2')
    ]

    # Run the function
    training_data_df, merge_logs_df = prp.merge_and_fill_training_data(input_dfs)

    # Assert that the merged DataFrame matches the expected DataFrame
    expected_df = pd.DataFrame({
        'coin_id': [1, 2, 3],
        'metric_1': [100, 200, 300],
        'metric_2': [400, 500, 600]
    })
    np.array_equal(training_data_df.values,expected_df.values)

    # Assert that the logs match the expected logs
    expected_logs = pd.DataFrame({
        'file': ['file1', 'file2'],
        'original_count': [3, 3],
        'filled_count': [0, 0],
    })
    np.array_equal(merge_logs_df.values,expected_logs.values)

    # drop_records happy path
    # ---------------------
    # Rerun the same function with drop_records and confirm that the output is identical
    input_dfs = [
        (df1, 'drop_records', 'file1'),
        (df2, 'drop_records', 'file2')
    ]

    # Run the function
    training_data_df, merge_logs_df = prp.merge_and_fill_training_data(input_dfs)

    # Assert that the merged DataFrame matches the expected DataFrame
    np.array_equal(training_data_df.values,expected_df.values)

    # Assert that the logs match the expected logs
    np.array_equal(merge_logs_df.values,expected_logs.values)




@pytest.mark.unit
def test_merge_and_fill_training_data_fill_zeros():
    """
    Test that merge_and_fill_training_data correctly applies the 'fill_zeros' strategy for missing coin_ids.
    """
    # Define mock DataFrames
    df1 = pd.DataFrame({'coin_id': [1, 2, 3], 'metric_1': [10, 20, 30]})
    df2 = pd.DataFrame({'coin_id': [2, 3], 'metric_2': [50, 60]})

    # Expected DataFrame
    expected_df = pd.DataFrame({
        'coin_id': [1, 2, 3],
        'metric_1': [10, 20, 30],
        'metric_2': [0, 50, 60]
    })

    # Call the function
    merged_df, merge_logs_df = prp.merge_and_fill_training_data([
        (df1, 'fill_zeros', 'df1'),
        (df2, 'fill_zeros', 'df2')
    ])

    # Assert that the merged DataFrame matches the expected DataFrame
    assert np.array_equal(merged_df.values, expected_df.values), "merged_df has unexpected values."

    # Check logs
    df1_log = merge_logs_df[merge_logs_df['file'] == 'df1']
    df2_log = merge_logs_df[merge_logs_df['file'] == 'df2']

    # df1 has no filling, but 1 dropped coin_id (coin_id 1 missing from df2)
    assert df1_log['filled_count'].iloc[0] == 0, "df1 should have no filled entries."

    # df2 has 1 filled entry for coin_id 1
    assert df2_log['filled_count'].iloc[0] == 1, "df2 should have 1 filled entry."


@pytest.mark.unit
def test_merge_and_fill_training_data_drop_records():
    """
    Test the merge_and_fill_training_data function when the 'drop_records' strategy is used.
    Ensures that the merge works correctly and that filled_count is logged appropriately.
    """
    # Mock DataFrames
    df1 = pd.DataFrame({
        'coin_id': [1, 2, 3],
        'metric_1': [10, 20, 30]
    })

    df2 = pd.DataFrame({
        'coin_id': [2, 3],
        'metric_2': [200, 300]
    })

    # Expected output when drop_records is applied: rows for coin 1 should be dropped
    expected_df = pd.DataFrame({
        'coin_id': [2, 3],
        'metric_1': [20, 30],
        'metric_2': [200, 300]
    })

    # Run the function
    merged_df, logs_df = prp.merge_and_fill_training_data([
        (df1, 'drop_records', 'df1'),
        (df2, 'drop_records', 'df2')
    ])

    # Assert the merged DataFrame is correct
    assert np.array_equal(merged_df.values, expected_df.values), "merged_df has unexpected values."

    # Assert the logs are correct
    # df1 should have no filled rows, and df2 should also have no filled rows (since we used drop_records)
    expected_logs = pd.DataFrame({
        'file': ['df1', 'df2'],
        'original_count': [3, 2],
        'filled_count': [0, 0]
    })

    assert np.array_equal(logs_df.values, expected_logs.values), "merged_df has unexpected values."


# ------------------------------------------ #
# calculate_coin_returns() unit tests
# ------------------------------------------ #

@pytest.fixture
def valid_prices_df():
    """
    Fixture to create a sample DataFrame with valid price data for multiple coins.
    """
    return pd.DataFrame({
        'coin_id': ['BTC', 'ETH', 'XRP'] * 2,
        'date': ['2023-01-01', '2023-01-01', '2023-01-01', '2023-12-31', '2023-12-31', '2023-12-31'],
        'price': [30000, 2000, 0.5, 35000, 2500, 0.6]
    })

@pytest.fixture
def valid_training_data_config():
    """
    Fixture to create a sample training data configuration.
    """
    return {
        'modeling_period_start': '2023-01-01',
        'modeling_period_end': '2023-12-31'
    }

@pytest.mark.unit
def test_calculate_coin_returns_valid_data(valid_prices_df, valid_training_data_config):
    """
    Test calculate_coin_returns function with valid data for multiple coins.

    This test ensures that the function correctly calculates returns and outcomes
    for all coins when given valid input data.
    """
    returns_df, outcomes_df = prp.calculate_coin_returns(valid_prices_df,
                                                                valid_training_data_config)

    expected_returns = pd.DataFrame({
        'coin_id': ['BTC', 'ETH', 'XRP'],
        'returns': [0.166667, 0.25, 0.2]
    })

    expected_outcomes = pd.DataFrame({
        'coin_id': ['BTC', 'ETH', 'XRP'],
        'outcome': ['returns calculated'] * 3
    })

    assert np.all(np.isclose(returns_df['returns'].values,
                            expected_returns['returns'].values,
                            rtol=1e-4, atol=1e-4))
    assert np.array_equal(outcomes_df.values, expected_outcomes.values)

    # Check if returns values are approximately equal
    for actual, expected in zip(returns_df['returns'], expected_returns['returns']):
        assert actual == pytest.approx(expected, abs=1e-4)


@pytest.fixture
def no_change_prices_df():
    """
    Fixture to create a sample DataFrame with no price change for some coins.
    """
    return pd.DataFrame({
        'coin_id': ['BTC', 'ETH', 'XRP'] * 2,
        'date': ['2023-01-01', '2023-01-01', '2023-01-01', '2023-12-31', '2023-12-31', '2023-12-31'],
        'price': [30000, 2000, 0.5, 30000, 2500, 0.5]
    })

@pytest.mark.unit
def test_calculate_coin_returns_no_change(no_change_prices_df, valid_training_data_config):
    """
    Test calculate_coin_returns function with no price change for some coins.

    This test ensures that the function correctly calculates zero returns for coins
    with no price change and correct returns for others.
    """
    returns_df, outcomes_df = prp.calculate_coin_returns(no_change_prices_df,
                                                                valid_training_data_config)

    expected_returns = pd.DataFrame({
        'coin_id': ['BTC', 'ETH', 'XRP'],
        'returns': [0.0, 0.25, 0.0]
    })

    assert (np.isclose(returns_df['returns'].values,
                       expected_returns['returns'].values,
                       rtol=1e-4, atol=1e-4)).all()

    expected_outcomes = pd.DataFrame({
        'coin_id': ['BTC', 'ETH', 'XRP'],
        'outcome': ['returns calculated'] * 3
    })

    assert np.array_equal(outcomes_df.values, expected_outcomes.values)


@pytest.fixture
def negative_returns_prices_df():
    """
    Fixture to create a sample DataFrame with negative returns for some coins.
    """
    return pd.DataFrame({
        'coin_id': ['BTC', 'ETH', 'XRP'] * 2,
        'date': ['2023-01-01', '2023-01-01', '2023-01-01', '2023-12-31', '2023-12-31', '2023-12-31'],
        'price': [30000, 2000, 0.5, 25000, 2500, 0.4]
    })

@pytest.mark.unit
def test_calculate_coin_returns_negative(negative_returns_prices_df, valid_training_data_config):
    """
    Test calculate_coin_returns function with negative returns for some coins.

    This test ensures that the function correctly calculates negative returns values
    for coins with price decreases and correct returns for others.
    """
    returns_df, outcomes_df = prp.calculate_coin_returns(negative_returns_prices_df,
                                                                valid_training_data_config)

    expected_returns = pd.DataFrame({
        'coin_id': ['BTC', 'ETH', 'XRP'],
        'returns': [-0.1667, 0.25, -0.2]
    })

    assert (np.isclose(returns_df['returns'].values,
                       expected_returns['returns'].values,
                       rtol=1e-4, atol=1e-4)).all()

    expected_outcomes = pd.DataFrame({
        'coin_id': ['BTC', 'ETH', 'XRP'],
        'outcome': ['returns calculated'] * 3
    })

    assert np.array_equal(outcomes_df.values, expected_outcomes.values)


@pytest.fixture
def multiple_datapoints_prices_df():
    """
    Fixture to create a sample DataFrame with multiple data points between start and end dates.
    """
    return pd.DataFrame({
        'coin_id': ['BTC', 'ETH', 'XRP'] * 4,
        'date': ['2023-01-01', '2023-01-01', '2023-01-01',
                 '2023-06-15', '2023-06-15', '2023-06-15',
                 '2023-09-30', '2023-09-30', '2023-09-30',
                 '2023-12-31', '2023-12-31', '2023-12-31'],
        'price': [30000, 2000, 0.5,
                  32000, 2200, 0.55,
                  34000, 2400, 0.58,
                  35000, 2500, 0.6]
    })

@pytest.mark.unit
def test_calculate_coin_returns_multiple_datapoints(multiple_datapoints_prices_df,
                                                        valid_training_data_config):
    """
    Test calculate_coin_returns function with multiple data points between start and end dates.

    This test ensures that the function correctly calculates returns using only start and end dates,
    ignoring intermediate data points.
    """
    returns_df, outcomes_df = prp.calculate_coin_returns(multiple_datapoints_prices_df,
                                                                valid_training_data_config)

    expected_returns = pd.DataFrame({
        'coin_id': ['BTC', 'ETH', 'XRP'],
        'returns': [0.1667, 0.25, 0.2]
    })

    assert (np.isclose(returns_df['returns'].values,
                       expected_returns['returns'].values,
                       rtol=1e-4, atol=1e-4)).all()

    expected_outcomes = pd.DataFrame({
        'coin_id': ['BTC', 'ETH', 'XRP'],
        'outcome': ['returns calculated'] * 3
    })

    assert np.array_equal(outcomes_df.values, expected_outcomes.values)



# ------------------------------------------ #
# create_target_variables_mooncrater() unit tests
# ------------------------------------------ #

@pytest.mark.unit
def test_calculate_mooncrater_targets():
    """
    Tests whether the is_moon and is_crater target variables are calculated correctly.
    """
    # Mock data
    data = {
        'coin_id': ['coin1', 'coin2', 'coin3', 'coin4', 'coin5'],
        # 5% increase, 55% increase, 5% decrease, 55% decrease, 50% increase
        'returns': [0.05, 0.55, -0.05, -0.55, 0.50]
    }
    returns_df = pd.DataFrame(data)

    # Mock configuration
    modeling_config = {
        'target_variables': {
            'moon_threshold': 0.5,  # 50% increase
            'moon_minimum_percent': 0.2,  # 20% of coins should be moons
            'crater_threshold': -0.5,  # 50% decrease
            'crater_minimum_percent': 0.2  # 20% of coins should be craters
        },
        'modeling': {
            'target_column': 'is_moon'
        }
    }

    # Call the function being tested
    target_variables_df = prp.calculate_mooncrater_targets(returns_df, modeling_config)

    # Assertions
    assert len(target_variables_df) == 5
    assert list(target_variables_df.columns) == ['coin_id', 'is_moon']

    # Check individual results
    assert target_variables_df[target_variables_df['coin_id'] == 'coin1']['is_moon'].values[0] == 0
    assert target_variables_df[target_variables_df['coin_id'] == 'coin2']['is_moon'].values[0] == 1
    assert target_variables_df[target_variables_df['coin_id'] == 'coin3']['is_moon'].values[0] == 0
    assert target_variables_df[target_variables_df['coin_id'] == 'coin4']['is_moon'].values[0] == 0
    assert target_variables_df[target_variables_df['coin_id'] == 'coin5']['is_moon'].values[0] == 1

    # Check minimum percentages
    total_coins = len(target_variables_df)
    assert (target_variables_df['is_moon'].sum() /
            total_coins >= modeling_config['target_variables']['moon_minimum_percent'])
