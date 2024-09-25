"""
tests used to audit the files in the data-science/src folder
"""
# pylint: disable=C0301 # line over 100 chars
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
import feature_engineering as fe
from utils import load_config

load_dotenv()
logger = dc.setup_logger()






# ===================================================== #
#                                                       #
#                 U N I T   T E S T S                   #
#                                                       #
# ===================================================== #

# ------------------------------------------ #
# calculate_stat() unit tests
# ------------------------------------------ #

@pytest.mark.unit
def test_fe_calculate_stat():
    """
    basic tests for calculation, ensuring that the inputs map to the correct
    functions and that an invalid input raises an error.
    """
    # Sample data for testing
    sample_series = pd.Series([1, 2, 3, 4, 5])
    empty_series = pd.Series([])

    # Test sum
    assert fe.calculate_stat(sample_series, 'sum') == 15

    # Test mean
    assert fe.calculate_stat(sample_series, 'mean') == 3

    # Test median
    assert fe.calculate_stat(sample_series, 'median') == 3

    # Test std (rounded for comparison)
    assert round(fe.calculate_stat(sample_series, 'std'), 5) == round(sample_series.std(), 5)

    # Test max
    assert fe.calculate_stat(sample_series, 'max') == 5

    # Test min
    assert fe.calculate_stat(sample_series, 'min') == 1

    # Test invalid statistic
    with pytest.raises(KeyError):
        fe.calculate_stat(sample_series, 'invalid_stat')

    # Test empty series
    assert np.isnan(fe.calculate_stat(empty_series, 'mean'))
    assert fe.calculate_stat(empty_series, 'sum') == 0
    assert np.isnan(fe.calculate_stat(empty_series, 'std'))


# ------------------------------------------ #
# calculate_adj_pct_change() unit tests
# ------------------------------------------ #

@pytest.mark.unit
def test_calculate_adj_pct_change():
    """
    Unit tests for the calculate_adj_pct_change function.

    Tests:
    1. Standard case (non-zero values)
    2. Zero start, non-zero end (capped)
    3. Zero start, zero end (0/0 case)
    4. Negative start to positive end
    5. Start value greater than end value
    6. Small positive change (cap not breached)
    7. Large increase (capped)
    """

    # Test 1: Standard Case
    result = fe.calculate_adj_pct_change(100, 150, 10)
    assert result == .50, f"Expected 50, got {result}"

    # Test 2: Zero Start, Non-Zero End (Capped)
    result = fe.calculate_adj_pct_change(0, 100, 10)
    assert result == 10, f"Expected 1000 (capped), got {result}"

    # Test 3: Zero Start, Zero End (0/0 case)
    result = fe.calculate_adj_pct_change(0, 0, 10)
    assert result == 0, f"Expected 0 for 0/0 case, got {result}"

    # Test 4: Negative Start to Positive End
    result = fe.calculate_adj_pct_change(-50, 100, 10)
    assert result == -3, f"Expected -300, got {result}"

    # Test 5: Start Greater than End (Decrease)
    result = fe.calculate_adj_pct_change(200, 100, 10)
    assert result == -.5, f"Expected -50, got {result}"

    # Test 6: Small Positive Change (Cap not breached)
    result = fe.calculate_adj_pct_change(0, 6, 10, 1)
    assert result == 5, f"Expected 500, got {result}"

    # Test 7: Large Increase (Capped at 1000%)
    result = fe.calculate_adj_pct_change(10, 800, 10)
    assert result == 10, f"Expected 1000 (capped), got {result}"


# ------------------------------------------ #
# calculate_rolling_window_features() unit tests
# ------------------------------------------ #

@pytest.mark.unit
def test_fe_calculate_rolling_window_features():
    """
    Unit tests for fe.calculate_rolling_window_features().

    Test Cases:
    1. **Multiple periods with complete windows**:
    - Uses 10 records with a window duration of 3 and 3 lookback periods.
    - Verifies that all requested statistics ('sum', 'max', 'min', 'median', 'std') are calculated correctly for complete periods.
    - Also checks for valid 'change' and 'pct_change' calculations.

    2. **Non-divisible records**:
    - Uses 8 records with a window duration of 3.
    - Verifies that only complete periods are processed and earlier incomplete data is disregarded.
    - Ensures that 'period_3' is not calculated, and correct results for 'sum' and 'change' are returned for periods 1 and 2.

    3. **Small dataset**:
    - Uses only 2 records, which is smaller than the window size.
    - Ensures that the function handles small datasets gracefully and returns an empty dictionary.

    4. **Standard deviation and median checks**:
    - Specifically tests the calculation of 'std' and 'median' over the last 3 periods for valid rolling windows.
    - Verifies that these statistics are calculated accurately for both period 1 and period 2.

    5. **Percentage change with impute_value logic**:
    - Tests how the function handles a time series containing zeros.
    - Ensures that the 'pct_change' is calculated correctly, handling cases where the start value is 0 using the impute logic, and verifies that large percentage changes are capped at 1000%.
    """

    # Sample data for testing
    ts = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ts_with_8_records = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
    small_ts = pd.Series([1, 2])

    # Configuration for window and lookback periods
    window_duration = 3
    lookback_periods = 3
    rolling_stats = ['sum', 'max', 'min', 'median', 'std']
    comparisons = ['change', 'pct_change']
    metric_name = 'buyers_new'


    # Test Case 1: Multiple periods with complete windows (10 records, 3 periods, window_duration=3)
    rolling_features = fe.calculate_rolling_window_features(
        ts, window_duration, lookback_periods, rolling_stats, comparisons, metric_name)

    assert rolling_features['buyers_new_sum_3d_period_1'] == 27  # Last 3 records: 9+10+8 = 27
    assert rolling_features['buyers_new_max_3d_period_1'] == 10
    assert rolling_features['buyers_new_min_3d_period_1'] == 8
    assert rolling_features['buyers_new_median_3d_period_1'] == 9
    assert round(rolling_features['buyers_new_std_3d_period_1'], 5) == round(ts.iloc[-3:].std(), 5)

    assert 'buyers_new_change_3d_period_1' in rolling_features
    assert 'buyers_new_pct_change_3d_period_1' in rolling_features


    # Test Case 2: Non-divisible records (8 records, window_duration=3)
    rolling_features_partial_window = fe.calculate_rolling_window_features(
        ts_with_8_records, 3, 3, ['sum', 'max'], ['change', 'pct_change'], metric_name)

    # Only two full periods (6-8 and 3-5), so period 3 should not exist
    assert rolling_features_partial_window['buyers_new_sum_3d_period_1'] == 21  # Last 3 records: 6+7+8
    assert rolling_features_partial_window['buyers_new_sum_3d_period_2'] == 12  # Next 3 records: 3+4+5

    # Ensure no period 3 is calculated
    assert 'buyers_new_sum_3d_period_3' not in rolling_features_partial_window
    assert 'buyers_new_change_3d_period_3' not in rolling_features_partial_window


    # Test Case 3: Small dataset (2 records)
    rolling_features_small_ts = fe.calculate_rolling_window_features(
        small_ts, window_duration, lookback_periods, rolling_stats, comparisons, metric_name)

    # No valid 3-period windows exist, so the function should handle it gracefully
    assert not rolling_features_small_ts  # Expect empty dict since window is larger than available data


    # Test Case 4: Check std and median specifically with window of 3 and valid lookback periods
    rolling_features_std_median = fe.calculate_rolling_window_features(
        ts, window_duration, lookback_periods, ['std', 'median'], comparisons, metric_name)

    # Check for standard deviation and median over the last 3 periods
    assert round(rolling_features_std_median['buyers_new_std_3d_period_1'], 5) == round(ts.iloc[-3:].std(), 5)
    assert rolling_features_std_median['buyers_new_median_3d_period_1'] == 9
    assert round(rolling_features_std_median['buyers_new_std_3d_period_2'], 5) == round(ts.iloc[-6:-3].std(), 5)
    assert rolling_features_std_median['buyers_new_median_3d_period_2'] == 6


    # Test Case 5: Handle pct_change with impute_value logic (start_value=0)
    ts_with_zeros = pd.Series([0, 0, 5, 10, 15, 20])
    rolling_features_zeros = fe.calculate_rolling_window_features(
        ts_with_zeros, window_duration, lookback_periods, ['sum'], comparisons, metric_name)

    assert 'buyers_new_pct_change_3d_period_1' in rolling_features_zeros
    assert rolling_features_zeros['buyers_new_pct_change_3d_period_2'] <= 1000  # Ensure capping at 1000%


# ------------------------------------------ #
# flatten_date_features() unit tests
# ------------------------------------------ #

@pytest.mark.unit
def test_fe_flatten_date_features():
    """
    Unit test for the flatten_date_features function, which flattens metrics for a single coin.

    Test Cases:
    1. Basic functionality: Tests that the function correctly aggregates columns like 'buyers_new'
       and 'sellers_new' with specified aggregations (sum, mean, max, etc.) based on the sample data.
    2. Missing metric column: Ensures that the function raises a ValueError if a required metric
       is missing from the input DataFrame.
    3. Missing 'coin_id' column: Verifies that the function raises a ValueError if the input
       DataFrame does not contain a 'coin_id' column.
    4. Invalid aggregation function: Tests that the function raises a KeyError if an unrecognized
       aggregation function is specified in the configuration.
    5. Rolling window metrics: Tests the rolling window functionality, ensuring that the correct
       rolling stats (e.g., sum, max) and comparisons (change, pct_change) are calculated over
       specified windows.
    """
    # Sample DataFrame for testing
    sample_coin_df = pd.DataFrame({
        'coin_id': [1] * 6,
        'buyers_new': [10, 20, 30, 40, 50, 60],
        'sellers_new': [5, 10, 15, 20, 25, 30]
    })

    # Sample configuration for metrics
    metrics_config = {
        'buyers_new': {
            'aggregations': {
                'sum': {'scaling': 'none'},
                'mean': {'scaling': 'none'},
                'max': {'scaling': 'none'},
                'min': {'scaling': 'none'},
                'median': {'scaling': 'none'},
                'std': {'scaling': 'none'}
            },
            'rolling': {
                'stats': ['sum', 'max'],
                'comparisons': ['change', 'pct_change'],
                'window_duration': 3,
                'lookback_periods': 2
            }
        },
        'sellers_new': {
            'aggregations': {
                'sum': {'scaling': 'none'},
                'mean': {'scaling': 'none'},
                'max': {'scaling': 'none'}
            }
        }
    }

    # Test Case 1: Basic functionality with all metrics present
    flat_features = fe.flatten_date_features(sample_coin_df, metrics_config)

    assert flat_features['buyers_new_sum'] == 210  # Sum of buyers_new column
    assert flat_features['buyers_new_mean'] == 35   # Mean of buyers_new column
    assert flat_features['buyers_new_max'] == 60    # Max of buyers_new column
    assert flat_features['buyers_new_min'] == 10    # Min of buyers_new column
    assert flat_features['buyers_new_median'] == 35 # Median of buyers_new column
    assert round(flat_features['buyers_new_std'], 5) == round(sample_coin_df['buyers_new'].std(), 5)

    assert flat_features['sellers_new_sum'] == 105  # Sum of sellers_new column
    assert flat_features['sellers_new_mean'] == 17.5  # Mean of sellers_new column
    assert flat_features['sellers_new_max'] == 30  # Max of sellers_new column

    # Test Case 2: Missing metric column in DataFrame
    with pytest.raises(ValueError, match="No metrics matched the columns in the DataFrame"):
        sample_coin_df_invalid = sample_coin_df.drop(columns=['buyers_new'])
        metrics_config_invalid = {
            'nonexistent_metric': {
                'aggregations': {
                    'sum': {'scaling': 'none'}
                }
            }
        }
        fe.flatten_date_features(sample_coin_df_invalid, metrics_config_invalid)

    # Test Case 3: Invalid aggregation function
    with pytest.raises(KeyError, match="Unsupported aggregation type: 'invalid_agg'."):
        metrics_config_invalid_agg = {
            'buyers_new': {
                'aggregations': {
                    'invalid_agg': {'scaling': 'none'}
                }
            }
        }
        fe.flatten_date_features(sample_coin_df, metrics_config_invalid_agg)

    # Test Case 4: Rolling window metrics
    rolling_features = fe.flatten_date_features(sample_coin_df, metrics_config)

    assert 'buyers_new_sum_3d_period_1' in rolling_features  # Ensure rolling stats are calculated
    assert 'buyers_new_max_3d_period_1' in rolling_features
    assert 'buyers_new_sum_3d_period_2' in rolling_features
    assert 'buyers_new_max_3d_period_2' in rolling_features
    assert 'buyers_new_sum_3d_period_3' not in rolling_features  # Ensure no extra periods


# ------------------------------------------ #
# flatten_date_features() unit tests
# ------------------------------------------ #

@pytest.mark.unit
def test_fe_flatten_coin_date_df():
    """
    Unit test for the flatten_coin_date_df function, which flattens metrics over multiple coins and dates.

    Test Cases:
    1. Basic functionality with multiple coins: Tests that the function correctly aggregates metrics
       (e.g., 'buyers_new', 'sellers_new') for multiple coins across multiple dates, and that the
       output contains all expected columns.
    2. Missing metric data: Ensures that the function raises a ValueError when a required metric
       (e.g., 'buyers_new') is missing from the input DataFrame.
    3. Empty DataFrame: Verifies that the function raises a ValueError when an empty DataFrame
       is provided as input.
    4. One coin in the dataset: Tests that the function correctly processes a dataset containing
       only one coin and generates the expected columns and values.
    """
    # Sample data for testing
    sample_df = pd.DataFrame({
        'date': [
            pd.Timestamp('2024-01-01'),
            pd.Timestamp('2024-01-02'),
            pd.Timestamp('2024-01-03'),
            pd.Timestamp('2024-01-01'),
            pd.Timestamp('2024-01-02'),
            pd.Timestamp('2024-01-03')
            ],
        'coin_id': [1, 1, 1, 2, 2, 2],
        'buyers_new': [10, 20, 30, 40, 50, 60],
        'sellers_new': [5, 10, 15, 20, 25, 30]
    })

    # Sample configuration for metrics
    df_metrics_config = {
        'buyers_new': {
            'aggregations': {
                'sum': {'scaling': 'none'},
                'mean': {'scaling': 'none'},
                'max': {'scaling': 'none'},
                'min': {'scaling': 'none'},
                'median': {'scaling': 'none'},
                'std': {'scaling': 'none'}
            }
        },
        'sellers_new': {
            'aggregations': {
                'sum': {'scaling': 'none'},
                'mean': {'scaling': 'none'},
                'max': {'scaling': 'none'}
            }
        }
    }

    # demo
    training_period_end = '2024-01-03'

    # Test Case 1: Basic functionality with multiple coins
    result = fe.flatten_coin_date_df(sample_df, df_metrics_config, training_period_end)

    # Check that there are two coins in the output
    assert len(result['coin_id'].unique()) == 2
    assert sorted(result['coin_id'].unique()) == [1, 2]

    # Check that all expected columns exist for both coins
    expected_columns = [
        'coin_id', 'buyers_new_sum', 'buyers_new_mean', 'buyers_new_max', 'buyers_new_min',
        'buyers_new_median', 'buyers_new_std', 'sellers_new_sum', 'sellers_new_mean', 'sellers_new_max'
    ]
    assert all(col in result.columns for col in expected_columns)

    # Test Case 2: Empty DataFrame (should raise ValueError)
    df_empty = pd.DataFrame(columns=['coin_id', 'buyers_new', 'sellers_new'])
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        fe.flatten_coin_date_df(df_empty, df_metrics_config, training_period_end)

    # Test Case 3: One coin in the dataset
    df_one_coin = pd.DataFrame({
        'date': [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-03')],
        'coin_id': [1, 1, 1],
        'buyers_new': [10, 20, 30],
        'sellers_new': [5, 10, 15]
    })
    result_one_coin = fe.flatten_coin_date_df(df_one_coin, df_metrics_config, training_period_end)

    # Check that the single coin is processed correctly and the columns are as expected
    assert len(result_one_coin['coin_id'].unique()) == 1
    assert 'buyers_new_sum' in result_one_coin.columns
    assert result_one_coin['buyers_new_sum'].iloc[0] == 60  # Sum of buyers_new for coin 1


# ------------------------------------------ #
# save_flattened_outputs() unit tests
# ------------------------------------------ #


@pytest.fixture
def mock_coin_df():
    """
    Simple mock flat file keyed on coin_id
    """
    data = {'coin_id': ['BTC', 'ETH'], 'buyers_new': [50000, 4000]}
    return pd.DataFrame(data)

@pytest.fixture
def mock_non_unique_coin_id_df():
    """
    Mock DataFrame with non-unique 'coin_id' values (keyed on coin_id-date)
    """
    data = {
        'coin_id': ['BTC', 'BTC', 'ETH', 'ETH'],
        'date': ['2024-01-01', '2024-01-02', '2024-01-01', '2024-01-02'],
        'buyers_new': [50000, 51000, 4000, 4200]
    }
    return pd.DataFrame(data)

@pytest.mark.unit
def test_save_flattened_outputs(mock_coin_df):
    """
    Confirms that the mock file saves correctly
    """
    # Hard-code test output location directly within the test function
    test_output_path = os.path.join(os.getcwd(), "tests", "test_modeling", "outputs", "flattened_outputs")
    metric_description = 'buysell'
    modeling_period_start = '2024-04-01'

    # Call the function to save the CSV and get the DataFrame and output path
    _, saved_file_path = fe.save_flattened_outputs(mock_coin_df, test_output_path, metric_description, modeling_period_start)

    # Assert that the file was created
    assert os.path.exists(saved_file_path), f"File was not saved at {saved_file_path}"

    # Cleanup (remove the test file after the test)
    os.remove(saved_file_path)

@pytest.mark.unit
def test_save_flattened_outputs_non_unique_coin_id(mock_non_unique_coin_id_df):
    """
    Confirms that the function raises a ValueError if 'coin_id' values are not unique
    """
    test_output_path = os.path.join(os.getcwd(), "tests", "test_modeling", "outputs", "flattened_outputs")
    metric_description = 'buysell'
    modeling_period_start = '2024-04-01'

    # Check for the ValueError due to non-unique 'coin_id' values
    with pytest.raises(ValueError, match="The 'coin_id' column must have fully unique values."):
        fe.save_flattened_outputs(mock_non_unique_coin_id_df, test_output_path, metric_description, modeling_period_start)



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
    output_df, output_path = fe.preprocess_coin_df(input_path, mock_modeling_config, mock_metrics_config)

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
    output_df, output_path = fe.preprocess_coin_df(
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
    merged_df, _ = fe.create_training_data_df(tmpdir, input_files)

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
        fe.create_training_data_df(tmpdir, filenames)


@pytest.mark.unit
def test_missing_coin_id(mock_input_files):
    """
    Confirms the error message when an input file does not have a coin_id column.
    """
    tmpdir, filenames = mock_input_files

    # Create a DataFrame missing the 'coin_id' column
    df_missing_coin_id = pd.DataFrame({'buyers_new': [100, 200]})
    preprocessed_output_dir = os.path.join(tmpdir, 'outputs', 'preprocessed_outputs')
    df_missing_coin_id.to_csv(os.path.join(preprocessed_output_dir, 'file_missing_coin_id_2024-09-13_14-47.csv'), index=False)

    filenames.append(('file_missing_coin_id_2024-09-13_14-47.csv', 'fill_zeros'))

    with pytest.raises(ValueError, match="coin_id column is missing in file_missing_coin_id_2024-09-13_14-47.csv"):
        fe.create_training_data_df(tmpdir, filenames)


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
        fe.create_training_data_df(tmpdir, filenames)


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
    training_data_df, merge_logs_df = fe.merge_and_fill_training_data(input_dfs)

    # Assert that the merged DataFrame matches the expected DataFrame
    expected_df = pd.DataFrame({
        'coin_id': [1, 2, 3],
        'metric_1': [100, 200, 300],
        'metric_2': [400, 500, 600]
    })
    pd.testing.assert_frame_equal(training_data_df, expected_df)

    # Assert that the logs match the expected logs
    expected_logs = pd.DataFrame({
        'file': ['file1', 'file2'],
        'original_count': [3, 3],
        'filled_count': [0, 0],
    })
    pd.testing.assert_frame_equal(merge_logs_df, expected_logs)

    # drop_records happy path
    # ---------------------
    # Rerun the same function with drop_records and confirm that the output is identical
    input_dfs = [
        (df1, 'drop_records', 'file1'),
        (df2, 'drop_records', 'file2')
    ]

    # Run the function
    training_data_df, merge_logs_df = fe.merge_and_fill_training_data(input_dfs)

    # Assert that the merged DataFrame matches the expected DataFrame
    pd.testing.assert_frame_equal(training_data_df, expected_df)

    # Assert that the logs match the expected logs
    pd.testing.assert_frame_equal(merge_logs_df, expected_logs)



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
    merged_df, merge_logs_df = fe.merge_and_fill_training_data([
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
    merged_df, logs_df = fe.merge_and_fill_training_data([
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
# create_target_variables_mooncrater() unit tests
# ------------------------------------------ #

@pytest.mark.unit
def test_create_target_variables_mooncrater():
    """
    tests whether ths is_moon and is_crater target variables are calculated correctly.
    """
    # Mock data
    data = {
        'coin_id': ['coin1', 'coin2', 'coin3', 'coin4', 'coin5', 'coin1', 'coin2', 'coin3', 'coin4', 'coin5'],
        'date': [
            '2024-01-01', '2024-01-01', '2024-01-01', '2024-01-01', '2024-01-01',
            '2024-12-31', '2024-12-31', '2024-12-31', '2024-12-31', '2024-12-31'
        ],
        'price': [
            100, 100, 100, 100, 100,   # Start prices
            105, 150, 95, 50, 150      # End prices: slightly positive, >moon, slightly negative, <crater, at moon threshold
        ]
    }
    prices_df = pd.DataFrame(data)
    prices_df['date'] = pd.to_datetime(prices_df['date'])

    # Mock configuration
    training_data_config = {
        'modeling_period_start': '2024-01-01',
        'modeling_period_end': '2024-12-31',
    }

    modeling_config = {
        'target_variables': {
            'moon_threshold': 0.5,  # 50% increase
            'moon_minimum_percent': 0.0,
            'crater_threshold': -0.5,  # 50% decrease
            'crater_minimum_percent': 0.0
        }
    }

    # Call the function being tested
    target_variables_df, outcomes_df = fe.create_target_variables_mooncrater(prices_df, training_data_config, modeling_config)

    # Assertions for target variables
    assert target_variables_df[target_variables_df['coin_id'] == 'coin1']['is_moon'].values[0] == 0
    assert target_variables_df[target_variables_df['coin_id'] == 'coin2']['is_moon'].values[0] == 1
    assert target_variables_df[target_variables_df['coin_id'] == 'coin3']['is_crater'].values[0] == 0
    assert target_variables_df[target_variables_df['coin_id'] == 'coin4']['is_crater'].values[0] == 1
    assert target_variables_df[target_variables_df['coin_id'] == 'coin5']['is_moon'].values[0] == 1  # Exactly at moon threshold

    # Assertions for outcomes
    assert outcomes_df[outcomes_df['coin_id'] == 'coin1']['outcome'].values[0] == 'target variable created'
    assert outcomes_df[outcomes_df['coin_id'] == 'coin2']['outcome'].values[0] == 'target variable created'
    assert outcomes_df[outcomes_df['coin_id'] == 'coin3']['outcome'].values[0] == 'target variable created'
    assert outcomes_df[outcomes_df['coin_id'] == 'coin4']['outcome'].values[0] == 'target variable created'
    assert outcomes_df[outcomes_df['coin_id'] == 'coin5']['outcome'].values[0] == 'target variable created'





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
def df_metrics_config():
    """
    Fixture to load the configuration from the YAML file.
    """
    metrics_config = load_config('tests/test_config/test_metrics_config.yaml')
    first_cohort_name, first_cohort_metrics = next(iter(metrics_config['wallet_cohorts'].items()))

    return first_cohort_metrics

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

@pytest.mark.integration
def test_metrics_config_alignment(buysell_metrics_df, df_metrics_config):
    """
    Test that at least one metric from the buysell_metrics_df is configured in the df_metrics_config.

    This test ensures that the buysell_metrics_df contains columns that are defined in the metrics
    configuration file. It checks whether there is any overlap between the metrics from the DataFrame
    and the metrics defined in the configuration, asserting that at least one match is found.
    """
    # Extract column names from the DataFrame
    df_columns = buysell_metrics_df.columns

    # Find the intersection of DataFrame columns and config metrics
    matching_metrics = [metric for metric in df_columns if metric in df_metrics_config]

    # Assert that at least one metric in the config applies to the buysell_metrics_df
    assert matching_metrics, "No matching metrics found between buysell_metrics_df and the metrics configuration"


@pytest.mark.integration
def test_aggregation_methods(buysell_metrics_df, df_metrics_config, config):
    """
    Test that the aggregation methods applied during flattening are correct.

    This test verifies that the aggregation of metrics (such as total_bought) is handled correctly
    by the flatten_coin_date_df function. It compares manually calculated sums for total_bought at
    the coin_id level with the corresponding values in the flattened DataFrame, ensuring that the
    sum matches the expected result.
    """
    # Flatten the buysell metrics DataFrame to the coin_id level
    flattened_buysell_metrics_df = fe.flatten_coin_date_df(buysell_metrics_df, df_metrics_config, config['training_data']['training_period_end'])

    # Example: Verify that total_bought_sum is aggregated correctly at the coin_id level
    # Manually group the original buysell_metrics_df by coin_id to compute expected sums
    expected_total_bought_sum = buysell_metrics_df.groupby('coin_id')['total_bought'].sum().reset_index(name='total_bought_sum')

    # Extract the result from the flattened DataFrame
    result_total_bought_sum = flattened_buysell_metrics_df[['coin_id', 'total_bought_sum']]

    # Assert that the sums match between the manually calculated and flattened DataFrame
    pd.testing.assert_frame_equal(expected_total_bought_sum, result_total_bought_sum, check_like=True)


@pytest.mark.integration
def test_outlier_handling(buysell_metrics_df, df_metrics_config, config):
    """
    Test that extreme values (outliers) are correctly handled by the flattening function.

    This test introduces an extreme value (outlier) into the buysell_metrics_df and passes it through
    the flatten_coin_date_df function. It asserts that the extreme value is properly included in the
    aggregated total_bought_sum column, ensuring that the function can handle large outliers without
    breaking or misrepresenting the data.
    """

    # Introduce an outlier in the buysell_metrics_df for total_bought
    outlier_df = buysell_metrics_df.copy()
    outlier_df.loc[0, 'total_bought'] = 1e12  # Extreme value

    # Flatten the modified DataFrame
    flattened_buysell_metrics_df = fe.flatten_coin_date_df(outlier_df, df_metrics_config, config['training_data']['training_period_end'])

    # Ensure the extreme value is handled and aggregated correctly
    assert flattened_buysell_metrics_df['total_bought_sum'].max() >= 1e12, "Outlier in total_bought not handled correctly"


@pytest.mark.integration
def test_all_coin_ids_present(buysell_metrics_df, df_metrics_config, config):
    """
    Test that all coin_ids from the original DataFrame are present in the flattened output.

    This test ensures that every coin_id from the buysell_metrics_df is retained in the flattened
    DataFrame after processing. It verifies that no coin_ids were lost during the flattening process
    by comparing the unique coin_ids in the input with those in the output.
    """

    # Flatten the buysell metrics DataFrame to the coin_id level
    flattened_buysell_metrics_df = fe.flatten_coin_date_df(buysell_metrics_df, df_metrics_config, config['training_data']['training_period_end'])

    # Get unique coin_ids from the original DataFrame
    expected_coin_ids = buysell_metrics_df['coin_id'].unique()

    # Get the coin_ids from the flattened DataFrame
    result_coin_ids = flattened_buysell_metrics_df['coin_id'].unique()

    # Assert that all expected coin_ids are present in the flattened result
    assert set(expected_coin_ids) == set(result_coin_ids), "Some coin_ids are missing from the flattened DataFrame"
