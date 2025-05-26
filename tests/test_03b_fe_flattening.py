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
import feature_engineering.flattening as flt
from utils import load_config

load_dotenv()
logger = dc.setup_logger()



# ===================================================== #
#                                                       #
#                 U N I T   T E S T S                   #
#                                                       #
# ===================================================== #

# ------------------------------------------ #
# calculate_aggregation() unit tests
# ------------------------------------------ #

@pytest.mark.unit
def test_fe_calculate_aggregation():
    """
    basic tests for calculation, ensuring that the inputs map to the correct
    functions and that an invalid input raises an error.
    """
    # Sample data for testing
    sample_series = pd.Series([1, 2, 3, 4, 5])
    empty_series = pd.Series([])

    # Test sum
    assert flt.calculate_aggregation(sample_series, 'sum') == 15

    # Test mean
    assert flt.calculate_aggregation(sample_series, 'mean') == 3

    # Test median
    assert flt.calculate_aggregation(sample_series, 'median') == 3

    # Test std (rounded for comparison)
    assert round(flt.calculate_aggregation(sample_series, 'std'), 5) == round(sample_series.std(), 5)

    # Test max
    assert flt.calculate_aggregation(sample_series, 'max') == 5

    # Test min
    assert flt.calculate_aggregation(sample_series, 'min') == 1

    # Test invalid statistic
    with pytest.raises(KeyError):
        flt.calculate_aggregation(sample_series, 'invalid_stat')

    # Test empty series
    assert np.isnan(flt.calculate_aggregation(empty_series, 'mean'))
    assert flt.calculate_aggregation(empty_series, 'sum') == 0
    assert np.isnan(flt.calculate_aggregation(empty_series, 'std'))


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
    result = flt.calculate_adj_pct_change(100, 150, 10)
    assert result == .50, f"Expected 50, got {result}"

    # Test 2: Zero Start, Non-Zero End (Capped)
    result = flt.calculate_adj_pct_change(0, 100, 10)
    assert result == 10, f"Expected 1000 (capped), got {result}"

    # Test 3: Zero Start, Zero End (0/0 case)
    result = flt.calculate_adj_pct_change(0, 0, 10)
    assert result == 0, f"Expected 0 for 0/0 case, got {result}"

    # Test 4: Negative Start to Positive End
    result = flt.calculate_adj_pct_change(-50, 100, 10)
    assert result == -3, f"Expected -300, got {result}"

    # Test 5: Start Greater than End (Decrease)
    result = flt.calculate_adj_pct_change(200, 100, 10)
    assert result == -.5, f"Expected -50, got {result}"

    # Test 6: Small Positive Change (Cap not breached)
    result = flt.calculate_adj_pct_change(0, 6, 10, 1)
    assert result == 5, f"Expected 500, got {result}"

    # Test 7: Large Increase (Capped at 1000%)
    result = flt.calculate_adj_pct_change(10, 800, 10)
    assert result == 10, f"Expected 1000 (capped), got {result}"


# ------------------------------------------ #
# calculate_rolling_window_features() unit tests
# ------------------------------------------ #

@pytest.mark.unit
def test_fe_calculate_rolling_window_features():
    """
    Unit tests for flt.calculate_rolling_window_features().

    Test Cases:
    1. **Multiple periods with complete windows**:
    - Uses 10 records with a window duration of 3 and 3 lookback periods.
    - Verifies that all requested statistics ('sum', 'max', 'min', 'median', 'std') are calculated
        correctly for complete periods.
    - Also checks for valid 'change' and 'pct_change' calculations.

    2. **Non-divisible records**:
    - Uses 8 records with a window duration of 3.
    - Verifies that only complete periods are processed and earlier incomplete data is disregarded.
    - Ensures that 'period_3' is not calculated, and correct results for 'sum' and 'change' are
        returned for periods 1 and 2.

    3. **Small dataset**:
    - Uses only 2 records, which is smaller than the window size.
    - Ensures that the function handles small datasets gracefully and returns an empty dictionary.

    4. **Standard deviation and median checks**:
    - Specifically tests the calculation of 'std' and 'median' over the last 3 periods for valid
        rolling windows.
    - Verifies that these statistics are calculated accurately for both period 1 and period 2.

    5. **Percentage change with impute_value logic**:
    - Tests how the function handles a time series containing zeros.
    - Ensures that the 'pct_change' is calculated correctly, handling cases where the start value is 0
        using the impute logic, and verifies that large percentage changes are capped at 1000%.
    """

    # Sample data for testing
    ts = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ts_with_8_records = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
    small_ts = pd.Series([1, 2])

    # Configuration for window and lookback periods
    window_duration = 3
    lookback_periods = 3
    rolling_aggregations = ['sum', 'max', 'min', 'median', 'std']
    comparisons = ['change', 'pct_change']
    metric_name = 'buyers_new'


    # Test Case 1: Multiple periods with complete windows (10 records, 3 periods, window_duration=3)
    rolling_features = flt.calculate_rolling_window_features(
        ts, window_duration, lookback_periods, rolling_aggregations, comparisons, metric_name)

    assert rolling_features['buyers_new_sum_3d_period_1'] == 27  # Last 3 records: 9+10+8 = 27
    assert rolling_features['buyers_new_max_3d_period_1'] == 10
    assert rolling_features['buyers_new_min_3d_period_1'] == 8
    assert rolling_features['buyers_new_median_3d_period_1'] == 9
    assert round(rolling_features['buyers_new_std_3d_period_1'], 5) == round(ts.iloc[-3:].std(), 5)

    assert 'buyers_new_change_3d_period_1' in rolling_features
    assert 'buyers_new_pct_change_3d_period_1' in rolling_features


    # Test Case 2: Non-divisible records (8 records, window_duration=3)
    rolling_features_partial_window = flt.calculate_rolling_window_features(
        ts_with_8_records, 3, 3, ['sum', 'max'], ['change', 'pct_change'], metric_name)

    # Only two full periods (6-8 and 3-5), so period 3 should not exist
    assert rolling_features_partial_window['buyers_new_sum_3d_period_1'] == 21  # Last 3 records: 6+7+8
    assert rolling_features_partial_window['buyers_new_sum_3d_period_2'] == 12  # Next 3 records: 3+4+5

    # Ensure no period 3 is calculated
    assert 'buyers_new_sum_3d_period_3' not in rolling_features_partial_window
    assert 'buyers_new_change_3d_period_3' not in rolling_features_partial_window


    # Test Case 3: Small dataset (2 records)
    rolling_features_small_ts = flt.calculate_rolling_window_features(
        small_ts, window_duration, lookback_periods, rolling_aggregations, comparisons, metric_name)

    # No valid 3-period windows exist, so the function should handle it gracefully
    assert not rolling_features_small_ts  # Expect empty dict since window is larger than available data


    # Test Case 4: Check std and median specifically with window of 3 and valid lookback periods
    rolling_features_std_median = flt.calculate_rolling_window_features(
        ts, window_duration, lookback_periods, ['std', 'median'], comparisons, metric_name)

    # Check for standard deviation and median over the last 3 periods
    assert round(rolling_features_std_median['buyers_new_std_3d_period_1'], 5) == round(ts.iloc[-3:].std(), 5)
    assert rolling_features_std_median['buyers_new_median_3d_period_1'] == 9
    assert round(rolling_features_std_median['buyers_new_std_3d_period_2'], 5) == round(ts.iloc[-6:-3].std(), 5)
    assert rolling_features_std_median['buyers_new_median_3d_period_2'] == 6


    # Test Case 5: Handle pct_change with impute_value logic (start_value=0)
    ts_with_zeros = pd.Series([0, 0, 5, 10, 15, 20])
    rolling_features_zeros = flt.calculate_rolling_window_features(
        ts_with_zeros, window_duration, lookback_periods, ['sum'], comparisons, metric_name)

    assert 'buyers_new_pct_change_3d_period_1' in rolling_features_zeros
    assert rolling_features_zeros['buyers_new_pct_change_3d_period_2'] <= 1000  # Ensure capping at 1000%




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
    test_output_path = os.path.join(os.getcwd(), "test_modeling", "outputs", "flattened_outputs")
    metric_description = 'buysell'
    modeling_period_start = '2024-04-01'

    # Call the function to save the CSV and get the DataFrame and output path
    _, saved_file_path = flt.save_flattened_outputs(mock_coin_df,
                                                   test_output_path,
                                                   metric_description,
                                                   modeling_period_start)

    # Assert that the file was created
    assert os.path.exists(saved_file_path), f"File was not saved at {saved_file_path}"

    # Cleanup (remove the test file after the test)
    os.remove(saved_file_path)

@pytest.mark.unit
def test_save_flattened_outputs_non_unique_coin_id(mock_non_unique_coin_id_df):
    """
    Confirms that the function raises a ValueError if 'coin_id' values are not unique
    """
    test_output_path = os.path.join(os.getcwd(), "test_modeling", "outputs", "flattened_outputs")
    metric_description = 'buysell'
    modeling_period_start = '2024-04-01'

    # Check for the ValueError due to non-unique 'coin_id' values
    with pytest.raises(ValueError, match="The 'coin_id' column must have fully unique values."):
        flt.save_flattened_outputs(mock_non_unique_coin_id_df,
                                  test_output_path,
                                  metric_description,
                                  modeling_period_start)


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
    flattened_buysell_metrics_df = flt.flatten_coin_date_df(buysell_metrics_df,
                                                           df_metrics_config,
                                                           config['training_data']['training_period_end'])

    # Example: Verify that total_bought_sum is aggregated correctly at the coin_id level
    # Manually group the original buysell_metrics_df by coin_id to compute expected sums
    expected_total_bought_sum = (buysell_metrics_df.groupby('coin_id')['total_bought']
                                                   .sum()
                                                   .reset_index(name='total_bought_sum'))

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
    flattened_buysell_metrics_df = flt.flatten_coin_date_df(outlier_df,
                                                           df_metrics_config,
                                                           config['training_data']['training_period_end'])

    # Ensure the extreme value is handled and aggregated correctly
    assert (flattened_buysell_metrics_df['total_bought_sum'].max() >= 1e12
            ),"Outlier in total_bought not handled correctly"


@pytest.mark.integration
def test_all_coin_ids_present(buysell_metrics_df, df_metrics_config, config):
    """
    Test that all coin_ids from the original DataFrame are present in the flattened output.

    This test ensures that every coin_id from the buysell_metrics_df is retained in the flattened
    DataFrame after processing. It verifies that no coin_ids were lost during the flattening process
    by comparing the unique coin_ids in the input with those in the output.
    """

    # Flatten the buysell metrics DataFrame to the coin_id level
    flattened_buysell_metrics_df = flt.flatten_coin_date_df(buysell_metrics_df,
                                                           df_metrics_config,
                                                           config['training_data']['training_period_end'])

    # Get unique coin_ids from the original DataFrame
    expected_coin_ids = buysell_metrics_df['coin_id'].unique()

    # Get the coin_ids from the flattened DataFrame
    result_coin_ids = flattened_buysell_metrics_df['coin_id'].unique()

    # Assert that all expected coin_ids are present in the flattened result
    assert set(expected_coin_ids) == set(result_coin_ids), "Some coin_ids are missing from the flattened DataFrame"
