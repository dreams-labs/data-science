"""
tests used to audit the files in the data-science/src folder
"""
# pylint: disable=C0301 # line over 100 chars
# pylint: disable=C0413 # import not at top of doc (due to local import)
# pylint: disable=C0303 trailing whitespace
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
import feature_engineering as fe # type: ignore[reportMissingImports]
from utils import load_config # type: ignore[reportMissingImports]

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
    result = fe.calculate_adj_pct_change(100, 150, 1000)
    assert result == 50, f"Expected 50, got {result}"
    
    # Test 2: Zero Start, Non-Zero End (Capped)
    result = fe.calculate_adj_pct_change(0, 100, 1000)
    assert result == 1000, f"Expected 1000 (capped), got {result}"
    
    # Test 3: Zero Start, Zero End (0/0 case)
    result = fe.calculate_adj_pct_change(0, 0, 1000)
    assert result == 0, f"Expected 0 for 0/0 case, got {result}"
    
    # Test 4: Negative Start to Positive End
    result = fe.calculate_adj_pct_change(-50, 100, 1000)
    assert result == -300, f"Expected -300, got {result}"
    
    # Test 5: Start Greater than End (Decrease)
    result = fe.calculate_adj_pct_change(200, 100, 1000)
    assert result == -50, f"Expected -50, got {result}"
    
    # Test 6: Small Positive Change (Cap not breached)
    result = fe.calculate_adj_pct_change(0, 6, 1000, 1)
    assert result == 500, f"Expected 500, got {result}"
    
    # Test 7: Large Increase (Capped at 1000%)
    result = fe.calculate_adj_pct_change(10, 800, 1000)
    assert result == 1000, f"Expected 1000 (capped), got {result}"


# ------------------------------------------ #
# calculate_adj_pct_change() unit tests
# ------------------------------------------ #

@pytest.mark.unit
def test_fe_calculate_global_stats():
    """
    Unit tests for the fe.calculate_global_stats() function.

    Test Cases:
    1. **Basic case**: Tests that 'sum' and 'mean' statistics are calculated correctly for a simple time series.
    2. **Multiple metrics**: Verifies that different metrics with different configurations of stats (e.g., 'sum', 'mean', 'median', 'std') are handled properly.
    3. **Empty time series**: Ensures that an empty time series is handled correctly, returning NaN for mean and 0 for sum.
    4. **Edge case (single value)**: Tests the function's behavior when the time series contains only a single value.
    5. **No stats in config**: Ensures that when no stats are defined for a metric, the function returns an empty dictionary without errors.
    """

    # Sample configurations and time series for testing
    basic_config = {
        'metrics': {
            'buyers_new': ['sum', 'mean']
        }
    }
    
    multiple_metrics_config = {
        'metrics': {
            'buyers_new': ['sum', 'mean'],
            'sellers_new': ['median', 'std']
        }
    }
    
    empty_ts = pd.Series([])
    single_value_ts = pd.Series([10])
    sample_ts = pd.Series([1, 2, 3, 4, 5])

    # Test Case 1: Basic case with simple sum and mean
    basic_stats = fe.calculate_global_stats(sample_ts, 'buyers_new', basic_config)
    assert basic_stats['buyers_new_sum'] == 15
    assert basic_stats['buyers_new_mean'] == 3
    
    # Test Case 2: Multiple metrics with different stats
    sample_coin_df = pd.DataFrame({
        'buyers_new': [1, 2, 3, 4, 5],
        'sellers_new': [5, 4, 3, 2, 1]
    })
    
    multiple_stats = fe.calculate_global_stats(sample_coin_df['buyers_new'], 'buyers_new', multiple_metrics_config)
    assert multiple_stats['buyers_new_sum'] == 15
    assert multiple_stats['buyers_new_mean'] == 3
    
    multiple_stats = fe.calculate_global_stats(sample_coin_df['sellers_new'], 'sellers_new', multiple_metrics_config)
    assert multiple_stats['sellers_new_median'] == 3
    assert round(multiple_stats['sellers_new_std'], 5) == round(sample_coin_df['sellers_new'].std(), 5)
    
    # Test Case 3: Empty time series should return NaN or 0 for certain stats
    empty_stats = fe.calculate_global_stats(empty_ts, 'buyers_new', basic_config)
    assert pd.isna(empty_stats['buyers_new_mean'])
    assert empty_stats['buyers_new_sum'] == 0
    
    # Test Case 4: Single value time series
    single_value_stats = fe.calculate_global_stats(single_value_ts, 'buyers_new', basic_config)
    assert single_value_stats['buyers_new_sum'] == 10
    assert single_value_stats['buyers_new_mean'] == 10
    
    # Test Case 5: No stats defined for the given metric
    no_stats_config = {
        'metrics': {
            'buyers_new': []
        }
    }
    
    no_stats = fe.calculate_global_stats(sample_ts, 'buyers_new', no_stats_config)
    assert no_stats == {}  # Should return an empty dictionary since no stats are defined



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
    rolling_features = fe.calculate_rolling_window_features(ts, window_duration, lookback_periods, rolling_stats, comparisons, metric_name)

    assert rolling_features['buyers_new_sum_3d_period_1'] == 27  # Last 3 records: 9+10+8 = 27
    assert rolling_features['buyers_new_max_3d_period_1'] == 10
    assert rolling_features['buyers_new_min_3d_period_1'] == 8
    assert rolling_features['buyers_new_median_3d_period_1'] == 9
    assert round(rolling_features['buyers_new_std_3d_period_1'], 5) == round(ts.iloc[-3:].std(), 5)

    assert 'buyers_new_change_3d_period_1' in rolling_features
    assert 'buyers_new_pct_change_3d_period_1' in rolling_features


    # Test Case 2: Non-divisible records (8 records, window_duration=3)
    rolling_features_partial_window = fe.calculate_rolling_window_features(ts_with_8_records, 3, 3, ['sum', 'max'], ['change', 'pct_change'], metric_name)

    # Only two full periods (6-8 and 3-5), so period 3 should not exist
    assert rolling_features_partial_window['buyers_new_sum_3d_period_1'] == 21  # Last 3 records: 6+7+8
    assert rolling_features_partial_window['buyers_new_sum_3d_period_2'] == 12  # Next 3 records: 3+4+5

    # Ensure no period 3 is calculated
    assert 'buyers_new_sum_3d_period_3' not in rolling_features_partial_window
    assert 'buyers_new_change_3d_period_3' not in rolling_features_partial_window


    # Test Case 3: Small dataset (2 records)
    rolling_features_small_ts = fe.calculate_rolling_window_features(small_ts, window_duration, lookback_periods, rolling_stats, comparisons, metric_name)

    # No valid 3-period windows exist, so the function should handle it gracefully
    assert rolling_features_small_ts == {}  # Expect empty dict since window is larger than available data


    # Test Case 4: Check std and median specifically with window of 3 and valid lookback periods
    rolling_features_std_median = fe.calculate_rolling_window_features(ts, window_duration, lookback_periods, ['std', 'median'], comparisons, metric_name)

    # Check for standard deviation and median over the last 3 periods
    assert round(rolling_features_std_median['buyers_new_std_3d_period_1'], 5) == round(ts.iloc[-3:].std(), 5)
    assert rolling_features_std_median['buyers_new_median_3d_period_1'] == 9
    assert round(rolling_features_std_median['buyers_new_std_3d_period_2'], 5) == round(ts.iloc[-6:-3].std(), 5)
    assert rolling_features_std_median['buyers_new_median_3d_period_2'] == 6


    # Test Case 5: Handle pct_change with impute_value logic (start_value=0)
    ts_with_zeros = pd.Series([0, 0, 5, 10, 15, 20])
    rolling_features_zeros = fe.calculate_rolling_window_features(ts_with_zeros, window_duration, lookback_periods, ['sum'], comparisons, metric_name)

    assert 'buyers_new_pct_change_3d_period_1' in rolling_features_zeros
    assert rolling_features_zeros['buyers_new_pct_change_3d_period_2'] <= 1000  # Ensure capping at 1000%


# ------------------------------------------ #
# flatten_coin_features() unit tests
# ------------------------------------------ #

@pytest.mark.unit
def test_fe_flatten_coin_features():
    # Sample DataFrame for testing
    sample_coin_df = pd.DataFrame({
        'coin_id': [1] * 6,
        'buyers_new': [10, 20, 30, 40, 50, 60],
        'sellers_new': [5, 10, 15, 20, 25, 30]
    })
    
    # Sample configuration for metrics
    metrics_config = {
        'metrics': {
            'buyers_new': {
                'aggregations': ['sum', 'mean', 'max', 'min', 'median', 'std'],
                'rolling': {
                    'stats': ['sum', 'max'],
                    'comparisons': ['change', 'pct_change'],
                    'window_duration': 3,
                    'lookback_periods': 2
                }
            },
            'sellers_new': {
                'aggregations': ['sum', 'mean', 'max']
            }
        }
    }

    # Test Case 1: Basic functionality with all metrics present
    flat_features = fe.flatten_coin_features(sample_coin_df, metrics_config)
    
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
    with pytest.raises(ValueError, match="Metric 'nonexistent_metric' is missing from the input DataFrame"):
        sample_coin_df_invalid = sample_coin_df.drop(columns=['buyers_new'])
        metrics_config_invalid = {'metrics': {'nonexistent_metric': {'aggregations': ['sum']}}}
        fe.flatten_coin_features(sample_coin_df_invalid, metrics_config_invalid)

    # Test Case 3: Missing 'coin_id' column in DataFrame
    with pytest.raises(ValueError, match="The input DataFrame is missing the required 'coin_id' column."):
        sample_coin_df_no_id = sample_coin_df.drop(columns=['coin_id'])
        fe.flatten_coin_features(sample_coin_df_no_id, metrics_config)

    # Test Case 4: Invalid aggregation function
    with pytest.raises(KeyError, match="Aggregation 'invalid_agg' for metric 'buyers_new' is not recognized"):
        metrics_config_invalid_agg = {
            'metrics': {
                'buyers_new': {
                    'aggregations': ['invalid_agg']
                }
            }
        }
        fe.flatten_coin_features(sample_coin_df, metrics_config_invalid_agg)

    # Test Case 5: Rolling window metrics
    rolling_features = fe.flatten_coin_features(sample_coin_df, metrics_config)
    
    assert 'buyers_new_sum_3d_period_1' in rolling_features  # Ensure rolling stats are calculated
    assert 'buyers_new_max_3d_period_1' in rolling_features
    assert 'buyers_new_sum_3d_period_2' in rolling_features
    assert 'buyers_new_max_3d_period_2' in rolling_features

    assert 'buyers_new_sum_3d_period_3' not in rolling_features  # Ensure no extra periods


# ------------------------------------------ #
# flatten_coin_features() unit tests
# ------------------------------------------ #

@pytest.mark.unit
def test_fe_flatten_coin_date_df():
    # Sample data for testing
    sample_df = pd.DataFrame({
        'coin_id': [1, 1, 1, 2, 2, 2],
        'buyers_new': [10, 20, 30, 40, 50, 60],
        'sellers_new': [5, 10, 15, 20, 25, 30]
    })

    # Sample configuration for metrics
    metrics_config = {
        'metrics': {
            'buyers_new': {
                'aggregations': ['sum', 'mean', 'max', 'min', 'median', 'std'],
            },
            'sellers_new': {
                'aggregations': ['sum', 'mean', 'max']
            }
        }
    }

    # Test Case 1: Basic functionality with multiple coins
    result = fe.flatten_coin_date_df(sample_df, metrics_config)

    # Check that there are two coins in the output
    assert len(result['coin_id'].unique()) == 2
    assert sorted(result['coin_id'].unique()) == [1, 2]

    # Check that all expected columns exist for both coins
    expected_columns = [
        'coin_id', 'buyers_new_sum', 'buyers_new_mean', 'buyers_new_max', 'buyers_new_min', 
        'buyers_new_median', 'buyers_new_std', 'sellers_new_sum', 'sellers_new_mean', 'sellers_new_max'
    ]
    assert all(col in result.columns for col in expected_columns)

    # Test Case 2: One coin with missing metric data (buyers_new should raise ValueError)
    df_missing_metric = pd.DataFrame({
        'coin_id': [1, 1, 1],
        'sellers_new': [5, 10, 15]
    })

    with pytest.raises(ValueError, match="Metric 'buyers_new' is missing from the input DataFrame."):
        fe.flatten_coin_date_df(df_missing_metric, metrics_config)

    # Test Case 3: Empty DataFrame (should raise ValueError)
    df_empty = pd.DataFrame(columns=['coin_id', 'buyers_new', 'sellers_new'])
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        fe.flatten_coin_date_df(df_empty, metrics_config)

    # Test Case 4: One coin in the dataset
    df_one_coin = pd.DataFrame({
        'coin_id': [1, 1, 1],
        'buyers_new': [10, 20, 30],
        'sellers_new': [5, 10, 15]
    })
    result_one_coin = fe.flatten_coin_date_df(df_one_coin, metrics_config)

    # Check that the single coin is processed correctly and the columns are as expected
    assert len(result_one_coin['coin_id'].unique()) == 1
    assert 'buyers_new_sum' in result_one_coin.columns
    assert result_one_coin['buyers_new_sum'].iloc[0] == 60  # Sum of buyers_new for coin 1


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
    config_path = os.path.join(os.path.dirname(__file__), 'test_config.yaml')
    return load_config(config_path)

@pytest.fixture(scope="session")
def cleaned_profits_df():
    """
    Fixture to load the cleaned_profits_df from the fixtures folder.
    """
    return pd.read_csv('tests/fixtures/cleaned_profits_df.csv')

@pytest.fixture(scope="session")
def shark_wallets_df():
    """
    Fixture to load the shark_wallets_df from the fixtures folder.
    """
    return pd.read_csv('tests/fixtures/shark_wallets_df.csv')

@pytest.fixture(scope="session")
def shark_coins_df():
    """
    Fixture to load the shark_coins_df from the fixtures folder.
    """
    return pd.read_csv('tests/fixtures/shark_coins_df.csv')


# ---------------------------------- #
# generate_buysell_metrics_df() integration tests
# ---------------------------------- #
