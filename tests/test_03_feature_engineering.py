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
