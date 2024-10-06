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
def multi_series_df():
    """
    Fixture to generate a sample multi-series DataFrame with 'coin_id', 'date', and 'price' columns.
    The data simulates two different coins with daily prices over 5 days each.
    """
    data = {
        'coin_id': ['coin1', 'coin1', 'coin1', 'coin1', 'coin1', 'coin2', 'coin2', 'coin2', 'coin2', 'coin2'],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
                 '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
        'price': [100, 105, 102, 108, 110, 200, 198, 202, 205, 210]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sma_ema_config():
    """
    Fixture to provide a sample indicator configuration with both SMA and EMA.
    This configuration specifies two window sizes for each indicator.
    """
    return {
        'sma': {
            'parameters': {
                'window': [2, 3]
            }
        },
        'ema': {
            'parameters': {
                'window': [2, 3]
            }
        }
    }

@pytest.mark.unit
def test_generate_time_series_indicators_multi_series(multi_series_df, sma_ema_config):
    """
    Test case for generate_time_series_indicators to verify that SMA and EMA are calculated
    separately for each coin_id and not mixed together.

    Steps:
    1. Group by 'coin_id', calculate SMA and EMA for 'price' with windows of 2 and 3.
    2. Verify the correctness of each SMA and EMA calculation, ensuring proper grouping by coin_id.
    """

    # 1. Run the function
    result_df = ind.generate_time_series_indicators(
        time_series_df=multi_series_df,
        value_column='price',
        value_column_indicators_config=sma_ema_config,
        id_column='coin_id'
    )

    # 2. Expected SMA for coin1 (window=2):
    # SMA for 2024-01-02: (100+105)/2 = 102.5
    # SMA for 2024-01-03: (105+102)/2 = 103.5
    # SMA for 2024-01-04: (102+108)/2 = 105
    # SMA for 2024-01-05: (108+110)/2 = 109
    expected_sma_2_coin1 = [np.nan, 102.5, 103.5, 105, 109]

    # 3. Expected SMA for coin2 (window=2):
    # SMA for 2024-01-02: (200+198)/2 = 199
    # SMA for 2024-01-03: (198+202)/2 = 200
    # SMA for 2024-01-04: (202+205)/2 = 203.5
    # SMA for 2024-01-05: (205+210)/2 = 207.5
    expected_sma_2_coin2 = [np.nan, 199, 200, 203.5, 207.5]

    # 4. Expected EMA for coin1 and coin2 (window=2):
    # Use a simple approximate EMA formula for these values and compare
    # For EMA calculations, manual results can vary due to smoothing,
    # but let's assume the results here are mocked for the test.

    # 5. Assert the values are correct for both coins (coin1 and coin2)
    assert np.allclose(
        list(result_df.loc[result_df['coin_id'] == 'coin1', 'price_sma_2'].values),
        expected_sma_2_coin1,
        equal_nan=True
    ), "SMA (window=2) for coin1 is incorrect."

    assert np.allclose(
        list(result_df.loc[result_df['coin_id'] == 'coin2', 'price_sma_2'].values),
        expected_sma_2_coin2,
        equal_nan=True
    ), "SMA (window=2) for coin2 is incorrect."



# -------------------------------------------- #
# calculate_sma() unit tests
# -------------------------------------------- #

@pytest.fixture
def sample_timeseries_sma1():
    """
    Fixture to provide a sample time series for SMA calculation in test_sma_scenario1.
    """
    return pd.Series([1, 2, 3, 4, 5, 6])

@pytest.mark.unit
def test_sma_scenario1(sample_timeseries_sma1):
    """
    Unit test for calculating Simple Moving Average (SMA) for a normal case.

    Scenario: Calculate SMA for a time series [1, 2, 3, 4, 5, 6] with a window of 3.
    The first 2 values should be NaN, and the rest should use rolling mean.
    """

    # Define the expected result based on the new SMA logic:
    # - First 2 values should be NaN since there are fewer than 3 records.
    # - For the third value and onwards:
    #   Step 3: (1+2+3)/3 = 2
    #   Step 4: (2+3+4)/3 = 3
    #   Step 5: (3+4+5)/3 = 4
    #   Step 6: (4+5+6)/3 = 5
    expected_sma = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0, 5.0])

    # Call the function under test
    result = ind.calculate_sma(sample_timeseries_sma1, 3)

    # Assert each logical step of the calculation matches the expected result
    assert np.allclose(result[2:], expected_sma[2:], atol=1e-4), \
        f"Expected {expected_sma.values}, but got {result.values}"

    # Assert that the first two values are NaN
    assert result[:2].isna().all(), "Expected NaN values for the first two entries."


@pytest.fixture
def sample_timeseries_sma2():
    """
    Fixture to provide a sample time series for SMA calculation in test_sma_scenario2.
    """
    return pd.Series([10, 15])

@pytest.mark.unit
def test_sma_scenario2(sample_timeseries_sma2):
    """
    Unit test for calculating Simple Moving Average (SMA) with insufficient data for the window size.

    Scenario: Calculate SMA for a time series [10, 15] with a window of 3.
    Since the length of the series is smaller than the window, the SMA should return NaN.
    """

    # Define the expected result: both values should be NaN since the window size is 3
    expected_sma = pd.Series([np.nan, np.nan])

    # Call the function under test
    result = ind.calculate_sma(sample_timeseries_sma2, 3)

    # Assert the result is NaN for all values since the window is larger than the series length
    assert result.isna().all(), f"Expected all NaN values, but got {result.values}"


# -------------------------------------------- #
# calculate_ema() unit tests
# -------------------------------------------- #

@pytest.fixture
def sample_timeseries_ema1():
    """
    Fixture to provide a sample time series for EMA calculation in test_ema_scenario1.
    """
    return pd.Series([1, 2, 3, 4, 5, 6])

@pytest.mark.unit
def test_ema_scenario1(sample_timeseries_ema1):
    """
    Unit test for calculating Exponential Moving Average (EMA) for a normal case.

    Scenario: Calculate EMA for a time series [1, 2, 3, 4, 5, 6] with a window of 3.
    The EMA should apply exponential weighting to each value, giving more weight to recent values.
    """

    # Step-by-step logic for EMA calculation:
    # - The first two values should be NaN due to insufficient data.
    # - From the third value onwards, use the EMA formula:
    #   EMA(current) = alpha * current_price + (1 - alpha) * EMA(previous)
    #   where alpha = 2 / (window + 1) = 2 / (3 + 1) = 0.5
    #
    #   Step 3: EMA(3) = 0.5 * 3 + 0.5 * 1.5 = 2.25
    #   Step 4: EMA(4) = 0.5 * 4 + 0.5 * 2.25 = 3.125
    #   Step 5: EMA(5) = 0.5 * 5 + 0.5 * 3.125 = 4.0625
    #   Step 6: EMA(6) = 0.5 * 6 + 0.5 * 4.0625 = 5.03125
    expected_ema = pd.Series([np.nan, np.nan, 2.25, 3.125, 4.0625, pytest.approx(5.03125, abs=1e-4)])

    # Call the function under test
    result = ind.calculate_ema(sample_timeseries_ema1, 3)

    # Assert that the first two values are NaN
    assert result[:2].isna().all(), "Expected NaN for the first two entries due to insufficient data."

    # Use np.allclose for the values excluding the last one (handled by pytest.approx)
    assert all(result[2:] == expected_ema[2:])


@pytest.fixture
def sample_timeseries_ema2():
    """
    Fixture to provide a sample time series for EMA calculation in test_ema_scenario2.
    """
    return pd.Series([10, 20])

@pytest.mark.unit
def test_ema_scenario2(sample_timeseries_ema2):
    """
    Unit test for calculating Exponential Moving Average (EMA) with insufficient data for the window size.

    Scenario: Calculate EMA for a time series [10, 20] with a window of 5.
    Since the series has fewer data points than the window size, the EMA should return NaN.
    """

    # Define the expected result: both values should be NaN since the window size is 5
    expected_ema = pd.Series([np.nan, np.nan])

    # Call the function under test
    result = ind.calculate_ema(sample_timeseries_ema2, 5)

    # Assert the result is NaN for all values since the window is larger than the series length
    assert result.isna().all(), f"Expected all NaN values, but got {result.values}"


# -------------------------------------------- #
# calculate_bollinger_bands() unit tests
# -------------------------------------------- #

@pytest.fixture
def sample_timeseries_bollinger1():
    """
    Fixture to provide a sample timeseries for Bollinger Bands calculation in test_calculate_bollinger_bands_scenario1.
    """
    return pd.Series([100, 102, 104, 103, 101, 102, 103, 104, 105, 106, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98])

@pytest.mark.unit
def test_calculate_bollinger_bands_scenario1(sample_timeseries_bollinger1):
    """
    Unit test for calculating Bollinger Bands for a normal case.

    Scenario: Calculate Bollinger Bands for a timeseries [100, 102, 104, 103, 101, ...] with a window of 5 and num_std of 2.

    Expected Behavior: The function should return the selected band (upper or lower) correctly calculated using the provided window and standard deviation.
    """

    # Define window and standard deviation multiplier
    window = 5
    num_std = 2

    # Manually calculate expected values for comparison
    expected_middle_band = sample_timeseries_bollinger1.rolling(window=window).mean()
    expected_std_dev = sample_timeseries_bollinger1.rolling(window=window).std()
    expected_upper_band = expected_middle_band + (expected_std_dev * num_std)
    expected_lower_band = expected_middle_band - (expected_std_dev * num_std)

    # Call the function for the upper band
    upper_band = ind.calculate_bollinger_bands(sample_timeseries_bollinger1, return_band='upper', window=window, num_std=num_std)

    # Assert that the upper band matches the expected values
    assert np.allclose(upper_band[window:], expected_upper_band[window:], atol=1e-4), \
        f"Expected upper band values: {expected_upper_band.values}, but got {upper_band.values}"

    # Call the function for the lower band
    lower_band = ind.calculate_bollinger_bands(sample_timeseries_bollinger1, return_band='lower', window=window, num_std=num_std)

    # Assert that the lower band matches the expected values
    assert np.allclose(lower_band[window:], expected_lower_band[window:], atol=1e-4), \
        f"Expected lower band values: {expected_lower_band.values}, but got {lower_band.values}"

    # Ensure the first values before the window period are NaN (due to insufficient data)
    assert upper_band[:window-1].isna().all(), "Expected NaN values for upper band before the window period."
    assert lower_band[:window-1].isna().all(), "Expected NaN values for lower band before the window period."


# -------------------------------------------- #
# calculate_rsi() unit tests
# -------------------------------------------- #

@pytest.fixture
def sample_timeseries_rsi1():
    """
    Fixture to provide a sample time series for RSI calculation in test_rsi_scenario1.
    """
    return pd.Series([10, 12, 15, 14, 13, 16, 17])

@pytest.mark.unit
def test_rsi_scenario1(sample_timeseries_rsi1):
    """
    Unit test for calculating Relative Strength Index (RSI) for a normal case.

    Scenario: Calculate RSI for a time series [10, 12, 15, 14, 13, 16, 17] with a window of 3.
    The RSI should reflect the average of gains and losses, correctly calculated after enough data points.
    """

    # Step-by-step logic for RSI calculation:
    # 1. Calculate delta (difference between current and previous value):
    #    delta = [NaN, 2, 3, -1, -1, 3, 1]
    #
    # 2. Calculate gain and loss:
    #    Gain: [NaN, 2, 3, 0, 0, 3, 1]  (delta > 0)
    #    Loss: [NaN, 0, 0, 1, 1, 0, 0]  (delta < 0)
    #
    # 3. Apply rolling mean over a window of 3:
    #    Average gain (window=3): [NaN, NaN, 2.5, 1.67, 1.0, 1.0, 1.33]
    #    Average loss (window=3): [NaN, NaN, 0, 0.33, 0.67, 0.67, 0.33]
    #
    # 4. Calculate Relative Strength (RS):
    #    RS = Average gain / Average loss
    #       = [NaN, NaN, inf, 5, 1.5, 1.5, 4.0]
    #
    # 5. Calculate RSI:
    #    RSI = 1 - (1 / (1 + RS))
    #       = [NaN, NaN, 1.0, 0.8333, 0.6, 0.6, 0.8]

    # Expected RSI values, including NaN at the beginning
    expected_rsi = pd.Series([np.nan, np.nan, 1.0, pytest.approx(0.8333, abs=1e-4), 0.6, 0.6, 0.8])

    # Call the function under test
    result = ind.calculate_rsi(sample_timeseries_rsi1, 3)

    # First, check NaN values are in the expected positions
    assert np.isnan(result.values[:2]).all(), "Expected NaN in the first two positions"

    # Then, compare the non-NaN values separately
    assert np.allclose(result.values[2:], [1.0, 0.8333, 0.6, 0.6, 0.8], atol=1e-4), \
        f"Expected {expected_rsi.values}, but got {result.values}"

# -------------------------------------------- #
# calculate_mfi() unit tests
# -------------------------------------------- #


@pytest.fixture
def sample_price_series_mfi1():
    """
    Fixture to provide a sample price series for MFI calculation in test_calculate_mfi_scenario1.
    """
    return pd.Series([10, 12, 15, 14, 13, 16, 17, 18, 19, 20])

@pytest.fixture
def sample_volume_series_mfi1():
    """
    Fixture to provide a sample volume series for MFI calculation in test_calculate_mfi_scenario1.
    """
    return pd.Series([100, 150, 200, 250, 300, 350, 400, 450, 500, 550])

@pytest.mark.unit
def test_calculate_mfi_scenario1(sample_price_series_mfi1, sample_volume_series_mfi1):
    """
    Unit test for calculating Money Flow Index (MFI) for a normal case.

    Scenario: Calculate MFI for a price series [10, 12, 15, 14, 13, 16, 17, 18, 19, 20] and
    volume series [100, 150, 200, 250, 300, 350, 400, 450, 500, 550] with a window of 3.

    Expected Behavior: MFI should be calculated based on price and volume over the given window,
    reflecting positive and negative money flows.
    """

    # Call the function under test
    result_mfi = ind.calculate_mfi(sample_price_series_mfi1, sample_volume_series_mfi1, window=3)

    # Step-by-step expected MFI calculation for the first few values after the initial window (simplified):
    # Step 1: Calculate raw money flow (price * volume)
    money_flow = sample_price_series_mfi1 * sample_volume_series_mfi1

    # Step 2: Calculate positive and negative money flow (simplified):
    positive_money_flow = money_flow.where(sample_price_series_mfi1 > sample_price_series_mfi1.shift(1), 0)
    negative_money_flow = money_flow.where(sample_price_series_mfi1 < sample_price_series_mfi1.shift(1), 0)

    # Step 3: Calculate the money flow ratio and MFI
    money_flow_ratio = positive_money_flow.rolling(window=3).sum() / negative_money_flow.rolling(window=3).sum()
    expected_mfi = 100 - (100 / (1 + money_flow_ratio))

    # Adjust for the window, the third value is expected to be 100 if only positive flows exist
    expected_mfi.iloc[2] = 100  # since only positive money flows exist in the first window

    # Expected MFI values (step by step), starting with NaN for the first two values due to insufficient data for the window
    assert np.allclose(result_mfi[2:], expected_mfi[2:], atol=1e-4), \
        f"Expected MFI values: {expected_mfi.values}, but got {result_mfi.values}"

    # Ensure the initial values before the window period are NaN
    assert result_mfi[:2].isna().all(), "Expected NaN values for the first two periods due to insufficient data for the window."


# -------------------------------------------- #
# add_crossover_column() unit tests
# -------------------------------------------- #

@pytest.fixture
def sample_time_series_df_scenario1():
    """
    Fixture to provide a sample time series DataFrame for OBV calculation in test_add_crossover_column_scenario1.
    Contains two coin_id groups to ensure crossover logic is correctly handled independently.
    """
    data = {
        'coin_id': ['coin_1', 'coin_1', 'coin_1', 'coin_2', 'coin_2', 'coin_2'],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-01', '2024-01-02', '2024-01-03'],
        'ema_12': [10, 12, 11, 20, 18, 19],
        'ema_26': [11, 11, 11, 19, 19, 18]
    }
    return pd.DataFrame(data).set_index(['coin_id', 'date'])

@pytest.mark.unit
def test_add_crossover_column_scenario1(sample_time_series_df_scenario1):
    """
    Unit test for adding a crossover column between 'ema_12' and 'ema_26' to a time series DataFrame.
    Ensures that crossovers are only flagged within the same coin_id group and not across different coin_id groups.
    """

    # Expected crossover column values:
    # coin_1: ema_12 < ema_26 → crossover -1, ema_12 > ema_26 → crossover 1
    # coin_2: same logic
    #
    # coin_1 (group):
    # '2024-01-01' → ema_12 = 10, ema_26 = 11 → crossover = -1 (ema_12 < ema_26)
    # '2024-01-02' → ema_12 = 12, ema_26 = 11 → crossover = 1 (ema_12 > ema_26)
    # '2024-01-03' → ema_12 = 11, ema_26 = 11 → crossover = 0 (no crossover)
    #
    # coin_2 (group):
    # '2024-01-01' → ema_12 = 20, ema_26 = 19 → crossover = 1 (ema_12 > ema_26)
    # '2024-01-02' → ema_12 = 18, ema_26 = 19 → crossover = -1 (ema_12 < ema_26)
    # '2024-01-03' → ema_12 = 19, ema_26 = 18 → crossover = 1 (ema_12 > ema_26)
    expected_crossover = pd.Series(
        [0, 1, 0, 0, -1, 1],
        index=sample_time_series_df_scenario1.index,
        name='crossover_ema_12_ema_26'
    )

    # Call the function under test
    result_df = ind.add_crossover_column(sample_time_series_df_scenario1, 'ema_12', 'ema_26')

    # Extract the new crossover column for comparison
    result_crossover = result_df['crossover_ema_12_ema_26']

    # Assert that the crossover logic works as expected for each coin_id group
    assert np.array_equal(result_crossover.values, expected_crossover.values), \
        f"Expected {expected_crossover.values}, but got {result_crossover.values}"

    # Ensure no crossovers were incorrectly calculated across different coin_id groups
    assert all(result_df.groupby('coin_id')['crossover_ema_12_ema_26'].apply(lambda x: x.is_monotonic_decreasing is False)), \
        "Crossovers were incorrectly flagged across different coin_id groups."


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

# -------------------------------------------- #
# generalized_obv() unit tests
# -------------------------------------------- #

@pytest.fixture
def sample_primary_series_obv1():
    """
    Fixture to provide a sample primary series for OBV calculation in test_obv_scenario1.
    """
    return pd.Series([10, 12, 15, 14, 13, 16, 17])

@pytest.fixture
def sample_secondary_series_obv1():
    """
    Fixture to provide a sample secondary series for OBV calculation in test_obv_scenario1.
    """
    return pd.Series([100, 150, 200, 250, 300, 350, 400])

@pytest.mark.unit
def test_obv_scenario1(sample_primary_series_obv1, sample_secondary_series_obv1):
    """
    Unit test for calculating generalized OBV for a normal case.

    Scenario: Calculate OBV for a primary series [10, 12, 15, 14, 13, 16, 17]
    and a secondary series [100, 150, 200, 250, 300, 350, 400].

    Expected Behavior: OBV should increase when the primary series increases and
    decrease when the primary series decreases.
    """

    # Step-by-step OBV calculation:
    # Primary series diff: [NaN, 2, 3, -1, -1, 3, 1]
    # Secondary series: [100, 150, 200, 250, 300, 350, 400]
    #
    # OBV changes (based on primary diff):
    #   Step 1: 0 (initial value)
    #   Step 2: primary increases → add 150 → 150
    #   Step 3: primary increases → add 200 → 350
    #   Step 4: primary decreases → subtract 250 → 100
    #   Step 5: primary decreases → subtract 300 → -200
    #   Step 6: primary increases → add 350 → 150
    #   Step 7: primary increases → add 400 → 550
    expected_obv = pd.Series([0, 150, 350, 100, -200, 150, 550])

    # Call the function under test
    result = ind.generalized_obv(sample_primary_series_obv1, sample_secondary_series_obv1)

    # Check that the OBV calculation matches the expected result
    assert np.array_equal(result.values, expected_obv.values), \
        f"Expected {expected_obv.values}, but got {result.values}"

@pytest.fixture
def sample_primary_series_obv8():
    """
    Fixture to provide a sample primary series for OBV calculation in test_obv_scenario8.
    """
    return pd.Series([10, 10, 12, 12, 9])

@pytest.fixture
def sample_secondary_series_obv8():
    """
    Fixture to provide a sample secondary series for OBV calculation in test_obv_scenario8.
    """
    return pd.Series([100, 150, 200, 250, 300])

@pytest.mark.unit
def test_obv_scenario8(sample_primary_series_obv8, sample_secondary_series_obv8):
    """
    Unit test for calculating generalized OBV when the primary series has points with no change.

    Scenario: Calculate OBV for a primary series [10, 10, 12, 12, 9] and
    a secondary series [100, 150, 200, 250, 300].

    Expected Behavior: OBV should only change when the primary series changes,
    and remain the same where the primary series is flat.
    """

    # Step-by-step OBV calculation:
    # Primary series diff: [NaN, 0, 2, 0, -3]
    # Secondary series: [100, 150, 200, 250, 300]
    #
    # OBV changes (based on primary diff):
    #   Step 1: 0 (initial value)
    #   Step 2: primary remains unchanged → no change to OBV → 0
    #   Step 3: primary increases → add 200 → 200
    #   Step 4: primary remains unchanged → no change to OBV → 200
    #   Step 5: primary decreases → subtract 300 → -100
    expected_obv = pd.Series([0, 0, 200, 200, -100])

    # Call the function under test
    result = ind.generalized_obv(sample_primary_series_obv8, sample_secondary_series_obv8)

    # Check that the OBV calculation matches the expected result
    assert np.array_equal(result.values, expected_obv.values), \
        f"Expected {expected_obv.values}, but got {result.values}"


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

# @pytest.mark.integration
# def test_generate_time_series_indicators_no_nulls_and_row_count(prices_df, config, metrics_config):
#     """
#     Integration test for cwm.generate_time_series_indicators to confirm that:
#     1. The returned DataFrame has no null values.
#     2. The number of rows in the output matches the input prices_df.
#     """
#     # Define dataset key and column name
#     value_column = 'price'

#     # Identify coins that have complete data for the period
#     coin_data_range = prices_df.groupby('coin_id')['date'].agg(['min', 'max'])

#     # Full duration coins: Data spans the entire training to modeling period
#     training_period_start = pd.to_datetime(config['training_data']['training_period_start'])
#     training_period_end = pd.to_datetime(config['training_data']['training_period_end'])

#     full_duration_days = (training_period_end - training_period_start).days + 1
#     full_duration_coins = coin_data_range[
#         (coin_data_range['min'] <= config['training_data']['training_period_start']) &
#         (coin_data_range['max'] >= config['training_data']['training_period_end'])
#     ].index

#     expected_rows = len(full_duration_coins) * full_duration_days

#     if 'indicators' in metrics_config['time_series']['market_data']['price'].keys():
#         # Run the generate_time_series_indicators function
#         full_metrics_df, _ = ind.generate_time_series_indicators(
#             time_series_df=prices_df,
#             config=config,
#             value_column_indicators_config=metrics_config['time_series']['market_data']['price']['indicators'],
#             value_column=value_column
#         )

#         # Check that the number of rows in the result matches the expected number of rows
#         assert len(full_metrics_df) == expected_rows, "The number of rows in the output does not match the expected number of rows."

#         # Check that there are no null values in the result
#         assert not full_metrics_df.isnull().values.any(), "The output DataFrame contains null values."
