"""
tests used to audit the files in the data-science/src folder
"""
# pylint: disable=W1203 # fstrings in logs
# pylint: disable=C0302 # file over 1000 lines
# pylint: disable=E0401 # can't find import (due to local import)
# pylint: disable=C0413 # import not at top of doc (due to local import)
# pylint: disable=W0612 # unused variables (due to test reusing functions with 2 outputs)
# pylint: disable=W0621 # redefining from outer scope triggering on pytest fixtures

import sys
import os
import contextlib
import threading
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pytest
from unittest import mock
from dreams_core import core as dc

# pyright: reportMissingImports=false
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import training_data.data_retrieval as dr
import training_data.profits_row_imputation as pri
import coin_wallet_metrics.coin_wallet_metrics as cwm
from utils import load_config

load_dotenv()
logger = dc.setup_logger()


# ===================================================== #
#                                                       #
#                 U N I T   T E S T S                   #
#                                                       #
# ===================================================== #




# ---------------------------------------- #
# clean_profits_df() unit tests
# ---------------------------------------- #
@pytest.mark.unit
def test_all_market_cap_present(mocker):
    """
    Test that when all 'market_cap' values are present (coverage = 1),
    the 'market_cap_imputed' column matches the original 'market_cap' values
    and no warnings are logged.
    """
    # Mock the root logger's warning method to track if any warnings are emitted
    mock_logger = mocker.patch('logging.warning')

    # Define the input DataFrame with full market_cap coverage
    input_data = {
        'coin_id': ['BTC', 'BTC', 'ETH', 'ETH'],
        'date': ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02'],
        'price': [30000, 31000, 2000, 2100],
        'market_cap': [600000000, 620000000, 400000000, 420000000]
    }
    input_df = pd.DataFrame(input_data)

    # Define the expected DataFrame with 'market_cap_imputed' identical to 'market_cap'
    expected_market_cap_imputed = input_df['market_cap'].astype('Int64')
    expected_df = input_df.copy()
    expected_df['market_cap_imputed'] = expected_market_cap_imputed

    # Invoke the impute_market_cap function
    result_df = dr.impute_market_cap(input_df, min_coverage=0.7)

    # Assert that 'market_cap_imputed' matches the original 'market_cap' values
    assert np.allclose(
        result_df['market_cap_imputed'],
        expected_df['market_cap_imputed'],
        equal_nan=True
    ), "The 'market_cap_imputed' does not match the original 'market_cap' values."

    # Assert that the 'market_cap_imputed' column is of type Int64
    assert result_df['market_cap_imputed'].dtype == 'Int64', (
        "'market_cap_imputed' column is not of type Int64."
    )

    # Assert that no warnings were logged
    mock_logger.assert_not_called()


@pytest.mark.unit
def test_market_cap_below_min_coverage(mocker):
    """
    Test that when a coin's 'market_cap' coverage is below 'min_coverage',
    it is excluded from imputation, and 'market_cap_imputed' matches the original
    'market_cap' values without any warnings being logged.
    """
    # Mock the root logger's warning method to track if any warnings are emitted
    mock_logger = mocker.patch('logging.warning')

    # Define the input DataFrame with partial market_cap coverage below min_coverage
    input_data = {
        'coin_id': ['BTC', 'BTC', 'ETH', 'ETH'],
        'date': ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02'],
        'price': [30000, 31000, 2000, 2100],
        'market_cap': [np.nan, np.nan, 400000000, np.nan]
    }
    input_df = pd.DataFrame(input_data)

    # Define the expected 'market_cap_imputed' column
    # Since coverage < min_coverage, no imputation occurs
    # 'market_cap_imputed' should match 'market_cap' where present and remain NaN otherwise
    expected_market_cap_imputed = input_df['market_cap'].astype('Int64')
    expected_df = input_df.copy()
    expected_df['market_cap_imputed'] = expected_market_cap_imputed

    # Invoke the impute_market_cap function with min_coverage=0.7
    result_df = dr.impute_market_cap(input_df, min_coverage=0.7)

    # Assert that 'market_cap_imputed' matches the expected values
    assert np.allclose(
        result_df['market_cap_imputed'],
        expected_df['market_cap_imputed'],
        equal_nan=True
    ), "The 'market_cap_imputed' does not match the expected values when coverage is below min_coverage."

    # Assert that the 'market_cap_imputed' column is of type Int64
    assert result_df['market_cap_imputed'].dtype == 'Int64', (
        "'market_cap_imputed' column is not of type Int64."
    )

    # Assert that no warnings were logged since no imputation should occur
    mock_logger.assert_not_called()




# ---------------------------------------- #
# clean_profits_df() unit tests
# ---------------------------------------- #
@pytest.fixture
def sample_profits_df_for_cleaning():
    """
    Fixture to create a sample profits DataFrame with multiple coins per wallet.
    """
    return pd.DataFrame({
        'coin_id': ['BTC', 'ETH', 'BTC', 'ETH', 'LTC', 'BTC', 'ETH'],
        'wallet_address': ['wallet1', 'wallet1', 'wallet2', 'wallet2','wallet2',
                           'wallet3', 'wallet3'],
        'date': pd.date_range(start='2023-01-01', periods=7),
        'profits_cumulative': [5000, 3000, 1000, 500, 500, 100, 50],
        'usd_inflows_cumulative': [10000, 8000, 2000, 1500, 1500, 500, 250]
    })

@pytest.fixture
def sample_data_cleaning_config():
    """
    Fixture to create a sample data cleaning configuration.
    """
    return {
        'profitability_filter': 7500,
        'inflows_filter': 15000
    }

@pytest.mark.unit
def test_multiple_coins_per_wallet(sample_profits_df_for_cleaning, sample_data_cleaning_config):
    """
    Test scenario where wallets own multiple coins, some exceeding thresholds when aggregated.
    """
    # Call the function
    cleaned_df, exclusions_logs_df = dr.clean_profits_df(sample_profits_df_for_cleaning,
                                                         sample_data_cleaning_config)

    # Expected results
    expected_cleaned_df = sample_profits_df_for_cleaning[
        sample_profits_df_for_cleaning['wallet_address'].isin(['wallet2', 'wallet3'])
    ].reset_index(drop=True)
    expected_exclusions = pd.DataFrame({
        'wallet_address': ['wallet1'],
        'profits_exclusion': [True],
        'inflows_exclusion': [True]
    })

    # Assertions
    assert len(cleaned_df) == 5  # wallet2 (3 records) and wallet3 (2 records) should remain
    assert np.array_equal(cleaned_df.values, expected_cleaned_df.values)
    assert np.array_equal(exclusions_logs_df.values, expected_exclusions.values)

    # Check if profits and inflows are approximately correct for the remaining wallets
    # 1000 + 500 + 500 + 100 + 50
    assert pytest.approx(cleaned_df['profits_cumulative'].sum(), abs=1e-4) == 2150
    # 2000 + 1500 + 1500 + 500 + 250
    assert pytest.approx(cleaned_df['usd_inflows_cumulative'].sum(), abs=1e-4) == 5750

@pytest.fixture
def profits_at_threshold_df():
    """
    Fixture to create a sample profits DataFrame with profits at, above, and below the threshold.
    """
    return pd.DataFrame({
        'coin_id': ['BTC', 'ETH', 'LTC', 'XRP', 'DOGE'],
        'wallet_address': ['wallet1', 'wallet2', 'wallet3', 'wallet4', 'wallet5'],
        'date': pd.date_range(start='2023-01-01', periods=5),
        'profits_cumulative': [5000, 7500, 7501, 7499, 3000],
        'usd_inflows_cumulative': [10000, 12000, 13000, 11000, 8000]
    })

@pytest.fixture
def profits_at_threshold_config():
    """
    Fixture to create a data cleaning configuration with a specific profitability threshold.
    """
    return {
        'profitability_filter': 7500,
        'inflows_filter': 15000
    }

@pytest.mark.unit
def test_profits_exactly_at_threshold(profits_at_threshold_df, profits_at_threshold_config):
    """
    Test scenario where some wallets have profits exactly at the threshold value.
    """
    # Call the function
    cleaned_df, exclusions_logs_df = dr.clean_profits_df(profits_at_threshold_df,
                                                         profits_at_threshold_config)

    # Expected results
    expected_cleaned_df = profits_at_threshold_df[
        profits_at_threshold_df['wallet_address'].isin(['wallet1', 'wallet4', 'wallet5'])
    ].reset_index(drop=True)

    expected_exclusions = pd.DataFrame({
        'wallet_address': ['wallet2', 'wallet3'],
        'profits_exclusion': [True, True],
        'inflows_exclusion': [False, False]
    })

    # Assertions
    assert len(cleaned_df) == 3  # wallet1, wallet4, and wallet5 should remain
    assert np.array_equal(cleaned_df.values, expected_cleaned_df.values)
    assert np.array_equal(exclusions_logs_df.values, expected_exclusions.values)

    # Check if the correct wallets are present in the cleaned DataFrame
    assert set(cleaned_df['wallet_address']) == {'wallet1', 'wallet4', 'wallet5'}

    # Verify that wallets at or above the threshold are excluded
    assert 'wallet2' not in cleaned_df['wallet_address'].values
    assert 'wallet3' not in cleaned_df['wallet_address'].values

    # Check if profits and inflows are approximately correct for the remaining wallets
    # 5000 + 7499 + 3000
    assert pytest.approx(cleaned_df['profits_cumulative'].sum(), abs=1e-4) == 15499
     # 10000 + 11000 + 8000
    assert pytest.approx(cleaned_df['usd_inflows_cumulative'].sum(), abs=1e-4) == 29000

@pytest.fixture
def negative_profits_df():
    """
    Fixture to create a sample profits DataFrame with various levels of negative profits (losses).
    """
    return pd.DataFrame({
        'coin_id': ['BTC', 'ETH', 'LTC', 'XRP', 'DOGE', 'ADA'],
        'wallet_address': ['wallet1', 'wallet2', 'wallet3', 'wallet4', 'wallet5', 'wallet6'],
        'date': pd.date_range(start='2023-01-01', periods=6),
        'profits_cumulative': [-5000, -7500, -7501, -7499, 3000, 0],
        'usd_inflows_cumulative': [10000, 12000, 13000, 11000, 8000, 5000]
    })

@pytest.fixture
def negative_profits_config():
    """
    Fixture to create a data cleaning configuration with a specific profitability threshold.
    """
    return {
        'profitability_filter': 7500,
        'inflows_filter': 15000
    }

@pytest.mark.unit
def test_negative_profits_losses(negative_profits_df, negative_profits_config):
    """
    Test scenario where some wallets have significant negative profits (losses).
    """
    # Call the function
    cleaned_df, exclusions_logs_df = dr.clean_profits_df(negative_profits_df,
                                                         negative_profits_config)

    # Expected results
    expected_cleaned_df = negative_profits_df[
        negative_profits_df['wallet_address'].isin(['wallet1', 'wallet4', 'wallet5', 'wallet6'])
    ].reset_index(drop=True)

    expected_exclusions = pd.DataFrame({
        'wallet_address': ['wallet2', 'wallet3'],
        'profits_exclusion': [True, True],
        'inflows_exclusion': [False, False]
    })

    # Assertions
    assert len(cleaned_df) == 4  # wallet1, wallet4, wallet5, and wallet6 should remain
    assert np.array_equal(cleaned_df.values, expected_cleaned_df.values)
    assert np.array_equal(exclusions_logs_df.values, expected_exclusions.values)

    # Check if the correct wallets are present in the cleaned DataFrame
    assert set(cleaned_df['wallet_address']) == {'wallet1', 'wallet4', 'wallet5', 'wallet6'}

    # Verify that wallets with losses at or beyond the threshold are excluded
    assert 'wallet2' not in cleaned_df['wallet_address'].values
    assert 'wallet3' not in cleaned_df['wallet_address'].values

    # Check if profits and inflows are approximately correct for the remaining wallets
    # -5000 + -7499 + 3000 + 0
    assert pytest.approx(cleaned_df['profits_cumulative'].sum(), abs=1e-4) == -9499
    # 10000 + 11000 + 8000 + 5000
    assert pytest.approx(cleaned_df['usd_inflows_cumulative'].sum(), abs=1e-4) == 34000

    # Verify that wallets with losses are present in the cleaned DataFrame
    assert (cleaned_df['profits_cumulative'] < 0).any()

    # Verify that the wallet with zero profit is included
    assert (cleaned_df['profits_cumulative'] == 0).any()


# ======================================================== #
#                                                          #
#            I N T E G R A T I O N   T E S T S             #
#                                                          #
# ======================================================== #


# ---------------------------------- #
# set up config and module-level variables
# ---------------------------------- #

config = load_config('tests/test_config/test_config.yaml')

# Module-level variables
TRAINING_PERIOD_START = config['training_data']['training_period_start']
TRAINING_PERIOD_END = config['training_data']['training_period_end']
MODELING_PERIOD_START = config['training_data']['modeling_period_start']
MODELING_PERIOD_END = config['training_data']['modeling_period_end']


# ---------------------------------------- #
# retrieve_transfers_data() integration tests
# ---------------------------------------- #

@pytest.fixture(scope='session')
def profits_df_base():
    """
    retrieves transfers_df for data quality checks
    """
    logger.info("Beginning integration testing...")
    logger.info("Generating profits_df fixture from production data...")
    # retrieve profits data
    profits_df = dr.retrieve_profits_data(TRAINING_PERIOD_START,
                                          MODELING_PERIOD_END,
                                          config['data_cleaning']['minimum_wallet_inflows'])

    # filter data to only 5% of coin_ids
    np.random.seed(42)
    unique_coin_ids = profits_df['coin_id'].unique()
    sample_coin_ids = np.random.choice(unique_coin_ids, size=int(0.05 * len(unique_coin_ids)), replace=False)
    profits_df = profits_df[profits_df['coin_id'].isin(sample_coin_ids)]

    # perform the rest of the data cleaning steps
    profits_df, _ = cwm.split_dataframe_by_coverage(
        profits_df,
        TRAINING_PERIOD_START,
        MODELING_PERIOD_END,
        id_column='coin_id'
    )
    return profits_df

@pytest.fixture(scope='session')
def cleaned_profits_df(profits_df_base):
    """
    Fixture to run clean_profits_df() and return both the cleaned DataFrame and exclusions DataFrame.
    Uses thresholds from the config file.
    """
    logger.info("Generating cleaned_profits_df from clean_profits_df()...")
    cleaned_df, exclusions_df = dr.clean_profits_df(profits_df_base, config['data_cleaning'])
    return cleaned_df, exclusions_df


@contextlib.contextmanager
def single_threaded():
    """
    helper function to avoid multithreading which breaks in pytest
    """
    _original_thread_count = threading.active_count()
    yield
    current_thread_count = threading.active_count()
    if current_thread_count > _original_thread_count:
        raise AssertionError(f"Test created new threads: {current_thread_count - _original_thread_count}")


@pytest.fixture(scope='session')
def profits_df(prices_df,cleaned_profits_df):
    """
    Returns the final profits_df after the full processing sequence.
    """
    profits_df, _ = cleaned_profits_df

    # 2. Filtering based on dataset overlap
    # -------------------------------------
    # Filter market_data to only coins with transfers data if configured to
    if config['data_cleaning']['exclude_coins_without_transfers']:
        prices_df = prices_df[prices_df['coin_id'].isin(profits_df['coin_id'])]

    # Filter profits_df to remove records for any coins that were removed in data cleaning
    profits_df = profits_df[profits_df['coin_id'].isin(prices_df['coin_id'])]

    # impute period boundary dates
    dates_to_impute = [
        TRAINING_PERIOD_END,
        MODELING_PERIOD_START,
        MODELING_PERIOD_END
    ]
    # this must use only 1 thread to work in a testing environment
    with single_threaded():
        profits_df = pri.impute_profits_for_multiple_dates(profits_df, prices_df, dates_to_impute, n_threads=1)

    return profits_df

# Save profits_df.csv in fixtures/
# ----------------------------------------
def test_save_profits_df(profits_df):
    """
    This is not a test! This function saves a cleaned_profits_df.csv in the fixtures folder so it can be \
    used for integration tests in other modules.
    """

    # Save the cleaned DataFrame to the fixtures folder
    profits_df.to_csv('tests/fixtures/cleaned_profits_df.csv', index=False)

    # Add some basic assertions to ensure the data was saved correctly
    assert profits_df is not None
    assert len(profits_df) > 0



@pytest.mark.integration
class TestProfitsDataQuality:
    """
    Various tests on the data quality of the final profits_df version
    """
    def test_no_duplicate_records(self, profits_df):
        """Test 1: No duplicate records"""
        deduped_df = profits_df[['coin_id', 'wallet_address', 'date']].drop_duplicates()
        logger.info(f"Original profits_df length: {len(profits_df)}, Deduplicated: {len(deduped_df)}")
        assert (len(profits_df) == len(deduped_df)
                ), "There are duplicate rows based on coin_id, wallet_address, and date"

    def test_records_at_training_period_end(self, profits_df):
        """Test 2: All coin-wallet pairs have a record at the end of the training period"""
        profits_df_filtered = profits_df[profits_df['date'] < MODELING_PERIOD_START]
        pairs_in_training_period = profits_df_filtered[['coin_id', 'wallet_address']].drop_duplicates()
        period_end_df = profits_df[profits_df['date'] == TRAINING_PERIOD_END]

        logger.info(f"Found {len(pairs_in_training_period)} total pairs in training period with \
                    {len(period_end_df)} having data at period end.")
        assert (len(pairs_in_training_period) == len(period_end_df)
                ), "Not all coin-wallet pairs have a record at the end of the training period"

    def test_no_negative_usd_balances(self, profits_df):
        """Test 3: No negative USD balances"""
        negative_balances = profits_df[profits_df['usd_balance'] < -0.1]
        logger.info(f"Found {len(negative_balances)} records with negative USD balances.")
        assert len(negative_balances) == 0, "There are negative USD balances in the dataset"

    def test_date_range(self, profits_df):
        """Test 4: Date range check"""
        min_date = profits_df['date'].min()
        max_date = profits_df['date'].max()
        expected_max_date = pd.to_datetime(MODELING_PERIOD_END)
        logger.info(f"profits_df date range: {min_date} to {max_date}")
        assert (max_date == expected_max_date
                ), f"The last date in the dataset should be {expected_max_date}"

    def test_no_missing_values(self, profits_df):
        """Test 5: No missing values"""
        missing_values = profits_df.isna().sum()
        assert (missing_values.sum() == 0
                ), f"There are missing values in the dataset: {missing_values[missing_values > 0]}"

    def test_records_at_training_period_end_all_wallets(self, profits_df):
        """Test 7: Ensure all applicable wallets have records as of the training_period_end"""
        training_profits_df = profits_df[profits_df['date'] <= TRAINING_PERIOD_END]
        training_wallets_df = training_profits_df[['coin_id', 'wallet_address']].drop_duplicates()

        training_end_df = profits_df[profits_df['date'] == TRAINING_PERIOD_END]
        training_end_df = training_end_df[['coin_id', 'wallet_address']].drop_duplicates()

        assert (len(training_wallets_df) == len(training_end_df)
                ), "Some wallets are missing a record as of the training_period_end"

    def test_records_at_modeling_period_start(self, profits_df):
        """Test 8: Ensure all wallets have records as of the modeling_period_start"""
        modeling_profits_df = profits_df[profits_df['date'] <= MODELING_PERIOD_START]
        modeling_wallets_df = modeling_profits_df[['coin_id', 'wallet_address']].drop_duplicates()

        modeling_start_df = profits_df[profits_df['date'] == MODELING_PERIOD_START]
        modeling_start_df = modeling_start_df[['coin_id', 'wallet_address']].drop_duplicates()

        assert (len(modeling_wallets_df) == len(modeling_start_df)
                ), "Some wallets are missing a record as of the modeling_period_start"

    def test_records_at_modeling_period_end(self, profits_df):
        """Test 9: Ensure all wallets have records as of the modeling_period_end"""
        modeling_profits_df = profits_df[profits_df['date'] <= MODELING_PERIOD_END]
        modeling_wallets_df = modeling_profits_df[['coin_id', 'wallet_address']].drop_duplicates()

        modeling_end_df = profits_df[profits_df['date'] == MODELING_PERIOD_END]
        modeling_end_df = modeling_end_df[['coin_id', 'wallet_address']].drop_duplicates()

        assert (len(modeling_wallets_df) == len(modeling_end_df)
                ), "Some wallets are missing a record as of the modeling_period_end"

    def test_no_records_before_training_period_start(self, profits_df):
        """Test 10: Confirm no records exist prior to the training period start"""
        early_records = profits_df[profits_df['date'] < TRAINING_PERIOD_START]
        assert (len(early_records) == 0
                ), f"Found {len(early_records)} records prior to training_period_start"


# ---------------------------------------- #
# retrieve_market_data() integration tests
# ---------------------------------------- #

@pytest.fixture(scope='session')
def market_data_df():
    """
    Retrieve and preprocess the market_data_df, filling gaps as needed.
    """
    logger.info("Generating market_data_df from production data...")
    market_data_df = dr.retrieve_market_data()
    market_data_df = dr.clean_market_data(market_data_df,
                                          config,
                                          config['training_data']['earliest_cohort_lookback_start'],
                                          config['training_data']['modeling_period_end']
                                          )
    market_data_df, _ = cwm.split_dataframe_by_coverage(
        market_data_df,
        start_date=config['training_data']['training_period_start'],
        end_date=config['training_data']['modeling_period_end'],
        id_column='coin_id'
    )
    return market_data_df

@pytest.fixture(scope='session')
def prices_df(market_data_df):
    """
    Retrieve and preprocess the market_data_df, filling gaps as needed.
    """
    prices_df = market_data_df[['coin_id','date','price']].copy()
    return prices_df


# Save market_data_df.csv in fixtures/
# ----------------------------------------
def test_save_market_data_df(market_data_df, prices_df):
    """
    This is not a test! This function saves a market_data_df.csv in the fixtures folder so it
    can be used for integration tests in other modules.
    """
    # Save the prices DataFrame to the fixtures folder
    market_data_df.to_csv('tests/fixtures/market_data_df.csv', index=False)
    prices_df.to_csv('tests/fixtures/prices_df.csv', index=False)


    # Add some basic assertions to ensure the data was saved correctly
    assert market_data_df is not None
    assert len(market_data_df) > 0

    # Assert that there are no null values
    assert market_data_df.isna().sum().sum() == 0



# ---------------------------------------- #
# retrieve_metadata_data() integration tests
# ---------------------------------------- #

@pytest.fixture(scope='session')
def metadata_df():
    """
    Retrieve and preprocess the metadata_df.
    """
    logger.info("Generating metadata_df from production data...")
    metadata_df = dr.retrieve_metadata_data()
    return metadata_df

# Save metadata_df.csv in fixtures/
# ----------------------------------------
def test_save_metadata_df(metadata_df):
    """
    This is not a test! This function saves a metadata_df.csv in the fixtures folder so it
    can be used for integration tests in other modules.
    """
    # Save the metadata DataFrame to the fixtures folder
    metadata_df.to_csv('tests/fixtures/metadata_df.csv', index=False)

    # Add some basic assertions to ensure the data was saved correctly
    assert metadata_df is not None
    assert len(metadata_df) > 0

    # Assert that coin_id is unique
    assert metadata_df['coin_id'].is_unique, "coin_id is not unique in metadata_df"


# ---------------------------------------- #
# clean_profits_df() integration tests
# ---------------------------------------- #

@pytest.mark.integration
def test_clean_profits_exclusions(cleaned_profits_df, profits_df_base):
    """
    Test that all excluded wallets breach either the profitability or inflows threshold.
    Uses thresholds from the config file.
    """
    cleaned_df, exclusions_df = cleaned_profits_df

    # Check that every excluded wallet breached at least one threshold
    exclusions_with_breaches = exclusions_df.merge(profits_df_base, on='wallet_address', how='inner')

    # Calculate the total profits and inflows per wallet
    wallet_coin_agg_df = (exclusions_with_breaches.sort_values('date')
                                                  .groupby(['wallet_address','coin_id'], observed=True)
                                                  .agg({
                                                     'profits_cumulative': 'last',
                                                     'usd_inflows_cumulative': 'last'
                                                 }).reset_index())

    wallet_agg_df = (wallet_coin_agg_df.groupby('wallet_address')
                                       .agg({
                                           'profits_cumulative': 'sum',
                                           'usd_inflows_cumulative': 'sum'
                                       })
                                       .reset_index())

    # Apply threshold check from the config
    profitability_filter = config['data_cleaning']['profitability_filter']
    inflows_filter = config['data_cleaning']['inflows_filter']

    breaches_df = wallet_agg_df[
        (wallet_agg_df['profits_cumulative'] >= profitability_filter) |
        (wallet_agg_df['profits_cumulative'] <= -profitability_filter) |
        (wallet_agg_df['usd_inflows_cumulative'] >= inflows_filter)
    ]
    # Assert that all excluded wallets breached a threshold
    assert len(exclusions_df) == len(breaches_df), "Some excluded wallets do not breach a threshold."

@pytest.mark.integration
def test_clean_profits_remaining_count(cleaned_profits_df, profits_df_base):
    """
    Test that the count of remaining records in the cleaned DataFrame matches the expected count.
    """
    cleaned_df, exclusions_df = cleaned_profits_df

    # Get the total number of unique wallets before and after cleaning
    input_wallet_count = profits_df_base['wallet_address'].nunique()
    cleaned_wallet_count = cleaned_df['wallet_address'].nunique()
    excluded_wallet_count = exclusions_df['wallet_address'].nunique()

    # Assert that the remaining records equal the difference between the input and excluded records
    assert input_wallet_count == cleaned_wallet_count + excluded_wallet_count, \
        "The count of remaining wallets does not match the input minus excluded records."

@pytest.mark.integration
def test_clean_profits_aggregate_sums(cleaned_profits_df):
    """
    Test that the aggregation of profits and inflows for the remaining wallets stays within the
    configured thresholds. Uses thresholds from the config file.
    """
    cleaned_df, _ = cleaned_profits_df

    # Aggregate the profits and inflows for the remaining wallets
    remaining_wallet_coins_agg_df = (cleaned_df.sort_values('date')
                                            .groupby(['coin_id','wallet_address'], observed=True)
                                            .agg({
                                                'profits_cumulative': 'last',
                                                'usd_inflows_cumulative': 'last'
                                            })
                                            .reset_index())

    remaining_wallets_agg_df = (remaining_wallet_coins_agg_df.groupby('wallet_address')
                                            .agg({
                                                'profits_cumulative': 'sum',
                                                'usd_inflows_cumulative': 'sum'
                                            })
                                            .reset_index())


    # Apply the thresholds from the config
    profitability_filter = config['data_cleaning']['profitability_filter']
    inflows_filter = config['data_cleaning']['inflows_filter']
    # Ensure no remaining wallets exceed the thresholds
    over_threshold_wallets = remaining_wallets_agg_df[
        (remaining_wallets_agg_df['profits_cumulative'] >= profitability_filter) |
        (remaining_wallets_agg_df['profits_cumulative'] <= -profitability_filter) |
        (remaining_wallets_agg_df['usd_inflows_cumulative'] >= inflows_filter)
    ]

    # Assert that no wallets in the cleaned DataFrame breach the thresholds
    assert over_threshold_wallets.empty, "Some remaining wallets exceed the thresholds."
