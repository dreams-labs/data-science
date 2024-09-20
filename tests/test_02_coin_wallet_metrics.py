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

import sys
import os
import pandas as pd
from dotenv import load_dotenv
import pytest
from dreams_core import core as dc

# pyright: reportMissingImports=false
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import coin_wallet_metrics as cwm
from utils import load_config

load_dotenv()
logger = dc.setup_logger()


# ===================================================== #
#                                                       #
#                 U N I T   T E S T S                   #
#                                                       #
# ===================================================== #

# ------------------------------------------ #
# test_generate_buysell_metrics_df() unit tests
# ------------------------------------------ #

# Updated mock data fixture for profits_df with wallet5 included
@pytest.fixture
def mock_profits_df():
    data = {
        'wallet_address': [
            'wallet1', 'wallet1', 'wallet1', 'wallet1', 'wallet1',  # wallet1 transactions (coin1)
            'wallet2', 'wallet2', 'wallet2', 'wallet2', 'wallet2',  # wallet2 transactions (coin2)
            'wallet3', 'wallet3', 'wallet3', 'wallet3', 'wallet3',  # wallet3 transactions (coin3)
            'wallet1', 'wallet1',  # wallet1 transactions (coin4 - outside cohort)
            'wallet4', 'wallet4',  # wallet4 transactions (coin1 - outside cohort)
            'wallet5', 'wallet5', 'wallet5'  # wallet5 transactions (coin1 and coin2)
        ],
        'coin_id': [
            'coin1', 'coin1', 'coin1', 'coin1', 'coin1',  # coin1 (wallet1)
            'coin2', 'coin2', 'coin2', 'coin2', 'coin2',  # coin2 (wallet2)
            'coin3', 'coin3', 'coin3', 'coin3', 'coin3',  # coin3 (wallet3)
            'coin4', 'coin4',  # coin4 (wallet1 - outside cohort)
            'coin1', 'coin1',  # wallet4 coin1 (outside cohort)
            'coin1', 'coin2', 'coin2'  # wallet5 purchases: coin1 (1/1/24), coin2 (1/2/24 and 1/3/24)
        ],
        'date': [
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',  # wallet1 coin1
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',  # wallet2 coin2
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',  # wallet3 coin3
            '2024-01-01', '2024-01-02',  # wallet1 coin4
            '2024-01-01', '2024-01-02',  # wallet4 coin1 (outside cohort)
            '2024-01-01', '2024-01-02', '2024-01-03'  # wallet5 transactions
        ],
        'balance': [
            100, 130, 230, 230, 220,  # wallet1 (coin1)
            200, 180, 230, 190, 190,  # wallet2 (coin2)
            50, 60, 40, 60, 60,  # wallet3 (coin3)
            400, 425,  # wallet1 (coin4 - outside cohort)
            600, 620,  # wallet4 coin1 (outside cohort)
            100, 200, 200  # wallet5 coin1 and coin2 purchases
        ],
        'net_transfers': [
            100, +30, +100, 0, -10,  # wallet1 (coin1)
            200, -20, +50, -40, 0,  # wallet2 (coin2)
            50, +10, -20, +20, 0,  # wallet3 (coin3)
            400, +25,  # wallet1 (coin4 - outside cohort)
            600, +20,  # wallet4 coin1 (outside cohort)
            100, 200, 200  # wallet5 transactions
        ]
    }

    return pd.DataFrame(data)

@pytest.mark.unit
def test_unit_generate_buysell_metrics_df(mock_profits_df):
    """
    tests the generation of buysell metrics for a wallet-coin cohort
    """
    cohort_wallets = ['wallet1', 'wallet2', 'wallet3', 'wallet5']  # Include wallet5
    training_period_end = '2024-01-05'  # Set a training period end date

    # Call the function
    result_df = cwm.generate_buysell_metrics_df(mock_profits_df, training_period_end, cohort_wallets)

    # Test the output structure
    expected_columns = [
        'date', 'buyers_new', 'buyers_repeat', 'total_buyers', 'sellers_new', 'sellers_repeat',
        'total_sellers', 'total_bought', 'total_sold', 'total_net_transfers', 'total_volume',
        'total_holders', 'total_balance', 'coin_id'
    ]

    for col in expected_columns:
        assert col in result_df.columns, f"Missing column: {col}"

    # Assertions for wallet5:
    # buyers_new for coin1 on 1/1/24 should be 2 (wallet1 and wallet5)
    assert result_df[(result_df['coin_id'] == 'coin1') & (result_df['date'] == '2024-01-01')]['buyers_new'].iloc[0] == 2

    # buyers_new for coin2 on 1/2/24 should be 1 (wallet5)
    assert result_df[(result_df['coin_id'] == 'coin2') & (result_df['date'] == '2024-01-02')]['buyers_new'].iloc[0] == 1

    # buyers_repeat for coin2 on 1/3/24 should be 2 (wallet2 and wallet5)
    assert result_df[(result_df['coin_id'] == 'coin2') & (result_df['date'] == '2024-01-03')]['buyers_repeat'].iloc[0] == 2

    # Filter mock_profits_df to only include cohort wallets
    cohort_profits_df = mock_profits_df[mock_profits_df['wallet_address'].isin(cohort_wallets)]

    # total_bought should match the sum of positive net_transfers in cohort_profits_df
    total_bought_mock = cohort_profits_df[cohort_profits_df['net_transfers'] > 0]['net_transfers'].sum()
    total_bought_result = result_df['total_bought'].sum()
    assert total_bought_mock == total_bought_result, f"Total bought does not match: {total_bought_mock} != {total_bought_result}"

    # total_sold should match the sum of absolute values of negative net_transfers in cohort_profits_df
    total_sold_mock = abs(cohort_profits_df[cohort_profits_df['net_transfers'] < 0]['net_transfers'].sum())
    total_sold_result = result_df['total_sold'].sum()
    assert total_sold_mock == total_sold_result, f"Total sold does not match: {total_sold_mock} != {total_sold_result}"

    # Assertions for total_balance of coin2 on all 5 days
    coin2_balances = mock_profits_df[mock_profits_df['coin_id'] == 'coin2'].groupby('date')['balance'].sum()
    for date, expected_balance in coin2_balances.items():
        result_balance = result_df[(result_df['coin_id'] == 'coin2') & (result_df['date'] == date)]['total_balance'].iloc[0]
        assert expected_balance == result_balance, f"Balance mismatch for coin2 on {date}: {expected_balance} != {result_balance}"


# ------------------------------------------ #
# fill_buysell_metrics_df() unit tests
# ------------------------------------------ #

@pytest.mark.unit
def test_fill_buysell_metrics_df():
    """
    Unit test for the fill_buysell_metrics_df function.

    This test checks the following:

    1. Missing dates within the input DataFrame's date range are correctly identified and filled.
    2. Missing rows beyond the latest date (up to the training_period_end) are added with NaN values initially, then filled appropriately.
    3. Forward-filling for 'total_balance' and 'total_holders' works as expected, ensuring that:
       - Values are forward-filled correctly for each coin_id.
       - Any missing dates earlier than the first record for 'total_balance' or 'total_holders' are filled with 0.
    4. 'buyers_new' and other transaction-related columns (e.g., 'total_bought', 'total_sold') are correctly filled with 0 for missing dates.

    Test Case Summary:
    - The test includes three coins: 'coin1', 'coin2', and 'coin3'.
    - 'coin1' has two records: one on 2024-01-01 and one on 2024-01-04, and is missing values for 2024-01-02, 2024-01-03, and 2024-01-05.
    - 'coin2' has two records: one on 2024-01-01 and one on 2024-01-03, and is missing values for 2024-01-02 and 2024-01-05.
    - 'coin3' has a single record on 2024-01-03 and is missing values for 2024-01-04 and 2024-01-05.

    Expected Assertions:
    - The 'total_balance' and 'buyers_new' columns are correctly forward-filled for each coin, ensuring no values leak across coin_ids.
    - For each coin_id, records with missing 'total_balance' or 'total_holders' before the first non-null date are filled with 0.
    - Transaction-related columns (e.g., 'buyers_new') are filled with 0 for any missing dates.
    """
    buysell_metrics_df = pd.DataFrame({
        'coin_id': ['coin1', 'coin1', 'coin2', 'coin2', 'coin3'],
        'date': [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-04'), pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-03'), pd.Timestamp('2024-01-03')],
        'total_balance': [100, 110, 200, None, 300],  # Added coin3 with balance 300 on 2024-01-03
        'total_bought': [50, 20, 75, None, 60],
        'total_sold': [10, 5, 15, None, 20],
        'total_net_transfers': [40, 15, 60, None, 40],
        'total_volume': [100, 35, 150, None, 80],
        'total_holders': [10, 11, 20, None, 30],  # Added coin3 with holders 30 on 2024-01-03
        'buyers_new': [1, 0, 2, None, 3],  # Added coin3 with 3 new buyers on 2024-01-03
        'buyers_repeat': [0, 1, 0, None, 1],
        'total_buyers': [1, 1, 2, None, 4],
        'sellers_new': [0, 1, 1, None, 2],
        'sellers_repeat': [1, 0, 1, None, 1],
        'total_sellers': [1, 1, 2, None, 3]
    })

    training_period_end = pd.Timestamp('2024-01-05')

    # Call the function to fill missing dates and values
    result = cwm.fill_buysell_metrics_df(buysell_metrics_df, training_period_end)

    # Assert total_balance for coin1 is filled correctly
    expected_total_balance_coin1 = [100, 100, 100, 110, 110]
    result_total_balance_coin1 = result[result['coin_id'] == 'coin1']['total_balance'].tolist()
    assert result_total_balance_coin1 == expected_total_balance_coin1, f"Expected {expected_total_balance_coin1}, but got {result_total_balance_coin1}"

    # Assert buyers_new for coin1 is filled correctly
    expected_buyers_new_coin1 = [1, 0, 0, 0, 0]
    result_buyers_new_coin1 = result[result['coin_id'] == 'coin1']['buyers_new'].tolist()
    assert result_buyers_new_coin1 == expected_buyers_new_coin1, f"Expected {expected_buyers_new_coin1}, but got {result_buyers_new_coin1}"

    # Assert total_balance for coin3 is filled correctly
    expected_total_balance_coin3 = [0, 0, 300, 300, 300]
    result_total_balance_coin3 = result[result['coin_id'] == 'coin3']['total_balance'].tolist()
    assert result_total_balance_coin3 == expected_total_balance_coin3, f"Expected {expected_total_balance_coin3}, but got {result_total_balance_coin3}"

    # Assert buyers_new for coin3 is filled correctly
    expected_buyers_new_coin3 = [0, 0, 3, 0, 0]
    result_buyers_new_coin3 = result[result['coin_id'] == 'coin3']['buyers_new'].tolist()
    assert result_buyers_new_coin3 == expected_buyers_new_coin3, f"Expected {expected_buyers_new_coin3}, but got {result_buyers_new_coin3}"




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
def cleaned_profits_df():
    """
    Fixture to load the cleaned_profits_df from the fixtures folder.
    """
    return pd.read_csv('tests/fixtures/cleaned_profits_df.csv')

@pytest.fixture(scope="session")
def wallet_cohort_df():
    """
    Fixture to load the wallet_cohort_df from the fixtures folder.
    """
    return pd.read_csv('tests/fixtures/wallet_cohort_df.csv')


# ---------------------------------- #
# generate_buysell_metrics_df() integration tests
# ---------------------------------- #

@pytest.fixture(scope="session")
def buysell_metrics_df(cleaned_profits_df, wallet_cohort_df, config):
    """
    Fixture to generate the buysell_metrics_df.
    """
    # Generate inputs for generate_buysell_metrics_df
    cohort_wallets = wallet_cohort_df[wallet_cohort_df['in_cohort']]['wallet_address'].unique()

    # Generate the buysell_metrics_df
    return cwm.generate_buysell_metrics_df(
        cleaned_profits_df,
        config['training_data']['training_period_end'],
        cohort_wallets
    )

# Save buysell_metrics_df.csv in fixtures/
# ----------------------------------------
@pytest.mark.integration
def test_save_buysell_metrics_df(buysell_metrics_df):
    """
    This is not a test! This function saves a buysell_metrics_df.csv in the fixtures folder
    so it can be used for integration tests in other modules.
    """
    # Save the cleaned DataFrame to the fixtures folder
    buysell_metrics_df.to_csv('tests/fixtures/buysell_metrics_df.csv', index=False)

    # Add some basic assertions to ensure the data was saved correctly
    assert buysell_metrics_df is not None
    assert len(buysell_metrics_df) > 0


@pytest.mark.integration
def test_integration_buysell_metrics_df(buysell_metrics_df, cleaned_profits_df, wallet_cohort_df, config):
    """
    Integration test for the buysell_metrics_df fixture.
    Validates the structure and key calculations in the final DataFrame.
    """

    # 1. Validate Structure: Check for expected columns in buysell_metrics_df
    expected_columns = [
        'date', 'buyers_new', 'buyers_repeat', 'total_buyers', 'sellers_new', 'sellers_repeat',
        'total_sellers', 'total_bought', 'total_sold', 'total_net_transfers', 'total_volume',
        'total_holders', 'total_balance', 'coin_id'
    ]
    assert set(expected_columns).issubset(buysell_metrics_df.columns), "Missing expected columns in buysell_metrics_df"

    # 2. Validate Key Feature Calculations
    # Filter the cleaned_profits_df to only include cohort wallets and coins
    cohort_wallets = wallet_cohort_df[wallet_cohort_df['in_cohort']]['wallet_address']

    cohort_profits_df = cleaned_profits_df[
        (cleaned_profits_df['wallet_address'].isin(cohort_wallets)) &
        (cleaned_profits_df['date'] <= config['training_data']['training_period_end'])  # Add date filtering
    ]

    # Check that total_bought matches the sum of positive net_transfers in cohort_profits_df
    total_bought_mock = cohort_profits_df[cohort_profits_df['net_transfers'] > 0]['net_transfers'].sum()
    total_bought_result = buysell_metrics_df['total_bought'].sum()
    assert total_bought_mock == pytest.approx(total_bought_result, rel=1e-9), f"Total bought mismatch: {total_bought_mock} != {total_bought_result}"

    # Check that total_sold matches the sum of negative net_transfers in cohort_profits_df
    total_sold_mock = abs(cohort_profits_df[cohort_profits_df['net_transfers'] < 0]['net_transfers'].sum())
    total_sold_result = buysell_metrics_df['total_sold'].sum()
    assert total_sold_mock == pytest.approx(total_sold_result, rel=1e-9), f"Total sold mismatch: {total_sold_mock} != {total_sold_result}"

    # Check that total_net_transfers matches the net of all net_transfers in cohort_profits_df
    total_net_transfers_mock = cohort_profits_df['net_transfers'].sum()
    total_net_transfers_result = buysell_metrics_df['total_net_transfers'].sum()
    assert total_net_transfers_mock == pytest.approx(total_net_transfers_result, rel=1e-9), f"Total net transfers mismatch: {total_net_transfers_mock} != {total_net_transfers_result}"

    # 3. Data Quality Checks
    # Ensure there are no NaN values in critical columns
    critical_columns = buysell_metrics_df.columns
    for col in critical_columns:
        assert buysell_metrics_df[col].isnull().sum() == 0, f"Found NaN values in {col}"

    # Check that all dates in buysell_metrics_df fall within the expected range
    assert buysell_metrics_df['date'].max() <= pd.to_datetime(config['training_data']['training_period_end']), \
        "Found data beyond the training period end date"
    assert buysell_metrics_df['date'].min() >= pd.to_datetime(config['training_data']['training_period_start']), \
        "Found data before the training period start date"

    # Check for missing dates in each coin-wallet pair up to the training_period_end
    missing_dates = buysell_metrics_df.groupby('coin_id').apply(
        lambda x: pd.date_range(start=x['date'].min(), end=pd.to_datetime(config['training_data']['training_period_end'])).difference(x['date'])
    ,include_groups=False)
    if any(len(missing) > 0 for missing in missing_dates):
        raise ValueError("Timeseries contains missing dates. Ensure all dates are filled up to the training_period_end before calling flatten_coin_date_df().")
