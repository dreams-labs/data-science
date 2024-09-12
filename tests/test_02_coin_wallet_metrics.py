"""
tests used to audit the files in the data-science/src folder
"""
# pylint: disable=C0301 # line over 100 chars
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


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import coin_wallet_metrics as cwm # type: ignore[reportMissingImports]
from utils import load_config # type: ignore[reportMissingImports]

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

def test_unit_generate_buysell_metrics_df(mock_profits_df):
    """
    tests the generation of buysell metrics for a wallet-coin cohort
    """
    cohort_wallets = ['wallet1', 'wallet2', 'wallet3', 'wallet5']  # Include wallet5
    cohort_coins = ['coin1', 'coin2', 'coin3']  # Cohort coins
    training_period_end = '2024-01-05'  # Set a training period end date

    # Call the function
    result_df = cwm.generate_buysell_metrics_df(mock_profits_df, training_period_end, cohort_wallets, cohort_coins)

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

    # Filter mock_profits_df to only include cohort wallets and coins
    cohort_profits_df = mock_profits_df[
        (mock_profits_df['wallet_address'].isin(cohort_wallets)) & 
        (mock_profits_df['coin_id'].isin(cohort_coins))
    ]

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


# # ---------------------------------- #
# # resample_profits_df() unit tests
# # ---------------------------------- #


# def test_single_buy_transfer():
#     data = {
#         'wallet_address': ['wallet1'],
#         'coin_id': ['coinA'],
#         'date': ['2024-09-01'],
#         'balance': [600],
#         'net_transfers': [200]
#     }
#     profits_df = pd.DataFrame(data)
#     profits_df['date'] = pd.to_datetime(profits_df['date'])

#     # Resample over 3 days
#     resampled_df = cwm.resample_profits_df(profits_df, resampling_period=3)

#     # Expected output
#     assert resampled_df['net_transfers'].iloc[0] == 200
#     assert resampled_df['balance'].iloc[0] == 600

# def test_single_sell_transfer():
#     data = {
#         'wallet_address': ['wallet1'],
#         'coin_id': ['coinA'],
#         'date': ['2024-09-01'],
#         'balance': [550],
#         'net_transfers': [-50]
#     }
#     profits_df = pd.DataFrame(data)
#     profits_df['date'] = pd.to_datetime(profits_df['date'])

#     # Resample over 3 days
#     resampled_df = cwm.resample_profits_df(profits_df, resampling_period=3)

#     # Expected output
#     assert resampled_df['net_transfers'].iloc[0] == -50
#     assert resampled_df['balance'].iloc[0] == 550

# def test_offsetting_transfers():
#     data = {
#         'wallet_address': ['wallet1', 'wallet1'],
#         'coin_id': ['coinA', 'coinA'],
#         'date': ['2024-09-01', '2024-09-02'],
#         'balance': [500, 400],
#         'net_transfers': [100, -100]
#     }
#     profits_df = pd.DataFrame(data)
#     profits_df['date'] = pd.to_datetime(profits_df['date'])

#     # Resample over 3 days
#     resampled_df = cwm.resample_profits_df(profits_df, resampling_period=3)

#     # Expect no output row since net_transfers cancel out
#     assert resampled_df.empty

# def test_mixed_transactions():
#     data = {
#         'wallet_address': ['wallet1', 'wallet1'],
#         'coin_id': ['coinA', 'coinA'],
#         'date': ['2024-09-01', '2024-09-03'],
#         'balance': [400, 350],
#         'net_transfers': [100, -50]
#     }
#     profits_df = pd.DataFrame(data)
#     profits_df['date'] = pd.to_datetime(profits_df['date'])

#     # Resample over 3 days
#     resampled_df = cwm.resample_profits_df(profits_df, resampling_period=3)

#     # Expected output
#     assert resampled_df['net_transfers'].iloc[0] == 50  # Sum of transfers
#     assert resampled_df['balance'].iloc[0] == 350  # Last balance

# def test_multiple_coins():
#     data = {
#         'wallet_address': ['wallet1', 'wallet1'],
#         'coin_id': ['coinA', 'coinB'],
#         'date': ['2024-09-01', '2024-09-01'],
#         'balance': [300, 200],
#         'net_transfers': [100, -50]
#     }
#     profits_df = pd.DataFrame(data)
#     profits_df['date'] = pd.to_datetime(profits_df['date'])

#     # Resample over 3 days
#     resampled_df = cwm.resample_profits_df(profits_df, resampling_period=3)

#     # Expected output: one row for each coin
#     assert resampled_df.shape[0] == 2
#     assert resampled_df['net_transfers'].iloc[0] == 100  # Coin A
#     assert resampled_df['net_transfers'].iloc[1] == -50  # Coin B

# def test_transactions_two_resampling_periods():
#     data = {
#         'wallet_address': ['wallet1', 'wallet1'],
#         'coin_id': ['coinA', 'coinA'],
#         'date': ['2024-09-01', '2024-09-05'],
#         'balance': [200, 250],
#         'net_transfers': [100, 50]
#     }
#     profits_df = pd.DataFrame(data)
#     profits_df['date'] = pd.to_datetime(profits_df['date'])

#     # Resample over 3 days
#     resampled_df = cwm.resample_profits_df(profits_df, resampling_period=3)

#     # Check for two distinct periods
#     assert resampled_df.shape[0] == 2

#     # Explicit checks for each period
#     first_period = resampled_df[resampled_df['date'] == '2024-09-01']
#     second_period = resampled_df[resampled_df['date'] == '2024-09-04']

#     assert first_period['net_transfers'].values[0] == 100
#     assert first_period['balance'].values[0] == 200
#     assert second_period['net_transfers'].values[0] == 50
#     assert second_period['balance'].values[0] == 250

# def test_transactions_three_resampling_periods():
#     data = {
#         'wallet_address': ['wallet1', 'wallet1', 'wallet1'],
#         'coin_id': ['coinA', 'coinA', 'coinA'],
#         'date': ['2024-09-02', '2024-09-06', '2024-09-09'],
#         'balance': [150, 250, 200],
#         'net_transfers': [50, 100, -50]
#     }
#     profits_df = pd.DataFrame(data)
#     profits_df['date'] = pd.to_datetime(profits_df['date'])

#     # Resample over 3 days
#     resampled_df = cwm.resample_profits_df(profits_df, resampling_period=3)

#     # Check for three distinct periods
#     assert resampled_df.shape[0] == 3

#     # Explicit checks for each period
#     period_1 = resampled_df[resampled_df['date'] == '2024-09-02']
#     period_2 = resampled_df[resampled_df['date'] == '2024-09-05']
#     period_3 = resampled_df[resampled_df['date'] == '2024-09-08']

#     assert period_1['net_transfers'].values[0] == 50
#     assert period_1['balance'].values[0] == 150
#     assert period_2['net_transfers'].values[0] == 100
#     assert period_2['balance'].values[0] == 250
#     assert period_3['net_transfers'].values[0] == -50
#     assert period_3['balance'].values[0] == 200

# def test_no_transactions_in_some_periods():
#     data = {
#         'wallet_address': ['wallet1', 'wallet1'],
#         'coin_id': ['coinA', 'coinA'],
#         'date': ['2024-09-01', '2024-09-08'],
#         'balance': [400, 500],
#         'net_transfers': [200, 100]
#     }
#     profits_df = pd.DataFrame(data)
#     profits_df['date'] = pd.to_datetime(profits_df['date'])

#     # Resample over 3 days
#     resampled_df = cwm.resample_profits_df(profits_df, resampling_period=3)

#     # Check for two periods
#     assert resampled_df.shape[0] == 2

#     # Explicit checks for each period
#     first_period = resampled_df[resampled_df['date'] == '2024-09-01']
#     third_period = resampled_df[resampled_df['date'] == '2024-09-07']

#     assert first_period['net_transfers'].values[0] == 200
#     assert first_period['balance'].values[0] == 400
#     assert third_period['net_transfers'].values[0] == 100
#     assert third_period['balance'].values[0] == 500

# def test_transactions_boundary_periods():
#     # Updated data with an additional transaction on 2024-09-01
#     data = {
#         'wallet_address': ['wallet1', 'wallet1', 'wallet1'],
#         'coin_id': ['coinA', 'coinA', 'coinA'],
#         'date': ['2024-09-01', '2024-09-03', '2024-09-04'],
#         'balance': [400, 300, 250],
#         'net_transfers': [150, 100, -50]
#     }
#     profits_df = pd.DataFrame(data)
#     profits_df['date'] = pd.to_datetime(profits_df['date'])

#     # Resample over 3 days
#     resampled_df = cwm.resample_profits_df(profits_df, resampling_period=3)

#     # Check for two periods
#     assert resampled_df.shape[0] == 2

#     # Explicit checks for each period
#     period_1 = resampled_df[resampled_df['date'] == '2024-09-01']
#     period_2 = resampled_df[resampled_df['date'] == '2024-09-04']

#     # First period should sum the net_transfers of the first two transactions
#     assert period_1['net_transfers'].values[0] == 250  # 150 + 100
#     assert period_1['balance'].values[0] == 300  # Last balance in the period

#     # Second period should retain the third transaction
#     assert period_2['net_transfers'].values[0] == -50
#     assert period_2['balance'].values[0] == 250

# def test_multiple_coins_multiple_periods():
#     data = {
#         'wallet_address': ['wallet1', 'wallet1', 'wallet1', 'wallet1'],
#         'coin_id': ['coinA', 'coinA', 'coinB', 'coinB'],
#         'date': ['2024-09-01', '2024-09-04', '2024-09-01', '2024-09-06'],
#         'balance': [300, 350, 100, 200],
#         'net_transfers': [100, 50, -50, 100]
#     }
#     profits_df = pd.DataFrame(data)
#     profits_df['date'] = pd.to_datetime(profits_df['date'])

#     # Resample over 3 days
#     resampled_df = cwm.resample_profits_df(profits_df, resampling_period=3)

#     # Check for four periods (2 for each coin)
#     assert resampled_df.shape[0] == 4

#     # Explicit checks for Coin A and Coin B
#     coin_a_period_1 = resampled_df[(resampled_df['coin_id'] == 'coinA') & (resampled_df['date'] == '2024-09-01')]
#     coin_a_period_2 = resampled_df[(resampled_df['coin_id'] == 'coinA') & (resampled_df['date'] == '2024-09-04')]
#     coin_b_period_1 = resampled_df[(resampled_df['coin_id'] == 'coinB') & (resampled_df['date'] == '2024-09-01')]
#     coin_b_period_2 = resampled_df[(resampled_df['coin_id'] == 'coinB') & (resampled_df['date'] == '2024-09-04')]

#     # Coin A checks
#     assert coin_a_period_1['net_transfers'].values[0] == 100
#     assert coin_a_period_1['balance'].values[0] == 300
#     assert coin_a_period_2['net_transfers'].values[0] == 50
#     assert coin_a_period_2['balance'].values[0] == 350

#     # Coin B checks
#     assert coin_b_period_1['net_transfers'].values[0] == -50
#     assert coin_b_period_1['balance'].values[0] == 100
#     assert coin_b_period_2['net_transfers'].values[0] == 100
#     assert coin_b_period_2['balance'].values[0] == 200






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

@pytest.fixture(scope="session")
def buysell_metrics_df(cleaned_profits_df, shark_wallets_df, shark_coins_df, config):
    """
    Fixture to generate the buysell_metrics_df from the cleaned_profits_df, shark_wallets_df, and shark_coins_df.
    """
    # Generate inputs for generate_buysell_metrics_df
    cohort_wallets = shark_wallets_df[shark_wallets_df['is_shark']]['wallet_address'].unique()
    cohort_coins = shark_coins_df['coin_id'].unique()

    # Generate the buysell_metrics_df
    return cwm.generate_buysell_metrics_df(
        cleaned_profits_df,
        config['training_data']['training_period_end'],
        cohort_wallets,
        cohort_coins
    )

@pytest.mark.integration
def test_integration_buysell_metrics_df(buysell_metrics_df, cleaned_profits_df, shark_wallets_df, shark_coins_df, config):
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
    cohort_wallets = shark_wallets_df[shark_wallets_df['is_shark']]['wallet_address'].unique()
    cohort_coins = shark_coins_df['coin_id'].unique()

    cohort_profits_df = cleaned_profits_df[
        (cleaned_profits_df['wallet_address'].isin(cohort_wallets)) &
        (cleaned_profits_df['coin_id'].isin(cohort_coins)) &
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
    critical_columns = ['total_bought', 'total_sold', 'total_net_transfers', 'total_balance']
    for col in critical_columns:
        assert buysell_metrics_df[col].isnull().sum() == 0, f"Found NaN values in {col}"

    # Ensure non-cohort wallets and coins are excluded
    assert set(buysell_metrics_df['coin_id']).issubset(cohort_coins), "Non-cohort coins found in buysell_metrics_df"
