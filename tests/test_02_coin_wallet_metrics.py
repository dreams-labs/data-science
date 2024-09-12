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
import yaml
import pandas as pd
from dotenv import load_dotenv
import pytest
from dreams_core import core as dc


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import coin_wallet_metrics as cwm # type: ignore[reportMissingImports]

load_dotenv()
logger = dc.setup_logger()

# ---------------------------------- #
# set up config and module-level variables
# ---------------------------------- #

def load_config():
    """
    Fixture to load test config from test_config.yaml.
    """
    config_path = os.path.join(os.path.dirname(__file__), 'test_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

config = load_config()


# ===================================================== #
#                                                       #
#                 U N I T   T E S T S                   #
#                                                       #
# ===================================================== #

# ---------------------------------- #
# resample_profits_df() unit tests
# ---------------------------------- #


def test_single_buy_transfer():
    data = {
        'wallet_address': ['wallet1'],
        'coin_id': ['coinA'],
        'date': ['2024-09-01'],
        'balance': [600],
        'net_transfers': [200]
    }
    profits_df = pd.DataFrame(data)
    profits_df['date'] = pd.to_datetime(profits_df['date'])

    # Resample over 3 days
    resampled_df = cwm.resample_profits_df(profits_df, resampling_period=3)

    # Expected output
    assert resampled_df['net_transfers'].iloc[0] == 200
    assert resampled_df['balance'].iloc[0] == 600

def test_single_sell_transfer():
    data = {
        'wallet_address': ['wallet1'],
        'coin_id': ['coinA'],
        'date': ['2024-09-01'],
        'balance': [550],
        'net_transfers': [-50]
    }
    profits_df = pd.DataFrame(data)
    profits_df['date'] = pd.to_datetime(profits_df['date'])

    # Resample over 3 days
    resampled_df = cwm.resample_profits_df(profits_df, resampling_period=3)

    # Expected output
    assert resampled_df['net_transfers'].iloc[0] == -50
    assert resampled_df['balance'].iloc[0] == 550

def test_offsetting_transfers():
    data = {
        'wallet_address': ['wallet1', 'wallet1'],
        'coin_id': ['coinA', 'coinA'],
        'date': ['2024-09-01', '2024-09-02'],
        'balance': [500, 400],
        'net_transfers': [100, -100]
    }
    profits_df = pd.DataFrame(data)
    profits_df['date'] = pd.to_datetime(profits_df['date'])

    # Resample over 3 days
    resampled_df = cwm.resample_profits_df(profits_df, resampling_period=3)

    # Expect no output row since net_transfers cancel out
    assert resampled_df.empty

def test_mixed_transactions():
    data = {
        'wallet_address': ['wallet1', 'wallet1'],
        'coin_id': ['coinA', 'coinA'],
        'date': ['2024-09-01', '2024-09-03'],
        'balance': [400, 350],
        'net_transfers': [100, -50]
    }
    profits_df = pd.DataFrame(data)
    profits_df['date'] = pd.to_datetime(profits_df['date'])

    # Resample over 3 days
    resampled_df = cwm.resample_profits_df(profits_df, resampling_period=3)

    # Expected output
    assert resampled_df['net_transfers'].iloc[0] == 50  # Sum of transfers
    assert resampled_df['balance'].iloc[0] == 350  # Last balance

def test_multiple_coins():
    data = {
        'wallet_address': ['wallet1', 'wallet1'],
        'coin_id': ['coinA', 'coinB'],
        'date': ['2024-09-01', '2024-09-01'],
        'balance': [300, 200],
        'net_transfers': [100, -50]
    }
    profits_df = pd.DataFrame(data)
    profits_df['date'] = pd.to_datetime(profits_df['date'])

    # Resample over 3 days
    resampled_df = cwm.resample_profits_df(profits_df, resampling_period=3)

    # Expected output: one row for each coin
    assert resampled_df.shape[0] == 2
    assert resampled_df['net_transfers'].iloc[0] == 100  # Coin A
    assert resampled_df['net_transfers'].iloc[1] == -50  # Coin B

def test_transactions_two_resampling_periods():
    data = {
        'wallet_address': ['wallet1', 'wallet1'],
        'coin_id': ['coinA', 'coinA'],
        'date': ['2024-09-01', '2024-09-05'],
        'balance': [200, 250],
        'net_transfers': [100, 50]
    }
    profits_df = pd.DataFrame(data)
    profits_df['date'] = pd.to_datetime(profits_df['date'])

    # Resample over 3 days
    resampled_df = cwm.resample_profits_df(profits_df, resampling_period=3)

    # Check for two distinct periods
    assert resampled_df.shape[0] == 2

    # Explicit checks for each period
    first_period = resampled_df[resampled_df['date'] == '2024-09-01']
    second_period = resampled_df[resampled_df['date'] == '2024-09-04']

    assert first_period['net_transfers'].values[0] == 100
    assert first_period['balance'].values[0] == 200
    assert second_period['net_transfers'].values[0] == 50
    assert second_period['balance'].values[0] == 250

def test_transactions_three_resampling_periods():
    data = {
        'wallet_address': ['wallet1', 'wallet1', 'wallet1'],
        'coin_id': ['coinA', 'coinA', 'coinA'],
        'date': ['2024-09-02', '2024-09-06', '2024-09-09'],
        'balance': [150, 250, 200],
        'net_transfers': [50, 100, -50]
    }
    profits_df = pd.DataFrame(data)
    profits_df['date'] = pd.to_datetime(profits_df['date'])

    # Resample over 3 days
    resampled_df = cwm.resample_profits_df(profits_df, resampling_period=3)

    # Check for three distinct periods
    assert resampled_df.shape[0] == 3

    # Explicit checks for each period
    period_1 = resampled_df[resampled_df['date'] == '2024-09-02']
    period_2 = resampled_df[resampled_df['date'] == '2024-09-05']
    period_3 = resampled_df[resampled_df['date'] == '2024-09-08']

    assert period_1['net_transfers'].values[0] == 50
    assert period_1['balance'].values[0] == 150
    assert period_2['net_transfers'].values[0] == 100
    assert period_2['balance'].values[0] == 250
    assert period_3['net_transfers'].values[0] == -50
    assert period_3['balance'].values[0] == 200

def test_no_transactions_in_some_periods():
    data = {
        'wallet_address': ['wallet1', 'wallet1'],
        'coin_id': ['coinA', 'coinA'],
        'date': ['2024-09-01', '2024-09-08'],
        'balance': [400, 500],
        'net_transfers': [200, 100]
    }
    profits_df = pd.DataFrame(data)
    profits_df['date'] = pd.to_datetime(profits_df['date'])

    # Resample over 3 days
    resampled_df = cwm.resample_profits_df(profits_df, resampling_period=3)

    # Check for two periods
    assert resampled_df.shape[0] == 2

    # Explicit checks for each period
    first_period = resampled_df[resampled_df['date'] == '2024-09-01']
    third_period = resampled_df[resampled_df['date'] == '2024-09-07']

    assert first_period['net_transfers'].values[0] == 200
    assert first_period['balance'].values[0] == 400
    assert third_period['net_transfers'].values[0] == 100
    assert third_period['balance'].values[0] == 500

def test_transactions_boundary_periods():
    # Updated data with an additional transaction on 2024-09-01
    data = {
        'wallet_address': ['wallet1', 'wallet1', 'wallet1'],
        'coin_id': ['coinA', 'coinA', 'coinA'],
        'date': ['2024-09-01', '2024-09-03', '2024-09-04'],
        'balance': [400, 300, 250],
        'net_transfers': [150, 100, -50]
    }
    profits_df = pd.DataFrame(data)
    profits_df['date'] = pd.to_datetime(profits_df['date'])

    # Resample over 3 days
    resampled_df = cwm.resample_profits_df(profits_df, resampling_period=3)

    # Check for two periods
    assert resampled_df.shape[0] == 2

    # Explicit checks for each period
    period_1 = resampled_df[resampled_df['date'] == '2024-09-01']
    period_2 = resampled_df[resampled_df['date'] == '2024-09-04']

    # First period should sum the net_transfers of the first two transactions
    assert period_1['net_transfers'].values[0] == 250  # 150 + 100
    assert period_1['balance'].values[0] == 300  # Last balance in the period

    # Second period should retain the third transaction
    assert period_2['net_transfers'].values[0] == -50
    assert period_2['balance'].values[0] == 250

def test_multiple_coins_multiple_periods():
    data = {
        'wallet_address': ['wallet1', 'wallet1', 'wallet1', 'wallet1'],
        'coin_id': ['coinA', 'coinA', 'coinB', 'coinB'],
        'date': ['2024-09-01', '2024-09-04', '2024-09-01', '2024-09-06'],
        'balance': [300, 350, 100, 200],
        'net_transfers': [100, 50, -50, 100]
    }
    profits_df = pd.DataFrame(data)
    profits_df['date'] = pd.to_datetime(profits_df['date'])

    # Resample over 3 days
    resampled_df = cwm.resample_profits_df(profits_df, resampling_period=3)

    # Check for four periods (2 for each coin)
    assert resampled_df.shape[0] == 4

    # Explicit checks for Coin A and Coin B
    coin_a_period_1 = resampled_df[(resampled_df['coin_id'] == 'coinA') & (resampled_df['date'] == '2024-09-01')]
    coin_a_period_2 = resampled_df[(resampled_df['coin_id'] == 'coinA') & (resampled_df['date'] == '2024-09-04')]
    coin_b_period_1 = resampled_df[(resampled_df['coin_id'] == 'coinB') & (resampled_df['date'] == '2024-09-01')]
    coin_b_period_2 = resampled_df[(resampled_df['coin_id'] == 'coinB') & (resampled_df['date'] == '2024-09-04')]

    # Coin A checks
    assert coin_a_period_1['net_transfers'].values[0] == 100
    assert coin_a_period_1['balance'].values[0] == 300
    assert coin_a_period_2['net_transfers'].values[0] == 50
    assert coin_a_period_2['balance'].values[0] == 350

    # Coin B checks
    assert coin_b_period_1['net_transfers'].values[0] == -50
    assert coin_b_period_1['balance'].values[0] == 100
    assert coin_b_period_2['net_transfers'].values[0] == 100
    assert coin_b_period_2['balance'].values[0] == 200






# ======================================================== #
#                                                          #
#            I N T E G R A T I O N   T E S T S             #
#                                                          #
# ======================================================== #

