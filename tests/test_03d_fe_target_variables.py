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
import feature_engineering.target_variables as tv

load_dotenv()
logger = dc.setup_logger()



# ===================================================== #
#                                                       #
#                 U N I T   T E S T S                   #
#                                                       #
# ===================================================== #

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
    target_variables_df = tv.calculate_mooncrater_targets(returns_df, modeling_config)

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
    returns_df = tv.calculate_coin_returns(valid_prices_df,
                                                                valid_training_data_config)

    expected_returns = pd.DataFrame({
        'coin_id': ['BTC', 'ETH', 'XRP'],
        'returns': [0.166667, 0.25, 0.2]
    })

    assert np.all(np.isclose(returns_df['returns'].values,
                            expected_returns['returns'].values,
                            rtol=1e-4, atol=1e-4))

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
    returns_df = tv.calculate_coin_returns(no_change_prices_df,
                                                                valid_training_data_config)

    expected_returns = pd.DataFrame({
        'coin_id': ['BTC', 'ETH', 'XRP'],
        'returns': [0.0, 0.25, 0.0]
    })

    assert (np.isclose(returns_df['returns'].values,
                       expected_returns['returns'].values,
                       rtol=1e-4, atol=1e-4)).all()



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
    returns_df = tv.calculate_coin_returns(negative_returns_prices_df,
                                                                valid_training_data_config)

    expected_returns = pd.DataFrame({
        'coin_id': ['BTC', 'ETH', 'XRP'],
        'returns': [-0.1667, 0.25, -0.2]
    })

    assert (np.isclose(returns_df['returns'].values,
                       expected_returns['returns'].values,
                       rtol=1e-4, atol=1e-4)).all()


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
    returns_df = tv.calculate_coin_returns(multiple_datapoints_prices_df,
                                                                valid_training_data_config)

    expected_returns = pd.DataFrame({
        'coin_id': ['BTC', 'ETH', 'XRP'],
        'returns': [0.1667, 0.25, 0.2]
    })

    assert (np.isclose(returns_df['returns'].values,
                       expected_returns['returns'].values,
                       rtol=1e-4,pytestatol=1e-4)).all()
