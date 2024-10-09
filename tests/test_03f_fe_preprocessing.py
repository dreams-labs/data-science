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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dotenv import load_dotenv
import pytest
from dreams_core import core as dc

# pyright: reportMissingImports=false
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import feature_engineering.preprocessing as prp

load_dotenv()
logger = dc.setup_logger()



# ===================================================== #
#                                                       #
#                 U N I T   T E S T S                   #
#                                                       #
# ===================================================== #

# ------------------------------------------ #
# ScalingProcessor unit tests
# ------------------------------------------ #

@pytest.fixture
def scaling_1_metrics_config():
    """
    Fixture providing a sample metrics configuration dictionary.
    """
    return {
        'time_series': {
            'market_data': {
                'price': {
                    'aggregations': {
                        'std': {
                            'scaling': 'none'
                        }
                    }
                },
                'volume': {
                    'aggregations': {
                        'sum': {
                            'scaling': 'standard'
                        }
                    }
                },
                'market_cap': {
                    'aggregations': {
                        'last': {
                            'scaling': 'log'
                        }
                    }
                }
            }
        }
    }

@pytest.fixture
def scaling_1_dummy_dataframe():
    """
    Fixture providing a dummy dataframe with a MultiIndex and sample data.
    """
    index = pd.MultiIndex.from_product(
        [
            pd.to_datetime(['2023-01-01', '2023-01-02']),
            ['bitcoin', 'ethereum']
        ],
        names=['time_window', 'coin_id']
    )
    data = {
        'time_series_market_data_price_std': [1.0, 2.0, 3.0, 4.0],
        'time_series_market_data_volume_sum': [100, 200, 300, 400],
        'time_series_market_data_market_cap_last': [1000, 2000, 3000, 4000]
    }
    df = pd.DataFrame(data, index=index)
    return df

@pytest.mark.unit
def test_scaling_processor(scaling_1_metrics_config, scaling_1_dummy_dataframe):
    """
    Test the ScalingProcessor class for correct column mapping and scaling application.
    """
    # Instantiate the ScalingProcessor with the provided metrics_config
    processor = prp.ScalingProcessor(scaling_1_metrics_config)

    # Expected column_scaling_map based on the metrics_config
    expected_column_scaling_map = {
        'time_series_market_data_price_std': 'none',
        'time_series_market_data_volume_sum': 'standard',
        'time_series_market_data_market_cap_last': 'log'
    }

    # Assert that the column_scaling_map is as expected
    assert processor.column_scaling_map == expected_column_scaling_map, (
        "Column scaling map does not match expected mapping."
    )

    # Apply scaling to the dummy_dataframe (as training data)
    scaled_df = processor.apply_scaling(scaling_1_dummy_dataframe, is_train=True)

    # Prepare expected scaled values for each column

    # For 'time_series_market_data_price_std', scaling is 'none',
    # so values should remain the same as in the original dataframe.
    expected_price_std = scaling_1_dummy_dataframe['time_series_market_data_price_std'].values

    # For 'time_series_market_data_volume_sum', scaling is 'standard'.
    # This means we need to standardize the values by subtracting the mean and dividing by the std deviation.
    volume_values = scaling_1_dummy_dataframe['time_series_market_data_volume_sum'].values.reshape(-1, 1)
    # Calculate mean and standard deviation of the volume values
    volume_mean = volume_values.mean()
    volume_std = volume_values.std()
    # Standardize the volume values
    expected_volume_sum = (volume_values - volume_mean) / volume_std
    expected_volume_sum = expected_volume_sum.flatten()

    # For 'time_series_market_data_market_cap_last', scaling is 'log'.
    # We apply the natural logarithm to the values (using np.log1p to handle zero values safely).
    market_cap_values = scaling_1_dummy_dataframe['time_series_market_data_market_cap_last'].values
    expected_market_cap_last = np.log1p(market_cap_values)

    # Now, we compare the scaled values in scaled_df to the expected values calculated above.

    # Compare 'time_series_market_data_price_std' values
    np.testing.assert_allclose(
        scaled_df['time_series_market_data_price_std'].values,
        expected_price_std,
        atol=1e-4,
        err_msg="Scaled values for 'price_std' do not match expected values."
    )

    # Compare 'time_series_market_data_volume_sum' values
    np.testing.assert_allclose(
        scaled_df['time_series_market_data_volume_sum'].values,
        expected_volume_sum,
        atol=1e-4,
        err_msg="Scaled values for 'volume_sum' do not match expected standardized values."
    )

    # Compare 'time_series_market_data_market_cap_last' values
    np.testing.assert_allclose(
        scaled_df['time_series_market_data_market_cap_last'].values,
        expected_market_cap_last,
        atol=1e-4,
        err_msg="Scaled values for 'market_cap_last' do not match expected log-transformed values."
    )


@pytest.fixture
def complex_metrics_config():
    """Complex metrics_config structure with nested aggregations"""
    return {
        "wallet_cohorts": {
            "whales": {
                "total_volume": {
                    "aggregations": {
                        "last": {
                            "scaling": "log"
                        }
                    },
                    "rolling": {
                        "aggregations": {
                            "mean": {
                                "scaling": "log"
                            }
                        },
                        "window_duration": 10,
                        "lookback_periods": 3
                    },
                    "indicators": {
                        "ema": {
                            "parameters": {
                                "window": [7]
                            },
                            "aggregations": {
                                "last": {
                                    "scaling": "none"
                                }
                            },
                            "rolling": {
                                "aggregations": {
                                    "last": {
                                        "scaling": "standard"
                                    }
                                },
                                "window_duration": 7,
                                "lookback_periods": 3
                            }
                        }
                    }
                }
            }
        }
    }
