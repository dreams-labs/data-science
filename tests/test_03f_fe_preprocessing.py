"""
tests used to audit the files in the data-science/src folder
"""
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
def rolling_metrics_config():
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
                    }
                }
            }
        }
    }

@pytest.fixture
def dummy_rolling_dataframe():
    """
    Fixture providing a dummy dataframe with a MultiIndex and sample data for rolling metrics.
    """
    index = pd.MultiIndex.from_product(
        [
            pd.to_datetime(['2023-01-01', '2023-01-02']),
            ['bitcoin', 'ethereum']
        ],
        names=['time_window', 'coin_id']
    )
    data = {
        'wallet_cohorts_whales_total_volume_last': [1000, 2000, 3000, 4000],
        'wallet_cohorts_whales_total_volume_mean_10d_period_1': [500, 600, 700, 800],
        'wallet_cohorts_whales_total_volume_mean_10d_period_2': [400, 500, 600, 700],
        'wallet_cohorts_whales_total_volume_mean_10d_period_3': [300, 400, 500, 600],
    }
    df = pd.DataFrame(data, index=index)
    return df

@pytest.mark.unit
def test_scaling_processor_with_rolling_metrics(rolling_metrics_config, dummy_rolling_dataframe):
    """
    Test the ScalingProcessor class for correct column mapping and scaling application with rolling metrics.
    """
    # Instantiate the ScalingProcessor with the provided rolling_metrics_config
    processor = prp.ScalingProcessor(rolling_metrics_config)

    # Expected column_scaling_map based on the rolling_metrics_config
    expected_column_scaling_map = {
        'wallet_cohorts_whales_total_volume_last': 'log',
        'wallet_cohorts_whales_total_volume_mean_10d_period_1': 'log',
        'wallet_cohorts_whales_total_volume_mean_10d_period_2': 'log',
        'wallet_cohorts_whales_total_volume_mean_10d_period_3': 'log',
    }

    # Assert that the column_scaling_map is as expected
    assert processor.column_scaling_map == expected_column_scaling_map, (
        "Column scaling map does not match expected mapping."
    )

    # Apply scaling to the dummy_rolling_dataframe (as training data)
    scaled_df = processor.apply_scaling(dummy_rolling_dataframe, is_train=True)

    # Prepare expected scaled values for each column
    # For each column, scaling is 'log', so we apply np.log1p to the original values

    # Logical steps for 'wallet_cohorts_whales_total_volume_last':
    # - Original values: [1000, 2000, 3000, 4000]
    # - Apply np.log1p to each value to get the expected scaled values
    original_values_last = dummy_rolling_dataframe['wallet_cohorts_whales_total_volume_last'].values
    expected_values_last = np.log1p(original_values_last)

    # Logical steps for 'wallet_cohorts_whales_total_volume_mean_10d_period_1':
    # - Original values: [500, 600, 700, 800]
    # - Apply np.log1p to each value
    original_values_mean1 = dummy_rolling_dataframe[
        'wallet_cohorts_whales_total_volume_mean_10d_period_1'
    ].values
    expected_values_mean1 = np.log1p(original_values_mean1)

    # Logical steps for 'wallet_cohorts_whales_total_volume_mean_10d_period_2':
    # - Original values: [400, 500, 600, 700]
    # - Apply np.log1p to each value
    original_values_mean2 = dummy_rolling_dataframe[
        'wallet_cohorts_whales_total_volume_mean_10d_period_2'
    ].values
    expected_values_mean2 = np.log1p(original_values_mean2)

    # Logical steps for 'wallet_cohorts_whales_total_volume_mean_10d_period_3':
    # - Original values: [300, 400, 500, 600]
    # - Apply np.log1p to each value
    original_values_mean3 = dummy_rolling_dataframe[
        'wallet_cohorts_whales_total_volume_mean_10d_period_3'
    ].values
    expected_values_mean3 = np.log1p(original_values_mean3)

    # Now, compare the scaled values in scaled_df to the expected values calculated above

    # Compare 'wallet_cohorts_whales_total_volume_last' values
    np.testing.assert_allclose(
        scaled_df['wallet_cohorts_whales_total_volume_last'].values,
        expected_values_last,
        atol=1e-4,
        err_msg=(
            "Scaled values for 'wallet_cohorts_whales_total_volume_last' do not match "
            "expected log-transformed values."
        )
    )

    # Compare 'wallet_cohorts_whales_total_volume_mean_10d_period_1' values
    np.testing.assert_allclose(
        scaled_df['wallet_cohorts_whales_total_volume_mean_10d_period_1'].values,
        expected_values_mean1,
        atol=1e-4,
        err_msg=(
            "Scaled values for 'wallet_cohorts_whales_total_volume_mean_10d_period_1' "
            "do not match expected log-transformed values."
        )
    )

    # Compare 'wallet_cohorts_whales_total_volume_mean_10d_period_2' values
    np.testing.assert_allclose(
        scaled_df['wallet_cohorts_whales_total_volume_mean_10d_period_2'].values,
        expected_values_mean2,
        atol=1e-4,
        err_msg=(
            "Scaled values for 'wallet_cohorts_whales_total_volume_mean_10d_period_2' "
            "do not match expected log-transformed values."
        )
    )

    # Compare 'wallet_cohorts_whales_total_volume_mean_10d_period_3' values
    np.testing.assert_allclose(
        scaled_df['wallet_cohorts_whales_total_volume_mean_10d_period_3'].values,
        expected_values_mean3,
        atol=1e-4,
        err_msg=(
            "Scaled values for 'wallet_cohorts_whales_total_volume_mean_10d_period_3' "
            "do not match expected log-transformed values."
        )
    )

@pytest.fixture
def indicators_metrics_config():
    """
    Fixture providing a metrics configuration that includes indicators.
    """
    return {
        'wallet_cohorts': {
            'whales': {
                'buyers_new': {
                    'aggregations': {
                        'sum': {
                            'scaling': 'standard'
                        }
                    },
                    'indicators': {
                        'rsi': {
                            'parameters': {
                                'window': [14, 60]
                            },
                            'aggregations': {
                                'last': {
                                    'scaling': 'minmax'
                                }
                            }
                        }
                    }
                }
            }
        }
    }

@pytest.fixture
def dummy_indicators_dataframe():
    """
    Fixture providing a dummy dataframe with sample data for indicators.
    """
    index = pd.MultiIndex.from_product(
        [
            pd.to_datetime(['2023-01-01', '2023-01-02']),
            ['bitcoin', 'ethereum']
        ],
        names=['time_window', 'coin_id']
    )
    data = {
        'wallet_cohorts_whales_buyers_new_sum': [100, 200, 300, 400],
        'wallet_cohorts_whales_buyers_new_rsi_14_last': [30, 40, 50, 60],
        'wallet_cohorts_whales_buyers_new_rsi_60_last': [35, 45, 55, 65],
    }
    df = pd.DataFrame(data, index=index)
    return df

@pytest.mark.unit
def test_scaling_processor_with_indicators(indicators_metrics_config, dummy_indicators_dataframe):
    """
    Test the ScalingProcessor class for correct column mapping and scaling application with indicators.
    """
    # Instantiate the ScalingProcessor with the provided indicators_metrics_config
    processor = prp.ScalingProcessor(indicators_metrics_config)

    # Expected column_scaling_map based on the indicators_metrics_config
    expected_column_scaling_map = {
        'wallet_cohorts_whales_buyers_new_sum': 'standard',
        'wallet_cohorts_whales_buyers_new_rsi_14_last': 'minmax',
        'wallet_cohorts_whales_buyers_new_rsi_60_last': 'minmax',
    }

    # Assert that the column_scaling_map is as expected
    assert processor.column_scaling_map == expected_column_scaling_map, (
        "Column scaling map does not match expected mapping."
    )

    # Apply scaling to the dummy_indicators_dataframe (as training data)
    scaled_df = processor.apply_scaling(dummy_indicators_dataframe, is_train=True)

    # Prepare expected scaled values for each column

    # Logical steps for 'wallet_cohorts_whales_buyers_new_sum':
    # - Scaling method is 'standard'.
    # - Original values: [100, 200, 300, 400]
    # - Calculate mean and standard deviation.
    sum_values = dummy_indicators_dataframe[
        'wallet_cohorts_whales_buyers_new_sum'
    ].values.reshape(-1, 1)
    sum_mean = sum_values.mean()
    sum_std = sum_values.std()
    # - Standardize the values.
    expected_sum = (sum_values - sum_mean) / sum_std
    expected_sum = expected_sum.flatten()

    # Logical steps for 'wallet_cohorts_whales_buyers_new_rsi_14_last':
    # - Scaling method is 'minmax'.
    # - Original values: [30, 40, 50, 60]
    # - Find min and max.
    rsi14_values = dummy_indicators_dataframe[
        'wallet_cohorts_whales_buyers_new_rsi_14_last'
    ].values.reshape(-1, 1)
    rsi14_min = rsi14_values.min()
    rsi14_max = rsi14_values.max()
    # - Apply min-max scaling.
    expected_rsi14 = (rsi14_values - rsi14_min) / (rsi14_max - rsi14_min)
    expected_rsi14 = expected_rsi14.flatten()

    # Logical steps for 'wallet_cohorts_whales_buyers_new_rsi_60_last':
    # - Scaling method is 'minmax'.
    # - Original values: [35, 45, 55, 65]
    # - Find min and max.
    rsi60_values = dummy_indicators_dataframe[
        'wallet_cohorts_whales_buyers_new_rsi_60_last'
    ].values.reshape(-1, 1)
    rsi60_min = rsi60_values.min()
    rsi60_max = rsi60_values.max()
    # - Apply min-max scaling.
    expected_rsi60 = (rsi60_values - rsi60_min) / (rsi60_max - rsi60_min)
    expected_rsi60 = expected_rsi60.flatten()

    # Now, compare the scaled values in scaled_df to the expected values calculated above

    # Compare 'wallet_cohorts_whales_buyers_new_sum' values
    np.testing.assert_allclose(
        scaled_df['wallet_cohorts_whales_buyers_new_sum'].values,
        expected_sum,
        atol=1e-4,
        err_msg=(
            "Scaled values for 'wallet_cohorts_whales_buyers_new_sum' do not match "
            "expected standardized values."
        )
    )

    # Compare 'wallet_cohorts_whales_buyers_new_rsi_14_last' values
    np.testing.assert_allclose(
        scaled_df['wallet_cohorts_whales_buyers_new_rsi_14_last'].values,
        expected_rsi14,
        atol=1e-4,
        err_msg=(
            "Scaled values for 'wallet_cohorts_whales_buyers_new_rsi_14_last' do not match "
            "expected min-max scaled values."
        )
    )

    # Compare 'wallet_cohorts_whales_buyers_new_rsi_60_last' values
    np.testing.assert_allclose(
        scaled_df['wallet_cohorts_whales_buyers_new_rsi_60_last'].values,
        expected_rsi60,
        atol=1e-4,
        err_msg=(
            "Scaled values for 'wallet_cohorts_whales_buyers_new_rsi_60_last' do not match "
            "expected min-max scaled values."
        )
    )

@pytest.fixture
def complex_metrics_config():
    """
    Fixture providing a complex metrics configuration incorporating all aggregation types.
    """
    return {
        'wallet_cohorts': {
            'whales': {
                'total_volume': {
                    'aggregations': {
                        'last': {
                            'scaling': 'log'
                        }
                    },
                    'rolling': {
                        'aggregations': {
                            'mean': {
                                'scaling': 'log'
                            }
                        },
                        'window_duration': 10,
                        'lookback_periods': 3
                    },
                    'indicators': {
                        'ema': {
                            'parameters': {
                                'window': [3, 14]
                            },
                            'aggregations': {
                                'last': {
                                    'scaling': 'none'
                                }
                            },
                            'rolling': {
                                'aggregations': {
                                    'last': {
                                        'scaling': 'standard'
                                    }
                                },
                                'window_duration': 7,
                                'lookback_periods': 3
                            }
                        }
                    }
                }
            }
        }
    }

@pytest.fixture
def dummy_complex_dataframe():
    """
    Fixture providing a dummy dataframe with sample data for complex metrics configurations.
    """
    index = pd.MultiIndex.from_product(
        [
            pd.to_datetime(['2023-01-01', '2023-01-02']),
            ['bitcoin', 'ethereum']
        ],
        names=['time_window', 'coin_id']
    )
    data = {
        'wallet_cohorts_whales_total_volume_last': [1000, 2000, 3000, 4000],
        'wallet_cohorts_whales_total_volume_mean_10d_period_1': [500, 600, 700, 800],
        'wallet_cohorts_whales_total_volume_mean_10d_period_2': [400, 500, 600, 700],
        'wallet_cohorts_whales_total_volume_mean_10d_period_3': [300, 400, 500, 600],
        'wallet_cohorts_whales_total_volume_ema_3_last': [1100, 2100, 3100, 4100],
        'wallet_cohorts_whales_total_volume_ema_3_last_7d_period_1': [1050, 2050, 3050, 4050],
        'wallet_cohorts_whales_total_volume_ema_3_last_7d_period_2': [1000, 2000, 3000, 4000],
        'wallet_cohorts_whales_total_volume_ema_3_last_7d_period_3': [950, 1950, 2950, 3950],
        'wallet_cohorts_whales_total_volume_ema_14_last': [1200, 2200, 3200, 4200],
        'wallet_cohorts_whales_total_volume_ema_14_last_7d_period_1': [1150, 2150, 3150, 4150],
        'wallet_cohorts_whales_total_volume_ema_14_last_7d_period_2': [1100, 2100, 3100, 4100],
        'wallet_cohorts_whales_total_volume_ema_14_last_7d_period_3': [1050, 2050, 3050, 4050],
    }
    df = pd.DataFrame(data, index=index)
    return df

@pytest.mark.unit
def test_scaling_processor_with_complex_metrics(
    complex_metrics_config, dummy_complex_dataframe
):
    """
    Test the ScalingProcessor class with a complex metrics configuration incorporating
    all aggregation types.
    """
    # Instantiate the ScalingProcessor with the provided complex_metrics_config
    processor = prp.ScalingProcessor(complex_metrics_config)

    # Expected column_scaling_map based on the complex_metrics_config
    expected_column_scaling_map = {
        'wallet_cohorts_whales_total_volume_last': 'log',
        'wallet_cohorts_whales_total_volume_mean_10d_period_1': 'log',
        'wallet_cohorts_whales_total_volume_mean_10d_period_2': 'log',
        'wallet_cohorts_whales_total_volume_mean_10d_period_3': 'log',
        'wallet_cohorts_whales_total_volume_ema_3_last': 'none',
        'wallet_cohorts_whales_total_volume_ema_3_last_7d_period_1': 'standard',
        'wallet_cohorts_whales_total_volume_ema_3_last_7d_period_2': 'standard',
        'wallet_cohorts_whales_total_volume_ema_3_last_7d_period_3': 'standard',
        'wallet_cohorts_whales_total_volume_ema_14_last': 'none',
        'wallet_cohorts_whales_total_volume_ema_14_last_7d_period_1': 'standard',
        'wallet_cohorts_whales_total_volume_ema_14_last_7d_period_2': 'standard',
        'wallet_cohorts_whales_total_volume_ema_14_last_7d_period_3': 'standard',
    }

    # Assert that the column_scaling_map is as expected
    assert processor.column_scaling_map == expected_column_scaling_map, (
        "Column scaling map does not match expected mapping."
    )

    # Apply scaling to the dummy_complex_dataframe (as training data)
    scaled_df = processor.apply_scaling(dummy_complex_dataframe, is_train=True)

    # Prepare expected scaled values for each column
    columns_to_test = expected_column_scaling_map.keys()

    for column in columns_to_test:
        scaling_method = expected_column_scaling_map[column]
        original_values = dummy_complex_dataframe[column].values.reshape(-1, 1)

        if scaling_method == 'log':
            # Logical steps:
            # - Original values: Retrieved from the dataframe
            # - Apply np.log1p to handle zero values safely
            expected_values = np.log1p(original_values).flatten()
        elif scaling_method == 'standard':
            # Logical steps:
            # - Calculate mean and standard deviation of the values
            mean = original_values.mean()
            std = original_values.std()
            # - Standardize the values using (value - mean) / std
            expected_values = ((original_values - mean) / std).flatten()
        elif scaling_method == 'none':
            # Logical steps:
            # - Values remain the same as original
            expected_values = original_values.flatten()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")

        # Compare the scaled values in scaled_df to the expected values
        np.testing.assert_allclose(
            scaled_df[column].values,
            expected_values,
            atol=1e-4,
            err_msg=(
                f"Scaled values for '{column}' do not match expected values with scaling '{scaling_method}'."
            )
        )


@pytest.fixture
def double_complex_metrics_config():
    """
    Fixture providing a double complex metrics configuration incorporating multiple levels.
    """
    return {
        'wallet_cohorts': {
            'whales': {
                'total_volume': {
                    'aggregations': {
                        'last': {
                            'scaling': 'log'
                        }
                    },
                    'rolling': {
                        'aggregations': {
                            'mean': {
                                'scaling': 'log'
                            }
                        },
                        'window_duration': 10,
                        'lookback_periods': 3
                    },
                    'indicators': {
                        'ema': {
                            'parameters': {
                                'window': [3, 14]
                            },
                            'aggregations': {
                                'last': {
                                    'scaling': 'none'
                                }
                            },
                            'rolling': {
                                'aggregations': {
                                    'last': {
                                        'scaling': 'standard'
                                    }
                                },
                                'window_duration': 7,
                                'lookback_periods': 3
                            }
                        }
                    }
                }
            }
        },
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
                        },
                        'std': {
                            'scaling': 'none'
                        }
                    }
                },
                'market_cap': {
                    'aggregations': {
                        'last': {
                            'scaling': 'log'
                        }
                    },
                    'indicators': {
                        'ema': {
                            'parameters': {
                                'window': [3, 14]
                            },
                            'aggregations': {
                                'last': {
                                    'scaling': 'log'
                                }
                            },
                            'rolling': {
                                'aggregations': {
                                    'sum': {
                                        'scaling': 'log'
                                    }
                                },
                                'window_duration': 10,
                                'lookback_periods': 2
                            }
                        },
                        'rsi': {
                            'parameters': {
                                'window': [14]
                            },
                            'aggregations': {
                                'last': {
                                    'scaling': 'none'
                                }
                            }
                        }
                    }
                }
            }
        }
    }


@pytest.fixture
def dummy_double_complex_dataframe():
    """
    Fixture providing a dummy dataframe with sample data for the double complex metrics configuration.
    """
    index = pd.MultiIndex.from_product(
        [
            pd.to_datetime(['2023-01-01', '2023-01-02']),
            ['bitcoin', 'ethereum']
        ],
        names=['time_window', 'coin_id']
    )
    data = {
        # wallet_cohorts.whales.total_volume
        'wallet_cohorts_whales_total_volume_last': [1000, 2000, 3000, 4000],
        'wallet_cohorts_whales_total_volume_mean_10d_period_1': [500, 600, 700, 800],
        'wallet_cohorts_whales_total_volume_mean_10d_period_2': [400, 500, 600, 700],
        'wallet_cohorts_whales_total_volume_mean_10d_period_3': [300, 400, 500, 600],
        'wallet_cohorts_whales_total_volume_ema_3_last': [1100, 2100, 3100, 4100],
        'wallet_cohorts_whales_total_volume_ema_3_last_7d_period_1': [1050, 2050, 3050, 4050],
        'wallet_cohorts_whales_total_volume_ema_3_last_7d_period_2': [1000, 2000, 3000, 4000],
        'wallet_cohorts_whales_total_volume_ema_3_last_7d_period_3': [950, 1950, 2950, 3950],
        'wallet_cohorts_whales_total_volume_ema_14_last': [1200, 2200, 3200, 4200],
        'wallet_cohorts_whales_total_volume_ema_14_last_7d_period_1': [1150, 2150, 3150, 4150],
        'wallet_cohorts_whales_total_volume_ema_14_last_7d_period_2': [1100, 2100, 3100, 4100],
        'wallet_cohorts_whales_total_volume_ema_14_last_7d_period_3': [1050, 2050, 3050, 4050],
        # time_series.market_data.price
        'time_series_market_data_price_std': [50, 60, 70, 80],
        # time_series.market_data.volume
        'time_series_market_data_volume_sum': [5000, 6000, 7000, 8000],
        'time_series_market_data_volume_std': [500, 600, 700, 800],
        # time_series.market_data.market_cap
        'time_series_market_data_market_cap_last': [100000, 200000, 300000, 400000],
        'time_series_market_data_market_cap_ema_3_last': [110000, 210000, 310000, 410000],
        'time_series_market_data_market_cap_ema_3_sum_10d_period_1': [105000, 205000, 305000, 405000],
        'time_series_market_data_market_cap_ema_3_sum_10d_period_2': [100000, 200000, 300000, 400000],
        'time_series_market_data_market_cap_ema_14_last': [120000, 220000, 320000, 420000],
        'time_series_market_data_market_cap_ema_14_sum_10d_period_1': [115000, 215000, 315000, 415000],
        'time_series_market_data_market_cap_ema_14_sum_10d_period_2': [110000, 210000, 310000, 410000],
        'time_series_market_data_market_cap_rsi_14_last': [30, 40, 50, 60],
    }
    df = pd.DataFrame(data, index=index)
    return df

@pytest.mark.unit
def test_scaling_processor_with_double_complex_metrics(
    double_complex_metrics_config, dummy_double_complex_dataframe
):
    """
    Test the ScalingProcessor class with a double complex metrics configuration incorporating
    multiple levels and nested configurations.
    """
    # Instantiate the ScalingProcessor with the provided double_complex_metrics_config
    processor = prp.ScalingProcessor(double_complex_metrics_config)

    # Expected column_scaling_map based on the double_complex_metrics_config
    expected_column_scaling_map = {
        # wallet_cohorts.whales.total_volume
        'wallet_cohorts_whales_total_volume_last': 'log',
        'wallet_cohorts_whales_total_volume_mean_10d_period_1': 'log',
        'wallet_cohorts_whales_total_volume_mean_10d_period_2': 'log',
        'wallet_cohorts_whales_total_volume_mean_10d_period_3': 'log',
        'wallet_cohorts_whales_total_volume_ema_3_last': 'none',
        'wallet_cohorts_whales_total_volume_ema_3_last_7d_period_1': 'standard',
        'wallet_cohorts_whales_total_volume_ema_3_last_7d_period_2': 'standard',
        'wallet_cohorts_whales_total_volume_ema_3_last_7d_period_3': 'standard',
        'wallet_cohorts_whales_total_volume_ema_14_last': 'none',
        'wallet_cohorts_whales_total_volume_ema_14_last_7d_period_1': 'standard',
        'wallet_cohorts_whales_total_volume_ema_14_last_7d_period_2': 'standard',
        'wallet_cohorts_whales_total_volume_ema_14_last_7d_period_3': 'standard',
        # time_series.market_data.price
        'time_series_market_data_price_std': 'none',
        # time_series.market_data.volume
        'time_series_market_data_volume_sum': 'standard',
        'time_series_market_data_volume_std': 'none',
        # time_series.market_data.market_cap
        'time_series_market_data_market_cap_last': 'log',
        'time_series_market_data_market_cap_ema_3_last': 'log',
        'time_series_market_data_market_cap_ema_3_sum_10d_period_1': 'log',
        'time_series_market_data_market_cap_ema_3_sum_10d_period_2': 'log',
        'time_series_market_data_market_cap_ema_14_last': 'log',
        'time_series_market_data_market_cap_ema_14_sum_10d_period_1': 'log',
        'time_series_market_data_market_cap_ema_14_sum_10d_period_2': 'log',
        'time_series_market_data_market_cap_rsi_14_last': 'none',
    }

    # Assert that the column_scaling_map is as expected
    assert processor.column_scaling_map == expected_column_scaling_map, (
        "Column scaling map does not match expected mapping."
    )

    # Apply scaling to the dummy_double_complex_dataframe (as training data)
    scaled_df = processor.apply_scaling(dummy_double_complex_dataframe, is_train=True)

    # Prepare expected scaled values for each column
    columns_to_test = expected_column_scaling_map.keys()

    for column in columns_to_test:
        scaling_method = expected_column_scaling_map[column]
        original_values = dummy_double_complex_dataframe[column].values.reshape(-1, 1)

        if scaling_method == 'log':
            # Logical steps:
            # - Apply np.log1p to the original values
            expected_values = np.log1p(original_values).flatten()
        elif scaling_method == 'standard':
            # Logical steps:
            # - Calculate mean and std, standardize the values
            mean = original_values.mean()
            std = original_values.std()
            expected_values = ((original_values - mean) / std).flatten()
        elif scaling_method == 'none':
            # Logical steps:
            # - Values remain the same as original
            expected_values = original_values.flatten()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")

        # Compare the scaled values in scaled_df to the expected values
        np.testing.assert_allclose(
            scaled_df[column].values,
            expected_values,
            atol=1e-4,
            err_msg=(
                f"Scaled values for '{column}' do not match expected values with scaling '{scaling_method}'."
            )
        )

@pytest.fixture
def scaling_metrics_config():
    """
    Fixture providing a metrics configuration with different scaling methods for testing.
    """
    return {
        'feature1': {
            'aggregations': {
                'sum': {
                    'scaling': 'standard'
                }
            }
        },
        'feature2': {
            'aggregations': {
                'last': {
                    'scaling': 'minmax'
                }
            }
        }
    }

@pytest.fixture
def training_dataset():
    """
    Fixture providing a training dataset with known values.
    """
    data = {
        'feature1_sum': [1, 2, 3, 4],
        'feature2_last': [10, 20, 30, 40]
    }
    df = pd.DataFrame(data)
    return df

@pytest.fixture
def test_dataset():
    """
    Fixture providing a test dataset with different values.
    """
    data = {
        'feature1_sum': [5, 6, 7, 8],
        'feature2_last': [15, 25, 35, 45]
    }
    df = pd.DataFrame(data)
    return df

@pytest.fixture
def minimal_config():
    """
    Fixture providing a minimal configuration dictionary required by DataPreprocessor.
    """
    return {
        'datasets': {}
    }

@pytest.fixture
def minimal_modeling_config():
    """
    Fixture providing a minimal modeling configuration required by DataPreprocessor.
    """
    return {
        'preprocessing': {}
    }

@pytest.mark.unit
def test_data_preprocessor_scaling_consistency(
    scaling_metrics_config, training_dataset, test_dataset,
    minimal_config, minimal_modeling_config
):
    """
    Test that the scaling parameters learned from the training set are applied to the test set
    without re-fitting the scalers.
    """
    # Instantiate the DataPreprocessor
    preprocessor = prp.DataPreprocessor(
        config=minimal_config,
        metrics_config=scaling_metrics_config,
        modeling_config=minimal_modeling_config
    )

    # Prepare datasets
    datasets = {
        'train': training_dataset,
        'test': test_dataset
    }

    # Preprocess the datasets
    preprocessed_datasets = preprocessor.preprocess(datasets)

    # Extract the processed training and test sets
    preprocessed_train = preprocessed_datasets['train']
    preprocessed_test = preprocessed_datasets['test']

    # Extract scalers from the ScalingProcessor after processing training data
    # For feature1_sum (standard scaling), get the mean and std from the scaler
    feature1_column = 'feature1_sum'
    feature2_column = 'feature2_last'

    # Check that the scaler for feature1_sum is StandardScaler and has been fitted
    assert feature1_column in preprocessor.scaler.scalers, "Scaler for feature1_sum not found."
    scaler_feature1 = preprocessor.scaler.scalers[feature1_column]
    assert isinstance(scaler_feature1, StandardScaler), (
        "Scaler for feature1_sum is not StandardScaler."
    )

    # Similarly for feature2_last (minmax scaling)
    assert feature2_column in preprocessor.scaler.scalers, "Scaler for feature2_last not found."
    scaler_feature2 = preprocessor.scaler.scalers[feature2_column]
    assert isinstance(scaler_feature2, MinMaxScaler), (
        "Scaler for feature2_last is not MinMaxScaler."
    )

    # Get the parameters from the scalers
    # For StandardScaler, mean_ and scale_
    mean_feature1 = scaler_feature1.mean_[0]
    std_feature1 = scaler_feature1.scale_[0]

    # For MinMaxScaler, data_min_ and data_max_
    min_feature2 = scaler_feature2.data_min_[0]
    max_feature2 = scaler_feature2.data_max_[0]

    # Manually scale the test data using these parameters
    # For feature1_sum
    original_test_feature1 = test_dataset[feature1_column].values
    expected_scaled_feature1 = (original_test_feature1 - mean_feature1) / std_feature1

    # For feature2_last
    original_test_feature2 = test_dataset[feature2_column].values
    expected_scaled_feature2 = (original_test_feature2 - min_feature2) / (max_feature2 - min_feature2)

    # Compare the scaled test data from preprocessed_test with expected scaled values
    np.testing.assert_allclose(
        preprocessed_test[feature1_column].values,
        expected_scaled_feature1,
        atol=1e-6,
        err_msg="Scaled feature1_sum in test set does not match expected scaled values."
    )

    np.testing.assert_allclose(
        preprocessed_test[feature2_column].values,
        expected_scaled_feature2,
        atol=1e-6,
        err_msg="Scaled feature2_last in test set does not match expected scaled values."
    )
