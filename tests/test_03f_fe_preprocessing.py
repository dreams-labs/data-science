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
# preprocess_coin_df() unit tests
# ------------------------------------------ #


@pytest.fixture
def sample_metrics_config():
    """basic sample config for initial unit tests"""
    return {
        "metric1": {
            "scaling": "standard"
        },
        "metric2": {
            "scaling": "minmax"
        },
        "metric3": {
            "scaling": "log"
        },
        "metric4": {
            "scaling": "none"
        },
        "nested_metric": {
            "aggregations": {
                "sum": {"scaling": "standard"}
            }
        }
    }

@pytest.fixture
def sample_dataframe():
    """simple df for initial unit tests"""
    return pd.DataFrame({
        "metric1": [1, 2, 3, 4, 5],
        "metric2": [10, 20, 30, 40, 50],
        "metric3": [1, 10, 100, 1000, 10000],
        "metric4": [1, 2, 3, 4, 5],
        "nested_metric_sum": [5, 10, 15, 20, 25]
    })

def test_basic_create_column_scaling_map(sample_metrics_config):
    """Confirms basic functionality of _create_column_scaling_map() method"""
    processor = prp.ScalingProcessor(sample_metrics_config)
    expected_map = {
        "metric1": "standard",
        "metric2": "minmax",
        "metric3": "log",
        "metric4": "none",
        "nested_metric_sum": "standard"
    }
    assert processor.column_scaling_map == expected_map

def test_basic_apply_scaling():
    sample_metrics_config = {
        "metric1": {"scaling": "standard"},
        "metric2": {"scaling": "minmax"},
        "metric3": {"scaling": "log"},
        "metric4": {"scaling": "none"}
    }

    sample_df = pd.DataFrame({
        "metric1": [1, 2, 3, 4, 5],
        "metric2": [10, 20, 30, 40, 50],
        "metric3": [1, 10, 100, 1000, 10000],
        "metric4": [1, 2, 3, 4, 5]
    })

    processor = prp.ScalingProcessor(sample_metrics_config)
    scaled_df = processor.apply_scaling(sample_df, is_train=True)

    # Check if scalers are created and stored
    assert "metric1" in processor.scalers
    assert "metric2" in processor.scalers
    assert "metric3" not in processor.scalers  # log scaling doesn't use a stored scaler
    assert "metric4" not in processor.scalers

    # Check if scaling is applied correctly
    np.testing.assert_allclose(
        scaled_df["metric1"].values,
        StandardScaler().fit_transform(sample_df[["metric1"]]).flatten(),
        rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        scaled_df["metric2"].values,
        MinMaxScaler().fit_transform(sample_df[["metric2"]]).flatten(),
        rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        scaled_df["metric3"].values,
        np.log1p(sample_df["metric3"]),
        rtol=1e-7, atol=1e-7
    )
    np.testing.assert_array_equal(scaled_df["metric4"].values, sample_df["metric4"].values)

    print("All tests passed!")
