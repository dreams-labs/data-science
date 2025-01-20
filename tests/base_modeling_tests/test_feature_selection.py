# pylint: disable=wrong-import-position  # import not at top of doc (due to local import)
# pylint: disable=redefined-outer-name  # redefining from outer scope triggering on pytest fixtures
# pylint: disable=unused-argument
# pyright: reportMissingModuleSource=false

import sys
from pathlib import Path
from dotenv import load_dotenv
import pytest
from dreams_core import core as dc

# pyright: reportMissingImports=false
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
import base_modeling.feature_selection as fs
from wallet_modeling.wallets_config_manager import WalletsConfig

load_dotenv()
logger = dc.setup_logger()

config_path = Path(__file__).parent.parent / 'test_config' / 'test_wallets_config.yaml'
wallets_config = WalletsConfig.load_from_yaml(config_path)


# ===================================================== #
#                                                       #
#                 U N I T   T E S T S                   #
#                                                       #
# ===================================================== #


# ------------------------------------------------ #
# identify_matching_columns() unit tests
# ------------------------------------------------ #
# pylint:disable=line-too-long

@pytest.fixture
def mock_columns():
    """
    Fixture providing a sample list of 15 column names for testing.
    """
    return [
        "training_clusters|k2_cluster/cluster_0|balances/usd_balance_241120|aggregations/sum",
        "training_clusters|k2_cluster/cluster_0|balances/usd_balance_241120|aggregations/count",
        "training_clusters|k2_cluster/cluster_0|balances/usd_balance_241120|aggregations/sum_pct",
        "training_clusters|k2_cluster/cluster_0|balances/usd_balance_241120|aggregations/count_pct",
        "training_clusters|k2_cluster/cluster_0|balances/usd_balance_241120|score_wtd/net_gain_winsorized_checkpoint_score",
        "training_clusters|k2_cluster/cluster_0|balances/usd_balance_241120|score_wtd/net_gain_winsorized_checkpoint_residual",
        "training_clusters|k2_cluster/cluster_0|balances/usd_balance_241120|score_dist/net_gain_winsorized_checkpoint_score_median",
        "training_clusters|k2_cluster/cluster_0|balances/usd_balance_241120|score_dist/net_gain_winsorized_checkpoint_residual_median",
        "training_clusters|k2_cluster/cluster_1|balances/usd_balance_241120|aggregations/sum",
        "training_clusters|k2_cluster/cluster_1|balances/usd_balance_241120|aggregations/count",
        "all_wallets|all/all|trading/max_investment|aggregations/sum",
        "all_wallets|all/all|trading/max_investment|score_dist/net_gain_winsorized_checkpoint_score_std",
        "score_quantile|net_gain_winsorized_checkpoint_score/15_50pct|trading/max_investment|aggregations/sum",
        "score_quantile|net_gain_winsorized_checkpoint_score/15_50pct|trading/max_investment|aggregations/count",
        "score_quantile|net_gain_winsorized_checkpoint_score/15_50pct|trading/max_investment|aggregations/sum_pct",
        "score_quantile|net_gain_winsorized_checkpoint_score/15_50pct|trading/max_investment|aggregations/count_pct",
    ]

@pytest.mark.unit
def test_identify_matching_columns_single_wildcard(mock_columns):
    """
    Test that a single wildcard pattern and a full column name match correctly.
    """
    # Define the input patterns
    patterns = [
        "training_clusters|k2_cluster/cluster_0|balances/*|aggregations/*",  # Single wildcard
        "all_wallets|all/all|trading/max_investment|aggregations/sum",        # Full column name
    ]

    # Expected matches
    expected_matches = {
        "training_clusters|k2_cluster/cluster_0|balances/usd_balance_241120|aggregations/sum",
        "training_clusters|k2_cluster/cluster_0|balances/usd_balance_241120|aggregations/count",
        "training_clusters|k2_cluster/cluster_0|balances/usd_balance_241120|aggregations/sum_pct",
        "training_clusters|k2_cluster/cluster_0|balances/usd_balance_241120|aggregations/count_pct",
        "all_wallets|all/all|trading/max_investment|aggregations/sum",
    }

    # Call the function
    matched_columns = fs.identify_matching_columns(patterns, mock_columns)

    # Assert the results
    assert matched_columns == expected_matches, \
        f"Expected {expected_matches}, but got {matched_columns}"


@pytest.mark.unit
def test_identify_matching_columns_double_wildcard(mock_columns):
    """
    Test that patterns with two wildcards (*) match the correct multiple columns.
    """
    # Define the input patterns
    patterns = [
        "training_clusters|k2_cluster/cluster_0|balances/*|aggregations/*",  # Double wildcard
        "score_quantile|*|trading/max_investment|aggregations/*",           # Double wildcard
    ]

    # Expected matches
    expected_matches = {
        # Matches for "training_clusters|k2_cluster/cluster_0|balances/*/aggregations/*"
        "training_clusters|k2_cluster/cluster_0|balances/usd_balance_241120|aggregations/sum",
        "training_clusters|k2_cluster/cluster_0|balances/usd_balance_241120|aggregations/count",
        "training_clusters|k2_cluster/cluster_0|balances/usd_balance_241120|aggregations/sum_pct",
        "training_clusters|k2_cluster/cluster_0|balances/usd_balance_241120|aggregations/count_pct",

        # Matches for "score_quantile|*/trading/max_investment|aggregations/*"
        "score_quantile|net_gain_winsorized_checkpoint_score/15_50pct|trading/max_investment|aggregations/sum",
        "score_quantile|net_gain_winsorized_checkpoint_score/15_50pct|trading/max_investment|aggregations/count",
        "score_quantile|net_gain_winsorized_checkpoint_score/15_50pct|trading/max_investment|aggregations/sum_pct",
        "score_quantile|net_gain_winsorized_checkpoint_score/15_50pct|trading/max_investment|aggregations/count_pct",
    }

    # Call the function
    matched_columns = fs.identify_matching_columns(patterns, mock_columns)

    # Walkthrough of logic:
    # 1. Pattern "training_clusters|k2_cluster/cluster_0|balances/*/aggregations/*":
    #    - Fixed parts are: ["training_clusters|k2_cluster/cluster_0|balances/", "/aggregations/"].
    #    - Matches all columns under "balances/..." that also include "/aggregations/...":
    #      - "training_clusters|k2_cluster/cluster_0|balances/usd_balance_241120|aggregations/sum"
    #      - "training_clusters|k2_cluster/cluster_0|balances/usd_balance_241120|aggregations/count"
    #      - "training_clusters|k2_cluster/cluster_0|balances/usd_balance_241120|aggregations/sum_pct"
    #      - "training_clusters|k2_cluster/cluster_0|balances/usd_balance_241120|aggregations/count_pct".
    # 2. Pattern "score_quantile|*/trading/max_investment|aggregations/*":
    #    - Fixed parts are: ["score_quantile|", "/trading/max_investment|aggregations/"].
    #    - Matches all columns with "score_quantile|..." and ending in "/aggregations/...":
    #      - "score_quantile|net_gain_winsorized_checkpoint_score/15_50pct|trading/max_investment|aggregations/sum"
    #      - "score_quantile|net_gain_winsorized_checkpoint_score/15_50pct|trading/max_investment|aggregations/count"
    #      - "score_quantile|net_gain_winsorized_checkpoint_score/15_50pct|trading/max_investment|aggregations/sum_pct"
    #      - "score_quantile|net_gain_winsorized_checkpoint_score/15_50pct|trading/max_investment|aggregations/count_pct".
    # 3. Combine results:
    #    - Final matched set includes all columns from both patterns.

    # Assert the results
    assert matched_columns == expected_matches, \
        f"Expected {expected_matches}, but got {matched_columns}"


@pytest.mark.unit
def test_identify_matching_columns_no_matches(mock_columns):
    """
    Test that a wildcard pattern with no matching columns correctly returns an empty set.
    """
    # Define the input patterns
    patterns = [
        "nonexistent_prefix|*/balances/*",        # Nonexistent prefix
        "training_clusters|k3_cluster/*/sum",    # Nonexistent cluster
        "all_wallets|all/all|invalid|*",         # Invalid structure
    ]

    # Expected matches
    expected_matches = set()  # No columns should match

    # Call the function
    matched_columns = fs.identify_matching_columns(patterns, mock_columns)

    # Walkthrough of logic:
    # 1. Pattern "nonexistent_prefix|*/balances/*":
    #    - Fixed parts are: ["nonexistent_prefix|", "/balances/"].
    #    - No columns in the fixture have the prefix "nonexistent_prefix|".
    #    - Result: No matches.
    # 2. Pattern "training_clusters|k3_cluster/*/sum":
    #    - Fixed parts are: ["training_clusters|k3_cluster/", "/sum"].
    #    - No columns in the fixture have "k3_cluster".
    #    - Result: No matches.
    # 3. Pattern "all_wallets|all/all|invalid|*":
    #    - Fixed part is: ["all_wallets|all/all|invalid|"].
    #    - No columns in the fixture include "invalid|".
    #    - Result: No matches.
    # 4. Combine results:
    #    - Final result is an empty set.

    # Assert the results
    assert matched_columns == expected_matches, \
        f"Expected {expected_matches}, but got {matched_columns}"
