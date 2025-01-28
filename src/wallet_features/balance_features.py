import logging
import pandas as pd

# Local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig
import utils as u

# set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()



# ------------------------------
#         Core Interface
# ------------------------------

@u.timing_decorator
def calculate_balance_features(
    profits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate wallet-level distribution features, given a profits_df with a multiindex of
    (coin_id, wallet_address, date). The last row (largest date) is used to represent
    each (coin_id, wallet_address)'s final balance.

    Params:
    - profits_df (DataFrame): MultiIndex on (coin_id, wallet_address, date). Must include
      'usd_balance'. Rows are sorted by date, or at least groupable by date.
    - balance_features_min_balance (float): Filter out wallets whose total_usd_balance
      is <= this threshold before aggregating features.

    Returns:
    - distribution_features_df (DataFrame): Wallet-indexed distribution features:
        - concentration_index (HHI)
        - largest_coin_frac
        - largest_coin_usd
        - total_usd_balance
        - n_coins
        - coin_fraction_std
        - min_coin_frac
        - median_coin_frac
    """
    profits_df = u.ensure_index(profits_df)
    balance_features_min_balance = wallets_config['features']['balance_features_min_balance']

    # Extract the final row per (coin_id, wallet_address) group
    profits_df_end = (
        profits_df
        .groupby(level=["coin_id", "wallet_address"], group_keys=False)
        .tail(1)
        .copy()
    )

    # Now we only need wallet-level grouping, so drop coin_id, date from index
    # Keep wallet_address as the index or as a column—whichever you prefer
    profits_df_end.reset_index(["coin_id", "date"], drop=True, inplace=True)

    # Calculate distribution features on these final balances
    distribution_features_df = calculate_distribution_features(
        profits_df_end,
        balance_features_min_balance=balance_features_min_balance
    )

    return distribution_features_df




# ------------------------------
#         Helper Functions
# ------------------------------

def calculate_distribution_features(
    profits_df_end: pd.DataFrame,
    balance_features_min_balance: float
) -> pd.DataFrame:
    """
    Build wallet-level distribution metrics from final balances, skipping wallets whose
    total balance is <= 'balance_features_min_balance'.

    Expects 'profits_df_end' to have:
      - Index or column: 'wallet_address'
      - Column: 'usd_balance'
      - One row per wallet-coin representing end-of-period balances.

    Returns:
    - features_df (DataFrame): Wallet-level features with index=wallet_address:
        - concentration_index (HHI)
        - largest_coin_frac
        - largest_coin_usd
        - total_usd_balance
        - n_coins
        - coin_fraction_std
        - min_coin_frac
        - median_coin_frac
    """
    profits_df_end = profits_df_end.copy()

    # Total USD balance per wallet
    profits_df_end["total_usd_balance"] = profits_df_end.groupby(
        "wallet_address", observed=True
    )["usd_balance"].transform("sum")

    # Filter out wallets below the min balance threshold
    profits_df_end = profits_df_end[profits_df_end["total_usd_balance"] > balance_features_min_balance]

    # Fraction of total balance for each coin
    profits_df_end["coin_fraction"] = (
        profits_df_end["usd_balance"] / profits_df_end["total_usd_balance"]
    )

    # Squared fraction for Herfindahl-Hirschman Index
    profits_df_end["coin_fraction_sq"] = profits_df_end["coin_fraction"] ** 2

    # Group to wallet level
    features_df = profits_df_end.groupby("wallet_address", observed=True).agg(
        concentration_index=("coin_fraction_sq", "sum"),
        largest_coin_frac=("coin_fraction", "max"),
        largest_coin_usd=("usd_balance", "max"),
        total_usd_balance=("usd_balance", "sum"),
        n_coins=("usd_balance", "count"),
        coin_fraction_std=("coin_fraction", "std"),
        min_coin_frac=("coin_fraction", "min"),
        median_coin_frac=("coin_fraction", "median")
    )

    return features_df