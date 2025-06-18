import logging
import pandas as pd
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)



# ------------------------------------
#       Main Interface Functions
# ------------------------------------

@u.timing_decorator
def compute_coin_wallet_metrics(
        wallets_coin_config: dict,
        profits_df: pd.DataFrame,
        wallet_training_data_df: pd.DataFrame,
        period_start: str,
        period_end: str
    ) -> pd.DataFrame:
    """
    Compute coin-wallet–level metrics: balances and trading metrics.

    Params:
    - wallets_coin_config (dict): dict from .yaml file
    - profits_df (DataFrame): profits_df for the wallet_modeling_period, meaning
        that the last date falls directly before the coin_modeling_period_start
    - wallet_training_data_df (DataFrame): wallet training features built with data that
        extends up to the coin_modeling_period_start.

    - period_start,period_end str(YYYY-MM-DD): period start and end dates

    Returns:
    - cw_metrics_df (DataFrame): MultiIndex [coin_id, wallet_address] with
        'balances/...’ and 'trading/...’ feature columns.
    """
    # Confirm time period is correct
    u.assert_period(profits_df, period_start, period_end)

    # 1) Build base index of all (coin, wallet) pairs
    idx = (
        profits_df[['coin_id', 'wallet_address']]
        .drop_duplicates()
        .set_index(['coin_id', 'wallet_address'])
        .index
    )
    cw_metrics_df = pd.DataFrame(index=idx)

    # 2) Calculate balances
    balances_df = calculate_coin_wallet_ending_balances(
        profits_df
    ).add_prefix('balances/')
    cw_metrics_df = (
        cw_metrics_df
        .join(balances_df, how='left')
        .fillna({col: 0 for col in balances_df.columns})
    )

    # 3) Extract configured wallet training data features
    wallet_features_df = select_wallet_features(
        wallets_coin_config, wallet_training_data_df
    ).add_prefix('wallet/')
    cw_metrics_df = (
        cw_metrics_df
        .join(wallet_features_df, how='left')
        # leave NaNs for XGB to process rather than filling
    )

    return cw_metrics_df




# ------------------------------
#         Helper Functions
# ------------------------------


@u.timing_decorator
def calculate_coin_wallet_ending_balances(profits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate metric value for each wallet-coin pair on specified dates.

    Params:
    - profits_df (DataFrame): Input profits data

    Returns:
    - balances_df (DataFrame): Wallet-coin level data with metric values for each specified date
        with indices ('coin_id', 'wallet_address').
    """
    # Create a DataFrame with coin_id and wallet_address as indices
    balances_df = profits_df[['coin_id', 'wallet_address']].drop_duplicates().set_index(['coin_id', 'wallet_address'])

    # Starting Balance
    period_starting_balance = profits_df['date'].min()
    starting_balances = profits_df.loc[
        profits_df['date'] == period_starting_balance,
        ['coin_id', 'wallet_address', 'usd_balance']
    ].set_index(['coin_id', 'wallet_address'])

    # Ending Balance
    period_end = profits_df['date'].max()
    ending_balances = profits_df.loc[
        profits_df['date'] == period_end,
        ['coin_id', 'wallet_address', 'usd_balance']
    ].set_index(['coin_id', 'wallet_address'])

    # Add the balance column to balances_df
    balances_df['usd_balance_starting'] = starting_balances['usd_balance']
    balances_df['usd_balance_ending'] = ending_balances['usd_balance']

    return balances_df



def select_wallet_features(
        wallets_coin_config: dict,
        wallet_training_data_df: pd.DataFrame,
    ) -> pd.DataFrame:
    """
    Selects the wallet_training_data_df columns that will be flattened into
     coin-level features. This df comes directly from the wallet model's training data where
     the wallet modeling_period_start matches the current coin_modeling_period_start.

    This dataset needs to be flattened separately from the compute_coin_wallet_metrics()
     output because it includes all wallets included in the multiwindow cohort, whereas
     the latter only includes profits_df from the modeling period directly prior to the
     coin modeling period, meaning that the multiwindow df has a much larger cohort.

    Params:
    - wallets_coin_config (dict): dict from .yaml file
    - wallet_training_data_df (DataFrame): wallet training features for the multiwindow
    period with the modeling_period_start equal to the current coin config's
    coin_modeling_period_start. Aggregated wallet features as of the coin modeling period
    are used as coin-level features.

    Returns:
    - wallet_features_df (df): The same df filtered to only the relevant
        columns, multiindexed on coin_id-wallet_address.
    """
    feature_cols = wallets_coin_config['features']['wallet_features_cols']
    missing_cols = [col for col in feature_cols if col not in wallet_training_data_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in training data: {missing_cols}")

    # Select columns
    wallet_features_df = wallet_training_data_df[feature_cols]

    # Replace characters for coin feature analysis
    wallet_features_df.columns = (wallet_features_df.columns
                                  .str.replace('|', '_')
                                  .str.replace('/', '_'))

    return wallet_features_df
