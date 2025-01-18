"""
Primary sequence functions used as part of the wallet modeling pipeline
"""

import logging
from datetime import datetime, timedelta
from typing import List
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import pandas_gbq
from google.cloud import bigquery
from dreams_core import core as dc

# Local module imports
import training_data.data_retrieval as dr
import wallet_features.market_cap_features as wmc
from wallet_modeling.wallets_config_manager import WalletsConfig
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()




# -----------------------------------
#      Training Data Preparation
# -----------------------------------

@u.timing_decorator
def retrieve_raw_datasets(period_start_date, period_end_date):
    """
    Retrieves raw market and profits data after applying the min_wallet_inflows filter at the
    wallet level. Because the filter is applied cumulatively, later period_end_dates will return
    additional wallets whose newly included inflows put them over the threshold.

    Params:
    - period_start_date,period_end_date (YYYY-MM-DD): The data period boundary dates.

    Returns:
    - profits_df, market_data_df: raw dataframes
    """
    # Identify the date we need ending balances from
    period_start_date = datetime.strptime(period_start_date,'%Y-%m-%d')
    starting_balance_date = period_start_date - timedelta(days=1)

    # Retrieve both datasets
    with ThreadPoolExecutor(max_workers=2) as executor:
        profits_future = executor.submit(
            dr.retrieve_profits_data,
            starting_balance_date,
            period_end_date,
            wallets_config['data_cleaning']['min_wallet_inflows'],
            wallets_config['training_data']['dataset']
        )
        market_future = executor.submit(dr.retrieve_market_data,
                                        wallets_config['training_data']['dataset'])

        profits_df = profits_future.result()
        market_data_df = market_future.result()

    return profits_df, market_data_df


def clean_market_dataset(market_data_df, profits_df, period_start_date, period_end_date, coin_cohort=None):
    """
    Cleans and filters market data.

    Params:
    - market_data_df (DataFrame): Raw market data
    - profits_df (DataFrame): Profits data for coin filtering
    - period_start_date,period_end_date: Period boundary dates
    - coin_cohort (set, optional): Coin IDs to filter data to, rather than using cleaning params


    Returns:
    - DataFrame: Cleaned market data
    """
    # Remove all records after the training period end to ensure no data leakage
    market_data_df = market_data_df[market_data_df['date']<=period_end_date]

    if not coin_cohort:
        # Clean market_data_df if there isn't an existing coin cohort
        market_data_df = market_data_df[market_data_df['coin_id'].isin(profits_df['coin_id'])]
        market_data_df = dr.clean_market_data(
            market_data_df,
            wallets_config,
            period_start_date,
            period_end_date
        )
    else:
        # Otherwise just filter to that cohort
        market_data_df = market_data_df[market_data_df['coin_id'].isin(coin_cohort)]


    # Intelligently impute market cap data in market_data_df when good data is available
    market_data_df = dr.impute_market_cap(market_data_df,
                                        wallets_config['data_cleaning']['min_mc_imputation_coverage'],
                                        wallets_config['data_cleaning']['max_mc_imputation_multiple'])

    # Crudely fill all remaining gaps in market cap data
    market_data_df = wmc.force_fill_market_cap(market_data_df)

    if not coin_cohort:
        # Apply initial market cap threshold filter if no coin cohort was passed
        max_initial_market_cap = wallets_config['data_cleaning']['max_initial_market_cap']
        above_initial_threshold_coins = market_data_df[
            (market_data_df['date']==wallets_config['training_data']['training_period_start'])
            & (market_data_df['market_cap_filled']>max_initial_market_cap)
        ]['coin_id']
        market_data_df = market_data_df[~market_data_df['coin_id'].isin(above_initial_threshold_coins)]
        logger.info("Removed data for %s coins with a market cap above $%s at the start of the training period."
                    ,len(above_initial_threshold_coins),dc.human_format(max_initial_market_cap))
    else:
        logger.info("Returned market data for the %s coins in the coin cohort passed as a parameter.",
                    len(coin_cohort))

    return market_data_df


def format_and_save_datasets(profits_df, market_data_df, period_start_date, parquet_prefix=None):
    """
    Formats and optionally saves the final datasets.

    Params:
    - profits_df, market_data_df (DataFrames): Input dataframes
    - starting_balance_date (datetime): Balance imputation date
    - parquet_prefix,parquet_folder (str): Save location params

    Returns:
    - tuple or None: (profits_df, market_data_df) if no save location specified
    """
    # Adjust all records on the starting_balance_date to be imputed with $0 transfers
    columns_to_update = ['is_imputed', 'usd_net_transfers', 'usd_inflows']
    new_values = [True, 0, 0]

    # Apply the updates
    # Training balance date (1 day before period start)
    period_start_date = datetime.strptime(period_start_date, "%Y-%m-%d")
    starting_balance_date = (period_start_date - timedelta(days=1)).strftime("%Y-%m-%d")

    mask = profits_df['date'] == starting_balance_date
    profits_df.loc[mask, columns_to_update] = new_values

    # Clean profits_df
    profits_df, _ = dr.clean_profits_df(profits_df, wallets_config['data_cleaning'])

    # Round relevant columns
    columns_to_round = [
        'profits_cumulative'
        ,'usd_balance'
        ,'usd_net_transfers'
        ,'usd_inflows'
        ,'usd_inflows_cumulative'
    ]
    profits_df.loc[:, columns_to_round] = (profits_df[columns_to_round]
                                        .round(2)
                                        .replace(-0, 0))

    # Remove rows with a rounded 0 balance and 0 transfers
    profits_df = profits_df[
        ~((profits_df['usd_balance'] == 0) &
        (profits_df['usd_net_transfers'] == 0))
    ]

    # If a parquet file location is specified, store the files and return None
    if parquet_prefix:
        # Store profits
        parquet_folder = wallets_config['training_data']['parquet_folder']
        profits_file = f"{parquet_folder}/{parquet_prefix}_profits_df_full.parquet"
        profits_df.to_parquet(profits_file,index=False)
        logger.info(f"Stored profits_df with shape {profits_df.shape} to {profits_file}.")

        # Store market data
        market_data_file = f"{parquet_folder}/{parquet_prefix}_market_data_df_full.parquet"
        market_data_df.to_parquet(market_data_file,index=False)
        logger.info(f"Stored market_data_df with shape {market_data_df.shape} to {market_data_file}.")
        return None, None

    return profits_df, market_data_df


def generate_training_window_imputation_dates() -> List[datetime]:
    """
    Generates a list of all dates that need imputation. Each period needs:
    1. an imputed row as of the balance end date
    2. an imputed row as of the period end date

    Because the period end date for window 1 is the same as the balance end date for window 2,
    we only need to impute one new row per window.

    The training_period_start is not included because it's already imputed in the base df.

    Returns:
    - imputation_dates (List[datetime]): list of dates that need imputation
    """
    # Make a list of the starting balance dates for all windows
    starting_balance_dates: List[datetime] = sorted([
        datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)
        for date in wallets_config['training_data']['training_window_starts']
    ])

    # Don't include the first value since training_period_start is already imputed
    imputation_dates: List[datetime] = starting_balance_dates[1:]

    # Add the end date for the final window
    final_window_end_date: datetime = datetime.strptime(wallets_config['training_data']['training_period_end'],
                                                        "%Y-%m-%d")
    imputation_dates += [final_window_end_date]

    return imputation_dates



def split_training_window_dfs(training_profits_df):
    """
    Splits the full profits_df into separate dfs for each training window

    Params:
    - windows_profits_df (df): dataframe containing profits data for the full training period, with imputed rows
        for each period window start and end

    Returns:
    - training_windows_dfs (list of dfs): list of profits_dfs for each training window

    """
    logger.debug("Generating window-specific profits_dfs...")

    # Convert training window starts to sorted datetime
    training_windows_starts = sorted([
        datetime.strptime(date, "%Y-%m-%d")
        for date in wallets_config['training_data']['training_window_starts']
    ])

    # Generate end dates for each period
    training_windows_ends = (
        [date - timedelta(days=1) for date in training_windows_starts[1:]]
        + [datetime.strptime(wallets_config['training_data']['training_period_end'], "%Y-%m-%d")]
    )

    # Generate starting balance dates for each period
    training_windows_starting_balance_dates = (
        [training_windows_starts[0] - timedelta(days=1)] # The day before the first window start
        + training_windows_ends[:-1] # the end date of one window is the starting balance date of the next
    )

    # Create array of DataFrames for each training period
    training_windows_profits_dfs = []
    for start_bal_date, end_date in zip(training_windows_starting_balance_dates, training_windows_ends):

        # Filter to between the starting balance date and end date
        window_df = training_profits_df[
            (training_profits_df['date'] >= start_bal_date) & (training_profits_df['date'] <= end_date)
        ]

        # Override records on the starting balance date to remove transfers
        mask = window_df['date'] == start_bal_date
        columns_to_update = ['usd_net_transfers', 'usd_inflows', 'is_imputed']
        values_to_set = [0, 0, True]
        window_df.loc[mask, columns_to_update] = values_to_set

        # Confirm the period boundaries are handled correctly
        start_date = start_bal_date + timedelta(days=1)
        u.assert_period(window_df, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        training_windows_profits_dfs.append(window_df)

    # Confirm that all window dfs' USD transfers add up to the USD transfers in the full training period
    full_sum = training_profits_df['usd_net_transfers'].sum()
    window_sum = sum(df['usd_net_transfers'].sum() for df in training_windows_profits_dfs)
    if not np.isclose(full_sum, window_sum, rtol=1e-5):
        raise ValueError(f"Net transfers in full training period ({full_sum}) do not match combined "
                        f"sum of transfers in windows dfs ({window_sum})")

    # Result: array of DataFrames
    for i, df in enumerate(training_windows_profits_dfs):
        logger.debug("Training Window %s (%s to %s): %s",
                    i + 1,
                    df['date'].min().strftime('%Y-%m-%d'),
                    df['date'].max().strftime('%Y-%m-%d'),
                    df.shape)
    logger.debug("Training Period (%s to %s): %s",
                training_profits_df['date'].min().strftime('%Y-%m-%d'),
                training_profits_df['date'].max().strftime('%Y-%m-%d'),
                training_profits_df.shape)

    return training_windows_profits_dfs




# -----------------------------------
#     Wallet Cohort Management
# -----------------------------------

def apply_wallet_thresholds(wallet_metrics_df):
    """
    Applies data cleaning filters to the a df keyed on wallet_address

    Params:
    - wallet_metrics_df (df): dataframe with index wallet_address and columns with
        metrics that the filters will be applied to

    """
    # Extract thresholds
    min_coins = wallets_config['data_cleaning']['min_coins_traded']
    max_coins = wallets_config['data_cleaning']['max_coins_traded']
    min_wallet_investment = wallets_config['data_cleaning']['min_wallet_investment']
    max_wallet_investment = wallets_config['data_cleaning']['max_wallet_investment']
    min_wallet_volume = wallets_config['data_cleaning']['min_wallet_volume']
    max_wallet_volume = wallets_config['data_cleaning']['max_wallet_volume']
    max_wallet_profits = wallets_config['data_cleaning']['max_wallet_profits']

    if wallets_config['training_data']['hybridize_wallet_ids'] is True:
        if min_coins > 1:
            raise ValueError('Hybrid IDs can only trade up to 1 coin.')

    # filter based on number of coins traded
    low_coins_traded_wallets = wallet_metrics_df[
        wallet_metrics_df['unique_coins_traded'] < min_coins
    ].index.values

    excess_coins_traded_wallets = wallet_metrics_df[
        wallet_metrics_df['unique_coins_traded'] > max_coins
    ].index.values

    # filter based on wallet investment amount
    low_investment_wallets = wallet_metrics_df[
        wallet_metrics_df['max_investment'] < min_wallet_investment
    ].index.values

    excess_investment_wallets = wallet_metrics_df[
        wallet_metrics_df['max_investment'] >= max_wallet_investment
    ].index.values

    # filter based on wallet volume
    low_volume_wallets = wallet_metrics_df[
        wallet_metrics_df['total_volume'] < min_wallet_volume
    ].index.values

    excess_volume_wallets = wallet_metrics_df[
        wallet_metrics_df['total_volume'] > max_wallet_volume
    ].index.values

    # max_wallet_coin_profits flagged wallets
    excess_profits_wallets = wallet_metrics_df[
        abs(wallet_metrics_df['crypto_net_gain']) >= max_wallet_profits
    ].index.values

    # combine all exclusion lists and apply them
    wallets_to_exclude = np.unique(np.concatenate([
        low_coins_traded_wallets, excess_coins_traded_wallets,
        low_investment_wallets, excess_investment_wallets,
        low_volume_wallets, excess_volume_wallets,
        excess_profits_wallets])
    )
    filtered_wallet_metrics_df = wallet_metrics_df[
        ~wallet_metrics_df.index.isin(wallets_to_exclude)
    ]

    logger.info("Retained %s wallets after filtering %s unique wallets:",
                len(filtered_wallet_metrics_df), len(wallets_to_exclude))

    logger.info(" - %s wallets fewer than %s coins traded, %s wallets with more than %s coins traded",
                len(low_coins_traded_wallets), min_coins,
                len(excess_coins_traded_wallets), max_coins)

    logger.info(" - %s wallets' max investment below $%s, %s wallets' max investment balance above $%s",
                len(low_investment_wallets), dc.human_format(min_wallet_investment),
                len(excess_investment_wallets), dc.human_format(max_wallet_investment))

    logger.info(" - %s wallets with volume below $%s, %s wallets with volume above $%s",
                len(low_volume_wallets), dc.human_format(min_wallet_volume),
                len(excess_volume_wallets), dc.human_format(max_wallet_volume))

    logger.info(" - %s wallets with net gain or loss exceeding $%s",
                len(excess_profits_wallets), dc.human_format(max_wallet_profits))

    return filtered_wallet_metrics_df



def upload_training_cohort(cohort_ids: np.array, hybridize_wallet_ids: bool) -> None:
    """
    Uploads the list of wallet_ids that are used in the model to BigQuery. This
    is used to pull additional metrics while limiting results to only relevant wallets.

    Params:
    - cohort_ids (np.array): the wallet_ids included in the cohort
    - hybridize_wallet_ids (bool): whether the IDs are regular wallet_ids or hybrid wallet-coin IDs

    """
    # 1. Generate upload_df from input df
    # -----------------------------------
    upload_df = pd.DataFrame()
    id_col = 'hybrid_id' if hybridize_wallet_ids else 'wallet_id'
    upload_df[id_col] = cohort_ids
    upload_df['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # set df datatypes of upload df
    dtype_mapping = {
        id_col: int,
        'updated_at': 'datetime64[ns, UTC]'
    }
    upload_df = upload_df.astype(dtype_mapping)


    # 2. Upload list of IDs to bigquery
    # ---------------------------------
    project_id = 'western-verve-411004'
    client = bigquery.Client(project=project_id)

    wallet_ids_table = f"{project_id}.temp.wallet_modeling_training_cohort"
    schema = [
        {'name':id_col, 'type': 'int64'},
        {'name':'updated_at', 'type': 'datetime'}
    ]
    pandas_gbq.to_gbq(
        upload_df
        ,wallet_ids_table
        ,project_id=project_id
        ,if_exists='replace'
        ,table_schema=schema
        ,progress_bar=False
    )


    # 3. Create updated table with full wallet_address values
    # -------------------------------------------------------

    if hybridize_wallet_ids is False:
        create_query = f"""
        CREATE OR REPLACE TABLE `{wallet_ids_table}` AS
        SELECT
            t.{id_col},
            w.wallet_address,
            t.updated_at
        FROM `{wallet_ids_table}` t
        LEFT JOIN `reference.wallet_ids` w
            ON t.wallet_id = w.wallet_id
        """
        client.query(create_query).result()
        logger.info('Uploaded cohort of %s wallets with addresses to %s.',
                    len(cohort_ids), wallet_ids_table)

    else:
        create_query = f"""
        CREATE OR REPLACE TABLE `{wallet_ids_table}` AS
        SELECT
            c.hybrid_id,
            hm.wallet_id,
            hm.wallet_address,
            hm.coin_id,
            c.updated_at
        FROM `{wallet_ids_table}` c
        JOIN `temp.wallet_modeling_hybrid_id_mapping` hm
            ON hm.hybrid_id = c.hybrid_id
        """
        client.query(create_query).result()
        logger.info('Uploaded cohort of %s hybrid_ids %s.',
                    len(cohort_ids), wallet_ids_table)
