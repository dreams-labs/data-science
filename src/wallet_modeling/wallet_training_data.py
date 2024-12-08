"""
Primary sequence functions used as part of the wallet modeling pipeline
"""

import logging
from datetime import datetime,timedelta
import pandas as pd
import numpy as np
import pandas_gbq

# Local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()


def generate_imputation_dates():
    """
    Generates a list of all dates that need imputation, including the first
    and last date of each training window and modeling period.

    The training period boundary dates are returned separateely so they can be used
    to

    Returns:
    - imputation_dates (list): list that includes all start and end dates
    """
    # Extract all window start dates
    window_start_dates = sorted([datetime.strptime(date, "%Y-%m-%d")
                    for date in wallets_config['training_data']['training_window_starts'].values()
                    ])

    # Generate the output array
    imputation_dates = [window_start_dates[0].strftime("%Y-%m-%d")]  # Include the first date
    for i in range(1, len(window_start_dates)):
        # Append the day before each window start
        imputation_dates.append((window_start_dates[i] - timedelta(days=1)).strftime("%Y-%m-%d"))
        # Append window start date
        imputation_dates.append(window_start_dates[i].strftime("%Y-%m-%d"))

    # Append day before modeling period
    modeling_start = datetime.strptime(wallets_config['training_data']['modeling_period_start'], "%Y-%m-%d")
    imputation_dates.append((modeling_start - timedelta(days=1)).strftime("%Y-%m-%d"))

    # Append modeling period dates
    imputation_dates.append(wallets_config['training_data']['modeling_period_start'])
    imputation_dates.append(wallets_config['training_data']['modeling_period_end'])

    return imputation_dates


def add_cash_flow_transfers_logic(profits_df):
    """
    Adds a cash_flow_transfers column to profits_df that can be used to compute
    the wallet's gain and investment amount by converting their starting and ending
    balances to cash flow equivilants.

    Params:
    - profits_df (df): profits_df that needs wallet investment peformance computed
        based on the earliest and latest dates in the df.

    Returns:
    - adj_profits_df (df): input df with the cash_flow_transfers column added
    """

    def adjust_end_transfers(df, target_date):
        df.loc[df['date'] == target_date, 'cash_flow_transfers'] -= df.loc[df['date'] == target_date, 'usd_balance']
        return df

    def adjust_start_transfers(df, target_date):
        df.loc[df['date'] == target_date, 'cash_flow_transfers'] = df.loc[df['date'] == target_date, 'usd_balance']
        return df

    # Copy df and add cash flow column
    adj_profits_df = profits_df.copy()
    adj_profits_df['cash_flow_transfers'] = adj_profits_df['usd_net_transfers']

    # Modify the records on the start and end dates to reflect the balances
    start_date = adj_profits_df['date'].min()
    end_date = adj_profits_df['date'].max()

    adj_profits_df = adjust_start_transfers(adj_profits_df,start_date)
    adj_profits_df = adjust_end_transfers(adj_profits_df,end_date)

    return adj_profits_df



def apply_wallet_thresholds(wallet_metrics_df):
    """
    Applies data cleaning filters to the a df keyed on wallet_address

    Params:
    - wallet_metrics_df (df): dataframe with index wallet_address and columns with
        metrics that the filters will be applied to
    """
    # inflows_filter flagged wallets
    excess_inflows_wallets = wallet_metrics_df[
        wallet_metrics_df['invested']>=wallets_config['data_cleaning']['inflows_filter']
        ].index.values

    # profitability_filter flagged wallets
    excess_profits_wallets = wallet_metrics_df[
        wallet_metrics_df['net_gain']>=wallets_config['data_cleaning']['profitability_filter']
        ].index.values

    # minimum_wallet_inflows flagged wallets
    low_inflows_wallets = wallet_metrics_df[
        wallet_metrics_df['invested']<wallets_config['data_cleaning']['minimum_wallet_inflows']
        ].index.values

    # minimum_volume flagged wallets
    low_volume_wallets = wallet_metrics_df[
        wallet_metrics_df['total_volume']<wallets_config['data_cleaning']['minimum_volume']
        ].index.values

    # minumum_coins_traded flagged wallets
    low_coins_traded_wallets = wallet_metrics_df[
        wallet_metrics_df['unique_coins_traded']<wallets_config['data_cleaning']['minumum_coins_traded']
        ].index.values

    # combine all exclusion lists and apply them
    wallets_to_exclude = np.unique(
        np.concatenate([excess_inflows_wallets,excess_profits_wallets,low_inflows_wallets,
                        low_volume_wallets,low_coins_traded_wallets])
    )
    filtered_wallet_metrics_df = wallet_metrics_df[
        ~wallet_metrics_df.index.isin(wallets_to_exclude)
    ]
    logger.info("Retained %s wallets after filtering %s unique wallets."
                "(%s low volume, %s low inflows, %s too few coins traded, "
                "%s excess inflows, %s excess profits)",
                len(filtered_wallet_metrics_df), len(wallets_to_exclude),
                len(low_volume_wallets), len(low_inflows_wallets), len(low_coins_traded_wallets),
                len(excess_inflows_wallets), len(excess_profits_wallets))

    return filtered_wallet_metrics_df



def split_window_dfs(windows_profits_df):
    """
    Splits the full profits_df into separate dfs for each training window and the modeling period.

    Params:
    - windows_profits_df (df): dataframe containing profits data for all periods, with imputed rows
        for each period start and end

    Returns:
    - training_windows_dfs (list of dfs): list of profits_dfs for each training window
    - modeling_period_df (df): profits_df for the modeling period only
    """
    # Extract modeling period boundaries
    modeling_period_start = datetime.strptime(wallets_config['training_data']['modeling_period_start'], "%Y-%m-%d")
    modeling_period_end = datetime.strptime(wallets_config['training_data']['modeling_period_end'], "%Y-%m-%d")

    # Convert training window starts to sorted datetime
    training_windows_starts = sorted([
        datetime.strptime(date, "%Y-%m-%d") - pd.Timedelta(days=1)
        for date in wallets_config['training_data']['training_window_starts'].values()
    ])

    # Generate end dates for each period
    training_windows_ends = (
        # the dates before each window starts (excluding the first)...
        [date - timedelta(days=1) for date in training_windows_starts[1:]]
        # ...plus the date before the modeling period starts
        + [modeling_period_start - pd.Timedelta(days=1)]
    )

    # Create array of DataFrames for each training period
    training_windows_dfs = []
    for start, end in zip(training_windows_starts, training_windows_ends):
        window_df = windows_profits_df[
            (windows_profits_df['date'] >= start) & (windows_profits_df['date'] < end)
        ]
        training_windows_dfs.append(window_df)

    # Result: array of DataFrames
    for i, df in enumerate(training_windows_dfs):
        logger.info("Training Window %s (%s to %s): %s",
                    i + 1,
                    training_windows_starts[i].strftime('%Y-%m-%d'),
                    training_windows_ends[i].strftime('%Y-%m-%d'),
                    df.shape)

    # Extract modeling period DataFrame
    modeling_period_df = windows_profits_df[
            (windows_profits_df['date'] >= modeling_period_start) & (windows_profits_df['date'] <= modeling_period_end)

        ]
    logger.info("Modeling Period (%s to %s): %s",
                modeling_period_start.strftime('%Y-%m-%d'),
                modeling_period_end.strftime('%Y-%m-%d'),
                modeling_period_df.shape)

    return training_windows_dfs, modeling_period_df



def upload_wallet_cohort(wallet_cohort):
    """
    Uploads the list of wallet_ids that are used in the model to BigQuery. This
    is used to pull additional metrics while limiting results to only relevant wallets.

    Params:
    - wallet_cohort (np.array): the wallet_ids included in the cohort
    """
    # generate upload_df from input df
    upload_df = pd.DataFrame()
    upload_df['wallet_id'] = wallet_cohort
    upload_df['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # set df datatypes of upload df
    dtype_mapping = {
        'wallet_id': int,
        'updated_at': 'datetime64[ns, UTC]'
    }
    upload_df = upload_df.astype(dtype_mapping)

    # upload df to bigquery
    project_id = 'western-verve-411004'
    table_name = 'temp.wallet_modeling_cohort'
    schema = [
        {'name':'wallet_id', 'type': 'int64'},
        {'name':'updated_at', 'type': 'datetime'}
    ]
    pandas_gbq.to_gbq(
        upload_df
        ,table_name
        ,project_id=project_id
        ,if_exists='replace'
        ,table_schema=schema
        ,progress_bar=False
    )
    logger.info('Uploaded cohort of %s wallets to temp.wallet_modeling_cohort.', len(upload_df))
