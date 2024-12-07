"""
Utility functions used as part of the wallet modeling pipeline
"""

from datetime import datetime,timedelta
import pandas as pd
import numpy as np
import pandas_gbq
import dreams_core.core as dc

# set up logger at the module level
logger = dc.setup_logger()


def apply_wallet_thresholds(wallet_metrics_df, wallets_config):
    """
    Applies data cleaning filters to the a df keyed on wallet_address

    Params:
    - wallet_metrics_df (df): dataframe with index wallet_address and columns with
        metrics that the filters will be applied to
    - wallets_config (dict): dict with data cleaning filter levels
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

    # combine all exclusion lists and apply them
    wallets_to_exclude = np.unique(
        np.concatenate([excess_inflows_wallets,excess_profits_wallets,low_inflows_wallets,low_volume_wallets])
    )
    filtered_wallet_metrics_df = wallet_metrics_df[
        ~wallet_metrics_df.index.isin(wallets_to_exclude)
    ]
    logger.info("Filtered %s unique wallets (%s low volume, %s low inflows, %s excess inflows, %s excess profits)",
                len(low_volume_wallets), len(low_inflows_wallets), len(wallets_to_exclude),
                len(excess_inflows_wallets), len(excess_profits_wallets))

    return filtered_wallet_metrics_df



def generate_imputation_dates(wallets_config):
    """
    Generates a list of all dates that need imputation, including the first
    and last date of each training and modeling period.

    Params:
    - wallets_config (dict): config with relevant training period and modeling period dates

    Returns:
    - imputation_dates (list): list that includes the start and end dates of each period
    """

    # Extract all period start dates
    period_start_dates = sorted([datetime.strptime(date, "%Y-%m-%d")
                    for date in wallets_config['training_data']['training_period_starts'].values()
                    ])

    # Generate the output array
    imputation_dates = [period_start_dates[0].strftime("%Y-%m-%d")]  # Include the first date
    for i in range(1, len(period_start_dates)):
        # Append the day before each period start
        imputation_dates.append((period_start_dates[i] - timedelta(days=1)).strftime("%Y-%m-%d"))
        # Append period start date
        imputation_dates.append(period_start_dates[i].strftime("%Y-%m-%d"))

    # Append day before modeling period
    modeling_start = datetime.strptime(wallets_config['training_data']['modeling_period_start'], "%Y-%m-%d")
    imputation_dates.append((modeling_start - timedelta(days=1)).strftime("%Y-%m-%d"))

    # Append modeling period dates
    imputation_dates.append(wallets_config['training_data']['modeling_period_start'])
    imputation_dates.append(wallets_config['training_data']['modeling_period_end'])

    return imputation_dates



def upload_wallet_cohort(merged_df):
    """
    Uploads the list of wallet_ids that are used in the model to BigQuery. This
    is used to pull additional metrics while limiting results to only relevant wallets.

    Params:
    - merged_df (pd.DataFrame):df with wallet_id as the index
    """
    # generate upload_df from input df
    upload_df = pd.DataFrame()
    upload_df['wallet_id'] = merged_df.index.values
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
