"""Primary sequence functions used to generate training data for the wallet modeling pipeline"""
import logging
import copy
from datetime import datetime, timedelta
from typing import List
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from google.cloud import bigquery
from dreams_core import core as dc

# Local module imports
import training_data.data_retrieval as dr
import wallet_features.market_cap_features as wmc
import utils as u
import utilities.bq_utils as bqu

# Set up logger at the module level
logger = logging.getLogger(__name__)



# ----------------------------------------
#       Primary Orchestration Class
# ----------------------------------------

class WalletTrainingData:
    """Wrapper for training data preparation functions to store the config state"""
    def __init__(self, wallets_config):
        self.wallets_config = copy.deepcopy(wallets_config)  # store the config at instance-level
        self.epoch_reference_date = wallets_config['training_data']['modeling_period_start'].replace('-','')

# -------------------------------------------
#      Training Data Preparation Methods
# -------------------------------------------

    @u.timing_decorator
    def retrieve_raw_datasets(self, period_start_date, period_end_date):
        """
        Retrieves raw market and profits data after applying the min_wallet_inflows filter at the
        wallet level. Because the filter is applied cumulatively, later period_end_dates will return
        additional wallets whose newly included inflows put them over the threshold.

        Params:
        - period_start_date,period_end_date (YYYY-MM-DD): The data period boundary dates.

        Returns:
        - profits_df, market_data_df, macro_trends_df: raw dataframes
        """
        # Identify the date we need ending balances from
        period_start_date = datetime.strptime(period_start_date,'%Y-%m-%d')
        period_end_date = datetime.strptime(period_end_date,'%Y-%m-%d')
        starting_balance_date = period_start_date - timedelta(days=1)

        # Retrieve all datasets
        with ThreadPoolExecutor(
            self.wallets_config['n_threads']['raw_data_retrieval']
        ) as executor:

            # Profits data
            profits_future = executor.submit(
                dr.retrieve_profits_data,
                starting_balance_date,
                period_end_date,
                self.wallets_config['data_cleaning']['min_wallet_inflows'],
                self.wallets_config['training_data']['dataset']
            )

            # Market data
            market_future = executor.submit(dr.retrieve_market_data,
                                            period_end_date,
                                            self.wallets_config['training_data']['dataset'])

            # Macro trends data
            macro_future = executor.submit(dr.retrieve_macro_trends_data)

            # Merge all dfs
            profits_df = profits_future.result()
            market_data_df = market_future.result()
            macro_trends_df = macro_future.result()

        # Remove all records after the training period end to ensure no data leakage
        market_data_df = market_data_df[market_data_df['date']<=period_end_date]
        macro_trends_df = macro_trends_df[macro_trends_df.index.get_level_values('date')<=period_end_date]

        return profits_df, market_data_df, macro_trends_df


    def clean_market_dataset(self, market_data_df, profits_df, period_start_date, period_end_date, coin_cohort=None):
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
                self.wallets_config,
                period_start_date,
                period_end_date
            )
        else:
            # Otherwise just filter to that cohort
            market_data_df = market_data_df[market_data_df['coin_id'].isin(coin_cohort)]


        # Intelligently impute market cap data in market_data_df when good data is available
        market_data_df = dr.impute_market_cap(
            market_data_df,
            self.wallets_config['data_cleaning']['min_mc_imputation_coverage'],
            self.wallets_config['data_cleaning']['max_mc_imputation_multiple']
        )

        # Crudely fill all remaining gaps in market cap data
        market_data_df = wmc.force_fill_market_cap(
            market_data_df,
            self.wallets_config['data_cleaning']['market_cap_default_fill']
        )

        if not coin_cohort:
            cfg = self.wallets_config
            start_date = cfg['training_data']['training_period_start']
            end_date   = cfg['training_data']['training_period_end']

            mc = market_data_df['market_cap_filled']
            dt = market_data_df['date']

            # thresholds
            max_init = cfg['data_cleaning']['max_initial_market_cap']
            min_init = cfg['data_cleaning']['min_initial_market_cap']
            max_end  = cfg['data_cleaning']['max_ending_market_cap']
            min_end  = cfg['data_cleaning']['min_ending_market_cap']

            # Masks for each filter
            mask_start_high = (dt == start_date) & (mc > max_init)
            mask_start_low  = (dt == start_date) & (mc < min_init)
            mask_end_high   = (dt == end_date)   & (mc > max_end)
            mask_end_low    = (dt == end_date)   & (mc < min_end)

            # Identify coin_ids for each
            drop_start_high = set(market_data_df.loc[mask_start_high, 'coin_id'])
            drop_start_low  = set(market_data_df.loc[mask_start_low,  'coin_id'])
            drop_end_high   = set(market_data_df.loc[mask_end_high,   'coin_id'])
            drop_end_low    = set(market_data_df.loc[mask_end_low,    'coin_id'])

            # Combine all drops and filter once
            coins_to_drop = drop_start_high | drop_start_low | drop_end_high | drop_end_low
            market_data_df = market_data_df[
                ~market_data_df['coin_id'].isin(coins_to_drop)
            ]

            # Log individual counts
            logger.info("Total coins removed for market cap thresholds: %s", len(coins_to_drop))
            logger.info("  %s coins above max_initial_market_cap at start date %s.",
                        len(drop_start_high), start_date)
            logger.info("  %s coins below min_initial_market_cap at start date %s.",
                        len(drop_start_low), start_date)
            logger.info("  %s coins above max_ending_market_cap at end date %s.",
                        len(drop_end_high), end_date)
            logger.info("  %s coins below min_ending_market_cap at end date %s.",
                        len(drop_end_low), end_date)

        else:
            logger.info("Returned market data for the %s coins in the coin cohort passed as a parameter.",
                        len(coin_cohort))

        return market_data_df


    def format_and_save_datasets(
            self,
            profits_df,
            market_data_df,
            macro_trends_df,
            period_start_date,
            parquet_prefix=None
        ):
        """
        Formats and optionally saves the final datasets.

        Params:
        - profits_df, market_data_df, macro_trends_df (DataFrames): Input dataframes
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
        profits_df, _ = dr.clean_profits_df(profits_df, self.wallets_config['data_cleaning'])

        # Round relevant columns
        columns_to_round = [
            'usd_balance',
            'usd_net_transfers',
            'usd_inflows',
            'profits_cumulative',
            'usd_inflows_cumulative',
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
            parquet_folder = self.wallets_config['training_data']['parquet_folder']
            profits_file = f"{parquet_folder}/{parquet_prefix}_profits_df_full.parquet"
            u.to_parquet_safe(profits_df, profits_file,index=False)
            logger.info(f"Stored profits_df with shape {profits_df.shape} to {profits_file}.")

            # Store market data
            market_data_file = f"{parquet_folder}/{parquet_prefix}_market_data_df_full.parquet"
            u.to_parquet_safe(market_data_df, market_data_file,index=False)
            logger.info(f"Stored market_data_df with shape {market_data_df.shape} to {market_data_file}.")

            # Store macro_trends_df
            macro_trends_file = f"{parquet_folder}/{parquet_prefix}_macro_trends_df_full.parquet"
            u.to_parquet_safe(macro_trends_df, macro_trends_file,index=True)  # retain index for macro trends
            logger.info(f"Stored macro_trends_df with shape {macro_trends_df.shape} to {macro_trends_file}.")
            return None, None, None

        return profits_df, market_data_df, macro_trends_df


    def generate_training_window_imputation_dates(self) -> List[datetime]:
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
            for date in self.wallets_config['training_data']['training_window_starts']
        ])

        # Don't include the first value since training_period_start is already imputed
        imputation_dates: List[datetime] = starting_balance_dates[1:]

        # Add the end date for the final window
        final_window_end_date: datetime = datetime.strptime(self.wallets_config['training_data']['training_period_end'],
                                                            "%Y-%m-%d")
        imputation_dates += [final_window_end_date]

        return imputation_dates



    def split_training_window_dfs(self, training_profits_df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Splits the full profits_df into separate dfs for each training window.
        Input df must have MultiIndex on (coin_id, wallet_address, date).

        Params:
        - training_profits_df (DataFrame): MultiIndexed df containing profits data

        Returns:
        - training_windows_dfs (list of DataFrames): list of profits_dfs for each window
        """
        logger.debug("Generating window-specific profits_dfs...")
        u.ensure_index(training_profits_df)

        # Convert training window starts to sorted datetime
        training_windows_starts = sorted([
            datetime.strptime(date, "%Y-%m-%d")
            for date in self.wallets_config['training_data']['training_window_starts']
        ])

        # Generate end dates for each period
        training_windows_ends = (
            [date - timedelta(days=1) for date in training_windows_starts[1:]]
            + [datetime.strptime(self.wallets_config['training_data']['training_period_end'], "%Y-%m-%d")]
        )

        # Generate starting balance dates for each period
        training_windows_starting_balance_dates = (
            [training_windows_starts[0] - timedelta(days=1)]  # The day before the first window start
            + training_windows_ends[:-1]  # the end date of one window is the starting balance date of the next
        )

        # Create array of DataFrames for each training period
        training_windows_profits_dfs = []
        for start_bal_date, end_date in zip(training_windows_starting_balance_dates, training_windows_ends):
            # Filter to between the starting balance date and end date
            window_df = training_profits_df[
                (training_profits_df.index.get_level_values('date') >= start_bal_date) &
                (training_profits_df.index.get_level_values('date') <= end_date)
            ]

            # Override records on starting balance date using MultiIndex
            start_date_idx = window_df.index[
                window_df.index.get_level_values('date') == start_bal_date
            ]
            window_df.loc[start_date_idx, ['usd_net_transfers', 'usd_inflows', 'is_imputed']] = [0, 0, True]

            # Confirm the period boundaries are handled correctly
            start_date = start_bal_date + timedelta(days=1)
            u.assert_period(window_df, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            training_windows_profits_dfs.append(window_df)

        # Confirm that all window dfs' transfers match the full df's transfers starting from the first window
        # Compare absolute USD flow so tiny signed‑sum drift doesn’t trigger a failure
        full_sum = (training_profits_df
                    [training_profits_df.index.get_level_values('date') >= training_windows_starts[0]]
                    ['usd_net_transfers']
                    .abs()
                    .sum())
        window_sum = sum(df['usd_net_transfers'].abs().sum()
                         for df in training_windows_profits_dfs)
        if not np.isclose(full_sum, window_sum, rtol=1e-4):
            raise ValueError(f"Net transfers in full training period ({full_sum}) do not match combined "
                            f"sum of transfers in windows dfs ({window_sum})")

        # Result: array of DataFrames
        for i, df in enumerate(training_windows_profits_dfs):
            logger.debug("Training Window %s (%s to %s): %s",
                        i + 1,
                        df.index.get_level_values('date').min().strftime('%Y-%m-%d'),
                        df.index.get_level_values('date').max().strftime('%Y-%m-%d'),
                        df.shape)
        logger.debug("Training Period (%s to %s): %s",
                    training_profits_df.index.get_level_values('date').min().strftime('%Y-%m-%d'),
                    training_profits_df.index.get_level_values('date').max().strftime('%Y-%m-%d'),
                    training_profits_df.shape)

        return training_windows_profits_dfs




    # -----------------------------------
    #     Wallet Cohort Management
    # -----------------------------------

    def apply_wallet_thresholds(self, wallet_metrics_df):
        """
        Applies data cleaning filters to the a df keyed on wallet_address

        Params:
        - wallet_metrics_df (df): dataframe with index wallet_address and columns with
            metrics that the filters will be applied to

        """
        # Extract thresholds
        min_coins = self.wallets_config['data_cleaning']['min_coins_traded']
        max_coins = self.wallets_config['data_cleaning']['max_coins_traded']
        min_wallet_investment = self.wallets_config['data_cleaning']['min_wallet_investment']
        max_wallet_investment = self.wallets_config['data_cleaning']['max_wallet_investment']
        min_wallet_volume = self.wallets_config['data_cleaning']['min_wallet_volume']
        max_wallet_volume = self.wallets_config['data_cleaning']['max_wallet_volume']
        max_wallet_profits = self.wallets_config['data_cleaning']['max_wallet_profits']

        if self.wallets_config['training_data']['hybridize_wallet_ids'] is True:
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



    def upload_training_cohort(
            self,
            cohort_ids: np.array
        ) -> None:
        """
        Uploads the list of wallet_ids that are used in the model to BigQuery. This
        is used to pull additional metrics while limiting results to only relevant wallets.

        Only non-hybridized wallet cohorts are uploaded; hybridized transfers queries can
        impute the corresponding coin_ids by using the reference.wallet_coin_ids table.

        Params:
        - cohort_ids (np.array): the wallet_ids included in the cohort
        """
        # 1. Prepare upload DataFrame
        upload_df = pd.DataFrame({
            'wallet_id': cohort_ids,
            'updated_at': datetime.now()
        })
        upload_df = upload_df.astype({
            'wallet_id': 'int64',
            'updated_at': 'datetime64[ns]'
        })

        project_id = 'western-verve-411004'
        table_id = f"{project_id}.temp.wallet_modeling_training_cohort_{self.epoch_reference_date}"

        # 2. Upload using centralized utility (honors global concurrency limit)
        job_config = bigquery.LoadJobConfig(
            schema=[
                bigquery.SchemaField('wallet_id', 'INT64'),
                bigquery.SchemaField('updated_at', 'DATETIME'),
            ],
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
        )
        bqu.upload_dataframe(upload_df, table_id, job_config=job_config)

        # 3. Populate wallet_address from reference table
        create_query = f"""
        CREATE OR REPLACE TABLE `{table_id}` AS
        SELECT
            t.wallet_id,
            w.wallet_address,
            t.updated_at
        FROM `{table_id}` t
        JOIN `reference.wallet_ids` w
            ON t.wallet_id = w.wallet_id
        """
        bqu.run_query(create_query)

        logger.info(
            "Uploaded cohort of %d wallets with addresses to %s.",
            len(cohort_ids),
            table_id
        )
