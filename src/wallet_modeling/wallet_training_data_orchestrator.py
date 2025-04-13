"""Orchestrates groups of functions to generate wallet model pipeline"""
import time
import logging
import copy
import gc
from datetime import datetime,timedelta
from typing import Tuple,Optional,Dict,List,Union
import concurrent
import pandas as pd
import numpy as np
import pandas_gbq
from google.cloud import bigquery

# Local module imports
from wallet_modeling.wallet_training_data import WalletTrainingData
import training_data.data_retrieval as dr
import training_data.profits_row_imputation as pri
import coin_wallet_metrics.indicators as ind
import wallet_features.wallet_features_orchestrator as wfo
import wallet_features.trading_features as wtf
import wallet_features.performance_features as wpf
import wallet_features.transfers_features as wts
import wallet_features.clustering_features as wcl
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)


# ----------------------------------------
#       Primary Orchestration Class
# ----------------------------------------

class WalletTrainingDataOrchestrator:
    """
    Orchestrates wallet model training data preparation and feature generation.
    """
    def __init__(
        self,
        wallets_config: dict,
        wallets_metrics_config: dict,
        wallets_features_config: dict,
        training_wallet_cohort: List[int] = None,
        profits_df = None,
        market_data_df = None,
        macro_trends_df = None,
    ):
        # Base configs
        self.wallets_config = copy.deepcopy(wallets_config)
        self.wallets_metrics_config = wallets_metrics_config
        self.wallets_features_config = wallets_features_config
        self.training_wallet_cohort = training_wallet_cohort

        # Generated objects
        self.parquet_folder = self.wallets_config['training_data']['parquet_folder']
        self.wtd = WalletTrainingData(wallets_config)  # pass config in

        # Preexisting raw dfs if provided
        self.profits_df = profits_df
        self.market_data_df = market_data_df
        self.macro_trends_df = macro_trends_df



    @u.timing_decorator
    def retrieve_period_datasets(
        self,
        period_start_date,
        period_end_date,
        coin_cohort: Optional[set] = None,
        parquet_prefix: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, set]:
        """
        Retrieves and processes data for a specific period. If coin_cohort provided,
        filters to those coins. Otherwise applies full cleaning pipeline to establish cohort.

        Params:
        - period_start_date,period_end_date (str): Period boundaries
        - coin_cohort (set, optional): Coin IDs from training cohort
        - parquet_prefix (str, optional): Prefix for saved parquet files

        Returns:
        - tuple: (profits_df, market_data_df, coin_cohort) for the period
        """
        # 1. Get raw period data
        if self.profits_df is None or self.market_data_df is None or self.macro_trends_df is None:
            profits_df, market_data_df, macro_trends_df = self.wtd.retrieve_raw_datasets(
                period_start_date,period_end_date
            )
        else:
            logger.info("Cleaning datasets from provided versions...")
            profits_df = self.profits_df
            market_data_df = self.market_data_df
            macro_trends_df = self.macro_trends_df


        # 2. Clean raw datasets
        # Apply cleaning process including coin cohort filter if specified
        market_data_df = self.wtd.clean_market_dataset(
            market_data_df, profits_df,
            period_start_date, period_end_date,
            coin_cohort
        )
        profits_df = profits_df[profits_df['coin_id'].isin(market_data_df['coin_id'])]

        # Macro trends imputation, cleaning, validation
        macro_trends_cols = list(self.wallets_metrics_config['time_series']['macro_trends'].keys())
        macro_trends_df = dr.clean_macro_trends(macro_trends_df, macro_trends_cols,
                                        start_date = None,  # historical data is needed for indicators
                                        end_date = period_end_date)

        # Set the coin_cohort if it hadn't already been passed
        if not coin_cohort:
            coin_cohort = set(market_data_df['coin_id'].unique())
            logger.info("Defined coin cohort of %s coins after applying data cleaning filters.",
                        len(coin_cohort))

        # 3. Impute the period end (period start is pre-imputed during profits_df generation)
        imputed_profits_df = pri.impute_profits_for_multiple_dates(
            profits_df, market_data_df,
            [period_end_date],
            n_threads=1
        )

        # 4. Format and optionally save the datasets
        profits_df_formatted, market_data_df_formatted, macro_trends_df_formatted = self.wtd.format_and_save_datasets(
            imputed_profits_df,
            market_data_df,
            macro_trends_df,
            period_start_date,
            parquet_prefix
        )

        return profits_df_formatted, market_data_df_formatted, macro_trends_df_formatted, coin_cohort



    def prepare_training_data(
        self,
        profits_df_full: pd.DataFrame,
        market_data_df_full: pd.DataFrame,
        macro_trends_df_full: pd.DataFrame,
        return_files: bool = False,
        period: str = 'training'
    ) -> Union[None, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Consolidated training data preparation pipeline that handles hybridization,
        indicator generation, cohort definition and transfers retrieval.

        Params:
        - profits_df_full: Full historical profits DataFrame
        - market_data_df_full: Full historical market data DataFrame
        - macro_trends_df_full: Full historical macro trends data DataFrame
        - return_files: If True, returns input dataframes as tuple
        - period: Which period to retrieve dates from

        Returns:
        - None or (profits_df, market_indicators_df, transfers_df) if return_files is True
        """
        # Remove market data before starting balance date and validate periods
        market_data_df = market_data_df_full[
            market_data_df_full['date'] >= self.wallets_config['training_data'][f'{period}_starting_balance_date']
        ]
        u.assert_period(market_data_df,
                        self.wallets_config['training_data'][f'{period}_period_start'],
                        self.wallets_config['training_data'][f'{period}_period_end'])

        # Generate market indicators
        def generate_market_indicators_df():
            logger.info("Generating market indicators...")
            market_indicators_df = self._generate_indicators_df(
                market_data_df_full,
                parquet_filename = None,
                period = period,
                metric_type = 'market_data'
            )
            return market_indicators_df

        # Generate macro indicators
        def generate_macro_indicators_df():
            logger.info("Generating macro trends indicators...")
            market_indicators_df = self._generate_indicators_df(
                macro_trends_df_full.reset_index(drop=True),
                parquet_filename = None,
                period = period,
                metric_type = 'macro_trends'
            )
            return market_indicators_df

        # Define training wallet cohort
        def generate_cohort_profits_df(profits_df_full):
            # Hybridize wallet IDs if configured
            if self.wallets_config['training_data']['hybridize_wallet_ids']:
                profits_df_full, hybrid_cw_id_map = hybridize_wallet_address(profits_df_full)
                pd.to_pickle(
                    hybrid_cw_id_map,
                    f"{self.parquet_folder}/hybrid_cw_id_map.pkl"
                )
                upload_hybrid_wallet_mapping(hybrid_cw_id_map)
                del hybrid_cw_id_map

            logger.info("Defining wallet cohort...")

            # Apply necessary transformations (row imputation, filtering by date)
            period_profits_df = self._transform_profits_for_period(
                profits_df_full,
                market_data_df,
                self.wallets_config['training_data'][f'{period}_period_end']
            )

            # If the cohort is already defined, just filter to it
            if self.training_wallet_cohort is None:
                self._define_training_wallet_cohort(
                    period_profits_df.copy(),
                    self.wallets_config['training_data']['hybridize_wallet_ids']
                )

            # Filter profits_df to cohort
            cohort_profits_df = period_profits_df[
                period_profits_df.index.get_level_values('wallet_address').isin(self.training_wallet_cohort)
            ]
            return cohort_profits_df

        # Modified to capture futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            market_indicators_df = executor.submit(generate_market_indicators_df).result()
            macro_indicators_df = executor.submit(generate_macro_indicators_df).result()
            cohort_profits_df = executor.submit(generate_cohort_profits_df, profits_df_full).result()

        # Retrieve transfers after cohort is in BigQuery
        logger.info("Retrieving transfers sequencing data...")
        transfers_df = wts.retrieve_transfers_sequencing(
            self.wallets_config['features']['timing_metrics_min_transaction_size'],
            self.wallets_config['training_data'][f'{period}_period_end'],
            self.wallets_config['training_data']['hybridize_wallet_ids'],
        )

        if return_files is True:
            # Return dfs without saving
            return (cohort_profits_df, market_indicators_df, macro_indicators_df, transfers_df)

        else:
            # Save all files
            cohort_profits_df.to_parquet(f"{self.parquet_folder}/{period}_profits_df.parquet",index=True)
            market_indicators_df.to_parquet(f"{self.parquet_folder}/{period}_market_indicators_data_df.parquet",index=False)  # pylint:disable=line-too-long
            macro_indicators_df.to_parquet(f"{self.parquet_folder}/{period}_macro_indicators_df.parquet",index=False)  # pylint:disable=line-too-long
            transfers_df.to_parquet(f"{self.parquet_folder}/{period}_transfers_sequencing_df.parquet",index=True)

            return None




    def generate_training_features(
        self,
        profits_df: pd.DataFrame,
        market_indicators_df: pd.DataFrame,
        macro_indicators_df: pd.DataFrame,
        transfers_df: pd.DataFrame,
        return_files: bool = False,
        period: str = 'training'
    ) -> Union[None, pd.DataFrame]:
        """
        Generates full period and window features concurrently.

        Params:
        - profits_df (DataFrame): Training profits data.
        - market_indicators_df (DataFrame): Market data with indicators.
        - transfers_df (DataFrame): Transfers data.
        - period (str): Period identifier.

        Returns:
        - DataFrame if return_files is True; otherwise writes to parquet.
        """
        # Ensure indices
        profits_df = u.ensure_index(profits_df)
        market_indicators_df = u.ensure_index(market_indicators_df)

        # Define cohort from profits
        wallet_cohort = list(
            profits_df.index.get_level_values('wallet_address').drop_duplicates()
        )

        # Prepare split windows ahead of time
        training_windows_profits_dfs = self._split_training_window_profits_dfs(
            profits_df, market_indicators_df, wallet_cohort
        )
        windows_profits_tuples = [
            (window_df, i + 1) for i, window_df in enumerate(training_windows_profits_dfs)
        ]

        # Run full period and window features concurrently
        with concurrent.futures.ThreadPoolExecutor(
                self.wallets_config['features']['max_workers']
        ) as executor:
            # Submit full period feature generation
            full_period_future = executor.submit(
                wfo.calculate_wallet_features,
                profits_df.copy(),
                market_indicators_df.copy(),
                macro_indicators_df.copy(),
                transfers_df.copy(),
                wallet_cohort,
                self.wallets_config['training_data'][f'{period}_period_start'],
                self.wallets_config['training_data'][f'{period}_period_end']
            )

            # Submit window feature calculations
            window_futures = [
                executor.submit(
                    self._calculate_window_features,
                    profits_tuple,
                    market_indicators_df,
                    macro_indicators_df,
                    transfers_df,
                    wallet_cohort
                )
                for profits_tuple in windows_profits_tuples
            ]

            # Retrieve full period features result
            training_wallet_features_df = full_period_future.result()
            # Initialize full features DataFrame
            wallet_training_data_df_full = training_wallet_features_df.add_suffix("|all_windows").copy()
            wallet_training_data_df_full.to_parquet(
                f"{self.parquet_folder}/wallet_training_data_df_full.parquet", index=True
            )
            del training_wallet_features_df
            gc.collect()

            # Collect window feature results as they complete
            window_features = [future.result() for future in concurrent.futures.as_completed(window_futures)]

        # Join all window features at once
        for window_feature_df in window_features:
            wallet_training_data_df_full = wallet_training_data_df_full.join(
                window_feature_df, how='left'
            )

        # Generate clusters if configured
        if 'clustering_n_clusters' in self.wallets_config.get('features', {}):
            training_cluster_features_df = wcl.create_kmeans_cluster_features(
                wallet_training_data_df_full
            )
            training_cluster_features_df = training_cluster_features_df.add_prefix('cluster|')
            wallet_training_data_df_full = wallet_training_data_df_full.join(
                training_cluster_features_df, how='inner'
            )

        # Verify cohort integrity
        missing_wallets = set(wallet_cohort) - set(wallet_training_data_df_full.index)
        if missing_wallets:
            raise ValueError(
                f"Lost {len(missing_wallets)} wallets from original cohort during feature "
                f"generation. First few missing: {list(missing_wallets)[:5]}"
            )

        # Convert index to int64
        wallet_training_data_df_full.index = wallet_training_data_df_full.index.astype('int64')

        # Return file if configured, else save final version
        if return_files is True:
            return wallet_training_data_df_full
        else:
            wallet_training_data_df_full.to_parquet(
                f"{self.parquet_folder}/wallet_training_data_df_full.parquet", index=True
            )




    # -----------------------------------------
    #   Modeling Data Orchestration Methods
    # -----------------------------------------

    @u.timing_decorator
    def prepare_modeling_features(
        self,
        modeling_profits_df_full: pd.DataFrame,
        hybrid_cw_id_map: Optional[Dict] = None,
        period: str = 'modeling'
    ) -> pd.DataFrame:
        """
        Orchestrates data preparation and feature generation for modeling.

        Params:
        - modeling_market_data_df_full: Full market data DataFrame
        - modeling_profits_df_full: Full profits DataFrame
        - config: Configuration dictionary
        - hybrid_cw_id_map: Optional mapping for hybrid wallet IDs
        - period: Which period to retrieve dates from


        Returns:
        - modeling_wallet_features_df: Generated wallet features
        """
        logger.info("Beginning modeling data preparation...")

        # Handle hybridization if configured
        if self.wallets_config['training_data']['hybridize_wallet_ids'] is True:
            logger.info("Applying wallet-coin hybridization...")
            modeling_profits_df_full, _ = hybridize_wallet_address(
                modeling_profits_df_full,
                hybrid_cw_id_map
            )

        # Filter profits to training cohort
        if self.training_wallet_cohort is not None:
            modeling_profits_df = modeling_profits_df_full[
                modeling_profits_df_full['wallet_address'].isin(self.training_wallet_cohort)
            ]
            del modeling_profits_df_full
        else:
            modeling_profits_df = modeling_profits_df_full

        # Assert period and save filtered/hybridized profits_df
        u.assert_period(modeling_profits_df,
                        self.wallets_config['training_data'][f'{period}_period_start'],
                        self.wallets_config['training_data'][f'{period}_period_end'])
        output_path = f"{self.wallets_config['training_data']['parquet_folder']}/{period}_profits_df.parquet"
        modeling_profits_df.to_parquet(output_path, index=False)

        # Initialize features DataFrame
        logger.info("Generating modeling features...")
        modeling_wallet_features_df = pd.DataFrame(index=self.training_wallet_cohort)
        modeling_wallet_features_df.index.name = 'wallet_address'

        # Generate trading features and identify modeling cohort
        modeling_trading_features_df = self._identify_modeling_cohort(modeling_profits_df)
        modeling_wallet_features_df = modeling_wallet_features_df.join(
            modeling_trading_features_df,
            how='left'
        ).fillna({col: 0 for col in modeling_trading_features_df.columns})

        # Generate performance features
        modeling_performance_features_df = wpf.calculate_performance_features(
            modeling_wallet_features_df,
            include_twb_metrics=False
        )
        modeling_wallet_features_df = modeling_wallet_features_df.join(
            modeling_performance_features_df,
            how='left'
        ).fillna({col: 0 for col in modeling_performance_features_df.columns})

        # Save features
        output_path = f"{self.wallets_config['training_data']['parquet_folder']}/{period}_wallet_features_df.parquet"
        modeling_wallet_features_df.to_parquet(output_path, index=True)
        logger.info(f"Saved {period} features to %s", output_path)

        # Clean up memory
        del modeling_trading_features_df, modeling_performance_features_df, modeling_profits_df
        gc.collect()

        return modeling_wallet_features_df


    # ----------------------------------
    #           Helper Methods
    # ----------------------------------

    def _calculate_window_features(
        self,
        window_data: tuple,
        market_indicators_df: pd.DataFrame,
        macro_indicators_df: pd.DataFrame,
        transfers_df: pd.DataFrame,
        wallet_cohort: List[int]
    ) -> pd.DataFrame:
        """
        Process a single training window for feature generation.

        Params:
        - window_data (tuple): Contains (window_profits_df, window_number)
        - market_indicators_df (DataFrame): Market indicators data
        - macro_indicators_df (DataFrame): Macroeconomic indicators data
        - transfers_df (DataFrame): Transfer sequence data
        - wallet_cohort (List[int]): List of wallet addresses

        Returns:
        - window_features (DataFrame): Window features with appropriate suffix
        """
        window_profits_df, window_number = window_data

        # Extract window dates from MultiIndex
        window_opening_balance_date = window_profits_df.index.get_level_values('date').min()
        window_start_date = window_opening_balance_date + timedelta(days=1)
        window_end_date = window_profits_df.index.get_level_values('date').max()

        # Calculate features for this window
        window_wallet_features_df = wfo.calculate_wallet_features(
            window_profits_df.copy(),
            market_indicators_df.copy(),
            macro_indicators_df.copy(),
            transfers_df.copy(),
            wallet_cohort,
            window_start_date.strftime('%Y-%m-%d'),
            window_end_date.strftime('%Y-%m-%d')
        )

        # Add window suffix
        return window_wallet_features_df.add_suffix(f'|w{window_number}')



    @u.timing_decorator
    def _define_training_wallet_cohort(
        self,
        training_profits_df: pd.DataFrame,
        hybridize_wallet_ids: bool
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Orchestrates the definition of a wallet cohort for model training by:
        1. Imputing profits at period boundaries
        2. Calculating wallet-level trading metrics
        3. Filtering wallets based on behavior thresholds
        4. Uploading filtered cohort to BigQuery

        Params:
        - profits_df (DataFrame): Historical profit and balance data for all wallets
        - hybridize_wallet_ids (bool): whether the IDs are regular wallet_ids or hybrid wallet-coin IDs
        """
        start_time = time.time()
        training_period_start = self.wallets_config['training_data']['training_period_start']
        training_period_end = self.wallets_config['training_data']['training_period_end']

        # Confirm valid dates for training period
        u.assert_period(training_profits_df, training_period_start, training_period_end)

        # Compute wallet level metrics over duration of training period
        training_wallet_metrics_df = wtf.calculate_wallet_trading_features(training_profits_df,
                                                                        training_period_start,
                                                                        training_period_end)

        # Apply filters based on wallet behavior during the training period
        filtered_training_wallet_metrics_df = self.wtd.apply_wallet_thresholds(training_wallet_metrics_df)
        training_wallet_cohort = filtered_training_wallet_metrics_df.index.values

        if len(training_wallet_cohort) == 0:
            raise ValueError("Cohort does not include any wallets. Cohort must include wallets.")

        # Upload the cohort to BigQuery for additional complex feature generation
        self.wtd.upload_training_cohort(training_wallet_cohort, hybridize_wallet_ids)
        logger.info("Training wallet cohort defined as %s wallets after %.2f seconds.",
                    len(training_wallet_cohort), time.time()-start_time)

        # Store wallet cohort
        self.training_wallet_cohort = training_wallet_cohort



    def _transform_profits_for_period(
        self,
        profits_df: pd.DataFrame,
        market_data_df: pd.DataFrame,
        period_end: str
    ) -> pd.DataFrame:
        """
        Impute and filter profits_df up to the given period end.
        """
        # Impute for the period end (training_period_start already pre-imputed)
        imputed_df = pri.impute_profits_for_multiple_dates(
            profits_df, market_data_df, [period_end], n_threads=1, reset_index=False
        )
        # Keep only records up to period_end
        transformed_df = imputed_df[
            imputed_df.index.get_level_values('date') <= period_end
        ].copy()
        return transformed_df



    @u.timing_decorator
    def _split_training_window_profits_dfs(
        self,
        training_profits_df,
        training_market_data_df,
        wallet_cohort
    ):
        """
        Adds imputed rows at the start and end date of all windows
        """
        # Filter to only wallet cohort
        cohort_profits_df = training_profits_df[
            training_profits_df.index.get_level_values('wallet_address').isin(wallet_cohort)
        ]

        # Impute all training window dates
        training_window_boundary_dates = self.wtd.generate_training_window_imputation_dates()
        training_windows_profits_df = pri.impute_profits_for_multiple_dates(cohort_profits_df,
                                                                            training_market_data_df,
                                                                            training_window_boundary_dates,
                                                                            n_threads=4, reset_index=False)

        # Split profits_df into training windows
        training_windows_profits_df = u.ensure_index(training_windows_profits_df)
        training_windows_profits_dfs = self.wtd.split_training_window_dfs(training_windows_profits_df)

        return training_windows_profits_dfs



    @u.timing_decorator
    def _generate_indicators_df(
        self,
        training_data_df_full,
        parquet_filename="training_market_indicators_data_df",
        parquet_folder="temp/wallet_modeling_dfs",
        period='training',
        metric_type='market_data'
    ):
        """
        Adds the configured indicators to the training period market_data_df and stores it
        as a parquet file by default (or returns it).

        Default save location: temp/wallet_modeling_dfs/market_indicators_data_df.parquet

        Params:
        - training_data_df_full (df): df with complete historical data, because indicators can
            have long lookback periods (e.g. SMA 200)
        - parquet_file, parquet_folder (strings): if these have values, the output df will be saved to this
            location instead of being returned
        - period: Which period to retrieve dates from
        - metric_type (str): the key in wallet_metrics_config, e.g. 'market_data', 'macro_trends'

        Returns:
        - indicators_df (df): indicators_df for the training period only

        """
        logger.info("Beginning indicator generation process...")

        # Validate that no records exist after the training period
        training_period_end = self.wallets_config['training_data'][f'{period}_period_end']
        latest_record = training_data_df_full['date'].max()
        if latest_record > pd.to_datetime(training_period_end):
            raise ValueError(
                f"Detected data after the end of the {period} period in indicators input df."
                f"Latest record found: {latest_record} vs period end of {training_period_end}"
            )
        group_column = None
        if 'coin_id' in training_data_df_full.reset_index().columns:
            group_column = 'coin_id'

        # Adds time series ratio metrics that can have additional indicators applied to them
        if (
            metric_type == 'market_data' and
            any(k in self.wallets_metrics_config['time_series']['market_data'] for k in ['mfi', 'obv'])
        ):
            indicators_df = ind.add_market_data_dualcolumn_indicators(training_data_df_full)
        else:
            indicators_df = training_data_df_full

        # Adds indicators to all configured time series
        indicators_df = ind.generate_time_series_indicators(
            indicators_df,
            self.wallets_metrics_config['time_series'][metric_type],
            group_column
        )

        # Filters out pre-training period records now that we've computed lookback and rolling metrics
        indicators_df = indicators_df[
            indicators_df['date'] >=
            self.wallets_config['training_data'][f'{period}_starting_balance_date']
        ]

        # Reset OBV to 0 at training start if it exists
        training_start = pd.to_datetime(self.wallets_config['training_data'][f'{period}_starting_balance_date'])
        if 'obv' in indicators_df.columns:
            # Group by coin_id since OBV is coin-specific
            for coin_id in indicators_df['coin_id'].unique():
                mask = (indicators_df['coin_id'] == coin_id) & \
                    (indicators_df['date'] >= training_start)
                coin_idx = indicators_df[mask].index
                if len(coin_idx) > 0:
                    # Reset OBV to start from 0 for each coin's training period
                    indicators_df.loc[coin_idx, 'obv'] -= \
                        indicators_df.loc[coin_idx[0], 'obv']

        # If a parquet file location is specified, store the files there and return nothing
        if parquet_filename:
            parquet_filepath = f"{parquet_folder}/{parquet_filename}.parquet"
            indicators_df.to_parquet(parquet_filepath,index=False)
            logger.info(f"Stored indicators_data_df with shape {indicators_df.shape} "
                        f"to {parquet_filepath}.")

            return None

        # If no parquet file is configured then return the df
        else:
            return indicators_df


    @u.timing_decorator
    def _identify_modeling_cohort(self,modeling_period_profits_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds boolean flag indicating if wallet meets modeling period activity criteria

        Params:
        - modeling_period_profits_df (DataFrame): Input profits data with index wallet_address

        Returns:
        - DataFrame: Original dataframe index wallet_address with added boolean in_wallet_cohort \
            column that indicates if the wallet met the wallet cohort thresholds
        """

        logger.info("Identifying modeling cohort...")

        # Validate date range
        u.assert_period(modeling_period_profits_df,
                        self.wallets_config['training_data']['modeling_period_start'],
                        self.wallets_config['training_data']['modeling_period_end'])

        # Calculate modeling period wallet metrics
        modeling_wallets_df = wtf.calculate_wallet_trading_features(
            modeling_period_profits_df,
            self.wallets_config['training_data']['modeling_period_start'],
            self.wallets_config['training_data']['modeling_period_end']
        )

        # Extract thresholds
        modeling_min_investment = self.wallets_config['modeling']['modeling_min_investment']
        modeling_min_coins_traded = self.wallets_config['modeling']['modeling_min_coins_traded']

        # Create boolean mask for qualifying wallets
        meets_criteria = (
            (modeling_wallets_df['max_investment'] >= modeling_min_investment) &
            (modeling_wallets_df['unique_coins_traded'] >= modeling_min_coins_traded)
        )

        # Log stats about wallet cohort
        total_wallets = len(modeling_wallets_df)
        qualifying_wallets = meets_criteria.sum()
        logger.info(
            f"Identified {qualifying_wallets} qualifying wallets ({100*qualifying_wallets/total_wallets:.2f}% "
            f"of {total_wallets} total wallets with modeling period activity) meeting modeling cohort criteria: "
            f"min_investment=${modeling_min_investment}, min_days={modeling_min_coins_traded}"
        )

        # Add boolean flag column as 1s and 0s
        modeling_wallets_df['in_modeling_cohort'] = meets_criteria.astype(int)


        return modeling_wallets_df



# -----------------------------------
#   Hybrid Index Utility Functions
# -----------------------------------

@u.timing_decorator
def hybridize_wallet_address(
    df: pd.DataFrame,
    hybrid_cw_id_map: Optional[Dict[Tuple[int, str], int]] = None
) -> Tuple[pd.DataFrame, Dict[Tuple[int, str], int]]:
    """
    Maps wallet_address-coin_id pairs to unique integers for efficient indexing.

    Params:
    - df (DataFrame): dataframe with columns ['coin_id','wallet_address']
    - hybrid_cw_id_map (dict): mapping of (wallet,coin) tuples to integers

    Returns:
    - df (DataFrame): input df with hybrid integer keys
    - hybrid_cw_id_map (dict): mapping of (wallet,coin) tuples to integers
    """
    # Create unique mapping for wallet-coin pairs
    unique_pairs = list(zip(df['wallet_address'], df['coin_id']))

    # Generate new mapping if none was provided
    if hybrid_cw_id_map is None:
        hybrid_cw_id_map = {pair: idx for idx, pair in enumerate(set(unique_pairs), 1)}

    # Vectorized mapping of pairs to integers
    df['wallet_address'] = pd.Series(unique_pairs).map(hybrid_cw_id_map)

    return df, hybrid_cw_id_map



def dehybridize_wallet_address(
    df: pd.DataFrame,
    hybrid_cw_id_map: Dict[Tuple[int, str], int],
    hybrid_col_name: str = 'wallet_address'
) -> pd.DataFrame:
    """
    Restores original wallet_address-coin_id pairs from hybrid integer keys.

    Params:
    - df (DataFrame): dataframe with hybrid integer keys in wallet_address
    - hybrid_cw_id_map (dict): mapping of (wallet,coin) tuples to integers
    - hybrid_col_name (str): name of the column containing the hybrid key

    Returns:
    - df (DataFrame): input df with original wallet_address restored
    """
    # Create reverse mapping
    reverse_map = {v: k for k, v in hybrid_cw_id_map.items()}

    # Vectorized mapping back to original tuples
    df[hybrid_col_name] = df[hybrid_col_name].map(reverse_map).map(lambda x: x[0])

    return df


@u.timing_decorator
def upload_hybrid_wallet_mapping(hybrid_cw_id_map: Dict[Tuple[int, str], int]) -> None:
    """
    Uploads the mapping of hybrid indices to all wallet-coin pairs to BigQuery.

    Params:
        hybrid_cw_id_map (Dict[Tuple[int, str], int]): Mapping of (wallet,coin) tuples to hybrid indices
    """
    # 1. Generate upload_df from hybrid map
    # -------------------------------------
    upload_df = pd.DataFrame(
        [(v, k[0], k[1]) for k, v in hybrid_cw_id_map.items()],
        columns=['hybrid_id', 'wallet_id', 'coin_id']
    )
    upload_df['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Set df datatypes
    dtype_mapping = {
        'hybrid_id': int,
        'wallet_id': int,
        'coin_id': str,
        'updated_at': 'datetime64[ns, UTC]'
    }
    upload_df = upload_df.astype(dtype_mapping)

    # 2. Upload to BigQuery
    # ---------------------
    project_id = 'western-verve-411004'
    client = bigquery.Client(project=project_id)

    hybrid_table = f"{project_id}.temp.wallet_modeling_hybrid_id_mapping"
    schema = [
        {'name': 'hybrid_id', 'type': 'int64'},
        {'name': 'wallet_id', 'type': 'int64'},
        {'name': 'coin_id', 'type': 'string'},
        {'name': 'updated_at', 'type': 'datetime'}
    ]

    pandas_gbq.to_gbq(
        upload_df,
        hybrid_table,
        project_id=project_id,
        if_exists='replace',
        table_schema=schema,
        progress_bar=False
    )

    # 3. Create final table with resolved wallet addresses
    # ----------------------------------------------------
    create_query = f"""
    CREATE OR REPLACE TABLE `{hybrid_table}` AS
    SELECT
        h.hybrid_id,
        h.coin_id,
        h.wallet_id,
        w.wallet_address,
        CURRENT_TIMESTAMP() as updated_at
    FROM `{hybrid_table}` h
    LEFT JOIN `reference.wallet_ids` w
        ON h.wallet_id = w.wallet_id
    """

    client.query(create_query).result()
    logger.info(
        'Uploaded complete hybrid ID mapping of %s wallet-coin pairs to %s.',
        len(hybrid_cw_id_map),
        hybrid_table
    )


def merge_wallet_hybrid_data(
    hybrid_parquet_path: str,
    base_parquet_path: str,
    output_parquet_path: str,
    hybrid_suffix: str = "/walletcoin",
    base_suffix: str = "/wallet"
) -> str:
    """
    Merges hybrid and non-hybrid wallet data frames and saves to parquet.

    Params:
    - hybrid_parquet_path (str): Path to hybrid wallet-coin data parquet
    - base_parquet_path (str): Path to base wallet data parquet
    - output_parquet_path (str): Where to save merged result
    - hybrid_suffix (str): Suffix for hybrid columns
    - base_suffix (str): Suffix for base columns

    Returns:
    - str: Path to saved merged parquet file
    """
    # Read input files
    hybrid_df = pd.read_parquet(hybrid_parquet_path)
    base_df = pd.read_parquet(base_parquet_path)

    # Create temporary base address column for merging
    hybrid_df['wallet_address_base'] = hybrid_df.index.values

    # Merge the dataframes
    merged_df = hybrid_df.merge(
        base_df,
        left_on='wallet_address_base',
        right_index=True,
        suffixes=(hybrid_suffix, base_suffix)
    )

    # Clean up temporary column
    merged_df = merged_df.drop('wallet_address_base', axis=1)

    # Save merged result
    merged_df.to_parquet(output_parquet_path, index=True)

    # Clean up memory
    del hybrid_df, base_df, merged_df
    gc.collect()

    return output_parquet_path
