"""Orchestrates groups of functions to generate wallet model pipeline"""
import time
import logging
import copy
import gc
from datetime import timedelta
from typing import Tuple,Optional,List,Union
import concurrent
import pandas as pd
import numpy as np

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
import coin_insights.coin_validation_analysis as civa
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
        profits_df: pd.DataFrame = None,
        market_data_df: pd.DataFrame = None,
        macro_trends_df: pd.DataFrame = None,
        complete_hybrid_cw_id_df: pd.DataFrame = None
    ):
        # Base configs
        self.wallets_config = copy.deepcopy(wallets_config)
        self.wallets_metrics_config = wallets_metrics_config
        self.wallets_features_config = wallets_features_config # currently unused pending DDA766
        self.training_wallet_cohort = training_wallet_cohort

        # Generated objects
        self.parquet_folder = self.wallets_config['training_data']['parquet_folder']
        self.wtd = WalletTrainingData(wallets_config)  # pass config in
        self.epoch_reference_date = self.wallets_config['training_data']['modeling_period_start'].replace('-','')
        self.features_orchestrator = wfo.WalletFeaturesOrchestrator(
            self.wallets_config,
            self.wallets_metrics_config,
            self.wallets_features_config
        )

        # Hybrid ID mapping
        self.complete_hybrid_cw_id_df = complete_hybrid_cw_id_df
        self.hybridize_wallet_ids = self.wallets_config['training_data']['hybridize_wallet_ids']

        # Preexisting raw dfs if provided
        self.profits_df = profits_df
        self.market_data_df = market_data_df
        self.macro_trends_df = macro_trends_df



    @u.timing_decorator
    def retrieve_cleaned_period_datasets(
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
            logger.info("Retrieving data from BigQuery due to missing DataFrames. ")
            logger.info(
                "Status: profits_df: %s, market_data_df: %s, macro_trends_df: %s.",
                "Missing" if self.profits_df is None else "Loaded",
                "Missing" if self.market_data_df is None else "Loaded",
                "Missing" if self.macro_trends_df is None else "Loaded"
            )
            profits_df, market_data_df, macro_trends_df = self.wtd.retrieve_raw_datasets(
                period_start_date, period_end_date
            )
        else:
            logger.info("Cleaning datasets from provided versions...")
            profits_df = self.profits_df
            market_data_df = self.market_data_df
            macro_trends_df = self.macro_trends_df


        # 2. Clean raw datasets
        if not market_data_df.index.is_unique:
            raise ValueError("market_data_df index has duplicate (coin_id, date) entries.")

        # Apply cleaning process including coin cohort filter if specified
        market_data_df = self.wtd.clean_market_dataset(
            market_data_df, profits_df,
            period_start_date, period_end_date,
            coin_cohort
        )
        profits_df = profits_df[profits_df['coin_id'].isin(market_data_df['coin_id'])]

        if not market_data_df.index.is_unique:
            raise ValueError("market_data_df index has duplicate (coin_id, date) entries.")

        # Macro trends imputation, cleaning, validation
        macro_trends_cols = list(self.wallets_metrics_config['time_series']['macro_trends'].keys())
        macro_trends_df = dr.clean_macro_trends(macro_trends_df, macro_trends_cols,
                                        start_date = None,  # retain historical data for indicators
                                        end_date = period_end_date)

        # Set the coin_cohort if it hadn't already been passed
        if not coin_cohort:
            coin_cohort = set(market_data_df['coin_id'].unique())
            logger.milestone("Defined coin cohort of %s coins after applying data cleaning filters.",
                             len(coin_cohort))

        # 3. Impute the period end (period start is pre-imputed during profits_df generation)
        imputed_profits_df = pri.impute_profits_for_multiple_dates(
            profits_df, market_data_df,
            [period_end_date],
            self.wallets_config['n_threads']['profits_row_imputation']
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
            market_indicators_df = self.generate_indicators_df(
                market_data_df_full,
                self.wallets_config['training_data'][f'{period}_period_start'],
                self.wallets_config['training_data'][f'{period}_period_end'],
                parquet_filename = None,
                metric_type = 'market_data'
            )
            return market_indicators_df

        # Generate macro indicators
        def generate_macro_indicators_df():
            logger.info("Generating macro trends indicators...")
            macro_indicators_df = self.generate_indicators_df(
                macro_trends_df_full.reset_index(),
                self.wallets_config['training_data'][f'{period}_period_start'],
                self.wallets_config['training_data'][f'{period}_period_end'],
                parquet_filename = None,
                metric_type = 'macro_trends',
            )
            macro_indicators_df = macro_indicators_df.set_index('date')
            return macro_indicators_df

        # Define training wallet cohort
        def generate_cohort_profits_df(profits_df_full):
            logger.info("Defining wallet cohort...")

            # Hybridize wallet IDs if configured using existing mapping
            if self.hybridize_wallet_ids:
                profits_df_full = hybridize_wallet_address(
                    profits_df_full,
                    self.complete_hybrid_cw_id_df
                )

            # Apply necessary transformations (row imputation, filtering by date)
            period_profits_df = self._transform_profits_for_period(
                profits_df_full,
                market_data_df,
                self.wallets_config['training_data'][f'{period}_period_end']
            )

            # If the cohort is already defined, just filter to it
            if self.training_wallet_cohort is None:
                self._define_training_wallet_cohort(period_profits_df.copy())

            # Filter profits_df to cohort
            cohort_profits_df = period_profits_df[
                period_profits_df.index.get_level_values('wallet_address').isin(self.training_wallet_cohort)
            ]
            return cohort_profits_df

        # Modified to capture futures
        with concurrent.futures.ThreadPoolExecutor(
            self.wallets_config['n_threads']['training_dfs_preparation']
        ) as executor:
            market_indicators_df = executor.submit(generate_market_indicators_df).result()
            macro_indicators_df = executor.submit(generate_macro_indicators_df).result()
            cohort_profits_df = executor.submit(generate_cohort_profits_df, profits_df_full).result()

        if self.wallets_config['features']['toggle_transfers_features']:
            # Retrieve transfers after cohort is in BigQuery
            logger.info("Retrieving transfers sequencing data...")
            transfers_df = wts.retrieve_transfers_sequencing(
                self.wallets_config['features']['timing_metrics_min_transaction_size'],
                self.wallets_config['training_data'][f'{period}_period_end'],
                self.epoch_reference_date,
                self.hybridize_wallet_ids,
                self.wallets_config['training_data']['dataset'],
            )
        else:
            transfers_df = pd.DataFrame()

        if return_files is True:
            # Return dfs without saving
            return (cohort_profits_df, market_indicators_df, macro_indicators_df, transfers_df)

        else:
            # Save all files
            cohort_profits_df.to_parquet(f"{self.parquet_folder}/{period}_profits_df.parquet",index=True)
            market_indicators_df.to_parquet(f"{self.parquet_folder}/{period}_market_indicators_data_df.parquet",index=False)  # pylint:disable=line-too-long
            macro_indicators_df.to_parquet(f"{self.parquet_folder}/{period}_macro_indicators_df.parquet",index=True)  # pylint:disable=line-too-long
            transfers_df.to_parquet(f"{self.parquet_folder}/{period}_transfers_sequencing_df.parquet",index=True)

            return None



    @u.timing_decorator
    def generate_training_features(
        self,
        profits_df: pd.DataFrame,
        market_indicators_df: pd.DataFrame,
        macro_indicators_df: pd.DataFrame,
        transfers_df: pd.DataFrame,
        period: str = 'training'
    ) -> pd.DataFrame:
        """
        Generates full period and window features concurrently.

        Params:
        - profits_df (DataFrame): Training profits data.
        - market_indicators_df (DataFrame): Market data with indicators.
        - macro_indicators_df (DataFrame): Macro trends data with indicators.
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

        if not market_indicators_df.index.is_unique:
            raise ValueError("market_data_df index has duplicate (coin_id, date) entries.")

        # Run full period and window features concurrently
        with concurrent.futures.ThreadPoolExecutor(
            self.wallets_config['n_threads']['concurrent_windows']
        ) as executor:
            # Submit full period feature generation
            full_period_future = executor.submit(
                self.features_orchestrator.calculate_wallet_features,
                profits_df.copy(),
                market_indicators_df.copy(deep=True),
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
                    market_indicators_df.copy(deep=True),
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
                self.wallets_config, wallet_training_data_df_full
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

        return wallet_training_data_df_full



    # -----------------------------------------
    #   Modeling Data Orchestration Methods
    # -----------------------------------------

    @u.timing_decorator
    def prepare_modeling_features(
        self,
        modeling_profits_df_full: pd.DataFrame,
        complete_hybrid_cw_id_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Orchestrates data preparation and feature generation for modeling.

        Params:
        - modeling_market_data_df_full: Full market data DataFrame
        - modeling_profits_df_full: Full profits DataFrame
        - config: Configuration dictionary
        - complete_hybrid_cw_id_df: Optional mapping for hybrid wallet IDs
        - period: Which period to retrieve dates from


        Returns:
        - wallet_target_vars_df: Generated wallet features
        """
        # Prepare modeling_profits_df
        logger.info("Beginning modeling data preparation...")
        modeling_profits_df = self._prepare_profits_for_modeling(
            modeling_profits_df_full,
            complete_hybrid_cw_id_df
        )

        # Initialize features DataFrame
        logger.info("Generating modeling features...")
        wallet_target_vars_df = pd.DataFrame(index=self.training_wallet_cohort)
        wallet_target_vars_df.index.name = 'wallet_address'

        # Generate trading features and identify modeling cohort
        # modeling_trading_features_df = self._identify_modeling_cohort(modeling_profits_df)
        modeling_trading_features_df = wtf.calculate_wallet_trading_features(
            modeling_profits_df,
            self.wallets_config['training_data']['modeling_period_start'],
            self.wallets_config['training_data']['modeling_period_end']
        )
        wallet_target_vars_df = wallet_target_vars_df.join(
            modeling_trading_features_df,
            how='left'
        ).fillna({col: 0 for col in modeling_trading_features_df.columns})

        # Generate performance features
        modeling_config = copy.deepcopy(self.wallets_config)
        modeling_config['features']['include_twb_metrics'] = False
        modeling_performance_features_df = wpf.calculate_performance_features(
            wallet_target_vars_df,
            modeling_config  # overridden to have no twb metrics
        )
        wallet_target_vars_df = wallet_target_vars_df.join(
            modeling_performance_features_df,
            how='left'
        ).fillna({col: 0 for col in modeling_performance_features_df.columns})

        # Add coin return features if hybridization is enabled
        if self.hybridize_wallet_ids and complete_hybrid_cw_id_df is not None:
            wallet_target_vars_df = self._add_coin_return_features(
                wallet_target_vars_df,
                self.market_data_df,
                complete_hybrid_cw_id_df
            )


        # Clean up memory
        del modeling_trading_features_df, modeling_performance_features_df, modeling_profits_df
        gc.collect()

        return wallet_target_vars_df


    # ----------------------------------
    #           Helper Methods
    # ----------------------------------

    def _prepare_profits_for_modeling(
        self,
        profits_df: pd.DataFrame,
        complete_hybrid_cw_id_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Standardized profits preprocessing: hybridization, cohort filtering, validation.

        Params:
        - profits_df: Raw profits DataFrame
        - complete_hybrid_cw_id_df: Optional mapping for wallet ID hybridization
        - period: Period identifier for config lookup

        Returns:
        - processed_profits_df: Filtered and validated profits DataFrame
        """
        # Handle hybridization if configured
        if self.hybridize_wallet_ids:
            logger.info("Applying wallet-coin hybridization...")
            profits_df = hybridize_wallet_address(
                profits_df,
                complete_hybrid_cw_id_df
            )

        # Filter profits to training cohort
        if self.training_wallet_cohort is not None:
            profits_df = profits_df[
                profits_df['wallet_address'].isin(self.training_wallet_cohort)
            ]

        # Assert period validity
        u.assert_period(profits_df,
                        self.wallets_config['training_data']['modeling_period_start'],
                        self.wallets_config['training_data']['modeling_period_end'])

        return profits_df


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

        if not market_indicators_df.index.is_unique:
            raise ValueError("market_data_df index has duplicate (coin_id, date) entries.")


        # Calculate features for this window
        window_wallet_features_df = self.features_orchestrator.calculate_wallet_features(
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
        training_profits_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Orchestrates the definition of a wallet cohort for model training by:
        1. Imputing profits at period boundaries
        2. Calculating wallet-level trading metrics
        3. Filtering wallets based on behavior thresholds
        4. Uploading filtered cohort to BigQuery

        Params:
        - profits_df (DataFrame): Historical profit and balance data for all wallets
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

        # Upload the cohort to BigQuery if needed for additional complex feature generation
        if (self.wallets_config['features']['toggle_transfers_features']
            or self.wallets_config['features']['toggle_scenario_features']):
            self.wtd.upload_training_cohort(training_wallet_cohort)

        logger.milestone("Training wallet cohort defined as %s wallets after %.2f seconds.",
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
        training_windows_profits_df = pri.impute_profits_for_multiple_dates(
            cohort_profits_df,
            training_market_data_df,
            training_window_boundary_dates,
            n_threads=self.wallets_config['n_threads']['profits_row_imputation'],
            reset_index=False
        )

        # Split profits_df into training windows
        training_windows_profits_df = u.ensure_index(training_windows_profits_df)
        training_windows_profits_dfs = self.wtd.split_training_window_dfs(training_windows_profits_df)

        return training_windows_profits_dfs



    @u.timing_decorator
    def generate_indicators_df(
        self,
        training_data_df_full,
        period_start_date: str,
        period_end_date: str,
        parquet_filename="training_market_indicators_data_df",
        parquet_folder="temp/wallet_modeling_dfs",
        metric_type='market_data',
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
        - period_start_date: Starting date of the period in YYYY-MM-DD format
        - period_end_date: Ending date of the period in YYYY-MM-DD format
        - metric_type (str): the key in wallet_metrics_config, e.g. 'market_data', 'macro_trends'

        Returns:
        - indicators_df (df): indicators_df for the training period only

        """
        logger.info("Beginning indicator generation process...")

        # Validate that no records exist after the training period
        latest_record = training_data_df_full['date'].max()
        if latest_record > pd.to_datetime(period_end_date):
            raise ValueError(
                f"Detected data after the end of the period in indicators input df."
                f"Latest record found: {latest_record} vs period end of {period_end_date}"
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

        # Reset OBV to 0 at training start if it exists
        period_starting_balance_date = pd.to_datetime(period_start_date) - timedelta(days=1)
        if 'obv' in indicators_df.columns:
            # Group by coin_id since OBV is coin-specific
            for coin_id in indicators_df['coin_id'].unique():
                mask = (indicators_df['coin_id'] == coin_id) & \
                    (indicators_df['date'] >= period_starting_balance_date)
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



    def _add_coin_return_features(
        self,
        wallet_target_vars_df: pd.DataFrame,
        market_data_df: pd.DataFrame,
        complete_hybrid_cw_id_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Adds coin return features to modeling target variables by dehybridizing
        wallet IDs and calculating coin performance during modeling period.

        Params:
        - wallet_target_vars_df: DataFrame with wallet features
        - market_data_df: Full market data for coin calculations
        - complete_hybrid_cw_id_df: Mapping for wallet ID dehybridization

        Returns:
        - wallet_target_vars_df: Enhanced with coin_return columns
        """
        # Extract coin_id from hybrid ids
        cw_target_vars_df = dehybridize_wallet_address(
            wallet_target_vars_df,
            complete_hybrid_cw_id_df
        )

        # Compute coin returns during modeling period
        coin_returns_df = civa.calculate_coin_performance(
            market_data_df,
            self.wallets_config['training_data']['modeling_period_start'],
            self.wallets_config['training_data']['modeling_period_end']
        )
        cw_returns = cw_target_vars_df.merge(
            coin_returns_df, on='coin_id', how='left'
        )['coin_return'].fillna(0).values

        # Append returns columns
        wallet_target_vars_df['coin_return'] = cw_returns
        wallet_target_vars_df['coin_return_winsorized'] = u.winsorize(
            cw_returns,
            self.wallets_config['features']['returns_winsorization']
        )
        wallet_target_vars_df['coin_return_rank'] = pd.Series(cw_returns).rank(method='average')

        return wallet_target_vars_df


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
    hybrid_cw_id_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Maps wallet_address‑coin_id pairs to unique integers for efficient indexing.

    Params:
    - df (DataFrame): dataframe with columns ['coin_id','wallet_address'] or those as index levels
    - hybrid_cw_id_map (DataFrame): mapping with columns ['hybrid_cw_id','wallet_id','coin_id']

    Returns:
    - df (DataFrame): input df with hybrid integer keys applied in the 'wallet_address' column
    """
    logger.info(f"Applying hybridization to DataFrame with shape {df.shape}...")
    df_copy = df.copy()
    original_index = df_copy.index.names

    # locate wallet & coin columns (or lift from index)
    if {'wallet_address', 'coin_id'}.issubset(df_copy.columns):
        wallet_col, coin_col = 'wallet_address', 'coin_id'
        used_index = False
    elif {'wallet_address', 'coin_id'}.issubset(df_copy.index.names):
        df_copy = df_copy.reset_index()
        wallet_col, coin_col = 'wallet_address', 'coin_id'
        used_index = True
    else:
        raise ValueError(
            "hybridize_wallet_address: 'wallet_address' and 'coin_id' "
            "must exist as columns or index levels"
        )

    # Merge to apply existing mapping
    merge_map = hybrid_cw_id_df.rename(columns={
        'wallet_id': wallet_col,
        'coin_id': coin_col
    })
    df_copy = df_copy.merge(merge_map, on=[wallet_col, coin_col], how='left')

    # Swap in hybrid IDs, drop helper column
    df_copy[wallet_col] = df_copy['hybrid_cw_id']
    df_copy = df_copy.drop(columns=['hybrid_cw_id'])

    # Restore original index if we reset it
    if used_index:
        df_copy = df_copy.set_index(original_index)

    # Verify all addresses were successfully mapped
    if df_copy[wallet_col].isna().any():
        unmapped_count = df_copy[wallet_col].isna().sum()
        total_rows = len(df_copy)
        raise ValueError(
            f"Failed to map {unmapped_count} of {total_rows} wallet addresses. "
            f"Missing mappings for {df_copy[df_copy[wallet_col].isna()][[coin_col]].drop_duplicates().shape[0]} "
            f"unique coin IDs."
        )

    return df_copy



def validate_hybrid_mapping_completeness(base_df, hybrid_cw_id_df):
    """
    Validates that all coin_id/wallet_address pairs in base_df exist in hybrid_cw_id_df.

    Params:
    - base_df (DataFrame): MultiIndexed profits dataframe with (coin_id, wallet_address)
        or just wallet_address
    - hybrid_cw_id_df (DataFrame): Mapping dataframe with coin_id, wallet_address, hybrid_cw_id columns

    Raises:
    - ValueError: If any coin_id/wallet_address pairs in base_df are missing from hybrid_cw_id_df
    """
    # Validate required columns exist
    required_cols = {'coin_id', 'wallet_address'}
    missing_cols = required_cols - set(base_df.reset_index().columns)
    if missing_cols:
        logger.warning("Cannot validate hybrid mapping of a df without both coin_id and "
                       "wallet_address columns.")
        return

    logger.info("Validating hybrid_cw_id coverage...")

    # -------------------------------------------------
    # 1) Quickly grab the wallet_address Series
    # -------------------------------------------------
    if 'wallet_address' in base_df.columns:
        wallet_series = base_df['wallet_address']
    else:
        wallet_series = base_df.index.get_level_values('wallet_address')

    # Detect whether the IDs are already hybrid ints (they start with '3')
    is_hybrid = wallet_series.astype(str).str.startswith('3').all()

    # -------------------------------------------------
    # 2) Hybrid‑ID case  →  simple vectorized membership test
    # -------------------------------------------------
    if is_hybrid:
        in_mapping = pd.Series(wallet_series.unique()).isin(
            hybrid_cw_id_df['hybrid_cw_id'].unique()
        )
        if not in_mapping.all():
            missing_ids = pd.Series(wallet_series.unique())[~in_mapping]
            raise ValueError(
                f"Found {missing_ids.size} hybrid_cw_id values with no mapping. "
                f"Examples: {missing_ids.iloc[:3].tolist()}"
            )

    # -------------------------------------------------
    # 3) Regular (coin_id, wallet_address) pair case
    #    →  use a left merge with an indicator to spot gaps
    # -------------------------------------------------
    else:
        # Build a de‑duplicated DataFrame of unique pairs from *base_df*
        if {'coin_id', 'wallet_address'}.issubset(base_df.columns):
            base_pairs = base_df[['coin_id', 'wallet_address']].drop_duplicates()
        else:
            base_pairs = (
                base_df.reset_index()[['coin_id', 'wallet_address']]
                .drop_duplicates()
            )

        mapping_pairs = hybrid_cw_id_df[['coin_id', 'wallet_address']].drop_duplicates()

        # Left‑join to find pairs that exist only in *base_df*
        merged = base_pairs.merge(
            mapping_pairs,
            on=['coin_id', 'wallet_address'],
            how='left',
            indicator=True
        )

        missing_rows = merged[merged['_merge'] == 'left_only']

        if not missing_rows.empty:
            raise ValueError(
                f"Found {len(missing_rows)} coin_id/wallet_address pairs in base_df with "
                f"no corresponding entry in hybrid_cw_id_df. "
                f"Examples: {missing_rows.head(3).to_dict('records')}"
            )

    logger.info("All hybrid_cw_ids have coverage.")



@u.timing_decorator
def dehybridize_wallet_address(
    df: pd.DataFrame,
    hybrid_cw_id_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Reverse hybridization by expanding the integer hybrid IDs back into their
    original wallet_address and coin_id components.

    Params:
    - df (DataFrame): Input DataFrame whose *wallet_address* column (or index
      level) contains `hybrid_cw_id` integers produced by
      `hybridize_wallet_address`.
    - hybrid_cw_id_df (DataFrame): Mapping DataFrame with columns
      ['hybrid_cw_id', 'wallet_id', 'coin_id'].

    Returns:
    - DataFrame identical to the input but with the hybrid IDs replaced by the
      original `wallet_address` values **and** a corresponding `coin_id`
      (either as a column or index level mirroring the input structure).

    Notes
    -----
    * The function preserves the original index structure:
      - If the hybrid IDs were in the index, the output will have a
        MultiIndex (wallet_address, coin_id).
      - If they were in a regular column, two explicit columns will be
        present.
    * Raises ValueError if any hybrid IDs fail to map.
    """
    logger.info("De‑hybridizing DataFrame with shape %s …", df.shape)
    df_out = df.copy()
    original_index = df_out.index.names

    # -------------------------------------------------
    # Detect where the hybrid IDs live and rename the
    # column/index level to *hybrid_cw_id* so we can
    # merge cleanly without duplicate label errors.
    # -------------------------------------------------
    if 'wallet_address' in df_out.columns:
        # Column case → rename in‑place
        df_out = df_out.rename(columns={'wallet_address': 'hybrid_cw_id'})
        wallet_col = 'hybrid_cw_id'
        used_index = False

    elif 'wallet_address' in df_out.index.names:
        # Index case → bring into columns, then rename
        # Drop any duplicate column first to avoid reset_index collision
        if 'wallet_address' in df_out.columns:
            df_out = df_out.drop(columns=['wallet_address'])

        df_out = df_out.reset_index()
        df_out = df_out.rename(columns={'wallet_address': 'hybrid_cw_id'})
        wallet_col = 'hybrid_cw_id'
        used_index = True

    else:
        raise ValueError(
            "dehybridize_wallet_address: expected 'wallet_address' either as "
            "a column or an index level containing hybrid IDs."
        )

    # Merge to add wallet_id + coin_id
    df_out = df_out.merge(hybrid_cw_id_df, on=wallet_col, how='left')

    # Verify mapping completeness
    if df_out['wallet_address'].isna().any():
        missing = df_out['wallet_address'].isna().sum()
        raise ValueError(
            f"Failed to de‑hybridize {missing} rows – missing hybrid_cw_id "
            "mappings."
        )

    # Replace hybrid id with original wallet_id and rename for consistency
    df_out = df_out.drop(columns=['hybrid_cw_id'])

    # Restore original index structure
    if used_index:
        # Re‑establish MultiIndex (wallet_address, coin_id, [other …])
        df_out = df_out.set_index(
            ['wallet_address', 'coin_id'] +
            [col for col in original_index if col not in ('wallet_address', 'coin_id')]
        )
    else:
        # Ensure wallet_address & coin_id are regular columns
        pass  # nothing required

    if len(df) != len(df_out):
        raise ValueError(f"Output df shape of {len(df_out)} does not match input df shape "
                         "of {len(df)}.")

    logger.info("Completed de‑hybridization; resulting shape %s.", df_out.shape)
    return df_out
