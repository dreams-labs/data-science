"""Orchestrates the creation of training_data_dfs for multiple training windows"""
import os
import logging
from pathlib import Path
import copy
from typing import List,Dict,Tuple
from datetime import datetime, timedelta
import pandas as pd

# Local module imports
import training_data.profits_row_imputation as pri
from wallet_modeling.wallet_training_data import WalletTrainingData
import wallet_modeling.wallets_config_manager as wcm
import wallet_modeling.wallet_training_data_orchestrator as wtdo
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)


# ----------------------------------------
#       Primary Orchestration Class
# ----------------------------------------

class MultiWindowOrchestrator:
    """
    Orchestrates training data generation across multiple time windows by
    offsetting base config dates and managing the resulting datasets.
    """
    def __init__(
        self,
        base_config: dict,
        metrics_config: dict,
        features_config: dict,
        windows_config: dict,
        complete_profits_df: pd.DataFrame = None,
        complete_market_data_df: pd.DataFrame = None,
        complete_macro_trends_df: pd.DataFrame = None,

    ):
        # Param Configs
        self.base_config = base_config
        self.metrics_config = metrics_config
        self.features_config = features_config
        self.windows_config = windows_config

        # Generated configs
        self.all_windows_configs = self._generate_window_configs()

        # Complete df objects
        self.complete_profits_df_file = None
        self.complete_market_data_df_file = None
        self.complete_macro_trends_df_file = None
        self.complete_profits_df = complete_profits_df
        self.complete_market_data_df = complete_market_data_df
        self.complete_macro_trends_df = complete_macro_trends_df
        self.wtd = WalletTrainingData(self.base_config)  # helper for complete df generation

        # Generated objects
        self.output_dfs = {}

        u.notify('wifi_drop')


    def load_complete_raw_datasets(self) -> None:
        """
        Identifies the earliest training_period_start and latest modeling_period_end
        across all windows, then retrieves the full profits and market data once,
        storing both in memory and in parquet files.

        Returns:
        - None
            """
        # 1. Find earliest and latest boundaries across all windows
        all_train_starts = []
        all_validation_ends = []
        for cfg in self.all_windows_configs:
            all_train_starts.extend(cfg['training_data']['training_window_starts'])
            all_validation_ends.append(cfg['training_data']['validation_period_end'])

        earliest_training_start = min(all_train_starts)
        latest_validation_end = max(all_validation_ends)

        logger.info("Retrieving complete dataset for range %s to %s.",
                    earliest_training_start, latest_validation_end)

        # Retrieve the full data once (BigQuery or otherwise)
        logger.info("Pulling complete raw datasets from %s through %s...",
                    earliest_training_start, latest_validation_end)
        self.complete_profits_df, self.complete_market_data_df, self.complete_macro_trends_df = \
            self.wtd.retrieve_raw_datasets(earliest_training_start, latest_validation_end)

        # Set index
        self.complete_profits_df = u.ensure_index(self.complete_profits_df)
        self.complete_market_data_df = u.ensure_index(self.complete_market_data_df)

        # Save them to parquet for future reuse
        parquet_folder = self.base_config['training_data']['parquet_folder']
        self.complete_profits_df_file = f"{parquet_folder}/complete_profits_df.parquet"
        self.complete_market_data_df_file = f"{parquet_folder}/complete_market_data_df.parquet"
        self.complete_macro_trends_df_file = f"{parquet_folder}/complete_macro_trends_df.parquet"

        self.complete_profits_df.to_parquet(self.complete_profits_df_file)
        self.complete_market_data_df.to_parquet(self.complete_market_data_df_file)
        self.complete_macro_trends_df.to_parquet(self.complete_macro_trends_df_file)

        logger.info("Saved complete profits to %s (%s rows), market data to %s (%s rows), " \
                    " market data to %s (%s rows).",
                    self.complete_profits_df_file, len(self.complete_profits_df),
                    self.complete_market_data_df_file, len(self.complete_market_data_df),
                    self.complete_macro_trends_df_file, len(self.complete_macro_trends_df)
                    )


    @u.timing_decorator
    def generate_windows_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates training data for all window configs in parallel.

        Returns:
        - merged_training_df: MultiIndexed on (wallet_address, window_start_date)
        - merged_modeling_df: MultiIndexed on (wallet_address, window_start_date)
        """
        # Ensure the complete dfs encompass the full range of training_window_starts
        self._assert_complete_coverage()

        # Initialize storage for window DataFrames
        training_window_dfs = {}
        modeling_window_dfs = {}

        u.notify('intro_3')
        for window_config in self.all_windows_configs:
            model_start = window_config['training_data']['modeling_period_start']
            logger.info(f"Generating data for window starting {model_start}")
            u.notify('futuristic')

            # Generate name of parquet folder and create it if necessary
            window_parquet_folder = window_config['training_data']['parquet_folder']
            os.makedirs(window_parquet_folder, exist_ok=True)

            try:
                # 1. Initialize data generator for window with presplit dfs
                training_profits_df = None
                training_market_data_df = None
                if self.complete_profits_df is not None:
                    training_profits_df,training_market_data_df = \
                        self._transform_complete_dfs_for_window(
                            window_config['training_data']['training_period_start'],
                            window_config['training_data']['training_period_end']
                        )

                training_generator = wtdo.WalletTrainingDataOrchestrator(
                    window_config,
                    self.metrics_config,
                    self.features_config,
                    profits_df = training_profits_df,
                    market_data_df = training_market_data_df
                )

                # 2. Generate TRAINING_DATA_DFs
                training_profits_df_full, training_market_data_df_full, macro_trends_df_full, training_coin_cohort = \
                    training_generator.retrieve_period_datasets(
                        window_config['training_data']['training_period_start'],
                        window_config['training_data']['training_period_end']
                    )

                training_profits_df, training_market_indicators_df, training_transfers_df = \
                    training_generator.prepare_training_data(
                        training_profits_df_full,
                        training_market_data_df_full,
                        return_files=True
                    )

                window_training_data_df = training_generator.generate_training_features(
                    training_profits_df,
                    training_market_indicators_df,
                    training_transfers_df,
                    return_files=True
                )

                # Store training df with window date
                window_date = datetime.strptime(
                    window_config['training_data']['modeling_period_start'], '%Y-%m-%d')
                training_window_dfs[window_date] = window_training_data_df

                # 3. Generate MODELING_DATA_DFs
                modeling_profits_df = None
                modeling_market_data_df = None
                if self.complete_profits_df is not None:
                    modeling_profits_df,modeling_market_data_df = \
                        self._transform_complete_dfs_for_window(
                            window_config['training_data']['modeling_period_start'],
                            window_config['training_data']['modeling_period_end']
                        )

                modeling_generator = wtdo.WalletTrainingDataOrchestrator(
                    window_config,
                    self.metrics_config,
                    self.features_config,
                    training_wallet_cohort = training_generator.training_wallet_cohort,
                    profits_df = modeling_profits_df,
                    market_data_df = modeling_market_data_df

                )

                modeling_profits_df_full,_,_,_ = modeling_generator.retrieve_period_datasets(
                    window_config['training_data']['modeling_period_start'],
                    window_config['training_data']['modeling_period_end'],
                    training_coin_cohort
                )

                hybrid_cw_id_map = None
                if window_config['training_data']['hybridize_wallet_ids']:
                    hybrid_cw_id_map = pd.read_pickle(
                        f"{window_config['training_data']['parquet_folder']}/hybrid_cw_id_map.pkl")

                window_modeling_features_df = modeling_generator.prepare_modeling_features(
                    modeling_profits_df_full,
                    hybrid_cw_id_map
                )

                # Store modeling df with window date
                modeling_window_dfs[window_date] = window_modeling_features_df

                logger.info(f"Successfully generated data for window {model_start}")

            except Exception as e:
                logger.error(f"Failed to generate data for {model_start}: {str(e)}")
                raise

        # Merge window DataFrames
        wallet_training_data_df = self._merge_window_dfs(training_window_dfs)
        modeling_wallet_features_df = self._merge_window_dfs(modeling_window_dfs)

        # Confirm indices match
        u.assert_matching_indices(wallet_training_data_df,modeling_wallet_features_df)

        logger.info("Generated multi-window DataFrames with shapes %s and %s",
                    wallet_training_data_df.shape, modeling_wallet_features_df.shape)
        u.notify('level_up')

        return wallet_training_data_df, modeling_wallet_features_df


    # -----------------------------------
    #           Helper Methods
    # -----------------------------------

    def _generate_window_configs(self) -> List[Dict]:
        """
        Generates config dicts for each offset window.

        Returns:
        - List[Dict]: List of config dicts, one per window.
        """
        all_windows_configs = []
        base_training_data = self.base_config['training_data']
        offsets = self.windows_config['offset_windows']['offsets']

        for offset_days in offsets:
            # Deep copy base config to prevent mutations
            window_config = copy.deepcopy(self.base_config)

            # Offset key dates
            for date_key in [
                'modeling_period_start',
                'modeling_period_end',
                # 'validation_period_end'
            ]:
                base_date = datetime.strptime(base_training_data[date_key], '%Y-%m-%d')
                new_date = base_date + timedelta(days=offset_days)
                window_config['training_data'][date_key] = new_date.strftime('%Y-%m-%d')

            # Offset every date in training_window_starts
            window_config['training_data']['training_window_starts'] = [
                (datetime.strptime(dt, '%Y-%m-%d') + timedelta(days=offset_days)).strftime('%Y-%m-%d')
                for dt in base_training_data['training_window_starts']
            ]

            # Update parquet folder based on new modeling_period_start
            model_start = window_config['training_data']['modeling_period_start']
            folder_suffix = datetime.strptime(model_start, '%Y-%m-%d').strftime('%y%m%d')
            base_folder = Path(base_training_data['parquet_folder'])
            window_config['training_data']['parquet_folder'] = str(base_folder / folder_suffix)

            # Use WalletsConfig to add derived values
            window_config = wcm.add_derived_values(window_config)

            # Log window configuration
            logger.debug(
                f"Generated config for {offset_days} day offset window: "
                f"modeling_period={window_config['training_data']['modeling_period_start']} "
                f"to {window_config['training_data']['modeling_period_end']}"
            )

            all_windows_configs.append(window_config)

        return all_windows_configs


    def _transform_complete_dfs_for_window(self,
                                           period_start: str,
                                           period_end: str) -> pd.DataFrame:
        """
        Impute and filter the complete_profits_df to remove all rows after the period_end
        and impute the starting values as of the period_start starting balance date.

        This function relies on the WalletTrainingData.split_training_window_dfs function
        that is used to generate the profits_dfs for the multiple training period features.

        Params:
        - period_start (str): The training_period_start to use for the filtered df.
        - period_end (str): The last date to retain in the self.complete_profits_df
        """
        # Keep only records up to period_end
        window_profits_df = (self.complete_profits_df.copy()
                      [self.complete_profits_df.index.get_level_values('date') <= period_end])
        window_market_data_df = (self.complete_market_data_df.copy()
                                 [self.complete_market_data_df.index.get_level_values('date') <= period_end])

        # Impute profits_df rows as of the period starting balance date and period end
        period_starting_balance_date = (pd.to_datetime(period_start) - timedelta(days=1)).strftime('%Y-%m-%d')
        window_profits_df = pri.impute_profits_for_multiple_dates(
            window_profits_df,
            window_market_data_df,
            [period_starting_balance_date,period_end],
            n_threads=4,
            reset_index=False
        )

        # Apply min_wallet_inflows filter
        valid_rows = (window_profits_df.xs(period_end, level='date')
                       ['usd_inflows_cumulative'] >= self.base_config['data_cleaning']['min_wallet_inflows'])
        valid_pairs = valid_rows[valid_rows].index
        window_profits_df = window_profits_df.loc[window_profits_df.index.droplevel('date').isin(valid_pairs)]

        # Create config with a training period of the param values
        splitter_config = copy.deepcopy(self.base_config)
        splitter_config['training_data']['training_window_starts'] = [period_start]
        splitter_config['training_data']['training_period_end'] = period_end

        # Initiate WalletTrainingData with the only window being the training_period_start
        complete_df_splitter = wtdo.WalletTrainingData(splitter_config)

        # Use the logic for splitting training window profits_dfs to split the complete profits_df
        window_profits_df = complete_df_splitter.split_training_window_dfs(u.ensure_index(window_profits_df))[0]

        return window_profits_df.reset_index(), window_market_data_df.reset_index()


    def _merge_window_dfs(self, window_dfs: Dict[datetime, pd.DataFrame]) -> pd.DataFrame:
        """
        Merges window DataFrames into a single MultiIndexed DataFrame with non-nullable indices.

        Params:
        - window_dfs: Dict mapping window dates to DataFrames

        Returns:
        - DataFrame: MultiIndexed on (wallet_address, window_start_date)
        """
        merged_dfs = []
        for window_date, df in window_dfs.items():
            # Convert wallet_address index to non-nullable int64
            wallet_index = df.index.astype("int64")

            # Build a MultiIndex from the non-nullable wallet_index and the window_date
            multi_idx = pd.MultiIndex.from_product(
                [wallet_index, [window_date]],
                names=['wallet_address', 'window_start_date']
            )

            # Use a copy to avoid modifying the original DataFrame
            window_df = df.copy()
            window_df.index = multi_idx

            merged_dfs.append(window_df)

        full_df = pd.concat(merged_dfs, axis=0).sort_index()

        return full_df



    # -----------------------------------
    #           Utility Methods
    # -----------------------------------

    def _assert_complete_coverage(self) -> None:
        """
        Verify that profits and market data fully cover all training windows.

        Raises:
        - ValueError: If data coverage is incomplete with specific boundary details
        """
        # Find earliest and latest boundaries
        all_train_starts = []
        all_model_ends = []
        for cfg in self.all_windows_configs:
            all_train_starts.extend(cfg['training_data']['training_window_starts'])
            all_model_ends.append(cfg['training_data']['modeling_period_end'])

        earliest_training_start = pd.to_datetime(min(all_train_starts))
        latest_modeling_end = pd.to_datetime(max(all_model_ends))
        earliest_starting_balance_date = earliest_training_start - timedelta(days=1)

        # Get actual data boundaries
        profits_start = self.complete_profits_df.index.get_level_values('date').min()
        profits_end = self.complete_profits_df.index.get_level_values('date').max()
        market_data_start = self.complete_market_data_df.index.get_level_values('date').min()
        market_data_end = self.complete_market_data_df.index.get_level_values('date').max()

        if not (
            (profits_start <= earliest_starting_balance_date) and
            (profits_end >= latest_modeling_end) and
            (market_data_start <= earliest_starting_balance_date) and
            (market_data_end >= latest_modeling_end)
        ):
            raise ValueError(
                f"Insufficient data coverage for specified windows.\n"
                f"Required coverage: {earliest_starting_balance_date.strftime('%Y-%m-%d')}"
                f" to {latest_modeling_end.strftime('%Y-%m-%d')}\n"
                f"Actual coverage:\n"
                f"- Profits data: {profits_start.strftime('%Y-%m-%d')} to {profits_end.strftime('%Y-%m-%d')}\n"
                f"- Market data: {market_data_start.strftime('%Y-%m-%d')} to {market_data_end.strftime('%Y-%m-%d')}"
            )
