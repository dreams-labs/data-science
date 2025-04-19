"""Orchestrates the creation of training_data_dfs for multiple training epochs"""
import os
import logging
from pathlib import Path
import copy
from typing import List,Dict,Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
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

class MultiEpochOrchestrator:
    """
    Orchestrates training data generation across multiple epochs by
    offsetting base config dates and managing the resulting datasets.
    """
    def __init__(
        self,
        base_config: dict,
        metrics_config: dict,
        features_config: dict,
        epochs_config: dict,
        complete_profits_df: pd.DataFrame = None,
        complete_market_data_df: pd.DataFrame = None,
        complete_macro_trends_df: pd.DataFrame = None,
        hybrid_cw_id_map: Dict = None
    ):
        # Param Configs
        self.base_config = base_config
        self.metrics_config = metrics_config
        self.features_config = features_config
        self.epochs_config = epochs_config

        # Hybrid ID mapping
        self.hybrid_cw_id_map = hybrid_cw_id_map

        # Generated configs
        self.all_epochs_configs = self._generate_epoch_configs()

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


    @u.timing_decorator(logging.MILESTONE)  # pylint: disable=no-member
    def load_complete_raw_datasets(self) -> None:
        """
        Identifies the earliest training_period_start and latest modeling_period_end
        across all epochs, then retrieves the full profits and market data once,
        storing both in memory and in parquet files.

        Returns:
        - None
        """
        u.notify('robotz_windows_exit')

        # 1. Find earliest and latest window boundaries across all epochs
        all_train_starts = []
        all_validation_ends = []
        for cfg in self.all_epochs_configs:
            all_train_starts.extend(cfg['training_data']['training_window_starts'])
            all_validation_ends.append(cfg['training_data']['validation_period_end'])

        earliest_training_start = min(all_train_starts)
        latest_validation_end = max(all_validation_ends)

        # Retrieve the full data once (BigQuery or otherwise)
        logger.milestone("Pulling complete raw datasets from %s through %s...",
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


    @u.timing_decorator(logging.MILESTONE)  # pylint: disable=no-member
    def generate_epochs_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates training data for all epoch configs in parallel using multithreading.

        Returns:
        - merged_training_df: MultiIndexed on (wallet_address, epoch_start_date)
        - merged_modeling_df: MultiIndexed on (wallet_address, epoch_start_date)
        """
        # Ensure the complete dfs encompass the full range of training_epoch_starts
        self._assert_complete_coverage()

        logger.milestone(f"Compiling training data for {len(self.all_epochs_configs)} epochs...")
        u.notify('intro_3')

        training_epoch_dfs = {}
        modeling_epoch_dfs = {}

        # Set a suitable number of threads. You could retrieve this from config; here we use 8 as an example.
        max_workers = self.base_config['n_threads']['concurrent_epochs']
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit each epoch processing task to the executor
            futures = {executor.submit(self._process_single_epoch, cfg): cfg for cfg in self.all_epochs_configs}
            for future in as_completed(futures):
                epoch_date, epoch_training_data_df, epoch_modeling_features_df = future.result()
                training_epoch_dfs[epoch_date] = epoch_training_data_df
                modeling_epoch_dfs[epoch_date] = epoch_modeling_features_df

        # Merge the epoch DataFrames into a single DataFrame for training and modeling respectively
        wallet_training_data_df = self._merge_epoch_dfs(training_epoch_dfs)
        modeling_wallet_features_df = self._merge_epoch_dfs(modeling_epoch_dfs)

        # Confirm indices match
        u.assert_matching_indices(wallet_training_data_df, modeling_wallet_features_df)

        logger.info("Generated multi-epoch DataFrames with shapes %s and %s",
                    wallet_training_data_df.shape, modeling_wallet_features_df.shape)
        u.notify('level_up')

        return wallet_training_data_df, modeling_wallet_features_df


    # -----------------------------------
    #           Helper Methods
    # -----------------------------------

    def _process_single_epoch(self, epoch_config: dict) -> Tuple[datetime, pd.DataFrame, pd.DataFrame]:
        """
        Process a single epoch configuration to generate training and modeling data, including
        handling of hybridization features.

        Params:
        - epoch_config (dict): Configuration for the specific epoch

        Returns:
        - epoch_date (datetime): The modeling period start date as datetime
        - epoch_training_data_df (DataFrame): Training features for this epoch
        - epoch_modeling_features_df (DataFrame): Modeling features for this epoch
        """

        # Build features with initial data
        epoch_date, epoch_training_data_df, epoch_modeling_features_df = self._build_epoch_features(epoch_config)

        if epoch_config['training_data']['hybridize_wallet_ids']:
            print('x')

        return epoch_date, epoch_training_data_df, epoch_modeling_features_df



    def _build_epoch_features(self, epoch_config: dict) -> Tuple[datetime, pd.DataFrame, pd.DataFrame]:
        """
        Process a single epoch configuration to generate training and modeling data.

        Params:
        - epoch_config (dict): Configuration for the specific epoch

        Returns:
        - epoch_date (datetime): The modeling period start date as datetime
        - epoch_training_data_df (DataFrame): Training features for this epoch
        - epoch_modeling_features_df (DataFrame): Modeling features for this epoch
        """
        model_start = epoch_config['training_data']['modeling_period_start']
        logger.info(f"Generating data for epoch starting {model_start}...")
        u.notify('futuristic')

        # Generate name of parquet folder and create it if necessary
        epoch_parquet_folder = epoch_config['training_data']['parquet_folder']
        os.makedirs(epoch_parquet_folder, exist_ok=True)

        # 1. Initialize data generator for epoch with presplit dfs
        training_profits_df = None
        training_market_data_df = None
        training_macro_trends_df = None
        if self.complete_profits_df is not None:
            training_profits_df, training_market_data_df, training_macro_trends_df = \
                self._transform_complete_dfs_for_epoch(
                    epoch_config['training_data']['training_period_start'],
                    epoch_config['training_data']['training_period_end']
                )

        training_generator = wtdo.WalletTrainingDataOrchestrator(
            epoch_config,
            self.metrics_config,
            self.features_config,
            profits_df=training_profits_df,
            market_data_df=training_market_data_df,
            macro_trends_df=training_macro_trends_df
        )

        # 2. Generate TRAINING_DATA_DFs
        training_profits_df_full, training_market_data_df_full, \
        training_macro_trends_df_full, training_coin_cohort = \
            training_generator.retrieve_cleaned_period_datasets(
                epoch_config['training_data']['training_period_start'],
                epoch_config['training_data']['training_period_end']
            )

        training_profits_df, training_market_indicators_df, \
        training_macro_indicators_df, training_transfers_df = \
            training_generator.prepare_training_data(
                training_profits_df_full,
                training_market_data_df_full,
                training_macro_trends_df_full,
                return_files=True
            )

        epoch_training_data_df = training_generator.generate_training_features(
            training_profits_df,
            training_market_indicators_df,
            training_macro_indicators_df,
            training_transfers_df,
            return_files=True
        )

        # Store training df with epoch date
        epoch_date = datetime.strptime(
            epoch_config['training_data']['modeling_period_start'], '%Y-%m-%d')

        # 3. Generate MODELING_DATA_DFs
        modeling_profits_df, modeling_market_data_df, modeling_macro_trends_df = \
            self._transform_complete_dfs_for_epoch(
                epoch_config['training_data']['modeling_period_start'],
                epoch_config['training_data']['modeling_period_end']
            )

        modeling_generator = wtdo.WalletTrainingDataOrchestrator(
            epoch_config,
            self.metrics_config,
            self.features_config,
            training_wallet_cohort=training_generator.training_wallet_cohort,  # add cohort
            profits_df=modeling_profits_df,
            market_data_df=modeling_market_data_df,
            macro_trends_df=modeling_macro_trends_df
        )

        modeling_profits_df_full, _, _, _ = modeling_generator.retrieve_cleaned_period_datasets(
            epoch_config['training_data']['modeling_period_start'],
            epoch_config['training_data']['modeling_period_end'],
            training_coin_cohort
        )

        epoch_modeling_features_df = modeling_generator.prepare_modeling_features(
            modeling_profits_df_full,
            training_generator.hybrid_cw_id_map
        )
        logger.milestone(f"Successfully generated features for epoch {model_start}.")

        return epoch_date, epoch_training_data_df, epoch_modeling_features_df


    def _generate_epoch_configs(self) -> List[Dict]:
        """
        Generates config dicts for each offset epoch.

        Returns:
        - List[Dict]: List of config dicts, one per epoch.
        """
        all_epochs_configs = []
        base_training_data = self.base_config['training_data']
        offsets = self.epochs_config['offset_epochs']['offsets']

        for offset_days in offsets:
            # Deep copy base config to prevent mutations
            epoch_config = copy.deepcopy(self.base_config)

            # Offset key dates
            for date_key in [
                'modeling_period_start',
                'modeling_period_end',
                # 'validation_period_end'
            ]:
                base_date = datetime.strptime(base_training_data[date_key], '%Y-%m-%d')
                new_date = base_date + timedelta(days=offset_days)
                epoch_config['training_data'][date_key] = new_date.strftime('%Y-%m-%d')

            # Offset every date in training_window_starts
            epoch_config['training_data']['training_window_starts'] = [
                (datetime.strptime(dt, '%Y-%m-%d') + timedelta(days=offset_days)).strftime('%Y-%m-%d')
                for dt in base_training_data['training_window_starts']
            ]

            # Update parquet folder based on new modeling_period_start
            model_start = epoch_config['training_data']['modeling_period_start']
            folder_suffix = datetime.strptime(model_start, '%Y-%m-%d').strftime('%y%m%d')
            base_folder = Path(base_training_data['parquet_folder'])
            epoch_config['training_data']['parquet_folder'] = str(base_folder / folder_suffix)

            # Use WalletsConfig to add derived values
            epoch_config = wcm.add_derived_values(epoch_config)

            # Log epoch configuration
            logger.debug(
                f"Generated config for {offset_days} day offset epoch: "
                f"modeling_period={epoch_config['training_data']['modeling_period_start']} "
                f"to {epoch_config['training_data']['modeling_period_end']}"
            )

            all_epochs_configs.append(epoch_config)

        return all_epochs_configs


    def _transform_complete_dfs_for_epoch(self,
                                           epoch_start: str,
                                           epoch_end: str) -> pd.DataFrame:
        """
        Impute and filter the complete_profits_df to remove all rows after the period_end
        and impute the starting values as of the period_start starting balance date.

        This function relies on the WalletTrainingData.split_training_epoch_dfs function
        that is used to generate the profits_dfs for the multiple training period features.

        Params:
        - epoch_start (str): The training_period_start to use for the filtered df.
        - epoch_end (str): The last date to retain in the self.complete_profits_df
        """
        # Keep only records up to period_end
        epoch_profits_df = (self.complete_profits_df.copy()
                      [self.complete_profits_df.index.get_level_values('date') <= epoch_end])
        epoch_market_data_df = (self.complete_market_data_df.copy()
                                 [self.complete_market_data_df.index.get_level_values('date') <= epoch_end])
        epoch_macro_trends_df = (self.complete_macro_trends_df.copy()
                                 [self.complete_macro_trends_df.index.get_level_values('date') <= epoch_end])

        # Impute profits_df rows as of the period starting balance date and period end
        epoch_starting_balance_date = (pd.to_datetime(epoch_start) - timedelta(days=1)).strftime('%Y-%m-%d')
        epoch_profits_df = pri.impute_profits_for_multiple_dates(
            epoch_profits_df,
            epoch_market_data_df,
            [epoch_starting_balance_date,epoch_end],
            n_threads=4,
            reset_index=False
        )

        # Apply min_wallet_inflows filter
        valid_rows = (epoch_profits_df.xs(epoch_end, level='date')
                       ['usd_inflows_cumulative'] >= self.base_config['data_cleaning']['min_wallet_inflows'])
        valid_pairs = valid_rows[valid_rows].index
        epoch_profits_df = epoch_profits_df.loc[epoch_profits_df.index.droplevel('date').isin(valid_pairs)]

        # Create config with a training period of the param values
        splitter_config = copy.deepcopy(self.base_config)
        splitter_config['training_data']['training_window_starts'] = [epoch_start]
        splitter_config['training_data']['training_period_end'] = epoch_end

        # Initiate WalletTrainingData with the only window being the training_epoch_start
        complete_df_splitter = wtdo.WalletTrainingData(splitter_config)

        # Use the logic for splitting training epoch profits_dfs to split the complete profits_df
        epoch_profits_df = complete_df_splitter.split_training_window_dfs(u.ensure_index(epoch_profits_df))[0]

        return epoch_profits_df.reset_index(), epoch_market_data_df.reset_index(), epoch_macro_trends_df


    def _merge_epoch_dfs(self, epoch_dfs: Dict[datetime, pd.DataFrame]) -> pd.DataFrame:
        """
        Merges epoch DataFrames into a single MultiIndexed DataFrame with non-nullable indices.

        Params:
        - epoch_dfs: Dict mapping epoch dates to DataFrames

        Returns:
        - DataFrame: MultiIndexed on (wallet_address, epoch_start_date)
        """
        merged_dfs = []
        for epoch_date, df in epoch_dfs.items():
            # Convert wallet_address index to non-nullable int64
            wallet_index = df.index.astype("int64")

            # Build a MultiIndex from the non-nullable wallet_index and the epoch_date
            multi_idx = pd.MultiIndex.from_product(
                [wallet_index, [epoch_date]],
                names=['wallet_address', 'epoch_start_date']
            )

            # Use a copy to avoid modifying the original DataFrame
            epoch_df = df.copy()
            epoch_df.index = multi_idx

            merged_dfs.append(epoch_df)

        full_df = pd.concat(merged_dfs, axis=0).sort_index()

        return full_df


    # def _merge_hybrid_wallet_features(
    #     self,
    #     hybrid_df: pd.DataFrame,
    #     wallet_df: pd.DataFrame,
    #     hybrid_cw_id_map: Dict[int, tuple],
    # ) -> pd.DataFrame:
    #     """
    #     For every hybrid wallet‑coin row, append the wallet‑level (/wallet‑suffixed)
    #     columns that describe the whole portfolio.
    #     """
    #     reverse_map = {v: k for k, v in hybrid_cw_id_map.items()}
    #     wallet_idx = hybrid_df.index.map(lambda h: reverse_map[h][0])

    #     # Ensure wallet-level features exist for all hybrid ID pairs
    #     missing_wallet_ids = set(wallet_idx) - set(wallet_df.index)
    #     if missing_wallet_ids:
    #         raise ValueError(f"Missing wallet-level features for hybrid wallet IDs: {missing_wallet_ids}")

    #     return (
    #         hybrid_df
    #         .assign(_wallet_id=wallet_idx)
    #         .join(wallet_df.add_prefix("wallet_"), on="_wallet_id", how="left")
    #         .drop(columns="_wallet_id")
    # )


    # def _generate_hybrid_and_wallet_features(
    #     self,
    #     epoch_config: dict,
    #     training_generator: wtdo.WalletTrainingDataOrchestrator,
    #     hybrid_training_df: pd.DataFrame,
    #     training_profits_df_full: pd.DataFrame,
    #     training_market_data_df_full: pd.DataFrame,
    #     training_macro_trends_df_full: pd.DataFrame,
    #     modeling_profits_df_full: pd.DataFrame,
    #     hybrid_modeling_df: pd.DataFrame
    # ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #     """
    #     Runs a second (non‑hybridized) wallet‑level feature pass for the same
    #     cohort and merges those columns onto the hybrid wallet‑coin rows.
    #     """
    #     # ——— wallet cohort from hybrid map ———
    #     hybrid_map = training_generator.hybrid_cw_id_map
    #     if hybrid_map is None:
    #         raise ValueError("Hybrid cw‑id map missing; cannot merge wallet‑level features.")

    #     reverse_map = {v: k for k, v in hybrid_map.items()}
    #     wallet_cohort = list(set([reverse_map[h][0] for h in training_generator.training_wallet_cohort]))

    #     # ——— wallet‑ID pass config ———
    #     dehybridized_config = copy.deepcopy(epoch_config)
    #     dehybridized_config['training_data']['hybridize_wallet_ids'] = False
    #     dehybridized_config['training_data']['parquet_folder'] = (
    #         f"{dehybridized_config['training_data']['parquet_folder']}_wallet"
    #     )
    #     os.makedirs(dehybridized_config['training_data']['parquet_folder'], exist_ok=True)

    #     wallet_gen = wtdo.WalletTrainingDataOrchestrator(
    #         dehybridized_config,
    #         self.metrics_config,
    #         self.features_config,
    #         training_wallet_cohort=wallet_cohort,
    #         profits_df=training_profits_df_full.copy(),
    #         market_data_df=training_market_data_df_full.copy(),
    #         macro_trends_df=training_macro_trends_df_full.copy()
    #     )

    #     # wallet‑level training features
    #     w_profits, w_mkt_ind, w_macro_ind, w_tx = wallet_gen.prepare_training_data(
    #         training_profits_df_full,
    #         training_market_data_df_full,
    #         training_macro_trends_df_full,
    #         return_files=True
    #     )
    #     wallet_train_df = wallet_gen.generate_training_features(
    #         w_profits, w_mkt_ind, w_macro_ind, w_tx, return_files=True
    #     )

    #     # wallet‑level modeling features
    #     wallet_model_df = wallet_gen.prepare_modeling_features(modeling_profits_df_full)

    #     # ——— merge & return ———
    #     merged_train = self._merge_hybrid_wallet_features(
    #         hybrid_training_df, wallet_train_df, hybrid_map
    #     )
    #     merged_model = self._merge_hybrid_wallet_features(
    #         hybrid_modeling_df, wallet_model_df, hybrid_map
    #     )
    #     return merged_train, merged_model



    # -----------------------------------
    #           Utility Methods
    # -----------------------------------

    def _assert_complete_coverage(self) -> None:
        """
        Verify that profits and market data fully cover all training epochs.

        Raises:
        - ValueError: If data coverage is incomplete with specific boundary details
        """
        # Find earliest and latest boundaries
        all_train_starts = []
        all_model_ends = []
        for cfg in self.all_epochs_configs:
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
                f"Insufficient data coverage for specified epochs.\n"
                f"Required coverage: {earliest_starting_balance_date.strftime('%Y-%m-%d')}"
                f" to {latest_modeling_end.strftime('%Y-%m-%d')}\n"
                f"Actual coverage:\n"
                f"- Profits data: {profits_start.strftime('%Y-%m-%d')} to {profits_end.strftime('%Y-%m-%d')}\n"
                f"- Market data: {market_data_start.strftime('%Y-%m-%d')} to {market_data_end.strftime('%Y-%m-%d')}"
            )
