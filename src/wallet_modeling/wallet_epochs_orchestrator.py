"""Orchestrates the creation of training_data_dfs for multiple training epochs"""
import os
import time
import logging
import gc
from pathlib import Path
import copy
from typing import List,Dict,Tuple,Set,Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

# Local module imports
import training_data.profits_row_imputation as pri
from wallet_modeling.wallet_training_data import WalletTrainingData
import wallet_modeling.wallets_config_manager as wcm
import wallet_modeling.wallet_training_data_orchestrator as wtdo
import base_modeling.pipeline as bp
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)


# ----------------------------------------
#       Primary Orchestration Class
# ----------------------------------------

class WalletEpochsOrchestrator:
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
        complete_hybrid_cw_id_df: pd.DataFrame = None
    ):
        # Param Configs
        self.base_config = base_config
        self.metrics_config = metrics_config
        self.features_config = features_config
        self.epochs_config = epochs_config

        # Generated configs
        self.all_epochs_configs = self.generate_epoch_configs()

        # Complete df objects
        self.complete_profits_df = complete_profits_df
        self.complete_market_data_df = complete_market_data_df
        self.complete_macro_trends_df = complete_macro_trends_df
        self.complete_hybrid_cw_id_df = complete_hybrid_cw_id_df
        self.wtd = WalletTrainingData(self.base_config)  # helper for complete df generation

        # Create hybrid ID mapping if configured and able
        self.complete_hybrid_cw_id_df = None
        if self.base_config['training_data']['hybridize_wallet_ids'] and self.complete_profits_df is not None:
            self.complete_hybrid_cw_id_df = self.create_hybrid_mapping()

        # Confirm all pairs in profits_df have a hybrid mapping
        if self.complete_hybrid_cw_id_df is not None:
            wtdo.validate_hybrid_mapping_completeness(
                self.complete_profits_df,
                self.complete_hybrid_cw_id_df
            )

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
        logger.milestone("<%s> Pulling complete raw datasets from %s through %s...",
                         self.base_config['training_data']['dataset'].upper(),
                         earliest_training_start, latest_validation_end)
        (
            self.complete_profits_df,
            self.complete_market_data_df,
            self.complete_macro_trends_df,
        ) = self.wtd.retrieve_raw_datasets(
            earliest_training_start,
            latest_validation_end
        )

        # Set index
        self.complete_profits_df = u.ensure_index(self.complete_profits_df)
        self.complete_market_data_df = u.ensure_index(self.complete_market_data_df)

        # Create hybrid mapping if configured
        if self.base_config['training_data']['hybridize_wallet_ids']:
            self.complete_hybrid_cw_id_df = self.create_hybrid_mapping()

        # Save them to parquet for future reuse
        parquet_folder = self.base_config['training_data']['parquet_folder']
        os.makedirs(parquet_folder, exist_ok=True)
        self.complete_profits_df.to_parquet(f"{parquet_folder}/complete_profits_df.parquet")
        self.complete_market_data_df.to_parquet(f"{parquet_folder}/complete_market_data_df.parquet")
        self.complete_macro_trends_df.to_parquet(f"{parquet_folder}/complete_macro_trends_df.parquet")
        if not getattr(self.complete_hybrid_cw_id_df, "empty", True):
            self.complete_hybrid_cw_id_df.to_parquet(f"{parquet_folder}/complete_hybrid_cw_id_df.parquet")

        parts = [
            f"complete profits ({len(self.complete_profits_df)} rows)",
            f"market data ({len(self.complete_market_data_df)} rows)",
            f"macro trends ({len(self.complete_macro_trends_df)} rows)",
        ]
        # only log hybrid ID mappings if there's rows in the df
        if not getattr(self.complete_hybrid_cw_id_df, "empty", True):
            parts.append(f"hybrid id mappings ({len(self.complete_hybrid_cw_id_df)} rows)")

        log_msg = "Saved " + ", ".join(parts) + f" to {parquet_folder}."
        logger.info(log_msg)


    @u.timing_decorator(logging.MILESTONE)  # pylint: disable=no-member
    def generate_epochs_training_data(self, training_only: bool = False
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generates epoch‑level training data.  When `training_only=True`, it returns **only** the merged
        `wallet_training_data_df` (historical training snapshots) and skips all future‑dated data
        generation.

        Returns:
        - wallet_training_data_df (always returned)

        When `training_only=False` it also returns:
        - wallet_target_vars_df: MultiIndexed on (wallet_address, epoch_start_date) for modeling epochs
        - validation_training_data_df: MultiIndexed on (wallet_address, epoch_start_date) for validation epochs
        - validation_target_vars_df: MultiIndexed on (wallet_address, epoch_start_date) for validation epochs
        """
        logger.milestone(f"Compiling wallet training data for {len(self.all_epochs_configs)} epochs...")
        if training_only:
            logger.milestone("Training‑only mode: Compiling wallet training data without validation or target variables.")

        u.notify('intro_3')

        training_modeling_dfs = {}
        if not training_only:
            # Ensure the complete dfs encompass the full range of training_epoch_starts
            self._assert_complete_coverage()

            modeling_modeling_dfs = {}
            training_validation_dfs = {}
            modeling_validation_dfs = {}

        # Set a suitable number of threads. You could retrieve this from config; here we use 8 as an example.
        max_workers = self.base_config['n_threads']['concurrent_epochs']
        i = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Skip validation epochs entirely in training‑only mode
            epoch_configs_to_process = (
                [cfg for cfg in self.all_epochs_configs if cfg.get('epoch_type') == 'modeling']
                if training_only else self.all_epochs_configs
            )
            futures = {
                executor.submit(self._process_single_epoch, cfg, training_only): cfg
                for cfg in epoch_configs_to_process
            }
            for future in as_completed(futures):
                cfg = futures[future]
                epoch_date, epoch_training_df, epoch_modeling_df = future.result()

                # Store data
                if cfg.get('epoch_type') == 'modeling':
                    training_modeling_dfs[epoch_date] = epoch_training_df
                    if not training_only:
                        modeling_modeling_dfs[epoch_date] = epoch_modeling_df
                elif not training_only and cfg.get('epoch_type') == 'validation':
                    training_validation_dfs[epoch_date] = epoch_training_df #pylint:disable=possibly-used-before-assignment
                    modeling_validation_dfs[epoch_date] = epoch_modeling_df #pylint:disable=possibly-used-before-assignment

                i += 1
                logger.milestone(f"Wallet epoch {i}/{len(epoch_configs_to_process)} completed (date: " \
                                 f"{pd.to_datetime(epoch_date).strftime('%Y-%m-%d')})")

        del epoch_training_df, epoch_modeling_df
        gc.collect()

        wallet_training_data_df = self._merge_epoch_dfs(training_modeling_dfs)

        # Training‑only fast path
        if training_only:
            logger.milestone(
                "Generated historical wallet training snapshots with shape: %s",
                wallet_training_data_df.shape,
            )
            return wallet_training_data_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Full four‑tuple path
        validation_training_data_df   = (self._merge_epoch_dfs(training_validation_dfs)
                                         if training_validation_dfs else pd.DataFrame())
        wallet_target_vars_df   = self._merge_epoch_dfs(modeling_modeling_dfs)
        validation_target_vars_df = (self._merge_epoch_dfs(modeling_validation_dfs)
                                         if modeling_validation_dfs else pd.DataFrame())

        # Confirm indices match
        u.assert_matching_indices(wallet_training_data_df, wallet_target_vars_df)

        logger.milestone(
            "Generated wallet multi‑epoch DataFrames with shapes:\n"
            " - modeling train: %s\n"
            " - modeling model: %s\n"
            " - validation train: %s\n"
            " - validation model: %s",
            wallet_training_data_df.shape,
            wallet_target_vars_df.shape,
            validation_training_data_df.shape,
            validation_target_vars_df.shape
        )
        u.notify('level_up')

        return (
            wallet_training_data_df,
            wallet_target_vars_df,
            validation_training_data_df,
            validation_target_vars_df
        )



    # -----------------------------------
    #           Helper Methods
    # -----------------------------------

    def _process_single_epoch(
            self,
            epoch_config: dict,
            training_only: bool = False
        ) -> Tuple[datetime, pd.DataFrame, pd.DataFrame]:
        """
        Process a single epoch configuration to generate training and modeling data, including
        handling of hybridization features.

        Params:
        - epoch_config (dict): Configuration for the specific epoch

        Returns:
        - epoch_date (datetime): The modeling period start date as datetime
        - epoch_training_data_df (DataFrame): Training features for this epoch
        - epoch_modeling_data_df (DataFrame): Modeling features for this epoch
        """
        generate_modeling = not training_only

        # Short-circuit: if dfs are already saved, just load them
        output_folder = epoch_config['training_data']['parquet_folder']
        training_path = f"{output_folder}/training_data_df.parquet"
        modeling_path = f"{output_folder}/modeling_data_df.parquet"
        if os.path.exists(training_path) and (not generate_modeling or os.path.exists(modeling_path)):
            logger.info(
                f"Loading precomputed features for epoch starting "
                f"{epoch_config['training_data']['modeling_period_start']} from {output_folder}"
            )
            epoch_date = datetime.strptime(
                epoch_config['training_data']['modeling_period_start'], '%Y-%m-%d'
            )
            epoch_training_data_df = pd.read_parquet(training_path)
            if generate_modeling:
                epoch_modeling_data_df = pd.read_parquet(modeling_path)
            else:
                epoch_modeling_data_df = pd.DataFrame()

            # Drop columns before returning if configured
            if epoch_config['training_data']['predrop_features']:
                drop_patterns=epoch_config['modeling']['feature_selection']['drop_patterns']
                col_dropper = bp.DropColumnPatterns(drop_patterns)
                epoch_training_data_df = col_dropper.fit_transform(epoch_training_data_df)

            return epoch_date, epoch_training_data_df, epoch_modeling_data_df

        # Begin generation of dfs
        model_start = epoch_config['training_data']['modeling_period_start']
        logger.info(f"Generating data for epoch with modeling_period_start of {model_start}...")
        u.notify('futuristic')

        # If using hybrid IDs, override for a non-hybridized run to define the wallet cohort
        epoch_config['training_data']['hybridize_wallet_ids'] = False

        # Build features with initial data
        epoch_date, epoch_training_data_df, epoch_modeling_data_df, cohorts = self._build_epoch_features(
            epoch_config,
            generate_modeling=generate_modeling
        )

        # If using hybrid IDs, generate hybridized feature set using IDs from the original cohorts
        if self.base_config['training_data']['hybridize_wallet_ids']:
            epoch_config['training_data']['hybridize_wallet_ids'] = True
            logger.info(f"Generating hybridized data for epoch starting {model_start}...")

            # Get the corresponding hybrid IDs for the nonhybridized wallet cohort
            hybridized_cohort = (self.complete_hybrid_cw_id_df
                                 [self.complete_hybrid_cw_id_df['wallet_address'].isin(cohorts['wallet_cohort'])]
                                 ['hybrid_cw_id'])
            cohorts['wallet_cohort'] = hybridized_cohort

            # Build features with hybrid IDs
            (
                _, hybridized_training_data_df, hybridized_modeling_data_df, _
            ) = self._build_epoch_features(epoch_config, cohorts, generate_modeling)

            # Merge hybrid/nonhybrid training and modeling dfs
            epoch_training_data_df = self._merge_hybrid_dfs(epoch_training_data_df,hybridized_training_data_df)

            # Downcast all features
            epoch_training_data_df = u.df_downcast(epoch_training_data_df)
            # Save features
            output_folder = f"{epoch_config['training_data']['parquet_folder']}/"
            epoch_training_data_df.to_parquet(f"{output_folder}/training_data_df.parquet", index=True)

            if generate_modeling:
                epoch_modeling_data_df = self._merge_hybrid_dfs(epoch_modeling_data_df,hybridized_modeling_data_df)
                # Remove modeling-only IDs - these represent wallets in the cohort but
                #  coin-wallet pairs that only became active during the modeling period
                epoch_modeling_data_df = (epoch_modeling_data_df[
                    epoch_modeling_data_df.index.isin(epoch_training_data_df.index.values)
                ])
                u.assert_matching_indices(epoch_training_data_df,epoch_modeling_data_df)
                # Downcast all features
                epoch_modeling_data_df = u.df_downcast(epoch_modeling_data_df)
                # Save features
                epoch_modeling_data_df.to_parquet(f"{output_folder}/modeling_data_df.parquet", index=True)
            logger.info(f"Saved {model_start} features to %s.", output_folder)

        # Drop columns before returning if configured
        if epoch_config['training_data']['predrop_features']:
            drop_patterns=epoch_config['modeling']['feature_selection']['drop_patterns']
            col_dropper = bp.DropColumnPatterns(drop_patterns)
            epoch_training_data_df = col_dropper.fit_transform(epoch_training_data_df)

        if training_only:
            epoch_modeling_data_df = pd.DataFrame()
        return epoch_date, epoch_training_data_df, epoch_modeling_data_df



    def _build_epoch_features(
        self,
        epoch_config: dict,
        cohorts: Optional[Dict[Set, np.array]] = None,
        generate_modeling: bool = True,
    ) -> Tuple[datetime, pd.DataFrame, pd.DataFrame, Dict[Set, np.array]]:
        """
        Process a single epoch configuration to generate training and modeling data.

        - generate_modeling (bool): If False, skip creation of modeling features and return an
            empty DataFrame for that output.

        Params:
        - epoch_config (dict): Configuration for the specific epoch
        - cohorts (Dict[Set,np.array]): Optional predefinition of coin and wallet cohorts

        Returns:
        - epoch_date (datetime): The modeling period start date as datetime
        - epoch_training_data_df (DataFrame): Training features for this epoch
        - epoch_modeling_data_df (DataFrame): Modeling features for this epoch
        - cohorts (Dict[Set,np.array]): The wallets and coins included in the training data
        """
        model_start = epoch_config['training_data']['modeling_period_start']
        if not generate_modeling:
            logger.info("Training‑only mode enabled: modeling features will be skipped for this epoch.")
        if cohorts is None:
            cohorts = {}

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
            training_wallet_cohort=cohorts.get('wallet_cohort'),  # use predefined cohort if provided
            profits_df=training_profits_df,
            market_data_df=training_market_data_df,
            macro_trends_df=training_macro_trends_df,
            complete_hybrid_cw_id_df=self.complete_hybrid_cw_id_df
        )

        training_profits_df_full, training_market_data_df_full, \
        training_macro_trends_df_full, training_coin_cohort = \
            training_generator.retrieve_cleaned_period_datasets(
                epoch_config['training_data']['training_period_start'],
                epoch_config['training_data']['training_period_end'],
                cohorts.get('coin_cohort')  # use predefined cohort if provided
            )

        # 2. Prepare training and modeling feature generation inputs
        training_profits_df, training_market_indicators_df, \
        training_macro_indicators_df, training_transfers_df = \
            training_generator.prepare_training_data(
                training_profits_df_full,
                training_market_data_df_full,
                training_macro_trends_df_full,
                return_files=True
            )

        if generate_modeling:
            modeling_profits_df, modeling_market_data_df, modeling_macro_trends_df = \
                self._transform_complete_dfs_for_epoch(
                    epoch_config['training_data']['modeling_period_start'],
                    epoch_config['training_data']['modeling_period_end'],
                )

            modeling_generator = wtdo.WalletTrainingDataOrchestrator(
                epoch_config,
                self.metrics_config,
                self.features_config,
                training_wallet_cohort=training_generator.training_wallet_cohort,  # add cohort
                profits_df=modeling_profits_df,
                market_data_df=modeling_market_data_df,
                macro_trends_df=modeling_macro_trends_df,
                complete_hybrid_cw_id_df=self.complete_hybrid_cw_id_df
            )

            modeling_profits_df_full, _, _, _ = modeling_generator.retrieve_cleaned_period_datasets(
                epoch_config['training_data']['modeling_period_start'],
                epoch_config['training_data']['modeling_period_end'],
                training_coin_cohort
            )

        # 3. Concurrently generate training and modeling features
        with ThreadPoolExecutor(max_workers=self.base_config['n_threads']['epoch_tm_features']) as executor:
            train_future = executor.submit(
                training_generator.generate_training_features,
                training_profits_df,
                training_market_indicators_df,
                training_macro_indicators_df,
                training_transfers_df
            )
            epoch_training_data_df = train_future.result()

            # Generate modeling data if configured; otherwise return empty df
            epoch_modeling_data_df = pd.DataFrame()
            if generate_modeling:
                model_future = executor.submit(
                    modeling_generator.prepare_modeling_features,
                    modeling_profits_df_full,
                    self.complete_hybrid_cw_id_df
                )
                epoch_modeling_data_df = (
                    model_future.result() if generate_modeling else pd.DataFrame()
                )

        # Store training df with epoch date
        epoch_date = datetime.strptime(
            epoch_config['training_data']['modeling_period_start'], '%Y-%m-%d')

        cohorts = {
            'coin_cohort': training_coin_cohort,
            'wallet_cohort': training_generator.training_wallet_cohort
        }

        # Save profits_dfs
        start_time = time.time()
        output_folder = f"{epoch_config['training_data']['parquet_folder']}/"
        training_profits_df.to_parquet(f"{output_folder}/training_profits_df.parquet", index=True)
        if generate_modeling:
            modeling_profits_df.to_parquet(f"{output_folder}/modeling_profits_df.parquet", index=True)
        logger.info("(%.1fs) Saved %s profits_df files to %s.",
            time.time() - start_time, model_start, output_folder)

        return epoch_date, epoch_training_data_df, epoch_modeling_data_df, cohorts


    def _assert_no_epoch_overlap(self, modeling_offsets: Set[int], validation_offsets: Set[int]) -> None:
        """
        Checks for overlap between modeling and validation epochs.
        Raises ValueError if the latest modeling end date is after
        the earliest validation start date.
        """
        modeling_latest_offset = max(modeling_offsets)
        modeling_latest_end_date = (
            pd.to_datetime(self.base_config['training_data']['modeling_period_end'])
            + timedelta(days=modeling_latest_offset)
        )

        validation_earliest_offset = min(validation_offsets)
        validation_earliest_start_date = (
            pd.to_datetime(self.base_config['training_data']['modeling_period_start'])
            + timedelta(days=validation_earliest_offset)
        )

        if modeling_latest_end_date > validation_earliest_start_date:
            raise ValueError(
                f"Latest modeling end date as of {modeling_latest_end_date.strftime('%Y-%m-%d')} "
                f"is later than earliest validation start date as of "
                f"{validation_earliest_start_date.strftime('%Y-%m-%d')}."
            )


    def build_epoch_config(
        self,
        offset_days: int,
        epoch_type: str,
        base_modeling_start: datetime,
        base_modeling_end: datetime,
        base_window_starts: List[datetime],
        base_parquet_folder_base: Path
    ) -> Dict:
        """
        Build a single epoch config by offsetting dates and updating folder based on offset.

        Public method that is also used by the CoinEpochsOrchestrator.
        """
        epoch_config = copy.deepcopy(self.base_config)
        epoch_config['epoch_type'] = epoch_type

        # Offset key dates
        new_start = (base_modeling_start + timedelta(days=offset_days)).strftime('%Y-%m-%d')
        new_end = (base_modeling_end + timedelta(days=offset_days)).strftime('%Y-%m-%d')
        epoch_config['training_data']['modeling_period_start'] = new_start
        epoch_config['training_data']['modeling_period_end'] = new_end

        # Offset every date in training_window_starts
        epoch_config['training_data']['training_window_starts'] = [
            (dt + timedelta(days=offset_days)).strftime('%Y-%m-%d')
            for dt in base_window_starts
        ]

        # Update parquet folder based on new modeling_period_start
        folder_suffix = datetime.strptime(new_start, '%Y-%m-%d').strftime('%y%m%d')
        epoch_config['training_data']['parquet_folder'] = str(base_parquet_folder_base / folder_suffix)

        # Add derived values
        return wcm.add_derived_values(epoch_config)



    def generate_epoch_configs(self) -> List[Dict]:
        """
        Generates config dicts for each offset epoch, including modeling and validation epochs.

        Also used by CoinEpochsOrchestrator.

        Returns:
        - List[Dict]: List of config dicts, one per epoch.
        """
        all_epochs_configs = []
        base_training_data = self.base_config['training_data']

        # Cache parsed base dates and base folder path
        base_modeling_start = datetime.strptime(base_training_data['modeling_period_start'], '%Y-%m-%d')
        base_modeling_end = datetime.strptime(base_training_data['modeling_period_end'], '%Y-%m-%d')
        base_training_window_starts = [
            datetime.strptime(dt, '%Y-%m-%d')
            for dt in base_training_data['training_window_starts']
        ]
        base_parquet_folder_base = Path(base_training_data['parquet_folder'])

        # Identify all offsets
        validation_offsets = self.epochs_config['offset_epochs'].get('validation_offsets', [])
        modeling_offsets = self.epochs_config['offset_epochs']['offsets']

        # Confirm there is no overlap between any modeling and validation periods
        if len(validation_offsets) > 0:
            self._assert_no_epoch_overlap(modeling_offsets, validation_offsets)

        # Generate modeling epoch configs
        for offset_days in modeling_offsets:
            all_epochs_configs.append(
                self.build_epoch_config(
                    offset_days, 'modeling',
                    base_modeling_start, base_modeling_end,
                    base_training_window_starts, base_parquet_folder_base
                )
            )

        # Add validation epoch configs if configured
        if self.base_config['training_data'].get('validation_period_end') is not None:
            for offset_days in validation_offsets:
                cfg = self.build_epoch_config(
                    offset_days, 'validation',
                    base_modeling_start, base_modeling_end,
                    base_training_window_starts, base_parquet_folder_base
                )
                all_epochs_configs.append(cfg)

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
        epoch_profits_df = (self.complete_profits_df.copy(deep=True)
                      [self.complete_profits_df.index.get_level_values('date') <= epoch_end])
        epoch_market_data_df = (self.complete_market_data_df.copy(deep=True)
                                 [self.complete_market_data_df.index.get_level_values('date') <= epoch_end])
        epoch_macro_trends_df = (self.complete_macro_trends_df.copy(deep=True)
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


    @u.timing_decorator
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


    def create_hybrid_mapping(self) -> pd.DataFrame:
        """
        Create hybrid wallet-coin ID mapping if configured in the base config.

        Also used by CoinEpochsOrchestrator.
        """
        if self.base_config['training_data']['hybridize_wallet_ids']:
            hybrid_df = (
                self.complete_profits_df
                    .reset_index()[['coin_id', 'wallet_address']]
                    .drop_duplicates()
            )
            hybrid_df['hybrid_cw_id'] = hybrid_df.index.values + 3e9
            if not hybrid_df['hybrid_cw_id'].is_unique:
                raise ValueError("Duplicate hybrid_cw_ids found, confirm index is not malformed.")

            return hybrid_df


    def _merge_hybrid_dfs(
        self,
        base_df: pd.DataFrame,
        hybrid_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        For every hybrid wallet‑coin row, append the wallet‑level feature
        columns that describe the full portfolio behavior.
        """
        # Add 'wc_' prefix to differentiate from wallet-level features
        hybrid_df = hybrid_df.add_prefix('cw_')

        # Merge to the base wallet_address value
        hybrid_df['hybrid_cw_id'] = hybrid_df.index.values
        hybrid_df = hybrid_df.merge(
            self.complete_hybrid_cw_id_df[['hybrid_cw_id','wallet_address']]
            ,how='left'
            ,on='hybrid_cw_id'
        )

        # Confirm every hybrid_cw_id got a wallet_address
        missing = hybrid_df['wallet_address'].isna()
        if missing.any():
            bad_ids = hybrid_df.loc[missing, 'hybrid_cw_id'].tolist()
            raise ValueError(f"Missing wallet_address for {len(bad_ids)} hybrid_cw_ids: {bad_ids[:10]}")


        # Merge to the base wallet_address features
        hybrid_df = hybrid_df.merge(
            base_df.reset_index(),
            how='left'
            ,on='wallet_address'
        )

        # Reapply index and drop helper columns
        hybrid_df = hybrid_df.set_index(hybrid_df['hybrid_cw_id'])
        hybrid_df = hybrid_df.drop(['hybrid_cw_id','wallet_address'],axis=1)

        return hybrid_df



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
        macro_trends_start = self.complete_macro_trends_df.index.get_level_values('date').min()
        macro_trends_end = self.complete_macro_trends_df.index.get_level_values('date').max()

        if not (
            (profits_start <= earliest_starting_balance_date) and
            (profits_end >= latest_modeling_end) and
            (market_data_start <= earliest_starting_balance_date) and
            (market_data_end >= latest_modeling_end) and
            (macro_trends_start <= earliest_starting_balance_date) and
            (macro_trends_end >= latest_modeling_end)
        ):
            raise ValueError(
                f"Insufficient wallet data coverage for specified epochs.\n"
                f"Required coverage: {earliest_starting_balance_date.strftime('%Y-%m-%d')}"
                    f" to {latest_modeling_end.strftime('%Y-%m-%d')}\n"
                f"Actual coverage:\n"
                f"- Profits data: {profits_start.strftime('%Y-%m-%d')} to {profits_end.strftime('%Y-%m-%d')}\n"
                f"- Market data: {market_data_start.strftime('%Y-%m-%d')} to {market_data_end.strftime('%Y-%m-%d')}\n"
                f"- Macro trends data: {macro_trends_start.strftime('%Y-%m-%d')} "
                    f"to {macro_trends_end.strftime('%Y-%m-%d')}"
            )

        # Confirm we have hybrid mappings for all pairs if applicable
        if self.base_config['training_data']['hybridize_wallet_ids']:
            if not isinstance(self.complete_hybrid_cw_id_df, pd.DataFrame):
                raise TypeError("complete_hybrid_cw_id_df must be a pandas DataFrame")
            if self.complete_hybrid_cw_id_df.empty:
                raise ValueError("complete_hybrid_cw_id_df cannot be empty")

            # Extract unique coin-wallet pairs from profits dataframe
            profits_pairs = set(zip(self.complete_profits_df.reset_index()['coin_id'],
                                    self.complete_profits_df.reset_index()['wallet_address']))

            # Extract unique coin-wallet pairs from hybrid dataframe
            hybrid_pairs = set(zip(self.complete_hybrid_cw_id_df['coin_id'],
                                   self.complete_hybrid_cw_id_df['wallet_address']))

            # Find pairs in profits that are not in hybrid
            missing_pairs = profits_pairs - hybrid_pairs

            if len(missing_pairs) > 0:
                raise ValueError(f"Found {len(missing_pairs)} coin-wallet pairs in profits_df "
                                 "without corresponding hybrid IDs.")
