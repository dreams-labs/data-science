"""Orchestrates the creation of training_data_dfs for multiple training epochs"""
import os
import logging
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
    ):
        # Param Configs
        self.base_config = base_config
        self.metrics_config = metrics_config
        self.features_config = features_config
        self.epochs_config = epochs_config

        # Confirm the validation_period_end is after the latest modeling_period_end
        modeling_end = pd.to_datetime(self.base_config['training_data']['modeling_period_end'])
        validation_end = pd.to_datetime(self.base_config['training_data']['validation_period_end'])
        latest_offset = pd.Series(self.epochs_config['offset_epochs'].get('validation_offsets')).max()
        latest_modeling_end = modeling_end + timedelta(days = int(latest_offset))
        if latest_modeling_end > validation_end:
            raise ValueError(f"Invalid config settings: latest epoch's modeling end of {latest_modeling_end} "
                             f"is later than the validation end of {validation_end}.")


        # Generated configs
        self.all_epochs_configs = self._generate_epoch_configs()

        # Complete df objects
        self.complete_profits_df = complete_profits_df
        self.complete_market_data_df = complete_market_data_df
        self.complete_macro_trends_df = complete_macro_trends_df
        self.wtd = WalletTrainingData(self.base_config)  # helper for complete df generation

        # Create hybrid ID mapping if configured and able
        self.complete_hybrid_cw_id_df = None
        if self.base_config['training_data']['hybridize_wallet_ids'] and self.complete_profits_df is not None:
            self.complete_hybrid_cw_id_df = self._create_hybrid_mapping()

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
            self.complete_hybrid_cw_id_df = self._create_hybrid_mapping()

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
    def generate_epochs_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generates training data for all epoch configs in parallel using multithreading.

        Returns:
        - wallet_training_data_df: MultiIndexed on (wallet_address, epoch_start_date) for modeling epochs
        - modeling_wallet_features_df: MultiIndexed on (wallet_address, epoch_start_date) for modeling epochs
        - validation_training_data_df: MultiIndexed on (wallet_address, epoch_start_date) for validation epochs
        - validation_wallet_features_df: MultiIndexed on (wallet_address, epoch_start_date) for validation epochs
        """
        # Ensure the complete dfs encompass the full range of training_epoch_starts
        self._assert_complete_coverage()

        logger.milestone(f"Compiling training data for {len(self.all_epochs_configs)} epochs...")
        u.notify('intro_3')

        training_modeling_dfs = {}
        modeling_modeling_dfs = {}
        training_validation_dfs = {}
        modeling_validation_dfs = {}

        # Set a suitable number of threads. You could retrieve this from config; here we use 8 as an example.
        max_workers = self.base_config['n_threads']['concurrent_epochs']
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit each epoch processing task to the executor
            futures = {executor.submit(self._process_single_epoch, cfg): cfg for cfg in self.all_epochs_configs}
            for future in as_completed(futures):
                cfg = futures[future]
                epoch_date, epoch_training_df, epoch_modeling_df = future.result()

                # Downcast dtypes
                epoch_training_df = u.df_downcast(epoch_training_df)
                epoch_modeling_df = u.df_downcast(epoch_modeling_df)

                # Store data in dicts
                if cfg.get('epoch_type') == 'validation':
                    training_validation_dfs[epoch_date] = epoch_training_df
                    modeling_validation_dfs[epoch_date] = epoch_modeling_df
                else:
                    training_modeling_dfs[epoch_date] = epoch_training_df
                    modeling_modeling_dfs[epoch_date] = epoch_modeling_df

        # Merge the epoch DataFrames into a single DataFrame for training and modeling respectively
        wallet_training_data_df       = self._merge_epoch_dfs(training_modeling_dfs)
        modeling_wallet_features_df   = self._merge_epoch_dfs(modeling_modeling_dfs)
        validation_training_data_df   = (self._merge_epoch_dfs(training_validation_dfs)
                                         if training_validation_dfs else pd.DataFrame())
        validation_wallet_features_df = (self._merge_epoch_dfs(modeling_validation_dfs)
                                         if modeling_validation_dfs else pd.DataFrame())

        # Confirm indices match
        u.assert_matching_indices(wallet_training_data_df, modeling_wallet_features_df)

        logger.milestone(
            "Generated multi-epoch DataFrames with shapes:\n"
            " - modeling train: %s\n"
            " - modeling model: %s\n"
            " - validation train: %s\n"
            " - validation model: %s",
            wallet_training_data_df.shape,
            modeling_wallet_features_df.shape,
            validation_training_data_df.shape,
            validation_wallet_features_df.shape
        )
        u.notify('level_up')

        return (
            wallet_training_data_df,
            modeling_wallet_features_df,
            validation_training_data_df,
            validation_wallet_features_df
        )

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
        - epoch_modeling_data_df (DataFrame): Modeling features for this epoch
        """
        model_start = epoch_config['training_data']['modeling_period_start']
        logger.info(f"Generating data for epoch with modeling_period_start of {model_start}...")
        u.notify('futuristic')

        # If using hybrid IDs, override for a non-hybridized run to define the wallet cohort
        epoch_config['training_data']['hybridize_wallet_ids'] = False

        # Build features with initial data
        epoch_date, epoch_training_data_df, epoch_modeling_data_df, cohorts = self._build_epoch_features(epoch_config)

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
            ) = self._build_epoch_features(epoch_config, cohorts)

            # Merge hybrid/nonhybrid training and modeling dfs
            epoch_training_data_df = self._merge_hybrid_dfs(epoch_training_data_df,hybridized_training_data_df)
            epoch_modeling_data_df = self._merge_hybrid_dfs(epoch_modeling_data_df,hybridized_modeling_data_df)

            # Remove modeling-only IDs - these represent wallets in the cohort but
            # coin-wallet pairs that only became active during the modeling period
            epoch_modeling_data_df = (epoch_modeling_data_df[
                epoch_modeling_data_df.index.isin(epoch_training_data_df.index.values)
            ])

            u.assert_matching_indices(epoch_training_data_df,epoch_modeling_data_df)

        return epoch_date, epoch_training_data_df, epoch_modeling_data_df



    def _build_epoch_features(
        self,
        epoch_config: dict,
        cohorts: Optional[Dict[Set,np.array]] = None
    ) -> Tuple[datetime, pd.DataFrame, pd.DataFrame, Dict[Set,np.array]]:
        """
        Process a single epoch configuration to generate training and modeling data.

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
                training_transfers_df,
                return_files=True
            )
            model_future = executor.submit(
                modeling_generator.prepare_modeling_features,
                modeling_profits_df_full,
                self.complete_hybrid_cw_id_df
            )
            epoch_training_data_df = train_future.result()
            epoch_modeling_data_df = model_future.result()

        # Store training df with epoch date
        epoch_date = datetime.strptime(
            epoch_config['training_data']['modeling_period_start'], '%Y-%m-%d')

        cohorts = {
            'coin_cohort': training_coin_cohort,
            'wallet_cohort': training_generator.training_wallet_cohort
        }

        logger.milestone(f"Successfully generated features for epoch {model_start}.")

        return epoch_date, epoch_training_data_df, epoch_modeling_data_df, cohorts


    def _generate_epoch_configs(self) -> List[Dict]:
        """
        Generates config dicts for each offset epoch, including modeling and validation epochs.

        Returns:
        - List[Dict]: List of config dicts, one per epoch.
        """
        all_epochs_configs = []
        base_training_data = self.base_config['training_data']
        offsets = self.epochs_config['offset_epochs']['offsets']

        for offset_days in offsets:
            # Deep copy base config to prevent mutations
            epoch_config = copy.deepcopy(self.base_config)
            epoch_config['epoch_type'] = 'modeling'

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

        # Add validation epochs if present
        if self.base_config['training_data'].get('validation_period_end') is not None:
            validation_offsets = self.epochs_config['offset_epochs'].get('validation_offsets', [])
            for offset_days in validation_offsets:
                epoch_config = copy.deepcopy(self.base_config)
                # Offset key dates
                for date_key in ['modeling_period_start','modeling_period_end']:
                    base_date = datetime.strptime(self.base_config['training_data'][date_key], '%Y-%m-%d')
                    new_date = base_date + timedelta(days=offset_days)
                    epoch_config['training_data'][date_key] = new_date.strftime('%Y-%m-%d')
                epoch_config['training_data']['training_window_starts'] = [
                    (datetime.strptime(dt, '%Y-%m-%d') + timedelta(days=offset_days)).strftime('%Y-%m-%d')
                    for dt in self.base_config['training_data']['training_window_starts']
                ]
                folder_suffix = (datetime.strptime(epoch_config['training_data']['modeling_period_start'], '%Y-%m-%d')
                                .strftime('%y%m%d'))
                base_folder = Path(self.base_config['training_data']['parquet_folder'])
                epoch_config['training_data']['parquet_folder'] = str(base_folder / folder_suffix)
                epoch_config = wcm.add_derived_values(epoch_config)
                epoch_config['epoch_type'] = 'validation'
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


    def _create_hybrid_mapping(self) -> pd.DataFrame:
        """
        Create hybrid wallet-coin ID mapping if configured in the base config.
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
