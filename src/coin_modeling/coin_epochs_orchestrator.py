"""
Orchestrates the creation of wallet training data, models, and scores for each coin lookback
 window, then converts them into coin features that are merged into a single dataframe along
 with their epoch reference date.
"""
import os
import logging
from typing import Tuple
from pathlib import Path
import copy
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import pandas as pd
import joblib

# Local module imports
import training_data.data_retrieval as dr
import wallet_modeling.wallets_config_manager as wcm
import wallet_modeling.wallet_epochs_orchestrator as weo
import wallet_modeling.wallet_training_data as wtd
import wallet_modeling.wallet_training_data_orchestrator as wtdo
import wallet_modeling.wallet_model_orchestrator as wmo
import coin_wallet_features.coin_features_orchestrator as cfo
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)


# ----------------------------------------
#       Primary Orchestration Class
# ----------------------------------------

class CoinEpochsOrchestrator:
    """
    Orchestrates training data generation across multiple epochs by
    offsetting base config dates and managing the resulting datasets.
    """
    def __init__(
        self,

        # wallets_coin model configs
        wallets_coin_config: dict,
        wallets_coins_metrics_config: dict,

        # wallets model configs
        wallets_config: dict,
        wallets_metrics_config: dict,
        wallets_features_config: dict,
        wallets_epochs_config: dict,

        # coin flow model configs
        coin_flow_config: dict = None,
        coin_flow_modeling_config: dict = None,
        coin_flow_metrics_config: dict = None,

        complete_profits_df: pd.DataFrame = None,
        complete_market_data_df: pd.DataFrame = None,
        complete_macro_trends_df: pd.DataFrame = None,
    ):
        # Ensure configs are dicts and not the custom config classes
        if not (isinstance(wallets_config,dict) and isinstance(wallets_coin_config,dict)):
            raise ValueError("CoinEpochsOrchestrator configs must be dtype=='dict'.")

        # Coin Configs
        self.wallets_coin_config =          copy.deepcopy(wallets_coin_config)
        self.wallets_coins_metrics_config = copy.deepcopy(wallets_coins_metrics_config)

        # Coin Params
        self.coin_training_data_only = None  # overridden by orchestrate_coin_epochs() param

        # WalletEpochsOrchestrator Configs
        self.wallets_config =               copy.deepcopy(wallets_config)
        self.wallets_metrics_config =       copy.deepcopy(wallets_metrics_config)
        self.wallets_features_config =      copy.deepcopy(wallets_features_config)
        self.wallets_epochs_config =        copy.deepcopy(wallets_epochs_config)

        # Coin Flow Configs
        self.coin_flow_config =             copy.deepcopy(coin_flow_config)
        self.coin_flow_modeling_config =    copy.deepcopy(coin_flow_modeling_config)
        self.coin_flow_metrics_config =     copy.deepcopy(coin_flow_metrics_config)

        # Complete DataFrames
        self.complete_profits_df = complete_profits_df
        self.complete_market_data_df = complete_market_data_df
        self.complete_macro_trends_df = complete_macro_trends_df



    # -----------------------------------
    #         Primary Interface
    # -----------------------------------

    def load_complete_raw_datasets(self) -> None:
        """
        Identifies the earliest training start date from the earliest coin epoch
        and generates complete dfs for them.
        """
        # Calculate earliest required modeling start date for coverage validation
        coins_earliest_epoch = min(self.wallets_coin_config['training_data']['coin_epochs_training'])
        wallets_earliest_epoch = min(self.wallets_epochs_config['offset_epochs']['offsets'])
        wallets_earliest_window = max(self.wallets_config['training_data']['training_window_lookbacks'])
        earliest_modeling_period_start = (
            pd.to_datetime(self.wallets_config['training_data']['modeling_period_start'])
            + timedelta(days=coins_earliest_epoch)
            + timedelta(days=wallets_earliest_epoch)
            - timedelta(days=wallets_earliest_window)
        )
        validation_period_end = pd.to_datetime(self.wallets_config['training_data']['validation_period_end'])

        # Return existing dfs if available
        if self.complete_profits_df is not None:
            # Validate coverage of stored dfs
            min_date = self.complete_profits_df.index.get_level_values('date').min()
            max_date = self.complete_profits_df.index.get_level_values('date').max()
            if min_date <= earliest_modeling_period_start and max_date >= validation_period_end:
                logger.info("Stored complete dfs cover required period and will be used.")
                return
            logger.warning(
                f"Stored complete dfs cover dates {min_date.date()} to {max_date.date()}, "
                f"which does not cover required range {earliest_modeling_period_start.date()} "
                f"to {validation_period_end.date()}. Regenerating complete dfs."
            )

        # Load the existing files if available
        parquet_folder = self.wallets_config['training_data']['parquet_folder']
        if os.path.exists(f"{parquet_folder}/complete_profits_df.parquet"):
            parquet_folder = self.wallets_config['training_data']['parquet_folder']
            # Store in self
            self.complete_profits_df = pd.read_parquet(f"{parquet_folder}/complete_profits_df.parquet")
            self.complete_market_data_df = pd.read_parquet(f"{parquet_folder}/complete_market_data_df.parquet")
            self.complete_macro_trends_df = pd.read_parquet(f"{parquet_folder}/complete_macro_trends_df.parquet")
            # Validate coverage of parquet-loaded dfs
            min_date = self.complete_profits_df.index.get_level_values('date').min()
            max_date = self.complete_profits_df.index.get_level_values('date').max()
            if min_date <= earliest_modeling_period_start and max_date >= validation_period_end:
                logger.info("Loaded complete dfs from parquet and they cover required period.")
                return
            logger.warning(
                f"Parquet-loaded complete dfs cover dates {min_date.date()} to {max_date.date()}, "
                f"which does not cover required range {earliest_modeling_period_start.date()} to "
                f"{validation_period_end.date()}. Regenerating complete dfs."
            )

        # Generate wallets_lookback_config with the earliest lookback date
        coins_earliest_epoch = min(self.wallets_coin_config['training_data']['coin_epochs_training'])
        wallets_lookback_config = copy.deepcopy(self.wallets_config)
        earliest_modeling_period_start_str = (
            pd.to_datetime(wallets_lookback_config['training_data']['modeling_period_start'])
            + timedelta(days=coins_earliest_epoch)
        ).strftime('%Y-%m-%d')
        wallets_lookback_config_dict = copy.deepcopy(wallets_lookback_config)
        wallets_lookback_config_dict['training_data']['modeling_period_start'] = earliest_modeling_period_start_str
        wallets_lookback_config = wcm.add_derived_values(wallets_lookback_config_dict)

        # Load and store complete dfs
        # This WEO has its earliest epoch set to the earliest relevant date before loading
        #  all complete datasets.
        #
        #  Example Scenario
        #  ----------------
        #  Earliest Date Calculations
        #    base modeling_period_start                                 2024-12-01
        #  - earliest (wallets_coin_config['coin_epochs_training'])   - 360
        #  - earliest (wallets_epochs_config['offsets'])              - 360
        #  - earliest (wallets_config['training_window_lookbacks'])   - 180
        #  - 1 day for starting_balance_date of training_period       -   1
        #  = earliest possible date                                   = 2022-06-15 (offset -901)
        #
        #  Latest Date Calculations
        #    base_validation_period_end                               = 2025-04-30 (from base config)
        #
        #  Complete Range
        #    2022-06-15 through 2022-04-30 (901 days)
        load_data_weo = weo.WalletEpochsOrchestrator(
            wallets_lookback_config, # lookback config
            self.wallets_metrics_config,
            self.wallets_features_config,
            self.wallets_epochs_config,
        )
        load_data_weo.load_complete_raw_datasets()

        self.complete_profits_df = load_data_weo.complete_profits_df
        self.complete_market_data_df = load_data_weo.complete_market_data_df
        self.complete_macro_trends_df = load_data_weo.complete_macro_trends_df
        logger.info("Successfully retrieved complete dfs.")



    @u.timing_decorator(logging.MILESTONE)  # pylint: disable=no-member
    def orchestrate_coin_epochs(
            self,
            custom_offset_days: list[int] | None = None,
            file_prefix: str = '',
            coin_training_data_only: bool = False
        ) -> None:
        """
        Orchestrate coin-level epochs by iterating through lookbacks and processing each epoch.

        Params:
        - custom_offset_days (list of ints): overrides coin_epochs_training in wallets_coin_config
        - file_prefix (str): changes the filename of the parquet files
        - coin_training_data_only (bool): whether to skip data generation of the coin target variables
            and validation datasets. This is used to generate training data that will be scored with
            an existing model that can be used to predict future price action.
        """
        # Override existing training data toggle with the param
        self.coin_training_data_only = coin_training_data_only
        self.wallets_config['training_data']['coin_training_data_only'] = coin_training_data_only

        # Notifications
        logger.milestone("")
        logger.milestone("Beginning orchestration of coin model data generation...")
        u.notify('whoosh_boom')
        ambient_player_coins = u.AmbientPlayer()
        ambient_player_coins.start('spaceship_ambient_loop')

        # 1. Build all wallet training data needed for all coin epochs
        # ------------------------------------------------------------
        # This increases execution speed by multithreading every wallet epoch, rather
        #  than generating individual additional wallet epochs needed as the coin epoch
        #  boundaries shift.
        self._build_all_wallet_data()


        # 2. Define and validate coin epoch offsets
        # -----------------------------------------
        offsets = custom_offset_days if custom_offset_days is not None \
            else self.wallets_coin_config['training_data']['coin_epochs_training']

        # Calculate maximum offset possible based on validation_period_end
        most_recent_data = self.complete_profits_df.index.get_level_values('date').max()
        base_coin_modeling_end = pd.to_datetime(self.wallets_config['training_data']['coin_modeling_period_end'])
        max_calculable_offset = (most_recent_data - base_coin_modeling_end).days

        # If we don't need target vars or training data, we can extend one period further forward
        if coin_training_data_only:
            max_calculable_offset+= self.wallets_config['training_data']['modeling_period_duration']

        if max(offsets) > max_calculable_offset:
            raise ValueError(f"Offset value of {max(offsets)} extends further into the future than the "
                            f"maximum calculable offset of {max_calculable_offset}.")


        # 3. Generate all coin epochs' training data
        # ------------------------------------------
        logger.milestone("Beginning generation of all coin epochs' training data...")

        # Tag each DataFrame with the epoch date
        def tag_with_epoch(df: pd.DataFrame, epoch_date: pd.Timestamp) -> pd.DataFrame:
            df = df.copy()
            df['coin_epoch_start_date'] = epoch_date
            return df.set_index('coin_epoch_start_date', append=True)

        def process_single_epoch(lookback: int) -> tuple:
            """Process a single epoch and return tagged results"""
            epoch_date, coin_features_df, coin_target_df = self._process_coin_epoch(lookback)
            return (
                tag_with_epoch(coin_features_df, epoch_date),
                tag_with_epoch(coin_target_df, epoch_date)
            )

        # Process epochs in parallel
        feature_dfs = []
        target_dfs = []

        n_threads = self.wallets_coin_config['n_threads']['concurrent_coin_epochs']
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            future_to_offset = {
                executor.submit(process_single_epoch, lookback): lookback
                for lookback in offsets
            }

            for i, future in enumerate(as_completed(future_to_offset), 1):
                lookback = future_to_offset[future]
                try:
                    features_df, targets_df = future.result()
                    feature_dfs.append(features_df)
                    target_dfs.append(targets_df)
                    logger.milestone(f"Completed coin epoch {i}/{len(offsets)} (lookback={lookback})")
                except Exception as e:
                    logger.error(f"Coin epoch with lookback {lookback} generated an exception: {e}", exc_info=True)
                    raise

        # Concatenate all epochs dfs
        multiwindow_features = pd.concat(feature_dfs).sort_index()
        multiwindow_targets = pd.concat(target_dfs).sort_index()

        def reset_index_codes(df):
            """Helper to ensure matching indices by reseting MultiIndex codes while preserving values"""
            df.index = pd.MultiIndex.from_tuples(df.index.values, names=df.index.names)
            return df


        # 4. Persist multiwindow parquet files
        # ------------------------------------
        root_folder = self.wallets_coin_config['training_data']['parquet_folder']
        multiwindow_features = reset_index_codes(multiwindow_features)
        multiwindow_features.to_parquet(f"{root_folder}/{file_prefix}multiwindow_coin_training_data_df.parquet")
        if not multiwindow_targets.empty:
            multiwindow_targets = reset_index_codes(multiwindow_targets)
            multiwindow_targets.to_parquet(f"{root_folder}/{file_prefix}multiwindow_coin_target_var_df.parquet")

        ambient_player_coins.stop()
        u.notify('robotz_windows_start')


    @staticmethod
    def score_coin_training_data(
            wallets_coin_config: dict,
            model_id: str,
            artifacts_path: str,
            features_df: pd.DataFrame = None
        ) -> pd.DataFrame:
        """
        Load a saved coin model pipeline and score current coin features.

        Params:
        - model_id (str): UUID of the saved model to load.
        - artifacts_path (str): Base directory where model artifacts are stored.
        - features_df (DataFrame, optional): DataFrame of features to score.

        Returns:
        - scores_df (DataFrame): DataFrame with coin_id index, score, model_id.
        """
        # 1) Use provided features_df or load from parquet
        if features_df is None:
            base_folder = wallets_coin_config['training_data']['parquet_folder']
            features_file = Path(base_folder) / 'current_como_coin_training_data_df_full.parquet'
            features_df = pd.read_parquet(features_file)

        # 2) Load the saved sklearn Pipeline
        pipeline_file = Path(artifacts_path) / 'coin_models' / f'coin_model_pipeline_{model_id}.pkl'
        if not pipeline_file.exists():
            raise FileNotFoundError(f'Pipeline file {pipeline_file} not found.')
        pipeline = joblib.load(str(pipeline_file))

        # 3) Validate that the pipeline can transform the raw features
        try:
            pipeline.x_transformer_.transform(features_df)
        except Exception as e:
            raise ValueError(f"Pipeline transform failed due to missing or invalid features: {e}") from e

        # 4) Predict using the pipeline
        if hasattr(pipeline.estimator, 'predict_proba'):
            preds = pipeline.predict_proba(features_df)[:, 1]
        else:
            preds = pipeline.predict(features_df)

        # 5) Assemble and save results (unchanged)...
        scores_df = features_df[[]].copy()
        scores_df['score'] = preds

        pred_folder = Path(artifacts_path) / 'coin_predictions'
        pred_folder.mkdir(parents=True, exist_ok=True)
        output_file = pred_folder / f'coin_predictions_{model_id}.csv'
        scores_df.to_csv(output_file, index=True)
        logger.info(f'Saved coin predictions to {output_file}')

        return scores_df




    # -----------------------------------
    #           Helper Methods
    # -----------------------------------

    # --------------
    # Primary Helper
    # --------------
    @u.timing_decorator
    def _process_coin_epoch(
            self,
            lookback_duration: int
        ) -> tuple[datetime, pd.DataFrame, pd.DataFrame]:
        """
        Process a single coin epoch: generate data, train wallet models, generate coin
         modeling data, and score wallets.

        Key Steps:
        1. Generate epoch-specific config files and check if the coin features df have already
            been generated.
        2. Generate wallet-level features and model scores for the epoch.
        3. Generate coin-level features, including transformations of wallet-level features into
            coin features.

        Params:
        - lookback_duration (int): how many days the dates will be offset from the base modeling period
        """
        # 1) Prepare config files
        # -----------------------
        epoch_wallets_config = self._prepare_coin_epoch_base_config(lookback_duration)

        # Debug: Log the resulting dates  # pylint:disable=line-too-long
        logger.debug(f"Will apply offset of {lookback_duration} days")
        logger.debug(f"Resulting training_period_start: {epoch_wallets_config['training_data']['training_period_start']}")
        logger.debug(f"Resulting training_period_end: {epoch_wallets_config['training_data']['training_period_end']}")
        logger.debug(f"Resulting modeling_period_start: {epoch_wallets_config['training_data']['modeling_period_start']}")
        logger.debug(f"Resulting modeling_period_end: {epoch_wallets_config['training_data']['modeling_period_end']}")
        logger.debug(f"Resulting coin_modeling_period_start: {epoch_wallets_config['training_data']['coin_modeling_period_start']}")
        logger.debug(f"Resulting coin_modeling_period_end: {epoch_wallets_config['training_data']['coin_modeling_period_end']}")

        epoch_coins_config = self._prepare_epoch_coins_config(epoch_wallets_config)
        epoch_date = pd.to_datetime(epoch_wallets_config['training_data']['coin_modeling_period_start'])

        # Shortcut: if both feature and target parquet files exist, load and return them
        toggle_rebuild_features = epoch_coins_config['training_data']['toggle_rebuild_all_features']
        base_folder = epoch_coins_config['training_data']['parquet_folder']
        feat_path = Path(base_folder) / "coin_training_data_df_full.parquet"
        tgt_path  = Path(base_folder) / "coin_target_var_df.parquet"

        # Load existing data if configured
        if (feat_path.exists() and tgt_path.exists()):
            if not toggle_rebuild_features:
                coin_features_df = pd.read_parquet(feat_path)
                coin_target_df   = pd.read_parquet(tgt_path)
                logger.milestone(
                    "Coin epoch %s training data loaded from existing feature and target files.",
                    epoch_date.strftime('%Y-%m-%d')
                )
                return epoch_date, coin_features_df, coin_target_df

            # Announce overwrite if applicable
            else:
                logger.warning("Overwriting existing coin features due to 'toggle_rebuild_all_features'.")

        # Epoch-specific wallets_epochs_config.yaml
        epoch_wallets_epochs_config = copy.deepcopy(self.wallets_epochs_config)

        # Delete wallet validation epochs if we aren't generating coin target variables, as the wallet
        #  validation period overlaps with the coin modeling period used for coin target variable data.
        if self.coin_training_data_only:
            epoch_wallets_epochs_config['offset_epochs']['validation_offsets'] = []


        # 2) Wallet-Level Features
        # ------------------------
        # Prepare epoch-specific orchestrator
        # This WEO is the base_config_weo offset by the lookback_duration param and modeling duration
        #  to generate training data for the relevant epoch.
        #
        #  Example Scenario
        #  ----------------
        #    base modeling_period_start                                 2024-12-01
        #  + offset_duration                                                  + 90
        #  + modeling_duration                                                + 30
        #  = offset epoch                                             = 2025-03-31 (offset +120)
        epoch_weo = weo.WalletEpochsOrchestrator(
            base_config=epoch_wallets_config,               # epoch-specific config
            metrics_config=self.wallets_metrics_config,
            features_config=self.wallets_features_config,
            epochs_config=epoch_wallets_epochs_config,      # epoch-specific config
            complete_profits_df=self.complete_profits_df,
            complete_market_data_df=self.complete_market_data_df,
            complete_macro_trends_df=self.complete_macro_trends_df,
        )

        # Generate wallets training & modeling data
        epoch_training_dfs = epoch_weo.generate_epochs_training_data()

        # Train and score wallet models for this epoch's coin modeling period
        wallet_training_data_df = self._train_and_score_wallet_epoch(
            epoch_weo,
            epoch_coins_config,
            epoch_training_dfs,
        )


        # 3) Coin-Level Features
        # ----------------------
        # Generate and save coin features for this epoch
        (
            coin_features_df,
            coin_market_data_df,
        ) = self._generate_coin_features(
            epoch_weo,
            epoch_coins_config,
            wallet_training_data_df
        )

        # Generate and save target variables for this epoch
        try:
            coin_target_var_df = self._generate_coin_target_vars(
                epoch_weo,
                epoch_coins_config,
                coin_features_df,
                coin_market_data_df
            )
        except Exception as e:
            logger.warning(
                "Target variable generation failed for epoch %s: %s",
                epoch_date.strftime('%Y-%m-%d'),
                e
            )
            # fallback to empty targets to allow features-only epochs
            coin_target_var_df = pd.DataFrame(index=coin_features_df.index)

        logger.milestone(
            "Coin epoch %s training data with shape (%s) generated successfully.",
            epoch_date.strftime('%Y-%m-%d'),
            coin_features_df.shape
        )

        return epoch_date, coin_features_df, coin_target_var_df



    # ---------------------
    # Coin Features Helpers
    # ---------------------
    def _generate_coin_features(
        self,
        epoch_weo,
        epoch_coins_config: dict,
        wallet_training_data_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate and persist coin features
        """
        # 1) Load base dfs needed for coin feature generation
        training_wallet_cohort = pd.Series(wallet_training_data_df.index.get_level_values('wallet_address'))
        (
            profits_df,
            coin_market_data_df,
            training_coin_cohort,
        ) = self._load_wallet_data_for_coin_features(
            epoch_weo.base_config,
            training_wallet_cohort
        )

        # 2) Prepare datasets
        macro_ind_df = self._generate_epoch_macro_indicators(
            epoch_weo.base_config['training_data']['modeling_period_start'],
            epoch_weo.base_config['training_data']['modeling_period_end'],
        )
        market_ind_df = self._generate_epoch_market_indicators(
            profits_df,
            training_coin_cohort,
            epoch_weo.base_config
        )

        # 3) Generate features
        cfo_inst = cfo.CoinFeaturesOrchestrator(
            epoch_weo.base_config,
            epoch_coins_config,
            self.wallets_coins_metrics_config,
            self.coin_flow_config,
            self.coin_flow_modeling_config,
            self.coin_flow_metrics_config,
            training_coin_cohort,
        )
        file_prefix = pd.to_datetime(
            epoch_weo.base_config['training_data']['coin_modeling_period_start']
        ).strftime('%Y%m%d')

        coin_features_df = cfo_inst.generate_coin_features_for_period(
            profits_df,
            wallet_training_data_df,
            market_ind_df,
            macro_ind_df,
            "modeling",
            file_prefix,
        )

        # 4) Persist results to parquet
        base_folder = epoch_coins_config['training_data']['parquet_folder']
        coin_features_df.to_parquet(
            f"{base_folder}/coin_training_data_df_full.parquet"
        )

        return coin_features_df, coin_market_data_df



    def _generate_coin_target_vars(
        self,
        epoch_weo,
        epoch_coins_config: dict,
        coin_features_df: pd.DataFrame,
        coin_market_data_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate and save target variablesfor the coin features.
        """
        base_folder = epoch_coins_config['training_data']['parquet_folder']
        # Instantiate a fresh CoinFeaturesOrchestrator for target calculation
        features_generator = cfo.CoinFeaturesOrchestrator(
            epoch_weo.base_config,
            epoch_coins_config,
            self.wallets_coins_metrics_config,
            self.coin_flow_config,
            self.coin_flow_modeling_config,
            self.coin_flow_metrics_config,
            None  # training_coin_cohort not required for target calc
        )

        # Calculate target variables
        coin_target_var_df = features_generator.calculate_target_variables(
            coin_market_data_df,
            epoch_weo.base_config['training_data']['coin_modeling_period_start'],
            epoch_weo.base_config['training_data']['coin_modeling_period_end'],
            set(coin_features_df.index)
        )
        coin_target_var_df.to_parquet(f"{base_folder}/coin_target_var_df.parquet", index=True)

        return coin_target_var_df



    def _load_wallet_data_for_coin_features(
            self,
            wallets_config: dict,
            wallet_cohort: pd.Series
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Loads DataFrames for generating coin model features and target variables. Filters
        out wallets that aren't in the wallet cohort from the wallet_training_data_df. If


        Params:
        - wallets_config (dict): epoch-specific wallets_config used for folder locations
            and date boundaries
        - wallet_cohort (pd.Series): All wallets with wallet model features

        Returns:
        - wamo_profits_df (pd.DataFrame): Contains profits data through the end of the
            wallet_modeling period. This is used to generate coin-level features used to
            predict the coin modeling period price action.
        - como_market_data_df (pd.DataFrame): Market data through the end of the coin_modeling
            period. This is used to create coin-level target variables.
        - training_coin_cohort (pd.Series): All coins included in wamo_profits_df
        """
        pf = wallets_config['training_data']['parquet_folder']

        # 1) Wallet modeling period profits_df for coin-level features
        wamo_date = datetime.strptime(
            wallets_config['training_data']['modeling_period_start'],
            '%Y-%m-%d'
        ).strftime('%y%m%d')
        wamo_profits_df = pd.read_parquet(f"{pf}/{wamo_date}/modeling_profits_df.parquet")
        u.assert_period(
            wamo_profits_df,
            wallets_config['training_data']['modeling_period_start'],
            wallets_config['training_data']['modeling_period_end']
        )

        # Hybridize wallet IDs if configured
        if wallets_config['training_data']['hybridize_wallet_ids']:
            hybrid_map = pd.read_parquet(f"{pf}/complete_hybrid_cw_id_df.parquet")
            wtdo.validate_hybrid_mapping_completeness(wamo_profits_df, hybrid_map)
            wamo_profits_df = wtdo.hybridize_wallet_address(
                wamo_profits_df, hybrid_map
            )

        # Filter to wallet cohort
        wamo_profits_df = wamo_profits_df[wamo_profits_df['wallet_address'].isin(wallet_cohort)]

        # Store coin cohort
        training_coin_cohort = (
            wamo_profits_df['coin_id'].drop_duplicates()
        )

        # 2) Coin modeling market data for coin target vars
        # skip como_market_data_df's modeling data if we're only returning training data
        if self.coin_training_data_only:
            return wamo_profits_df, pd.DataFrame(), training_coin_cohort

        # Filter complete market data to coin modeling period
        complete_md = pd.read_parquet(f"{pf}/complete_market_data_df.parquet")
        como_market_data_df = complete_md.loc[
            (complete_md.index.get_level_values('date') >=
                wallets_config['training_data']['modeling_period_end']) &
            (complete_md.index.get_level_values('date') <=
                wallets_config['training_data']['coin_modeling_period_end'])
        ]
        u.assert_period(
            como_market_data_df,
            wallets_config['training_data']['coin_modeling_period_start'],
            wallets_config['training_data']['coin_modeling_period_end']
        )

        return wamo_profits_df, como_market_data_df, training_coin_cohort



    def _generate_epoch_macro_indicators(
            self,
            period_start_date: str,
            period_end_date: str) -> pd.DataFrame:
        """
        Generate macro indicators for the specified period using WalletTrainingDataOrchestrator.

        Args:
            epoch_weo: WalletEpochsOrchestrator instance with configs
            period_start_date: Start date in 'YYYY-MM-DD' format
            period_end_date: End date in 'YYYY-MM-DD' format

        Returns:
            DataFrame with generated indicators
        """

        # Trim, clean, and impute missing values in complete_macro_trends_df
        period_macro_trends_df = dr.clean_macro_trends(
            self.complete_macro_trends_df,
            macro_trends_cols=list(self.wallets_coins_metrics_config['time_series']['macro_trends'].keys()),
            start_date = None,  # retain historical data for indicators
            end_date = period_end_date
        )

        # Use existing training data orchestrator for consistency
        wtdo_instance = wtdo.WalletTrainingDataOrchestrator(
            self.wallets_config,                # has no impact on indicators output
            self.wallets_coins_metrics_config,  # coins metrics coCoinFeaturesOrchestratornfig
            self.wallets_features_config        # has no impact on indicators output
        )

        # Call the public indicator generation method
        macro_indicators_df = wtdo_instance.generate_indicators_df(
            period_macro_trends_df.reset_index(),
            period_start_date=period_start_date,
            period_end_date=period_end_date,
            metric_type='macro_trends',
            parquet_filename=None
        )

        # Set date index for consistency with rest of pipeline
        macro_indicators_df = macro_indicators_df.set_index('date')

        logger.info(f"Generated {len(macro_indicators_df.columns)} macro indicators for period "
                    f"{period_start_date} to {period_end_date}")

        return macro_indicators_df



    def _generate_epoch_market_indicators(
            self,
            profits_df: pd.DataFrame,
            training_coin_cohort: pd.Series,
            wallets_config: dict) -> pd.DataFrame:
        """
        Generate market data indicators for the specified period using WalletTrainingDataOrchestrator.

        Args:
        - profits_df (pd.DataFrame): profits_df for the epoch
        - training_coin_cohort (pd.Series): which coins to include in the output df
        - wallets_config (dict): epoch-specific wallets_config.yaml

        Returns:
            DataFrame with generated indicators
        """
        wtd_orch = wtd.WalletTrainingData(self.wallets_config)
        period_start_date = wallets_config['training_data']['modeling_period_start']
        period_end_date = wallets_config['training_data']['modeling_period_end']

        # Trim, clean, and impute missing values in complete_macro_trends_df
        period_market_data_df = wtd_orch.clean_market_dataset(
            self.complete_market_data_df.reset_index(),
            profits_df,
            period_start_date,
            period_end_date,
            set(training_coin_cohort)
        )

        # Use existing training data orchestrator for consistency
        wtdo_instance = wtdo.WalletTrainingDataOrchestrator(
            wallets_config,                     # has no impact on indicators output
            self.wallets_coins_metrics_config,  # coins metrics coCoinFeaturesOrchestratornfig
            self.wallets_features_config        # has no impact on indicators output
        )

        # Call the public indicator generation method
        market_indicators_df = wtdo_instance.generate_indicators_df(
            period_market_data_df.reset_index(),
            period_start_date=period_start_date,
            period_end_date=period_end_date,
            metric_type='market_data',
            parquet_filename=None
        )

        # Set date index for consistency with rest of pipeline
        market_indicators_df = market_indicators_df.set_index('date')

        logger.info(f"Generated {len(market_indicators_df.columns) - len(period_market_data_df.columns)} "
                    f"market data indicators for period {period_start_date} to {period_end_date}")

        return market_indicators_df




    # --------------
    # Config Helpers
    # --------------
    def _prepare_coin_epoch_base_config(self, lookback_duration: int):
        """
        Creates an epoch-specific wallets_config with dates offset by the epoch's lookback_duration.

        Params:
        - lookback_duration (int): How many days the coin lookback is offset vs the base config.
        """
        # Cache parsed base dates and base folder path
        base_training_data = copy.deepcopy(self.wallets_config['training_data'])
        base_modeling_start = datetime.strptime(base_training_data['modeling_period_start'], '%Y-%m-%d')
        base_modeling_end = datetime.strptime(base_training_data['modeling_period_end'], '%Y-%m-%d')
        base_training_window_starts = [
            datetime.strptime(dt, '%Y-%m-%d')
            for dt in base_training_data['training_window_starts']
        ]
        base_parquet_folder_base = Path(base_training_data['parquet_folder'])

        # Load and store complete dfs
        # This WEO generates the base wallets config without making any modifications to
        #  date ranges.
        base_config_weo = weo.WalletEpochsOrchestrator(
            copy.deepcopy(self.wallets_config), # base config
            self.wallets_metrics_config,
            self.wallets_features_config,
            self.wallets_epochs_config
        )

        # The build_epoch_wallets_config method should apply the lookback_duration offset
        epoch_base_wallets_config = base_config_weo.build_epoch_wallets_config(
            lookback_duration,          # This offset gets applied to base dates
            'coin_modeling',
            base_modeling_start,        # Base date before offset
            base_modeling_end,
            base_training_window_starts,
            base_parquet_folder_base
        )

        # Retain the parquet_folder so all periods are built in the same root directory
        epoch_base_wallets_config['training_data']['parquet_folder'] = base_training_data['parquet_folder']

        return epoch_base_wallets_config


    def _prepare_epoch_coins_config(self, epoch_wallets_config) -> dict:
        """
        Prepare epoch-specific coins config with date suffix folders.
        """
        # Build a date suffix from the modeling_period_start
        date_suffix = pd.to_datetime(
            epoch_wallets_config['training_data']['modeling_period_start']
        ).strftime('%Y%m%d')

        # Deep-copy the coin config and adjust folder paths
        epoch_coins_config = copy.deepcopy(self.wallets_coin_config)
        base_folder = epoch_coins_config['training_data']['parquet_folder']

        parquet_folder = f"{base_folder}/{date_suffix}"
        epoch_coins_config['training_data']['parquet_folder'] = parquet_folder
        Path(parquet_folder).mkdir(exist_ok=True)

        scores_folder = f"{parquet_folder}/scores"
        epoch_coins_config['training_data']['coins_wallet_scores_folder'] = scores_folder
        Path(scores_folder).mkdir(exist_ok=True)
        return epoch_coins_config


    @staticmethod
    def numerize_wallets_coin_config(wallets_config:dict, wallets_coin_config: dict) -> dict:
        """
        Convert string-based y_pred_thresholds to numeric values using latest model metadata.

        Params:
        - wallets_coin_config (dict): Raw wallets coin config with potential string thresholds

        Returns:
        - numerized_config (dict): Config with numeric y_pred_thresholds
        """
        # Find latest epoch with model data
        latest_epoch = max(
            wallets_coin_config['training_data']['coin_epochs_validation'] +
            wallets_coin_config['training_data']['coin_epochs_training']
        )
        latest_epoch_date = (
            pd.to_datetime(wallets_config['training_data']['modeling_period_start'])
            + timedelta(days=latest_epoch)
        ).strftime('%Y%m%d')

        # Load model metadata
        latest_model_ids_path = (
            Path(wallets_coin_config['training_data']['parquet_folder']) /
            latest_epoch_date / "wallet_model_ids.json"
        )

        if not latest_model_ids_path.exists():
            raise ValueError(f"No 'wallet_model_ids.json' found at {latest_model_ids_path}")

        with open(latest_model_ids_path, 'r', encoding='utf-8') as f:
            latest_models_dict = json.load(f)

        # Validate all required models exist
        missing_keys = (
            set(wallets_coin_config['wallet_scores']['score_params'].keys()) -
            set(latest_models_dict.keys())
        )
        if missing_keys:
            raise ValueError(f"Missing model metrics for score_params: {missing_keys}")

        # Convert string thresholds to numeric
        numerized_config = copy.deepcopy(wallets_coin_config)
        for model in numerized_config['wallet_scores']['score_params']:
            threshold = wallets_coin_config['wallet_scores']['score_params'][model]['y_pred_threshold']
            if isinstance(threshold, str):
                numeric_threshold = latest_models_dict[model]['metrics']['y_pred_threshold']
                numerized_config['wallet_scores']['score_params'][model]['y_pred_threshold'] = numeric_threshold

        return numerized_config.config


    # -----------------------
    # Wallet Features Helpers
    # -----------------------
    @u.timing_decorator(logging.MILESTONE)  # pylint: disable=no-member
    def _build_all_wallet_data(self) -> None:
        """
        Pre-build every wallet-epoch parquet so later coin loops just `pd.read_parquet`.
        """
        # Identify all offsets
        all_offsets = self._compute_all_wallet_offsets()

        # Find offsets that still need computation
        base_parquet_folder = self.wallets_config['training_data']['parquet_folder']
        base_model_start_dt = pd.to_datetime(
            self.wallets_config['training_data']['modeling_period_start']
        )
        missing_offsets: list[int] = []
        for offs in all_offsets:
            date_suffix = (base_model_start_dt + timedelta(days=offs)).strftime('%y%m%d')
            epoch_folder = f"{base_parquet_folder}/{date_suffix}"
            train_file   = f"{epoch_folder}/training_data_df.parquet"
            model_file   = f"{epoch_folder}/modeling_data_df.parquet"

            # If either core parquet is missing, mark this offset for build
            if not (os.path.exists(train_file) and os.path.exists(model_file)):
                missing_offsets.append(offs)

        if not missing_offsets:
            logger.milestone("All wallet‑epoch parquet files already exist — warm‑up skipped.")
            return

        logger.milestone(
            "Coin epoch orchestrator warm‑up: generating %d missing wallet‑epoch(s) %s",
            len(missing_offsets),
            missing_offsets
        )

        # Clone the epochs config and inject the superset
        bulk_epochs_cfg = copy.deepcopy(self.wallets_epochs_config)
        bulk_epochs_cfg['offset_epochs']['offsets'] = all_offsets
        bulk_epochs_cfg['offset_epochs']['validation_offsets'] = []
        # (keep validation_offsets as-is; they’re already in all_offsets)

        # Load and generate all wallets epochs data
        # This WEO has its date range set separately for each wallet epoch's training data.
        #  It shifts the full range of wallet model offsets by the modeling_period_duration
        #  to identify the earliest training data wallet epoch.
        #
        #  Example Scenario
        #  ----------------
        #  Earliest Training Data Epoch Calculations
        #    base modeling_period_start                                 2024-12-01
        #  - earliest (wallets_epochs_config['offsets'])              - 360
        #  - earliest (wallets_coin_config['coin_epochs_training'])   - 360
        #    earliest offset                                          =-720

        #
        #  Latest Training Data Epoch Calculations
        #    base modeling_period_start                                 2024-12-01
        #  - wallets_config['modeling_period_duration']               - 30
        #  = latest wallet epoch                                      = 2023-01-15 (offset -30)
        bulk_weo = weo.WalletEpochsOrchestrator(
            base_config         = self.wallets_config,
            metrics_config      = self.wallets_metrics_config,
            features_config     = self.wallets_features_config,
            epochs_config       = wcm.add_derived_values(bulk_epochs_cfg),
            complete_profits_df = self.complete_profits_df,
            complete_market_data_df = self.complete_market_data_df,
            complete_macro_trends_df = self.complete_macro_trends_df,
        )

        # One parallel run builds every epoch (training+modeling) in its own date-suffix folder
        bulk_weo.generate_epochs_training_data()



    def _compute_all_wallet_offsets(self) -> dict[str, list[int]]:
        """
        Pulls all day-shifts that WalletEpochsOrchestrator will need, using only
        the config objects already stored on `self`.
        """
        # coin-epoch lookbacks
        coin_train = self.wallets_coin_config['training_data']['coin_epochs_training']
        coin_val   = self.wallets_coin_config['training_data'].get('coin_epochs_validation', [])

        # wallet-epoch lookbacks
        w_train = self.wallets_epochs_config['offset_epochs']['offsets']
        w_val   = self.wallets_epochs_config['offset_epochs'].get('validation_offsets', [])

        # Build Cartesian combos
        train_epochs  = [c + w for c in coin_train for w in w_train]
        val_epochs    = [c + v for c in coin_val  for v in w_val]

        # Single merged list
        all_offsets = sorted(
            set(train_epochs)        |
            set(val_epochs)          |
            set(w_train) | set(w_val)|
            set(coin_train) | set(coin_val)
        )

        return all_offsets



    def _train_and_score_wallet_epoch(
        self,
        epoch_weo,
        epoch_coins_config: dict,
        epoch_training_dfs: tuple
    ) -> pd.DataFrame:
        """
        Train wallet models for a single epoch and score wallets.

        Returns:
        - wallet_training_data_df (pd.DataFrame): wallet training data for the epoch
        """
        # 1) Train all models
        # Instantiate the WalletModelOrchestrator for this epoch
        epoch_wmo = wmo.WalletModelOrchestrator(
            epoch_weo.base_config,         # epoch-specific config
            self.wallets_metrics_config,
            self.wallets_features_config,
            self.wallets_epochs_config,
            epoch_coins_config
        )

        # Train all wallet models (or score with existing models)
        models_dict = epoch_wmo.train_wallet_models(*epoch_training_dfs)

        # 2) Generate this epoch's wallet training data
        logger.info("Generating wallet training_data_df for scoring in the coin_modeling period...")
        modeling_offset = self.wallets_config['training_data']['modeling_period_duration']

        # This epoch's training data is built during the base modeling_period. As a result,
        #  no target variables or validation data will be generated.
        coin_modeling_epochs_config = {
            'offset_epochs': {
                'offsets': [modeling_offset],
                'validation_offsets': []
            }
        }

        # This WEO has its date range set to match the wallets_config['modeling_period_end']
        #  in order to generate training data as of that date that can be scored using the
        #  model generated by the bulk_weo.
        #
        #  Example Scenario
        #  ----------------
        #  Wallet Training Data Epoch Calculations
        #    base modeling_period_start                                 2024-12-01
        #  + wallets_config['modeling_period_duration']               + 30
        #  = wallet training data epoch                               = 2024-12-31 (offset +30)
        epoch_weo = weo.WalletEpochsOrchestrator(
            base_config=epoch_weo.base_config,
            metrics_config=epoch_weo.metrics_config,
            features_config=epoch_weo.features_config,
            epochs_config=coin_modeling_epochs_config,          # custom config
            complete_profits_df=epoch_weo.complete_profits_df,
            complete_market_data_df=epoch_weo.complete_market_data_df,
            complete_macro_trends_df=epoch_weo.complete_macro_trends_df,
            training_only=True # we're scoring with prebuilt models so only generate training data
        )
        # Generate TRAINING_DATA_DF
        # training_only is toggled to skip generation of target variables and validation dfs
        wallet_training_data_df, _, _, _ = epoch_weo.generate_epochs_training_data()

        # 3) Score wallets on the training data using the models in wallets_coin_config
        configured_models = set(epoch_coins_config['wallet_scores']['score_params'].keys())
        filtered_models_dict = {k: models_dict[k] for k in configured_models}
        epoch_wmo.predict_and_store(filtered_models_dict, wallet_training_data_df)

        return wallet_training_data_df
