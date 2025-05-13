"""
Orchestrates the creation of wallet training data, models, and scores for each coin lookback
 window, then converts them into coin features that are merged into a single dataframe along
 with their epoch reference date.
"""
import os
import logging
from pathlib import Path
import copy
from datetime import datetime, timedelta
import pandas as pd

# Local module imports
import training_data.data_retrieval as dr
import wallet_modeling.wallet_epochs_orchestrator as weo
import wallet_modeling.wallets_config_manager as wcm
import wallet_modeling.wallet_model_orchestrator as wmo
import wallet_modeling.wallet_training_data_orchestrator as wtdo
import coin_wallet_features.coin_features_orchestrator as cfo

# import utils as u

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
        wallets_coin_config: dict,
        wallets_coins_metrics_config: dict,
        wallets_config: dict,
        wallets_metrics_config: dict,
        wallets_features_config: dict,
        wallets_epochs_config: dict,
        complete_profits_df: pd.DataFrame = None,
        complete_market_data_df: pd.DataFrame = None,
        complete_macro_trends_df: pd.DataFrame = None,
    ):
        # Coin Params
        self.wallets_coin_config = wallets_coin_config
        self.wallets_coins_metrics_config = wallets_coins_metrics_config

        # WalletEpochsOrchestrator Configs and DataFrames
        self.wallets_config = wallets_config
        self.wallets_metrics_config = wallets_metrics_config
        self.wallets_features_config = wallets_features_config
        self.wallets_epochs_config = wallets_epochs_config
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
        # Return existing dfs if available
        if self.complete_profits_df is not None:
            logger.info("Stored complete dfs will be used.")
            return

        # Load the existing files if avilable
        parquet_folder = self.wallets_config['training_data']['parquet_folder']
        if os.path.exists(f"{parquet_folder}/complete_profits_df.parquet"):
            parquet_folder = self.wallets_config['training_data']['parquet_folder']

            # Store in self
            self.complete_profits_df = pd.read_parquet(f"{parquet_folder}/complete_profits_df.parquet")
            self.complete_market_data_df = pd.read_parquet(f"{parquet_folder}/complete_market_data_df.parquet")
            self.complete_macro_trends_df = pd.read_parquet(f"{parquet_folder}/complete_macro_trends_df.parquet")
            logger.info("Loaded complete dfs from parquet.")
            return

        # Generate wallets_lookback_config with the earliest lookback date
        coins_earliest_epoch = min(self.wallets_coin_config['training_data']['coin_epoch_lookbacks'])
        wallets_lookback_config = copy.deepcopy(self.wallets_config)
        earliest_modeling_period_start = (
            pd.to_datetime(wallets_lookback_config['training_data']['modeling_period_start'])
            + timedelta(days=coins_earliest_epoch)
        ).strftime('%Y-%m-%d')
        wallets_lookback_config_dict = copy.deepcopy(wallets_lookback_config.config)
        wallets_lookback_config_dict['training_data']['modeling_period_start'] = earliest_modeling_period_start
        wallets_lookback_config = wcm.add_derived_values(wallets_lookback_config_dict)

        # Load and store complete dfs
        wallet_orchestrator = weo.WalletEpochsOrchestrator(
            wallets_lookback_config, # lookback config
            self.wallets_metrics_config,
            self.wallets_features_config,
            self.wallets_epochs_config,
        )
        wallet_orchestrator.load_complete_raw_datasets()

        self.complete_profits_df = wallet_orchestrator.complete_profits_df
        self.complete_market_data_df = wallet_orchestrator.complete_market_data_df
        self.complete_macro_trends_df = wallet_orchestrator.complete_macro_trends_df
        logger.info("Successfully retrieved complete dfs.")



    def orchestrate_coin_epochs(self) -> None:
        """
        Orchestrate coin-level epochs by iterating through lookbacks and processing each epoch.
        """
        wamo_feature_dfs = []
        wamo_target_dfs = []
        como_feature_dfs = []
        como_target_dfs = []

        # Tag each DataFrame with the epoch date
        def tag_with_epoch(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df['coin_epoch_start_date'] = epoch_date
            return df.set_index('coin_epoch_start_date', append=True)
        for lookback in self.wallets_coin_config['training_data']['coin_epoch_lookbacks']:
            epoch_date, wamo_features_df, wamo_target_df, como_features_df, como_target_df = \
                self._process_coin_epoch(lookback)
            wamo_feature_dfs.append(tag_with_epoch(wamo_features_df))
            wamo_target_dfs.append(tag_with_epoch(wamo_target_df))
            como_feature_dfs.append(tag_with_epoch(como_features_df))
            como_target_dfs.append(tag_with_epoch(como_target_df))

        # Concatenate across epochs
        multiwindow_wamo = pd.concat(wamo_feature_dfs).sort_index()
        multiwindow_wamo_target = pd.concat(wamo_target_dfs).sort_index()
        multiwindow_como = pd.concat(como_feature_dfs).sort_index()
        multiwindow_como_target = pd.concat(como_target_dfs).sort_index()

        # Persist multiwindow parquet files
        root_folder = self.wallets_coin_config['training_data']['parquet_folder']
        multiwindow_wamo.to_parquet(f"{root_folder}/multiwindow_wamo_coin_training_data_df_full.parquet")
        multiwindow_wamo_target.to_parquet(f"{root_folder}/multiwindow_wamo_coin_target_var_df.parquet")
        multiwindow_como.to_parquet(f"{root_folder}/multiwindow_como_coin_training_data_df_full.parquet")
        multiwindow_como_target.to_parquet(f"{root_folder}/multiwindow_como_coin_target_var_df.parquet")


    # Primary helper
    def _process_coin_epoch(
            self,
            lookback_duration: int
        ) -> tuple[datetime, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Process a single coin epoch: generate data, train wallet models, generate coin modeling data, and score wallets.
        """
        # 1) Generate epoch-specific WalletEpochsOrchestrator and training data
        epoch_weo, epoch_training_dfs = self._generate_coin_epoch_training_data(lookback_duration)
        epoch_date = pd.to_datetime(
            epoch_weo.base_config['training_data']['coin_modeling_period_start']
        )

        # 2) Prepare coin config with date suffix folders
        epoch_coins_config = self._prepare_epoch_coins_config(epoch_weo)

        # 3) Train and score wallet models for this epoch's coin modeling period
        wamo_como_dfs = self._train_and_score_wallet_epoch(epoch_weo, epoch_coins_config, epoch_training_dfs)

        # 4) Generate and persist coin features for this epoch
        (
            wamo_features_df,
            como_features_df,
            como_market_data_df,
            investing_market_data_df
        ) = self._generate_coin_features(
            epoch_weo,
            epoch_coins_config,
            wamo_como_dfs[0],  # wamo_training_data_df
            wamo_como_dfs[2]   # como_training_data_df
        )

        # 5) Calculate and persist target variables for this epoch
        wamo_target_df, como_target_df = self._generate_coin_target_vars(
            epoch_weo,
            epoch_coins_config,
            wamo_features_df,
            como_features_df,
            como_market_data_df,
            investing_market_data_df
        )

        return epoch_date, wamo_features_df, wamo_target_df, como_features_df, como_target_df




    # -----------------------------------
    #           Helper Methods
    # -----------------------------------

    def _generate_coin_epoch_training_data(self, lookback_duration: int):
        """
        Generates a coin epoch's training data for all wallet windows and epochs.

        Params:
         - lookback_duration (int): How many days the coin lookback is offset vs the base config.

         Returns:
         - epoch_orch (WalletEpochsOrchestrator): orchestrator with config set for lookback
         - (
                wallet_training_data_df,modeling_wallet_features_df,
                validation_training_data_df,validation_wallet_features_df
            ) Tuple(pd.DataFrames): training_data_df and target variables to the epoch
        """
        # 1) Generate coin epoch-specific wallets_config
        epoch_wallets_config = self._prepare_coin_epoch_base_config(lookback_duration)

        # 2) Instantiate a new WalletEpochsOrchestrator for this epoch
        epoch_weo = weo.WalletEpochsOrchestrator(
            base_config=epoch_wallets_config,       # epoch-specific config
            metrics_config=self.wallets_metrics_config,
            features_config=self.wallets_features_config,
            epochs_config=self.wallets_epochs_config,
            complete_profits_df=self.complete_profits_df,
            complete_market_data_df=self.complete_market_data_df,
            complete_macro_trends_df=self.complete_macro_trends_df,
        )

        # 3) Generate the coin epoch's wallet configs
        epoch_weo.all_epochs_configs = epoch_weo.generate_epoch_configs()

        # 4) Generate and return its training & modeling data
        return epoch_weo, epoch_weo.generate_epochs_training_data()



    def _prepare_coin_epoch_base_config(self, lookback_duration: int):
        """
        Creates a config with dates offset by a wallets_coin_config lookback_duration.

        Params:
         - lookback_duration (int): How many days the coin lookback is offset vs the base config.
        """
        # Cache parsed base dates and base folder path
        base_training_data = copy.deepcopy(self.wallets_config.config['training_data'])
        base_modeling_start = datetime.strptime(base_training_data['modeling_period_start'], '%Y-%m-%d')
        base_modeling_end = datetime.strptime(base_training_data['modeling_period_end'], '%Y-%m-%d')
        base_training_window_starts = [
            datetime.strptime(dt, '%Y-%m-%d')
            for dt in base_training_data['training_window_starts']
        ]
        base_parquet_folder_base = Path(base_training_data['parquet_folder'])

        # Load and store complete dfs
        wallet_epochs_orchestrator = weo.WalletEpochsOrchestrator(
            self.wallets_config.config, # base config
            self.wallets_metrics_config,
            self.wallets_features_config,
            self.wallets_epochs_config
        )
        coin_epoch_base_config = wallet_epochs_orchestrator.build_epoch_config(
            lookback_duration,
            'coin_modeling',  # epoch_type, which has no impact on this class's data
            base_modeling_start,
            base_modeling_end,
            base_training_window_starts,
            base_parquet_folder_base
        )

        # Retain the parquet_folder so all periods are built in the same root directory
        coin_epoch_base_config['training_data']['parquet_folder'] = base_training_data['parquet_folder']

        return coin_epoch_base_config



    def _prepare_epoch_coins_config(self, epoch_weo) -> dict:
        """
        Prepare epoch-specific coins config with date suffix folders.
        """
        # Build a date suffix from the modeling_period_start
        wamo_date_suffix = pd.to_datetime(
            epoch_weo.base_config['training_data']['modeling_period_start']
        ).strftime('%Y%m%d')

        # Deep-copy the coin config and adjust folder paths
        epoch_coins_config = copy.deepcopy(self.wallets_coin_config.config)
        base_folder = epoch_coins_config['training_data']['parquet_folder']

        parquet_folder = f"{base_folder}/{wamo_date_suffix}"
        epoch_coins_config['training_data']['parquet_folder'] = parquet_folder
        Path(parquet_folder).mkdir(exist_ok=True)

        scores_folder = f"{parquet_folder}/scores"
        epoch_coins_config['training_data']['coins_wallet_scores_folder'] = scores_folder
        Path(scores_folder).mkdir(exist_ok=True)
        return epoch_coins_config



    def _generate_macro_indicators(self,
                                epoch_weo,
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
        # Use existing training data orchestrator for consistency
        wtdo_instance = wtdo.WalletTrainingDataOrchestrator(
            epoch_weo.base_config,
            self.wallets_metrics_config,
            self.wallets_features_config
        )

        period_macro_trends_df = dr.clean_macro_trends(
            self.complete_macro_trends_df,
            macro_trends_cols=list(self.wallets_coins_metrics_config['macro_trends'].keys()),
            start_date = None,  # retain historical data for indicators
            end_date = self.wallets_config['training_data']['modeling_period_end'])

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



    def _train_and_score_wallet_epoch(
        self,
        epoch_weo,
        epoch_coins_config: dict,
        epoch_training_dfs: tuple
    ) -> tuple[pd.DataFrame]:
        """
        Train wallet models for a single epoch and score wallets.

        Returns:
        - wamo_como_dfs: Tuple of four dfs:
            (wamo_training_data_df, wamo_modeling_data_df,
             como_training_data_df, como_modeling_data_df)
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

        # Train wallet models using all epoch dfs
        models_dict = epoch_wmo.train_wallet_models(*epoch_training_dfs)


        # 2) Generate this epoch's wamo/como training and modeling dfs
        # Build epochs config for only the coin and wallet modeling periods
        modeling_offset = self.wallets_config['training_data']['modeling_period_duration']
        coin_modeling_epochs_config = {
            'offset_epochs': {
                'offsets': [modeling_offset],
                'validation_offsets': [modeling_offset*2]
            }
        }
        epoch_wamo_weo = weo.WalletEpochsOrchestrator(
            base_config=epoch_weo.base_config,
            metrics_config=epoch_weo.metrics_config,
            features_config=epoch_weo.features_config,
            epochs_config=coin_modeling_epochs_config,          # custom wamo/como config
            complete_profits_df=epoch_weo.complete_profits_df,
            complete_market_data_df=epoch_weo.complete_market_data_df,
            complete_macro_trends_df=epoch_weo.complete_macro_trends_df,
        )
        # Generate TRAINING_DATA_DF for the WAllet MOdeling period and COin MOdeling periods
        wamo_como_dfs = epoch_wamo_weo.generate_epochs_training_data()

        # 3) Score wallets on the wamo training data
        epoch_wmo.predict_and_store(models_dict, wamo_como_dfs[0])

        return wamo_como_dfs



    def _generate_coin_features(
        self,
        epoch_weo,
        epoch_coins_config: dict,
        wamo_training_data_df: pd.DataFrame,
        como_training_data_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate and persist coin features for WaMo (modeling) and CoMo (validation) periods.
        """
        # 1) Load base dfs needed for coin feature generation
        (
            training_coin_cohort,
            wamo_profits_df,
            como_market_data_df,
            como_profits_df,
            investing_market_data_df
        ) = cfo.load_wallet_data_for_coin_features(
            epoch_weo.base_config,
            self.wallets_config['training_data']['parquet_folder']
        )

        # 2) Instantiate the CoinFeaturesOrchestrator for this epoch
        epoch_cfo = cfo.CoinFeaturesOrchestrator(
            epoch_weo.base_config,        # epoch-specific wallet config
            epoch_coins_config,           # epoch-specific coin config
            self.wallets_metrics_config,  # metrics config
            self.wallets_features_config, # features config
            self.wallets_epochs_config,   # epochs config or modeling-specific config
            training_coin_cohort          # initial coin cohort
        )

        # 3) Generate features for modeling (WaMo) period
        wamo_suffix = pd.to_datetime(
            epoch_weo.base_config['training_data']['coin_modeling_period_start']
        ).strftime('%Y%m%d')
        wamo_coin_features = epoch_cfo.generate_coin_features_for_period(
            wamo_profits_df,
            wamo_training_data_df,
            'modeling',
            wamo_suffix
        )

        # 4) Generate features for coin-modeling (CoMo) period
        como_coin_features = epoch_cfo.generate_coin_features_for_period(
            como_profits_df,
            como_training_data_df,
            'coin_modeling',
            wamo_suffix
        )

        # 5) Persist results to parquet
        base_folder = epoch_coins_config['training_data']['parquet_folder']
        wamo_coin_features.to_parquet(
            f"{base_folder}/wamo_coin_training_data_df_full.parquet", index=True
        )
        como_coin_features.to_parquet(
            f"{base_folder}/como_coin_training_data_df_full.parquet", index=True
        )
        # Return the generated feature and market dataframes for target calculation
        return wamo_coin_features, como_coin_features, como_market_data_df, investing_market_data_df



    def _generate_coin_target_vars(
        self,
        epoch_weo,
        epoch_coins_config: dict,
        wamo_features_df: pd.DataFrame,
        como_features_df: pd.DataFrame,
        como_market_data_df: pd.DataFrame,
        investing_market_data_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate and save target variable tables for WaMo and CoMo periods.
        """
        base_folder = epoch_coins_config['training_data']['parquet_folder']
        # Instantiate a fresh CoinFeaturesOrchestrator for target calculation
        features_generator = cfo.CoinFeaturesOrchestrator(
            epoch_weo.base_config,
            epoch_coins_config,
            self.wallets_metrics_config,
            self.wallets_features_config,
            self.wallets_epochs_config,
            None  # training_coin_cohort not required for target calc
        )

        # Calculate WaMo target variables
        wamo_target = features_generator.calculate_target_variables(
            como_market_data_df,
            epoch_weo.base_config['training_data']['coin_modeling_period_start'],
            epoch_weo.base_config['training_data']['coin_modeling_period_end'],
            set(wamo_features_df.index)
        )
        wamo_target.to_parquet(f"{base_folder}/wamo_coin_target_var_df.parquet", index=True)

        # Calculate CoMo target variables
        como_target = features_generator.calculate_target_variables(
            investing_market_data_df,
            epoch_weo.base_config['training_data']['investing_period_start'],
            epoch_weo.base_config['training_data']['investing_period_end'],
            set(como_features_df.index)
        )
        como_target.to_parquet(f"{base_folder}/como_coin_target_var_df.parquet", index=True)

        return wamo_target, como_target
