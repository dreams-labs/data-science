"""
Orchestrates the scoring of wallet training data across multiple investing epochs using
a pre-trained wallet model to evaluate long-term prediction performance.
"""
from pathlib import Path
import logging
import concurrent.futures
import copy
from datetime import datetime
import pandas as pd

# Local module imports
import wallet_modeling.wallet_epochs_orchestrator as weo
import wallet_modeling.wallet_training_data_orchestrator as wtdo
import wallet_insights.wallet_validation_analysis as wiva
import coin_modeling.coin_epochs_orchestrator as ceo
import coin_insights.coin_validation_analysis as civa
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)


# ----------------------------------------
#       Primary Orchestration Class
# ----------------------------------------

class InvestingEpochsOrchestrator(ceo.CoinEpochsOrchestrator):
    """
    Orchestrates wallet model prediction scoring across multiple investing epochs by
    offsetting base config dates and scoring with a pre-trained model.

    Inherits data loading, config management, and orchestration infrastructure
    from CoinEpochsOrchestrator while focusing on prediction scoring rather than training.
    """

    def __init__(
        self,

        # investing config
        investing_config: dict,

        # wallets model configs
        wallets_config: dict,
        wallets_metrics_config: dict,
        wallets_features_config: dict,
        wallets_epochs_config: dict,

        # complete datasets
        complete_profits_df: pd.DataFrame,
        complete_market_data_df: pd.DataFrame,
        complete_macro_trends_df: pd.DataFrame,
        complete_hybrid_cw_id_df: pd.DataFrame
    ):
        """
        Initialize the investing epochs orchestrator with a pre-trained model.
        """
        # Ensure configs are dicts and not the custom config classes
        if not isinstance(wallets_config,dict):
            raise ValueError("InvestingEpochsOrchestrator configs must be dtype=='dict'.")

        # investing-specific configs
        self.investing_config = investing_config
        self.parquet_folder = (wallets_config['training_data']['parquet_folder']
                               .replace('wallet_modeling_dfs','wallet_investing_dfs'))
        self.model_id = None

        # wallets model configs
        self.wallets_config = wallets_config
        self.wallets_metrics_config = wallets_metrics_config
        self.wallets_features_config = wallets_features_config
        self.wallets_epochs_config = wallets_epochs_config

        # complete datasets
        self.complete_profits_df = complete_profits_df
        self.complete_market_data_df = complete_market_data_df
        self.complete_macro_trends_df = complete_macro_trends_df
        self.complete_hybrid_cw_id_df = complete_hybrid_cw_id_df

    # -----------------------------------
    #         Primary Interface
    # -----------------------------------

    @u.timing_decorator(logging.MILESTONE)  # pylint: disable=no-member
    def orchestrate_investing_epochs(
        self,
        model_id
    ) -> pd.DataFrame:
        """
        Orchestrate wallet model scoring across multiple investing epochs.

        Params:
        - model_id (str): The ID of the model to use for predictions

        Returns:
        - epoch_metrics_df (DataFrame): Performance metrics for each epoch offset
        """
        # Store model ID for later reference
        self.model_id = model_id

        epoch_metrics_list = []
        trading_dfs_list = []

        offsets = self.investing_config['investing_epochs']

        # Process each epoch concurrently
        n_threads = self.investing_config['n_threads']['investing_epochs']
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            trading_dfs = list(executor.map(self._process_investing_epoch, offsets))

        # Compute metrics for each epoch result
        for offset, trading_df in zip(offsets, trading_dfs):
            trading_dfs_list.append(trading_df)

            epoch_modeling_start = trading_df.index.get_level_values('epoch_modeling_start').unique()[0]
            coins_bought = trading_df[trading_df['is_buy']]['coin_return'].count()

            median_buy_return = trading_df[trading_df['is_buy']]['coin_return'].median()
            mean_buy_return = trading_df[trading_df['is_buy']]['coin_return'].mean()
            wins_buy_return = u.winsorize(trading_df[trading_df['is_buy']]['coin_return'], 0.01).mean()

            median_overall_return = trading_df['coin_return'].median()
            mean_overall_return = trading_df['coin_return'].mean()
            wins_overall_return = u.winsorize(trading_df['coin_return'], 0.01).mean()

            epoch_metrics_list.append({
                'offset': offset,
                'epoch_modeling_start': epoch_modeling_start,
                'coins_bought': coins_bought,
                'median_buy_return': median_buy_return,
                'median_overall_return': median_overall_return,
                'wins_buy_return': wins_buy_return,
                'wins_overall_return': wins_overall_return,
                'mean_overall_return': mean_overall_return,
                'mean_buy_return': mean_buy_return,
            })

            logger.milestone(f"Identified {coins_bought} coins to buy for epoch {offset}.")

        epoch_metrics_df = pd.DataFrame(epoch_metrics_list).sort_values(by='offset')
        all_trading_df = pd.concat(trading_dfs_list, ignore_index=False)

        u.notify('soft_twinkle_musical')

        return epoch_metrics_df,all_trading_df



    # -----------------------------------
    #           Helper Methods
    # -----------------------------------

    # --------------
    # Primary Helper
    # --------------
    def _process_investing_epoch(
        self,
        offset_days: int
    ) -> tuple[datetime, pd.DataFrame]:
        """
        Process a single investing epoch: generate wallet training data and score with pre-trained model.

        Key Steps:
        1. Generate epoch-specific config files
        2. Generate wallet-level training features for the epoch
        3. Score features using the pre-trained model
        4. Calculate actual performance if target data available

        Params:
        - offset_days (int): Days offset from base modeling period. Positive values move the modeling
            period later.

        Returns:
        - trading_df (DataFrame): Actual returns of all coins with columns:
            'coin_id' (str): multiindex
            'modeling_epoch_start' (datetime): multiindex of modeling_period_start dates
            'return' (float): actual coin performance
            'is_buy' (bool): identifies which coins were bought.
        """
        # Prepare config files
        epoch_wallets_config, epoch_wallets_epochs_config = self._prepare_epoch_configs(offset_days)

        # Generate training_data_df
        epoch_training_data_df = self._generate_wallet_training_data_for_epoch(
            epoch_wallets_config,
            epoch_wallets_epochs_config
        )

        # Identify buy signals using the model
        cw_preds = wiva.load_and_predict(
            self.model_id,
            epoch_training_data_df,
            self.wallets_config['training_data']['model_artifacts_folder']
        )
        buy_coins = self._identify_buy_signals(cw_preds)

        # Compute actual coin returns
        coin_returns_df = civa.calculate_coin_performance(
            self.complete_market_data_df,
            epoch_wallets_config['training_data']['modeling_period_start'],
            epoch_wallets_config['training_data']['modeling_period_end']
        )

        # Merge to create trading_df
        trading_df = coin_returns_df[~coin_returns_df['coin_return'].isna()].copy()
        missing_coins = set(buy_coins) - set(trading_df.index.values)
        if len(missing_coins) > 0:
            raise ValueError(f"Not all buy coins had actual return values. Missing coins: {missing_coins}")

        # Append buys to returns_df
        trading_df['is_buy'] = trading_df.index.isin(buy_coins)

        # Append epoch to df
        trading_df['epoch_modeling_start'] = epoch_wallets_config['training_data']['modeling_period_start']
        trading_df = trading_df.reset_index().set_index(['coin_id','epoch_modeling_start'])

        return trading_df



    # ------------------------
    # Epoch Processing Helpers
    # ------------------------
    def _prepare_epoch_configs(self, lookback_duration: int) -> tuple[dict, dict, datetime]:
        """
        Prepare epoch-specific configuration files for investing predictions.

        Params:
        - lookback_duration (int): Days offset from base modeling period. Positive values
            move the modeling period later and negative move it earlier.

        Returns:
        - epoch_wallets_config (dict): Wallets config with offset dates and training_data_only flag
        - epoch_wallets_epochs_config (dict): Epochs config without validation offsets
        """
        logger.milestone(f"Beginning generation of investing data for offset '{lookback_duration}' days.")

        # Generate epoch-specific wallets config by offsetting base dates
        epoch_wallets_config = self._prepare_coin_epoch_base_config(lookback_duration)

        # Set training_data_only flag to skip target variable generation
        epoch_wallets_config['training_data']['training_data_only'] = True

        # Create wallets_epochs_config without any validation offsets
        epoch_wallets_epochs_config = copy.deepcopy(self.wallets_epochs_config)
        epoch_wallets_epochs_config['offset_epochs']['validation_offsets'] = []

        logger.info(f"Configured investing epoch for offset '{lookback_duration}' days.")

        return epoch_wallets_config, epoch_wallets_epochs_config


    def _generate_wallet_training_data_for_epoch(
        self,
        epoch_wallets_config: dict,
        epoch_wallets_epochs_config: dict
    ) -> pd.DataFrame:
        """
        Generate wallet training data for a single investing epoch.

        Params:
        - epoch_wallets_config: Epoch-specific wallet configuration
        - epoch_wallets_epochs_config: Epoch-specific epochs configuration

        Returns:
        - wallet_training_data_df: Training features for the epoch
        """
        epoch_date = datetime.strptime(
            epoch_wallets_config['training_data']['modeling_period_start'],
            '%Y-%m-%d'
        ).strftime('%y%m%d')
        file_location = Path(self.parquet_folder) / epoch_date / 'training_data_df.parquet'
        # Ensure epoch directory exists
        file_location.parent.mkdir(parents=True, exist_ok=True)

        # Load existing training data if available
        if (file_location.exists() and
            not self.investing_config['training_data']['toggle_overwrite_parquet']
        ):
            logger.info(f"Loading existing wallet training data for epoch {epoch_date} from {file_location}")
            return pd.read_parquet(file_location)

        epoch_weo = weo.WalletEpochsOrchestrator(
            base_config=epoch_wallets_config,               # epoch-specific config
            metrics_config=self.wallets_metrics_config,
            features_config=self.wallets_features_config,
            epochs_config=epoch_wallets_epochs_config,      # epoch-specific config
            complete_profits_df=self.complete_profits_df,
            complete_market_data_df=self.complete_market_data_df,
            complete_macro_trends_df=self.complete_macro_trends_df,
            complete_hybrid_cw_id_df = self.complete_hybrid_cw_id_df
        )

        # Generate wallets training & modeling data
        epoch_training_data_df,_,_,_ = epoch_weo.generate_epochs_training_data()
        # Ensure output directory exists
        file_location.parent.mkdir(parents=True, exist_ok=True)
        # Save generated training data
        epoch_training_data_df.to_parquet(file_location)
        logger.info(f"Saved wallet training data for epoch {epoch_date} to {file_location}")

        return epoch_training_data_df


    def _identify_buy_signals(
        self,
        cw_preds: pd.Series
    ) -> list:
        """
        Convert coin-wallet predictions into buy signals for coins based on scoring thresholds.

        Params:
        - cw_preds (Series): Predictions indexed by hybrid coin-wallet IDs
        - complete_hybrid_cw_id_df (DataFrame): Mapping for dehybridizing wallet addresses

        Returns:
        - buy_coins (Series): Coin IDs that meet buy criteria
        """
        score_threshold = self.investing_config['trading']['score_threshold']
        min_scores = self.investing_config['trading']['min_scores']

        # Extract coin_ids from coin-wallet pair hybrid IDs
        preds_df = pd.DataFrame(cw_preds)
        preds_df.columns = ['score']
        preds_df = wtdo.dehybridize_wallet_address(preds_df, self.complete_hybrid_cw_id_df)

        # Count how many coin-wallet pairs are above the score_threshold
        buys_df = preds_df[preds_df['score'] > score_threshold]
        buys_df = pd.DataFrame(buys_df.reset_index()
                                .groupby('coin_id', observed=True)
                                .size())
        buys_df.columns = ['high_scores']

        # Identify coins with enough high scores to buy
        buy_coins = list(buys_df[buys_df['high_scores'] > min_scores].index)

        logger.info(f"Identified {len(buy_coins)} coins to buy.")

        return buy_coins
