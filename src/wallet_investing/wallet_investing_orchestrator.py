"""
Orchestrates the scoring of wallet training data across multiple investing epochs using
a pre-trained wallet model to evaluate long-term prediction performance.
"""
import logging
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

        Params:
        - wallets_config, wallets_metrics_config, etc.: Standard wallet model configs
        - investing_epochs (list[int]): Day offsets for future prediction periods
        - trained_model_pipeline: Pre-trained sklearn pipeline for scoring
        - model_id (str): UUID identifier for the trained model
        - complete_*_df: Pre-loaded datasets (optional, will load if not provided)
        """
        # Ensure configs are dicts and not the custom config classes
        if not isinstance(wallets_config,dict):
            raise ValueError("InvestingEpochsOrchestrator configs must be dtype=='dict'.")

        # investing-specific configs
        self.investing_config = investing_config
        self.investing_epochs = None
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
        model_id,
        file_prefix: str = 'investing_',
    ) -> pd.DataFrame:
        """
        Orchestrate wallet model scoring across multiple investing epochs.

        Params:
        - model_id (str): The ID of the model to use for predictions
        - file_prefix (str): Prefix for output parquet files

        Returns:
        - investing_predictions_df (DataFrame): Predictions across all epochs with performance metrics
        """
        # Store model ID for later reference
        self.model_id = model_id

    @staticmethod
    def analyze_investing_performance(
        investing_predictions_df: pd.DataFrame,
        model_id: str,
        artifacts_path: str
    ) -> dict:
        """
        Analyze model prediction performance across investing epochs.

        Params:
        - investing_predictions_df: Predictions across epochs
        - model_id: Model identifier
        - artifacts_path: Base directory for saving analysis artifacts

        Returns:
        - performance_metrics (dict): Aggregated performance statistics
        """

    # -----------------------------------
    #           Helper Methods
    # -----------------------------------

    # --------------
    # Primary Helper
    # --------------
    def _process_investing_epoch(
        self,
        lookback_duration: int
    ) -> tuple[datetime, pd.DataFrame]:
        """
        Process a single investing epoch: generate wallet training data and score with pre-trained model.

        Key Steps:
        1. Generate epoch-specific config files
        2. Generate wallet-level training features for the epoch
        3. Score features using the pre-trained model
        4. Calculate actual performance if target data available

        Params:
        - lookback_duration (int): Days offset from base modeling period

        Returns:
        - coin_returns_df (DataFrame): Actual returns of all coins with boolean column 'is_buy' that
            identifies which coins were bought.
        """
        # Prepare config files
        epoch_wallets_config, epoch_wallets_epochs_config = self._prepare_epoch_configs(lookback_duration)

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
        missing_coins = set(buy_coins) - set(coin_returns_df.index.values)
        if len(missing_coins) > 0:
            raise ValueError(f"Not all buy coins had actual return values. Missing coins: {missing_coins}")

        # Append buys to returns_df
        coin_returns_df['is_buy'] = coin_returns_df.index.isin(buy_coins)

        return coin_returns_df



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




    # -----------------------------------
    #     Performance Analysis Methods
    # -----------------------------------

    def _aggregate_investing_results(
        self,
        epoch_predictions: dict[datetime, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Aggregate predictions and performance across all investing epochs.

        Params:
        - epoch_predictions: Dict mapping epoch dates to prediction DataFrames

        Returns:
        - aggregated_df: MultiIndexed DataFrame with all epochs and performance metrics
        """

    def _validate_investing_epochs_coverage(self) -> None:
        """
        Verify that complete datasets cover all required investing epoch dates.
        """

    def _calculate_prediction_stability(self, predictions_df: pd.DataFrame) -> dict:
        """
        Calculate how stable model predictions are across different epochs.
        """

    def _calculate_prediction_accuracy(self, predictions_df: pd.DataFrame) -> dict:
        """
        Calculate prediction accuracy metrics where actual performance data exists.
        """

    def _generate_investing_performance_charts(
        self,
        predictions_df: pd.DataFrame,
        artifacts_path: str
    ) -> None:
        """
        Generate charts showing model performance across investing epochs.
        """
