"""
Orchestrates the scoring of coin training data across multiple investing epochs.
"""
from pathlib import Path
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta
import pandas as pd

# Local module imports
import coin_modeling.coin_model as cm
import coin_modeling.coin_epochs_orchestrator as ceo
import coin_insights.coin_model_reporting as cimr
import coin_insights.coin_validation_analysis as civa
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)


# ----------------------------------------
#       Primary Orchestration Class
# ----------------------------------------

# CoinModel
class CoinInvestingOrchestrator(ceo.CoinEpochsOrchestrator):
    """
    Orchestrates wallet model prediction scoring across multiple investing epochs by
    offsetting base config dates and scoring with a pre-trained model.

    Inherits data loading, config management, and orchestration infrastructure
    from CoinEpochsOrchestrator while focusing on prediction scoring rather than training.
    """

    def __init__(
        self,

        # coin investing config
        coins_investing_config: dict,

        # coin model configs (inherits from CoinEpochsOrchestrator)
        wallets_coin_config: dict,
        wallets_coins_metrics_config: dict,

        # wallets model configs
        wallets_config: dict,
        wallets_metrics_config: dict,
        wallets_features_config: dict,
        wallets_epochs_config: dict,
    ):
        """
        Initialize the investing epochs orchestrator with a pre-trained model.
        """
        # Ensure configs are dicts and not the custom config classes
        if not (isinstance(wallets_config,dict) and isinstance(wallets_coin_config,dict)):
            raise ValueError("CoinEpochsOrchestrator configs must be dtype=='dict'.")

        # investing-specific configs
        self.coins_investing_config = coins_investing_config

        # coin model configs (inherits from CoinEpochsOrchestrator)
        self.wallets_coin_config = wallets_coin_config
        self.wallets_coins_metrics_config = wallets_coins_metrics_config

        # wallets model configs
        self.wallets_config = wallets_config
        self.wallets_metrics_config = wallets_metrics_config
        self.wallets_features_config = wallets_features_config
        self.wallets_epochs_config = wallets_epochs_config

        # coin epochs orchestrator
        self.coin_epochs_orchestrator = ceo.CoinEpochsOrchestrator(
            self.wallets_coin_config,
            self.wallets_coins_metrics_config,
            self.wallets_config,
            self.wallets_metrics_config,
            self.wallets_features_config,
            self.wallets_epochs_config,
        )
        self.coin_epochs_orchestrator.load_complete_raw_datasets()





    # -----------------------------------
    #         Primary Interface
    # -----------------------------------

    @u.timing_decorator(logging.MILESTONE)
    def orchestrate_coin_investment_cycles(
        self,
    ) -> pd.DataFrame:
        """
        Main orchestration method that:
        1. For each epoch, generates coin training data
        2. Trains a coin model on that data
        3. Scores coins in the subsequent period
        4. Consolidates results across all epochs

        Returns:
        - consolidated_results_df: Multi-indexed on (coin_id, epoch_start_date)
          with columns: score, actual_return, is_buy
        """

        # # Store model ID for later reference
        # investment_cycles = self.coins_investing_config['investment_cycles']
        # n_threads = self.coins_investing_config['n_threads']['investment_cycles']

        # # Process each epoch concurrently
        # with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        #     cw_scores_dfs = list(executor.map(self._score_investing_epoch, investment_cycles))

        # all_cw_scores_df = pd.concat(cw_scores_dfs, ignore_index=False)
        # u.notify('soft_twinkle_musical')

        # return all_cw_scores_df




    # -----------------------------------
    #       Scoring Helper Methods
    # -----------------------------------

    def _process_single_investment_cycle(
        self,
        investment_cycle: int,
        hold_time: int
    ) -> pd.DataFrame:
        """
        Process one epoch:
        1. Generate training data for epoch
        2. Train coin model
        3. Generate scoring data for prediction period
        4. Score coins and calculate actual returns
        5. Determine buy signals
        """
        # Generate training data
        cycle_modeling_dfs, buy_date = self._build_cycle_training_data(investment_cycle)

        # Build model
        model_id = self._train_cycle_coin_model(cycle_modeling_dfs)

        # Score investment data
        y_pred = self._score_prediction_period(model_id, cycle_modeling_dfs[2])

        # Calculate actual performance
        coin_returns_df = self._calculate_actual_performance(buy_date, hold_time)

        # Combine predictions with performance
        cycle_performance_df = coin_returns_df.join(y_pred,how='inner')

        # Validate returns completeness
        if len(y_pred) > len(cycle_performance_df):
            raise ValueError(f"Only {len(cycle_performance_df)}/{len(y_pred)} of scored coins had price data.")
        # Validate NaNs
        nan_count = cycle_performance_df.isnull().sum().sum()
        if nan_count > 0:
            raise ValueError(f"NaN values detected in output: {nan_count} total NaNs")

        return cycle_performance_df


    def _build_cycle_training_data(
        self,
        investment_cycle: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate coin training and validation data for a specific investment cycle by
        orchestrating coin epochs and loading the resulting multiwindow datasets.

        Params:
        - investment_cycle (int): Days offset from base modeling period for this cycle

        Returns:
        - Tuple of dataframes containing:
            - training_data_df (DataFrame): Coin features for model training
            - training_target_var_df (DataFrame): Target variables for model training
            - val_data_df (DataFrame): Coin features for model validation
            - val_target_var_df (DataFrame): Target variables for model validation
        - investment_start_date: when the investment purchases are made
        """
        # Identify file locations
        date_prefix = (
            pd.to_datetime(self.wallets_config['training_data']['modeling_period_start'])
            + timedelta(days=investment_cycle)
        ).strftime('%Y%m%d')
        parquet_folder = f"{self.wallets_coin_config['training_data']['parquet_folder']}/{date_prefix}"

        # Calculate epochs that are shifted by investment_cycle days
        base_epochs = self.wallets_coin_config['training_data']['coin_epochs_training']
        training_epochs = [x + investment_cycle for x in base_epochs]
        validation_epoch = [
            (max(training_epochs)
            + self.wallets_config['training_data']['modeling_period_duration'])
        ]
        investment_start_date = (
            pd.to_datetime(self.wallets_config['training_data']['coin_modeling_period_start'])
            + timedelta(days=validation_epoch[0])
        )

        # Generate all coin training data
        self.coin_epochs_orchestrator.orchestrate_coin_epochs(
            training_epochs,
            file_prefix=f'{date_prefix}/training_'
        )
        # Generate all coin validation data
        self.coin_epochs_orchestrator.orchestrate_coin_epochs(
            validation_epoch,
            file_prefix=f'{date_prefix}/validation_'
        )

        training_data_df = pd.read_parquet(f"{parquet_folder}/training_multiwindow_coin_training_data_df.parquet")
        training_target_var_df = pd.read_parquet(f"{parquet_folder}/training_multiwindow_coin_target_var_df.parquet")
        val_data_df = pd.read_parquet(f"{parquet_folder}/validation_multiwindow_coin_training_data_df.parquet")
        val_target_var_df = pd.read_parquet(f"{parquet_folder}/validation_multiwindow_coin_target_var_df.parquet")

        return (training_data_df, training_target_var_df, val_data_df, val_target_var_df), investment_start_date


    def _train_cycle_coin_model(
        self,
        cycle_modeling_dfs: pd.DataFrame
    ) -> str:
        """
        Train a coin model for this specific epoch.
        Returns model_id for scoring.
        """
        # Initialize and run model
        coin_model = cm.CoinModel(modeling_config=self.wallets_coin_config['coin_modeling'])
        coin_model_results = coin_model.construct_coin_model(*cycle_modeling_dfs)

        # Generate and save all model artifacts
        coin_model_id, coin_evaluator, coin_scores_df = cimr.generate_and_save_coin_model_artifacts(
            model_results=coin_model_results,
            base_path='../artifacts/coin_modeling',
            configs = {
                'wallets_coin_config': self.wallets_coin_config,
                'wallets_config': self.wallets_config,
                'wallets_epochs_config': self.wallets_epochs_config,
                'wallets_features_config': self.wallets_features_config,
                'wallets_metrics_config': self.wallets_metrics_config,
            }
        )
        coin_evaluator.plot_wallet_evaluation()

        return coin_model_id


    def _score_prediction_period(
        self,
        model_id: str,
        training_data_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Score coins using the trained model and calculate actual returns.
        """
        # Load and predict
        y_pred = ceo.CoinEpochsOrchestrator.score_coin_training_data(
            self.wallets_coin_config,
            model_id,
            self.wallets_coin_config['training_data']['model_artifacts_folder'],
            training_data_df,
        )

        return y_pred


    def _calculate_actual_performance(
        self,
        buy_date: datetime,
        hold_time: int
    ) -> pd.DataFrame:
        """
        Determines the actual returns of coins over the investment cycle.
        """
        sell_date = buy_date + timedelta(days=hold_time)

        # Compute actual coin returns
        coin_returns_df = civa.calculate_coin_performance(
            self.coin_epochs_orchestrator.complete_market_data_df,
            buy_date,
            sell_date
        )

        return coin_returns_df

