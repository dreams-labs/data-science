"""
Orchestrates the scoring of coin training data across multiple investing epochs.
"""
from pathlib import Path
import logging
import concurrent.futures
from datetime import datetime,timedelta
import json
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

        # Propagate loaded complete dfs to this orchestrator for use in build_all_wallet_data
        self.complete_profits_df = self.coin_epochs_orchestrator.complete_profits_df
        self.complete_market_data_df = self.coin_epochs_orchestrator.complete_market_data_df
        self.complete_macro_trends_df = self.coin_epochs_orchestrator.complete_macro_trends_df



    # -----------------------------------
    #         Primary Interface
    # -----------------------------------

    @u.timing_decorator(logging.MILESTONE)  # pylint: disable=no-member
    def orchestrate_coin_investment_cycles(
        self,
    ) -> pd.DataFrame:
        """
        Main orchestration method that:
        1. For each cycle, generates coin training data
        2. Trains a coin model on that data
        3. Scores coins in the subsequent period
        4. Consolidates results across all epochs
        """
        if self.wallets_coin_config['coin_modeling']['grid_search_params'].get('enabled',False):
            raise u.ConfigError("Grid search cannot be enabled for investment cycle orchestration.")

        # Extract investment cycles
        investment_cycles = self.coins_investing_config['investment_cycles']

        # Process each cycle concurrently
        n_threads = self.coins_investing_config['n_threads']['investment_cycles']
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            cycle_scores_dfs = list(executor.map(self._process_single_investment_cycle, investment_cycles))

        coin_scores_df = pd.concat(cycle_scores_dfs, ignore_index=False)
        u.notify('soft_twinkle_musical')

        return coin_scores_df





    # -----------------------------------
    #           Helper Methods
    # -----------------------------------

    def _compute_all_wallet_offsets(self) -> list[int]:
        """
        Extend the base wallet offsets by including combinations with investment cycles.
        """
        # Retrieve the base offsets from the parent CoinEpochsOrchestrator
        base_offsets = super()._compute_all_wallet_offsets()
        # Get the investment cycle offsets from this orchestrator's config
        investment_cycles = self.coins_investing_config['investment_cycles']
        # Generate extra offsets by adding each investment cycle to each base offset
        extra_offsets = {offset + cycle for offset in base_offsets for cycle in investment_cycles}
        # Combine and sort unique offsets
        combined_offsets = sorted(set(base_offsets) | extra_offsets)

        return combined_offsets


    def _process_single_investment_cycle(
        self,
        investment_cycle_offset: int
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
        cycle_modeling_dfs, buy_date = self._build_cycle_training_data(investment_cycle_offset)

        # Build model
        model_id, model_evaluator = self._train_cycle_coin_model(cycle_modeling_dfs)

        # Score investment data
        y_pred = self._score_prediction_period(model_id, cycle_modeling_dfs[2])

        # Calculate actual performance
        coin_returns_df = self._calculate_actual_performance(buy_date)

        # Combine predictions with performance
        cycle_performance_df = coin_returns_df.join(y_pred,how='inner')
        self._store_model_metrics(cycle_performance_df, model_evaluator)

        # Validate returns completeness
        if len(y_pred) > len(cycle_performance_df):
            raise ValueError(f"Only {len(cycle_performance_df)}/{len(y_pred)} of scored coins had price data.")
        # Validate NaNs
        nan_count = cycle_performance_df.isnull().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count}/{len(cycle_performance_df)} NaN values in coin returns.")

        return cycle_performance_df


    def _build_cycle_training_data(
        self,
        investment_cycle_offset: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate coin training and validation data for a specific investment cycle by
        orchestrating coin epochs and loading the resulting multiwindow datasets.

        Params:
        - investment_cycle_offset (int): Days offset from base modeling period for this cycle

        Returns:
        - Tuple of dataframes containing:
            - training_data_df (DataFrame): Coin features for model training
            - training_target_var_df (DataFrame): Target variables for model training
            - val_data_df (DataFrame): Coin features for model validation
            - val_target_var_df (DataFrame): Target variables for model validation
        - investment_start_date: when the investment purchases are made
        """
        # Identify file locations
        parquet_folder = f"{self.wallets_coin_config['training_data']['parquet_folder']}"
        date_prefix = (
            pd.to_datetime(self.wallets_config['training_data']['coin_modeling_period_start'])
            + timedelta(days=investment_cycle_offset)
        ).strftime('%y%m%d')

        # Calculate epochs that are shifted by investment_cycle days
        base_epochs = self.wallets_coin_config['training_data']['coin_epochs_training']
        training_epochs = [x + investment_cycle_offset for x in base_epochs]
        validation_epoch = [
            (max(training_epochs)
            + self.wallets_config['training_data']['modeling_period_duration'])
        ]
        investment_start_date = (
            pd.to_datetime(self.wallets_config['training_data']['coin_modeling_period_start'])
            + timedelta(days=validation_epoch[0])
        )

        # ----- Cached data handling -----
        # If all expected parquet files already exist for this investment cycle,
        # load them directly and skip the (expensive) regeneration step.
        training_data_path = Path(parquet_folder) / date_prefix / "training_multiwindow_coin_training_data_df.parquet"
        training_target_path = Path(parquet_folder) / date_prefix / "training_multiwindow_coin_target_var_df.parquet"
        val_data_path = Path(parquet_folder) / date_prefix / "validation_multiwindow_coin_training_data_df.parquet"
        val_target_path = Path(parquet_folder) / date_prefix / "validation_multiwindow_coin_target_var_df.parquet"

        if (all(p.exists() for p in [training_data_path, training_target_path, val_data_path, val_target_path])
            and not (self.coins_investing_config.get('training_data') or {}).get('toggle_overwrite_parquet', False)
            ):
            logger.milestone(
                "Investment cycle %s: using cached coin training/validation data.",
                date_prefix
            )
            training_data_df = pd.read_parquet(training_data_path)
            training_target_var_df = pd.read_parquet(training_target_path)
            val_data_df = pd.read_parquet(val_data_path)
            val_target_var_df = pd.read_parquet(val_target_path)
            return (
                (training_data_df,
                 training_target_var_df,
                 val_data_df,
                 val_target_var_df),
                investment_start_date
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

        # pylint:disable=line-too-long
        training_data_df = pd.read_parquet(f"{parquet_folder}/{date_prefix}/training_multiwindow_coin_training_data_df.parquet")
        training_target_var_df = pd.read_parquet(f"{parquet_folder}/{date_prefix}/training_multiwindow_coin_target_var_df.parquet")
        val_data_df = pd.read_parquet(f"{parquet_folder}/{date_prefix}/validation_multiwindow_coin_training_data_df.parquet")
        val_target_var_df = pd.read_parquet(f"{parquet_folder}/{date_prefix}/validation_multiwindow_coin_target_var_df.parquet")

        return (training_data_df, training_target_var_df, val_data_df, val_target_var_df), investment_start_date


    def _train_cycle_coin_model(
        self,
        cycle_modeling_dfs: pd.DataFrame
    ):
        """
        Train a coin model for this specific epoch.
        Returns model_id for scoring.
        """
        # Initialize and run model
        coin_model = cm.CoinModel(modeling_config=self.wallets_coin_config['coin_modeling'])
        coin_model_results = coin_model.construct_coin_model(*cycle_modeling_dfs)

        # Generate and save all model artifacts
        coin_model_id, coin_evaluator, _ = cimr.generate_and_save_coin_model_artifacts(
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

        return coin_model_id, coin_evaluator


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
        buy_date: datetime
    ) -> pd.DataFrame:
        """
        Determines the actual returns of coins over the investment cycle.
        """
        hold_time = self.coins_investing_config['trading']['hold_time']
        sell_date = buy_date + timedelta(days=hold_time)

        # Compute actual coin returns
        coin_returns_df = civa.calculate_coin_performance(
            self.coin_epochs_orchestrator.complete_market_data_df,
            buy_date,
            sell_date
        )

        return coin_returns_df


    def _store_model_metrics(
        self,
        cycle_performance_df: pd.DataFrame,
        model_evaluator
    ) -> dict:
        """
        Store quantile performance metrics and macro indicators for model evaluation.

        Params:
        - cycle_performance_df (DataFrame): Performance data with score and coin_return columns
        - model_evaluator: Model evaluator object with X_test and X_validation attributes

        Returns:
        - coin_model_dict (dict): Serializable metrics dictionary
        """
        # 1) Compute quantile model metrics
        all_quantiles = [x/20 for x in range(21)]
        all_quantiles_df = pd.DataFrame(index=all_quantiles)
        all_quantiles_df.index.name = 'quantile'

        coin_returns_df = cycle_performance_df.copy()
        coin_returns_df['quantile'] = round(coin_returns_df['score']*20).astype(int)/20
        coin_returns_df['return_wins'] = u.winsorize(coin_returns_df['coin_return'])

        quantile_returns_df = (coin_returns_df
                            .groupby('quantile')
                            .agg({
                                'coin_return': ['mean', 'median', 'count'],
                                'return_wins': 'mean'
                            })
                            .round(4))
        quantile_returns_df.columns = ['coin_return_mean', 'coin_return_median',
                                       'coin_return_count', 'return_wins_mean']

        all_quantiles_df = all_quantiles_df.join(quantile_returns_df)
        all_quantiles_df['coin_return_count'] = all_quantiles_df['coin_return_count'].fillna(0)

        # 2) Extract macro indicators
        max_date = model_evaluator.X_test.index.get_level_values('coin_epoch_start_date').max()
        selected_row = model_evaluator.X_test.loc[
            model_evaluator.X_test.index.get_level_values('coin_epoch_start_date') == max_date
        ].iloc[0]

        macro_mask = model_evaluator.X_validation.columns.astype(str).str.startswith('macro|')
        macro_cols = model_evaluator.X_validation.columns[macro_mask]
        macro_indicators = selected_row[macro_cols]

        # 3) Convert to dict and save
        coin_model_dict = {
            'quantile_returns': all_quantiles_df.reset_index().to_dict(),
            'macro_indicators': macro_indicators.to_dict(),
            'model_id': model_evaluator.model_id
        }

        json_save_path = (
            f"{self.wallets_coin_config['training_data']['parquet_folder']}/"
            f"{pd.to_datetime(self.wallets_config['training_data']['coin_modeling_period_start']).strftime('%y%m%d')}/"
            "coin_model_ids.json"
        )
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(coin_model_dict, f, indent=4, default=u.numpy_type_converter)
        logger.milestone(f"Saved coin_model_ids.json to {json_save_path}.")
