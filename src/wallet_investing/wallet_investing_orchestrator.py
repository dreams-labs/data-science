"""
Orchestrates the scoring of wallet training data across multiple investing epochs using
a pre-trained wallet model to evaluate long-term prediction performance.
"""
from pathlib import Path
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta
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

# WalletModel
class WalletsInvestingOrchestrator(ceo.CoinEpochsOrchestrator):
    """
    Orchestrates wallet model prediction scoring across multiple investing epochs by
    offsetting base config dates and scoring with a pre-trained model.

    Inherits data loading, config management, and orchestration infrastructure
    from CoinEpochsOrchestrator while focusing on prediction scoring rather than training.
    """

    def __init__(
        self,

        # investing config
        wallets_investing_config: dict,

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
            raise ValueError("WalletsInvestingOrchestrator configs must be dtype=='dict'.")

        # investing-specific configs
        self.wallets_investing_config = wallets_investing_config
        self.parquet_folder = (wallets_config['training_data']['parquet_folder']
                               .replace('wallet_modeling_dfs','wallet_investing_dfs'))
        self.model_id = None
        self.buy_duration = None

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
    def score_all_investment_cycles(
        self,
        model_id: str,
    ) -> pd.DataFrame:
        """
        Orchestrate wallet model scoring across multiple investing epochs.

        Params:
        - model_id (str): The ID of the model to use for predictions

        Returns:
        - all_cw_scores_df (pd.DataFrame): df with column 'score' for every coin-wallet-epoch multiindex tuple.
        """
        # Store model ID for later reference
        self.model_id = model_id

        offsets = self.wallets_investing_config['investment_cycles']
        n_threads = self.wallets_investing_config['n_threads']['investment_cycles']

        # Process each epoch concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            cw_scores_dfs = list(executor.map(self._score_investing_epoch, offsets))

        all_cw_scores_df = pd.concat(cw_scores_dfs, ignore_index=False)
        u.notify('soft_twinkle_musical')

        return all_cw_scores_df


    def determine_epoch_buys(
        self,
        cw_scores_df: pd.DataFrame,
        buy_duration: int = None
    ) -> pd.DataFrame:
        """
        Assigns buys to coins in a boolean is_buy column based on params in the wallets_investing_config.yaml.

        Params:
        - cw_scores_df (pd.DataFrame): df with column 'score' for every coin-wallet-epoch multiindex tuple.

        Returns:
        - trading_df (pd.DataFrame): df with multiindex on coin_id,epoch_start_date and columns:
            coin_return: showing the actual performance of the coin over the epoch modeling period
            is_buy: boolean showing whether the coin would be bought based on config params
        """
        # Determine how many days after the prediction to check the price change
        if buy_duration is not None:
            self.buy_duration = buy_duration
        else:
            self.buy_duration = self.wallets_config['training_data']['modeling_period_duration'] - 1

        unique_epochs = cw_scores_df.index.get_level_values('epoch_start_date').unique()
        max_workers = self.wallets_investing_config['n_threads']['buy_logic_epochs']

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all epoch processing tasks
            future_to_epoch = {
                executor.submit(self._determine_single_epoch_buys, epoch_start, cw_scores_df): epoch_start
                for epoch_start in unique_epochs
            }

            # Collect results as they complete
            all_returns_dfs = []
            for future in concurrent.futures.as_completed(future_to_epoch):
                epoch_start = future_to_epoch[future]
                try:
                    epoch_result = future.result()
                    all_returns_dfs.append(epoch_result)
                except Exception as exc:
                    raise ValueError(f'Epoch {epoch_start} generated an exception: {exc}') from exc

        trading_df = pd.concat(all_returns_dfs)
        return trading_df

    # -----------------------------------
    #       Scoring Helper Methods
    # -----------------------------------


    def _determine_single_epoch_buys(self, epoch_start, cw_scores_df):
        """
        Helper method to process a single epoch for multithreading.

        Params:
        - epoch_start: the start date for this epoch
        - cw_scores_df: full scores dataframe (filtered within method)

        Returns:
        - epoch_results_df: processed results for this epoch
        """
        scored_coins = set(cw_scores_df.index.get_level_values('coin_id').values.astype(str))

        # 1) Compute actual coin returns
        # ------------------------------
        epoch_end = epoch_start + timedelta(days=self.buy_duration)
        coin_returns_df = civa.calculate_coin_performance(
            self.complete_market_data_df,
            epoch_start,
            epoch_end
        )
        coin_returns_df = coin_returns_df[coin_returns_df.index.isin(scored_coins)]

        # Confirm completeness
        missing_coins = set(coin_returns_df.index.astype(str)) - scored_coins
        if len(missing_coins) > 0:
            raise ValueError(f"No returns found for scored coins {missing_coins}")

        # Append epoch date
        coin_returns_df['epoch_start_date'] = epoch_start
        coin_returns_df = coin_returns_df.copy().reset_index().set_index(['coin_id','epoch_start_date'])

        # 2) Identify buy signals
        # -----------------------
        epoch_scores_df = cw_scores_df[cw_scores_df.index.get_level_values('epoch_start_date') == epoch_start]
        epoch_buy_coins = self._identify_buy_signals(epoch_scores_df)

        # Add boolean
        coin_returns_df['is_buy'] = (coin_returns_df.index.get_level_values('coin_id').astype(str)
                                    .isin(epoch_buy_coins))

        return coin_returns_df



    def _score_investing_epoch(
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
        logger.milestone(f"Scoring coin-wallet pairs for offset of '{offset_days}' days with "
                         f"modeling start {epoch_wallets_config['training_data']['modeling_period_start']}...")

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

        # Extract coin_ids from coin-wallet pair hybrid IDs
        preds_df = pd.DataFrame(cw_preds)
        preds_df.columns = ['score']
        cw_scores_df = wtdo.dehybridize_wallet_address(preds_df, self.complete_hybrid_cw_id_df)

        return cw_scores_df


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

        # Create wallets_epochs_config for only the current lookback. Note that the
        #  modeling period boundaries have already been shifted in the epoch_wallets_config
        #  so the offset here is 0.
        epoch_wallets_epochs_config = {
            'offset_epochs': {
                'offsets': [0],
                'validation_offsets': []
            }
        }

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
            not self.wallets_investing_config['training_data']['toggle_overwrite_parquet']
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



    def _compute_all_wallet_offsets(self) -> list[int]:
        """
        Compute all wallet-epoch offsets for investing, combining base wallet offsets
        and investing-specific offsets.

        This override replaces the coin-based offsets logic. It:
        1. Reads investing epoch offsets from self.wallets_investing_config['investment_cycles'].
        2. Reads base wallet training and validation offsets from self.wallets_epochs_config.
        3. Combines them, then for each investing offset adds it to each base offset.
        4. Returns a sorted unique list of all offsets.

        Returns:
        - list[int]: sorted unique day offsets to build wallet data for.
        """
        # Investing-specific epochs
        investing_offsets: list[int] = self.wallets_investing_config['investment_cycles']

        # Base wallet epochs from config
        wallets_epochs_cfg = self.wallets_epochs_config['offset_epochs']
        base_wallet_offsets: list[int] = wallets_epochs_cfg.get('offsets', [])
        base_wallet_val_offsets: list[int] = wallets_epochs_cfg.get('validation_offsets', [])

        # Start with all raw offsets
        all_offsets: list[int] = investing_offsets + base_wallet_offsets + base_wallet_val_offsets

        # Generate combined offsets by adding each investing offset to each base offset
        for invest in investing_offsets:
            all_offsets += [invest + base for base in base_wallet_offsets]

        # Deduplicate and sort
        unique_offsets: list[int] = sorted(set(all_offsets))
        return unique_offsets




    # -----------------------------------
    #       Trading Helper Methods
    # -----------------------------------

    def _identify_buy_signals(
        self,
        cw_scores_df: pd.DataFrame
    ) -> list:
        """
        Convert coin-wallet predictions into buy signals for coins based on scoring thresholds.

        Params:
        - cw_scores_df (DataFrame): Scores of coin-wallet pairs for a given epoch.

        Returns:
        - buy_coins (Series): Coin IDs that meet buy criteria
        """
        high_score_threshold = self.wallets_investing_config['trading']['high_score_threshold']
        min_high_scores = self.wallets_investing_config['trading']['min_high_scores']
        min_average_score = self.wallets_investing_config['trading']['min_average_score']
        max_coins_per_epoch = self.wallets_investing_config['trading']['max_coins_per_epoch']

        # Count how many coin-wallet pairs are above the high_score_threshold
        buys_df = cw_scores_df[cw_scores_df['score'] > high_score_threshold]
        buys_df = pd.DataFrame(buys_df.reset_index()
                                .groupby('coin_id', observed=True)
                                .size())
        buys_df.columns = ['high_scores']

        # Identify coins with enough high scores to buy
        buys_df = buys_df[buys_df['high_scores'] > min_high_scores]

        # Determine mean scores
        mean_scores = cw_scores_df.copy().groupby('coin_id',observed=True)['score'].mean()
        mean_scores.name = 'mean_score'
        buy_coins_df = buys_df.join(mean_scores)

        # Filter based on mean score
        buy_coins_df = buy_coins_df[buy_coins_df['mean_score'] >= min_average_score]

        # Limit buys to max number of coins
        buy_coins = (buy_coins_df.sort_values(by='mean_score', ascending=False)
                     .head(max_coins_per_epoch).index.values)

        logger.info(f"Identified {len(buy_coins)} coins to buy.")

        return buy_coins
