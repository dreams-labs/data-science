"""
Multi-temporal grid search orchestrator for coin models to assess feature stability across time periods.
"""
import logging
import copy
import gc
from datetime import datetime, timedelta
from typing import List
from pathlib import Path
import pandas as pd

# Local modules
import coin_modeling.coin_epochs_orchestrator as ceo
import coin_modeling.coin_model as cm
import coin_insights.coin_model_reporting as cimr
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)


class CoinTemporalGridSearcher:
    """
    Orchestrates grid search experiments across multiple time periods for coin models
    to assess feature stability and consistency of model parameters.
    """

    def __init__(
        self,
        # Coin model configs
        wallets_coin_config: dict,
        wallets_coins_metrics_config: dict,

        # Wallet model configs (for underlying data generation)
        wallets_config: dict,
        wallets_metrics_config: dict,
        wallets_features_config: dict,
        wallets_epochs_config: dict,

        # Coin flow model configs (optional)
        coin_flow_config: dict = None,
        coin_flow_modeling_config: dict = None,
        coin_flow_metrics_config: dict = None,

        # Search parameters
        coin_modeling_dates: List[str] = None,
    ):
        """
        Initialize the multi-temporal coin grid search orchestrator.

        Params:
        - wallets_coin_config: Coin modeling configuration
        - wallets_coins_metrics_config: Coin metrics configuration
        - wallets_config: Base wallet configuration
        - wallets_metrics_config: Wallet metrics configuration
        - wallets_features_config: Wallet features configuration
        - wallets_epochs_config: Wallet epochs configuration
        - coin_flow_config: Optional coin flow configuration
        - coin_flow_modeling_config: Optional coin flow modeling configuration
        - coin_flow_metrics_config: Optional coin flow metrics configuration
        - coin_modeling_dates: List of coin modeling period start dates (YYYY-MM-DD format)
        """
        # Store all configurations
        self.wallets_coin_config = copy.deepcopy(wallets_coin_config)
        self.wallets_coins_metrics_config = copy.deepcopy(wallets_coins_metrics_config)
        self.wallets_config = copy.deepcopy(wallets_config)
        self.wallets_metrics_config = copy.deepcopy(wallets_metrics_config)
        self.wallets_features_config = copy.deepcopy(wallets_features_config)
        self.wallets_epochs_config = copy.deepcopy(wallets_epochs_config)
        self.coin_flow_config = copy.deepcopy(coin_flow_config) if coin_flow_config else None
        self.coin_flow_modeling_config = copy.deepcopy(coin_flow_modeling_config) if coin_flow_modeling_config else None
        self.coin_flow_metrics_config = copy.deepcopy(coin_flow_metrics_config) if coin_flow_metrics_config else None

        # Generate coin modeling dates if not provided
        if coin_modeling_dates is None:
            self.coin_modeling_dates = self._generate_default_modeling_dates()
        else:
            self.coin_modeling_dates = coin_modeling_dates

        # Force regeneration flag
        self.force_regenerate_data = wallets_coin_config['training_data'].get('toggle_rebuild_all_features', False)

        # Storage for results
        self.training_data_cache = {}
        self.grid_search_results = {}
        self.consolidated_results = None
        self.model_comparison_results = None

        # Initialize coin epochs orchestrator
        self.coin_epochs_orchestrator = ceo.CoinEpochsOrchestrator(
            self.wallets_coin_config,
            self.wallets_coins_metrics_config,
            self.wallets_config,
            self.wallets_metrics_config,
            self.wallets_features_config,
            self.wallets_epochs_config,
            self.coin_flow_config,
            self.coin_flow_modeling_config,
            self.coin_flow_metrics_config,
        )


    # ---------------------------------
    #      Primary Search Sequence
    # ---------------------------------

    def generate_all_coin_training_data(self) -> None:
        """
        Generate coin training data for all specified modeling dates.
        Caches results for subsequent grid search experiments.
        """
        logger.milestone(f"Generating coin training data for {len(self.coin_modeling_dates)} time periods...")

        # Load complete datasets once
        self.coin_epochs_orchestrator.load_complete_raw_datasets()

        for i, modeling_date in enumerate(self.coin_modeling_dates, 1):
            # Check if data already exists and skip if not forcing regeneration
            if not self.force_regenerate_data and self._check_coin_data_exists(modeling_date):
                logger.info(f"({i}/{len(self.coin_modeling_dates)}) Skipping {modeling_date} - "
                            "coin data already exists")
                continue

            logger.info(f"({i}/{len(self.coin_modeling_dates)}) Generating coin data for {modeling_date}...")

            # Create date-specific config
            date_config = self._create_coin_date_config(modeling_date)

            # Update coin epochs orchestrator with new config
            self._update_orchestrator_config(date_config)
            self.coin_epochs_orchestrator.load_complete_raw_datasets()

            # Generate coin training and validation data
            self._generate_coin_epoch_data(modeling_date)

            # Clear memory
            gc.collect()

        logger.milestone("Completed coin training data generation for all periods")


    def load_all_coin_training_data(self) -> None:
        """
        Load pre-generated coin training data for all modeling dates into memory cache.
        """
        logger.milestone("Loading coin training data for all time periods...")

        for modeling_date in self.coin_modeling_dates:
            date_str = datetime.strptime(modeling_date, '%Y-%m-%d').strftime('%y%m%d')

            try:
                # Load training and validation DataFrames for this date
                base_path = self._get_coin_data_path(modeling_date)

                training_data_df = pd.read_parquet(f"{base_path}/training_multiwindow_coin_training_data_df.parquet")
                training_target_df = pd.read_parquet(f"{base_path}/training_multiwindow_coin_target_var_df.parquet")
                validation_data_df = pd.read_parquet(f"{base_path}/validation_multiwindow_coin_training_data_df.parquet")  # pylint:disable=line-too-long
                validation_target_df = pd.read_parquet(f"{base_path}/validation_multiwindow_coin_target_var_df.parquet")

                # Cache the data
                self.training_data_cache[date_str] = (
                    training_data_df,
                    training_target_df,
                    validation_data_df,
                    validation_target_df
                )

                logger.info(f"Loaded coin training data for {modeling_date} (shapes: "
                           f"{training_data_df.shape}, {training_target_df.shape}, "
                           f"{validation_data_df.shape}, {validation_target_df.shape})")

            except FileNotFoundError as e:
                logger.error(f"Coin training data not found for {modeling_date}: {e}")
                raise FileNotFoundError(
                    f"Coin training data missing for {modeling_date}. "
                    f"Run generate_all_coin_training_data() first or set force_regenerate_data=True"
                ) from e

        logger.milestone(f"Successfully loaded coin training data for {len(self.coin_modeling_dates)} periods")


    @u.timing_decorator(logging.MILESTONE)      #pylint:disable=no-member
    def run_multi_temporal_coin_grid_search(self) -> None:
        """
        Execute grid search across all time periods for coin models and cache results.
        """
        # Validate grid search is enabled
        if not self.wallets_coin_config['coin_modeling']['grid_search_params'].get('enabled', False):
            raise ValueError("Grid search must be enabled in coin modeling configuration")

        if not self.training_data_cache:
            raise ValueError("No training data loaded. Call load_all_coin_training_data() first")

        logger.milestone(f"Running coin grid search across {len(self.coin_modeling_dates)} time periods...")
        u.notify('startup')

        for i, modeling_date in enumerate(self.coin_modeling_dates, 1):
            date_str = datetime.strptime(modeling_date, '%Y-%m-%d').strftime('%y%m%d')

            if date_str not in self.training_data_cache:
                logger.warning(f"Skipping {modeling_date} - no training data cached")
                continue

            logger.milestone(f"({i}/{len(self.coin_modeling_dates)}) Running coin grid search for {modeling_date}...")

            # Create date-specific modeling config
            date_config = self._create_coin_date_config(modeling_date)

            # Initialize coin model with date-specific config
            coin_model = cm.CoinModel(copy.deepcopy(date_config['coin_modeling']))

            # Run grid search experiment
            training_data = self.training_data_cache[date_str]
            coin_model_results = coin_model.construct_coin_model(*training_data)

            # Generate and cache search report
            if hasattr(coin_model, 'generate_search_report'):
                report_df = coin_model.generate_search_report()
                self.grid_search_results[date_str] = report_df
                logger.info(f"Coin grid search completed for {modeling_date}: "
                           f"{len(report_df)} parameter combinations tested")
            else:
                logger.warning(f"No grid search results available for {modeling_date}")

            # Clear model memory
            del coin_model, coin_model_results
            gc.collect()

        logger.milestone("Completed coin grid search across all time periods")
        u.notify('ui_1')


    def consolidate_coin_results(self) -> pd.DataFrame:
        """
        Consolidate coin grid search results across all time periods into a summary DataFrame.

        Returns:
        - consolidated_df: DataFrame with parameters as rows, dates as columns, plus summary stats
        """
        if not self.grid_search_results:
            raise ValueError("No grid search results available. Run run_multi_temporal_coin_grid_search() first")

        logger.info("Consolidating coin grid search results across time periods...")

        # Stack all report DataFrames with date identifiers
        consolidated_data = []
        for date_str, report_df in self.grid_search_results.items():
            temp_df = report_df.reset_index()
            temp_df['date'] = date_str
            consolidated_data.append(temp_df)

        if not consolidated_data:
            raise ValueError("No valid grid search results to consolidate")

        # Combine all results
        stacked_df = pd.concat(consolidated_data, ignore_index=True)

        # Pivot to get dates as columns
        consolidated_df = stacked_df.pivot_table(
            index=['param', 'param_value'],
            columns='date',
            values='avg_score'
        ).reset_index()

        # Calculate summary statistics across time periods
        date_columns = [col for col in consolidated_df.columns
                       if col not in ['param', 'param_value']]

        consolidated_df['mean_score'] = consolidated_df[date_columns].mean(axis=1)
        consolidated_df['median_score'] = consolidated_df[date_columns].median(axis=1)
        consolidated_df['std_dev'] = consolidated_df[date_columns].std(axis=1)
        consolidated_df['min_score'] = consolidated_df[date_columns].min(axis=1)
        consolidated_df['max_score'] = consolidated_df[date_columns].max(axis=1)
        consolidated_df['score_range'] = consolidated_df['max_score'] - consolidated_df['min_score']

        # Round all score columns to 3 decimal places
        score_columns = date_columns + ['mean_score', 'median_score', 'std_dev',
                                        'min_score', 'max_score', 'score_range']
        consolidated_df[score_columns] = consolidated_df[score_columns].round(3)

        # Count non-null values (periods where parameter was tested)
        consolidated_df['periods_tested'] = consolidated_df[date_columns].notna().sum(axis=1)

        # Sort by parameter and mean score
        consolidated_df = consolidated_df.sort_values(
            by=['param', 'median_score'],
            ascending=[True, False]
        )

        # Cache results
        self.consolidated_results = consolidated_df

        logger.info(f"Consolidated coin results: {len(consolidated_df)} unique parameter combinations "
                   f"across {len(date_columns)} time periods")

        return consolidated_df


    def analyze_coin_parameter_stability(self, stability_threshold: float = 0.05) -> pd.DataFrame:
        """
        Analyze coin parameter stability across time periods.

        Params:
        - stability_threshold: Maximum acceptable coefficient of variation for stable parameters

        Returns:
        - stability_df: DataFrame with stability metrics for each parameter combination
        """
        if self.consolidated_results is None:
            raise ValueError("No consolidated results available. Run consolidate_coin_results() first")

        logger.info("Analyzing coin parameter stability across time periods...")

        # Calculate coefficient of variation (std/mean) for each parameter
        stability_df = self.consolidated_results.copy()

        # Coefficient of variation (only for positive means to avoid division issues)
        mask_positive_mean = stability_df['mean_score'] > 0
        stability_df.loc[mask_positive_mean, 'coeff_variation'] = (
            stability_df.loc[mask_positive_mean, 'std_dev'] /
            stability_df.loc[mask_positive_mean, 'mean_score']
        )

        # Mark stable parameters
        stability_df['is_stable'] = (
            (stability_df['coeff_variation'] <= stability_threshold) &
            (stability_df['periods_tested'] >= len(self.coin_modeling_dates) * 0.8)  # Tested in 80%+ periods
        )

        # Rank by stability and performance
        stability_df['stability_rank'] = stability_df['coeff_variation'].rank(ascending=True)
        stability_df['performance_rank'] = stability_df['mean_score'].rank(ascending=False)
        stability_df['combined_rank'] = (
            stability_df['stability_rank'] * 0.3 +
            stability_df['performance_rank'] * 0.7
        )

        # Sort by combined ranking
        stability_df = stability_df.sort_values('combined_rank')

        stable_count = stability_df['is_stable'].sum()
        logger.info(f"Found {stable_count} stable coin parameter combinations out of {len(stability_df)}")

        return stability_df.sort_values(by='median_score', ascending=False)


    def run_multi_temporal_coin_model_comparison(self) -> pd.DataFrame:
        """
        Build coin models for each time period using base parameters (no grid search)
        and consolidate their performance metrics for comparison.

        Returns:
        - comparison_df: DataFrame with performance metrics across all time periods
        """
        if not self.training_data_cache:
            raise ValueError("No training data loaded. Call load_all_coin_training_data() first")

        logger.milestone(f"Building and evaluating coin models across {len(self.coin_modeling_dates)} time periods...")
        u.notify('startup')

        performance_results = {}

        for i, modeling_date in enumerate(self.coin_modeling_dates, 1):
            date_str = datetime.strptime(modeling_date, '%Y-%m-%d').strftime('%y%m%d')

            if date_str not in self.training_data_cache:
                logger.warning(f"Skipping {modeling_date} - no training data cached")
                continue

            logger.milestone(f"({i}/{len(self.coin_modeling_dates)}) Building coin model for {modeling_date}...")

            # Create date-specific modeling config with grid search disabled
            date_config = self._create_coin_date_config(modeling_date)
            date_config['coin_modeling']['grid_search_params']['enabled'] = False

            # Initialize coin model with date-specific config
            coin_model = cm.CoinModel(copy.deepcopy(date_config['coin_modeling']))

            # Build model without grid search
            training_data = self.training_data_cache[date_str]
            coin_model_results = coin_model.construct_coin_model(*training_data)

            # Generate evaluator and extract metrics
            coin_model_id, coin_evaluator, _ = cimr.generate_and_save_coin_model_artifacts(
                model_results=coin_model_results,
                base_path=self.wallets_coin_config['training_data']['model_artifacts_folder'],
                configs={
                    'wallets_coin_config': self.wallets_coin_config,
                    'wallets_config': self.wallets_config,
                }
            )

            # Store performance metrics for this period
            performance_results[date_str] = {
                'modeling_date': modeling_date,
                'model_type': coin_model_results['model_type'],
                'model_id': coin_model_id,
                **self._extract_coin_performance_metrics(coin_evaluator)
            }

            coin_evaluator.summary_report()
            coin_evaluator.plot_wallet_evaluation()

            logger.info(f"Coin model completed for {modeling_date}")

            # Clear model memory
            del coin_model, coin_model_results, coin_evaluator
            gc.collect()

        # Consolidate results into DataFrame
        comparison_df = pd.DataFrame.from_dict(performance_results, orient='index')
        comparison_df = comparison_df.reset_index().rename(columns={'index': 'date_str'})

        # Add summary statistics across time periods
        self._add_coin_performance_summary_stats(comparison_df)

        # Cache results
        self.model_comparison_results = comparison_df

        logger.milestone("Completed coin model comparison across all time periods")
        u.notify('ui_1')

        return comparison_df


    # ----------------------------------
    #           Helper Methods
    # ----------------------------------

    def _generate_default_modeling_dates(self) -> List[str]:
        """
        Generate default coin modeling dates based on coin epochs configuration.
        """
        base_date = pd.to_datetime(self.wallets_config['training_data']['coin_modeling_period_start'])
        coin_epochs = self.wallets_coin_config['training_data']['coin_epochs_training']

        modeling_dates = []
        for epoch_offset in coin_epochs:
            modeling_date = (base_date + timedelta(days=epoch_offset)).strftime('%Y-%m-%d')
            modeling_dates.append(modeling_date)

        return sorted(modeling_dates)


    def _create_coin_date_config(self, modeling_date: str) -> dict:
        """Create coin configuration with specified coin_modeling_period_start date."""
        date_config = copy.deepcopy(self.wallets_coin_config)

        # Calculate offset from base modeling date
        base_modeling_date = pd.to_datetime(self.wallets_config['training_data']['modeling_period_start'])
        target_modeling_date = pd.to_datetime(modeling_date)
        offset_days = (target_modeling_date - base_modeling_date).days

        # Update coin modeling period dates
        date_config['training_data']['coin_modeling_period_start'] = modeling_date
        modeling_duration = self.wallets_config['training_data']['modeling_period_duration']
        modeling_end = (target_modeling_date + timedelta(days=modeling_duration - 1)).strftime('%Y-%m-%d')
        date_config['training_data']['coin_modeling_period_end'] = modeling_end

        # CRITICAL: Also update the underlying wallets_config dates that the coin orchestrator uses
        # This ensures wallet training data is generated for the correct time periods
        updated_wallets_config = copy.deepcopy(self.wallets_config)

        # Offset all relevant dates in wallets_config
        for date_field in ['modeling_period_start', 'modeling_period_end',
                           'coin_modeling_period_start', 'coin_modeling_period_end']:
            if date_field in updated_wallets_config['training_data']:
                original_date = pd.to_datetime(updated_wallets_config['training_data'][date_field])
                new_date = (original_date + timedelta(days=offset_days)).strftime('%Y-%m-%d')
                updated_wallets_config['training_data'][date_field] = new_date

        # Update training window starts
        if 'training_window_starts' in updated_wallets_config['training_data']:
            original_windows = [pd.to_datetime(d) for d in
                                updated_wallets_config['training_data']['training_window_starts']]
            new_windows = [(d + timedelta(days=offset_days)).strftime('%Y-%m-%d') for d in original_windows]
            updated_wallets_config['training_data']['training_window_starts'] = new_windows

        # Store the updated wallets config in the coin config for the orchestrator to use
        date_config['_updated_wallets_config'] = updated_wallets_config

        return date_config


    def _update_orchestrator_config(self, date_config: dict) -> None:
        """Update the coin epochs orchestrator with new configuration."""
        self.coin_epochs_orchestrator.wallets_coin_config = date_config

        # CRITICAL: Also update the underlying wallets_config if it was modified
        if '_updated_wallets_config' in date_config:
            self.coin_epochs_orchestrator.wallets_config = date_config['_updated_wallets_config']


    def _generate_coin_epoch_data(self, modeling_date: str) -> None:
        """Generate coin training and validation data for a specific modeling date."""
        date_str = datetime.strptime(modeling_date, '%Y-%m-%d').strftime('%y%m%d')

        # Generate training epochs
        training_epochs = self.wallets_coin_config['training_data']['coin_epochs_training']
        self.coin_epochs_orchestrator.orchestrate_coin_epochs(
            training_epochs,
            file_prefix=f'{date_str}/training_'
        )

        # Generate validation epochs
        validation_epochs = self.wallets_coin_config['training_data'].get('coin_epochs_validation', [])
        if validation_epochs:
            self.coin_epochs_orchestrator.orchestrate_coin_epochs(
                validation_epochs,
                file_prefix=f'{date_str}/validation_'
            )


    def _check_coin_data_exists(self, modeling_date: str) -> bool:
        """Check if coin training data already exists for the specified date."""
        base_path = self._get_coin_data_path(modeling_date)

        required_files = [
            f"{base_path}/training_multiwindow_coin_training_data_df.parquet",
            f"{base_path}/training_multiwindow_coin_target_var_df.parquet",
            f"{base_path}/validation_multiwindow_coin_training_data_df.parquet",
            f"{base_path}/validation_multiwindow_coin_target_var_df.parquet"
        ]

        return all(Path(file_path).exists() for file_path in required_files)


    def _get_coin_data_path(self, modeling_date: str) -> str:
        """Get the data path for a specific modeling date."""
        date_str = datetime.strptime(modeling_date, '%Y-%m-%d').strftime('%y%m%d')
        parquet_folder = self.wallets_coin_config['training_data']['parquet_folder']
        return f"{parquet_folder}/{date_str}"


    def _extract_coin_performance_metrics(self, evaluator) -> dict:
        """
        Extract key performance metrics from coin model evaluator.

        Params:
        - evaluator: Coin model evaluator instance

        Returns:
        - metrics_dict: Dictionary of performance metrics
        """
        metrics = {}

        # Common metrics for both model types
        if hasattr(evaluator, 'metrics'):
            # Sample sizes
            if 'test_samples' in evaluator.metrics:
                metrics['test_samples'] = evaluator.metrics['test_samples']
            if 'train_samples' in evaluator.metrics:
                metrics['train_samples'] = evaluator.metrics['train_samples']

            # Feature importance summary
            if 'importances' in evaluator.metrics and hasattr(evaluator, 'feature_names'):
                metrics['n_features'] = len(evaluator.feature_names)
                # Top 3 most important features
                if len(evaluator.metrics['importances']['feature']) >= 3:
                    top_features = evaluator.metrics['importances']['feature'][:3]
                    top_importances = evaluator.metrics['importances']['importance'][:3]
                    for i, (feat, imp) in enumerate(zip(top_features, top_importances)):
                        metrics[f'top_feature_{i+1}'] = feat
                        metrics[f'top_importance_{i+1}'] = round(imp, 4)

        # Model-type specific metrics
        if evaluator.modeling_config.get('model_type') == 'classification':
            metrics.update(self._extract_coin_classification_metrics(evaluator))
        elif evaluator.modeling_config.get('model_type') == 'regression':
            metrics.update(self._extract_coin_regression_metrics(evaluator))

        return metrics


    def _extract_coin_classification_metrics(self, evaluator) -> dict:
        """Extract classification-specific metrics for coin models."""
        metrics = {}

        # Test set metrics (if available)
        if hasattr(evaluator, 'train_test_data_provided') and evaluator.train_test_data_provided:
            metrics.update({
                'test_accuracy': round(evaluator.metrics.get('accuracy', 0), 4),
                'test_precision': round(evaluator.metrics.get('precision', 0), 4),
                'test_recall': round(evaluator.metrics.get('recall', 0), 4),
                'test_f1': round(evaluator.metrics.get('f1', 0), 4),
                'test_roc_auc': round(evaluator.metrics.get('roc_auc', 0), 4),
            })

        # Validation set metrics (if available)
        if hasattr(evaluator, 'validation_data_provided') and evaluator.validation_data_provided:
            metrics.update({
                'val_accuracy': round(evaluator.metrics.get('val_accuracy', 0), 4),
                'val_precision': round(evaluator.metrics.get('val_precision', 0), 4),
                'val_recall': round(evaluator.metrics.get('val_recall', 0), 4),
                'val_f1': round(evaluator.metrics.get('val_f1', 0), 4),
                'val_roc_auc': round(evaluator.metrics.get('val_roc_auc', 0), 4),
            })

            # Threshold information
            if 'y_pred_threshold' in evaluator.metrics:
                metrics['y_pred_threshold'] = round(evaluator.metrics['y_pred_threshold'], 4)

        return metrics


    def _extract_coin_regression_metrics(self, evaluator) -> dict:
        """Extract regression-specific metrics for coin models."""
        metrics = {}

        # Test set metrics (if available)
        if hasattr(evaluator, 'train_test_data_provided') and evaluator.train_test_data_provided:
            metrics.update({
                'test_r2': round(evaluator.metrics.get('r2', 0), 4),
                'test_rmse': round(evaluator.metrics.get('rmse', 0), 4),
                'test_mae': round(evaluator.metrics.get('mae', 0), 4),
            })

        # Validation set metrics (if available)
        if 'validation_metrics' in evaluator.metrics:
            vm = evaluator.metrics['validation_metrics']
            metrics.update({
                'val_r2': round(vm.get('r2', 0), 4),
                'val_rmse': round(vm.get('rmse', 0), 4),
                'val_mae': round(vm.get('mae', 0), 4),
                'val_spearman': round(vm.get('spearman', 0), 4),
            })

        return metrics


    def _add_coin_performance_summary_stats(self, comparison_df: pd.DataFrame) -> None:
        """
        Add summary statistics across time periods to the coin comparison DataFrame.

        Params:
        - comparison_df: DataFrame with performance metrics for each time period
        """
        # Identify numeric metric columns (exclude metadata columns)
        metadata_cols = ['date_str', 'modeling_date', 'model_type', 'model_id', 'n_features'] + \
                    [col for col in comparison_df.columns if col.startswith('top_feature_')]
        numeric_cols = [col for col in comparison_df.columns if col not in metadata_cols]

        # Calculate summary statistics for numeric columns
        summary_stats = {}
        for col in numeric_cols:
            if comparison_df[col].dtype in ['float64', 'int64'] and not comparison_df[col].isna().all():
                values = comparison_df[col].dropna()
                if len(values) > 0:
                    summary_stats[f'{col}_mean'] = round(values.mean(), 4)
                    summary_stats[f'{col}_std'] = round(values.std(), 4)
                    summary_stats[f'{col}_min'] = round(values.min(), 4)
                    summary_stats[f'{col}_max'] = round(values.max(), 4)
                    # Coefficient of variation for stability assessment
                    if values.mean() != 0:
                        summary_stats[f'{col}_cv'] = round(values.std() / abs(values.mean()), 4)

        # Add summary row
        summary_row = {'date_str': 'SUMMARY', 'modeling_date': 'ACROSS_ALL_PERIODS',
                    'model_type': comparison_df['model_type'].iloc[0], **summary_stats}

        # Append summary row to DataFrame
        comparison_df.loc[len(comparison_df)] = summary_row


    def save_coin_results(self, output_folder: str) -> None:
        """
        Save all coin results to the specified folder.

        Params:
        - output_folder: Path to save results
        """
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save consolidated results
        if self.consolidated_results is not None:
            self.consolidated_results.to_csv(
                output_path / "consolidated_coin_grid_search_results.csv",
                index=False
            )
            logger.info(f"Saved consolidated coin results to {output_path}")

        # Save model comparison results
        if self.model_comparison_results is not None:
            self.model_comparison_results.to_csv(
                output_path / "coin_model_comparison_results.csv",
                index=False
            )
            logger.info(f"Saved coin model comparison results to {output_path}")

        # Save individual period results
        results_folder = output_path / "individual_periods"
        results_folder.mkdir(exist_ok=True)

        for date_str, report_df in self.grid_search_results.items():
            report_df.to_csv(results_folder / f"coin_grid_search_{date_str}.csv")

        logger.info(f"Saved {len(self.grid_search_results)} individual coin period results")
