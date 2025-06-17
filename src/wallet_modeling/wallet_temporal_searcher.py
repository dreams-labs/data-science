"""
Multi-temporal grid search orchestrator for assessing feature stability across time periods.
"""
import logging
import copy
import gc
from datetime import datetime
from typing import List, Dict
from pathlib import Path
import pandas as pd

# Local modules
import wallet_modeling.wallet_epochs_orchestrator as weo
import wallet_modeling.wallets_config_manager as wcm
import wallet_modeling.wallet_model as wm
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)


class TemporalGridSearcher:
    """
    Orchestrates grid search experiments across multiple time periods to assess
    feature stability and consistency of model parameters.
    """

    def __init__(
        self,
        base_wallets_config: dict,
        wallets_metrics_config: dict,
        wallets_features_config: dict,
        wallets_epochs_config: dict,
        modeling_dates: List[str],
    ):
        """
        Initialize the multi-temporal grid search orchestrator.

        Params:
        - base_wallets_config: Base wallet configuration to be modified for each date
        - wallets_metrics_config: Metrics configuration
        - wallets_features_config: Features configuration
        - wallets_epochs_config: Epochs configuration
        - modeling_dates: List of modeling period start dates (YYYY-MM-DD format)
        """
        self.base_wallets_config = copy.deepcopy(base_wallets_config)
        self.wallets_metrics_config = wallets_metrics_config
        self.wallets_features_config = wallets_features_config
        self.wallets_epochs_config = wallets_epochs_config
        self.modeling_dates = modeling_dates

        # If True, regenerate all training data regardless of cache
        self.force_regenerate_data = base_wallets_config['training_data']['rebuild_multiwindow_dfs']

        # Storage for results
        self.training_data_cache = {}
        self.grid_search_results = {}
        self.consolidated_results = None


    @u.timing_decorator(logging.MILESTONE)
    def generate_all_training_data(self) -> None:
        """
        Generate training data for all specified modeling dates.
        Caches results for subsequent grid search experiments.
        """
        logger.milestone(f"Generating training data for {len(self.modeling_dates)} time periods...")
        u.notify('robotz_windows_exit')

        for i, modeling_date in enumerate(self.modeling_dates, 1):
            date_str = datetime.strptime(modeling_date, '%Y-%m-%d').strftime('%y%m%d')

            # Check if data already exists and skip if not forcing regeneration
            if not self.force_regenerate_data and self._check_data_exists(modeling_date):
                logger.info(f"({i}/{len(self.modeling_dates)}) Skipping {modeling_date} - data already exists")
                continue

            logger.info(f"({i}/{len(self.modeling_dates)}) Generating data for {modeling_date}...")

            # Create date-specific config
            date_config = self._create_date_config(modeling_date)

            # Initialize orchestrator for this date
            epochs_orchestrator = weo.WalletEpochsOrchestrator(
                date_config,
                self.wallets_metrics_config,
                self.wallets_features_config,
                self.wallets_epochs_config,
            )

            # Load complete datasets
            epochs_orchestrator.load_complete_raw_datasets()

            # Generate training data
            epochs_orchestrator.generate_epochs_training_data()

            # Clear memory
            del epochs_orchestrator
            gc.collect()

        logger.milestone("Completed training data generation for all periods")


    @u.timing_decorator(logging.MILESTONE)
    def load_all_training_data(self) -> None:
        """
        Load pre-generated training data for all modeling dates into memory cache.
        """
        logger.milestone("Loading training data for all time periods...")

        parquet_folder = self.base_wallets_config['training_data']['parquet_folder']

        for modeling_date in self.modeling_dates:
            date_str = datetime.strptime(modeling_date, '%Y-%m-%d').strftime('%y%m%d')

            try:
                # Load all four DataFrames for this date
                base_path = f"{parquet_folder}/{date_str}"
                wallet_training_data_df = pd.read_parquet(f"{base_path}/multiwindow_wallet_training_data_df.parquet")
                wallet_target_vars_df = pd.read_parquet(f"{base_path}/multiwindow_wallet_target_vars_df.parquet")
                validation_training_data_df = pd.read_parquet(f"{base_path}/multiwindow_validation_training_data_df.parquet")
                validation_target_vars_df = pd.read_parquet(f"{base_path}/multiwindow_validation_target_vars_df.parquet")

                # Cache the data
                self.training_data_cache[date_str] = (
                    wallet_training_data_df,
                    wallet_target_vars_df,
                    validation_training_data_df,
                    validation_target_vars_df
                )

                logger.info(f"Loaded training data for {modeling_date} (shapes: "
                           f"{wallet_training_data_df.shape}, {wallet_target_vars_df.shape}, "
                           f"{validation_training_data_df.shape}, {validation_target_vars_df.shape})")

            except FileNotFoundError as e:
                logger.error(f"Training data not found for {modeling_date}: {e}")
                raise FileNotFoundError(
                    f"Training data missing for {modeling_date}. "
                    f"Run generate_all_training_data() first or set force_regenerate_data=True"
                )

        logger.milestone(f"Successfully loaded training data for {len(self.modeling_dates)} periods")


    @u.timing_decorator(logging.MILESTONE)
    def run_multi_temporal_grid_search(self) -> None:
        """
        Execute grid search across all time periods and cache results.
        """
        # Validate grid search is enabled
        if not self.base_wallets_config['modeling']['grid_search_params'].get('enabled',False):
            raise ValueError("Grid search must be enabled in base configuration")

        if not self.training_data_cache:
            raise ValueError("No training data loaded. Call load_all_training_data() first")

        logger.milestone(f"Running grid search across {len(self.modeling_dates)} time periods...")
        u.notify('startup')

        for i, modeling_date in enumerate(self.modeling_dates, 1):
            date_str = datetime.strptime(modeling_date, '%Y-%m-%d').strftime('%y%m%d')

            if date_str not in self.training_data_cache:
                logger.warning(f"Skipping {modeling_date} - no training data cached")
                continue

            logger.info(f"({i}/{len(self.modeling_dates)}) Running grid search for {modeling_date}...")

            # Create date-specific modeling config
            date_config = self._create_date_config(modeling_date)

            # Initialize model with date-specific config
            wallet_model = wm.WalletModel(copy.deepcopy(date_config['modeling']))

            # Run grid search experiment
            training_data = self.training_data_cache[date_str]
            wallet_model_results = wallet_model.construct_wallet_model(*training_data)

            # Generate and cache search report
            if hasattr(wallet_model, 'generate_search_report'):
                report_df = wallet_model.generate_search_report()
                self.grid_search_results[date_str] = report_df
                logger.info(f"Grid search completed for {modeling_date}: "
                           f"{len(report_df)} parameter combinations tested")
            else:
                logger.warning(f"No grid search results available for {modeling_date}")

            # Clear model memory
            del wallet_model, wallet_model_results
            gc.collect()

        logger.milestone("Completed grid search across all time periods")
        u.notify('ui_1')


    def consolidate_results(self) -> pd.DataFrame:
        """
        Consolidate grid search results across all time periods into a summary DataFrame.

        Returns:
        - consolidated_df: DataFrame with parameters as rows, dates as columns, plus summary stats
        """
        if not self.grid_search_results:
            raise ValueError("No grid search results available. Run run_multi_temporal_grid_search() first")

        logger.info("Consolidating grid search results across time periods...")

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

        # Round all date columns to 3 decimal places
        consolidated_df[date_columns] = consolidated_df[date_columns].round(3)
        consolidated_df['mean_score'] = consolidated_df['mean_score'].round(3)
        consolidated_df['median_score'] = consolidated_df['median_score'].round(3)
        consolidated_df['stddev'] = consolidated_df['stddev'].round(3)
        consolidated_df['min_score'] = consolidated_df['min_score'].round(3)
        consolidated_df['max_score'] = consolidated_df['max_score'].round(3)
        consolidated_df['score_range'] = consolidated_df['score_range'].round(3)

        # Count non-null values (periods where parameter was tested)
        consolidated_df['periods_tested'] = consolidated_df[date_columns].notna().sum(axis=1)

        # Sort by parameter and mean score
        consolidated_df = consolidated_df.sort_values(
            by=['param', 'mean_score'],
            ascending=[True, False]
        )

        # Cache results
        self.consolidated_results = consolidated_df

        logger.info(f"Consolidated results: {len(consolidated_df)} unique parameter combinations "
                   f"across {len(date_columns)} time periods")

        return consolidated_df


    def analyze_parameter_stability(self, stability_threshold: float = 0.05) -> pd.DataFrame:
        """
        Analyze parameter stability across time periods.

        Params:
        - stability_threshold: Maximum acceptable coefficient of variation for stable parameters

        Returns:
        - stability_df: DataFrame with stability metrics for each parameter combination
        """
        if self.consolidated_results is None:
            raise ValueError("No consolidated results available. Run consolidate_results() first")

        logger.info("Analyzing parameter stability across time periods...")

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
            (stability_df['periods_tested'] >= len(self.modeling_dates) * 0.8)  # Tested in 80%+ periods
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
        logger.info(f"Found {stable_count} stable parameter combinations out of {len(stability_df)}")

        return stability_df


    def get_best_parameters_by_stability(self, top_n: int = 5) -> Dict[str, any]:
        """
        Get the top N most stable and high-performing parameter combinations.

        Params:
        - top_n: Number of top parameter combinations to return

        Returns:
        - best_params: Dictionary of parameter recommendations
        """
        stability_df = self.analyze_parameter_stability()

        # Get top N most stable and high-performing combinations
        top_combinations = stability_df.head(top_n)

        best_params = {}
        for _, row in top_combinations.iterrows():
            param_name = row['param']
            param_value = row['param_value']

            if param_name not in best_params:
                best_params[param_name] = {
                    'value': param_value,
                    'mean_score': row['mean_score'],
                    'stability': row['coeff_variation'],
                    'periods_tested': row['periods_tested']
                }

        logger.info(f"Identified best parameters across {len(best_params)} categories")
        return best_params


    def save_results(self, output_folder: str) -> None:
        """
        Save all results to the specified folder.

        Params:
        - output_folder: Path to save results
        """
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save consolidated results
        if self.consolidated_results is not None:
            self.consolidated_results.to_csv(
                output_path / "consolidated_grid_search_results.csv",
                index=False
            )
            logger.info(f"Saved consolidated results to {output_path}")

        # Save individual period results
        results_folder = output_path / "individual_periods"
        results_folder.mkdir(exist_ok=True)

        for date_str, report_df in self.grid_search_results.items():
            report_df.to_csv(results_folder / f"grid_search_{date_str}.csv")

        logger.info(f"Saved {len(self.grid_search_results)} individual period results")


    # Helper Methods
    def _create_date_config(self, modeling_date: str) -> dict:
        """Create configuration with specified modeling_period_start date."""
        date_config = copy.deepcopy(self.base_wallets_config)
        date_config['training_data']['modeling_period_start'] = modeling_date
        return wcm.add_derived_values(date_config)


    def _check_data_exists(self, modeling_date: str) -> bool:
        """Check if training data already exists for the specified date."""
        parquet_folder = self.base_wallets_config['training_data']['parquet_folder']
        date_str = datetime.strptime(modeling_date, '%Y-%m-%d').strftime('%y%m%d')

        required_files = [
            f"{parquet_folder}/{date_str}/multiwindow_wallet_training_data_df.parquet",
            f"{parquet_folder}/{date_str}/multiwindow_wallet_target_vars_df.parquet",
            f"{parquet_folder}/{date_str}/multiwindow_validation_training_data_df.parquet",
            f"{parquet_folder}/{date_str}/multiwindow_validation_target_vars_df.parquet"
        ]

        return all(Path(file_path).exists() for file_path in required_files)


    def display_summary(self) -> None:
        """Display a formatted summary of results."""
        if self.consolidated_results is None:
            logger.warning("No results to display. Run consolidate_results() first.")
            return

        # Get stability analysis
        stability_df = self.analyze_parameter_stability()

        # Build summary message as single string
        summary_lines = [
            "",
            "="*60,
            "MULTI-TEMPORAL GRID SEARCH SUMMARY",
            "="*60,
            f"Time periods analyzed: {len(self.modeling_dates)}",
            f"Date range: {min(self.modeling_dates)} to {max(self.modeling_dates)}",
            f"Total parameter combinations: {len(self.consolidated_results)}",
            f"Stable combinations: {stability_df['is_stable'].sum()}",
            "",
            "TOP 5 MOST STABLE PARAMETERS:",
            "-" * 40
        ]

        # Add top 5 most stable parameters
        top_stable = stability_df.head(5)
        for _, row in top_stable.iterrows():
            summary_lines.extend([
                f"{row['param']}: {row['param_value']}",
                f"  Mean Score: {row['mean_score']:.4f}",
                f"  Stability (CV): {row['coeff_variation']:.4f}",
                f"  Periods Tested: {row['periods_tested']}/{len(self.modeling_dates)}",
                ""
            ])

        # Log as single message
        logger.info("\n".join(summary_lines))