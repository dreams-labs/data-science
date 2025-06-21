"""
Multi-temporal grid search orchestrator for assessing feature stability across time periods.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
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
import base_modeling.pipeline as bp
import wallet_insights.model_evaluation as wime
import wallet_insights.wallet_model_reporting as wimr
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
        wallets_investing_config: dict,
        wallets_metrics_config: dict,
        wallets_features_config: dict,
        wallets_epochs_config: dict,
        modeling_dates: List[str],
        complete_hybrid_cw_id_df: pd.DataFrame,
        complete_market_data_df: pd.DataFrame,
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
        self.wallets_investing_config = wallets_investing_config
        self.wallets_metrics_config = wallets_metrics_config
        self.wallets_features_config = wallets_features_config
        self.wallets_epochs_config = wallets_epochs_config
        self.modeling_dates = modeling_dates

        # If True, regenerate all training data regardless of cache
        self.force_regenerate_data = wallets_investing_config['training_data']['toggle_overwrite_multioffset_parquet']

        # Storage for results
        self.training_data_cache = {}
        self.grid_search_results = {}
        self.consolidated_results = None
        self.model_comparison_results = None

        # Complete dfs for analysis
        self.complete_hybrid_cw_id_df = complete_hybrid_cw_id_df
        self.complete_market_data_df = complete_market_data_df



    # ---------------------------------
    #      Primary Search Sequence
    # ---------------------------------

    @u.timing_decorator(logging.MILESTONE)  # pylint: disable=no-member
    def generate_all_training_data(self) -> None:
        """
        Generate training data for all specified modeling dates using multithreading.
        Caches results for subsequent grid search experiments.
        """
        logger.milestone(f"Generating training data for {len(self.modeling_dates)} time periods...")
        u.notify('robotz_windows_exit')

        max_workers = self.wallets_investing_config['n_threads']['training_data_loading']

        def generate_single_date(modeling_date: str) -> tuple:
            """Generate training data for a single modeling date."""
            try:
                # Check if data already exists and skip if not forcing regeneration
                if not self.force_regenerate_data and self._check_data_exists(modeling_date):
                    return modeling_date, "skipped", None

                # Warn if forcing regeneration of existing data
                if self.force_regenerate_data and self._check_data_exists(modeling_date):
                    logger.warning(f"Force regenerating existing data for {modeling_date}")

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

                return modeling_date, "generated", None

            except Exception as e:
                return modeling_date, "failed", e

        # Execute parallel generation
        failed_dates = []
        skipped_count = 0
        generated_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_date = {
                executor.submit(generate_single_date, modeling_date): modeling_date
                for modeling_date in self.modeling_dates
            }

            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_date), 1):
                modeling_date = future_to_date[future]

                try:
                    date, status, error = future.result()

                    if error is not None:
                        failed_dates.append((modeling_date, error))
                        continue

                    if status == "skipped":
                        skipped_count += 1
                        logger.info(f"[{i}/{len(self.modeling_dates)}] Skipping {modeling_date} - data already exists")
                    elif status == "generated":
                        generated_count += 1
                        logger.info(f"[{i}/{len(self.modeling_dates)}] Generated data for {modeling_date}")

                except Exception as e:
                    failed_dates.append((modeling_date, e))

        # Handle any failures
        if failed_dates:
            error_details = "\n".join([f"  {date}: {error}" for date, error in failed_dates])
            logger.error(f"Failed to generate training data for {len(failed_dates)} dates:\n{error_details}")

            # Raise error for the first failure
            first_failed_date, first_error = failed_dates[0]
            raise RuntimeError(
                f"Training data generation failed for {first_failed_date}. "
                f"Check logs for details."
            ) from first_error

        logger.milestone(f"Training data generation completed using {max_workers} threads. "
                        f"Generated: {generated_count}, Skipped: {skipped_count}, Failed: {len(failed_dates)}")



    @u.timing_decorator(logging.MILESTONE)  # pylint: disable=no-member
    def load_all_training_data(self) -> None:
        """
        Load pre-generated training data for all modeling dates into memory cache using multithreading.

        Params:
        - max_workers: Maximum number of threads for parallel loading (default: 4)
        """

        logger.milestone("Loading training data for all time periods...")

        parquet_folder = self.base_wallets_config['training_data']['parquet_folder']
        max_workers = self.wallets_investing_config['n_threads']['training_data_loading']

        def load_single_date(modeling_date: str) -> tuple:
            """Load training data for a single modeling date."""
            date_str = datetime.strptime(modeling_date, '%Y-%m-%d').strftime('%y%m%d')

            try:
                # Load all four DataFrames for this date   # pylint:disable=line-too-long
                base_path = f"{parquet_folder}/{date_str}"
                wallet_training_data_df = pd.read_parquet(f"{base_path}/multioffset_wallet_training_data_df.parquet")
                wallet_target_vars_df = pd.read_parquet(f"{base_path}/multioffset_wallet_target_vars_df.parquet")
                validation_training_data_df = pd.read_parquet(f"{base_path}/multioffset_validation_training_data_df.parquet")
                validation_target_vars_df = pd.read_parquet(f"{base_path}/multioffset_validation_target_vars_df.parquet")

                wallet_target_vars_df['cw_coin_return_rank'] = pd.Series(
                    wallet_target_vars_df['cw_coin_return'],
                    index=wallet_target_vars_df.index
                ).rank(method='average', pct=True)
                validation_target_vars_df['cw_coin_return_rank'] = pd.Series(
                    validation_target_vars_df['cw_coin_return'],
                    index=validation_target_vars_df.index
                ).rank(method='average', pct=True)

                # Apply predrop_features logic if configured
                date_config = self._create_date_config(modeling_date)
                if date_config['training_data']['predrop_features']:
                    drop_patterns = date_config['modeling']['feature_selection']['drop_patterns']
                    col_dropper = bp.DropColumnPatterns(drop_patterns)
                    wallet_training_data_df = col_dropper.fit_transform(wallet_training_data_df)
                    validation_training_data_df = col_dropper.fit_transform(validation_training_data_df)

                training_data = (
                    wallet_training_data_df,
                    wallet_target_vars_df,
                    validation_training_data_df,
                    validation_target_vars_df
                )

                shapes_info = (
                    wallet_training_data_df.shape,
                    wallet_target_vars_df.shape,
                    validation_training_data_df.shape,
                    validation_target_vars_df.shape
                )

                return date_str, training_data, shapes_info, None

            except FileNotFoundError as e:
                return date_str, None, None, e

        # Execute parallel loading
        failed_dates = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_date = {
                executor.submit(load_single_date, modeling_date): modeling_date
                for modeling_date in self.modeling_dates
            }

            # Collect results as they complete
            for future in as_completed(future_to_date):
                modeling_date = future_to_date[future]

                try:
                    date_str, training_data, shapes_info, error = future.result()

                    if error is not None:
                        failed_dates.append((modeling_date, error))
                        continue

                    # Cache the successfully loaded data
                    self.training_data_cache[date_str] = training_data

                    logger.info(f"Loaded training data for {modeling_date} (shapes: "
                            f"{shapes_info[0]}, {shapes_info[1]}, "
                            f"{shapes_info[2]}, {shapes_info[3]})")

                except Exception as e:
                    failed_dates.append((modeling_date, e))

        # Handle any failures
        if failed_dates:
            error_details = "\n".join([f"  {date}: {error}" for date, error in failed_dates])
            logger.error(f"Failed to load training data for {len(failed_dates)} dates:\n{error_details}")

            # Raise error for the first failure
            first_failed_date, first_error = failed_dates[0]
            raise FileNotFoundError(
                f"Training data missing for {first_failed_date}. "
                f"Run generate_all_training_data() first or set force_regenerate_data=True"
            ) from first_error

        logger.milestone(f"Successfully loaded training data for {len(self.modeling_dates)} "
                         "periods using {max_workers} threads")

    @u.timing_decorator(logging.MILESTONE)  # pylint: disable=no-member
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

        for i, modeling_date in enumerate(self.modeling_dates, 1):
            date_str = datetime.strptime(modeling_date, '%Y-%m-%d').strftime('%y%m%d')

            if date_str not in self.training_data_cache:
                logger.warning(f"Skipping {modeling_date} - no training data cached")
                continue

            logger.milestone(f"[{i}/{len(self.modeling_dates)}] Running grid search for {modeling_date}...")

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




    # ---------------------------------
    #         Reporting Methods
    # ---------------------------------

    def consolidate_results(self) -> pd.DataFrame:
        """
        Consolidate grid search results across all time periods into a summary DataFrame.
        Includes period-specific baseline comparisons.

        Returns:
        - consolidated_df: DataFrame with parameters as rows, dates as columns, plus baseline comparisons
        """
        if not self.grid_search_results:
            raise ValueError("No grid search results available. Run run_multi_temporal_grid_search() first")

        logger.info("Consolidating grid search results across time periods...")

        # Calculate period-specific baselines first
        period_baselines = {}
        for date_str, sample_data in self.training_data_cache.items():
            period_baselines[date_str] = self._calculate_sample_baseline(sample_data)

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

        # Get date columns for processing
        date_columns = [col for col in consolidated_df.columns
                    if col not in ['param', 'param_value']]

        # Add baseline comparison columns for each date
        baseline_columns = []
        for date_col in date_columns:
            baseline_col = f"{date_col}_v_baseline"
            baseline_columns.append(baseline_col)

            # Calculate difference from period-specific baseline
            baseline_value = period_baselines.get(date_col, 0.0)
            consolidated_df[baseline_col] = consolidated_df[date_col] - baseline_value

            # Round the baseline comparison columns
            consolidated_df[baseline_col] = consolidated_df[baseline_col].round(3)

        # Calculate summary statistics across time periods (original scores only)
        consolidated_df['mean_score'] = consolidated_df[date_columns].mean(axis=1)
        consolidated_df['median_score'] = consolidated_df[date_columns].median(axis=1)
        consolidated_df['std_dev'] = consolidated_df[date_columns].std(axis=1)
        consolidated_df['min_score'] = consolidated_df[date_columns].min(axis=1)
        consolidated_df['max_score'] = consolidated_df[date_columns].max(axis=1)
        consolidated_df['score_range'] = consolidated_df['max_score'] - consolidated_df['min_score']

        # Calculate summary statistics for baseline comparisons
        consolidated_df['mean_vs_baseline'] = consolidated_df[baseline_columns].mean(axis=1)
        consolidated_df['median_vs_baseline'] = consolidated_df[baseline_columns].median(axis=1)
        consolidated_df['baseline_improvement_consistency'] = (consolidated_df[baseline_columns] > 0).sum(axis=1)

        # Round all columns to 3 decimal places
        consolidated_df[date_columns] = consolidated_df[date_columns].round(3)
        consolidated_df[baseline_columns] = consolidated_df[baseline_columns].round(3)
        consolidated_df['mean_score'] = consolidated_df['mean_score'].round(3)
        consolidated_df['median_score'] = consolidated_df['median_score'].round(3)
        consolidated_df['std_dev'] = consolidated_df['std_dev'].round(3)
        consolidated_df['min_score'] = consolidated_df['min_score'].round(3)
        consolidated_df['max_score'] = consolidated_df['max_score'].round(3)
        consolidated_df['score_range'] = consolidated_df['score_range'].round(3)
        consolidated_df['mean_vs_baseline'] = consolidated_df['mean_vs_baseline'].round(3)
        consolidated_df['median_vs_baseline'] = consolidated_df['median_vs_baseline'].round(3)

        # Count non-null values (periods where parameter was tested)
        consolidated_df['periods_tested'] = consolidated_df[date_columns].notna().sum(axis=1)

        # Reorder columns for better readability: interleave dates with their baselines
        ordered_columns = ['param', 'param_value']
        for date_col in sorted(date_columns):
            ordered_columns.extend([date_col, f"{date_col}_v_baseline"])

        # Add summary columns at the end
        summary_columns = [
            'mean_score', 'median_score', 'std_dev', 'min_score', 'max_score', 'score_range',
            'mean_vs_baseline', 'median_vs_baseline', 'baseline_improvement_consistency', 'periods_tested'
        ]
        ordered_columns.extend(summary_columns)

        # Reorder DataFrame
        available_columns = [col for col in ordered_columns if col in consolidated_df.columns]
        consolidated_df = consolidated_df[available_columns]

        # Sort by parameter and mean score
        consolidated_df = consolidated_df.sort_values(
            by=['param', 'median_score'],
            ascending=[True, False]
        )

        # Cache results
        self.consolidated_results = consolidated_df

        # Log baseline information
        baseline_summary = {date: round(baseline, 3) for date, baseline in period_baselines.items()}
        logger.info(f"Period-specific baselines: {baseline_summary}")
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

        return stability_df.sort_values(by='median_score',ascending=False)


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
            f"{parquet_folder}/{date_str}/multioffset_wallet_training_data_df.parquet",
            f"{parquet_folder}/{date_str}/multioffset_wallet_target_vars_df.parquet",
            f"{parquet_folder}/{date_str}/multioffset_validation_training_data_df.parquet",
            f"{parquet_folder}/{date_str}/multioffset_validation_target_vars_df.parquet"
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


    def _calculate_sample_baseline(self, training_data_tuple) -> float:
        """Calculate population baseline from sample training data."""
        _, _, _, validation_target_vars_df = training_data_tuple

        target_var = self.base_wallets_config['modeling']['target_variable']
        if target_var in validation_target_vars_df.columns:
            returns = validation_target_vars_df[target_var]
            return u.winsorize(returns, 0.001).mean()

        return 0.0





    # ----------------------------------
    #    Non-Search Modeling Sequence
    # ----------------------------------

    @u.timing_decorator(logging.MILESTONE)  # pylint: disable=no-member
    def run_multi_temporal_model_comparison(
        self,
        score_filter: float = 0.2,
        min_scores: int = 10,
    ) -> pd.DataFrame:
        """
        Build models for each time period using base parameters (no grid search)
        and consolidate their performance metrics for comparison.

        Params:
        - score_filter (float): reporting chart wallet cohort minimum prediction score threshold
        - min_scores (int): reporting chart coin cohort minimum number of scores per coin


        Returns:
        - comparison_df: DataFrame with performance metrics across all time periods
        """
        if not self.training_data_cache:
            raise ValueError("No training data loaded. Call load_all_training_data() first")

        logger.milestone(f"Building and evaluating models across {len(self.modeling_dates)} time periods...")

        performance_results = {}

        for i, modeling_date in enumerate(self.modeling_dates, 1):
            date_str = datetime.strptime(modeling_date, '%Y-%m-%d').strftime('%y%m%d')

            if date_str not in self.training_data_cache:
                logger.warning(f"Skipping {modeling_date} - no training data cached")
                continue

            logger.milestone(f"[{i}/{len(self.modeling_dates)}] Building model for {modeling_date}...")

            # Create date-specific modeling config with grid search disabled
            date_config = self._create_date_config(modeling_date)
            date_config['modeling']['grid_search_params']['enabled'] = False

            # Initialize model with date-specific config
            wallet_model = wm.WalletModel(copy.deepcopy(date_config['modeling']))

            # Build model without grid search
            training_data = self.training_data_cache[date_str]
            wallet_model_results = wallet_model.construct_wallet_model(*training_data)

            # Generate evaluator and extract metrics
            model_id, evaluator, _ = wimr.generate_and_save_wallet_model_artifacts(
                model_results=wallet_model_results,
                base_path='../artifacts/wallet_modeling',
                configs = {
                    'wallets_config': date_config,
                    'wallets_investing_config': self.wallets_investing_config,
                    'wallets_metrics_config': self.wallets_metrics_config,
                    'wallets_features_config': self.wallets_features_config,
                    'wallets_epochs_config': self.wallets_epochs_config,
                }
            )

            # Store performance metrics for this period
            performance_results[date_str] = {
                'modeling_date': modeling_date,
                'model_type': wallet_model_results['model_type'],
                **self._extract_performance_metrics(evaluator)
            }

            evaluator.summary_report()

            if self.wallets_investing_config['training_data']['toggle_graph_wallet_performance']:
                evaluator.plot_wallet_evaluation()
                validation_training_data_df = training_data[2]
                validation_target_vars_df = training_data[3]
                wime.run_validation_analysis(
                    date_config,
                    validation_training_data_df,
                    validation_target_vars_df,
                    self.complete_hybrid_cw_id_df,
                    self.complete_market_data_df,
                    model_id,
                    score_filter,
                    min_scores
                )


            logger.info(f"Model completed for {modeling_date}")

            # Clear model memory
            del wallet_model, wallet_model_results, evaluator
            gc.collect()

        # Consolidate results into DataFrame
        comparison_df = pd.DataFrame.from_dict(performance_results, orient='index')
        comparison_df = comparison_df.reset_index().rename(columns={'index': 'date_str'})

        # Add summary statistics across time periods
        self._add_performance_summary_stats(comparison_df)

        # Cache results
        self.model_comparison_results = comparison_df

        logger.milestone("Completed model comparison across all time periods")
        u.notify('ui_1')

        return comparison_df


    def _extract_performance_metrics(self, evaluator) -> dict:
        """
        Extract key performance metrics from model evaluator.

        Params:
        - evaluator: Model evaluator instance (RegressorEvaluator or ClassifierEvaluator)

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
            # Feature importance summary (removed top features/importance)
            # Do not add top_feature_* or top_importance_*
            if 'importances' in evaluator.metrics and evaluator.feature_names:
                metrics['n_features'] = len(evaluator.feature_names)

        # Model-type specific metrics
        if evaluator.modeling_config.get('model_type') == 'classification':
            metrics.update(self._extract_classification_metrics(evaluator))
        elif evaluator.modeling_config.get('model_type') == 'regression':
            metrics.update(self._extract_regression_metrics(evaluator))

        return metrics


    def _extract_classification_metrics(self, evaluator) -> dict:
        """Extract classification-specific metrics."""
        metrics = {}

        # Test set metrics (if available)
        if hasattr(evaluator, 'train_test_data_provided') and evaluator.train_test_data_provided:
            metrics.update({
                'test_accuracy': round(evaluator.metrics.get('accuracy', 0), 3),
                'test_precision': round(evaluator.metrics.get('precision', 0), 3),
                'test_recall': round(evaluator.metrics.get('recall', 0), 3),
                'test_f1': round(evaluator.metrics.get('f1', 0), 3),
                'test_roc_auc': round(evaluator.metrics.get('roc_auc', 0), 3),
            })

        # Validation set metrics (if available)
        if hasattr(evaluator, 'validation_data_provided') and evaluator.validation_data_provided:
            metrics.update({
                'val_accuracy': round(evaluator.metrics.get('val_accuracy', 0), 3),
                'val_precision': round(evaluator.metrics.get('val_precision', 0), 3),
                'val_recall': round(evaluator.metrics.get('val_recall', 0), 3),
                'val_f1': round(evaluator.metrics.get('val_f1', 0), 3),
                'val_roc_auc': round(evaluator.metrics.get('val_roc_auc', 0), 3),
                'val_ret_mean_overall': round(evaluator.metrics.get('val_ret_mean_overall', 0), 3),
                'val_ret_mean_top1': round(evaluator.metrics.get('val_ret_mean_top1', 0), 3),
                'val_ret_mean_top5': round(evaluator.metrics.get('val_ret_mean_top5', 0), 3),
                'positive_pred_return': round(evaluator.metrics.get('positive_pred_return', 0), 3),
            })

            # Threshold information
            if 'y_pred_threshold' in evaluator.metrics:
                metrics['y_pred_threshold'] = round(evaluator.metrics['y_pred_threshold'], 3)

        return metrics


    def _extract_regression_metrics(self, evaluator) -> dict:
        """Extract regression-specific metrics."""
        metrics = {}

        # Test set metrics (if available)
        if hasattr(evaluator, 'train_test_data_provided') and evaluator.train_test_data_provided:
            metrics.update({
                'test_r2': round(evaluator.metrics.get('r2', 0), 3),
                'test_rmse': round(evaluator.metrics.get('rmse', 0), 3),
                'test_mae': round(evaluator.metrics.get('mae', 0), 3),
            })

        # Validation set metrics (if available)
        if 'validation_metrics' in evaluator.metrics:
            vm = evaluator.metrics['validation_metrics']
            metrics.update({
                'val_r2': round(vm.get('r2', 0), 3),
                'val_rmse': round(vm.get('rmse', 0), 3),
                'val_mae': round(vm.get('mae', 0), 3),
                'val_spearman': round(vm.get('spearman', 0), 3),
                'val_top1pct_mean': round(vm.get('top1pct_mean', 0), 3),
            })

        # Training cohort metrics (if available)
        if 'training_cohort' in evaluator.metrics:
            tc = evaluator.metrics['training_cohort']
            metrics.update({
                'training_cohort_r2': round(tc.get('r2', 0), 3),
                'training_cohort_rmse': round(tc.get('rmse', 0), 3),
            })

        return metrics


    def _add_performance_summary_stats(self, comparison_df: pd.DataFrame) -> None:
        """
        Add summary statistics across time periods to the comparison DataFrame.

        Params:
        - comparison_df: DataFrame with performance metrics for each time period
        """
        # Exclude SUMMARY row if it already exists and metadata columns
        data_rows = comparison_df[comparison_df['date_str'] != 'SUMMARY'].copy()
        metadata_cols = ['date_str', 'modeling_date', 'model_type']

        # Initialize summary row with metadata
        summary_data = {
            'date_str': 'SUMMARY',
            'modeling_date': 'ACROSS_ALL_PERIODS',
            'model_type': data_rows['model_type'].iloc[0] if len(data_rows) > 0 else 'classification'
        }

        # Calculate means for all numeric columns
        for col in data_rows.columns:
            if col not in metadata_cols:
                # Convert to numeric, coercing errors to NaN
                numeric_series = pd.to_numeric(data_rows[col], errors='coerce')

                # Calculate mean only if there are valid values
                valid_values = numeric_series.dropna()
                if len(valid_values) > 0:
                    summary_data[col] = round(valid_values.mean(), 4)
                else:
                    summary_data[col] = None  # Use None instead of NaN for cleaner display

        # Remove existing summary row if present
        comparison_df_clean = comparison_df[comparison_df['date_str'] != 'SUMMARY'].copy()

        # Add the new summary row
        comparison_df_clean.loc[len(comparison_df_clean)] = summary_data

        # Update the original DataFrame
        comparison_df.drop(comparison_df.index, inplace=True)
        comparison_df.update(comparison_df_clean)
        for i, row in comparison_df_clean.iterrows():
            comparison_df.loc[i] = row


    def display_model_comparison_summary(self) -> None:
        """Display a formatted summary of model comparison results."""
        if not hasattr(self, 'model_comparison_results') or self.model_comparison_results is None:
            logger.warning("No model comparison results to display. Run run_multi_temporal_model_comparison() first.")
            return

        df = self.model_comparison_results
        model_type = df['model_type'].iloc[0]

        # Build summary message
        summary_lines = [
            "",
            "="*60,
            "MULTI-TEMPORAL MODEL COMPARISON SUMMARY",
            "="*60,
            f"Model Type: {model_type}",
            f"Time periods analyzed: {len(self.modeling_dates)}",
            f"Date range: {min(self.modeling_dates)} to {max(self.modeling_dates)}",
            "",
            "PERFORMANCE STABILITY METRICS:",
            "-" * 40
        ]

        # Key metrics to highlight based on model type
        if model_type == 'classification':
            key_metrics = ['val_roc_auc', 'val_f1', 'val_ret_mean_top1', 'positive_pred_return']
            metric_labels = ['Validation AUC', 'Validation F1', 'Top 1% Return', 'Positive Pred Return']
        else:  # regression
            key_metrics = ['val_r2', 'val_rmse', 'val_spearman', 'val_top1pct_mean']
            metric_labels = ['Validation R²', 'Validation RMSE', 'Validation Spearman', 'Top 1% Mean']

        # Display stability for key metrics
        for metric, label in zip(key_metrics, metric_labels):
            mean_col = f'{metric}_mean'
            cv_col = f'{metric}_cv'
            if mean_col in df.columns and cv_col in df.columns:
                summary_row = df[df['date_str'] == 'SUMMARY'].iloc[0]
                mean_val = summary_row[mean_col]
                cv_val = summary_row[cv_col]
                stability = "Stable" if cv_val < 0.1 else "Moderate" if cv_val < 0.2 else "Unstable"
                summary_lines.extend([
                    f"{label}:",
                    f"  Mean: {mean_val:.4f}",
                    f"  Coefficient of Variation: {cv_val:.4f} ({stability})",
                    ""
                ])

        # Add sample size information
        if 'test_samples_mean' in df.columns:
            summary_row = df[df['date_str'] == 'SUMMARY'].iloc[0]
            summary_lines.extend([
                "SAMPLE SIZES:",
                "-" * 20,
                f"Average Test Samples: {summary_row.get('test_samples_mean', 'N/A')}",
                f"Average Train Samples: {summary_row.get('train_samples_mean', 'N/A')}",
                f"Average Features: {summary_row.get('n_features', 'N/A')}",
                ""
            ])

        # Log as single message
        logger.info("\n".join(summary_lines))
