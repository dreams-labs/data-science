"""
Business logic
--------------
* Goal – Orchestrate multi-temporal grid search experiments to assess feature stability
  and parameter consistency across different investing cycles, enabling robust model selection
  for cryptocurrency wallet behavior prediction.

* Core Responsibilities
  - Generate training data across multiple investing cycles with configurable offsets
  - Execute parallel grid search experiments on each temporal dataset
  - Consolidate results to identify temporally stable parameter combinations
  - Build and compare models across investing cycles without grid search
  - Analyze parameter stability using coefficient of variation metrics

* Workflow Sequence
  1. Data Generation: `generate_all_training_data()` creates epoch-specific training
     datasets using multithreaded `WalletEpochsOrchestrator` instances
  2. Data Loading: `load_all_training_data()` caches pre-generated datasets in memory
     for efficient repeated access during experiments
  3. Grid Search: `run_multi_temporal_grid_search()` executes parameter optimization
     across all investing cycles with period-specific baseline comparisons
  4. Model Comparison: `run_multi_temporal_model_comparison()` builds models using
     base parameters (no search) and evaluates performance stability
  5. Results Analysis: `consolidate_results()` and `analyze_parameter_stability()`
     provide comprehensive temporal stability metrics

* Key Features
  - Parallel processing with configurable thread pools for data generation and loading
  - Automatic period-specific baseline calculations for normalized comparisons
  - Comprehensive stability analysis using coefficient of variation thresholds
  - Feature importance aggregation across temporal models
  - Escape hatch for S3 training data export without model building

Downstream consumers
--------------------
* Model Selection Workflows – identifies robust parameters for production deployment
* Research & Development – temporal stability analysis for feature engineering
* Risk Management – parameter sensitivity assessment across market conditions
* MLOps Pipelines – automated model validation across different investing cycles

Key dependencies
----------------
* WalletEpochsOrchestrator – generates epoch-specific training datasets
* WalletModel – provides grid search and model building capabilities
* Model Evaluation Framework – extracts performance metrics across model types
* Configuration Management – handles temporal offset and parameter grid definitions

Export capabilities
-------------------
* Consolidated grid search results with temporal stability metrics
* Individual period performance comparisons and model artifacts
* Parameter ranking by stability and performance combination
* Feature importance aggregation across all temporal experiments
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import ast
import gc
import time
from datetime import datetime
from collections import defaultdict
from typing import List
from pathlib import Path
import pandas as pd
import numpy as np

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
    Multi-temporal grid search orchestrator for cryptocurrency wallet behavior models.

    Technical overview
    ------------------
    * Architecture: Coordinates parallel execution of grid search experiments across
      multiple investing cycles, with intelligent caching and result consolidation

    * Core Pipeline:
        1. Generate epoch-specific training datasets with configurable date offsets
        2. Execute parameter optimization on each temporal dataset with baseline computation
        3. Build models using base parameters across periods for stability assessment
        4. Consolidate results with coefficient of variation metrics and stability rankings

    * Concurrency & Memory:
        - Configurable thread pools for data generation, loading, and processing
        - Intelligent caching of training datasets in `training_data_cache`
        - Explicit garbage collection and memory cleanup between experiments
        - Thread-safe result collection with automatic failure handling

    Key methods
    -----------
    * generate_all_training_data(): Parallel epoch dataset generation with caching
    * run_multi_temporal_grid_search(): Grid search execution across investing cycles
    * run_multi_temporal_model_comparison(): Model building and evaluation workflow
    * consolidate_results(): Result aggregation with stability metrics

    Output artifacts
    ----------------
    * Consolidated grid search results with temporal stability rankings
    * Parameter recommendations based on stability and performance metrics
    * Feature importance aggregation across all temporal experiments
    """

    def __init__(
        self,
        wallets_config: dict,
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
        - wallets_config: Base wallet configuration to be modified for each date
        - wallets_metrics_config: Metrics configuration
        - wallets_features_config: Features configuration
        - wallets_epochs_config: Epochs configuration
        - modeling_dates: List of modeling period start dates (YYYY-MM-DD format)
        """
        self.wallets_config = copy.deepcopy(wallets_config)
        self.wallets_investing_config = wallets_investing_config
        self.wallets_metrics_config = wallets_metrics_config
        self.wallets_features_config = wallets_features_config
        self.wallets_epochs_config = wallets_epochs_config
        self.modeling_dates = modeling_dates

        # If True, regenerate all training data regardless of cache
        self.force_regenerate_data = (wallets_investing_config['training_data']
                                      .get('toggle_overwrite_multioffset_parquet',False))
        # Override companion toggle in wallets_config
        if self.force_regenerate_data:
            self.wallets_config['training_data']['rebuild_multioffset_dfs'] = True

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
        logger.milestone(f"Generating training data for {len(self.modeling_dates)} investing cycles...")

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
                        logger.milestone(f"[{i}/{len(self.modeling_dates)}] Generated data for {modeling_date}.")

                except Exception as e:
                    failed_dates.append((modeling_date, e))

        # Handle any failures
        if failed_dates:
            error_details = "\n".join([f"  {date}: {error}" for date, error in failed_dates])
            logger.error(f"Failed to generate training data for dates {failed_dates}:\n{error_details}")

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

        logger.milestone("Loading training data for all investing cycles...")

        parquet_folder = self.wallets_config['training_data']['parquet_folder']
        max_workers = self.wallets_investing_config['n_threads']['training_data_loading']

        def load_single_date(modeling_date: str) -> tuple:
            """
            Load training data for a single modeling date.

            Filepaths are explicitly declared to provide more debugging information.
            """
            date_str = datetime.strptime(modeling_date, '%Y-%m-%d').strftime('%y%m%d')
            base_path = f"{parquet_folder}/{date_str}"
            current_file = None

            try:
                # Load all four DataFrames for this date
                current_file = f"{base_path}/multioffset_wallet_training_data_df.parquet"
                wallet_training_data_df = pd.read_parquet(current_file)

                current_file = f"{base_path}/multioffset_wallet_target_vars_df.parquet"
                wallet_target_vars_df = pd.read_parquet(current_file)

                current_file = f"{base_path}/multioffset_validation_training_data_df.parquet"
                validation_training_data_df = pd.read_parquet(current_file)

                current_file = f"{base_path}/multioffset_validation_target_vars_df.parquet"
                validation_target_vars_df = pd.read_parquet(current_file)

                # Processing operations (track which DataFrame is being processed)
                current_file = f"{base_path}/multioffset_wallet_target_vars_df.parquet (ranking operation)"
                wallet_target_vars_df['cw_coin_return_rank'] = pd.Series(
                    wallet_target_vars_df['cw_coin_return'],
                    index=wallet_target_vars_df.index
                ).rank(method='average', pct=True)

                current_file = f"{base_path}/multioffset_validation_target_vars_df.parquet (ranking operation)"
                validation_target_vars_df['cw_coin_return_rank'] = pd.Series(
                    validation_target_vars_df['cw_coin_return'],
                    index=validation_target_vars_df.index
                ).rank(method='average', pct=True)

                # Apply predrop_features logic if configured
                date_config = self._create_date_config(modeling_date)
                if date_config['training_data'].get('predrop_features',False):
                    drop_patterns = date_config['modeling']['feature_selection']['drop_patterns']
                    protected_columns = self.wallets_config['modeling']['feature_selection']['protected_features']
                    col_dropper = bp.DropColumnPatterns(drop_patterns, protected_columns)
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

            except Exception as e:
                # Add specific file path context to the error
                error_with_context = Exception(f"Error processing {current_file}: {str(e)}")
                error_with_context.__cause__ = e
                return date_str, None, None, error_with_context
        # Execute parallel loading
        failed_dates = []

        i = 0
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

                    i+=1
                    logger.info(f"[{i}/{len(self.modeling_dates)}] "
                                f"Loaded training data for {modeling_date} (shapes: "
                                f"{shapes_info[0]}, {shapes_info[1]}, "
                                f"{shapes_info[2]}, {shapes_info[3]})")

                except Exception as e:
                    failed_dates.append((modeling_date, e))

        # Handle any failures
        if failed_dates:
            error_details = "\n".join([f"  {date}: {error}" for date, error in failed_dates])
            logger.error(f"Failed to load training data for {len(failed_dates)} dates:\n{error_details}")

            # Raise the actual error instead of masking it
            _, first_error = failed_dates[0]
            raise first_error

        logger.milestone(f"Successfully loaded training data for {len(self.modeling_dates)} "
                         f"periods using {max_workers} threads")

    @u.timing_decorator(logging.MILESTONE)  # pylint: disable=no-member
    def run_multi_temporal_grid_search(self) -> None:
        """
        Execute grid search across all investing cycles and cache results.
        """
        # Validate grid search is enabled
        if not self.wallets_config['modeling']['grid_search_params'].get('enabled',False):
            raise ValueError("Grid search must be enabled in base configuration")

        if not self.training_data_cache:
            raise ValueError("No training data loaded. Call load_all_training_data() first")

        logger.milestone(f"Running grid search across {len(self.modeling_dates)} investing cycles...")

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

        logger.milestone("Completed grid search across all investing cycles")
        u.notify('ui_1')




    # ---------------------------------
    #         Reporting Methods
    # ---------------------------------

    def consolidate_results(self) -> pd.DataFrame:
        """
        Consolidate grid search results across all investing cycles into a summary DataFrame.
        Includes period-specific baseline comparisons.

        Returns:
        - consolidated_df: DataFrame with parameters as rows, dates as columns, plus baseline comparisons
        """
        if not self.grid_search_results:
            raise ValueError("No grid search results available. Run run_multi_temporal_grid_search() first")

        logger.info("Consolidating grid search results across investing cycles...")

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

        # Calculate summary statistics across investing cycles (original scores only)
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
                f"across {len(date_columns)} investing cycles")

        return consolidated_df


    def aggregate_by_individual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create performance metrics for each individual feature across all feature sets.

        Params:
        - df (DataFrame): input performance data with param_value column containing feature sets

        Returns:
        - feature_performance_df (DataFrame): performance metrics averaged by individual feature
        """
        # Identify numeric columns to aggregate
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Parse param_value strings to extract individual features
        feature_rows = []
        for _, row in df.iterrows():
            # Extract features from the set string representation
            param_str = row['param_value']
            # Remove "Retained features: " prefix and parse the set
            features_str = param_str.replace("Retained features: ", "")
            # Use eval to parse the set (assuming it's properly formatted)
            features_set = ast.literal_eval(features_str)

            # Create a row for each feature
            for feature in features_set:
                feature_row = row[numeric_cols].to_dict()
                feature_row['feature'] = feature
                feature_rows.append(feature_row)

        # Convert to DataFrame
        exploded_df = pd.DataFrame(feature_rows)

        # Group by feature and calculate mean performance
        feature_performance_df = exploded_df.groupby('feature')[numeric_cols].mean().reset_index()

        # Add count of how many feature sets included each feature
        feature_performance_df['inclusion_count'] = exploded_df.groupby('feature').size().reset_index(drop=True)

        # Sort by mean_score descending
        if 'mean_score' in feature_performance_df.columns:
            feature_performance_df = feature_performance_df.sort_values('mean_score', ascending=False)

        return feature_performance_df





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
        date_config = copy.deepcopy(self.wallets_config)
        date_config['training_data']['modeling_period_start'] = modeling_date
        return wcm.add_derived_values(date_config)


    def _check_data_exists(self, modeling_date: str) -> bool:
        """Check if training data already exists for the specified date."""
        parquet_folder = self.wallets_config['training_data']['parquet_folder']
        date_str = datetime.strptime(modeling_date, '%Y-%m-%d').strftime('%y%m%d')

        required_files = [
            f"{parquet_folder}/{date_str}/multioffset_wallet_training_data_df.parquet",
            f"{parquet_folder}/{date_str}/multioffset_wallet_target_vars_df.parquet",
            f"{parquet_folder}/{date_str}/multioffset_validation_training_data_df.parquet",
            f"{parquet_folder}/{date_str}/multioffset_validation_target_vars_df.parquet"
        ]

        return all(Path(file_path).exists() for file_path in required_files)



    def _calculate_sample_baseline(self, training_data_tuple) -> float:
        """Calculate population baseline from sample training data."""
        _, _, _, validation_target_vars_df = training_data_tuple

        target_var = self.wallets_config['modeling']['target_variable']
        if target_var in validation_target_vars_df.columns:
            returns = validation_target_vars_df[target_var]
            return u.winsorize(returns, 0.001).mean()

        return 0.0





    # ---------------------------------
    #    Temporal Modeling Sequence
    # ---------------------------------

    @u.timing_decorator(logging.MILESTONE)  # pylint: disable=no-member
    def run_multi_temporal_model_comparison(
        self,
        score_filter: float = 0.2,  # only used for data viz
        min_scores: int = 10,       # only used for data viz
    ) -> pd.DataFrame:
        """
        Build models for each investing cycle using base parameters (no grid search)
        and consolidate their performance metrics for comparison.

        Params:
        - score_filter (float): reporting chart wallet cohort minimum prediction score threshold
        - min_scores (int): reporting chart coin cohort minimum number of scores per coin


        Returns:
        - comparison_df: DataFrame with performance metrics across all investing cycles
        """
        if not self.training_data_cache:
            raise ValueError("No training data loaded. Call load_all_training_data() first")

        logger.milestone(f"Building and evaluating models across {len(self.modeling_dates)} investing cycles...")

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
            wallet_model = wm.WalletModel(copy.deepcopy(date_config['modeling'], self.wallets_config))

            # Build model without grid search
            training_data = self.training_data_cache[date_str]
            wallet_model_results = wallet_model.construct_wallet_model(*training_data, self.wallets_config)

            # Escape early when only exporting S3 training data
            if date_config['modeling'].get('export_s3_training_data', {}).get('enabled', False):
                logger.info(f"S3 training data export completed for modeling date '{modeling_date}'.")
                continue

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
                'model_id': model_id,
                **self._extract_performance_metrics(evaluator)
            }

            evaluator.summary_report()

            if self.wallets_investing_config['training_data'].get('toggle_graph_wallet_performance',False):
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
                    min_scores,
                    self.wallets_investing_config['training_data'].get('toggle_score_agg_coin_graphs',False)
                )


            logger.info(f"Model completed for {modeling_date}")
            time.sleep(0.5)

            # Clear model memory
            del wallet_model, wallet_model_results, evaluator
            gc.collect()

        # Escape if we're just exporting data
        if date_config['modeling'].get('export_s3_training_data', {}).get('enabled', False):
            logger.milestone(f"S3 training data export completed for all {len(self.modeling_dates)} "
                             "modeling dates.")
            return

        # Consolidate results into DataFrame
        comparison_df = pd.DataFrame.from_dict(performance_results, orient='index')
        comparison_df = comparison_df.reset_index().rename(columns={'index': 'date_str'})

        # Add summary statistics across investing cycles
        self._add_performance_summary_stats(comparison_df)

        # Cache results
        self.model_comparison_results = comparison_df

        logger.milestone("Completed model comparison across all investing cycles.")
        u.notify('ui_1')

        return comparison_df






    # ----------------------------
    #    Reporting and Analysis
    # ----------------------------

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
        Add summary statistics across investing cycles to the comparison DataFrame.

        Params:
        - comparison_df: DataFrame with performance metrics for each investing cycle
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



    def aggregate_temporal_feature_importance(self) -> pd.DataFrame:
        """
        Aggregate feature importance across all models generated by temporal grid search.

        Returns:
        - importance_stats_df: DataFrame with importance statistics across investing cycles
        """
        if self.model_comparison_results is None:
            raise ValueError("No model comparison results available. Run run_multi_temporal_model_comparison() first")

        # Collect all importance data
        feature_importance_data = defaultdict(list)

        for _, row in self.model_comparison_results.iterrows():
            if row['date_str'] == 'SUMMARY':  # Skip summary row
                continue

            model_id = row['model_id']  # Now this will work!

            # Load the model report for this model_id
            try:
                report = wimr.load_model_report(
                    model_id,
                    self.wallets_config['training_data']['model_artifacts_folder']
                )

                if 'evaluation' in report and 'importances' in report['evaluation']:
                    importances = report['evaluation']['importances']

                    # Validate structure
                    if 'feature' not in importances or 'importance' not in importances:
                        continue

                    features = importances['feature']
                    importance_values = importances['importance']

                    # Only process non-zero importance values
                    for feature, importance in zip(features, importance_values):
                        feature_importance_data[feature].append(importance)

            except FileNotFoundError:
                logger.warning(f"Model report not found for {model_id}")
                continue

        # Calculate statistics for each feature
        importance_stats = []

        for feature, values in feature_importance_data.items():
            if values:  # Only process features with at least one non-zero value
                stats = {
                    'feature': feature,
                    'mean_importance': np.mean(values),
                    'median_importance': np.median(values),
                    'std_importance': np.std(values),
                    'count': len(values),
                    'min_importance': np.min(values),
                    'max_importance': np.max(values)
                }
                importance_stats.append(stats)

        # Convert to DataFrame and sort by mean importance
        importance_stats_df = pd.DataFrame(importance_stats)

        if not importance_stats_df.empty:
            importance_stats_df = importance_stats_df.sort_values('mean_importance', ascending=False).reset_index(drop=True)

            logger.info(f"Aggregated feature importance across {len(self.model_comparison_results) - 1} temporal models, "
                    f"for {len(importance_stats_df)} features.")
        else:
            logger.warning("No feature importance data found across temporal models")

        return importance_stats_df
