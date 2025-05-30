"""
functions used to orchestrate time windows. the major steps are:
1. generating shared datasets that are unbounded by time windows
2. generating time window-specific features based off those values
3. merging all time window features together into a training data df
    ready for preprocessing
"""
# pylint: disable=E0401  # unable to import modules from parent folders
# pylint: disable=C0413  # wrong import position
# pylint: disable=C0103  # X_train violates camelcase

import os
import re
import logging
from datetime import timedelta
from typing import List, Dict, Tuple
import pandas as pd

# project modules
import training_data.data_retrieval as dr
import coin_wallet_metrics.indicators as ind
import feature_engineering.feature_generation as fg
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)



# -------------------------------------------------
#            CoinFlowFeaturesOrchestrator
# -------------------------------------------------

class CoinFlowFeaturesOrchestrator:
    """
    Thin wrapper around the current free-function workflow.
    Step 1 only adds an object to carry the three config blobs plus
    lazy placeholders for heavy data pulls; core logic stays where it is
    to minimise diff size and risk.
    """

    def __init__(
        self,
        config: dict,
        metrics_config: dict,
        modeling_config: dict,
        complete_profits_df: pd.DataFrame = None,
        complete_market_data_df: pd.DataFrame = None,
        complete_macro_trends_df: pd.DataFrame = None
    ):
        """
        Params
        ------
        config : config.yaml
        metrics_config : metrics_config.yaml
        modeling_config : modeling_config.yaml
        """
        self.config = config
        self.metrics_config = metrics_config
        self.modeling_config = modeling_config

        # All time dataframes
        self.complete_profits_df = complete_profits_df
        self.complete_market_data_df = complete_market_data_df
        self.complete_macro_trends_df = complete_macro_trends_df

        # Window-trimmed dataframes
        self.profits_df = None
        self.market_data_df = None
        self.macro_trends_df = None
        self.prices_df = None



    # -------------------------------
    #       Primary Interface
    # -------------------------------

    def generate_all_time_windows_model_inputs(self):
        """
        The full sequence to generate X and y splits for all sets that
        incorporates all datasets, time windows, metrics, and indicators.

        Sequence:
        1. Retrieve the base datasets that contain records across all windows
        2. Loop through each time window and generate flattened features
        3a. Concat each dataset's window dfs, then join the datasets together
            to create a comprehensive feature set keyed on coin_id.
        3b. Split the full feature set into train/test/validation/future sets.

        Returns
        -------
        training_data_df : DataFrame
            MultiIndex (time_window, coin_id) with all configured features.
        prices_df : DataFrame
            Prices for all coins on all dates.
        join_logs_df : DataFrame
            Outcomes of each dataset join/fill step.
        """
        logger.info(
            "Beginning generation of all time windows' data for epoch with "
            f"modeling period start of {self.config['training_data']['modeling_period_start']}."
        )
        u.notify('default')

        # 1. Prepare base datasets
        if self.macro_trends_df is not None:
            self._transform_complete_dfs_to_base_data()

        else:
            # Repull complete data if not provided
            (
                self.macro_trends_df,
                self.market_data_df,
                self.profits_df,
                self.prices_df,
            ) = self._prepare_all_windows_base_data()

        # 2. Flattened features per window
        time_windows = self._generate_time_windows()
        all_flattened_filepaths = []

        for time_window in time_windows:

            # Generate custom configs for each window
            window_config, window_metrics_config, window_modeling_config = (
                prepare_configs(
                    self.modeling_config['modeling']['config_folder'], time_window
                )
            )

            # Pass the window configs into feature generators
            _, window_flattened_filepaths = self._generate_window_flattened_dfs(
                window_config, window_metrics_config, window_modeling_config
            )
            all_flattened_filepaths.extend(window_flattened_filepaths)

        # 3. Combine features & targets
        concatenated_dfs = self._concat_dataset_time_windows_dfs(all_flattened_filepaths)
        training_data_df, join_logs_df = self._join_dataset_all_windows_dfs(concatenated_dfs)

        logger.info(
            "Generated training_data_df with shape %s for epoch with modeling period start %s.",
            training_data_df.shape,
            self.config['training_data']['modeling_period_start'],
        )

        return training_data_df, self.prices_df, join_logs_df





    # ----------------------------------
    #           Helper Methods
    # ----------------------------------


    def _transform_complete_dfs_to_base_data(self):
        """
        Converts the complete_* dfs to the windowed dfs associate with the
         config.yaml-defined training period.
        """
        # training_period_end = pd.to_datetime(self.config['training_data']['modeling_period_start'])
        # training_period_duration_= self.config['training_data']['training_period_duration']
        # training_period_start = training_period_end - timedelta(days=training_period_duration_)

        # # DDA 752 LOGIC TO TRIM GOES HERE
        # macro_trends_df = self.complete_macro_trends_df
        # market_data_df = self.complete_market_data_df
        # profits_df = self.complete_profits_df

        # self.macro_trends_df = macro_trends_df
        # self.market_data_df = market_data_df
        # self.profits_df = profits_df
        # self.prices_df = market_data_df[['coin_id','date','price']].copy()

        # u.assert_period(profits_df,training_period_start,training_period_end)
        # u.assert_period(market_data_df,training_period_start,training_period_end)

        pass



    def _prepare_all_windows_base_data(self):
        """
        Retrieves, cleans and adds indicators to the all windows training data.

        Wallet cohort indicators will be generated for each window as the cohort groups will
        change based on the boundary dates of the cohort lookback_period specified in the self.config.

        Returns:
        - macro_trends_df (DataFrame): macro trends keyed on date only
        - market_data_df, prices_df (DataFrame): market data for coin-date pairs
        - profits_df (DataFrame): wallet transfer activity keyed on coin-date-wallet that will be used
            to compute wallet cohorts metrics
        """
        # 1. Data Retrieval, Cleaning, Indicator Calculation
        # --------------------------------------------------
        # Market data: retrieve and clean full history
        market_data_df = dr.retrieve_market_data(self.config['training_data']['training_period_end'],
                                                 dataset=self.config['training_data']['dataset'])
        market_data_df = dr.clean_market_data(market_data_df, self.config,
                                                self.config['training_data']['earliest_window_start'],
                                                self.config['training_data']['training_period_end'])

        # Profits: retrieve and clean profits data spanning the earliest to latest training periods
        profits_df = dr.retrieve_profits_data(
            start_date = self.config['training_data']['earliest_cohort_lookback_start'],
            end_date = self.config['training_data']['training_period_end'],
            min_wallet_inflows = self.config['data_cleaning']['min_wallet_inflows'],
            dataset = self.config['training_data']['dataset'])
        profits_df, _ = dr.clean_profits_df(profits_df, self.config['data_cleaning'])

        # Macro trends: retrieve and clean full history
        macro_trends_df = dr.retrieve_macro_trends_data()
        macro_trends_cols = (
            list(self.config['datasets']['macro_trends'].keys())
            if 'macro_trends' in self.config['datasets']
            else []
        )
        macro_trends_df = dr.clean_macro_trends(macro_trends_df, macro_trends_cols)


        # 2. Filtering based on dataset overlap
        # -------------------------------------
        # Filter market_data to only coins with transfers data if self.configured to
        if self.config['data_cleaning']['exclude_coins_without_transfers']:
            market_data_df = market_data_df[market_data_df['coin_id'].isin(profits_df['coin_id'])]
        # Create prices_df: lightweight reference for other functions
        prices_df = market_data_df[['coin_id','date','price']].copy()

        # Filter profits_df to remove records for any coins that were removed in data cleaning
        profits_df = profits_df[profits_df['coin_id'].isin(market_data_df['coin_id'])]


        # 3. Add indicators (additional time series)
        # ------------------------------------------
        # Macro trends: add indicators if there are metrics self.configured
        if self.metrics_config.get('macro_trends'):
            macro_trends_df = ind.generate_time_series_indicators(macro_trends_df.reset_index(),
                                                                self.metrics_config['macro_trends'],
                                                                None)
        # Market data: add indicators
        market_data_df = ind.generate_time_series_indicators(market_data_df,
                                                            self.metrics_config['time_series']['market_data'],
                                                            'coin_id')
        market_data_df = ind.add_market_data_dualcolumn_indicators(market_data_df)

        return macro_trends_df, market_data_df, profits_df, prices_df



    def _generate_time_windows(self):
        """
        Generates the parameter dicts used by i.prepare_configs() to generate the full set
        of config files.

        Returns:
            time_windows (list of dicts): a list of dicts that can be used to override the
            config.yaml settings for each time window.
        """
        start_date = pd.to_datetime(self.config['training_data']['modeling_period_start'])
        window_frequency = self.config['training_data']['time_window_frequency']

        time_windows = [
            {'config.training_data.modeling_period_start': start_date.strftime('%Y-%m-%d')}
        ]

        for _ in range(self.config['training_data']['additional_windows']):
            start_date -= timedelta(days=window_frequency)
            time_windows.append({'config.training_data.modeling_period_start': start_date.strftime('%Y-%m-%d')})

        return time_windows



    def _generate_window_flattened_dfs(
            self,
            window_config: dict,
            window_metrics_config: dict,
            window_modeling_config: dict):
        """
        Takes the all windows datasets and filters and transforms them as needed to generate
        flattened dfs with all configured aggregations and indicators for all columns.

        Params:
        - window_config, window_metrics_config, window_modeling_config (dicts):
            Full config files for the given window. Note that the window's config files
            will differ from the base configs stored in the class and must be passed as params.

        Returns:
        - window_flattened_dfs (list of DataFrames): all flattened dfs keyed on coin_id with columns
            for every specified aggregation and indicator
        - window_flattened_filepaths (list of strings): filepaths to csv versions of the flattened dfs
        """
        window_flattened_dfs = []
        window_flattened_filepaths = []

        # Market data: generate window-specific flattened metrics
        flattened_market_data_df, flattened_market_data_filepath = fg.generate_window_time_series_features(
            self.market_data_df,
            'time_series-market_data',
            window_config,
            window_metrics_config['time_series']['market_data'],
            window_modeling_config
        )
        window_flattened_dfs.extend([flattened_market_data_df])
        window_flattened_filepaths.extend([flattened_market_data_filepath])

        # Macro trends: generate window-specific flattened metrics
        if not self.macro_trends_df.reset_index().drop(columns='date').empty:
            flattened_macro_trends_df, flattened_macro_trends_filepath = fg.generate_window_macro_trends_features(
                self.macro_trends_df,
                'macro_trends',
                window_config,
                window_metrics_config,
                window_modeling_config
            )
            window_flattened_dfs.extend([flattened_macro_trends_df])
            window_flattened_filepaths.extend([flattened_macro_trends_filepath])

        # Cohorts: generate window-specific flattened metrics
        flattened_cohort_dfs, flattened_cohort_filepaths = fg.generate_window_wallet_cohort_features(
            self.profits_df,
            self.prices_df,
            window_config,
            window_metrics_config,
            window_modeling_config
        )
        window_flattened_dfs.extend(flattened_cohort_dfs)
        window_flattened_filepaths.extend(flattened_cohort_filepaths)

        return window_flattened_dfs, window_flattened_filepaths



    def _concat_dataset_time_windows_dfs(
            self,
            filepaths: List[str],
        ) -> Dict[str, Tuple[pd.DataFrame, str]]:
        """
        For each dataset, concatenates the flattened dfs from all windows together and returns a
        tuple containing the merged df with its fill method.

        This function reads multiple CSV files, extracts their dataset prefixes,
        concatenates DataFrames with the same prefix, and determines the fill method for each dataset.
        It handles subgroups within datasets by prefixing column names with the subgroup name.

        Args:
        - filepaths (List[str]): A list of file paths to the CSV files to be processed.

        Returns:
        - concatenated_dfs (Dict[str, Tuple[pd.DataFrame, str]]): A dictionary where keys are
            dataset prefixes and values are tuples containing:
             - The concatenated DataFrame for each prefix
             - The fill method for the dataset
        """
        # Dictionary to store DataFrames grouped by dataset prefix
        grouped_dfs = {}
        fill_methods = {}

        parent_directory = os.path.join(self.modeling_config['modeling']['modeling_folder'],
                                        'outputs/preprocessed_outputs/')

        for filepath in filepaths:
            # Validate file existence
            if not os.path.exists(filepath):
                raise KeyError(f"File {filepath} does not exist.")

            # Extract the dataset prefix from the filename
            dataset_prefix = self._extract_dataset_key_from_filepath(filepath, parent_directory)

            # Read the CSV
            df = pd.read_csv(filepath)

            # Handle dataset subkeys if present
            if '-' in dataset_prefix:
                dataset, dataset_subkey = dataset_prefix.split('-')  # pylint: disable=W0612
                # Prefix the columns with the subgroup
            else:
                dataset = dataset_prefix

            # Add the DataFrame to the appropriate group
            if dataset_prefix not in grouped_dfs:

                grouped_dfs[dataset_prefix] = []
                # Identify the fill method only once per dataset
                try:
                    fill_methods[dataset_prefix] = self.modeling_config['preprocessing']['fill_methods'][dataset]
                except KeyError as exc:
                    raise KeyError(f"Fill method not found for dataset: {dataset}") from exc

            grouped_dfs[dataset_prefix].append(df)

        # Concatenate DataFrames within each group and pair with fill method
        concatenated_dfs = {
            dataset_prefix: (pd.concat(dfs, ignore_index=True), fill_methods[dataset_prefix])
            for dataset_prefix, dfs in grouped_dfs.items()
        }

        return concatenated_dfs


    @staticmethod
    def _join_dataset_all_windows_dfs(concatenated_dfs):
        """
        Merges the all-windows dataframes of each dataset together according to the fill method
        specified in the model_config. The param is the format from cffo.concat_dataset_time_windows_dfs().

        Params:
        - concatenated_dfs (Dict[str, Tuple[pd.DataFrame, str]]): A dictionary where keys are
            dataset prefixes and values are tuples containing:
                - The concatenated DataFrame for each prefix
                - The fill method for the dataset

        Returns
        - training_data_df (DataFrame): df multiindexed on time_window-coin_id, with columns for
            all flattened features for all datasets.
        - join_log_df (DataFrame): df that contains briefs logs of join methods and outcomes
        """
        # List to store all combinations
        all_coin_windows = []
        log_data = []

        # Pull a unique set of all coin_id-time_window pairs across all dfs
        for dataset, (df, _) in concatenated_dfs.items():
            if 'coin_id' in df.columns and 'time_window' in df.columns:
                combinations = df[['time_window', 'coin_id']].drop_duplicates()
                all_coin_windows.append(combinations)

        # Concatenate all combinations and remove duplicates
        training_data_df = pd.concat(all_coin_windows, ignore_index=True).drop_duplicates()
        training_data_df = training_data_df.set_index(['time_window', 'coin_id'])

        # Iterate through df_list and merge each one
        for dataset, (df, fill_method) in concatenated_dfs.items():
            rows_start = len(training_data_df)

            # Get columns to track from current df so we can fill them appropriately
            merge_cols = df.columns.difference(['time_window', 'coin_id'])

            if fill_method == 'retain_nulls':
                df = df.set_index(['time_window', 'coin_id'])
                training_data_df = training_data_df.join(df, on=['time_window', 'coin_id'], how='left')

            elif fill_method == 'drop_records':
                df = df.set_index(['time_window', 'coin_id'])
                df = df.dropna()
                training_data_df = training_data_df.join(df, on=['time_window', 'coin_id'], how='inner')

            elif fill_method == 'fill_zeros':
                df = df.set_index(['time_window', 'coin_id'])
                training_data_df = training_data_df.join(df, on=['time_window', 'coin_id'], how='left')
                # Only fill the columns from current df
                training_data_df[merge_cols] = training_data_df[merge_cols].fillna(0)

            elif fill_method == 'extend_coin_ids':
                df = df.set_index('time_window')
                training_data_df = training_data_df.join(df, on='time_window', how='inner')

            elif fill_method == 'extend_time_windows':
                df = df.set_index('coin_id')
                training_data_df = training_data_df.join(df, on='coin_id', how='inner')

            else:
                raise ValueError(f"Invalid fill method '{fill_method}' found in config.yaml.")

            rows_end = len(training_data_df)
            log_data.append({
                'dataset': dataset,
                'fill_method': fill_method,
                'rows_start': rows_start,
                'rows_end': rows_end,
                'rows_removed': rows_start - rows_end
            })

        join_log_df = pd.DataFrame(log_data)

        return training_data_df, join_log_df



    @staticmethod
    def _extract_dataset_key_from_filepath(filepath, parent_directory):
        """
        Helper function for concat_dataset_time_windows_dfs().

        Extracts the dataset key (e.g. 'time_series', 'wallet_cohorts', etc) from the
        flattened filepath.

        Params:
        - filepath (str): filepath output by generate_window_flattened_dfs().
        - parent_directory (string): the parent directory of the flattened files

        Returns:
        - dataset_key (string): dataset key that matches to the config and metrics_config
        """
        # Remove the parent directory part from the filepath
        relative_path = filepath.replace(parent_directory, '')

        # Split the remaining path into parts
        parts = relative_path.split(os.sep)

        # The filename should be the last part
        filename = parts[-1]

        # Define the date pattern regex
        date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}')

        # Find the date in the filename
        date_match = date_pattern.search(filename)

        if date_match:
            # Extract the part before the date
            dataset_key = filename[:date_match.start()].rstrip('_')
            return dataset_key
        else:
            raise ValueError(f"Could not parse dataset of file {filepath}")




# ------------------------------------
#       Config Utility Functions
# ------------------------------------

def prepare_configs(config_folder, override_params):
    """
    Loads config files from the config_folder using load_config and applies overrides specified
    in override_params.

    Args:
    - config_folder (str): Path to the folder containing the configuration files.
    - override_params (dict): Dictionary of flattened parameters to override in the loaded configs.

    Returns:
    - config (dict): The main config file with overrides applied.
    - metrics_config (dict): The metrics configuration with overrides applied.
    - modeling_config (dict): The modeling configuration with overrides applied.

    Raises:
    - KeyError: if any key from override_params does not match an existing key in the
        corresponding config.
    """

    # Load the main config files using load_config
    config_path = os.path.join(config_folder, 'config.yaml')
    metrics_config_path = os.path.join(config_folder, 'metrics_config.yaml')
    modeling_config_path = os.path.join(config_folder, 'modeling_config.yaml')

    config = u.load_config(config_path)
    metrics_config = u.load_config(metrics_config_path)
    modeling_config = u.load_config(modeling_config_path)

    # Apply the flattened overrides to each config
    for full_key, value in override_params.items():
        if full_key.startswith('config.'):
            # Validate the key exists before setting the value
            validate_key_in_config(config, full_key[len('config.'):])
            set_nested_value(config, full_key[len('config.'):], value)
        elif full_key.startswith('metrics_config.'):
            # Validate the key exists before setting the value
            validate_key_in_config(metrics_config, full_key[len('metrics_config.'):])
            set_nested_value(metrics_config, full_key[len('metrics_config.'):], value)
        elif full_key.startswith('modeling_config.'):
            # Validate the key exists before setting the value
            validate_key_in_config(modeling_config, full_key[len('modeling_config.'):])
            set_nested_value(modeling_config, full_key[len('modeling_config.'):], value)
        else:
            raise ValueError(f"Unknown config section in key: {full_key}")

    # reapply the period boundary dates based on the current config['training_data'] params
    period_dates = u.calculate_period_dates(config)
    config['training_data'].update(period_dates)

    return config, metrics_config, modeling_config

# helper function for prepare_configs()
def set_nested_value(config, key_path, value):
    """
    Sets a value in a nested dictionary based on a flattened key path.

    Args:
    - config (dict): The configuration dictionary to update.
    - key_path (str): The flattened key path (e.g., 'config.data_cleaning.max_wallet_inflows').
    - value: The value to set at the given key path.
    """
    keys = key_path.split('.')
    sub_dict = config
    for key in keys[:-1]:  # Traverse down to the second-to-last key
        sub_dict = sub_dict.setdefault(key, {})
    sub_dict[keys[-1]] = value  # Set the value at the final key

# helper function for prepare_configs
def validate_key_in_config(config, key_path):
    """
    Validates that a given key path exists in the nested configuration.

    Args:
    - config (dict): The configuration dictionary to validate.
    - key_path (str): The flattened key path to check.
        (e.g. 'config.data_cleaning.max_wallet_inflows')

    Raises:
    - KeyError: If the key path does not exist in the config.
    """
    keys = key_path.split('.')
    sub_dict = config
    for key in keys[:-1]:  # Traverse down to the second-to-last key
        if key not in sub_dict:
            raise KeyError(
                f"Key '{key}' not found in config at level '{'.'.join(keys[:-1])}'")
        sub_dict = sub_dict[key]
    if keys[-1] not in sub_dict:
        raise KeyError(
            f"Key '{keys[-1]}' not found in config at final level '{'.'.join(keys[:-1])}'")

