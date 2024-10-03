"""
functions used to build coin-level features from training data
"""
# pylint: disable=C0302

import os
from datetime import datetime
import time
import copy
import re
import pandas as pd
import numpy as np
import dreams_core.core as dc
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# pylint: disable=E0401
# project module imports
import coin_wallet_metrics as cwm


# set up logger at the module level
logger = dc.setup_logger()


def generate_time_series_features(
        dataset_name,
        dataset_df,
        config,
        metrics_config,
        modeling_config
    ):
    """
    Generates features for a time series dataset by calculating all metrics defined in the
    metrics_config for each coin_id. The metrics are then flattened into features and saved
    in the format needed to merge them using fe.create_training_data_df().

    Params:
    - dataset_name (string): the key of the dataset,
        e.g. config['datasets']['time_series'][{dataset_name}]
    - dataset_df (pd.DataFrame): the dataframe containing the columns with defined metrics,
        e.g. a column for each of metrics_config['time_series'][{dataset_name}].keys()
    - config (dict): config.yaml
    - metrics_config (dict): metrics_config.yaml
    - modeling_config (dict): modeling_config.yaml

    Returns:
    - training_data_tuples (list of tuples): Each tuple contains the preprocessed file name
        and fill method for the value column contained in dataset_metrics_config, formatted for
        input to fe.create_training_data_df().
    - training_data_dfs (list of pd.DataFrames): A list of preprocessed DataFrames for each
        value_column included in the dataset_metrics_config.
    """
    training_data_tuples = []
    training_data_dfs = []

    dataset_metrics_config = metrics_config['time_series'][dataset_name]

    # calculate metrics for each value column
    for value_column in list(dataset_metrics_config.keys()):

        # a value_column-specific df will be used for feature generation
        value_column_config = config['datasets']['time_series'][dataset_name][value_column]
        value_column_metrics_config = dataset_metrics_config[value_column]
        value_column_df = dataset_df[['date','coin_id',value_column]].copy()

        # check if there are any time series indicators to add, e.g. sma, ema, etc
        if 'indicators' in value_column_metrics_config:
            value_column_metrics_df, _ = cwm.generate_time_series_indicators(
                value_column_df,
                config,
                value_column_metrics_config['indicators'],
                value_column,
                id_column='coin_id'
            )

        else:
            # if no indicators are needed, pass through coins with complete date coverage
            value_column_metrics_df, _ = cwm.split_dataframe_by_coverage(
                value_column_df,
                config['training_data']['training_period_start'],
                config['training_data']['training_period_end'],
                id_column='coin_id'
            )

        # generate features from the metrics
        value_column_features_df, value_column_tuple = convert_dataset_metrics_to_features(
            value_column_metrics_df,
            value_column_config,
            dataset_metrics_config,
            config,
            modeling_config
        )

        logger.info('Generated features for %s.%s.%s',
                    'time_series', dataset_name, value_column)

        training_data_tuples.append(value_column_tuple)
        training_data_dfs.append(value_column_features_df)

    return training_data_tuples, training_data_dfs



def generate_wallet_cohort_features(
        profits_df,
        config,
        metrics_config,
        modeling_config
    ):
    """
    Generates features for all wallet cohorts by calculating the metrics defined in the
    metrics_config for each coin_id. The metrics are then flattened into features and saved
    in the format needed to merge them using fe.create_training_data_df().
    """
    wallet_cohort_tuples = []
    wallet_data_dfs = []

    for cohort_name in metrics_config['wallet_cohorts']:

        # load configs
        dataset_metrics_config = metrics_config['wallet_cohorts'][cohort_name]
        dataset_config = config['datasets']['wallet_cohorts'][cohort_name]

        # identify wallets in the cohort
        cohort_summary_df = cwm.classify_wallet_cohort(profits_df, dataset_config, cohort_name)
        cohort_wallets = cohort_summary_df[cohort_summary_df['in_cohort']]['wallet_address']

        # If no cohort members were identified, continue
        if len(cohort_wallets) == 0:
            logger.info("No wallets identified as members of cohort '%s'", cohort_name)
            continue

        # generate cohort buysell_metrics
        cohort_metrics_df = cwm.generate_buysell_metrics_df(
            profits_df,config['training_data']['training_period_end'],cohort_wallets)

        # generate features from the metrics
        dataset_features_df, dataset_tuple = convert_dataset_metrics_to_features(
            cohort_metrics_df,
            dataset_config,
            dataset_metrics_config,
            config,
            modeling_config
        )

        # identify columns for logging
        dataset_features = dataset_features_df.columns.tolist()
        dataset_features.remove('coin_id')

        logger.info("Generated %s features for wallet cohort '%s'.",
                    len(dataset_features), cohort_name)
        logger.debug('Features generated: %s', dataset_features)

        wallet_cohort_tuples.append(dataset_tuple)
        wallet_data_dfs.append(dataset_features_df)

    return wallet_cohort_tuples, wallet_data_dfs



def convert_dataset_metrics_to_features(
    dataset_metrics_df,
    dataset_config,
    dataset_metrics_config,
    config,
    modeling_config,
):
    """
    Converts a dataset keyed on coin_id-date into features by flattening and preprocessing.

    Args:
        dataset_metrics_df (pd.DataFrame): Input DataFrame containing raw metrics data.
        dataset_config (dict): The component of config['datasets'] relating to this dataset
        dataset_metrics_config (dict): The component of metrics_config relating to this dataset
        config (dict): The whole main config, which includes period boundary dates
        modeling_config (dict): The whole modeling_config, which includes a list of tables to
            drop in preprocessing

    Returns:
        preprocessed_df (pd.DataFrame): The preprocessed DataFrame ready for model training.
        dataset_tuple (tuple): Contains the preprocessed file name and fill method for the dataset.
    """

    # Flatten the metrics DataFrame into the required format for feature engineering
    flattened_metrics_df = flatten_coin_date_df(
        dataset_metrics_df,
        dataset_metrics_config,
        config['training_data']['training_period_end']  # Ensure data is up to training period end
    )

    # Save the flattened output and retrieve the file path
    _, flattened_metrics_filepath = save_flattened_outputs(
        flattened_metrics_df,
        os.path.join(
            modeling_config['modeling']['modeling_folder'],  # Folder to store flattened outputs
            'outputs/flattened_outputs'
        ),
        dataset_config['description'],  # Descriptive metadata for the dataset
        config['training_data']['modeling_period_start']  # Ensure data starts from modeling period
    )

    # Preprocess the flattened data and return the preprocessed file path
    preprocessed_df, preprocessed_filepath = preprocess_coin_df(
        flattened_metrics_filepath,
        modeling_config,
        dataset_config,
        dataset_metrics_config
    )

    # this tuple is the input for create_training_data_df() that will merge all the files
    dataset_tuple = (
        preprocessed_filepath.split('preprocessed_outputs/')[1],  # Extract file name from the path
        dataset_config['fill_method']  # fill method to be used as part of the merge process
    )

    return preprocessed_df, dataset_tuple



def flatten_coin_date_df(df, df_metrics_config, training_period_end):
    """
    Processes all coins in the DataFrame and flattens relevant time series metrics for each coin.

    Params:
    - df (pd.DataFrame): DataFrame containing time series data for multiple coins (coin_id-date).
    - df_metrics_config (dict): Configuration object showing the settings for the metrics
        that apply to the specific input df.
    - training_period_end (datetime): The end of the training period to ensure dates are filled
        until this date and that rolling windows end on the training_period_end

    Returns:
    - result (pd.DataFrame): A DataFrame of flattened features for all coins.

    Raises:
    - ValueError:
        - If the input DataFrame is empty.
        - If the time series data contains missing dates or NaN values.
    """

    # Step 1: Data Quality Checks
    # ---------------------------
    # Check if df is empty
    if df.empty:
        raise ValueError("Input DataFrame is empty. Check your data and ensure it's populated.")

    # Check that all coin-date pairs are complete for the full df
    missing_dates_dict = {}
    for coin_id in df['coin_id'].unique():

        coin_df = df[df['coin_id'] == coin_id]

        # Create the full date range for the coin, explicitly cast to pd.Timestamp
        full_date_range = pd.date_range(start=coin_df['date'].min(), end=training_period_end)

        # Get the existing dates for the coin, explicitly cast to pd.Timestamp
        existing_dates = set(pd.to_datetime(coin_df['date'].unique()).to_pydatetime())

        # Find the missing dates by subtracting existing from full date range
        missing_dates = set(full_date_range) - existing_dates

        # Store the missing dates for the current coin_id
        missing_dates_dict[coin_id] = sorted(missing_dates)

    # Convert to DataFrame for easier display
    missing_dates_df = pd.DataFrame(list(missing_dates_dict.items())
                                    , columns=['coin_id', 'missing_dates'])
    if any(len(missing_dates_df) > 0 for missing in missing_dates):
        raise ValueError(
            "Timeseries has missing dates. Ensure all dates are filled up to training_period_end."
            f"Missing dates: {missing_dates}"
        )

    # Check for NaN values
    if df.isnull().values.any():
        raise ValueError(
            "Timeseries contains NaN values. Ensure imputation is done upstream before flattening."
        )
    # Check if df columns have configurations in df_metrics_config
    configured_metrics = []
    for m in df_metrics_config.keys():
        if m in df.columns:
            configured_metrics.append(m)
    if len(configured_metrics) == 0:
        raise ValueError("No columns in the input_df were found in df_metrics_config.")


    # Step 2: Flatten the metrics
    # ---------------------------
    start_time = time.time()
    logger.debug("Flattening columns %s into coin-level features...", configured_metrics)

    all_flat_features = []

    # Loop through each unique coin_id
    for coin_id in df['coin_id'].unique():
        # Filter the data for the current coin
        coin_df = df[df['coin_id'] == coin_id].copy()

        # Flatten the features for this coin
        flat_features = flatten_date_features(coin_df, df_metrics_config)

        # Add coin_id to the flattened features
        flat_features['coin_id'] = coin_id

        all_flat_features.append(flat_features)

    # Convert the list of feature dictionaries into a DataFrame
    flattened_df = pd.DataFrame(all_flat_features)

    logger.debug('Flattened input df into coin-level features with shape %s after %.2f seconds.',
                flattened_df.shape, time.time() - start_time)


    return flattened_df



def flatten_date_features(time_series_df, df_metrics_config):
    """
    Flattens all relevant time series metrics for a single coin into a row of features.

    Params:
    - time_series_df (pd.DataFrame): DataFrame keyed on date with a time series dataset
    - df_metrics_config (dict): Configuration object with metric rules from the metrics file for
        the specific input df.

    Returns:
    - flat_features (dict): A dictionary of flattened features for the coin.

    Raises:
    - ValueError: If no columns were matched to the config.
    """
    flat_features = {}
    matched_columns = False

    # promote indicators to the same key level as primary metrics
    df_metrics_indicators_config = promote_indicators_to_metrics(df_metrics_config)

    # Apply global stats calculations for each metric
    for metric, config in df_metrics_indicators_config.items():
        if metric not in time_series_df.columns:
            continue

        matched_columns = True
        ts = time_series_df[metric].copy()  # Get the time series for this metric

        # Standard aggregations
        if 'aggregations' in config:
            for agg, agg_config in config['aggregations'].items():
                agg_value = calculate_aggregation(ts, agg)

                # Generate bucket columns if buckets are specified in the config
                if agg_config and 'buckets' in agg_config:
                    bucket_category = bucketize_value(agg_value, agg_config['buckets'])
                    flat_features[f'{metric}_{agg}_bucket'] = bucket_category

                # Return the aggregate metric if it is not bucketized
                else:
                    flat_features[f'{metric}_{agg}'] = agg_value


        # Rolling window calculations (unchanged)
        rolling = config.get('rolling', False)
        if rolling:

            rolling_aggregations = config['rolling'].get('aggregations', [])
            comparisons = config['rolling'].get('comparisons', [])
            window_duration = config['rolling']['window_duration']
            lookback_periods = config['rolling']['lookback_periods']

            # Calculate rolling metrics and update flat_features
            rolling_features = calculate_rolling_window_features(
                ts, window_duration, lookback_periods, rolling_aggregations, comparisons, metric)
            flat_features.update(rolling_features)

    if not matched_columns:
        raise ValueError("No metrics matched the columns in the DataFrame.")

    return flat_features


def promote_indicators_to_metrics(df_metrics_config):
    """
    Moves indicators to the same level as other columns in the config file so that their
    features can be generated the same way they are for primary metrics.

    Params:
    - df_metrics_config (dict): Configuration object with metric rules from the metrics file for
        the specific input df.

    Returns:
    """
    # make a deep copy to avoid impacting the original dict
    df_metrics_indicators_config = copy.deepcopy(df_metrics_config)

    for key, value in df_metrics_config.items():
        # Check if indicators are present
        if 'indicators' in value:
            for indicator, indicator_data in value['indicators'].items():
                # Create new top-level key: original key + indicator name
                new_key = f"{key}_{indicator}"
                df_metrics_indicators_config[new_key] = indicator_data

            # Optionally, remove indicators from the original key
            df_metrics_indicators_config[key].pop('indicators')

    return df_metrics_indicators_config




def bucketize_value(value, buckets):
    """
    Categorizes a value based on the defined buckets.

    Params:
    - value (float): The value to categorize
    - buckets (list): List of dictionaries defining the buckets

    Returns:
    - str: The category the value falls into
    """
    for bucket in buckets:
        for category, threshold in bucket.items():
            if threshold == "remainder" or value <= threshold:
                return category

    # This should never be reached if buckets are properly defined
    raise ValueError(f"Value {value} does not fit in any defined bucket.")



def calculate_rolling_window_features(
        ts,
        window_duration,
        lookback_periods,
        rolling_aggregations,
        comparisons,
        metric_name
    ):
    """
    Calculates rolling window features and comparisons for a given time series based on
    configurable window duration and lookback periods.

    Params:
    - ts (pd.Series): The time series data for the metric.
    - window_duration (int): The size of each rolling window (e.g., 7 for 7 days).
    - lookback_periods (int): The number of lookback periods to calculate (e.g., 2 for two periods).
    - rolling_aggregations (list): The aggregations for each rolling window (e.g. ['sum', 'max']).
    - comparisons (list): The comparisons to make between the first and last value in the window.
    - metric_name (str): The name of the metric to include in the feature names.

    Returns:
    - features (dict): A dictionary containing calculated rolling window features.
    """

    features = {}  # Initialize an empty dictionary to store the rolling window features

    # Calculate the total number of complete periods that fit within the time series
    num_complete_periods = len(ts) // window_duration

    # Ensure we're not calculating more periods than we have data for
    actual_lookback_periods = min(lookback_periods, num_complete_periods)

    # Start processing from the last complete period, moving backwards
    for i in range(actual_lookback_periods):
        # Define the start and end of the current rolling window
        end_period = len(ts) - i * window_duration
        start_period = end_period - window_duration

        # Ensure that the start index is not out of bounds
        if start_period >= 0:
            rolling_window = ts.iloc[start_period:end_period]

            # Loop through each statistic to calculate for the rolling window
            for agg in rolling_aggregations:
                agg_key = f'{metric_name}_{agg}_{window_duration}d_period_{i+1}'
                features[agg_key] = calculate_aggregation(rolling_window, agg)

            # If the rolling window has enough data, calculate comparisons
            if len(rolling_window) > 0:
                for comparison in comparisons:
                    comparison_key = f'{metric_name}_{comparison}_{window_duration}d_period_{i+1}'
                    features[comparison_key] = calculate_comparisons(rolling_window, comparison)

    return features  # Return the dictionary of rolling window features



def calculate_adj_pct_change(start_value, end_value, cap=5, impute_value=1):
    """
    Calculates the adjusted percentage change between two values.
    Handles cases where the start_value is 0 by imputing a value and capping extreme
    percentage changes.

    Params:
    - start_value (float): The value at the beginning of the period.
    - end_value (float): The value at the end of the period.
    - cap (float): The maximum percentage change allowed (default = 1000%).
    - impute_value (float): The value to impute when the start_value is 0 (default = 1).

    Returns:
    - pct_change (float): The calculated or capped percentage change.
    """
    if start_value == 0:
        if end_value == 0:
            return 0  # 0/0 case
        else:
            # Use imputed value to maintain scale of increase
            return min((end_value / impute_value) - 1, cap)
    else:
        pct_change = (end_value / start_value) - 1
        return min(pct_change, cap)  # Cap the percentage change if it's too large


def calculate_aggregation(ts, stat):
    """
    Centralized function to calculate various aggregations on a time series.

    Parameters:
    - ts (pd.Series): The time series to aggregate.
    - stat (str): The type of aggregation to perform.

    Returns:
    - The calculated aggregation value.
    """
    agg_functions = {
        'sum': ts.sum,
        'mean': ts.mean,
        'median': ts.median,
        'std': ts.std,
        'max': ts.max,
        'min': ts.min,
        'first': lambda: ts.iloc[0],
        'last': lambda: ts.iloc[-1]
    }

    if stat not in agg_functions:
        raise KeyError(f"Unsupported aggregation type: '{stat}'.")

    return agg_functions[stat]()


# Helper function for comparison metric formulas
def calculate_comparisons(rolling_window, comparison):
    """
    helper function that calculates the comparison metric for the specific rolling window.
    this helps make the for loop more readable.

    params:
        - rolling_window (pd.Series): the metric series for the given window only
        - comparison (string): the type of comparison calculation to perform
    """
    if comparison == 'change':
        return rolling_window.iloc[-1] - rolling_window.iloc[0]
    elif comparison == 'pct_change':
        start_value, end_value = rolling_window.iloc[0], rolling_window.iloc[-1]
        return calculate_adj_pct_change(start_value, end_value)


def save_flattened_outputs(flattened_df, output_dir, metric_description, modeling_period_start):
    """
    Saves the flattened DataFrame with descriptive metrics into a CSV file.

    Params:
    - flattened_df (pd.DataFrame): The DataFrame containing flattened data.
    - output_dir (str): Directory where the CSV file will be saved.
    - metric_description (str): Description of metrics (e.g., 'buysell_metrics').
    - modeling_period_start (str): Start of the modeling period (e.g., '2023-01-01').
    - description (str, optional): A description to be added to the filename.

    Returns:
    - flattened_df (pd.DataFrame): The same DataFrame that was passed in.
    - output_path (str): The full path to the saved CSV file.
    """

    # if the coin_id column exists, confirm it is fully unique
    if 'coin_id' in flattened_df.columns:
        if not flattened_df['coin_id'].is_unique:
            raise ValueError("The 'coin_id' column must have fully unique values.")

    # Define filename with metric description and optional description
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    filename = f"{metric_description}_{timestamp}_model_period_{modeling_period_start}.csv"

    # Save file
    output_path = os.path.join(output_dir, filename)
    flattened_df.to_csv(output_path, index=False)

    logger.debug("Saved flattened outputs to %s", output_path)

    return flattened_df, output_path



def preprocess_coin_df(input_path, modeling_config, dataset_config, df_metrics_config=None):
    """
    Preprocess the flattened coin DataFrame by applying feature selection based on the modeling
    and dataset-specific configs, and optional scaling based on the metrics config.

    Params:
    - input_path (str): Path to the flattened CSV file.
    - modeling_config (dict): Configuration with modeling-specific parameters.
    - dataset_config (dict): Dataset-specific configuration (e.g., sharks, coin_metadata).
    - df_metrics_config (dict, optional): Configuration for scaling metrics, can be None if scaling
        is not required.

    Returns:
    - df (pd.DataFrame): The preprocessed DataFrame.
    - output_path (str): The full path to the saved preprocessed CSV file.
    """

    # Step 1: Load and Validate Data
    # ----------------------------------------------------
    df = pd.read_csv(input_path)

    # Check for missing values and raise an error if any are found
    if df.isnull().values.any():
        raise ValueError("Missing values detected in the DataFrame.")


    # Step 2: Convert categorical and boolean columns to integers
    # ---------------------------------------------------------------
    # Convert categorical columns to one-hot encoding (get_dummies)
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    categorical_columns = [col for col in categorical_columns if col != 'coin_id']
    for col in categorical_columns:
        num_categories = df[col].nunique()
        if num_categories > 8:
            logger.warning("Column '%s' has %s categories, consider reducing categories.",
                           col, num_categories)
        df = pd.get_dummies(df, columns=[col], drop_first=True)


    # Convert boolean columns to integers
    df = df.apply(lambda col: col.astype(int) if col.dtype == bool else col)


    # Step 3: Feature Selection Based on Config
    # ----------------------------------------------------
    # Drop features specified in modeling_config['drop_features']
    drop_features = modeling_config['preprocessing'].get('drop_features', [])
    if drop_features:
        df = df.drop(columns=drop_features, errors='ignore')

    # Apply feature selection based on sameness_threshold and retain_columns from dataset_config
    sameness_threshold = dataset_config.get('sameness_threshold', 1.0)
    retain_columns = dataset_config.get('retain_columns', [])

    # Drop columns with more than `sameness_threshold` of the same value, unless in retain_columns
    for column in df.columns:
        if column not in retain_columns:
            max_value_ratio = df[column].value_counts(normalize=True).max()
            if max_value_ratio > sameness_threshold:
                df = df.drop(columns=[column])
                logger.debug("Dropped column %s due to sameness_threshold", column)


    # Step 4: Scaling and Transformation
    # ----------------------------------------------------
    # Apply scaling if df_metrics_config is provided
    if df_metrics_config:
        df = apply_scaling(df, df_metrics_config)


    # Step 5: Save and Log Preprocessed Data
    # ----------------------------------------------------
    # Generate output path and filename based on input
    base_filename = os.path.basename(input_path).replace(".csv", "")
    output_filename = f"{base_filename}_preprocessed.csv"
    output_path = os.path.join(
        os.path.dirname(input_path).replace("flattened_outputs", "preprocessed_outputs"),
        output_filename
    )

    # Save the preprocessed data
    df.to_csv(output_path, index=False)

    # Log the changes made
    logger.debug("Preprocessed file saved at: %s", output_path)

    return df, output_path



def apply_scaling(df, df_metrics_config):
    """
    Apply scaling to the relevant columns in the DataFrame based on the metrics config.

    Params:
    - df (pd.DataFrame): The DataFrame to scale.
    - df_metrics_config (dict): The input file's configuration with metrics and their scaling
        methods, aggregations, etc.

    Returns:
    - df (pd.DataFrame): The scaled DataFrame.
    """
    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler()
    }

    # Loop through each metric and its settings in the df_metrics_config
    for metric, settings in df_metrics_config.items():

        # if there are no aggregations for the metric, continue
        if 'aggregations' not in settings.keys():
            continue
        for agg, agg_settings in settings['aggregations'].items():
            # Ensure agg_settings exists and is not None
            if not agg_settings:
                continue

            column_name = f"{metric}_{agg}"
            if column_name in df.columns:
                scaling_method = agg_settings.get('scaling', None)

                # Skip scaling if set to "none" or if no scaling is provided
                if scaling_method is None or scaling_method == "none":
                    continue

                if scaling_method == "log":
                    # Apply log1p scaling (log(1 + x)) to avoid issues with zero values
                    df[[column_name]] = np.log1p(df[[column_name]])
                elif scaling_method in scalers:
                    scaler = scalers[scaling_method]
                    df[[column_name]] = scaler.fit_transform(df[[column_name]])
                else:
                    logger.info("Unknown scaling method %s for column %s",
                                scaling_method, column_name)

    return df



def create_training_data_df(
    modeling_folder: str,
    input_file_tuples: list[tuple[str, str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merges specified preprocessed output CSVs into a single DataFrame, applies fill strategies,
    and ensures consistency of coin_ids across datasets. Adds suffixes to column names to
    avoid duplicates.

    Additionally, raises an error if any of the input files have duplicate coin_ids or are missing
    the coin_id column.

    Params:
    - modeling_folder (str): Location of the parent modeling folder.
    - input_file_tuples (list of tuples): List of tuples where each tuple contains:
        - filename (str): The name of the CSV file to process.
        - fill_strategy (str): Strategy for handling missing values ('fill_zeros', 'drop_records').

    Returns:
    - training_data_df (pd.DataFrame): Merged DataFrame with all specified preprocessed and all
        fill strategies applied.
    - merge_logs_df (pd.DataFrame): Log DataFrame detailing record counts for each input DataFrame.
    """
    # Initialize location of the preprocessed_outputs directory
    input_directory = os.path.join(modeling_folder,'outputs/preprocessed_outputs/')

    # Initialize an empty list to hold DataFrames
    df_list = []
    missing_files = []

    # Regex to extract the date pattern %Y-%m-%d_%H-%M from the filename
    date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}')

    # Dictionary to track how many times each column name has been used
    column_suffix_count = {}

    # Count occurrences of each metric_string
    metric_string_count = {}

    # First loop to count how often each metric_string appears
    for filename, _ in input_file_tuples:
        match = date_pattern.search(filename)
        if not match:
            raise ValueError(f"No valid date string found in the filename: {filename}")

        date_string = match.group()
        metric_string = filename.split(date_string)[0].rstrip('_')

        if metric_string not in metric_string_count:
            metric_string_count[metric_string] = 1
        else:
            metric_string_count[metric_string] += 1

    # Loop through the input_file_tuples (filename, fill_strategy)
    for filename, fill_strategy in input_file_tuples:
        file_path = os.path.join(input_directory, filename)

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

            # Check if coin_id column exists if the fill strategy requires it
            if fill_strategy in ['drop_records','fill_zeros']:
                if 'coin_id' not in df.columns:
                    raise ValueError(f"coin_id column is missing in {filename}")

                # Raise an error if there are duplicate coin_ids in the file
                if df['coin_id'].duplicated().any():
                    raise ValueError(f"Duplicate coin_ids found in file: {filename}")

            # Extract the date string from the filename
            match = date_pattern.search(filename)
            if not match:
                raise ValueError(f"No valid date string found in the filename: {filename}")
            date_string = match.group()  # e.g., '2024-09-13_14-44'
            metric_string = filename.split(date_string)[0].rstrip('_')

            # Add column suffixes based on the count of metric_string
            if metric_string_count[metric_string] > 1:
                suffix = f"{metric_string}_{date_string}"
            else:
                suffix = metric_string

            # Check if this suffix has been used before and append a numerical suffix if necessary
            for column in df.columns:
                if column != 'coin_id':
                    column_with_suffix = f"{column}_{suffix}"

                    # Check if suffix exists in the count, increment if necessary
                    if column_with_suffix in column_suffix_count:
                        column_suffix_count[column_with_suffix] += 1
                        suffix_count = column_suffix_count[column_with_suffix]
                        column_with_suffix = f"{column_with_suffix}_{suffix_count}"
                    else:
                        column_suffix_count[column_with_suffix] = 1

                    df = df.rename(columns={column: column_with_suffix})

            # Append DataFrame and fill strategy to the list for processing
            df_list.append((df, fill_strategy, filename))
        else:
            missing_files.append(filename)

    # Merge the output DataFrames based on their fill strategies
    training_data_df, merge_logs_df = merge_and_fill_training_data(df_list)

    # Log the results
    logger.debug("%d files were successfully merged into training_data_df.", len(df_list))
    if missing_files:
        logger.warning("%d files could not be found: %s",
                        len(missing_files), ', '.join(missing_files))
    else:
        logger.debug("All specified files were found and merged into training_data_df.")

    return training_data_df, merge_logs_df



def merge_and_fill_training_data(
    df_list: list[tuple[pd.DataFrame, str, str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merges a list of DataFrames on 'coin_id' and applies specified fill strategies for missing
    values. Generates a log of the merging process with details on record counts and modifications.

    Params:
    - df_list (list of tuples): Each tuple contains:
        - df (pd.DataFrame): A DataFrame to merge.
        - fill_strategy (str): The strategy to handle missing values ('fill_zeros', 'drop_records').
        - filename (str): The name of the input file, used for logging.

    Returns:
    - training_data_df (pd.DataFrame): Merged DataFrame with applied fill strategies.
    - merge_logs_df (pd.DataFrame): Log DataFrame detailing record counts for each input DataFrame.
    """
    if not df_list:
        raise ValueError("No DataFrames to merge.")

    # Initialize the log DataFrame
    merge_logs = []

    # Pull a unique set of all coin_ids across all DataFrames
    all_coin_ids = set()
    for df, _, _ in df_list:
        if 'coin_id' in df.columns:
            all_coin_ids.update(df['coin_id'].unique())

    # Start merging with the full set of coin_ids in a DataFrame
    training_data_df = pd.DataFrame(all_coin_ids, columns=['coin_id'])

    # Iterate through df_list and merge each one
    for df, fill_strategy, filename in df_list:

        # if the df is a macro_series without a coin_id, cross join it to all coin_ids
        if fill_strategy == 'expand':
            original_coin_ids = set()
            training_data_df = training_data_df.merge(df, how='cross')

        else:
            # Merge with the full coin_id set (outer join)
            original_coin_ids = set(df['coin_id'].unique())  # Track original coin_ids
            training_data_df = pd.merge(training_data_df, df, on='coin_id', how='outer')

            # Apply the fill strategy
            if fill_strategy == 'fill_zeros':
                # Fill missing values with 0
                training_data_df.fillna(0, inplace=True)
            elif fill_strategy == 'drop_records':
                # Drop rows with missing values for this DataFrame's columns
                training_data_df.dropna(inplace=True)
            else:
                raise ValueError(f"Invalid fill strategy '{fill_strategy}' found in config.yaml.")

        # Calculate log details
        final_coin_ids = set(training_data_df['coin_id'].unique())
        filled_ids = final_coin_ids - original_coin_ids  # Present in final, missing in original

        # Add log information for this DataFrame
        merge_logs.append({
            'file': filename,  # Use filename for logging
            'original_count': len(original_coin_ids),
            'filled_count': len(filled_ids)
        })

    # Ensure no duplicate columns after merging
    if training_data_df.columns.duplicated().any():
        raise ValueError("Duplicate columns found after merging.")

    # Raise an error if there are any null values in the final DataFrame
    if training_data_df.isnull().any().any():
        raise ValueError("Null values detected in the final merged DataFrame.")

    # Convert logs to a DataFrame
    merge_logs_df = pd.DataFrame(merge_logs)

    return training_data_df, merge_logs_df


def prepare_and_compute_performance(prices_df, training_data_config):
    """
    Prepares the data and computes price performance for each coin.

    Parameters:
    - prices_df: DataFrame containing price data with columns 'coin_id', 'date', and 'price'.
    - training_data_config: Configuration with modeling period dates.

    Returns:
    - performance_df: DataFrame with columns 'coin_id' and 'performance'.
    - outcomes_df: DataFrame tracking outcomes for each coin.
    """
    prices_df = prices_df.copy()
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    modeling_period_start = pd.to_datetime(training_data_config['modeling_period_start'])
    modeling_period_end = pd.to_datetime(training_data_config['modeling_period_end'])

    # Filter data for start and end dates
    start_prices = prices_df[prices_df['date'] == modeling_period_start].set_index('coin_id')['price']
    end_prices = prices_df[prices_df['date'] == modeling_period_end].set_index('coin_id')['price']

    # Identify coins with both start and end prices
    valid_coins = start_prices.index.intersection(end_prices.index)

    # Check for missing data
    all_coins = prices_df['coin_id'].unique()
    coins_missing_price = set(all_coins) - set(valid_coins)

    if coins_missing_price:
        missing = ', '.join(map(str, coins_missing_price))
        raise ValueError(f"Missing price for coins at start or end date: {missing}")

    # Compute performance
    performance = (end_prices[valid_coins] - start_prices[valid_coins]) / start_prices[valid_coins]
    performance_df = pd.DataFrame({'coin_id': valid_coins,
                                   'performance': performance}
                                   ).reset_index(drop=True)

    # Create outcomes DataFrame
    outcomes_df = pd.DataFrame({
        'coin_id': valid_coins,
        'outcome': 'performance calculated'
    })

    return performance_df, outcomes_df



def create_target_variables_mooncrater(prices_df, training_data_config, modeling_config):
    """
    Creates a DataFrame with target variables 'is_moon' and 'is_crater' based on price performance
    during the modeling period, using the thresholds from modeling_config.

    Parameters:
    - prices_df: DataFrame containing price data with columns 'coin_id', 'date', and 'price'.
    - config: General configuration file with modeling period dates.
    - modeling_config: Configuration for modeling with target variable thresholds.

    Returns:
    - target_variable_df: DataFrame with columns 'coin_id', 'is_moon', and 'is_crater'.
    - outcomes_df: DataFrame tracking outcomes for each coin.
    """
    # 1. Retrieve Variables and Prepare DataFrame
    # -------------------------------------------
    prices_df = prices_df.copy()
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    modeling_period_start = pd.to_datetime(training_data_config['modeling_period_start'])
    modeling_period_end = pd.to_datetime(training_data_config['modeling_period_end'])

    modeling_period_df = prices_df.loc[
        prices_df['date'].between(modeling_period_start, modeling_period_end)
    ]

    # Raise an exception if any coins are missing a price at the start or end date
    start_price_coins = modeling_period_df[
        modeling_period_df['date'] == modeling_period_start
    ]['coin_id'].unique()

    end_price_coins = modeling_period_df[
        modeling_period_df['date'] == modeling_period_end
    ]['coin_id'].unique()

    all_coins = modeling_period_df['coin_id'].unique()
    coins_missing_price = set(all_coins) - set(start_price_coins) - set(end_price_coins)

    if coins_missing_price:
        missing = ', '.join(map(str, coins_missing_price))
        raise ValueError(f"Missing price for coins at start or end date: {missing}")


    # 2. Generate Target Variables
    # ----------------------------
    moon_threshold = modeling_config['target_variables']['moon_threshold']
    crater_threshold = modeling_config['target_variables']['crater_threshold']
    moon_minimum_percent = modeling_config['target_variables']['moon_minimum_percent']
    crater_minimum_percent = modeling_config['target_variables']['crater_minimum_percent']

    # Process coins with data
    target_data = []
    outcomes = []
    price_performance = []
    for coin_id, group in modeling_period_df.groupby('coin_id', observed=True):
        # Get the price on the start and end dates
        price_start = group[group['date'] == modeling_period_start]['price'].values
        price_end = group[group['date'] == modeling_period_end]['price'].values

        # Check if both start and end prices exist
        if len(price_start) > 0 and len(price_end) > 0:
            price_start = price_start[0]
            price_end = price_end[0]
            performance = (price_end - price_start) / price_start
            is_moon = int(performance >= moon_threshold)
            is_crater = int(performance <= crater_threshold)
            target_data.append({'coin_id': coin_id, 'is_moon': is_moon, 'is_crater': is_crater})
            outcomes.append({'coin_id': coin_id, 'outcome': 'target variable created'})
            price_performance.append({'coin_id': coin_id, 'performance': performance})

        else:
            if len(price_start) == 0 and len(price_end) == 0:
                outcomes.append({'coin_id': coin_id, 'outcome': 'missing both'})
            elif len(price_start) == 0:
                outcomes.append({'coin_id': coin_id, 'outcome': 'missing start price'})
            else:
                outcomes.append({'coin_id': coin_id, 'outcome': 'missing end price'})


    # 3. Ensure minimum percentage for moon and crater
    # ------------------------------------------------
    target_variables_df = pd.DataFrame(target_data)
    price_performance_df = pd.DataFrame(price_performance)

    # Sort by performance for filling the remaining moons and craters
    price_performance_df = price_performance_df.sort_values(by='performance', ascending=False)

    total_coins = len(target_variables_df)
    moons = target_variables_df[target_variables_df['is_moon'] == 1].shape[0]
    craters = target_variables_df[target_variables_df['is_crater'] == 1].shape[0]

    # Check if moons meet minimum percentage
    if moons / total_coins < moon_minimum_percent:
        additional_moons_needed = int(total_coins * moon_minimum_percent) - moons
        # Mark additional top-performing coins as moons
        moon_candidates = price_performance_df[~price_performance_df['coin_id'].isin(  # pylint: disable=E1136
            target_variables_df[target_variables_df['is_moon'] == 1]['coin_id']
            )].head(additional_moons_needed)
        target_variables_df.loc[
            target_variables_df['coin_id'].isin(moon_candidates['coin_id']), 'is_moon'
            ] = 1

    # Check if craters meet minimum percentage
    if craters / total_coins < crater_minimum_percent:
        additional_craters_needed = int(total_coins * crater_minimum_percent) - craters
        # Mark additional worst-performing coins as craters
        crater_candidates = price_performance_df[~price_performance_df['coin_id'].isin(  # pylint: disable=E1136
            target_variables_df[target_variables_df['is_crater'] == 1]['coin_id']
            )].tail(additional_craters_needed)
        target_variables_df.loc[
            target_variables_df['coin_id'].isin(crater_candidates['coin_id']), 'is_crater'
            ] = 1


    # 4. Log and Format Output DataFrame
    # ----------------------------------
    outcomes_df = pd.DataFrame(outcomes)

    logger.info(
        "Target variables created for %s coins with %s/%s (%s) moons and %s/%s (%s) craters.",
        len(target_variables_df),
        target_variables_df[target_variables_df['is_moon'] == 1].shape[0], total_coins,
        dc.human_format(
            100 * target_variables_df[target_variables_df['is_moon'] == 1].shape[0] / total_coins
            ) + '%',
        target_variables_df[target_variables_df['is_crater'] == 1].shape[0], total_coins,
        dc.human_format(
            100 * target_variables_df[target_variables_df['is_crater'] == 1].shape[0] / total_coins
            ) + '%'
    )

    return target_variables_df, outcomes_df



def prepare_model_input_df(training_data_df, target_variable_df, target_column):
    """
    Prepares the final model input DataFrame by merging the training data with the target variables
    on 'coin_id' and selects the specified target column. Checks for data quality issues such as
    missing columns, duplicate coin_ids, and missing target variables.

    Parameters:
    - training_data_df: DataFrame containing the features for training the model.
    - target_variable_df: DataFrame containing target variables for each coin_id.
    - target_column: The name of the target variable to train on (e.g., 'is_moon' or 'is_crater').

    Returns:
    - model_input_df: Merged DataFrame with both features and the specified target variable.
    """

    # Step 1: Ensure that both DataFrames have 'coin_id' as a column
    if 'coin_id' not in training_data_df.columns:
        raise ValueError("The 'coin_id' column is missing in training_data_df")
    if 'coin_id' not in target_variable_df.columns:
        raise ValueError("The 'coin_id' column is missing in target_variable_df")

    # Step 2: Check for duplicated coin_id entries in both DataFrames
    if training_data_df['coin_id'].duplicated().any():
        raise ValueError("Duplicate 'coin_id' found in training_data_df")
    if target_variable_df['coin_id'].duplicated().any():
        raise ValueError("Duplicate 'coin_id' found in target_variable_df")

    # Step 3: Ensure that the target column exists in the target_variable_df
    if target_column not in target_variable_df.columns:
        raise ValueError(f"The target column '{target_column}' is missing in target_variable_df")

    # Step 4: Merge the training data with the target variable DataFrame on 'coin_id'
    model_input_df = pd.merge(training_data_df, target_variable_df[['coin_id', target_column]],
                              on='coin_id', how='inner')

    # Step 5: Remove coins without target variables and output a warning with the number removed
    removed_coins = len(training_data_df) - len(model_input_df)
    if removed_coins > 0:
        logger.warning("%s coins were removed due to missing target variables", removed_coins)

    # Step 6: Perform final quality checks (e.g., no NaNs in important columns)
    if model_input_df.isnull().any().any():
        logger.warning("NaN values found in the merged DataFrame")

    # Step 7: Return the final model input DataFrame
    return model_input_df
