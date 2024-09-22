"""
functions used to build coin-level features from training data
"""
import os
from datetime import datetime
import time
import re
import pandas as pd
import dreams_core.core as dc
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# set up logger at the module level
logger = dc.setup_logger()


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
        full_date_range = pd.to_datetime(pd.date_range(start=coin_df['date'].min()
                                                       , end=training_period_end)).to_pydatetime()

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
    logger.info("Flattening columns %s into coin-level features...", configured_metrics)

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

    logger.info('Flattened input df into coin-level features with shape %s after %.2f seconds.',
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
    - ValueError: If an expected column (e.g., a metric) is missing from the DataFrame.
    """
    flat_features = {}

    # Apply global stats calculations for each metric
    for metric, config in df_metrics_config.items():
        if metric not in time_series_df.columns:
            raise ValueError(f"Metric '{metric}' is missing from the input DataFrame.")

        ts = time_series_df[metric].copy()  # Get the time series for this metric

        # Standard aggregations
        if 'aggregations' in config:
            for agg in config['aggregations']:
                flat_features[f'{metric}_{agg}'] = calculate_stat(ts, agg)

        # Rolling window calculations
        rolling = config.get('rolling', False)
        if rolling:
            rolling_stats = config['rolling']['stats']
            comparisons = config['rolling'].get('comparisons', [])
            window_duration = config['rolling']['window_duration']
            lookback_periods = config['rolling']['lookback_periods']

            # Calculate rolling metrics and update flat_features
            rolling_features = calculate_rolling_window_features(
                ts, window_duration, lookback_periods, rolling_stats, comparisons, metric)
            flat_features.update(rolling_features)

    return flat_features


def calculate_rolling_window_features(
        ts,
        window_duration,
        lookback_periods,
        rolling_stats,
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
    - rolling_stats (list): The metrics to calculate for each rolling window (e.g. ['sum', 'max']).
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

    # Start processing from the last complete period, moving backwards
    for i in range(actual_lookback_periods):
        # Define the start and end of the current rolling window
        end_period = len(ts) - i * window_duration
        start_period = end_period - window_duration

        # Ensure that the start index is not out of bounds
        if start_period >= 0:
            rolling_window = ts.iloc[start_period:end_period]

            # Loop through each statistic to calculate for the rolling window
            for stat in rolling_stats:
                stat_key = f'{metric_name}_{stat}_{window_duration}d_period_{i+1}'
                features[stat_key] = calculate_stat(rolling_window, stat)

            # If the rolling window has enough data, calculate comparisons
            if len(rolling_window) > 0:
                for comparison in comparisons:
                    comparison_key = f'{metric_name}_{comparison}_{window_duration}d_period_{i+1}'
                    features[comparison_key] = calculate_comparisons(rolling_window, comparison)

    return features  # Return the dictionary of rolling window features



def calculate_adj_pct_change(start_value, end_value, cap=1000, impute_value=1):
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
            return min((end_value / impute_value - 1) * 100, cap)
    else:
        pct_change = (end_value / start_value - 1) * 100
        return min(pct_change, cap)  # Cap the percentage change if it's too large



def calculate_global_stats(ts, metric_name, config):
    """
    Calculates global statistics for a given time series based on the configuration.

    Params:
    - ts (pd.Series): The time series data for the metric.
    - metric_name (str): The name of the metric (e.g., 'buyers_new').
    - config (dict): The configuration object that specifies which statistics to calculate
                     for each metric.

    Returns:
    - stats (dict): A dictionary containing calculated statistics. The keys are in the format
                    of '{metric_name}_{aggregation}' (e.g., 'buyers_new_sum').

    Example:
    If the config specifies 'sum' and 'mean' for 'buyers_new', the result will include:
    {
        'buyers_new_sum': value,
        'buyers_new_mean': value
    }
    """
    stats = {}  # Initialize an empty dictionary to hold the calculated stats.

    # Get the aggregation functions for the given metric from the config.
    metric_config = config['metrics'].get(metric_name, [])

    # Loop through each aggregation function and calculate the stat.
    for agg in metric_config:
        # Use the helper function 'calculate_stat' to calculate each aggregation.
        stats[f'{metric_name}_{agg}'] = calculate_stat(ts, agg)

    return stats  # Return the dictionary of calculated statistics.



def calculate_stat(ts, stat):
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


def save_flattened_outputs(coin_df, output_dir, metric_description, modeling_period_start):
    """
    Saves the flattened DataFrame with descriptive metrics into a CSV file.

    Params:
    - coin_df (pd.DataFrame): The DataFrame containing flattened data.
    - output_dir (str): Directory where the CSV file will be saved.
    - metric_description (str): Description of metrics (e.g., 'buysell_metrics').
    - modeling_period_start (str): Start of the modeling period (e.g., '2023-01-01').
    - description (str, optional): A description to be added to the filename.

    Returns:
    - coin_df (pd.DataFrame): The same DataFrame that was passed in.
    - output_path (str): The full path to the saved CSV file.
    """

    # Check if 'coin_id' exists and is fully unique
    if 'coin_id' not in coin_df.columns:
        raise ValueError("The DataFrame must contain a 'coin_id' column.")

    if not coin_df['coin_id'].is_unique:
        raise ValueError("The 'coin_id' column must have fully unique values.")

    # Define filename with metric description and optional description
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    filename = f"{metric_description}_{timestamp}_model_period_{modeling_period_start}.csv"

    # Save file
    output_path = os.path.join(output_dir, filename)
    coin_df.to_csv(output_path, index=False)

    logger.debug("Saved flattened outputs to %s", output_path)

    return coin_df, output_path



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


    # Step 2: Preprocess Data
    # ----------------------------------------------------
    # Convert boolean columns to integers
    df = df.apply(lambda col: col.astype(int) if col.dtype == bool else col)

    # Apply feature selection based on drop_features from modeling_config
    drop_features = modeling_config['preprocessing'].get('drop_features', [])
    if drop_features:
        df = df.drop(columns=drop_features, errors='ignore')


    # Step 3: Feature Selection Based on Config
    # ----------------------------------------------------
    # Apply feature selection based on sameness_threshold and retain_columns from dataset_config
    sameness_threshold = dataset_config.get('sameness_threshold', 1.0)
    retain_columns = dataset_config.get('retain_columns', [])

    # Drop columns with more than `sameness_threshold` of the same value, unless in retain_columns
    for column in df.columns:
        if column not in retain_columns:
            max_value_ratio = df[column].value_counts(normalize=True).max()
            if max_value_ratio >= sameness_threshold:
                df = df.drop(columns=[column])
                logger.info("Dropped column %s due to sameness_threshold", column)


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
    logger.info("Preprocessed file saved at: %s", output_path)

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

                if scaling_method in scalers:
                    scaler = scalers[scaling_method]
                    df[[column_name]] = scaler.fit_transform(df[[column_name]])
                else:
                    logger.info("Unknown scaling method %s for column %s",
                                scaling_method, column_name)

    return df



def create_training_data_df(
    modeling_folder: str,
    input_file_tuples: list[tuple[str, str]]
) -> pd.DataFrame:
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
        print(file_path)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(df.shape)
            # Check if coin_id column exists; raise an error if missing
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
    logger.info("%d files were successfully merged.", len(df_list))
    if missing_files:
        logger.info("%d files could not be found: %s", len(missing_files), ', '.join(missing_files))
    else:
        logger.info("All specified files were found and merged successfully.")

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
        all_coin_ids.update(df['coin_id'].unique())

    # Convert set of coin_ids to a DataFrame
    all_coin_ids_df = pd.DataFrame(all_coin_ids, columns=['coin_id'])

    # Start merging with the full set of coin_ids
    training_data_df = all_coin_ids_df

    # Iterate through df_list and merge each one
    for df, fill_strategy, filename in df_list:
        original_coin_ids = set(df['coin_id'].unique())  # Track original coin_ids

        # Merge with the full coin_id set (outer join)
        training_data_df = pd.merge(training_data_df, df, on='coin_id', how='outer')

        # Apply the fill strategy
        if fill_strategy == 'fill_zeros':
            # Fill missing values with 0
            training_data_df.fillna(0, inplace=True)
        elif fill_strategy == 'drop_records':
            # Drop rows with missing values for this DataFrame's columns
            training_data_df.dropna(inplace=True)
        else:
            raise ValueError("Invalid fill strategy listed. Valid strategies include "
                             "['fill_zeros','drop_records']")

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
    # Retrieve the necessary values from modeling_config
    modeling_period_start = pd.to_datetime(training_data_config['modeling_period_start'])
    modeling_period_end = pd.to_datetime(training_data_config['modeling_period_end'])
    moon_threshold = modeling_config['target_variables']['moon_threshold']
    crater_threshold = modeling_config['target_variables']['crater_threshold']

    # Filter for the modeling period and sort the DataFrame
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    modeling_period_df = prices_df[
        (prices_df['date'] >= modeling_period_start) &
        (prices_df['date'] <= modeling_period_end)
    ]
    modeling_period_df = modeling_period_df.sort_values(by=['coin_id', 'date'])

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

    # Process coins with data
    target_data = []
    outcomes = []
    for coin_id, group in modeling_period_df.groupby('coin_id'):
        # Get the price on the start and end dates
        price_start = group[group['date'] == modeling_period_start]['price'].values
        price_end = group[group['date'] == modeling_period_end]['price'].values

        # Check if both start and end prices exist
        if len(price_start) > 0 and len(price_end) > 0:
            price_start = price_start[0]
            price_end = price_end[0]
            is_moon = int(price_end >= (1 + moon_threshold) * price_start)
            is_crater = int(price_end <= (1 + crater_threshold) * price_start)
            target_data.append({'coin_id': coin_id, 'is_moon': is_moon, 'is_crater': is_crater})
            outcomes.append({'coin_id': coin_id, 'outcome': 'target variable created'})

        else:
            if len(price_start) == 0 and len(price_end) == 0:
                outcomes.append({'coin_id': coin_id, 'outcome': 'missing both'})
            elif len(price_start) == 0:
                outcomes.append({'coin_id': coin_id, 'outcome': 'missing start price'})
            else:
                outcomes.append({'coin_id': coin_id, 'outcome': 'missing end price'})

    # Log outcomes for coins with no data in the modeling period
    all_coins = set(prices_df['coin_id'].unique())
    modeling_period_coins = set(modeling_period_df['coin_id'].unique())
    coins_with_no_data = all_coins - modeling_period_coins
    for coin_id in coins_with_no_data:
        outcomes.append({'coin_id': coin_id, 'outcome': 'missing both'})

    # Convert target data and outcomes to DataFrames
    target_variables_df = pd.DataFrame(target_data)
    outcomes_df = pd.DataFrame(outcomes)

    # Log summary based on outcomes
    target_variable_count = len(outcomes_df[outcomes_df['outcome'] == 'target variable created'])
    moons = target_variables_df[target_variables_df['is_moon'] == 1].shape[0]
    craters = target_variables_df[target_variables_df['is_crater'] == 1].shape[0]

    logger.info(
        "Target variables created for %s coins with %s/%s (%s) moons and %s/%s (%s) craters.",
        target_variable_count,
        moons, target_variable_count, dc.human_format(100 * moons / target_variable_count) + '%',
        craters, target_variable_count, dc.human_format(100 * craters / target_variable_count) + '%'
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

    # Step 5: Check if any coin_id from training_data_df is missing a target variable
    missing_targets = set(training_data_df['coin_id']) - set(target_variable_df['coin_id'])
    if missing_targets:
        logger.warning("Some 'coin_id's are missing target variables: %s"
                       , ', '.join(map(str, missing_targets)))

    # Step 6: Perform final quality checks (e.g., no NaNs in important columns)
    if model_input_df.isnull().any().any():
        logger.warning("NaN values found in the merged DataFrame")

    # Step 7: Return the final model input DataFrame
    return model_input_df
