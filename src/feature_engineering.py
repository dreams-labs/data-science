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
        flat_features = flatten_coin_features(coin_df, df_metrics_config)
        all_flat_features.append(flat_features)

    # Convert the list of feature dictionaries into a DataFrame
    flattened_df = pd.DataFrame(all_flat_features)

    logger.info('Flattened input df into coin-level features with shape %s after %.2f seconds.'
                , flattened_df.shape, time.time() - start_time)


    return flattened_df



def flatten_coin_features(coin_df, df_metrics_config):
    """
    Flattens all relevant time series metrics for a single coin into a row of features.

    Params:
    - coin_df (pd.DataFrame): DataFrame with time series data for a single coin (coin_id-date).
    - df_metrics_config (dict): Configuration object with metric rules from the metrics file for
        the specific input df.

    Returns:
    - flat_features (dict): A dictionary of flattened features for the coin.

    Raises:
    - KeyError: If a metric in the DataFrame is not found in the configuration.
    - ValueError: If an expected column (e.g., a metric) is missing from the DataFrame.
    """
    # Check if the required 'coin_id' column is present
    if 'coin_id' not in coin_df.columns:
        raise ValueError("The input DataFrame is missing the required 'coin_id' column.")

    flat_features = {'coin_id': coin_df['coin_id'].iloc[0]}  # Initialize with coin_id

    # Apply global stats calculations for each metric
    for metric, config in df_metrics_config.items():
        if metric not in coin_df.columns:
            raise ValueError(f"Metric '{metric}' is missing from the input DataFrame.")

        ts = coin_df[metric].copy()  # Get the time series for this metric

        # Standard aggregations
        if 'aggregations' in config:
            for agg in config['aggregations']:
                if agg == 'sum':
                    flat_features[f'{metric}_sum'] = ts.sum()
                elif agg == 'mean':
                    flat_features[f'{metric}_mean'] = ts.mean()
                elif agg == 'median':
                    flat_features[f'{metric}_median'] = ts.median()
                elif agg == 'min':
                    flat_features[f'{metric}_min'] = ts.min()
                elif agg == 'max':
                    flat_features[f'{metric}_max'] = ts.max()
                elif agg == 'std':
                    flat_features[f'{metric}_std'] = ts.std()
                else:
                    raise KeyError(f"Aggregation '{agg}' for metric '{metric}' is not recognized.")

        # Rolling window calculations
        rolling = config.get('rolling', False)
        if rolling:
            rolling_stats = config['rolling']['stats']  # Get rolling stats
            comparisons = config['rolling'].get('comparisons', [])  # Get comparisons
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
    Helper function to calculate a given statistic for a time series.

    Params:
    - ts (pd.Series): Time series data.
    - stat (str): The statistic to calculate (e.g., 'sum', 'mean').

    Returns:
    - The calculated statistic value.

    Raises:
    - KeyError: If the statistic is not recognized.
    """
    if stat == 'sum':
        return ts.sum()
    elif stat == 'mean':
        return ts.mean()
    elif stat == 'median':
        return ts.median()
    elif stat == 'std':
        return ts.std()
    elif stat == 'max':
        return ts.max()
    elif stat == 'min':
        return ts.min()
    else:
        raise KeyError(f"Invalid statistic: '{stat}'")



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



def preprocess_coin_df(input_path, modeling_config, df_metrics_config):
    """
    Preprocess the flattened coin DataFrame by applying feature selection and scaling based on
    the metrics config.

    Params:
    - input_path (str): Path to the flattened CSV file.
    - modeling_config (dict): Configuration with modeling-specific parameters.
    - df_metrics_config (dict): The input file's configuration with metrics and their scaling
        methods, aggregations, etc. This needs to be metrics_config['metrics'][dataset_name],
        rather than the full config file.

    Returns:
    - df (pd.DataFrame): The preprocessed DataFrame.
    - output_path (str): The full path to the saved preprocessed CSV file.
    """
    # Step 1: Load the flattened data
    df = pd.read_csv(input_path)

    # Step 2: Check for missing values and raise an error if any are found
    if df.isnull().values.any():
        raise ValueError("Missing values detected in the DataFrame.")

    # Step 3: Apply feature selection (using drop_features)
    drop_features = modeling_config['preprocessing'].get('drop_features', [])
    if drop_features is not None:
        df = df.drop(columns=drop_features, errors='ignore')

    # Step 4: Apply scaling to the relevant columns based on the metrics config
    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler()
    }

    # Loop through each metric and its settings in the df_metrics_config
    for metric, settings in df_metrics_config.items():
        # Loop through each aggregation (e.g., sum, mean) and its associated settings
        for agg, agg_settings in settings['aggregations'].items():
            # Construct the column name based on the metric and aggregation (e.g., 'buyers_new_sum')
            column_name = f"{metric}_{agg}"

            # Check if the column exists in the DataFrame
            if column_name in df.columns:
                # Retrieve the scaling method from the configuration (e.g., 'standard', 'minmax')
                try:
                    scaling_method = agg_settings.get('scaling')
                except AttributeError:
                    scaling_method = None

                # If a scaling method is specified
                if scaling_method:
                    # Check if the scaling method is recognized (exists in the 'scalers' dictionary)
                    if scaling_method in scalers:
                        # Retrieve the appropriate scaler (e.g., StandardScaler, MinMaxScaler)
                        scaler = scalers[scaling_method]

                        # Apply the scaling transformation to the specified column
                        df[[column_name]] = scaler.fit_transform(df[[column_name]])
                    else:
                        # Log a warning if the scaling method specified is not recognized
                        logger.info("Unknown scaling method %s for column %s"
                                    , scaling_method, column_name)

    # Step 5: Generate output path and filename based on input
    base_filename = os.path.basename(input_path).replace(".csv", "")
    output_filename = f"{base_filename}_preprocessed.csv"
    output_path = os.path.join(os.path.dirname(input_path)
                               .replace("flattened_outputs", "preprocessed_outputs")
                               , output_filename)

    # Step 6: Save the preprocessed data
    df.to_csv(output_path, index=False)

    # Step 7: Log the changes made
    logger.info("Preprocessed file saved at: %s", output_path)
    return df, output_path




def create_training_data_df(input_directory, input_filenames):
    """
    Merges specified preprocessed output CSVs into a single DataFrame and checks if all have
    identical coin_ids to ensure consistency. Adds suffixes to column names based on filename
    components to avoid duplicates.

    Additionally, raises an error if any of the input files have duplicate coin_ids or are
    missing the coin_id column.

    Params:
    - input_directory (str): Directory containing preprocessed output CSVs.
    - input_filenames (list): List of filenames to be merged (without directory path).

    Returns:
    - training_data_df (pd.DataFrame): DataFrame with merged data from all specified
        preprocessed outputs.
    """
    # Initialize an empty list to hold DataFrames
    df_list = []
    missing_files = []
    coin_id_sets = []

    # Regex to extract the date pattern %Y-%m-%d_%H-%M from the filename
    date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}')

    # Dictionary to track how many times each column name has been used
    column_suffix_count = {}

    # Count occurrences of each metric_string
    metric_string_count = {}

    # First loop to count how often each metric_string appears
    for filename in input_filenames:
        match = date_pattern.search(filename)
        if not match:
            raise ValueError(f"No valid date string found in the filename: {filename}")

        date_string = match.group()
        metric_string = filename.split(date_string)[0].rstrip('_')

        if metric_string not in metric_string_count:
            metric_string_count[metric_string] = 1
        else:
            metric_string_count[metric_string] += 1

    # Loop through the input filenames and attempt to read each
    for filename in input_filenames:
        file_path = os.path.join(input_directory, filename)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

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

            df_list.append(df)
            coin_id_sets.append(set(df['coin_id'].unique()))
        else:
            missing_files.append(filename)

    # Merge all DataFrames iteratively on 'coin_id'
    if df_list:
        training_data_df = df_list[0]
        for df in df_list[1:]:
            training_data_df = pd.merge(training_data_df, df, on='coin_id', how='inner')

        # Ensure no duplicate columns after merging
        training_data_df = training_data_df.loc[:, ~training_data_df.columns.duplicated()]

    else:
        raise ValueError("No preprocessed output files found for the given filenames.")

    # Check if all files have identical coin_ids
    base_coin_ids = coin_id_sets[0]
    non_matching_ids = [coin_id_set for coin_id_set in coin_id_sets if coin_id_set != base_coin_ids]
    if non_matching_ids:
        raise ValueError(
            f"Input files do not have matching coin_ids. Non-matching ids: {non_matching_ids}"
        )

    # Log the results
    logger.info("%d files were successfully merged.", len(df_list))
    if missing_files:
        logger.info("%d files could not be found: %s", len(missing_files), ', '.join(missing_files))
    else:
        logger.info("All specified files were found and merged successfully.")

    return training_data_df




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
