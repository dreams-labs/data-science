"""
functions used to build coin-level features from training data
"""
import os
import pandas as pd
import numpy as np
import dreams_core.core as dc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# pylint: disable=C0103  # X_train doesn't conform to snake case
# project module imports

# set up logger at the module level
logger = dc.setup_logger()


def perform_train_test_validation_future_splits(training_data_df, target_variable_df, modeling_config):
    """
    Prepares the final model input DataFrame by merging the training data with the target variables
    on the multiindex (coin_id, time_window) and selects the specified target column. Checks for
    data quality issues such as missing columns, duplicate index entries, and missing target variables.

    Parameters:
    - training_data_df: DataFrame with model training features and multiindex (time_window, coin_id).
    - target_variable_df: DataFrame containing target variables with multiindex (coin_id, time_window).
    - target_column: The name of the target variable to train on (e.g., 'is_moon' or 'is_crater').

    Returns:
    - sets_X_y_dict (dict[pd.DataFrame, pd.Series]): Dict with keys for each set type (e.g. train_set,
        future_set, etc) that contains the X and y data for the set.
    """

    # 1. Data Quality Checks
    # ----------------------
    # Set type to object now that there's only one row per coin_id
    target_variable_df = target_variable_df.reset_index()
    target_variable_df['coin_id'] = target_variable_df['coin_id'].astype('object')
    target_variable_df = target_variable_df.set_index(['time_window', 'coin_id'])

    original_row_count = len(training_data_df)
    perform_model_input_data_quality_checks(training_data_df, target_variable_df, modeling_config)

    # 2. Train Test Split
    # -------------------
    data_partitioning_config = modeling_config['preprocessing']['data_partitioning']
    np.random.seed(modeling_config['modeling']['random_seed'])

    # Split future set if specified
    X_future, y_future, temp_training_data_df, temp_target_variable_df = split_future_set(
        training_data_df,
        target_variable_df,
        data_partitioning_config)

    # Split validation set
    X_validation, y_validation, temp_training_data_df, temp_target_variable_df = split_validation_set(
        temp_training_data_df,
        temp_target_variable_df,
        data_partitioning_config,
        training_data_df)

    # Split train and test sets
    X_train, X_test, y_train, y_test = split_train_test_sets(
        temp_training_data_df,
        temp_target_variable_df,
        data_partitioning_config,
        training_data_df,
    )

    # Create the result dictionary
    sets_X_y_dict = {
        'train_set': (X_train, y_train),
        'test_set': (X_test, y_test),
        'validation_set': (X_validation, y_validation),
        'future_set': (X_future, y_future)
    }

    # 3. Logs and additional data quality checks
    # ------------------------------------------

    # Prepare log message
    target_column = modeling_config['modeling']['target_column']
    unique_values = target_variable_df[target_column].unique()
    is_binary = len(unique_values) == 2 and set(unique_values).issubset({0, 1})

    log_message = "Data Partitioning Results:\n"
    total_partitioned_rows = 0
    for set_name, (X, y) in sets_X_y_dict.items():
        row_count = len(X)
        total_partitioned_rows += row_count
        log_message += f"- {set_name}: {row_count} rows"
        if is_binary:

            positive_count = (y[target_column] == 1).sum()
            total_count = len(y)
            percentage = (positive_count / total_count) * 100 if total_count > 0 else 0
            log_message += f", Positive samples: {positive_count} ({percentage:.2f}%)"
        log_message += "\n"

    # Check if total rows in all sets equals original row count
    if total_partitioned_rows != original_row_count:
        raise ValueError(f"Data partitioning error: Total rows in all sets ({total_partitioned_rows}) "
                            f"does not match original row count ({original_row_count})")

    # Log the consolidated message
    logger.info(log_message)

    return sets_X_y_dict



def split_future_set(training_data_df, target_variable_df, data_partitioning_config):
    """
    Splits off the future set from the full training_data_df by assigning it the latest
    n time_window values from the index. This will allow the set to see how well the
    model generalizes to periods after the train/test/validation sets.

    Params:
    - training_data_df (pd.DataFrame): the full set of training data rows
    - target_variable_df (pd.DataFrame): the full set of target data rows
    - data_partitioning_config (dict): modeling_config['preprocessing']['data_partitioning']

    Returns:
    - X_future, y_future (pd.DataFrame, pd.Series): the X and y for the future set
    - temp_training_data_df (pd.DataFrame): the remaining training data rows
    - temp_target_variable_df (pd.DataFrame): the remaining target variable rows
    """
    unique_time_windows = training_data_df.index.get_level_values('time_window').unique()

    # Select the first n windows, counting in reverse order
    if data_partitioning_config['future_set_time_windows'] == 0:
        future_time_windows = []
    else:
        future_time_windows = unique_time_windows[-data_partitioning_config['future_set_time_windows']:]
    future_mask = training_data_df.index.get_level_values('time_window').isin(future_time_windows)

    X_future = training_data_df[future_mask]
    y_future = target_variable_df[future_mask]
    temp_training_data_df = training_data_df[~future_mask]
    temp_target_variable_df = target_variable_df[~future_mask]

    return X_future, y_future, temp_training_data_df, temp_target_variable_df



def split_validation_set(temp_training_data_df, temp_target_variable_df,
                         data_partitioning_config, training_data_df
    ):
    """
    Splits off the validation set from the remaining rows in training_data_df after the future
    set has been split by assigning n coins to the validation set, where n is the % of total
    coins n time_window values from the index. This will allow the set to see how well the
    model generalizes to periods after the train/test/validation sets.

    Params:
    - temp_training_data_df (pd.DataFrame): the remaining training data rows
    - temp_target_variable_df (pd.DataFrame): the remaining target data rows
    - data_partitioning_config (dict): modeling_config['preprocessing']['data_partitioning']
    - training_data_df (pd.DataFrame): the full training data df, used for calculating how many
        coins to assign the validation set

    Returns:
    - X_val, y_val (pd.DataFrame, pd.Series): the X and y for the validation set
    - temp_training_data_df (pd.DataFrame): the remaining training data rows after the
        validation set has been split off
    - temp_target_variable_df (pd.DataFrame): the remaining target variable rows after the
        validation set has been split off
    """
    # Get unique coin_ids
    unique_coin_ids = training_data_df.index.get_level_values('coin_id').unique()
    total_coin_ids = len(unique_coin_ids)

    # Calculate the number of coin_ids for the validation set
    num_validation_coins = int(np.round(data_partitioning_config['validation_set_share'] * total_coin_ids))

    # Randomly select coin_ids for the validation set
    validation_coin_ids = np.random.choice(unique_coin_ids, size=num_validation_coins, replace=False)

    # Create masks for the validation and training sets
    validation_mask = temp_training_data_df.index.get_level_values('coin_id').isin(validation_coin_ids)

    # Split the data
    X_val = temp_training_data_df[validation_mask]
    y_val = temp_target_variable_df[validation_mask]
    temp_training_data_df = temp_training_data_df[~validation_mask]
    temp_target_variable_df = temp_target_variable_df[~validation_mask]

    return X_val, y_val, temp_training_data_df, temp_target_variable_df



def split_train_test_sets(temp_training_data_df, temp_target_variable_df, data_partitioning_config,
                     training_data_df):
    """
    Splits the remaining records into the train and test sets, allocating
    [test_set_share] of the original record count to the test set.

    Params:
    - temp_training_data_df (pd.DataFrame): the remaining training data rows
    - temp_target_variable_df (pd.DataFrame): the remaining target data rows
    - data_partitioning_config (dict): modeling_config['preprocessing']['data_partitioning']
    - training_data_df (pd.DataFrame): the full training data df, used for calculating how many
        records to assign the test set

    Returns:
    - X_val, y_val (pd.DataFrame, pd.Series): the X and y for the validation set
    - temp_training_data_df (pd.DataFrame): the remaining training data rows after the
        validation set has been split off
    - temp_target_variable_df (pd.DataFrame): the remaining target variable rows after the
        validation set has been split off
    """
    # Calculate record counts
    original_record_count = len(training_data_df)
    current_record_count = len(training_data_df)

    # Calculate the number of records for the test set based on the original record count
    test_set_size = int(np.round(data_partitioning_config['test_set_share']
                                 * original_record_count))

    # Ensure we don't try to allocate more records to test set than available
    test_set_size = min(test_set_size, current_record_count - 1)

    # Calculate the proportion of remaining records to allocate to test set
    test_proportion = test_set_size / current_record_count

    # Check if there is more than one unique time_window
    if temp_training_data_df.index.get_level_values('time_window').nunique() > 1:
        stratify_col = temp_training_data_df.index.get_level_values('time_window')
    else:
        stratify_col = None  # No stratification if only one time_window remains

    # Perform the split
    X_train, X_test, y_train, y_test = train_test_split(
        temp_training_data_df, temp_target_variable_df,
        test_size=test_proportion,
        stratify=stratify_col
    )

    return X_train, X_test, y_train, y_test



def perform_model_input_data_quality_checks(training_data_df, target_variable_df, modeling_config):
    """
    Performs data quality checks on the input DataFrames.

    Parameters:
    - training_data_df: DataFrame with model training features and multiindex (time_window, coin_id).
    - target_variable_df: DataFrame containing target variables with multiindex (coin_id, time_window).
    - modeling_config: Configuration dictionary containing modeling parameters.

    Raises:
    - ValueError: If any data quality check fails.
    """
    expected_index_names = ['time_window', 'coin_id']

    # Check MultiIndex structure
    for df, name in [(training_data_df, 'training_data_df'), (target_variable_df, 'target_variable_df')]:
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError(f"{name} must have a MultiIndex")
        if df.index.names != expected_index_names:
            raise ValueError(f"{name} must have a MultiIndex with levels named {expected_index_names}")

    # Check if indices are identical and in the same order
    training_data_df = training_data_df.sort_index()
    target_variable_df = target_variable_df.sort_index()
    if not training_data_df.index.equals(target_variable_df.index):
        raise ValueError("The MultiIndex is not identical for both DataFrames")

    # Check for duplicate index entries
    if training_data_df.index.duplicated().any():
        raise ValueError("Duplicate index entries found in training_data_df")
    if target_variable_df.index.duplicated().any():
        raise ValueError("Duplicate index entries found in target_variable_df")

    # Check for NaN values
    if training_data_df.isnull().any().any():
        logger.warning("NaN values found in training_data_df")
    if target_variable_df.isnull().any().any():
        logger.warning("NaN values found in target_data_df")

    # Check target column existence
    if modeling_config['modeling']['target_column'] not in target_variable_df.columns:
        raise ValueError(f"The target column '{modeling_config['modeling']['target_column']}' "
                         "is missing in target_variable_df")

    # Check dataset size
    if len(training_data_df) < 10:
        raise ValueError("Dataset is too small to perform a meaningful split. Need at least 10 data points.")

    # Check for imbalanced target
    if target_variable_df[modeling_config['modeling']['target_column']].value_counts(normalize=True).max() > 0.95:
        raise ValueError("Target is heavily imbalanced. Consider rebalancing or using specialized techniques.")

    # Check for non-numeric features
    if not all(np.issubdtype(dtype, np.number) for dtype in training_data_df.dtypes):
        raise ValueError("Features contain non-numeric data. Consider encoding categorical features.")



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

    # Makes a new df with all coin_ids in a column
    training_data_df = pd.DataFrame(all_coin_ids, columns=['coin_id'])

    # Iterate through df_list and merge each one
    for df, fill_strategy, filename in df_list:

        # if the df is a macro_series without a coin_id, cross join it to all coin_ids
        if fill_strategy == 'extend':
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



def create_target_variables(prices_df, training_data_config, modeling_config):
    """
    Main function to create target variables based on coin returns.

    Parameters:
    - prices_df: DataFrame containing price data with columns 'coin_id', 'date', and 'price'.
    - training_data_config: Configuration with modeling period dates.
    - modeling_config: Configuration for modeling with target variable settings.

    Returns:
    - target_variables_df: DataFrame with target variables.
    - returns_df: DataFrame with coin returns data.
    """
    returns_df = calculate_coin_returns(prices_df, training_data_config)

    target_variable = modeling_config['modeling']['target_column']

    if target_variable in ['is_moon','is_crater']:
        target_variables_df = calculate_mooncrater_targets(returns_df, modeling_config)
    elif target_variable == 'returns':
        target_variables_df = returns_df.reset_index()
    else:
        raise ValueError(f"Unsupported target variable type: {target_variable}")

    return target_variables_df, returns_df



def calculate_coin_returns(prices_df, training_data_config):
    """
    Prepares the data and computes price returns for each coin.

    Parameters:
    - prices_df: DataFrame containing price data with columns 'coin_id', 'date', and 'price'.
    - training_data_config: Configuration with modeling period dates.

    Returns:
    - returns_df: DataFrame with columns 'coin_id' and 'returns'.
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

    # Compute returns
    returns = (end_prices[valid_coins] - start_prices[valid_coins]) / start_prices[valid_coins]
    returns_df = pd.DataFrame({'returns': returns})

    return returns_df



def calculate_mooncrater_targets(returns_df, modeling_config):
    """
    Calculates 'is_moon' and 'is_crater' target variables based on returns.

    Parameters:
    - returns_df: DataFrame with columns 'coin_id' and 'returns'.
    - modeling_config: Configuration for modeling with target variable thresholds.

    Returns:
    - target_variables_df: DataFrame with columns 'coin_id', 'is_moon', and 'is_crater'.
    """
    moon_threshold = modeling_config['target_variables']['moon_threshold']
    crater_threshold = modeling_config['target_variables']['crater_threshold']
    moon_minimum_percent = modeling_config['target_variables']['moon_minimum_percent']
    crater_minimum_percent = modeling_config['target_variables']['crater_minimum_percent']

    target_variables_df = returns_df.copy().reset_index()
    target_variables_df['is_moon'] = (target_variables_df['returns'] >= moon_threshold).astype(int)
    target_variables_df['is_crater'] = (target_variables_df['returns'] <= crater_threshold).astype(int)

    total_coins = len(target_variables_df)
    moons = target_variables_df['is_moon'].sum()
    craters = target_variables_df['is_crater'].sum()

    # Ensure minimum percentage for moons and craters
    if moons / total_coins < moon_minimum_percent:
        additional_moons_needed = int(total_coins * moon_minimum_percent) - moons
        moon_candidates = (target_variables_df[target_variables_df['is_moon'] == 0]
                           .nlargest(additional_moons_needed, 'returns'))
        target_variables_df.loc[moon_candidates.index, 'is_moon'] = 1

    if craters / total_coins < crater_minimum_percent:
        additional_craters_needed = int(total_coins * crater_minimum_percent) - craters
        crater_candidates = (target_variables_df[target_variables_df['is_crater'] == 0]
                             .nsmallest(additional_craters_needed, 'returns'))
        target_variables_df.loc[crater_candidates.index, 'is_crater'] = 1

    # Log results
    total_coins = len(target_variables_df)
    moons = target_variables_df['is_moon'].sum()
    craters = target_variables_df['is_crater'].sum()

    logger.info(
        "Target variables created for %s coins with %s/%s (%s) moons and %s/%s (%s) craters.",
        total_coins, moons, total_coins, f"{moons/total_coins:.2%}",
        craters, total_coins, f"{craters/total_coins:.2%}"
    )

    if modeling_config['modeling']['target_column']=="is_moon":
        target_column_df = target_variables_df[['coin_id', 'is_moon']]
    elif modeling_config['modeling']['target_column']=="is_crater":
        target_column_df = target_variables_df[['coin_id', 'is_crater']]
    else:
        raise KeyError("Cannot run calculate_mooncrater_targets() if target column is not 'is_moon' or 'is_crater'.")

    return target_column_df






# def convert_dataset_metrics_to_features(
#     dataset_metrics_df,
#     dataset_config,
#     dataset_metrics_config,
#     config,
#     modeling_config,
# ):
#     """
#     Converts a dataset keyed on coin_id-date into features by flattening and preprocessing.

#     Args:
#         dataset_metrics_df (pd.DataFrame): Input DataFrame containing raw metrics data.
#         dataset_config (dict): The component of config['datasets'] relating to this dataset
#         dataset_metrics_config (dict): The component of metrics_config relating to this dataset
#         config (dict): The whole main config, which includes period boundary dates
#         modeling_config (dict): The whole modeling_config, which includes a list of tables to
#             drop in preprocessing

#     Returns:
#         preprocessed_df (pd.DataFrame): The preprocessed DataFrame ready for model training.
#         dataset_tuple (tuple): Contains the preprocessed file name and fill method for the dataset.
#     """

#     # Flatten the metrics DataFrame into the required format for feature engineering
#     flattened_metrics_df = flatten_coin_date_df(
#         dataset_metrics_df,
#         dataset_metrics_config,
#         config['training_data']['training_period_end']  # Ensure data is up to training period end
#     )

#     # Save the flattened output and retrieve the file path
#     _, flattened_metrics_filepath = save_flattened_outputs(
#         flattened_metrics_df,
#         os.path.join(
#             modeling_config['modeling']['modeling_folder'],  # Folder to store flattened outputs
#             'outputs/flattened_outputs'
#         ),
#         dataset_config['description'],  # Descriptive metadata for the dataset
#         config['training_data']['modeling_period_start']  # Ensure data starts from modeling period
#     )

#     # Preprocess the flattened data and return the preprocessed file path
#     preprocessed_df, preprocessed_filepath = preprocess_coin_df(
#         flattened_metrics_filepath,
#         modeling_config,
#         dataset_config,
#         dataset_metrics_config
#     )

#     # this tuple is the input for join_all_feature_dfs() that will merge all the files
#     dataset_tuple = (
#         preprocessed_filepath.split('preprocessed_outputs/')[1],  # Extract file name from the path
#         dataset_config['fill_method']  # fill method to be used as part of the merge process
#     )

#     return preprocessed_df, dataset_tuple
