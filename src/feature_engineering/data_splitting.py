"""
functions used to build coin-level features from training data
"""
import pandas as pd
import numpy as np
import dreams_core.core as dc
from sklearn.model_selection import train_test_split


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

