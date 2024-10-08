"""
functions used to build coin-level features from training data
"""
from typing import Dict, Any
import pandas as pd
import numpy as np
import dreams_core.core as dc
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# pylint: disable=C0103  # X_train doesn't conform to snake case
# project module imports

# set up logger at the module level
logger = dc.setup_logger()






def preprocess_coin_df(df, config, modeling_config):
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
    # Confirm there are no null values
    if df.isnull().values.any():
        raise ValueError("Missing values detected in the DataFrame.")

    # 1. Column Formatting
    # --------------------
    # Convert all columns to numeric
    df = preprocess_categorical_and_boolean(df)

    # 2. Feature Selection
    # --------------------
    # Drop features specified in modeling_config['drop_features']
    drop_features = modeling_config['preprocessing'].get('drop_features', [])
    if drop_features:
        df = df.drop(columns=drop_features, errors='warn')

    # Apply feature selection based on sameness_threshold from dataset_config
    df = check_sameness_and_drop_columns(df, config)

    # 3: Scaling and Transformation
    # ----------------------------------------------------
    # Apply scaling if df_metrics_config is provided
    # if df_metrics_config:
    #     df = apply_scaling(df, df_metrics_config)


    # # Step 5: Save and Log Preprocessed Data
    # # ----------------------------------------------------
    # # Generate output path and filename based on input
    # base_filename = os.path.basename(input_path).replace(".csv", "")
    # output_filename = f"{base_filename}_preprocessed.csv"
    # output_path = os.path.join(
    #     os.path.dirname(input_path).replace("flattened_outputs", "preprocessed_outputs"),
    #     output_filename
    # )

    # # Save the preprocessed data
    # df.to_csv(output_path, index=False)

    # # Log the changes made
    # logger.debug("Preprocessed file saved at: %s", output_path)

    # return df, output_path



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


def preprocess_categorical_and_boolean(df):
    """
    Preprocess categorical columns by one-hot encoding and convert boolean columns to integers.

    Args:
    df (pd.DataFrame): Input DataFrame

    Returns:
    pd.DataFrame: Processed DataFrame
    """
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        num_categories = df[col].nunique()
        if num_categories > 8:
            logger.warning("Column '%s' has %s categories, consider reducing categories.",
                           col, num_categories)
        df = pd.get_dummies(df, columns=[col], drop_first=True)

    # Convert boolean columns to integers
    df = df.apply(lambda col: col.astype(int) if col.dtype == bool else col)

    return df



def check_sameness_and_drop_columns(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Check column sameness and drop columns exceeding the threshold.

    This function analyzes each column in the DataFrame, calculates its sameness percentage,
    and drops columns that exceed the threshold specified in the configuration.

    Parameters:
    df (pd.DataFrame): The input DataFrame to process.
    config (Dict[str, Any]): The configuration dictionary containing sameness thresholds.

    Returns:
    pd.DataFrame: A new DataFrame with columns dropped based on the sameness criteria.

    Raises:
    ValueError: If any columns can't be mapped to a sameness threshold or if any config keys
                can't be mapped to columns.
    """
    prefix_mapping = create_prefix_mapping(config)
    columns_to_drop = []
    unmapped_columns = []
    used_config_keys = set()

    for column in df.columns:
        mapped = False
        for prefix, config_info in prefix_mapping.items():
            if column.startswith(prefix):
                mapped = True
                used_config_keys.add(prefix)
                sameness = calculate_sameness_percentage(df[column])
                if sameness > config_info['threshold']:
                    columns_to_drop.append(column)
                break
        if not mapped:
            unmapped_columns.append(column)

    unused_config_keys = set(prefix_mapping.keys()) - used_config_keys

    if unmapped_columns:
        raise ValueError(f"The following columns could not be mapped to a sameness threshold: {unmapped_columns}")

    if unused_config_keys:
        raise ValueError(f"The following config keys could not be mapped to columns: {unused_config_keys}")

    # Drop the columns
    df.drop(columns=columns_to_drop)
    logger.info("Dropped %s columns %s due to sameness thresholds.", len(columns_to_drop), columns_to_drop)

    return df

def calculate_sameness_percentage(column: pd.Series) -> float:
    """
    Calculate the percentage of the most common value in a column.

    Parameters:
    column (pd.Series): The column to analyze.

    Returns:
    float: The percentage (0 to 1) of the most common value in the column.
    """
    return column.value_counts().iloc[0] / len(column)

def create_prefix_mapping(config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Create a mapping of column prefixes to their config paths and sameness thresholds.

    Parameters:
    config (Dict[str, Any]): The configuration dictionary containing dataset information.

    Returns:
    Dict[str, Dict[str, float]]: A dictionary where keys are column prefixes and values are
    dictionaries containing 'path' (str) and 'threshold' (float) for each prefix.
    """
    mapping = {}

    for dataset_type, dataset_config in config['datasets'].items():
        for category, category_config in dataset_config.items():
            if isinstance(category_config, dict) and 'sameness_threshold' in category_config:
                prefix = f"{category}_"
                mapping[prefix] = {
                    'path': f"datasets.{dataset_type}.{category}",
                    'threshold': category_config['sameness_threshold']
                }
            elif isinstance(category_config, dict):
                for subcategory, subcategory_config in category_config.items():
                    if 'sameness_threshold' in subcategory_config:
                        prefix = f"{subcategory}_"
                        mapping[prefix] = {
                            'path': f"datasets.{dataset_type}.{category}.{subcategory}",
                            'threshold': subcategory_config['sameness_threshold']
                        }

    return mapping
