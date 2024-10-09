"""
functions used to build coin-level features from training data
"""
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import dreams_core.core as dc
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


# pylint: disable=C0103  # X_train doesn't conform to snake case
# project module imports

# set up logger at the module level
logger = dc.setup_logger()


class DataPreprocessor:
    """
    Object to apply scaling to the X features for all split sets using the scaling outcomes
    from the training set. This is important to ensure an accurate assessment of how well the
    model generalizes to the test/validation/future sets based on only the training data.

    The preprocessing methods are called in sequence through self.preprocessing_steps.
    """
    def __init__(self, config, metrics_config, modeling_config):
        # Configuration
        self.config = config
        self.modeling_config = modeling_config

        # Scaling
        self.scaler = ScalingProcessor(metrics_config)

        # Mapping and encoders
        self.prefix_mapping = self._create_prefix_mapping()
        self.categorical_encoders: Dict[str, OneHotEncoder] = {}
        self.categorical_columns: List[str] = []

        # Preprocessing state
        self.columns_to_drop: List[str] = []

        # Define preprocessing pipeline
        self.preprocessing_steps = [
            self._preprocess_categorical_and_boolean,
            self._drop_features_config,
            self._drop_same_columns,
            self._apply_scaling
        ]

    # -------------------------------------------------------------- #
    # Configuration and Mapping Methods
    # -------------------------------------------------------------- #

    def _create_prefix_mapping(self) -> Dict[str, Dict[str, float]]:
        """
        Create a mapping of column prefixes to their config paths and sameness thresholds.

        Parameters:
        config (Dict[str, Any]): The configuration dictionary containing dataset information.

        Returns:
        Dict[str, Dict[str, float]]: A dictionary where keys are column prefixes and values are
        dictionaries containing 'path' (str) and 'threshold' (float) for each prefix.
        """
        # Creates a mapping of which columns relate to which config keys
        mapping = {}
        for dataset_type, dataset_config in self.config['datasets'].items():
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


    # -------------------------------------------------------------- #
    # Data Preprocessing Methods
    # -------------------------------------------------------------- #

    def _preprocess_categorical_and_boolean(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """
        Preprocess categorical columns by one-hot encoding and convert booleans to integers.
        Ensures consistency across all datasets by using the categories from the training set.
        """
        if is_train:
            # Identify categorical columns
            self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

            # Initialize and fit OneHotEncoder for each categorical column
            for col in self.categorical_columns:
                num_categories = df[col].nunique()
                if num_categories > 8:
                    logger.warning("Warning: Column '%s' has %s categories, consider reducing categories.",
                                   col, num_categories)

                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
                encoder.fit(df[[col]])
                self.categorical_encoders[col] = encoder

        # One-hot encode categorical columns
        for col in self.categorical_columns:
            encoder = self.categorical_encoders[col]
            encoded_cols = encoder.transform(df[[col]])
            encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names([col]))
            df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)

        # Convert boolean columns to integers
        bool_columns = df.select_dtypes(include=['bool']).columns
        for col in bool_columns:
            df[col] = df[col].astype(int)

        return df


    def _apply_scaling(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """Apply scaling with the ScalingProcessor class"""
        return self.scaler.apply_scaling(df, is_train)



    # -------------------------------------------------------------- #
    # Feature Selection Methods
    # -------------------------------------------------------------- #

    def _drop_features_config(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """Drop features specified in modeling_config."""
        drop_features = self.modeling_config['preprocessing'].get('drop_features', [])
        if drop_features:
            df = df.drop(columns=drop_features, errors='ignore')

            # If preprocessing the training set, log dropped columns
            if is_train:
                dropped = set(drop_features) & set(df.columns)
                not_dropped = set(drop_features) - dropped
                if dropped:
                    logger.debug("Dropped specified features: %s", list(dropped))
                if not_dropped:
                    logger.warning("Some specified features were not found in the dataset: %s",
                                   list(not_dropped))
        return df

    def _drop_same_columns(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """
        Drop columns based on sameness threshold of the training set.
        """
        # If preprocessing the training set, compile the list of columns to drop
        if is_train:
            self.columns_to_drop = []
            for column in df.columns:
                for prefix, config_info in self.prefix_mapping.items():
                    if column.startswith(prefix):
                        sameness = self._calculate_sameness_percentage(df[column])
                        if sameness > config_info['threshold']:
                            self.columns_to_drop.append(column)
                        break

        # For all dfs, drop the training set columns_to_drop
        df = df.drop(columns=self.columns_to_drop)
        return df

    def _calculate_sameness_percentage(self, column: pd.Series) -> float:
        """Calculate the percentage of the most common value in a column."""
        return column.value_counts().iloc[0] / len(column)


    # -------------------------------------------------------------------------
    # Main Preprocessing Pipeline
    # -------------------------------------------------------------------------

    def preprocess(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Preprocess all datasets consistently based on the training set.

        This method applies all preprocessing steps in the order defined in self.preprocessing_steps.
        It first processes the training set, learning any necessary parameters, and then
        applies the same transformations to the test, validation, and future datasets.

        Parameters:
        datasets (Dict[str, pd.DataFrame]): A dictionary containing 'train', 'test', 'validation',
                                            and 'future' DataFrames.

        Returns:
        Dict[str, pd.DataFrame]: A dictionary containing the preprocessed DataFrames.
        """
        preprocessed_datasets = {}

        # Preprocess training set and learn parameters
        if 'train' in datasets:
            preprocessed_datasets['train'] = datasets['train']
            for step in self.preprocessing_steps:
                preprocessed_datasets['train'] = step(preprocessed_datasets['train'], is_train=True)

        # Apply same preprocessing to all other datasets
        for dataset_name, dataset in datasets.items():
            if dataset_name != 'train':
                preprocessed_datasets[dataset_name] = dataset
                for step in self.preprocessing_steps:
                    preprocessed_datasets[dataset_name] = step(preprocessed_datasets[dataset_name], is_train=False)

        return preprocessed_datasets


import pdb
class ScalingProcessor:
    """
    Class to apply scaling to all columns as configured in the metrics_config.

    High level sequence:
    * Makes a mapping from column to the metrics_config using _create_column_scaling_map()
    * For the training set, applies scaling and stores the scaler
    * For other sets, retrieves the training set scaler and applies it
    """
    def __init__(self, metrics_config: Dict[str, Any]):
        self.metrics_config = metrics_config
        self.column_scaling_map = self._create_column_scaling_map()
        self.scalers = {}

    def _create_column_scaling_map(self) -> Dict[str, str]:
        """
        Create a mapping between column names and their corresponding scaling methods.

        This method recursively parses the metrics configuration to build a flat dictionary
        where keys are column names and values are scaling methods.

        Returns:
            Dict[str, str]: A dictionary mapping column names to scaling methods.
        """
        def recursive_parse(config: Dict[str, Any], prefix: str = '') -> Dict[str, str]:
            """
            Recursively parse the configuration to extract scaling methods for columns.

            Args:
                config (Dict[str, Any]): The configuration dictionary to parse.
                prefix (str): The current prefix for column names.

            Returns:
                Dict[str, str]: A dictionary mapping column names to scaling methods.
            """
            mapping = {}
            for key, value in config.items():
                new_prefix = f"{prefix}_{key}" if prefix else key

                if isinstance(value, dict):
                    # Direct scaling for the current level
                    if 'scaling' in value:
                        mapping[new_prefix] = value['scaling']

                    # Handle aggregations
                    if 'aggregations' in value:
                        for agg_type, agg_config in value['aggregations'].items():
                            if isinstance(agg_config, dict) and 'scaling' in agg_config:
                                mapping[f"{new_prefix}_{agg_type}"] = agg_config['scaling']
                            elif isinstance(agg_config, str):
                                # If agg_config is directly a scaling method
                                mapping[f"{new_prefix}_{agg_type}"] = agg_config
                        # Do not recurse into 'aggregations'

                    # Handle comparisons
                    if 'comparisons' in value:
                        for comp_type, comp_config in value['comparisons'].items():
                            if isinstance(comp_config, dict) and 'scaling' in comp_config:
                                mapping[f"{new_prefix}_{comp_type}"] = comp_config['scaling']
                        # Do not recurse into 'comparisons'

                    # Handle rolling metrics
                    if 'rolling' in value:
                        rolling_config = value['rolling']
                        if 'aggregations' in rolling_config:
                            for agg_type, agg_config in rolling_config['aggregations'].items():
                                if isinstance(agg_config, dict) and 'scaling' in agg_config:
                                    # Iterate over each period
                                    for period in range(1, rolling_config['lookback_periods'] + 1):
                                        mapping[f"{new_prefix}_{agg_type}_"
                                                f"{rolling_config['window_duration']}d_period_{period}"] = agg_config['scaling']
                        if 'comparisons' in rolling_config:
                            for comp_type, comp_config in rolling_config['comparisons'].items():
                                if isinstance(comp_config, dict) and 'scaling' in comp_config:
                                    # Iterate over each period
                                    for period in range(1, rolling_config['lookback_periods'] + 1):
                                        mapping[f"{new_prefix}_{comp_type}_"
                                                f"{rolling_config['window_duration']}d_period_{period}"] = comp_config['scaling']
                        # Do not recurse into 'rolling'

                    # Exclude specific keys from recursion
                    keys_to_exclude = {'aggregations', 'comparisons', 'rolling', 'scaling'}
                    sub_config = {k: v for k, v in value.items() if k not in keys_to_exclude}
                    # Recursive call for nested structures
                    mapping.update(recursive_parse(sub_config, new_prefix))

                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            mapping.update(recursive_parse(item, new_prefix))

            return mapping

        return recursive_parse(self.metrics_config)


    def apply_scaling(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """
        Apply scaling to the dataframe based on the column_scaling_map.

        Args:
            df (pd.DataFrame): The input dataframe to scale.
            is_train (bool): Whether this is the training set (to fit scalers) or not.

        Returns:
            pd.DataFrame: The scaled dataframe.
        """
        scaled_df = df.copy()
        for column in df.columns:
            if column in self.column_scaling_map:
                scaling_method = self.column_scaling_map[column]
                if scaling_method != 'none':
                    scaler = self._get_scaler(scaling_method)
                    if scaling_method == 'log':
                        scaled_values = scaler(df[[column]])
                    else:
                        if is_train:
                            scaled_values = scaler.fit_transform(df[[column]])
                            self.scalers[column] = scaler
                        else:
                            scaler = self.scalers[column]
                            scaled_values = scaler.transform(df[[column]])
                    scaled_df[column] = scaled_values
        return scaled_df

    def _get_scaler(self, scaling_method: str):
        """
        Get the appropriate scaler based on the scaling method.

        Args:
            scaling_method (str): The scaling method to use.

        Returns:
            object: A scaler object or function.

        Raises:
            ValueError: If an unknown scaling method is provided.
        """
        if scaling_method == 'standard':
            return StandardScaler()
        elif scaling_method == 'minmax':
            return MinMaxScaler()
        elif scaling_method == 'log':
            return lambda x: np.log1p(x.values)
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")