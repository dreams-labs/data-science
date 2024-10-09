"""
functions used to build coin-level features from training data
"""
from typing import Dict, List, Any
import itertools
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

        Returns:
            Dict[str, str]: A dictionary mapping column names to scaling methods.
        """
        def recursive_parse(config: Dict[str, Any], prefix: str = '') -> Dict[str, str]:
            mapping = {}
            for key, value in config.items():
                new_prefix = f"{prefix}_{key}" if prefix else key

                if isinstance(value, dict):
                    # Direct scaling for the current level
                    if 'scaling' in value:
                        mapping[new_prefix] = value['scaling']

                    # Handle different configurations
                    if 'aggregations' in value:
                        mapping.update(self._parse_aggregations(value['aggregations'], new_prefix))
                    if 'comparisons' in value:
                        mapping.update(self._parse_comparisons(value['comparisons'], new_prefix))
                    if 'rolling' in value:
                        mapping.update(self._parse_rolling(value['rolling'], new_prefix))
                    if 'indicators' in value:
                        mapping.update(self._parse_indicators(value['indicators'], new_prefix))

                    # Exclude specific keys from recursion
                    keys_to_exclude = {'aggregations', 'comparisons', 'rolling', 'scaling', 'indicators', 'parameters'}
                    sub_config = {k: v for k, v in value.items() if k not in keys_to_exclude}

                    # Recursive call for nested structures
                    mapping.update(recursive_parse(sub_config, new_prefix))

                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            mapping.update(recursive_parse(item, new_prefix))

            return mapping

        return recursive_parse(self.metrics_config)

    def _parse_aggregations(self, aggregations: Dict[str, Any], prefix: str) -> Dict[str, str]:
        """
        Parse the 'aggregations' section of the configuration.
        """
        mapping = {}
        for agg_type, agg_config in aggregations.items():
            column_name = f"{prefix}_{agg_type}"
            if isinstance(agg_config, dict) and 'scaling' in agg_config:
                mapping[column_name] = agg_config['scaling']
            elif isinstance(agg_config, str):
                # If agg_config is directly a scaling method
                mapping[column_name] = agg_config
        return mapping

    def _parse_comparisons(self, comparisons: Dict[str, Any], prefix: str) -> Dict[str, str]:
        """
        Parse the 'comparisons' section of the configuration.
        """
        mapping = {}
        for comp_type, comp_config in comparisons.items():
            column_name = f"{prefix}_{comp_type}"
            if isinstance(comp_config, dict) and 'scaling' in comp_config:
                mapping[column_name] = comp_config['scaling']
        return mapping

    def _parse_rolling(self, rolling_config: Dict[str, Any], prefix: str) -> Dict[str, str]:
        """
        Parse the 'rolling' section of the configuration.
        """
        mapping = {}
        window_duration = rolling_config['window_duration']
        lookback_periods = rolling_config['lookback_periods']

        if 'aggregations' in rolling_config:
            mapping.update(self._parse_rolling_aggregations(
                rolling_config['aggregations'], prefix, window_duration, lookback_periods
            ))
        if 'comparisons' in rolling_config:
            mapping.update(self._parse_rolling_comparisons(
                rolling_config['comparisons'], prefix, window_duration, lookback_periods
            ))
        return mapping

    def _parse_rolling_aggregations(self, aggregations: Dict[str, Any], prefix: str,
                                    window_duration: int, lookback_periods: int) -> Dict[str, str]:
        """
        Parse the 'aggregations' within 'rolling' configuration.
        """
        mapping = {}
        for agg_type, agg_config in aggregations.items():
            if isinstance(agg_config, dict) and 'scaling' in agg_config:
                for period in range(1, lookback_periods + 1):
                    column_name = (f"{prefix}_{agg_type}_{window_duration}d_period_{period}")
                    mapping[column_name] = agg_config['scaling']
        return mapping

    def _parse_rolling_comparisons(self, comparisons: Dict[str, Any], prefix: str,
                                   window_duration: int, lookback_periods: int) -> Dict[str, str]:
        """
        Parse the 'comparisons' within 'rolling' configuration.
        """
        mapping = {}
        for comp_type, comp_config in comparisons.items():
            if isinstance(comp_config, dict) and 'scaling' in comp_config:
                for period in range(1, lookback_periods + 1):
                    column_name = (f"{prefix}_{comp_type}_{window_duration}d_period_{period}")
                    mapping[column_name] = comp_config['scaling']
        return mapping

    def _parse_indicators(self, indicators: Dict[str, Any], prefix: str) -> Dict[str, str]:
        """
        Parse the 'indicators' section of the configuration.
        """
        mapping = {}
        for indicator_type, indicator_config in indicators.items():
            indicator_prefix = f"{prefix}_{indicator_type}"

            # Handle parameters
            if 'parameters' in indicator_config:
                # Get parameter names and values
                param_names = list(indicator_config['parameters'].keys())  # pylint: disable=W0612
                param_values_list = list(indicator_config['parameters'].values())

                # Create combinations of parameters
                param_combinations = list(itertools.product(*param_values_list))

                for params in param_combinations:
                    # Build parameter string
                    param_str = '_'.join(map(str, params))
                    # Full indicator prefix with parameters
                    full_indicator_prefix = f"{indicator_prefix}_{param_str}"

                    # Handle aggregations within the indicator
                    if 'aggregations' in indicator_config:
                        mapping.update(self._parse_aggregations(
                            indicator_config['aggregations'], full_indicator_prefix
                        ))

                    # Handle rolling within the indicator
                    if 'rolling' in indicator_config:
                        mapping.update(self._parse_rolling(
                            indicator_config['rolling'], full_indicator_prefix
                        ))
            else:
                # No parameters, directly process aggregations
                full_indicator_prefix = indicator_prefix
                if 'aggregations' in indicator_config:
                    mapping.update(self._parse_aggregations(
                        indicator_config['aggregations'], full_indicator_prefix
                    ))
                if 'rolling' in indicator_config:
                    mapping.update(self._parse_rolling(
                        indicator_config['rolling'], full_indicator_prefix
                    ))
        return mapping

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
