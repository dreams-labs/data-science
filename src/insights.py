"""
functions used to build coin-level features from training data
"""
# pylint: disable=C0301 # line over 100 chars
# pylint: disable=C0103 # X_train violates camelcase
# pylint: disable=E0401 # can't find utils import
# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false

import os
import uuid
from datetime import datetime
import random
import hashlib
import json
import pandas as pd
from sklearn.model_selection import ParameterGrid, ParameterSampler
import matplotlib.pyplot as plt
import seaborn as sns
import dreams_core.core as dc

# project files
import utils as u
import training_data as td
import feature_engineering as fe
import modeling as m

# set up logger at the module level
logger = dc.setup_logger()



def run_experiment(modeling_config):
    """
    Runs an experiment using configurations from the experiments_config.yaml,
    builds models, and logs the results of each trial.

    Args:
    - modeling_config (dict): Configuration dictionary containing paths, model params, etc.

    Returns:
    - experiment_id (string): the ID of the experiment that can be used to retrieve metadata
        from modeling/experiment metadat
    """

    # 1. Extract config variables and store experiment metadata
    # ---------------------------------------------------------
    # Extract folder paths from modeling_config
    modeling_folder = modeling_config['modeling']['modeling_folder']
    config_folder = modeling_config['modeling']['config_folder']

    # Load experiments_config.yaml
    experiments_config = u.load_config(os.path.join(config_folder, 'experiments_config.yaml'))

    # Extract metadata and experiment details from experiments_config
    metadata = experiments_config['metadata']
    experiment_name = metadata['experiment_name']
    search_method = metadata['search_method']
    max_evals = metadata['max_evals']

    # Add experiment_id and timestamp to the metadata
    experiment_id = f"{experiment_name}_{uuid.uuid4()}"
    metadata['experiment_id'] = experiment_id
    metadata['start_time'] = datetime.now().isoformat()
    metadata['trial_logs'] = []  # Initialize the array for trial log filenames

    # Write initial metadata file with start time
    experiment_tracking_path = os.path.join(modeling_folder, "experiment_metadata")
    os.makedirs(experiment_tracking_path, exist_ok=True)
    experiment_metadata_file = os.path.join(experiment_tracking_path, f"{experiment_id}.json")

    with open(experiment_metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

    logger.info('Experiment %s started.', experiment_id)


    # 2. Initialize trial configurations and initial variables
    # -------------------------------------------------------------------------
    # Generate the trial configurations based on variable_overrides
    trial_configurations = generate_experiment_configurations(config_folder, method=search_method, max_evals=max_evals)

    # Cap the number of trials if 'max_evals' is set
    max_evals = experiments_config['metadata'].get('max_evals', len(trial_configurations))
    total_trials = min(len(trial_configurations), max_evals)

    # Retrieve all base datasets
    config = u.load_config(os.path.join(config_folder, 'config.yaml'))
    market_data_df = td.retrieve_market_data()
    market_data_df, _ = td.fill_market_data_gaps(market_data_df, config['data_cleaning']['max_gap_days'])
    prices_df = market_data_df[['coin_id','date','price']].copy()
    profits_df = rebuild_profits_df_if_necessary(config, modeling_folder, prices_df, profits_df=None)

    # remove records from market_data_df that don't have transfers if configured to do so
    if config['data_cleaning']['exclude_coins_without_transfers']:
        market_data_df = market_data_df[market_data_df['coin_id'].isin(profits_df['coin_id'])]


    # Initialize progress bar and empty variables
    trials_bar = u.create_progress_bar(total_trials)


    # 3. Iterate through each trial configuration
    # -------------------------------------------
    for n, trial in enumerate(trial_configurations[:total_trials]):

        # 3.1 Prepare the full configuration by applying overrides from the current trial config
        config, metrics_config, modeling_config = prepare_configs(config_folder, trial)

        # Store the configuration settings used in this trial in metadata
        metadata['config_settings'] = {
            "config": config,
            "metrics_config": metrics_config,
            "modeling_config": modeling_config
        }

        # 3.2 Retrieve or rebuild profits_df based on config changes
        profits_df = rebuild_profits_df_if_necessary(config, modeling_folder, prices_df, profits_df)

        # 3.3 Build the configured model input data (train/test data)
        X_train, X_test, y_train, y_test = build_configured_model_input(profits_df, market_data_df, config, metrics_config, modeling_config)

        # 3.4 Train the model using the current configuration and log the results
        model, model_id = m.train_model(X_train, y_train, modeling_folder, modeling_config['modeling']['model_params'])

        # 3.5 Evaluate and save the model's performance on the test set to a CSV
        _ = m.evaluate_model(model, X_test, y_test, model_id, modeling_folder)

        # 3.6 Log the trial results for this configuration
        # Include the trial name, metadata, and other relevant details
        trial_log_filename = m.log_trial_results(modeling_folder, model_id, experiment_id, trial)

        # Append the trial log filename to the metadata
        metadata['trial_logs'].append(trial_log_filename)

        # Update the progress bar
        trials_bar.update(n + 1)

    # Finish the progress bar
    trials_bar.finish()


    # 4. Log experiment metadata
    # -------------------------------------------
    # Add the experiment end time and calculate the duration
    metadata['end_time'] = datetime.now().isoformat()
    metadata['duration'] = (datetime.fromisoformat(metadata['end_time']) -
                            datetime.fromisoformat(metadata['start_time'])).total_seconds()

    # Log experiment metadata
    experiment_tracking_path = os.path.join(modeling_folder, "experiment_metadata")
    os.makedirs(experiment_tracking_path, exist_ok=True)
    experiment_metadata_file = os.path.join(experiment_tracking_path, f"{experiment_id}.json")
    with open(experiment_metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

    logger.info('Experiment %s complete.', experiment_id)

    return experiment_id



@u.timing_decorator
def generate_experiment_configurations(config_folder, method='grid', max_evals=50):
    """
    Generates experiment configurations based on the validated experiment config YAML file and the search method.

    Args:
    - config_folder (str): Path to the folder containing the config files.
    - method (str): 'grid' or 'random' to select the search method.
    - max_evals (int): Number of iterations for Random search (default is 50).

    Returns:
    - configurations (list): List of generated configurations.
    """

    # Load and validate the experiment configuration
    experiment_config = validate_experiments_yaml(config_folder)

    # Flatten the experiment configuration into a dictionary that can be used for grid/random search
    param_grid = {}

    # Flatten the config dictionary so it can be used for the search
    for section, parameters in experiment_config:
        param_grid.update(flatten_dict(parameters, section))

    # Generate configurations based on the chosen method
    if method == 'grid':
        # Grid search: generate all possible combinations
        configurations = list(ParameterGrid(param_grid))
    elif method == 'random':
        # Random search: sample a subset of combinations
        configurations = list(ParameterSampler(param_grid, n_iter=max_evals, random_state=random.randint(1, 100)))
    else:
        raise ValueError(f"Invalid method: {method}. Must be 'grid' or 'random'.")

    return configurations

# helper function for generate_experiment_configurations()
def flatten_dict(d, parent_key='', sep='.'):
    """
    Helper function for generate_experiment_configurations().
    Flattens a nested dictionary.

    Args:
    - d (dict): The dictionary to flatten.
    - parent_key (str): The base key (used during recursion).
    - sep (str): Separator between keys.

    Returns:
    - flat_dict (dict): Flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# helper function for generate_experiment_configurations()
def validate_experiments_yaml(config_folder):
    """
    Ingests the experiment configuration file and checks if all variables
    map correctly to the specified config files.

    Args:
    - config_folder (str): Path to the folder containing all config files, including experiments_config.yaml.

    Returns:
    - configurations (list): List of valid configurations or raises an error if any issues are found.
    """

    # Path to the experiments_config.yaml file
    experiment_config_path = os.path.join(config_folder, 'experiments_config.yaml')

    # Load the experiments_config.yaml file
    experiment_config = u.load_config(experiment_config_path)

    # Extract the variable_overrides section
    if 'variable_overrides' not in experiment_config:
        raise ValueError("Missing 'variable_overrides' section in experiments_config.yaml.")

    variable_overrides = experiment_config['variable_overrides']

    # Dynamically generate the list of config files based on the variable_overrides keys
    config_files = {section: f"{section}.yaml" for section in variable_overrides.keys()}

    # Load all referenced config files dynamically
    loaded_configs = {}
    for section, file_name in config_files.items():
        file_path = os.path.join(config_folder, file_name)
        if os.path.exists(file_path):
            loaded_configs[section] = u.load_config(file_path)
        else:
            raise FileNotFoundError(f"{file_name} not found in {config_folder}")

    # Validate that each variable in variable_overrides maps correctly to the loaded config files
    for section, section_values in variable_overrides.items():
        if section not in loaded_configs:
            raise ValueError(f"Section '{section}' in variable_overrides not found in any loaded config file.")

        # Get the corresponding config file for this section
        corresponding_config = loaded_configs[section]

        # Validate that keys and values exist in the corresponding config
        for key, values in section_values.items():
            if key not in corresponding_config:
                raise ValueError(f"Key '{key}' in variable_overrides not found in {section}.yaml.")

            # Ensure the values are valid in the corresponding config
            for value in values:
                if value not in corresponding_config[key]:
                    raise ValueError(f"Value '{value}' for key '{key}' in variable_overrides is invalid.")

    # If all checks pass, return configurations
    return list(variable_overrides.items())



def prepare_configs(config_folder, override_params):
    """
    Loads config files from the config_folder using load_config and applies overrides specified in override_params.
    Raises KeyError if any key from override_params does not match an existing key in the corresponding config.

    Args:
    - config_folder (str): Path to the folder containing the configuration files.
    - override_params (dict): Dictionary of flattened parameters to override in the loaded configs.

    Returns:
    - config (dict): The main config file with overrides applied.
    - metrics_config (dict): The metrics configuration with overrides applied.
    - modeling_config (dict): The modeling configuration with overrides applied.
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
    period_dates = u.calculate_period_dates(config['training_data'])
    config['training_data'].update(period_dates)

    return config, metrics_config, modeling_config

# helper function for prepare_configs()
def set_nested_value(config, key_path, value):
    """
    Sets a value in a nested dictionary based on a flattened key path.

    Args:
    - config (dict): The configuration dictionary to update.
    - key_path (str): The flattened key path (e.g., 'config.data_cleaning.inflows_filter').
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
    - key_path (str): The flattened key path to check (e.g., 'config.data_cleaning.inflows_filter').

    Raises:
    - KeyError: If the key path does not exist in the config.
    """
    keys = key_path.split('.')
    sub_dict = config
    for key in keys[:-1]:  # Traverse down to the second-to-last key
        if key not in sub_dict:
            raise KeyError(f"Key '{key}' not found in config at level '{'.'.join(keys[:-1])}'")
        sub_dict = sub_dict[key]
    if keys[-1] not in sub_dict:
        raise KeyError(f"Key '{keys[-1]}' not found in config at final level '{'.'.join(keys[:-1])}'")




def rebuild_profits_df_if_necessary(config, modeling_folder, prices_df, profits_df=None):
    """
    Checks if the config has changed and reruns time-intensive steps if needed.

    Args:
    - config (dict): The config containing training_data and data_cleaning.
    - modeling_folder (str): Folder for outputs, including temp files.
    - prices_df (DataFrame): The prices dataframe needed to compute profits_df.
    - profits_df (DataFrame, optional): The profits dataframe passed in memory.

    Returns:
    - profits_df (DataFrame): The profits dataframe.
    """

    # Set up the temp folder inside the modeling folder and raise an error if it doesn't exist
    temp_folder = os.path.join(modeling_folder, 'outputs/temp')
    if not os.path.exists(temp_folder):
        raise FileNotFoundError(f"Required temp folder '{temp_folder}' does not exist.")

    # Combine 'training_data' and 'data_cleaning' for a single hash
    relevant_config = {**config['training_data'], **config['data_cleaning']}
    config_hash = generate_config_hash(relevant_config)

    # Load the previous hash
    previous_hash = handle_hash(config_hash, temp_folder, 'load')

    # If the hash hasn't changed and profits_df is passed, skip rerun
    if config_hash == previous_hash and profits_df is not None:
        logger.debug("Using passed profits_df from memory.")
        return profits_df

    # Otherwise, rerun time-intensive steps
    logger.debug("Config changes detected or missing profits_df, rerunning time-intensive steps...")

    # Example time-intensive logic to regenerate profits_df
    transfers_df = td.retrieve_transfers_data(
        config['training_data']['training_period_start'],
        config['training_data']['modeling_period_start'],
        config['training_data']['modeling_period_end']
    )

    profits_df = td.prepare_profits_data(transfers_df, prices_df)
    profits_df = td.calculate_wallet_profitability(profits_df)
    profits_df, _ = td.clean_profits_df(profits_df, config['data_cleaning'])

    # Save the new hash for future runs
    handle_hash(config_hash, temp_folder, 'save')

    return profits_df

# helper function for rebuild_profits_df_if_necessary()
def generate_config_hash(config):
    """
    Generates a hash for a given config section.

    Args:
    - config (dict): The configuration section.

    Returns:
    - str: A hash representing the config state.
    """
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode('utf-8')).hexdigest()

# helper function for rebuild_profits_df_if_necessary()
def handle_hash(config_hash, temp_folder, operation='load'):
    """
    Handles saving or loading the config hash for checking.

    Args:
    - config_hash (str): The generated hash to save or load.
    - temp_folder (str): The folder where the hash file is stored.
    - operation (str): Either 'save' or 'load' to perform the operation.

    Returns:
    - str: The loaded hash if operation is 'load', None if it doesn't exist.
    """
    hash_file = os.path.join(temp_folder, 'config_hash.txt')

    if operation == 'load':
        return open(hash_file, 'r', encoding='utf-8').read() if os.path.exists(hash_file) else None
    elif operation == 'save':
        with open(hash_file, 'w', encoding='utf-8') as f:
            f.write(config_hash)



def build_configured_model_input(profits_df, market_data_df, config, metrics_config, modeling_config):
    """
    Build the model input data (train/test sets) based on the configuration settings.

    Args:
    - profits_df (DataFrame): DataFrame containing profits information for wallets.
    - market_data_df (DataFrame): DataFrame containing market data for coins.
    - config (dict): Overall configuration containing details for wallet cohorts and training data periods.
    - metrics_config (dict): Configuration for metric generation.
    - modeling_config (dict): Configuration for model training parameters.

    Returns:
    - X_train (DataFrame): Training feature set.
    - X_test (DataFrame): Testing feature set.
    - y_train (Series): Training target variable.
    - y_test (Series): Testing target variable.
    """

    # 1. Generate and merge features for all datasets
    # -------------------------------------
    # Time series features
    market_data_tuples, _ = fe.generate_time_series_features(
            'time_series',
            market_data_df,
            config,
            metrics_config,
            modeling_config
        )

    # Wallet cohort features
    wallet_cohort_tuples, _ = fe.generate_wallet_cohort_features(
            profits_df,
            config,
            metrics_config,
            modeling_config
        )

    # Merge all the features
    training_data_tuples = market_data_tuples + wallet_cohort_tuples
    training_data_df, _ = fe.create_training_data_df(modeling_config['modeling']['modeling_folder'], training_data_tuples)


    # 2. Add target variable to training_data_df
    # ------------------------------------------
    # create the target variable df
    target_variable_df,_ = fe.create_target_variables_mooncrater(market_data_df, config['training_data'], modeling_config)

    # merge the two into the final model input df
    model_input_df = fe.prepare_model_input_df(training_data_df, target_variable_df, modeling_config['modeling']['target_column'])


    # 3. Split into train/test datasets
    # ---------------------------------
    # split the df into train and test sets
    X_train, X_test, y_train, y_test = m.split_model_input(
        model_input_df,
        modeling_config['modeling']['target_column'],
        modeling_config['modeling']['train_test_split'],
        modeling_config['modeling']['random_state']
    )

    return X_train, X_test, y_train, y_test





def generate_trial_df(modeling_folder, experiment_id):
    """
    Generates a DataFrame by loading and processing trial logs from a specified modeling folder and experiment ID.

    Parameters:
    - modeling_folder: The path to the folder where model data is stored.
    - experiment_id: The ID of the experiment to retrieve trial logs from.

    Returns:
    - A pandas DataFrame (trial_df) containing the processed trial logs.
    """

    # 1. Construct the path to the experiment metadata file
    experiment_metadata_path = os.path.join(modeling_folder, "experiment_metadata", f"{experiment_id}.json")

    # 2. Load the experiment metadata
    with open(experiment_metadata_path, 'r', encoding='utf-8') as f:
        experiment_metadata = json.load(f)

    # Retrieve trial log filenames
    trial_logs = experiment_metadata['trial_logs']

    # 3. Initialize lists to store trial data
    trial_data = []

    # 4. Loop through each trial log and extract relevant information
    for trial_log_path in trial_logs:
        with open(trial_log_path, 'r', encoding='utf-8') as f:
            trial_log_data = json.load(f)

        # Extract trial_overrides and performance metrics
        trial_overrides = trial_log_data.get('trial_overrides', {})
        performance_metrics = trial_log_data.get('metrics', {})

        # Merge trial_overrides and performance metrics into a single dictionary
        trial_info = {**trial_overrides, **performance_metrics}

        # Append trial info to the list
        trial_data.append(trial_info)

    # 5. Convert the list of dictionaries to a pandas DataFrame
    trial_df = pd.DataFrame(trial_data)

    return trial_df


def summarize_feature_performance(trial_df):
    """
    Summarizes the performance of the values for each feature that was experimented on.
    """
    #  Identify the value and feature columns
    value_vars = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'log_loss']
    id_vars = [col for col in trial_df.columns if col not in (value_vars + ['confusion_matrix'])]

    # Melt the dataframe
    melted_df = pd.melt(trial_df, id_vars=id_vars, value_vars=value_vars,
                        var_name='metric', value_name='value')

    # Create a list to store the results
    results = []

    # Iterate through each feature
    for feature in id_vars:
        # Group by feature value and metric, then calculate the mean
        grouped = melted_df.groupby([feature, 'metric'])['value'].mean().unstack()

        # Calculate the count of models for each feature value
        model_count = melted_df.groupby(feature)['metric'].count().div(len(value_vars)).astype(int)

        # Reset index and rename columns
        grouped = grouped.reset_index()
        grouped.columns.name = None
        grouped = grouped.rename(columns={col: f'avg_{col}' for col in value_vars})

        # Rename the feature column and add the model count
        grouped = grouped.rename(columns={feature: 'value'})
        grouped['model_count'] = grouped['value'].map(model_count)
        grouped['feature'] = feature

        # Reorder columns
        column_order = ['feature', 'value', 'model_count'] + [f'avg_{metric}' for metric in value_vars]
        grouped = grouped[column_order]

        # Append to results
        results.append(grouped)

    # Concatenate all results and clean up formatting
    feature_performance_df = pd.concat(results, ignore_index=True)
    feature_performance_df = feature_performance_df.sort_values(['feature', 'value'])
    feature_performance_df = feature_performance_df.reset_index(drop=True)

    # Define the columns that start with 'avg'
    columns_to_format = [f'avg_{metric}' for metric in value_vars]

    # Apply conditional formatting to those columns
    feature_performance_df = feature_performance_df.style.background_gradient(subset=columns_to_format, cmap='RdYlGn')

    return feature_performance_df





def plot_roc_auc_performance(trial_df, top_n):
    """
    Plot the average ROC AUC performance for the top N features in the trial_df.

    Parameters:
    - trial_df: DataFrame containing trial data with columns to be grouped and evaluated.
    - top_n: The number of top features based on average ROC AUC to plot.
    """
    # Ensure all columns are converted to strings to handle arrays, lists, and other non-numeric data types
    trial_df = trial_df.applymap(lambda x: str(x) if isinstance(x, (list, dict, tuple)) else x)

    # Calculate mean roc_auc for each category in the relevant columns
    roc_auc_means = pd.DataFrame()

    for column in trial_df.columns:
        if column not in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'log_loss', 'confusion_matrix']:
            grouped = trial_df.groupby(column)['roc_auc'].mean().reset_index()
            grouped['feature'] = column
            roc_auc_means = pd.concat([roc_auc_means, grouped], ignore_index=True)

    # Sort by mean ROC AUC and select the top N
    roc_auc_means = roc_auc_means.sort_values(by='roc_auc', ascending=False).head(top_n)

    # Plot the results in a single bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x='roc_auc', y='feature', data=roc_auc_means)
    plt.title(f'Top {top_n} Features by ROC AUC Performance')
    plt.xlabel('Average ROC AUC')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()


def plot_top_feature_importance(modeling_folder, experiment_id, top_n=10):
    """
    Plot the top features by mean importance from an experiment's feature importance logs.

    Parameters:
    - modeling_folder: str, path to the folder where the experiment data is stored.
    - experiment_id: str, unique identifier for the experiment to retrieve the logs.
    - top_n: int, number of top features to display in the bar chart (default: 10).

    This function retrieves trial logs from an experiment's metadata, extracts feature importance
    data, calculates the mean importance across all trials, and displays a bar chart of the top_n
    most important features.
    """
    # 1. Construct the path to the experiment metadata file
    experiment_metadata_path = os.path.join(modeling_folder, "experiment_metadata", f"{experiment_id}.json")

    # Load the experiment metadata to retrieve trial logs
    with open(experiment_metadata_path, 'r', encoding='utf-8') as f:
        experiment_metadata = json.load(f)

    # Retrieve trial log filenames
    trial_logs = experiment_metadata['trial_logs']

    # Initialize an empty list to store DataFrames
    all_feature_importances = []

    # Loop through each trial log and process feature importance
    for trial_log_path in trial_logs:
        with open(trial_log_path, 'r', encoding='utf-8') as f:
            trial_log_data = json.load(f)

        # Extract feature importance and convert to DataFrame
        feature_importance = trial_log_data['feature_importance']
        feature_importance_df = pd.DataFrame(list(feature_importance.items()), columns=['feature', 'importance'])

        # Append the DataFrame to the list
        all_feature_importances.append(feature_importance_df)

    # Concatenate all DataFrames
    combined_feature_importance_df = pd.concat(all_feature_importances)

    # Group by feature and calculate mean importance
    feature_stats = combined_feature_importance_df.groupby('feature')['importance'].agg(['mean', 'var', 'std']).reset_index()

    # Sort by mean importance
    sorted_features = feature_stats.sort_values(by='mean', ascending=False)

    # Plot the top features by importance
    sorted_features.head(top_n).sort_values(by='mean',ascending=True).plot(kind='barh', x='feature', y='mean', title=f'Top {top_n} Features by Mean Importance')

    # Display the plot
    plt.xlabel('Mean Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()



def analyze_experiment(modeling_folder, experiment_id, top_n=10):
    """
    Analyze experiment results by generating two visualizations:
    1. A bar chart of the top N features by ROC AUC performance.
    2. A bar chart of the top N features by mean importance from the trial logs.

    Parameters:
    - modeling_folder: str, path to the folder where the experiment data is stored.
    - experiment_id: str, unique identifier for the experiment to retrieve the logs.
    - top_n: int, number of top features to display in the bar charts (default: 10).
    """
    # Generate trial DataFrame from the experiment logs
    trial_df = generate_trial_df(modeling_folder, experiment_id)

    # Plot ROC AUC performance for the top N features
    plot_roc_auc_performance(trial_df, top_n)

    # Plot top feature importance based on the trial logs
    plot_top_feature_importance(modeling_folder, experiment_id, top_n)
