"""
functions used to build coin-level features from training data
"""
# pylint: disable=C0301 # line over 100 chars
# pylint: disable=C0303 # trailing whitespace
# pylint: disable=C0103 # X_train violates camelcase
# pylint: disable=E0401 # can't find utils import

import os
import random
import hashlib
import json
from sklearn.model_selection import ParameterGrid, ParameterSampler
import dreams_core.core as dc
import progressbar

# project files
from utils import load_config
import training_data as td
import feature_engineering as fe
import coin_wallet_metrics as cwm
import modeling as m

# set up logger at the module level
logger = dc.setup_logger()


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
    experiment_config = load_config(experiment_config_path)

    # Dynamically generate the list of config files based on the experiment_config keys
    config_files = {section: f"{section}.yaml" for section in experiment_config.keys()}

    # Load all referenced config files dynamically
    loaded_configs = {}
    for section, file_name in config_files.items():
        file_path = os.path.join(config_folder, file_name)
        if os.path.exists(file_path):
            loaded_configs[section] = load_config(file_path)
        else:
            raise FileNotFoundError(f"{file_name} not found in {config_folder}")

    # Validate that each variable in experiment_config maps correctly to the loaded config files
    for section, section_values in experiment_config.items():
        if section not in loaded_configs:
            raise ValueError(f"Section '{section}' in experiments_config.yaml not found in any loaded config file.")

        # Get the corresponding config file for this section
        corresponding_config = loaded_configs[section]
        
        # Validate that keys and values exist in the corresponding config
        for key, values in section_values.items():
            if key not in corresponding_config:
                raise ValueError(f"Key '{key}' in experiments_config.yaml not found in {section}.yaml.")
            
            # Ensure the values are valid in the corresponding config
            for value in values:
                if value not in corresponding_config[key]:
                    raise ValueError(f"Value '{value}' for key '{key}' in experiments_config.yaml is invalid.")

    # If all checks pass, return configurations
    return list(experiment_config.items())




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

    config = load_config(config_path)
    metrics_config = load_config(metrics_config_path)
    modeling_config = load_config(modeling_config_path)

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




def rebuild_profits_df_if_necessary(config, modeling_folder, profits_df=None):
    """
    Checks if the config has changed and reruns time-intensive steps if needed.

    Args:
    - config (dict): The config containing training_data and data_cleaning.
    - modeling_folder (str): Folder for outputs, including temp files.
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
    prices_df = td.retrieve_prices_data()
    prices_df, _ = td.fill_prices_gaps(prices_df, config['data_cleaning']['max_gap_days'])
    
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



def build_configured_model_input(profits_df, prices_df, config, metrics_config, modeling_config):
    """
    Build the model input data (train/test sets) based on the configuration settings.

    Args:
    - profits_df (DataFrame): DataFrame containing profits information for wallets.
    - prices_df (DataFrame): DataFrame containing price data for coins.
    - config (dict): Overall configuration containing details for wallet cohorts and training data periods.
    - metrics_config (dict): Configuration for metric generation.
    - modeling_config (dict): Configuration for model training parameters.

    Returns:
    - X_train (DataFrame): Training feature set.
    - X_test (DataFrame): Testing feature set.
    - y_train (Series): Training target variable.
    - y_test (Series): Testing target variable.
    """
    
    # 1. Identify cohort of wallets (e.g., sharks) based on the cohort classification logic
    cohort_summary_df = td.classify_wallet_cohort(profits_df, config['wallet_cohorts']['sharks'])

    # 2. Generate buysell metrics for wallets in the identified cohort
    cohort_wallets = cohort_summary_df[cohort_summary_df['in_cohort']]['wallet_address']
    buysell_metrics_df = cwm.generate_buysell_metrics_df(
        profits_df,
        config['training_data']['training_period_end'],
        cohort_wallets
    )

    # 3. Flatten the buysell metrics DataFrame, save it, and preprocess it
    flattened_output_directory = os.path.join(
        modeling_config['modeling']['modeling_folder'],
        'outputs/flattened_outputs/'
    )
    cohort_name = list(config['wallet_cohorts'].keys())[0]
    metric_description = f"{cohort_name}_cohort"

    flattened_buysell_metrics_df = fe.flatten_coin_date_df(
        buysell_metrics_df,
        metrics_config,
        config['training_data']['training_period_end']
    )
    _, flattened_filepath = fe.save_flattened_outputs(
        flattened_buysell_metrics_df,
        flattened_output_directory,
        metric_description,
        config['training_data']['modeling_period_start']
    )
    _, preprocessed_filepath = fe.preprocess_coin_df(
        flattened_filepath,
        modeling_config,
        metrics_config
    )

    # 4. Create training data from the preprocessed DataFrame
    input_directory = f"{preprocessed_filepath.split('preprocessed_outputs/')[0]}preprocessed_outputs/"
    input_filenames = [preprocessed_filepath.split('preprocessed_outputs/')[1]]
    training_data_df = fe.create_training_data_df(input_directory, input_filenames)

    # 5. Create the target variable DataFrame based on price changes
    target_variable_df, _ = fe.create_target_variables_mooncrater(
        prices_df,
        config['training_data'],
        modeling_config
    )

    # 6. Merge the training data with the target variables to create the model input DataFrame
    model_input_df = fe.prepare_model_input_df(
        training_data_df,
        target_variable_df,
        modeling_config['modeling']['target_column']
    )

    # 7. Split the data into train and test sets
    X_train, X_test, y_train, y_test = m.split_model_input(
        model_input_df,
        modeling_config['modeling']['target_column'],
        modeling_config['modeling']['train_test_split'],
        modeling_config['modeling']['random_state']
    )

    return X_train, X_test, y_train, y_test




def run_experiments(method, config_folder, modeling_folder, max_evals=50):
    """
    Runs experiments using a specified search method (grid or random), builds models,
    and logs the results of each experiment.

    Args:
    - method (str): 'grid' or 'random' to select the search method.
    - config_folder (str): Path to the folder containing all configuration files.
    - modeling_folder (str): Path to the folder where models, logs, and results will be saved.
    - max_evals (int): Number of iterations for Random search (default is 50).
    """

    # 1. Generate the experiment configurations
    experiment_configurations = generate_experiment_configurations(config_folder, method=method, max_evals=max_evals)

    # Generate prices_df
    config = load_config(os.path.join(config_folder,'config.yaml'))
    prices_df = td.retrieve_prices_data()
    prices_df,_ = td.fill_prices_gaps(prices_df, config['data_cleaning']['max_gap_days'])

    # 2. Create the progress bar
    total_experiments = len(experiment_configurations)
    experiments_bar = progressbar.ProgressBar(maxval=total_experiments, widgets=[
        ' [', progressbar.Percentage(), '] ',
        progressbar.Bar(), ' (', progressbar.ETA(), ') '
    ]).start()

    # 3. Iterate through each configuration
    for n, experiment in enumerate(experiment_configurations):
        
        # 3.1 Prepare the full configuration by applying overrides from the current experiment config
        config, metrics_config, modeling_config = prepare_configs(config_folder, experiment)
        
        # 3.2 Retrieve or rebuild profits_df based on config changes
        profits_df = rebuild_profits_df_if_necessary(config, modeling_folder)
        
        # 3.3 Build the configured model input data (train/test data)
        X_train, X_test, y_train, y_test = build_configured_model_input(profits_df, prices_df, config, metrics_config, modeling_config)

        # 3.4 Train the model using the current configuration and log the results
        model, model_id = m.train_model(X_train, y_train, modeling_folder, modeling_config['modeling']['model_params'])
        
        # 3.5 Evaluate and log the model's performance on the test set
        _ = m.evaluate_model(model, X_test, y_test, model_id, modeling_folder)

        # 3.6 Log the experiment results for this configuration
        m.log_experiment_results(modeling_folder, model_id)

        # Update the progress bar
        experiments_bar.update(n + 1)

    # Finish the progress bar
    experiments_bar.finish()

    # # 4. Compare all experiments and analyze the best-performing configuration
    # analyze_experiments(modeling_folder)
