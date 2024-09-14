"""
functions used to build coin-level features from training data
"""
# pylint: disable=C0301 # line over 100 chars
# pylint: disable=C0303 # trailing whitespace
# pylint: disable=C0103 # X_train violates camelcase
# pylint: disable=E0401 # can't find utils import

import os
import random
import pandas as pd
import numpy as np
import dreams_core.core as dc
from sklearn.model_selection import ParameterGrid, ParameterSampler
from utils import load_config

# set up logger at the module level
logger = dc.setup_logger()


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
