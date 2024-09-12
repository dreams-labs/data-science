"""
utility functions use in data science notebooks
"""

import yaml


def load_config(file_path='../notebooks/config.yaml'):
    """
    Load configuration from a YAML file.
    
    Args:
        file_path (str): Path to the config file.
    
    Returns:
        dict: Parsed YAML configuration.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def cw_filter_df(df, coin_id, wallet_address):
    """
    Filter DataFrame by coin_id and wallet_address.
    
    Args:
        df (pd.DataFrame): The DataFrame to filter.
        coin_id (str): The coin ID to filter by.
        wallet_address (str): The wallet address to filter by.
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    filtered_df = df[
        (df['coin_id'] == coin_id) &
        (df['wallet_address'] == wallet_address)
    ]
    return filtered_df
