"""
functions used to build coin-level features from training data
"""
import pandas as pd
import dreams_core.core as dc


# pylint: disable=C0103  # X_train doesn't conform to snake case
# project module imports

# set up logger at the module level
logger = dc.setup_logger()


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
