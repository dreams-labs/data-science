"""Orchestrates groups of functions to generate wallet model pipeline"""
import time
import logging
import gc
from datetime import datetime,timedelta
from typing import Tuple,Optional,Dict,List
import pandas as pd
import numpy as np
import pandas_gbq
from google.cloud import bigquery

# Local module imports
import training_data.profits_row_imputation as pri
import coin_wallet_metrics.indicators as ind
import wallet_modeling.wallet_training_data as wtd
import wallet_features.trading_features as wtf
import wallet_features.performance_features as wpf
import wallet_features.transfers_features as wts
import wallet_features.wallet_features_orchestrator as wfo
import wallet_features.clustering_features as wcl
from wallet_modeling.wallets_config_manager import WalletsConfig
import utils as u

# Set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()



# ------------------------------------------
#      Primary Orchestration Functions
# ------------------------------------------

@u.timing_decorator
def retrieve_period_datasets(period_start_date, period_end_date, coin_cohort=None, parquet_prefix=None):
    """
    Retrieves and processes data for a specific period. If coin_cohort provided,
    filters to those coins. Otherwise applies full cleaning pipeline to establish cohort.

    Params:
    - period_start_date,period_end_date (str): Period boundaries
    - coin_cohort (set, optional): Coin IDs from training cohort
    - parquet_prefix (str, optional): Prefix for saved parquet files

    Returns:
    - tuple: (profits_df, market_data_df, coin_cohort) for the period
    """
    # Get raw period data
    profits_df, market_data_df = wtd.retrieve_raw_datasets(period_start_date, period_end_date)

    # Apply cleaning process including coin cohort filter if specified
    market_data_df = wtd.clean_market_dataset(market_data_df, profits_df,
                                                period_start_date, period_end_date,
                                                coin_cohort)
    profits_df = profits_df[profits_df['coin_id'].isin(market_data_df['coin_id'])]

    # Set the coin_cohort if it hadn't already been passed
    if not coin_cohort:
        coin_cohort = set(market_data_df['coin_id'].unique())
        logger.info("Defined coin cohort of %s coins after applying data cleaning filters.",
                    len(coin_cohort))

    # Impute the period end (period start is pre-imputed during profits_df generation)
    imputed_profits_df = pri.impute_profits_for_multiple_dates(profits_df, market_data_df,
                                                               [period_end_date], n_threads=1)

    # Format and optionally save the datasets
    profits_df_formatted, market_data_df_formatted = wtd.format_and_save_datasets(
        imputed_profits_df,
        market_data_df,
        period_start_date,
        parquet_prefix
    )

    return profits_df_formatted, market_data_df_formatted, coin_cohort



def prepare_training_data(
    profits_df_full: pd.DataFrame,
    market_data_df_full: pd.DataFrame,
    wallets_metrics_config: dict,
    parquet_folder: str
) -> List[str]:
    """
    Consolidated training data preparation pipeline that handles hybridization,
    indicator generation, cohort definition and transfers retrieval.

    Params:
    - profits_df_full: Full historical profits DataFrame
    - market_data_df_full: Full historical market data DataFrame
    - wallets_metrics_config: Configuration for market indicators
    - parquet_folder: Required folder for parquet storage

    Returns:
    - List of generated parquet file paths
    """
    generated_files = []

    # Remove market data before starting balance date and validate periods
    market_data_df = market_data_df_full[
        market_data_df_full['date'] >= wallets_config['training_data']['training_starting_balance_date']
    ]
    u.assert_period(market_data_df,
                    wallets_config['training_data']['training_period_start'],
                    wallets_config['training_data']['training_period_end'])

    # Generate market indicators
    logger.info("Generating market indicators...")
    market_indicators_df = generate_training_indicators_df(
        market_data_df_full,
        wallets_metrics_config,
        parquet_filename=None
    )
    market_indicators_path = f"{parquet_folder}/training_market_indicators_data_df.parquet"
    market_indicators_df.to_parquet(market_indicators_path, index=False)
    generated_files.append(market_indicators_path)
    del market_data_df_full, market_data_df

    # Hybridize wallet IDs if configured
    if wallets_config['training_data']['hybridize_wallet_ids']:
        profits_df_full, hybrid_cw_id_map = hybridize_wallet_address(profits_df_full)
        hybrid_map_path = f"{parquet_folder}/hybrid_cw_id_map.pkl"
        pd.to_pickle(hybrid_cw_id_map, hybrid_map_path)
        generated_files.append(hybrid_map_path)

        upload_hybrid_wallet_mapping(hybrid_cw_id_map)
        del hybrid_cw_id_map

    # Define training wallet cohort
    logger.info("Defining wallet cohort...")
    profits_df, _ = define_training_wallet_cohort(
        profits_df_full,
        market_indicators_df,
        wallets_config['training_data']['hybridize_wallet_ids']
    )
    profits_path = f"{parquet_folder}/training_profits_df.parquet"
    profits_df.to_parquet(profits_path, index=True)
    generated_files.append(profits_path)

    # Retrieve transfers after cohort is in BigQuery
    logger.info("Retrieving transfers sequencing data...")
    transfers_df = wts.retrieve_transfers_sequencing(
        wallets_config['training_data']['hybridize_wallet_ids']
    )
    transfers_path = f"{parquet_folder}/training_transfers_sequencing_df.parquet"
    transfers_df.to_parquet(transfers_path, index=True)
    generated_files.append(transfers_path)

    # Clean up memory
    del profits_df_full, profits_df, market_indicators_df, transfers_df
    gc.collect()

    return generated_files


def generate_training_features(
    profits_df: pd.DataFrame,
    market_indicators_df: pd.DataFrame,
    transfers_df: pd.DataFrame,
    wallet_cohort: List[int],
    parquet_folder: str
) -> None:
    """
    Orchestrates end-to-end feature generation maintaining existing logic.

    Params:
    - profits_df: Training period profits data
    - market_indicators_df: Market data with indicators
    - transfers_df: Transfers sequencing data
    - wallet_cohort: List of wallet addresses
    - parquet_folder: Location for parquet storage
    """
    # Generate full period features
    logger.info("Generating features for full training period...")
    training_wallet_features_df = wfo.calculate_wallet_features(
        profits_df,
        market_indicators_df,
        transfers_df,
        wallet_cohort,
        wallets_config['training_data']['training_period_start'],
        wallets_config['training_data']['training_period_end']
    )

    # Initialize full features df with suffixed columns
    wallet_training_data_df_full = training_wallet_features_df.add_suffix("|all_windows").copy()
    wallet_training_data_df_full.to_parquet(f"{parquet_folder}/wallet_training_data_df_full.parquet", index=True)
    del training_wallet_features_df
    gc.collect()

    # Generate window features
    training_windows_profits_dfs = split_training_window_profits_dfs(
        profits_df,
        market_indicators_df,
        wallet_cohort
    )

    # Process each window
    for i, window_profits_df in enumerate(training_windows_profits_dfs, 1):
        logger.info("Generating features for window %s...", i)

        window_opening_balance_date = window_profits_df['date'].min()
        window_start_date = window_opening_balance_date + timedelta(days=1)
        window_end_date = window_profits_df['date'].max()

        window_wallet_features_df = wfo.calculate_wallet_features(
            window_profits_df,
            market_indicators_df,
            transfers_df,
            wallet_cohort,
            window_start_date.strftime('%Y-%m-%d'),
            window_end_date.strftime('%Y-%m-%d')
        )

        window_wallet_features_df = window_wallet_features_df.add_suffix(f'|w{i}')
        wallet_training_data_df_full = wallet_training_data_df_full.join(window_wallet_features_df, how='left')

    # Save unclustered version
    wallet_training_data_df_full.to_parquet(f"{parquet_folder}/wallet_training_data_df_full_unclustered.parquet", index=True)  # pylint:disable=line-too-long

    # Generate clusters
    if 'clustering_n_clusters' in wallets_config.get('features', {}):
        training_cluster_features_df = wcl.create_kmeans_cluster_features(wallet_training_data_df_full)
        training_cluster_features_df = training_cluster_features_df.add_prefix('cluster|')
        wallet_training_data_df_full = wallet_training_data_df_full.join(training_cluster_features_df, how='inner')

    # Verify cohort integrity
    missing_wallets = set(wallet_cohort) - set(wallet_training_data_df_full.index)
    if missing_wallets:
        raise ValueError(f"Lost {len(missing_wallets)} wallets from original cohort during feature "
                         "generation. First few missing: {list(missing_wallets)[:5]}")

    # Save final version
    wallet_training_data_df_full.to_parquet(f"{parquet_folder}/wallet_training_data_df_full.parquet", index=True)



# -----------------------------------------
#   Modeling Data Orchestration Function
# -----------------------------------------

@u.timing_decorator
def prepare_modeling_features(
    modeling_profits_df_full: pd.DataFrame,
    hybrid_cw_id_map: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Orchestrates data preparation and feature generation for modeling.

    Params:
    - modeling_market_data_df_full: Full market data DataFrame
    - modeling_profits_df_full: Full profits DataFrame
    - config: Configuration dictionary
    - hybrid_cw_id_map: Optional mapping for hybrid wallet IDs

    Returns:
    - modeling_wallet_features_df: Generated wallet features
    """
    logger.info("Beginning modeling data preparation...")

    # Handle hybridization if configured
    if wallets_config['training_data']['hybridize_wallet_ids'] is True:
        logger.info("Applying wallet-coin hybridization...")
        modeling_profits_df_full, _ = hybridize_wallet_address(
            modeling_profits_df_full,
            hybrid_cw_id_map
        )

    # Get training wallet cohort
    logger.info("Loading training wallet cohort...")
    training_wallet_cohort = pd.read_parquet(
        f"{wallets_config['training_data']['parquet_folder']}/wallet_training_data_df_full.parquet",
        columns=[]
    ).index.values

    # Filter profits to training cohort
    modeling_profits_df = modeling_profits_df_full[
        modeling_profits_df_full['wallet_address'].isin(training_wallet_cohort)
    ]
    del modeling_profits_df_full

    # Assert period and save filtered/hybridized profits_df
    u.assert_period(modeling_profits_df,
                    wallets_config['training_data']['modeling_period_start'],
                    wallets_config['training_data']['modeling_period_end'])
    output_path = f"{wallets_config['training_data']['parquet_folder']}/modeling_profits_df.parquet"
    modeling_profits_df.to_parquet(output_path, index=False)

    # Initialize features DataFrame
    logger.info("Generating modeling features...")
    modeling_wallet_features_df = pd.DataFrame(index=training_wallet_cohort)
    modeling_wallet_features_df.index.name = 'wallet_address'

    # Generate trading features and identify modeling cohort
    modeling_trading_features_df = identify_modeling_cohort(modeling_profits_df)
    modeling_wallet_features_df = modeling_wallet_features_df.join(
        modeling_trading_features_df,
        how='left'
    ).fillna({col: 0 for col in modeling_trading_features_df.columns})

    # Generate performance features
    modeling_performance_features_df = wpf.calculate_performance_features(
        modeling_wallet_features_df,
        include_twb_metrics=False
    )
    modeling_wallet_features_df = modeling_wallet_features_df.join(
        modeling_performance_features_df,
        how='left'
    ).fillna({col: 0 for col in modeling_performance_features_df.columns})

    # Save features
    output_path = f"{wallets_config['training_data']['parquet_folder']}/modeling_wallet_features_df.parquet"
    modeling_wallet_features_df.to_parquet(output_path, index=True)
    logger.info("Saved modeling features to %s", output_path)

    # Clean up memory
    del modeling_trading_features_df, modeling_performance_features_df, modeling_profits_df
    gc.collect()

    return modeling_wallet_features_df


# -----------------------------------
#           Helper Functions
# -----------------------------------

@u.timing_decorator
def define_training_wallet_cohort(profits_df: pd.DataFrame,
                                  market_data_df: pd.DataFrame,
                                  hybridize_wallet_ids: bool
                                ) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Orchestrates the definition of a wallet cohort for model training by:
    1. Imputing profits at period boundaries
    2. Calculating wallet-level trading metrics
    3. Filtering wallets based on behavior thresholds
    4. Uploading filtered cohort to BigQuery

    Params:
    - profits_df (DataFrame): Historical profit and balance data for all wallets
    - market_data_df (DataFrame): Market prices and metadata for relevant period
    - hybridize_wallet_ids (bool): whether the IDs are regular wallet_ids or hybrid wallet-coin IDs


    Returns:
    - training_cohort_profits_df (DataFrame): Profits data filtered to selected wallets
    - training_wallet_cohort (ndarray): Array of wallet addresses that pass filters
    """
    start_time = time.time()
    training_period_start = wallets_config['training_data']['training_period_start']
    training_period_end = wallets_config['training_data']['training_period_end']

    # Impute the training period end (training period start is pre-imputed into profits_df generation)
    imputed_profits_df = pri.impute_profits_for_multiple_dates(profits_df, market_data_df,
                                                               [training_period_end], n_threads=24)

    # Create a training period only profits_df
    training_profits_df = (
        imputed_profits_df[imputed_profits_df['date'] <= training_period_end]
        .copy()
    )

    # Confirm valid dates for training period
    u.assert_period(profits_df, training_period_start, training_period_end)
    u.assert_period(market_data_df, training_period_start, training_period_end)

    # Compute wallet level metrics over duration of training period
    training_wallet_metrics_df = wtf.calculate_wallet_trading_features(training_profits_df,
                                                                       training_period_start,
                                                                       training_period_end)

    # Apply filters based on wallet behavior during the training period
    filtered_training_wallet_metrics_df = wtd.apply_wallet_thresholds(training_wallet_metrics_df)
    training_wallet_cohort = filtered_training_wallet_metrics_df.index.values

    if len(training_wallet_cohort) == 0:
        raise ValueError("Cohort does not include any wallets. Cohort must include wallets.")

    # Upload the cohort to BigQuery for additional complex feature generation
    wtd.upload_training_cohort(training_wallet_cohort, hybridize_wallet_ids)
    logger.info("Training wallet cohort defined as %s wallets after %.2f seconds.",
                len(training_wallet_cohort), time.time()-start_time)

    # Create a profits_df that only includes the wallet cohort
    training_cohort_profits_df = training_profits_df[training_profits_df['wallet_address'].isin(training_wallet_cohort)]

    return training_cohort_profits_df, training_wallet_cohort



@u.timing_decorator
def split_training_window_profits_dfs(training_profits_df,training_market_data_df,wallet_cohort):
    """
    Adds imputed rows at the start and end date of all windows
    """
    # Filter to only wallet cohort
    cohort_profits_df = training_profits_df[training_profits_df['wallet_address'].isin(wallet_cohort)]

    # Impute all training window dates
    training_window_boundary_dates = wtd.generate_training_window_imputation_dates()
    training_windows_profits_df = pri.impute_profits_for_multiple_dates(cohort_profits_df,
                                                                        training_market_data_df,
                                                                        training_window_boundary_dates,
                                                                        n_threads=1)

    # Split profits_df into training windows
    training_windows_profits_dfs = wtd.split_training_window_dfs(training_windows_profits_df)

    return training_windows_profits_dfs



@u.timing_decorator
def generate_training_indicators_df(training_market_data_df_full,wallets_metrics_config,
                                    parquet_filename="training_market_indicators_data_df",
                                    parquet_folder="temp/wallet_modeling_dfs"):
    """
    Adds the configured indicators to the training period market_data_df and stores it
    as a parquet file by default (or returns it).

    Default save location: temp/wallet_modeling_dfs/market_indicators_data_df.parquet

    Params:
    - training_market_data_df_full (df): market_data_df with complete historical data, because indicators can
        have long lookback periods (e.g. SMA 200)
    - wallets_metrics_config (dict): metrics_config.py compatible metrics definitions
    - parquet_file, parquet_folder (strings): if these have values, the output df will be saved to this
        location instead of being returned

    Returns:
    - market_indicators_data_df (df): market_data_df for the training period only

    """
    logger.info("Beginning indicator generation process...")

    # Validate that no records exist after the training period
    training_period_end = wallets_config['training_data']['training_period_end']
    latest_market_data_record = training_market_data_df_full['date'].max()
    if latest_market_data_record > pd.to_datetime(training_period_end):
        raise ValueError(
            f"Detected data after the end of the training period in training_market_data_df_full."
            f"Latest record found: {latest_market_data_record} vs period end of {training_period_end}"
        )

    # Adds time series ratio metrics that can have additional indicators applied to them
    if any(k in wallets_metrics_config['time_series']['market_data'] for k in ['mfi', 'obv']):
        market_indicators_data_df = ind.add_market_data_dualcolumn_indicators(training_market_data_df_full)
    else:
        market_indicators_data_df = training_market_data_df_full

    # Adds indicators to all configured time series
    market_indicators_data_df = ind.generate_time_series_indicators(market_indicators_data_df,
                                                            wallets_metrics_config['time_series']['market_data'],
                                                            'coin_id')

    # Filters out pre-training period records now that we've computed lookback and rolling metrics
    market_indicators_data_df = market_indicators_data_df[market_indicators_data_df['date']
                                                    >=wallets_config['training_data']['training_starting_balance_date']]

    # Reset OBV to 0 at training start if it exists
    training_start = pd.to_datetime(wallets_config['training_data']['training_starting_balance_date'])
    if 'obv' in market_indicators_data_df.columns:
        # Group by coin_id since OBV is coin-specific
        for coin_id in market_indicators_data_df['coin_id'].unique():
            mask = (market_indicators_data_df['coin_id'] == coin_id) & \
                  (market_indicators_data_df['date'] >= training_start)
            coin_idx = market_indicators_data_df[mask].index
            if len(coin_idx) > 0:
                # Reset OBV to start from 0 for each coin's training period
                market_indicators_data_df.loc[coin_idx, 'obv'] -= \
                    market_indicators_data_df.loc[coin_idx[0], 'obv']

    # If a parquet file location is specified, store the files there and return nothing
    if parquet_filename:
        parquet_filepath = f"{parquet_folder}/{parquet_filename}.parquet"
        market_indicators_data_df.to_parquet(parquet_filepath,index=False)
        logger.info(f"Stored market_indicators_data_df with shape {market_indicators_data_df.shape} "
                    f"to {parquet_filepath}.")

        return None

    # If no parquet file is configured then return the df
    else:
        return market_indicators_data_df



@u.timing_decorator
def identify_modeling_cohort(modeling_period_profits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds boolean flag indicating if wallet meets modeling period activity criteria

    Params:
    - modeling_period_profits_df (DataFrame): Input profits data with index wallet_address

    Returns:
    - DataFrame: Original dataframe index wallet_address with added boolean in_wallet_cohort \
        column that indicates if the wallet met the wallet cohort thresholds
    """

    logger.info("Identifying modeling cohort...")

    # Validate date range
    u.assert_period(modeling_period_profits_df,
                    wallets_config['training_data']['modeling_period_start'],
                    wallets_config['training_data']['modeling_period_end'])

    # Calculate modeling period wallet metrics
    modeling_wallets_df = wtf.calculate_wallet_trading_features(modeling_period_profits_df,
                                            wallets_config['training_data']['modeling_period_start'],
                                            wallets_config['training_data']['modeling_period_end'])

    # Extract thresholds
    modeling_min_investment = wallets_config['modeling']['modeling_min_investment']
    modeling_min_coins_traded = wallets_config['modeling']['modeling_min_coins_traded']

    # Create boolean mask for qualifying wallets
    meets_criteria = (
        (modeling_wallets_df['max_investment'] >= modeling_min_investment) &
        (modeling_wallets_df['unique_coins_traded'] >= modeling_min_coins_traded)
    )

    # Log stats about wallet cohort
    total_wallets = len(modeling_wallets_df)
    qualifying_wallets = meets_criteria.sum()
    logger.info(
        f"Identified {qualifying_wallets} qualifying wallets ({100*qualifying_wallets/total_wallets:.2f}% "
        f"of {total_wallets} total wallets with modeling period activity) meeting modeling cohort criteria: "
        f"min_investment=${modeling_min_investment}, min_days={modeling_min_coins_traded}"
    )

    # Add boolean flag column as 1s and 0s
    modeling_wallets_df['in_modeling_cohort'] = meets_criteria.astype(int)


    return modeling_wallets_df



# -----------------------------------
#   Hybrid Index Utility Functions
# -----------------------------------

@u.timing_decorator
def hybridize_wallet_address(
    df: pd.DataFrame,
    hybrid_cw_id_map: Optional[Dict[Tuple[int, str], int]] = None
) -> Tuple[pd.DataFrame, Dict[Tuple[int, str], int]]:
    """
    Maps wallet_address-coin_id pairs to unique integers for efficient indexing.

    Params:
    - df (DataFrame): dataframe with columns ['coin_id','wallet_address']
    - hybrid_cw_id_map (dict): mapping of (wallet,coin) tuples to integers

    Returns:
    - df (DataFrame): input df with hybrid integer keys
    - hybrid_cw_id_map (dict): mapping of (wallet,coin) tuples to integers
    """
    # Create unique mapping for wallet-coin pairs
    unique_pairs = list(zip(df['wallet_address'], df['coin_id']))

    # Generate new mapping if none was provided
    if hybrid_cw_id_map is None:
        hybrid_cw_id_map = {pair: idx for idx, pair in enumerate(set(unique_pairs), 1)}

    # Vectorized mapping of pairs to integers
    df['wallet_address'] = pd.Series(unique_pairs).map(hybrid_cw_id_map)

    return df, hybrid_cw_id_map



def dehybridize_wallet_address(
    df: pd.DataFrame,
    hybrid_cw_id_map: Dict[Tuple[int, str], int],
    hybrid_col_name: str = 'wallet_address'
) -> pd.DataFrame:
    """
    Restores original wallet_address-coin_id pairs from hybrid integer keys.

    Params:
    - df (DataFrame): dataframe with hybrid integer keys in wallet_address
    - hybrid_cw_id_map (dict): mapping of (wallet,coin) tuples to integers
    - hybrid_col_name (str): name of the column containing the hybrid key

    Returns:
    - df (DataFrame): input df with original wallet_address restored
    """
    # Create reverse mapping
    reverse_map = {v: k for k, v in hybrid_cw_id_map.items()}

    # Vectorized mapping back to original tuples
    df[hybrid_col_name] = df[hybrid_col_name].map(reverse_map).map(lambda x: x[0])

    return df


@u.timing_decorator
def upload_hybrid_wallet_mapping(hybrid_cw_id_map: Dict[Tuple[int, str], int]) -> None:
    """
    Uploads the mapping of hybrid indices to all wallet-coin pairs to BigQuery.

    Params:
        hybrid_cw_id_map (Dict[Tuple[int, str], int]): Mapping of (wallet,coin) tuples to hybrid indices
    """
    # 1. Generate upload_df from hybrid map
    # -------------------------------------
    upload_df = pd.DataFrame(
        [(v, k[0], k[1]) for k, v in hybrid_cw_id_map.items()],
        columns=['hybrid_id', 'wallet_id', 'coin_id']
    )
    upload_df['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Set df datatypes
    dtype_mapping = {
        'hybrid_id': int,
        'wallet_id': int,
        'coin_id': str,
        'updated_at': 'datetime64[ns, UTC]'
    }
    upload_df = upload_df.astype(dtype_mapping)

    # 2. Upload to BigQuery
    # ---------------------
    project_id = 'western-verve-411004'
    client = bigquery.Client(project=project_id)

    hybrid_table = f"{project_id}.temp.wallet_modeling_hybrid_id_mapping"
    schema = [
        {'name': 'hybrid_id', 'type': 'int64'},
        {'name': 'wallet_id', 'type': 'int64'},
        {'name': 'coin_id', 'type': 'string'},
        {'name': 'updated_at', 'type': 'datetime'}
    ]

    pandas_gbq.to_gbq(
        upload_df,
        hybrid_table,
        project_id=project_id,
        if_exists='replace',
        table_schema=schema,
        progress_bar=False
    )

    # 3. Create final table with resolved wallet addresses
    # ----------------------------------------------------
    create_query = f"""
    CREATE OR REPLACE TABLE `{hybrid_table}` AS
    SELECT
        h.hybrid_id,
        h.coin_id,
        h.wallet_id,
        w.wallet_address,
        CURRENT_TIMESTAMP() as updated_at
    FROM `{hybrid_table}` h
    LEFT JOIN `reference.wallet_ids` w
        ON h.wallet_id = w.wallet_id
    """

    client.query(create_query).result()
    logger.info(
        'Uploaded complete hybrid ID mapping of %s wallet-coin pairs to %s.',
        len(hybrid_cw_id_map),
        hybrid_table
    )


def merge_wallet_hybrid_data(
    hybrid_parquet_path: str,
    base_parquet_path: str,
    output_parquet_path: str,
    hybrid_suffix: str = "/walletcoin",
    base_suffix: str = "/wallet"
) -> str:
    """
    Merges hybrid and non-hybrid wallet data frames and saves to parquet.

    Params:
    - hybrid_parquet_path (str): Path to hybrid wallet-coin data parquet
    - base_parquet_path (str): Path to base wallet data parquet
    - output_parquet_path (str): Where to save merged result
    - hybrid_suffix (str): Suffix for hybrid columns
    - base_suffix (str): Suffix for base columns

    Returns:
    - str: Path to saved merged parquet file
    """
    # Read input files
    hybrid_df = pd.read_parquet(hybrid_parquet_path)
    base_df = pd.read_parquet(base_parquet_path)

    # Create temporary base address column for merging
    hybrid_df['wallet_address_base'] = hybrid_df.index.values

    # Merge the dataframes
    merged_df = hybrid_df.merge(
        base_df,
        left_on='wallet_address_base',
        right_index=True,
        suffixes=(hybrid_suffix, base_suffix)
    )

    # Clean up temporary column
    merged_df = merged_df.drop('wallet_address_base', axis=1)

    # Save merged result
    merged_df.to_parquet(output_parquet_path, index=True)

    # Clean up memory
    del hybrid_df, base_df, merged_df
    gc.collect()

    return output_parquet_path
