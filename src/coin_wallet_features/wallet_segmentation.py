"""
Functions for assigning wallets to different segments or cohorts, which can then be
used to compare metrics between the features.
"""
import logging
import pandas as pd
import numpy as np

# local module imports
import wallet_features.clustering_features as wcl


# Set up logger at the module level
logger = logging.getLogger(__name__)


# -----------------------------------
#       Main Interface Function
# -----------------------------------

def build_wallet_segmentation(
        wallets_coin_config: dict,
        wallets_config: dict,
        score_suffix: str = None) -> pd.DataFrame:
    """
    Build wallet segmentation DataFrame with score quantiles and optional clusters.

    Params:
    - wallets_coin_config, wallets_config (dicts): dicts from .yaml file
    - score_suffix (str): specifies which score column to load (if applicable)

    Returns:
    - wallet_segmentation_df (pd.DataFrame): df containing segment categories for each wallet
    """
    # Load wallet scores
    wallet_scores_df = load_wallet_scores(
        wallets_coin_config,
        score_suffix
    )
    wallet_segmentation_df = wallet_scores_df.copy()

    # Add "all" segment for full-population aggregations
    wallet_segmentation_df['all_wallets|all'] = 'all'
    wallet_segmentation_df['all_wallets|all'] = wallet_segmentation_df['all_wallets|all'].astype('category')

    # Assign score quantiles if configured
    score_quantiles = wallets_coin_config['wallet_segments'].get('score_segment_quantiles')
    if score_quantiles:
        wallet_segmentation_df = assign_wallet_score_quantiles(
            wallets_coin_config,
            wallet_segmentation_df
        )

    # Add training period cluster labels if configured
    cluster_groups = wallets_coin_config['wallet_segments'].get('training_period_cluster_groups')
    if cluster_groups:
        # Load full-window wallet features to generate clusters
        training_df_path = (
            f"{wallets_config['training_data']['parquet_folder']}"
            "/wallet_training_data_df_full.parquet"
        )
        training_data_df = pd.read_parquet(training_df_path)
        wallet_clusters_df = assign_cluster_labels(
            training_data_df,
            cluster_groups
        )
        # Join and verify no rows dropped
        orig_len = len(wallet_segmentation_df)
        wallet_segmentation_df = wallet_segmentation_df.join(wallet_clusters_df, how='inner')
        joined_len = len(wallet_segmentation_df)
        if joined_len < orig_len:
            raise ValueError(
                f"Join dropped {orig_len - joined_len} rows from original {orig_len} rows"
            )

    return wallet_segmentation_df



# ------------------------------
#         Helper Functions
# ------------------------------

def load_wallet_scores(wallets_coin_config: dict, score_suffix: str = None) -> pd.DataFrame:
    """
    Params:
    - wallets_coin_config (dict): config from .yaml file
    - score_suffix (str): specifies which score column to load (if applicable)

    Returns:
    - wallet_scores_df (DataFrame):
        wallet_address (index): contains all wallet addresses included in any score
        score|{score_name} (float): the predicted score
        residual|{score_name} (float): the residual of the score
    """
    wallet_scores = wallets_coin_config['wallet_segments']['wallet_scores']
    wallet_scores_path = wallets_coin_config['wallet_segments']['wallet_scores_path']
    wallet_scores_df = pd.DataFrame()

    for score_name in wallet_scores:
        score_df = pd.read_parquet(f"{wallet_scores_path}/{score_name}{score_suffix}.parquet")
        feature_cols = []

        # Add scores column
        score_df[f'scores|{score_name}_score'] = score_df[f'score|{score_name}']
        feature_cols.append(f'scores|{score_name}_score')

        # Add residuals column
        if wallets_coin_config['wallet_segments']['wallet_scores_residuals_segments'] is True:
            score_df[f'scores|{score_name}_residual'] = (
                score_df[f'score|{score_name}'] - score_df[f'actual|{score_name}']
            )
            feature_cols.append(f'scores|{score_name}_residual')

        # Add confidence if provided
        if ((wallets_coin_config['wallet_segments']['wallet_scores_confidence_segments'] is True)
            & (f'confidence|{score_name}' in score_df.columns)
            ):
            score_df[f'scores|{score_name}_confidence'] = score_df[f'confidence|{score_name}']
            feature_cols.append(f'scores|{score_name}_confidence')

        # Full outer join with existing results
        wallet_scores_df = (
            score_df[feature_cols] if wallet_scores_df.empty
            else wallet_scores_df.join(score_df[feature_cols], how='outer')
        )

    return wallet_scores_df


def calculate_wallet_quantiles(score_series: pd.Series, quantiles: list[float]) -> pd.DataFrame:
    """
    Assigns each wallet to a single quantile bucket based on score.

    Params:
    - score_series (Series): Score values indexed by wallet_address
    - quantiles (list[float]): List of quantile thresholds in ascending order (e.g. [0.4, 0.6, 0.8])
        Represents percentile breakpoints for binning

    Returns:
    - DataFrame: DataFrame with new score quantile column named 'score_{score_name}_quantile'
        indicating which quantile bucket the wallet belongs to (e.g. '0_40pct')
    """
    # Validate and sort quantiles
    quantiles = sorted(quantiles)
    if not all(0 < q < 1 for q in quantiles):
        raise ValueError("Quantiles must be between 0 and 1")

    # Calculate score thresholds based on data distribution
    score_thresholds = score_series.quantile(quantiles)

    # Create bin edges using actual score distribution
    bin_edges = [-float('inf')] + list(score_thresholds) + [float('inf')]

    # Create labels for each bin (e.g. '0_40pct', '40_60pct', etc)
    bin_labels = []

    for i in range(len(bin_edges) - 1):
        start_pct = int(quantiles[i-1] * 100) if i > 0 else 0
        end_pct = int(quantiles[i] * 100) if i < len(quantiles) else 100
        bin_labels.append(f'{start_pct}_{end_pct}pct')

    # Merge if any boundaries are duplicated
    bin_edges, bin_labels = consolidate_duplicate_bins(bin_edges, bin_labels)

    # Create result DataFrame with quantile assignments
    result_df = pd.DataFrame(index=score_series.index)
    column_name = f'score_quantile|{score_series.name.split("|")[1]}'
    result_df[column_name] = pd.cut(
        score_series,
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
        duplicates='drop'
    ).astype('category')

    # Validate all wallets have segments and segment count is correct
    unique_segments = result_df[column_name].unique()
    if result_df[column_name].isna().any():
        raise ValueError("Some wallets are missing segment assignments")
    if len(bin_edges) != len(bin_labels) + 1:
        raise ValueError(f"Expected {len(bin_edges) + 1} segments but found {len(unique_segments)}: {unique_segments}")

    return result_df


def consolidate_duplicate_bins(bin_edges, bin_labels):
    """
    Consolidate duplicate bin edges by merging their corresponding labels.

    Params:
    - bin_edges (list): List of bin edges with possible duplicates
    - bin_labels (list): List of bin labels corresponding to bin_edges

    Returns:
    - tuple: (new_bin_edges, new_bin_labels) with duplicates resolved
    """
    # 1) Build the ordered list of unique edges
    unique_edges = [bin_edges[0]]
    for edge in bin_edges[1:]:
        if edge != unique_edges[-1]:
            unique_edges.append(edge)

    # 2) For each new interval, merge original labels that fall within it
    new_labels = []
    for i in range(len(unique_edges) - 1):
        left, right = unique_edges[i], unique_edges[i + 1]
        # indices of original intervals fully within [left, right]
        overlap_idxs = [
            j for j, (l, r) in enumerate(zip(bin_edges[:-1], bin_edges[1:]))
            if l >= left and r <= right
        ]
        grp = [bin_labels[j] for j in overlap_idxs]
        if len(grp) == 1:
            new_labels.append(grp[0])
        else:
            # pick prefix from the first non-zero-width interval
            non_zero = [
                j for j in overlap_idxs
                if bin_edges[j] != bin_edges[j + 1]
            ]
            if non_zero:
                prefix = bin_labels[non_zero[0]].split('_')[0]
            else:
                prefix = grp[0].split('_')[0]
            # suffix from the last original label
            suffix = grp[-1].split('_')[-1]
            new_labels.append(f"{prefix}_{suffix}")

    return unique_edges, new_labels


def assign_wallet_score_quantiles(wallets_coin_config, wallet_segmentation_df):
    """
    Generates categorical quantiles for each wallet across each score and residual.

    Params:
    - wallets_coin_config (dict): dict from .yaml file
    - wallet_segmentation_df (df): index wallet_address with columns score|{} and residual|{}
        for all scores in wallet_scores param

    Returns:
    - wallet_segmentation_df (df): the param df but with added score_quantile|{} columns
        for each score and residual
    """
    wallet_scores = wallets_coin_config['wallet_segments']['wallet_scores']
    score_segment_quantiles = wallets_coin_config['wallet_segments']['score_segment_quantiles']
    for score_name in wallet_scores:

        # Append score quantiles
        wallet_quantiles = calculate_wallet_quantiles(wallet_segmentation_df[f'scores|{score_name}_score'],
                                                   score_segment_quantiles)
        wallet_segmentation_df = wallet_segmentation_df.join(wallet_quantiles,how='inner')

        # Append residual quantiles
        if wallets_coin_config['wallet_segments']['wallet_scores_residuals_segments'] is True:
            wallet_quantiles = calculate_wallet_quantiles(wallet_segmentation_df[f'scores|{score_name}_residual'],
                                                    score_segment_quantiles)
            wallet_segmentation_df = wallet_segmentation_df.join(wallet_quantiles,how='inner')

        # Append confidence quantiles
        if ((wallets_coin_config['wallet_segments']['wallet_scores_confidence_segments'] is True)
            & (f'scores|{score_name}_confidence' in wallet_segmentation_df.columns)
            ):
            wallet_quantiles = calculate_wallet_quantiles(wallet_segmentation_df[f'scores|{score_name}_confidence'],
                                                    score_segment_quantiles)
            wallet_segmentation_df = wallet_segmentation_df.join(wallet_quantiles,how='inner')

    return wallet_segmentation_df


def assign_cluster_labels(training_data_df: pd.DataFrame, cluster_groups: list) -> pd.DataFrame:
    """
    Params:
    - training_data_df (DataFrame): Training data for clustering
    - n_clusters (list): List of cluster counts to generate

    Returns:
    - combined_clusters_df (DataFrame): Single DataFrame with all cluster labels
    """
    # Create list of cluster dataframes
    cluster_dfs = []

    for n in cluster_groups:
        wallet_clusters_df = wcl.assign_clusters_from_distances(training_data_df.copy(), [n])
        wallet_clusters_df[f'k{n}_cluster'] = 'cluster_' + wallet_clusters_df[f'k{n}_cluster'].astype(str)
        wallet_clusters_df[f'k{n}_cluster'] = wallet_clusters_df[f'k{n}_cluster'].astype('category')
        wallet_clusters_df = wallet_clusters_df.add_prefix('training_clusters|')
        cluster_dfs.append(wallet_clusters_df)

    # Single join operation
    combined_clusters_df = pd.concat(cluster_dfs, axis=1)

    return combined_clusters_df


def add_feature_quantile_columns(validation_coin_wallet_features_df: pd.DataFrame, n_quantiles: int) -> pd.DataFrame:
    """
    Adds quantile columns for each base metric in the dataframe.

    Params:
    - validation_coin_wallet_features_df (DataFrame): Input features dataframe
    - n_quantiles (int): Number of quantiles to split into (e.g. 4 for quartiles)

    Returns:
    - DataFrame: Original dataframe with added quantile columns
    """
    result_df = validation_coin_wallet_features_df.copy()
    base_cols = [col for col in result_df.columns if col != 'coin_return']

    for col in base_cols:
        quantile_col = f"{col}/quantile_{n_quantiles}"
        try:
            # Convert categorical output to int
            result_df[quantile_col] = pd.qcut(
                result_df[col],
                q=n_quantiles,
                labels=False,  # Returns 0-based indices
                duplicates='drop'
            ) + 1  # Shift to 1-based indices
        except ValueError:
            bins = np.linspace(
                result_df[col].min(),
                result_df[col].max(),
                n_quantiles + 1
            )
            result_df[quantile_col] = pd.cut(
                result_df[col],
                bins=bins,
                labels=False,
                include_lowest=True
            ) + 1

    return result_df
