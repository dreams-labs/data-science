"""
Conducts PCA anlysis of wallet facts and outputs wallet-keyed clustering features
"""
import logging
from typing import Dict,List
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from kneed import KneeLocator

# Local module imports
import utils as u

# set up logger at the module level
logger = logging.getLogger(__name__)


# -----------------------------------
#           Core Interface
# -----------------------------------

@u.timing_decorator
def create_kmeans_cluster_features(
        wallets_config,
        training_data_df,
        include_pca=False,
        include_categorical=False
    ) -> pd.DataFrame:
    """
    Add cluster-related features to the original dataframe.

    Parameters:
    - wallets_config (dict): dict from .yaml file
    - training_data_df (df): DataFrame keyed on wallet_address
    - include_pca (bool): whether to include PCA component features in the output
    - include_categorical (bool): whether to include the cluster number as categorical feature

    Returns:
    - cluster_features_df (df): dataframe with original index plus new cluster features
    """
    # Store original index and extract params
    original_index = training_data_df.index
    n_components = wallets_config['features']['clustering_n_components']
    cluster_counts = wallets_config['features']['clustering_n_clusters']
    include_booleans = wallets_config['features'].get('clustering_include_booleans', False)

    # Get preprocessed data using helper
    logger.info("Preprocessing training data for clustering...")
    scaled_data = preprocess_clustering_data(wallets_config,training_data_df)

    # Apply PCA
    logger.info("Reducing to %s PCA components...", n_components)
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)

    # Create new features DataFrame
    cluster_features_df = pd.DataFrame(index=original_index)

    # Add PCA components once
    if include_pca:
        for i in range(n_components):
            cluster_features_df[f'pca_component_{i+1}'] = pca_result[:, i]

    # Generate features for each cluster count
    for n_clusters in cluster_counts:
        logger.info("Generating clusters for cluster count %s...", n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(pca_result)

        if include_categorical:
            cluster_features_df[f'cluster_{n_clusters}'] = pd.Categorical(
                [f'cluster_{i}' for i in cluster_labels],
                categories=[f'cluster_{i}' for i in range(n_clusters)]
            )

        distances = kmeans.transform(pca_result)
        for i in range(n_clusters):
            cluster_features_df[f'k{n_clusters}/distance_to_cluster_{i}'] = distances[:, i]

        # Add one-hot encoded cluster assignments if configured
        if include_booleans:
            min_cluster_idx = np.argmin(distances, axis=1)
            for i in range(n_clusters):
                cluster_features_df[f'k{n_clusters}_cluster_k{i}'] = (min_cluster_idx == i).astype(int)

    return cluster_features_df



# -----------------------------------
#          Helper Functions
# -----------------------------------

def preprocess_clustering_data(wallets_config: dict, training_data_df: pd.DataFrame) -> np.ndarray:
    """
    Preprocesses data for clustering analysis with consistent scaling.

    Params:
    - wallets_config (dict): dict from .yaml file
    - training_data_df (DataFrame): Input features DataFrame

    Returns:
    - scaled_data (ndarray): Preprocessed and scaled numeric data
    """
    # Validate fill method
    fill_method = wallets_config['features']['clustering_fill_method']
    if fill_method not in ['fill_0', 'fill_mean']:
        raise ValueError(f"Unknown clustering fill value {fill_method}")

    # Select numeric columns
    numeric_df = training_data_df.select_dtypes(include=[np.number])

    # Fill 0s if configured to
    if fill_method == 'fill_0':
        numeric_df = numeric_df.fillna(0)

    # Scale data
    scaled_data = (numeric_df - numeric_df.mean()) / numeric_df.std()

    # Zero out columns that have no variance to avoid NaNs that break PCA)
    scaled_data.loc[:, numeric_df.std() == 0] = 0

    if fill_method == 'fill_mean':
        scaled_data = scaled_data.fillna(0)

    # # Remove low variance features to speed up PCA
    # scaled_data = fs.remove_low_variance_features(
    #     scaled_data,
    #     wallets_config['features']['feature_selection']['variance_threshold']
    #     ,scale_before_selection=False  # the data is already scaled
    # )

    if scaled_data.isna().sum().sum() > 0:
        raise ValueError("Unexpected null values found in pre-PCA scaled data.")

    return scaled_data



# -----------------------------------
#         Utility Functions
# -----------------------------------

def assign_clusters_from_distances(modeling_df: pd.DataFrame, cluster_counts: List[int]) -> pd.DataFrame:
    """
    Assign cluster labels as one-hot encoded features based on minimum distances.

    Params:
    - modeling_df (DataFrame): DataFrame with distance features, indexed by wallet_address
    - cluster_counts (List[int]): List of k values to process [e.g. 2, 4]

    Returns:
    - cluster_assignments_df (DataFrame): One-hot encoded cluster assignments
    """
    cluster_assignments_df = pd.DataFrame(index=modeling_df.index)

    for k in cluster_counts:
        # Get distance columns for this k value
        distance_cols = [f'cluster|k{k}/distance_to_cluster_{i}' for i in range(k)]

        # Find the cluster with minimum distance
        min_cluster_idx = modeling_df[distance_cols].idxmin(axis=1).str[-1].astype(int)

        # Create one-hot encoded columns
        for i in range(k):
            cluster_assignments_df[f'k{k}_cluster_k{i}'] = (min_cluster_idx == i).astype(int)

    return cluster_assignments_df



def optimize_cluster_parameters(
        wallets_config: dict,
        training_data_df: pd.DataFrame,
        max_components: int = 120,
        max_clusters: int = 20,
        minik_batch_size: int = None,
    ) -> Dict[str, Dict]:
    """
    Analyze optimal number of components and clusters using multiple methods.

    Parameters:
    - wallets_config (dict): dict from .yaml file
    - training_data_df: DataFrame with features
    - max_components: maximum number of PCA components to consider
    - max_clusters: maximum number of clusters to consider
    - minik_batch: how many records to use in the MiniBatchKMeans batches

    Returns:
    - Dictionary with analysis results and plots

    Example Use:
    results = wcl.optimize_cluster_parameters(wallets_training_data_df)

    """
    # Filter to numeric, scale, and fill data
    logger.info("Preprocessing training data for PCA analysis...")
    scaled_data = preprocess_clustering_data(wallets_config, training_data_df)

    # Analyze PCA components
    logger.info("Computing PCA with %s components...", max_components)
    pca = PCA(n_components=max_components)
    pca.fit(scaled_data)
    logger.info("PCA calculations complete.")

    # Calculate explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Plot explained variance
    plt.figure(figsize=(12, 4))

    plt.subplot(121)
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot')

    plt.subplot(122)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.axhline(y=0.8, color='k', linestyle='--', label='80% Threshold')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Find optimal components using explained variance threshold
    n_components_80 = min(
        np.argmax(cumulative_variance >= 0.8) + 1,
        max_components
    )

    # Analyze optimal number of clusters
    # Use optimal number of components found above
    logger.info("Assessing optimal cluster counts based on PCA results...")
    reduced_data = pca.transform(scaled_data)[:, :n_components_80]

    # Sample data for silhouette scores if dataset is large
    sample_size = 10000
    logger.info("Using %s sample size for MiniBatchKMeans...", sample_size)
    if reduced_data.shape[0] > sample_size:
        indices = np.random.choice(reduced_data.shape[0], sample_size, replace=False)
        sample_data = reduced_data[indices]

    inertias = []
    silhouette_scores = []

    # Define batch size for mini k
    if not minik_batch_size:
        minik_batch_size = int(np.sqrt(reduced_data.shape[0]))
    for k in range(2, max_clusters + 1):
        logger.debug("Computing scores for %s clusters...", k)
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42,
                                batch_size=minik_batch_size)
        kmeans.fit(reduced_data)
        inertias.append(kmeans.inertia_)

        # Use sampled data for silhouette score
        if reduced_data.shape[0] > sample_size:
            labels = kmeans.predict(sample_data)
            silhouette_scores.append(silhouette_score(sample_data, labels))
        else:
            silhouette_scores.append(silhouette_score(reduced_data, kmeans.labels_))

    # Plot clustering metrics
    plt.figure(figsize=(12, 4))

    plt.subplot(121)
    plt.plot(range(2, max_clusters + 1), inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')

    # Find elbow point
    logger.info("Computing knee locations...")
    kl = KneeLocator(range(2, max_clusters + 1), inertias, curve='convex', direction='decreasing')
    if kl.elbow:
        plt.axvline(x=kl.elbow, color='g', linestyle='--', label=f'Elbow at k={kl.elbow}')
        plt.legend()

    plt.subplot(122)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, 'ro-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')

    # Find optimal k from silhouette score
    optimal_k_silhouette = np.argmax(silhouette_scores) + 2
    plt.axvline(x=optimal_k_silhouette, color='r', linestyle='--',
                label=f'Best k={optimal_k_silhouette}')
    plt.legend()

    plt.tight_layout()
    plt.show()

    results = {
        'optimal_components': {
            'n_components_80_variance': n_components_80,
            'explained_variance_ratio': explained_variance,
            'cumulative_variance': cumulative_variance
        },
        'optimal_clusters': {
            'elbow_k': kl.elbow if kl.elbow else None,
            'silhouette_k': optimal_k_silhouette,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores
        }
    }

    logger.info(f"Number of components explaining 80% variance: \
                {results['optimal_components']['n_components_80_variance']}")
    logger.info(f"Optimal k from elbow method: {results['optimal_clusters']['elbow_k']}")
    logger.info(f"Optimal k from silhouette score: {results['optimal_clusters']['silhouette_k']}")
    return results
