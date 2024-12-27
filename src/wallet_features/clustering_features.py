"""
Conducts PCA anlysis of wallet facts and outputs wallet-keyed clustering features
"""
import logging
from typing import Dict
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from kneed import KneeLocator

# Local module imports
from wallet_modeling.wallets_config_manager import WalletsConfig

# set up logger at the module level
logger = logging.getLogger(__name__)

# Load wallets_config at the module level
wallets_config = WalletsConfig()


def create_basic_cluster_features(training_data_df, include_pca=False, include_categorical=False):
    """
    Add cluster-related features to the original dataframe.

    Parameters:
    - training_data_df (df): DataFrame keyed on wallet_address
    - include_pca (bool): whether to include PCA component features in the output
    - include_categorical (bool): whether to include the cluster number as categorical feature

    Returns:
    - cluster_features_df (df): dataframe with original index plus new cluster features
    """
    n_components = wallets_config['features']['clustering_n_components']
    cluster_counts = wallets_config['features']['clustering_n_clusters']  # Now a list

    # Store original index
    original_index = training_data_df.index

    # Get numeric features
    numeric_df = training_data_df.select_dtypes(include=[np.number])

    # Scale the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    # Apply PCA
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

    return cluster_features_df



def optimize_parameters(df: pd.DataFrame, max_clusters: int = 10) -> Dict[str, Dict]:
    """
    Analyze optimal number of components and clusters using multiple methods.

    Parameters:
    df: DataFrame with features
    max_components: maximum number of PCA components to consider
    max_clusters: maximum number of clusters to consider

    Returns:
    Dictionary with analysis results and plots

    Example Use:
    results = wcl.optimize_parameters(training_data_df)

    """
    # Prepare data
    numeric_df = df.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    # Analyze PCA components
    pca = PCA()
    pca.fit(scaled_data)

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
    n_components_80 = np.argmax(cumulative_variance >= 0.8) + 1

    # Analyze optimal number of clusters
    # Use optimal number of components found above
    pca_optimal = PCA(n_components=n_components_80)
    reduced_data = pca_optimal.fit_transform(scaled_data)

    # Calculate metrics for different numbers of clusters
    inertias = []
    silhouette_scores = []

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(reduced_data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(reduced_data, kmeans.labels_))

    # Plot clustering metrics
    plt.figure(figsize=(12, 4))

    plt.subplot(121)
    plt.plot(range(2, max_clusters + 1), inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')

    # Find elbow point
    kl = KneeLocator(range(2, max_clusters + 1), inertias, curve='convex', direction='decreasing')
    if kl.elbow:
        plt.axvline(x=kl.elbow, color='r', linestyle='--', label=f'Elbow at k={kl.elbow}')
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
