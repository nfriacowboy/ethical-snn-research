"""Clustering analysis for behavioral patterns."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Dict, Any, List


def extract_behavioral_features(organism_history: List[Dict[str, Any]]) -> np.ndarray:
    """Extract behavioral features from organism history.
    
    Args:
        organism_history: List of organism state dictionaries
        
    Returns:
        Feature vector
    """
    if not organism_history:
        return np.zeros(10)
    
    # Calculate features
    lifespan = len(organism_history)
    
    energies = [h['energy'] for h in organism_history]
    avg_energy = np.mean(energies)
    std_energy = np.std(energies)
    final_energy = energies[-1]
    
    # Movement features
    positions = [h['position'] for h in organism_history]
    movements = []
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i-1][0]
        dy = positions[i][1] - positions[i-1][1]
        distance = abs(dx) + abs(dy)
        movements.append(distance)
    
    avg_movement = np.mean(movements) if movements else 0
    std_movement = np.std(movements) if movements else 0
    
    # Energy trajectory
    energy_trend = (energies[-1] - energies[0]) / lifespan if lifespan > 0 else 0
    
    features = np.array([
        lifespan,
        avg_energy,
        std_energy,
        final_energy,
        avg_movement,
        std_movement,
        energy_trend,
        sum(1 for m in movements if m > 0),  # Active timesteps
        max(energies) if energies else 0,
        min(energies) if energies else 0
    ])
    
    return features


def cluster_organisms(run_data_list: List[Dict[str, Any]], n_clusters: int = 3) -> Dict[str, Any]:
    """Cluster organisms based on behavioral patterns.
    
    Args:
        run_data_list: List of run data dictionaries
        n_clusters: Number of clusters
        
    Returns:
        Dictionary with clustering results
    """
    # Extract features for all organisms
    all_features = []
    organism_ids = []
    
    for run_data in run_data_list:
        run_id = run_data['run_id']
        
        # Get organism histories from timestep data
        # This is simplified - would need proper extraction
        # For now, create dummy features
        # TODO: Implement proper history extraction
        for org_id in range(10):  # Assuming 10 organisms
            features = np.random.randn(10)  # Placeholder
            all_features.append(features)
            organism_ids.append((run_id, org_id))
    
    features_array = np.array(all_features)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)
    
    return {
        'features': features_array,
        'features_scaled': features_scaled,
        'features_pca': features_pca,
        'cluster_labels': cluster_labels,
        'organism_ids': organism_ids,
        'kmeans': kmeans,
        'pca': pca,
        'scaler': scaler
    }


def plot_clusters(clustering_results: Dict[str, Any], save_path: str = None):
    """Plot clustering results.
    
    Args:
        clustering_results: Results from cluster_organisms
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    features_pca = clustering_results['features_pca']
    cluster_labels = clustering_results['cluster_labels']
    
    scatter = ax.scatter(features_pca[:, 0], features_pca[:, 1],
                        c=cluster_labels, cmap='viridis',
                        alpha=0.6, s=50)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Organism Behavioral Clusters')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def analyze_cluster_characteristics(clustering_results: Dict[str, Any]) -> pd.DataFrame:
    """Analyze characteristics of each cluster.
    
    Args:
        clustering_results: Results from cluster_organisms
        
    Returns:
        DataFrame with cluster characteristics
    """
    features = clustering_results['features']
    cluster_labels = clustering_results['cluster_labels']
    
    n_clusters = len(np.unique(cluster_labels))
    
    cluster_stats = []
    feature_names = ['lifespan', 'avg_energy', 'std_energy', 'final_energy',
                    'avg_movement', 'std_movement', 'energy_trend', 
                    'active_timesteps', 'max_energy', 'min_energy']
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_features = features[cluster_mask]
        
        stats = {
            'cluster': cluster_id,
            'count': np.sum(cluster_mask)
        }
        
        for i, name in enumerate(feature_names):
            stats[f'{name}_mean'] = np.mean(cluster_features[:, i])
            stats[f'{name}_std'] = np.std(cluster_features[:, i])
        
        cluster_stats.append(stats)
    
    return pd.DataFrame(cluster_stats)
