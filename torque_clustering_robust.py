# -*- coding: utf-8 -*-
"""
Torque Clustering - Robust Implementation

A physics-inspired clustering algorithm where data points are treated as particles
that exert forces on each other. Points with higher "mass" pull other points more
strongly, and the pull weakens with distance.

This implementation consolidates all functionality from the original notebook
into a single, well-organized module.

Author: Consolidated from Colab notebook
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple, Literal


# =============================================================================
# MASS CALCULATION
# =============================================================================

def calculate_mass(
    data: np.ndarray,
    method: Literal['uniform', 'density', 'feature_sum', 'feature_variance'] = 'uniform',
    k: int = 5
) -> np.ndarray:
    """
    Calculate the "mass" of each data point.

    In Torque Clustering, mass represents the "importance" or "influence" of a point.
    Points with higher mass exert stronger gravitational pull on other points.

    Args:
        data: A NumPy array of shape (n_samples, n_features) where each row is a data point.
        method: The method to use for mass calculation:
            - 'uniform': All points have equal mass (simplest, unbiased).
            - 'density': Mass proportional to local density (dense regions have higher mass).
            - 'feature_sum': Mass proportional to sum of absolute feature values.
            - 'feature_variance': Mass proportional to variance across features.
        k: Number of neighbors for density estimation (only used when method='density').

    Returns:
        A NumPy array of shape (n_samples,) containing the mass of each data point.
        Masses are normalized (sum to 1) for non-uniform methods to ensure stability.

    Raises:
        ValueError: If an invalid mass calculation method is provided.

    Example:
        >>> data = np.array([[1, 2], [3, 4], [5, 6]])
        >>> masses = calculate_mass(data, method='uniform')
        >>> print(masses)  # [1. 1. 1.]
    """
    n_samples = data.shape[0]

    if method == 'uniform':
        # All points have equal influence - simplest approach
        return np.ones(n_samples)

    elif method == 'density':
        # Mass proportional to local density
        # Points in dense regions have higher mass, pulling sparse points toward them
        # This helps clusters form around dense cores

        # Compute pairwise distances
        distances = squareform(pdist(data, 'euclidean'))

        # Sort distances for each point to find k-nearest neighbors
        distances_sorted = np.sort(distances, axis=1)

        # Get distances to k nearest neighbors (excluding self at index 0)
        # Limit k to n_samples - 1 to avoid index errors
        k_actual = min(k, n_samples - 1)
        knn_distances = distances_sorted[:, 1:k_actual + 1]

        # Local density is inverse of mean distance to k neighbors
        # Add small epsilon to avoid division by zero
        local_density = 1.0 / (np.mean(knn_distances, axis=1) + 1e-9)

        # Normalize so masses sum to 1
        return local_density / np.sum(local_density)

    elif method == 'feature_sum':
        # Mass proportional to the magnitude of feature values
        # Points with larger feature values have higher mass
        feature_sums = np.sum(np.abs(data), axis=1)

        # Handle edge case where all values are zero
        total = np.sum(feature_sums)
        if total == 0:
            return np.ones(n_samples) / n_samples

        return feature_sums / total

    elif method == 'feature_variance':
        # Mass proportional to variance across features
        # Points with high variance in their features have higher mass
        variances = np.var(data, axis=1)

        # Handle edge case where all variances are zero
        total = np.sum(variances)
        if total == 0:
            return np.ones(n_samples) / n_samples

        return variances / total

    else:
        raise ValueError(
            f"Invalid mass calculation method: '{method}'. "
            f"Must be one of: 'uniform', 'density', 'feature_sum', 'feature_variance'."
        )


# =============================================================================
# DISTANCE CALCULATION
# =============================================================================

def calculate_distance_matrix(data: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise Euclidean distance matrix for the dataset.

    Args:
        data: A NumPy array of shape (n_samples, n_features).

    Returns:
        A symmetric NumPy array of shape (n_samples, n_samples) where
        element [i, j] is the Euclidean distance between points i and j.
        Diagonal elements are 0 (distance from a point to itself).

    Example:
        >>> data = np.array([[0, 0], [3, 4]])
        >>> dist_matrix = calculate_distance_matrix(data)
        >>> print(dist_matrix[0, 1])  # 5.0 (3-4-5 triangle)
    """
    return squareform(pdist(data, 'euclidean'))


# =============================================================================
# TORQUE CALCULATION
# =============================================================================

def calculate_torque(
    data: np.ndarray,
    masses: np.ndarray,
    distances: np.ndarray,
    k_neighbors: Optional[int] = None,
    distance_power: float = 1.0
) -> np.ndarray:
    """
    Calculate the "torque" (net force vector) exerted on each data point by others.

    The torque represents the direction and magnitude each point should move.
    It's computed as a weighted sum of direction vectors from each point to others,
    where weights depend on mass and inverse distance.

    Physics analogy:
        - Each point is a particle with mass
        - Particles exert gravitational pull on each other
        - Torque is the net force vector acting on each particle

    Args:
        data: A NumPy array of shape (n_samples, n_features).
        masses: A NumPy array of shape (n_samples,) containing mass of each point.
        distances: A NumPy array of shape (n_samples, n_samples) of pairwise distances.
        k_neighbors: If specified, only consider k nearest neighbors for torque calculation.
                    If None, all points contribute to torque (global influence).
        distance_power: Power of distance in denominator (1 for inverse, 2 for inverse square).
                       Higher values make torque more local (distant points contribute less).

    Returns:
        A NumPy array of shape (n_samples, n_features) representing the net torque
        vector on each data point. This indicates direction and strength of pull.

    Note:
        The returned torques are normalized by total weight when k_neighbors is used,
        making the algorithm more stable with varying k values.
    """
    n_samples = data.shape[0]
    torques = np.zeros_like(data, dtype=float)

    for i in range(n_samples):
        # Determine which neighbors to consider
        if k_neighbors is None:
            # Use all points except self
            neighbors = np.concatenate([np.arange(i), np.arange(i + 1, n_samples)])
        else:
            # Use k-nearest neighbors (excluding self)
            distances_from_i = distances[i, :]
            # Get indices sorted by distance, exclude self (distance 0)
            sorted_indices = np.argsort(distances_from_i)
            # Skip index 0 which is self (distance 0), take next k
            k_actual = min(k_neighbors, n_samples - 1)
            neighbors = sorted_indices[1:k_actual + 1]

        total_weight = 0.0

        for j in neighbors:
            # Direction vector pointing from i toward j
            direction_vector = data[j] - data[i]

            # Distance between points i and j
            distance = distances[i, j]

            # Weight: mass of j divided by distance^power
            # Higher mass = stronger pull, larger distance = weaker pull
            weight = masses[j] / (distance ** distance_power + 1e-9)

            # Accumulate weighted direction
            torques[i] += weight * direction_vector
            total_weight += weight

        # Normalize by total weight for stability (weighted average)
        if total_weight > 0 and k_neighbors is not None:
            torques[i] /= total_weight

    return torques


# =============================================================================
# NEAREST HEAVIER POINT (for connection-based clustering)
# =============================================================================

def find_nearest_heavier_point(
    masses: np.ndarray,
    distances: np.ndarray
) -> np.ndarray:
    """
    For each point, find the nearest point that has a strictly higher mass.

    This creates a directed graph where edges point from lighter to heavier points.
    Used in connection-based clustering to identify cluster structure.

    Args:
        masses: A NumPy array of shape (n_samples,) containing mass of each point.
        distances: A NumPy array of shape (n_samples, n_samples) of pairwise distances.

    Returns:
        A NumPy array of shape (n_samples,) where element i contains the index
        of the nearest heavier neighbor, or -1 if no heavier neighbor exists
        (i.e., this point has the maximum mass).

    Example:
        If point 0 has mass 1.0 and points 1, 2 have mass 2.0, 3.0:
        - Point 0 will point to whichever of 1, 2 is closer
        - The heaviest point(s) will have -1 (no heavier neighbor)
    """
    n_samples = masses.shape[0]
    nearest_heavier = np.full(n_samples, -1, dtype=int)

    for i in range(n_samples):
        # Find all points with strictly higher mass
        heavier_indices = np.where(masses > masses[i])[0]

        if heavier_indices.size > 0:
            # Among heavier points, find the nearest one
            distances_to_heavier = distances[i, heavier_indices]
            nearest_idx = np.argmin(distances_to_heavier)
            nearest_heavier[i] = heavier_indices[nearest_idx]

    return nearest_heavier


# =============================================================================
# CLUSTERING METHOD 1: ITERATIVE POSITION UPDATE
# =============================================================================

def torque_clustering_iterative(
    data: np.ndarray,
    mass_method: str = 'uniform',
    learning_rate: float = 0.01,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    k_neighbors: Optional[int] = None,
    k_density: int = 5,
    n_clusters: int = 3,
    final_clustering: Literal['kmeans', 'nearest'] = 'kmeans',
    distance_power: float = 1.0,
    visualize_iterations: bool = False,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Perform Torque Clustering using iterative position updates.

    Algorithm:
        1. Assign mass to each point based on chosen method
        2. Iteratively:
            a. Calculate torque (net force) on each point
            b. Move points in direction of torque: position += learning_rate * torque
            c. Check convergence (if average movement < tolerance, stop)
        3. Apply final clustering (KMeans or nearest neighbor) on transformed positions

    This approach physically "moves" points toward their attractors, causing
    points in the same cluster to converge toward each other.

    Args:
        data: Input data array of shape (n_samples, n_features).
        mass_method: Method for mass calculation ('uniform', 'density', 'feature_sum', 'feature_variance').
        learning_rate: Step size for position updates. Smaller = more stable but slower.
        max_iterations: Maximum number of iterations before stopping.
        tolerance: Convergence threshold. Stop if average movement < tolerance.
        k_neighbors: Number of neighbors for torque calculation. None = use all points.
        k_density: Number of neighbors for density-based mass calculation.
        n_clusters: Number of clusters for final KMeans clustering.
        final_clustering: Method for final cluster assignment:
            - 'kmeans': Apply KMeans on transformed positions
            - 'nearest': Assign to nearest point (creates many micro-clusters)
        distance_power: Power of distance in torque calculation (1 or 2).
        visualize_iterations: If True, show live plot of point movement (2D data only).
        verbose: If True, print convergence information.

    Returns:
        cluster_labels: Array of cluster labels for each point.
        transformed_data: The data after torque-based position updates.
        iterations: Number of iterations performed.

    Example:
        >>> data, true_labels = make_blobs(n_samples=300, centers=3)
        >>> labels, transformed, iters = torque_clustering_iterative(data, n_clusters=3)
    """
    n_samples, n_features = data.shape

    # Calculate mass for each point
    masses = calculate_mass(data, method=mass_method, k=k_density)

    # Work with a copy to preserve original data
    data_transformed = data.copy().astype(float)

    # Setup visualization if requested
    if visualize_iterations and n_features >= 2:
        plt.figure(figsize=(8, 6))
        plt.ion()  # Interactive mode for live updates

    iterations_performed = 0
    previous_positions = None

    for iteration in range(max_iterations):
        # Calculate current distance matrix
        distances = calculate_distance_matrix(data_transformed)

        # Calculate torque on each point
        torques = calculate_torque(
            data_transformed, masses, distances,
            k_neighbors=k_neighbors,
            distance_power=distance_power
        )

        # Update positions: move in direction of torque
        data_transformed += learning_rate * torques

        iterations_performed = iteration + 1

        # Visualize current state
        if visualize_iterations and n_features >= 2:
            plt.clf()
            plt.scatter(data_transformed[:, 0], data_transformed[:, 1],
                       c='blue', alpha=0.5, s=30)
            plt.title(f'Torque Clustering - Iteration {iteration + 1}')
            plt.xlabel('Feature 1: Mean HU')
            plt.ylabel('Feature 2: GLCM Contrast')
            plt.draw()
            plt.pause(0.05)

        # Check for convergence
        if previous_positions is not None:
            # Average movement per point
            avg_change = np.linalg.norm(data_transformed - previous_positions) / n_samples
            if avg_change < tolerance:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations (avg change: {avg_change:.6f})")
                break

        previous_positions = data_transformed.copy()

    else:
        if verbose:
            print(f"Reached maximum iterations ({max_iterations})")

    # Close visualization
    if visualize_iterations and n_features >= 2:
        plt.ioff()
        plt.show()

    # Final clustering on transformed data
    if final_clustering == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(data_transformed)
    elif final_clustering == 'nearest':
        # Simple nearest-neighbor: each point's label is its own index
        # This creates many micro-clusters (not very useful in practice)
        distances_final = calculate_distance_matrix(data_transformed)
        cluster_labels = np.argmin(distances_final, axis=1)
    else:
        raise ValueError(f"Invalid final_clustering: '{final_clustering}'")

    return cluster_labels, data_transformed, iterations_performed


# =============================================================================
# CLUSTERING METHOD 2: CONNECTION-BASED (ABNORMAL CONNECTION REMOVAL)
# =============================================================================

def torque_clustering_connection_based(
    data: np.ndarray,
    mass_method: str = 'uniform',
    k_density: int = 5,
    torque_threshold: Union[str, float] = 'median',
    verbose: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Perform Torque Clustering using connection-based approach.

    Algorithm:
        1. Assign mass to each point
        2. Build a directed graph: each point connects to its nearest heavier neighbor
        3. Calculate torque for each connection: torque = (mass_i * mass_j) / distance^2
        4. Identify "abnormal" connections where torque exceeds threshold
        5. Remove abnormal connections to separate clusters
        6. Assign cluster labels based on connected components

    This approach identifies cluster boundaries as "weak" connections (low torque)
    between dense regions (high torque within clusters).

    Args:
        data: Input data array of shape (n_samples, n_features).
        mass_method: Method for mass calculation.
        k_density: Number of neighbors for density-based mass calculation.
        torque_threshold: Threshold for identifying abnormal connections:
            - 'median': Use median of all torque values
            - 'mean': Use mean of all torque values
            - float: Use specified value directly
        verbose: If True, print clustering information.

    Returns:
        cluster_labels: Array of cluster labels for each point.
        n_clusters: Number of clusters found.

    Note:
        This method automatically determines the number of clusters based on
        the torque threshold, unlike the iterative method which requires n_clusters.
    """
    n_samples = data.shape[0]

    # Calculate masses and distances
    masses = calculate_mass(data, method=mass_method, k=k_density)
    distances = calculate_distance_matrix(data)

    # Find nearest heavier neighbor for each point
    nearest_heavier = find_nearest_heavier_point(masses, distances)

    # Calculate torque matrix: torque[i,j] = (mass_i * mass_j) / distance_ij^2
    mass_product = masses[:, None] * masses[None, :]
    distance_squared = distances ** 2
    torque_matrix = mass_product / (distance_squared + 1e-9)

    # Determine threshold for abnormal connections
    if torque_threshold == 'median':
        threshold = np.median(torque_matrix)
    elif torque_threshold == 'mean':
        threshold = np.mean(torque_matrix)
    elif isinstance(torque_threshold, (int, float)):
        threshold = float(torque_threshold)
    else:
        raise ValueError(
            f"Invalid torque_threshold: '{torque_threshold}'. "
            f"Must be 'median', 'mean', or a numeric value."
        )

    # Identify abnormal connections (torque > threshold indicates strong connection)
    # We cut these to separate clusters
    abnormal_connections = set()
    for i in range(n_samples):
        if nearest_heavier[i] != -1:
            if torque_matrix[i, nearest_heavier[i]] > threshold:
                abnormal_connections.add((i, nearest_heavier[i]))

    # Build clusters by following connections (excluding abnormal ones)
    visited = np.zeros(n_samples, dtype=bool)
    clusters = []

    for i in range(n_samples):
        if not visited[i]:
            cluster = [i]
            visited[i] = True

            # Follow the chain of nearest heavier neighbors
            current = i
            while nearest_heavier[current] != -1:
                # Stop if this connection is abnormal
                if (current, nearest_heavier[current]) in abnormal_connections:
                    break

                next_point = nearest_heavier[current]

                # Stop if we hit an already visited point (avoid cycles)
                if visited[next_point]:
                    break

                cluster.append(next_point)
                visited[next_point] = True
                current = next_point

            clusters.append(cluster)

    # Assign cluster labels
    cluster_labels = np.full(n_samples, -1, dtype=int)
    for label, cluster in enumerate(clusters):
        for point in cluster:
            cluster_labels[point] = label

    n_clusters_found = len(clusters)

    if verbose:
        print(f"Connection-based clustering found {n_clusters_found} clusters")
        print(f"Threshold used: {threshold:.6f}")
        print(f"Abnormal connections removed: {len(abnormal_connections)}")

    return cluster_labels, n_clusters_found


# =============================================================================
# UNIFIED INTERFACE
# =============================================================================

def torque_clustering(
    data: np.ndarray,
    method: Literal['iterative', 'connection'] = 'iterative',
    mass_method: str = 'uniform',
    k_density: int = 5,
    # Iterative method parameters
    learning_rate: float = 0.01,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    k_neighbors: Optional[int] = None,
    n_clusters: int = 3,
    final_clustering: str = 'kmeans',
    distance_power: float = 1.0,
    # Connection method parameters
    torque_threshold: Union[str, float] = 'median',
    # Common parameters
    visualize_iterations: bool = False,
    verbose: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Unified interface for Torque Clustering.

    Provides access to both clustering methods through a single function.

    Args:
        data: Input data array of shape (n_samples, n_features).
        method: Clustering method to use:
            - 'iterative': Position update method (good for visualization)
            - 'connection': Connection-based method (auto-determines n_clusters)
        mass_method: Mass calculation method ('uniform', 'density', 'feature_sum', 'feature_variance').
        k_density: Neighbors for density-based mass.

        # Iterative method only:
        learning_rate: Step size for updates.
        max_iterations: Maximum iterations.
        tolerance: Convergence threshold.
        k_neighbors: Neighbors for torque (None = all).
        n_clusters: Number of clusters for KMeans.
        final_clustering: 'kmeans' or 'nearest'.
        distance_power: Distance power in torque formula.

        # Connection method only:
        torque_threshold: Threshold for abnormal connections.

        # Common:
        visualize_iterations: Show live plot (iterative only).
        verbose: Print progress information.

    Returns:
        cluster_labels: Array of cluster labels.
        transformed_data: Transformed data (iterative method only, None for connection).
    """
    if method == 'iterative':
        labels, transformed, _ = torque_clustering_iterative(
            data=data,
            mass_method=mass_method,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            tolerance=tolerance,
            k_neighbors=k_neighbors,
            k_density=k_density,
            n_clusters=n_clusters,
            final_clustering=final_clustering,
            distance_power=distance_power,
            visualize_iterations=visualize_iterations,
            verbose=verbose
        )
        return labels, transformed

    elif method == 'connection':
        labels, _ = torque_clustering_connection_based(
            data=data,
            mass_method=mass_method,
            k_density=k_density,
            torque_threshold=torque_threshold,
            verbose=verbose
        )
        return labels, None

    else:
        raise ValueError(f"Invalid method: '{method}'. Must be 'iterative' or 'connection'.")


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_clustering(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray
) -> Tuple[float, float]:
    """
    Evaluate clustering performance using standard metrics.

    Args:
        true_labels: Ground truth cluster labels.
        predicted_labels: Predicted cluster labels.

    Returns:
        ari: Adjusted Rand Index (-1 to 1, higher is better, 1 = perfect).
        nmi: Normalized Mutual Information (0 to 1, higher is better, 1 = perfect).
    """
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    return ari, nmi


def train_and_evaluate(
    data: np.ndarray,
    true_labels: np.ndarray,
    method: str = 'iterative',
    **kwargs
) -> Tuple[float, float, np.ndarray]:
    """
    Convenience function to cluster data and evaluate performance.

    Args:
        data: Input data.
        true_labels: Ground truth labels.
        method: Clustering method ('iterative' or 'connection').
        **kwargs: Additional arguments passed to torque_clustering.

    Returns:
        ari: Adjusted Rand Index.
        nmi: Normalized Mutual Information.
        predicted_labels: Predicted cluster labels.
    """
    predicted_labels, _ = torque_clustering(data, method=method, verbose=False, **kwargs)
    ari, nmi = evaluate_clustering(true_labels, predicted_labels)
    return ari, nmi, predicted_labels


# =============================================================================
# DATA GENERATION
# =============================================================================

def create_synthetic_data(
    n_samples: int = 300,
    n_clusters: int = 3,
    n_features: int = 2,
    cluster_std: float = 1.0,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic clustered data for testing.

    Args:
        n_samples: Total number of samples.
        n_clusters: Number of clusters to generate.
        n_features: Number of features per sample.
        cluster_std: Standard deviation of clusters (higher = more overlap).
        random_state: Random seed for reproducibility.

    Returns:
        data: Generated data of shape (n_samples, n_features).
        labels: True cluster labels of shape (n_samples,).
    """
    data, labels = make_blobs(
        n_samples=n_samples,
        centers=n_clusters,
        n_features=n_features,
        cluster_std=cluster_std,
        random_state=random_state
    )
    return data, labels


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_clustering_results(
    data: np.ndarray,
    labels_dict: dict,
    true_labels: Optional[np.ndarray] = None,
    title: str = "Clustering Results"
) -> None:
    """
    Visualize clustering results with multiple methods side by side.

    Args:
        data: Original data (uses first 2 features for 2D plot).
        labels_dict: Dictionary mapping method names to cluster labels.
        true_labels: Ground truth labels (optional).
        title: Overall plot title.
    """
    n_plots = len(labels_dict) + (1 if true_labels is not None else 0)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    plot_idx = 0

    # Plot ground truth if available
    if true_labels is not None:
        axes[plot_idx].scatter(data[:, 0], data[:, 1], c=true_labels, cmap='viridis', s=30, alpha=0.7)
        axes[plot_idx].set_title('Ground Truth')
        axes[plot_idx].set_xlabel('Feature 1: Mean HU')
        axes[plot_idx].set_ylabel('Feature 2: GLCM Contrast')
        plot_idx += 1

    # Plot each clustering result
    for name, labels in labels_dict.items():
        axes[plot_idx].scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=30, alpha=0.7)
        axes[plot_idx].set_title(name)
        axes[plot_idx].set_xlabel('Feature 1: Mean HU')
        axes[plot_idx].set_ylabel('Feature 2: GLCM Contrast')
        plot_idx += 1

    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_comparison(
    data_train: np.ndarray,
    data_test: np.ndarray,
    labels_train: np.ndarray,
    labels_test: np.ndarray,
    true_train: np.ndarray,
    true_test: np.ndarray,
    method_name: str
) -> None:
    """
    Plot train/test comparison for a clustering method.

    Args:
        data_train, data_test: Training and test data.
        labels_train, labels_test: Predicted labels.
        true_train, true_test: True labels.
        method_name: Name of the clustering method.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Ground truth
    axes[0].scatter(data_train[:, 0], data_train[:, 1], c=true_train,
                   cmap='viridis', marker='o', s=30, label='Train', alpha=0.7)
    axes[0].scatter(data_test[:, 0], data_test[:, 1], c=true_test,
                   cmap='viridis', marker='x', s=50, label='Test', alpha=0.7)
    axes[0].set_title('Ground Truth')
    axes[0].legend()

    # Predicted
    axes[1].scatter(data_train[:, 0], data_train[:, 1], c=labels_train,
                   cmap='viridis', marker='o', s=30, label='Train', alpha=0.7)
    axes[1].scatter(data_test[:, 0], data_test[:, 1], c=labels_test,
                   cmap='viridis', marker='x', s=50, label='Test', alpha=0.7)
    axes[1].set_title(f'Predicted ({method_name})')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN EXECUTION - COMPREHENSIVE EXPERIMENTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TORQUE CLUSTERING - COMPREHENSIVE EVALUATION")
    print("=" * 70)

    # Generate synthetic data
    print("\n[1] Generating synthetic data...")
    data, true_labels = create_synthetic_data(n_samples=300, n_clusters=3, cluster_std=1.0)
    X_train, X_test, y_train, y_test = train_test_split(
        data, true_labels, test_size=0.2, random_state=42
    )
    print(f"    Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Store results for summary
    results = []

    # ==========================================================================
    # ITERATIVE METHOD EXPERIMENTS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ITERATIVE METHOD EXPERIMENTS")
    print("=" * 70)

    # Experiment 1: Uniform Mass, All Neighbors
    print("\n[2] Uniform Mass, All Neighbors")
    ari_train, nmi_train, labels_train = train_and_evaluate(
        X_train, y_train, method='iterative', mass_method='uniform', k_neighbors=None, n_clusters=3
    )
    ari_test, nmi_test, labels_test = train_and_evaluate(
        X_test, y_test, method='iterative', mass_method='uniform', k_neighbors=None, n_clusters=3
    )
    print(f"    Train - ARI: {ari_train:.4f}, NMI: {nmi_train:.4f}")
    print(f"    Test  - ARI: {ari_test:.4f}, NMI: {nmi_test:.4f}")
    results.append(('Uniform (All)', ari_train, nmi_train, ari_test, nmi_test))

    # Experiment 2: Uniform Mass, 5 Neighbors
    print("\n[3] Uniform Mass, 5 Neighbors")
    ari_train, nmi_train, _ = train_and_evaluate(
        X_train, y_train, method='iterative', mass_method='uniform', k_neighbors=5, n_clusters=3
    )
    ari_test, nmi_test, _ = train_and_evaluate(
        X_test, y_test, method='iterative', mass_method='uniform', k_neighbors=5, n_clusters=3
    )
    print(f"    Train - ARI: {ari_train:.4f}, NMI: {nmi_train:.4f}")
    print(f"    Test  - ARI: {ari_test:.4f}, NMI: {nmi_test:.4f}")
    results.append(('Uniform (k=5)', ari_train, nmi_train, ari_test, nmi_test))

    # Experiment 3: Density Mass, All Neighbors
    print("\n[4] Density Mass, All Neighbors")
    ari_train, nmi_train, _ = train_and_evaluate(
        X_train, y_train, method='iterative', mass_method='density', k_neighbors=None, k_density=5, n_clusters=3
    )
    ari_test, nmi_test, _ = train_and_evaluate(
        X_test, y_test, method='iterative', mass_method='density', k_neighbors=None, k_density=5, n_clusters=3
    )
    print(f"    Train - ARI: {ari_train:.4f}, NMI: {nmi_train:.4f}")
    print(f"    Test  - ARI: {ari_test:.4f}, NMI: {nmi_test:.4f}")
    results.append(('Density (All, k=5)', ari_train, nmi_train, ari_test, nmi_test))

    # Experiment 4: Density Mass, 5 Neighbors
    print("\n[5] Density Mass, 5 Neighbors")
    ari_train, nmi_train, _ = train_and_evaluate(
        X_train, y_train, method='iterative', mass_method='density', k_neighbors=5, k_density=5, n_clusters=3
    )
    ari_test, nmi_test, _ = train_and_evaluate(
        X_test, y_test, method='iterative', mass_method='density', k_neighbors=5, k_density=5, n_clusters=3
    )
    print(f"    Train - ARI: {ari_train:.4f}, NMI: {nmi_train:.4f}")
    print(f"    Test  - ARI: {ari_test:.4f}, NMI: {nmi_test:.4f}")
    results.append(('Density (k=5, k=5)', ari_train, nmi_train, ari_test, nmi_test))

    # Experiment 5: Feature Sum Mass
    print("\n[6] Feature Sum Mass")
    ari_train, nmi_train, _ = train_and_evaluate(
        X_train, y_train, method='iterative', mass_method='feature_sum', n_clusters=3
    )
    ari_test, nmi_test, _ = train_and_evaluate(
        X_test, y_test, method='iterative', mass_method='feature_sum', n_clusters=3
    )
    print(f"    Train - ARI: {ari_train:.4f}, NMI: {nmi_train:.4f}")
    print(f"    Test  - ARI: {ari_test:.4f}, NMI: {nmi_test:.4f}")
    results.append(('Feature Sum', ari_train, nmi_train, ari_test, nmi_test))

    # Experiment 6: Feature Variance Mass
    print("\n[7] Feature Variance Mass")
    ari_train, nmi_train, _ = train_and_evaluate(
        X_train, y_train, method='iterative', mass_method='feature_variance', n_clusters=3
    )
    ari_test, nmi_test, _ = train_and_evaluate(
        X_test, y_test, method='iterative', mass_method='feature_variance', n_clusters=3
    )
    print(f"    Train - ARI: {ari_train:.4f}, NMI: {nmi_train:.4f}")
    print(f"    Test  - ARI: {ari_test:.4f}, NMI: {nmi_test:.4f}")
    results.append(('Feature Variance', ari_train, nmi_train, ari_test, nmi_test))

    # ==========================================================================
    # CONNECTION-BASED METHOD EXPERIMENTS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("CONNECTION-BASED METHOD EXPERIMENTS")
    print("=" * 70)

    # Experiment 7: Connection-based, Uniform Mass, Median Threshold
    print("\n[8] Connection-based, Uniform Mass, Median Threshold")
    ari_train, nmi_train, _ = train_and_evaluate(
        X_train, y_train, method='connection', mass_method='uniform', torque_threshold='median'
    )
    ari_test, nmi_test, _ = train_and_evaluate(
        X_test, y_test, method='connection', mass_method='uniform', torque_threshold='median'
    )
    print(f"    Train - ARI: {ari_train:.4f}, NMI: {nmi_train:.4f}")
    print(f"    Test  - ARI: {ari_test:.4f}, NMI: {nmi_test:.4f}")
    results.append(('Connection (Uniform, Median)', ari_train, nmi_train, ari_test, nmi_test))

    # Experiment 8: Connection-based, Density Mass, Median Threshold
    print("\n[9] Connection-based, Density Mass, Median Threshold")
    ari_train, nmi_train, _ = train_and_evaluate(
        X_train, y_train, method='connection', mass_method='density', torque_threshold='median'
    )
    ari_test, nmi_test, _ = train_and_evaluate(
        X_test, y_test, method='connection', mass_method='density', torque_threshold='median'
    )
    print(f"    Train - ARI: {ari_train:.4f}, NMI: {nmi_train:.4f}")
    print(f"    Test  - ARI: {ari_test:.4f}, NMI: {nmi_test:.4f}")
    results.append(('Connection (Density, Median)', ari_train, nmi_train, ari_test, nmi_test))

    # Experiment 9: Connection-based, Uniform Mass, Mean Threshold
    print("\n[10] Connection-based, Uniform Mass, Mean Threshold")
    ari_train, nmi_train, _ = train_and_evaluate(
        X_train, y_train, method='connection', mass_method='uniform', torque_threshold='mean'
    )
    ari_test, nmi_test, _ = train_and_evaluate(
        X_test, y_test, method='connection', mass_method='uniform', torque_threshold='mean'
    )
    print(f"    Train - ARI: {ari_train:.4f}, NMI: {nmi_train:.4f}")
    print(f"    Test  - ARI: {ari_test:.4f}, NMI: {nmi_test:.4f}")
    results.append(('Connection (Uniform, Mean)', ari_train, nmi_train, ari_test, nmi_test))

    # ==========================================================================
    # RESULTS SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<35} {'Train ARI':>10} {'Train NMI':>10} {'Test ARI':>10} {'Test NMI':>10}")
    print("-" * 75)
    for name, ari_tr, nmi_tr, ari_te, nmi_te in results:
        print(f"{name:<35} {ari_tr:>10.4f} {nmi_tr:>10.4f} {ari_te:>10.4f} {nmi_te:>10.4f}")

    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("VISUALIZATION")
    print("=" * 70)

    # Get labels for visualization
    labels_uniform, _ = torque_clustering(X_train, method='iterative', mass_method='uniform', n_clusters=3, verbose=False)
    labels_density, _ = torque_clustering(X_train, method='iterative', mass_method='density', n_clusters=3, verbose=False)
    labels_conn, _ = torque_clustering(X_train, method='connection', mass_method='uniform', verbose=False)

    # Plot results
    plot_clustering_results(
        X_train,
        {
            'Iterative (Uniform)': labels_uniform,
            'Iterative (Density)': labels_density,
            'Connection-based': labels_conn
        },
        true_labels=y_train,
        title="Torque Clustering Comparison"
    )

    # Optional: Visualize iterations (uncomment to see)
    print("\n[11] Visualizing iterative process...")
    labels, transformed, iters = torque_clustering_iterative(
        X_train, mass_method='density', k_neighbors=5,
        visualize_iterations=True, n_clusters=3
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
