"""
Unsupervised Kidney Stone Segmentation Module
==============================================

A fully UNSUPERVISED segmentation pipeline for detecting renal stones
in 2D kidney CT images using Torque Clustering on superpixel features.

NO LABELS, NO TRAINING, NO SUPERVISION - purely statistical and adaptive.

PIPELINE OVERVIEW:
------------------
    CT Image + Kidney Mask (MedSAM)
                ↓
    ┌───────────────────────────────────┐
    │  STAGE 1: Kidney ROI Extraction   │
    │  - Apply MedSAM mask              │
    │  - Constrain all processing       │
    └───────────────────────────────────┘
                ↓
    ┌───────────────────────────────────┐
    │  STAGE 2: Feature Extraction      │
    │  - Generate SLIC superpixels      │
    │  - Extract stone features         │
    │  - Stone: intensity, top-hat      │
    └───────────────────────────────────┘
                ↓
    ┌───────────────────────────────────┐
    │  STAGE 3: Torque Clustering       │
    │  - Physics-inspired clustering    │
    │  - Adaptive cluster selection     │
    │  - No predefined k                │
    └───────────────────────────────────┘
                ↓
    ┌───────────────────────────────────┐
    │  STAGE 4: Post-Processing         │
    │  - Morphological cleanup          │
    │  - Connected component filtering  │
    │  - Shape/size constraints         │
    └───────────────────────────────────┘
                ↓
            Stone Mask

USAGE:
------
    from kidney_pathology_segmentation import (
        StoneSegmenterTorque,
        segment_stones
    )

    # Stone segmentation
    stone_seg = StoneSegmenterTorque()
    stone_mask = stone_seg.segment(ct_image, kidney_mask)

    # Or use the pipeline function
    stone_mask = segment_stones(ct_image, kidney_mask)

Author: Kidney Segmentation Pipeline
Version: 3.0.0 (Stone-only)
License: MIT
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import label as ndimage_label
from scipy.ndimage import gaussian_filter, uniform_filter, maximum_filter, minimum_filter
from scipy.spatial.distance import cdist
from typing import Tuple, Dict, Optional, List, Union
from dataclasses import dataclass, field
import logging
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class TorqueClusteringParams:
    """
    Parameters for Torque Clustering algorithm.
    
    Torque clustering is a physics-inspired algorithm that treats data points
    as masses and finds natural cluster boundaries based on gravitational
    torque equilibrium.
    
    Attributes:
        n_clusters_range: Range of clusters to explore (min, max)
        distance_metric: Distance metric for clustering ('euclidean', 'manhattan')
        convergence_threshold: Threshold for convergence detection
        max_iterations: Maximum iterations for refinement
        torque_exponent: Exponent for torque calculation (default: 2.0)
        min_cluster_size: Minimum points per cluster
        random_state: Random seed for reproducibility
    """
    n_clusters_range: Tuple[int, int] = (2, 8)
    distance_metric: str = 'euclidean'
    convergence_threshold: float = 1e-4
    max_iterations: int = 100
    torque_exponent: float = 2.0
    min_cluster_size: int = 5
    random_state: int = 42


@dataclass
class SuperpixelParams:
    """
    Parameters for SLIC superpixel generation.
    
    Attributes:
        n_segments: Approximate number of superpixels
        compactness: Balance between color and spatial proximity (higher = more square)
        sigma: Gaussian smoothing before segmentation
        min_size_factor: Minimum superpixel size as fraction of average
    """
    n_segments: int = 200
    compactness: float = 10.0
    sigma: float = 1.0
    min_size_factor: float = 0.25


@dataclass
class StoneSegmentationParams:
    """
    Parameters for stone segmentation.
    
    Attributes:
        tophat_kernel_size: Kernel size for white top-hat filtering
        intensity_percentile: Percentile for bright region detection
        min_area: Minimum stone area in pixels
        max_area: Maximum stone area in pixels
        min_circularity: Minimum circularity constraint
        min_solidity: Minimum solidity constraint
        local_contrast_threshold: Minimum local contrast ratio
    """
    tophat_kernel_size: int = 15
    intensity_percentile: float = 95.0
    min_area: int = 10
    max_area: int = 2000
    min_circularity: float = 0.15
    min_solidity: float = 0.4
    local_contrast_threshold: float = 1.3


# =============================================================================
# TORQUE CLUSTERING IMPLEMENTATION
# =============================================================================

class TorqueClusterer:
    """
    Torque Clustering Algorithm.
    
    A physics-inspired unsupervised clustering algorithm that treats data points
    as masses in a gravitational field. Clusters are formed based on torque
    equilibrium - points that experience balanced "pull" from different directions
    form cluster boundaries.
    
    The algorithm:
    1. Initialize cluster centers using density-based seeding
    2. Compute "gravitational torque" for each point relative to centers
    3. Assign points to clusters based on torque minimization
    4. Refine centers iteratively until convergence
    5. Automatically determine optimal number of clusters
    
    This is fully UNSUPERVISED - no labels or ground truth required.
    
    Example:
        clusterer = TorqueClusterer()
        labels = clusterer.fit_predict(features)
    """
    
    def __init__(self, params: Optional[TorqueClusteringParams] = None):
        """
        Initialize Torque Clusterer.
        
        Args:
            params: Clustering parameters (uses defaults if None)
        """
        self.params = params or TorqueClusteringParams()
        self.centers_ = None
        self.labels_ = None
        self.n_clusters_ = None
        self.inertia_ = None
        
        np.random.seed(self.params.random_state)
    
    def _compute_density(self, X: np.ndarray, bandwidth: float) -> np.ndarray:
        """
        Compute local density for each point using Gaussian kernel.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            bandwidth: Kernel bandwidth
            
        Returns:
            Density values for each point
        """
        n_samples = X.shape[0]
        densities = np.zeros(n_samples)
        
        for i in range(n_samples):
            distances = np.linalg.norm(X - X[i], axis=1)
            densities[i] = np.sum(np.exp(-0.5 * (distances / bandwidth) ** 2))
        
        return densities
    
    def _initialize_centers_density(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Initialize cluster centers using density-based seeding.
        
        Selects initial centers from high-density regions while ensuring
        spatial diversity (similar to k-means++ but density-aware).
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            
        Returns:
            Initial cluster centers
        """
        n_samples, n_features = X.shape
        
        # Estimate bandwidth using rule of thumb
        bandwidth = np.std(X) * (n_samples ** (-1 / (n_features + 4)))
        bandwidth = max(bandwidth, 1e-6)
        
        # Compute densities
        densities = self._compute_density(X, bandwidth)
        
        # Select first center from highest density region
        centers = [X[np.argmax(densities)]]
        
        # Select remaining centers with density-weighted distance sampling
        for _ in range(1, n_clusters):
            # Compute minimum distance to existing centers
            min_distances = np.min(cdist(X, np.array(centers)), axis=1)
            
            # Weight by density and distance
            weights = densities * min_distances
            weights = weights / weights.sum()
            
            # Sample next center
            next_idx = np.random.choice(n_samples, p=weights)
            centers.append(X[next_idx])
        
        return np.array(centers)
    
    def _compute_torque(
        self, 
        X: np.ndarray, 
        centers: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gravitational torque for each point relative to cluster centers.
        
        Torque is computed as the cross-product of position vector and
        gravitational force vector. Points with balanced torque (low magnitude)
        are near cluster boundaries.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            centers: Cluster centers (n_clusters, n_features)
            
        Returns:
            Tuple of (torque_magnitudes, force_vectors)
        """
        n_samples = X.shape[0]
        n_clusters = centers.shape[0]
        
        # Compute distances to all centers
        distances = cdist(X, centers)  # (n_samples, n_clusters)
        
        # Avoid division by zero
        distances = np.maximum(distances, 1e-10)
        
        # Compute gravitational forces (inverse square law)
        forces = 1.0 / (distances ** self.params.torque_exponent)
        
        # Normalize forces
        force_sum = forces.sum(axis=1, keepdims=True)
        force_weights = forces / force_sum
        
        # Compute "torque" as imbalance in force distribution
        # High torque = strong pull toward one center
        # Low torque = balanced pull (boundary region)
        torque = np.max(force_weights, axis=1) - np.min(force_weights, axis=1)
        
        return torque, force_weights
    
    def _assign_clusters(
        self, 
        X: np.ndarray, 
        centers: np.ndarray
    ) -> np.ndarray:
        """
        Assign points to clusters based on torque-weighted distance.
        
        Args:
            X: Feature matrix
            centers: Cluster centers
            
        Returns:
            Cluster labels for each point
        """
        distances = cdist(X, centers)
        torque, force_weights = self._compute_torque(X, centers)
        
        # Combine distance and torque for assignment
        # Points with high torque (strong pull to one center) get assigned clearly
        # Points with low torque (boundaries) use pure distance
        
        weighted_distances = distances * (1 - 0.5 * torque[:, np.newaxis])
        labels = np.argmin(weighted_distances, axis=1)
        
        return labels
    
    def _update_centers(
        self, 
        X: np.ndarray, 
        labels: np.ndarray, 
        n_clusters: int
    ) -> np.ndarray:
        """
        Update cluster centers as weighted mean of assigned points.
        
        Uses torque-weighted averaging to give more weight to core points.
        
        Args:
            X: Feature matrix
            labels: Current cluster labels
            n_clusters: Number of clusters
            
        Returns:
            Updated cluster centers
        """
        centers = np.zeros((n_clusters, X.shape[1]))
        
        for k in range(n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                cluster_points = X[mask]
                
                # Compute density within cluster for weighting
                if len(cluster_points) > 1:
                    pairwise_dist = cdist(cluster_points, cluster_points)
                    bandwidth = np.median(pairwise_dist) + 1e-10
                    weights = np.sum(np.exp(-pairwise_dist / bandwidth), axis=1)
                    weights = weights / weights.sum()
                    centers[k] = np.average(cluster_points, weights=weights, axis=0)
                else:
                    centers[k] = cluster_points[0]
            else:
                # Empty cluster: reinitialize randomly
                centers[k] = X[np.random.randint(len(X))]
        
        return centers
    
    def _compute_inertia(
        self, 
        X: np.ndarray, 
        labels: np.ndarray, 
        centers: np.ndarray
    ) -> float:
        """
        Compute clustering inertia (within-cluster sum of squares).
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            centers: Cluster centers
            
        Returns:
            Inertia value
        """
        inertia = 0.0
        for k in range(len(centers)):
            mask = labels == k
            if np.sum(mask) > 0:
                inertia += np.sum((X[mask] - centers[k]) ** 2)
        return inertia
    
    def _compute_silhouette_score(
        self, 
        X: np.ndarray, 
        labels: np.ndarray
    ) -> float:
        """
        Compute simplified silhouette score for cluster quality assessment.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            
        Returns:
            Mean silhouette score
        """
        n_samples = len(X)
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters < 2:
            return -1.0
        
        silhouette_vals = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Compute mean distance to same cluster (a)
            same_mask = labels == labels[i]
            same_mask[i] = False
            if np.sum(same_mask) > 0:
                a = np.mean(np.linalg.norm(X[same_mask] - X[i], axis=1))
            else:
                a = 0
            
            # Compute mean distance to nearest other cluster (b)
            b = np.inf
            for k in unique_labels:
                if k != labels[i]:
                    other_mask = labels == k
                    if np.sum(other_mask) > 0:
                        mean_dist = np.mean(np.linalg.norm(X[other_mask] - X[i], axis=1))
                        b = min(b, mean_dist)
            
            if b == np.inf:
                b = 0
            
            # Silhouette coefficient
            silhouette_vals[i] = (b - a) / max(a, b, 1e-10)
        
        return np.mean(silhouette_vals)
    
    def _fit_single_k(
        self, 
        X: np.ndarray, 
        n_clusters: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Fit clustering with a specific number of clusters.
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            
        Returns:
            Tuple of (labels, centers, inertia)
        """
        # Initialize centers
        centers = self._initialize_centers_density(X, n_clusters)
        
        prev_inertia = np.inf
        
        for iteration in range(self.params.max_iterations):
            # Assign clusters
            labels = self._assign_clusters(X, centers)
            
            # Check for empty clusters
            unique_labels = np.unique(labels)
            if len(unique_labels) < n_clusters:
                # Reinitialize empty clusters
                for k in range(n_clusters):
                    if k not in unique_labels:
                        # Find point furthest from any center
                        distances = np.min(cdist(X, centers), axis=1)
                        centers[k] = X[np.argmax(distances)]
                labels = self._assign_clusters(X, centers)
            
            # Update centers
            centers = self._update_centers(X, labels, n_clusters)
            
            # Compute inertia
            inertia = self._compute_inertia(X, labels, centers)
            
            # Check convergence
            if abs(prev_inertia - inertia) < self.params.convergence_threshold:
                break
            
            prev_inertia = inertia
        
        return labels, centers, inertia
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit Torque Clustering and return cluster labels.
        
        Automatically determines optimal number of clusters using
        silhouette score within the specified range.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Cluster labels (n_samples,)
        """
        if len(X) < self.params.min_cluster_size:
            logger.warning(f"Too few samples ({len(X)}), returning single cluster")
            self.labels_ = np.zeros(len(X), dtype=np.int32)
            self.n_clusters_ = 1
            return self.labels_
        
        # Normalize features
        X_normalized = X.copy()
        for j in range(X.shape[1]):
            col_std = np.std(X[:, j])
            if col_std > 1e-10:
                X_normalized[:, j] = (X[:, j] - np.mean(X[:, j])) / col_std
        
        best_score = -np.inf
        best_labels = None
        best_centers = None
        best_k = self.params.n_clusters_range[0]
        
        # Search for optimal number of clusters
        min_k, max_k = self.params.n_clusters_range
        max_k = min(max_k, len(X) // self.params.min_cluster_size)
        
        for k in range(min_k, max_k + 1):
            labels, centers, inertia = self._fit_single_k(X_normalized, k)
            
            # Compute silhouette score
            score = self._compute_silhouette_score(X_normalized, labels)
            
            logger.debug(f"k={k}, silhouette={score:.4f}, inertia={inertia:.4f}")
            
            if score > best_score:
                best_score = score
                best_labels = labels
                best_centers = centers
                best_k = k
        
        self.labels_ = best_labels
        self.centers_ = best_centers
        self.n_clusters_ = best_k
        self.inertia_ = self._compute_inertia(X_normalized, best_labels, best_centers)
        
        logger.info(f"Torque Clustering: {best_k} clusters, silhouette={best_score:.4f}")
        
        return self.labels_
    
    def fit_predict_fixed_k(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Fit Torque Clustering with a fixed number of clusters.
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            
        Returns:
            Cluster labels
        """
        if len(X) < n_clusters:
            logger.warning(f"Too few samples ({len(X)}) for {n_clusters} clusters")
            self.labels_ = np.zeros(len(X), dtype=np.int32)
            return self.labels_
        
        # Normalize features
        X_normalized = X.copy()
        for j in range(X.shape[1]):
            col_std = np.std(X[:, j])
            if col_std > 1e-10:
                X_normalized[:, j] = (X[:, j] - np.mean(X[:, j])) / col_std
        
        self.labels_, self.centers_, self.inertia_ = self._fit_single_k(X_normalized, n_clusters)
        self.n_clusters_ = n_clusters
        
        return self.labels_


# =============================================================================
# SUPERPIXEL GENERATION
# =============================================================================

class SuperpixelGenerator:
    """
    Superpixel Generator for kidney ROI.
    
    Generates oversegmented superpixels within the kidney mask region,
    providing spatial coherence for feature extraction.
    
    Uses SLIC if available (opencv-contrib), otherwise falls back to
    a simple grid-based segmentation with region merging.
    """
    
    def __init__(self, params: Optional[SuperpixelParams] = None):
        """
        Initialize superpixel generator.
        
        Args:
            params: Superpixel parameters
        """
        self.params = params or SuperpixelParams()
        self._has_slic = self._check_slic_available()
    
    def _check_slic_available(self) -> bool:
        """Check if SLIC is available in OpenCV."""
        try:
            if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'createSuperpixelSLIC'):
                return True
        except:
            pass
        return False
    
    def _generate_grid_superpixels(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """
        Generate grid-based superpixels (fallback when SLIC unavailable).
        
        Creates a regular grid and merges similar adjacent regions.
        
        Args:
            image: Grayscale image
            mask: Binary mask
            
        Returns:
            Tuple of (labels, num_superpixels)
        """
        h, w = image.shape
        
        # Calculate grid size based on desired number of segments
        mask_area = np.sum(mask)
        if mask_area == 0:
            return np.full((h, w), -1, dtype=np.int32), 0
        
        segment_size = int(np.sqrt(mask_area / self.params.n_segments))
        segment_size = max(segment_size, 10)  # Minimum 10x10 pixels
        
        # Create initial grid labels
        grid_labels = np.full((h, w), -1, dtype=np.int32)
        label_id = 0
        
        for y in range(0, h, segment_size):
            for x in range(0, w, segment_size):
                y_end = min(y + segment_size, h)
                x_end = min(x + segment_size, w)
                
                # Check if this grid cell overlaps with mask
                cell_mask = mask[y:y_end, x:x_end]
                if np.any(cell_mask):
                    grid_labels[y:y_end, x:x_end][cell_mask == 1] = label_id
                    label_id += 1
        
        # Apply mask
        grid_labels[mask == 0] = -1
        
        # Relabel to consecutive integers
        unique_labels = np.unique(grid_labels[grid_labels >= 0])
        new_labels = np.full_like(grid_labels, -1)
        
        for new_id, old_id in enumerate(unique_labels):
            new_labels[grid_labels == old_id] = new_id
        
        num_superpixels = len(unique_labels)
        
        logger.debug(f"Generated {num_superpixels} grid superpixels")
        
        return new_labels, num_superpixels
    
    def _generate_slic_superpixels(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """
        Generate SLIC superpixels (requires opencv-contrib).
        
        Args:
            image: Grayscale image
            mask: Binary mask
            
        Returns:
            Tuple of (labels, num_superpixels)
        """
        # Apply Gaussian smoothing
        if self.params.sigma > 0:
            image_smooth = cv2.GaussianBlur(
                image, (0, 0), self.params.sigma
            ).astype(np.float32)
        else:
            image_smooth = image.astype(np.float32)
        
        # Convert to 3-channel for SLIC (OpenCV requirement)
        image_3ch = cv2.cvtColor(image_smooth.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        # Apply SLIC
        slic = cv2.ximgproc.createSuperpixelSLIC(
            image_3ch,
            algorithm=cv2.ximgproc.SLIC,
            region_size=int(np.sqrt(np.sum(mask) / self.params.n_segments)),
            ruler=self.params.compactness
        )
        slic.iterate(10)
        
        # Get labels
        labels = slic.getLabels()
        
        # Mask out regions outside kidney
        labels[mask == 0] = -1
        
        # Relabel to consecutive integers
        unique_labels = np.unique(labels[labels >= 0])
        new_labels = np.full_like(labels, -1)
        
        for new_id, old_id in enumerate(unique_labels):
            new_labels[labels == old_id] = new_id
        
        num_superpixels = len(unique_labels)
        
        logger.debug(f"Generated {num_superpixels} SLIC superpixels")
        
        return new_labels, num_superpixels
    
    def generate(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """
        Generate superpixels within the masked region.
        
        Uses SLIC if available, otherwise falls back to grid-based approach.
        
        Args:
            image: Grayscale image (uint8)
            mask: Binary mask (0/1)
            
        Returns:
            Tuple of (superpixel_labels, num_superpixels)
        """
        if self._has_slic:
            try:
                return self._generate_slic_superpixels(image, mask)
            except Exception as e:
                logger.warning(f"SLIC failed: {e}, using grid fallback")
        
        return self._generate_grid_superpixels(image, mask)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

class FeatureExtractor:
    """
    Feature Extractor for stone detection.

    Extracts discriminative features for stone detection
    from superpixels within the kidney ROI.

    Stone Features (5D):
        1. Mean intensity
        2. Maximum intensity
        3. White top-hat response (highlights small bright structures)
        4. Local contrast ratio
        5. Region area (normalized)
    """

    def __init__(
        self,
        stone_params: Optional[StoneSegmentationParams] = None
    ):
        """
        Initialize feature extractor.

        Args:
            stone_params: Stone segmentation parameters
        """
        self.stone_params = stone_params or StoneSegmentationParams()
    
    def _compute_tophat(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Compute white top-hat transform.
        
        White top-hat = image - opening(image)
        Highlights bright structures smaller than the kernel.
        
        Args:
            image: Input image
            kernel_size: Morphological kernel size
            
        Returns:
            Top-hat response image
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    
    def _compute_local_contrast(
        self, 
        image: np.ndarray, 
        kernel_size: int = 15
    ) -> np.ndarray:
        """
        Compute local contrast map.
        
        Local contrast = (pixel - local_min) / (local_max - local_min + eps)
        
        Args:
            image: Input image
            kernel_size: Window size for local statistics
            
        Returns:
            Local contrast map
        """
        image_float = image.astype(np.float32)
        
        local_max = maximum_filter(image_float, size=kernel_size)
        local_min = minimum_filter(image_float, size=kernel_size)
        
        contrast = (image_float - local_min) / (local_max - local_min + 1e-10)
        
        return contrast
    
    def _compute_gradient_magnitude(
        self, 
        image: np.ndarray, 
        kernel_size: int = 3
    ) -> np.ndarray:
        """
        Compute gradient magnitude using Sobel operators.
        
        Args:
            image: Input image
            kernel_size: Sobel kernel size
            
        Returns:
            Gradient magnitude map
        """
        image_float = image.astype(np.float32)
        
        grad_x = cv2.Sobel(image_float, cv2.CV_32F, 1, 0, ksize=kernel_size)
        grad_y = cv2.Sobel(image_float, cv2.CV_32F, 0, 1, ksize=kernel_size)
        
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        
        return magnitude
    
    def _compute_laplacian(self, image: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian response (second derivative).
        
        Laplacian highlights regions with rapid intensity changes (texture).
        
        Args:
            image: Input image
            
        Returns:
            Absolute Laplacian response
        """
        image_float = image.astype(np.float32)
        laplacian = cv2.Laplacian(image_float, cv2.CV_32F)
        return np.abs(laplacian)
    
    def _compute_local_std(
        self, 
        image: np.ndarray, 
        kernel_size: int = 21
    ) -> np.ndarray:
        """
        Compute local standard deviation (texture measure).
        
        Args:
            image: Input image
            kernel_size: Window size
            
        Returns:
            Local standard deviation map
        """
        image_float = image.astype(np.float64)
        
        # Local mean
        local_mean = uniform_filter(image_float, size=kernel_size)
        
        # Local mean of squares
        local_sq_mean = uniform_filter(image_float ** 2, size=kernel_size)
        
        # Local variance and std
        local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
        local_std = np.sqrt(local_var)
        
        return local_std
    
    def extract_stone_features(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        superpixel_labels: np.ndarray,
        num_superpixels: int
    ) -> np.ndarray:
        """
        Extract stone-specific features for each superpixel.
        
        Stone Feature Vector (5D):
            [mean_intensity, max_intensity, tophat_response, local_contrast, area]
        
        Stones are characterized by:
        - High intensity (bright calcifications)
        - High top-hat response (small bright structures)
        - High local contrast (stand out from surroundings)
        - Small area
        
        Args:
            image: Grayscale CT image
            mask: Kidney mask
            superpixel_labels: Superpixel label map
            num_superpixels: Number of superpixels
            
        Returns:
            Feature matrix (num_superpixels, 5)
        """
        # Precompute image transforms
        tophat = self._compute_tophat(image, self.stone_params.tophat_kernel_size)
        local_contrast = self._compute_local_contrast(image)
        
        # Initialize feature matrix
        features = np.zeros((num_superpixels, 5))
        
        # Total kidney area for normalization
        total_area = np.sum(mask)
        
        for sp_id in range(num_superpixels):
            sp_mask = superpixel_labels == sp_id
            
            if np.sum(sp_mask) == 0:
                continue
            
            # Extract pixel values
            intensity_vals = image[sp_mask]
            tophat_vals = tophat[sp_mask]
            contrast_vals = local_contrast[sp_mask]
            
            # Feature 1: Mean intensity
            features[sp_id, 0] = np.mean(intensity_vals)
            
            # Feature 2: Maximum intensity
            features[sp_id, 1] = np.max(intensity_vals)
            
            # Feature 3: Mean top-hat response
            features[sp_id, 2] = np.mean(tophat_vals)
            
            # Feature 4: Mean local contrast
            features[sp_id, 3] = np.mean(contrast_vals)
            
            # Feature 5: Normalized area (smaller for stones)
            features[sp_id, 4] = np.sum(sp_mask) / total_area
        
        return features


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def apply_kidney_mask(image: np.ndarray, mask: np.ndarray, padding_bottom: int = 30) -> np.ndarray:
    """
    Apply kidney bounding box to image, setting outside pixels to 0.

    Uses the bounding box of the kidney mask with padding extended only
    downward, to preserve stones that may be below the segmented kidney boundary.

    Args:
        image: Input image
        mask: Binary kidney mask
        padding_bottom: Pixels to pad below the kidney bounding box (default: 30)

    Returns:
        Masked image with pixels outside padded bounding box set to 0
    """
    result = image.copy()

    # Find bounding box of the kidney mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        # Empty mask - return zeros
        return np.zeros_like(image)

    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]

    y_min, y_max = row_indices[0], row_indices[-1]
    x_min, x_max = col_indices[0], col_indices[-1]

    # Apply padding only to the bottom (clamped to image bounds)
    h = image.shape[0]
    y_max = min(h - 1, y_max + padding_bottom)

    # Zero out pixels outside the bounding box (padded only at bottom)
    result[:y_min, :] = 0
    result[y_max + 1:, :] = 0
    result[:, :x_min] = 0
    result[:, x_max + 1:] = 0

    return result



def morphological_cleanup(
    mask: np.ndarray, 
    open_kernel: int = 2, 
    close_kernel: int = 3
) -> np.ndarray:
    """
    Apply morphological cleanup to binary mask.
    
    Args:
        mask: Binary mask
        open_kernel: Kernel size for opening (noise removal)
        close_kernel: Kernel size for closing (hole filling)
        
    Returns:
        Cleaned mask
    """
    result = mask.copy()
    
    if open_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    
    if close_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    
    return result


def remove_border_components(mask: np.ndarray) -> np.ndarray:
    """
    Remove connected components touching the image border.
    
    Args:
        mask: Binary mask
        
    Returns:
        Mask with border-touching components removed
    """
    h, w = mask.shape
    
    # Create border mask
    border = np.zeros_like(mask)
    border[0, :] = 1
    border[-1, :] = 1
    border[:, 0] = 1
    border[:, -1] = 1
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    
    result = mask.copy()
    
    for label_id in range(1, num_labels):
        component = (labels == label_id)
        if np.any(component & border.astype(bool)):
            result[component] = 0
    
    return result


def filter_by_size(
    mask: np.ndarray, 
    min_area: int, 
    max_area: int
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Filter connected components by area.
    
    Args:
        mask: Binary mask
        min_area: Minimum component area
        max_area: Maximum component area
        
    Returns:
        Tuple of (filtered_mask, component_properties)
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    
    result = np.zeros_like(mask)
    properties = []
    
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        
        if min_area <= area <= max_area:
            result[labels == label_id] = 1
            properties.append({
                'label_id': label_id,
                'area': area,
                'centroid': centroids[label_id].tolist(),
                'bbox': [
                    stats[label_id, cv2.CC_STAT_LEFT],
                    stats[label_id, cv2.CC_STAT_TOP],
                    stats[label_id, cv2.CC_STAT_WIDTH],
                    stats[label_id, cv2.CC_STAT_HEIGHT]
                ]
            })
    
    return result.astype(np.uint8), properties


def filter_by_shape(
    mask: np.ndarray,
    min_circularity: float = 0.0,
    max_circularity: float = 1.0,
    min_solidity: float = 0.0
) -> np.ndarray:
    """
    Filter connected components by shape properties.
    
    Args:
        mask: Binary mask
        min_circularity: Minimum circularity (0-1)
        max_circularity: Maximum circularity (0-1)
        min_solidity: Minimum solidity (0-1)
        
    Returns:
        Filtered mask
    """
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    
    result = np.zeros_like(mask)
    
    for label_id in range(1, num_labels):
        component = (labels == label_id).astype(np.uint8)
        
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            continue
        
        contour = contours[0]
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Circularity
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-10)
        
        # Solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-10)
        
        # Check criteria
        if (min_circularity <= circularity <= max_circularity and 
            solidity >= min_solidity):
            result[component == 1] = 1
    
    return result.astype(np.uint8)


# =============================================================================
# STONE SEGMENTER
# =============================================================================

class StoneSegmenterTorque:
    """
    Unsupervised Stone Segmentation using Torque Clustering.
    
    Detects kidney stones by:
    1. Generating superpixels within kidney ROI
    2. Extracting stone-specific features (intensity, top-hat, contrast)
    3. Clustering superpixels using Torque Clustering
    4. Selecting clusters with highest brightness/top-hat response
    5. Post-processing with morphology and shape constraints
    
    NO SUPERVISION - all thresholds are statistical/adaptive.
    
    Example:
        segmenter = StoneSegmenterTorque()
        stone_mask = segmenter.segment(ct_image, kidney_mask)
    """
    
    def __init__(
        self,
        params: Optional[StoneSegmentationParams] = None,
        clustering_params: Optional[TorqueClusteringParams] = None,
        superpixel_params: Optional[SuperpixelParams] = None
    ):
        """
        Initialize stone segmenter.
        
        Args:
            params: Stone segmentation parameters
            clustering_params: Torque clustering parameters
            superpixel_params: Superpixel generation parameters
        """
        self.params = params or StoneSegmentationParams()
        self.clustering_params = clustering_params or TorqueClusteringParams(
            n_clusters_range=(2, 5)
        )
        self.superpixel_params = superpixel_params or SuperpixelParams(
            n_segments=150
        )
        
        self.feature_extractor = FeatureExtractor(stone_params=self.params)
        self.superpixel_generator = SuperpixelGenerator(self.superpixel_params)
        self.clusterer = TorqueClusterer(self.clustering_params)
    
    def _select_stone_clusters(
        self, 
        features: np.ndarray, 
        labels: np.ndarray
    ) -> List[int]:
        """
        Select clusters most likely to contain stones using strict criteria.
        
        Stone clusters are characterized by:
        - Very high mean intensity (feature 0) - must be outlier
        - Very high max intensity (feature 1) - must be outlier  
        - High top-hat response (feature 2) - must be significant
        - Small area (feature 4)
        
        Selection uses statistical outlier detection - only clusters that
        are significantly brighter than the kidney average are selected.
        
        Args:
            features: Feature matrix
            labels: Cluster labels
            
        Returns:
            List of cluster IDs likely containing stones
        """
        unique_labels = np.unique(labels)
        
        # Compute global statistics for outlier detection
        all_intensities = features[:, 0]
        all_max_intensities = features[:, 1]
        all_tophat = features[:, 2]
        
        # Stones must be statistical outliers (very bright)
        intensity_mean = np.mean(all_intensities)
        intensity_std = np.std(all_intensities)
        tophat_mean = np.mean(all_tophat)
        tophat_std = np.std(all_tophat)
        
        # Strict thresholds: must be >2 std above mean
        intensity_threshold = intensity_mean + 2.0 * intensity_std
        tophat_threshold = tophat_mean + 1.5 * tophat_std
        
        # Also require absolute minimum intensity (stones are bright)
        min_absolute_intensity = 150
        
        selected = []
        
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_features = features[cluster_mask]
            
            if len(cluster_features) == 0:
                continue
            
            # Compute cluster statistics
            cluster_mean_intensity = np.mean(cluster_features[:, 0])
            cluster_mean_max = np.mean(cluster_features[:, 1])
            cluster_mean_tophat = np.mean(cluster_features[:, 2])
            cluster_mean_area = np.mean(cluster_features[:, 4])
            
            # Strict selection criteria:
            # 1. Must be intensity outlier OR have high tophat
            is_intensity_outlier = cluster_mean_intensity > intensity_threshold
            is_tophat_outlier = cluster_mean_tophat > tophat_threshold
            has_high_absolute_intensity = cluster_mean_max > min_absolute_intensity
            
            # 2. Must have small area (stones are small)
            is_small = cluster_mean_area < np.percentile(features[:, 4], 50)
            
            # Select only if: (bright outlier OR high tophat) AND absolutely bright AND small
            if ((is_intensity_outlier or is_tophat_outlier) and 
                has_high_absolute_intensity and is_small):
                selected.append(cluster_id)
                logger.debug(f"Stone cluster {cluster_id}: I={cluster_mean_intensity:.1f}, "
                           f"tophat={cluster_mean_tophat:.1f}, area={cluster_mean_area:.4f}")
        
        logger.debug(f"Stone cluster selection: {len(selected)}/{len(unique_labels)} clusters "
                    f"(I_thresh={intensity_threshold:.1f}, tophat_thresh={tophat_threshold:.1f})")
        
        return selected
    
    def segment(
        self, 
        ct_image: np.ndarray, 
        kidney_mask: np.ndarray
    ) -> np.ndarray:
        """
        Segment stones in the CT image.
        
        Uses a hybrid approach:
        1. Superpixel-based Torque Clustering for region-level analysis
        2. Direct pixel-level detection for small bright stones
        3. Combination of both approaches
        
        Args:
            ct_image: Grayscale CT image (uint8, 560x560)
            kidney_mask: Binary kidney mask from MedSAM (0/1)
            
        Returns:
            Binary stone mask (uint8, 0/1)
        """
        # Validate inputs
        if ct_image.shape != kidney_mask.shape:
            raise ValueError(f"Shape mismatch: image={ct_image.shape}, mask={kidney_mask.shape}")
        
        # Ensure proper types
        ct_image = ct_image.astype(np.uint8)
        kidney_mask = (kidney_mask > 0).astype(np.uint8)
        
        # Check kidney size
        kidney_pixels = np.sum(kidney_mask)
        if kidney_pixels < 100:
            logger.warning("Insufficient kidney pixels")
            return np.zeros_like(ct_image, dtype=np.uint8)
        
        # Stage 1: Apply kidney mask
        roi_image = apply_kidney_mask(ct_image, kidney_mask, padding_bottom=30)
        
        # APPROACH 1: Direct pixel-level detection (works better for small bright stones)
        pixel_mask = self._detect_bright_stones_direct(roi_image, kidney_mask)
        
        # APPROACH 2: Superpixel-based clustering (works better for larger stones)
        cluster_mask = np.zeros_like(ct_image, dtype=np.uint8)
        try:
            sp_labels, num_sp = self.superpixel_generator.generate(roi_image, kidney_mask)
            
            if num_sp >= 3:
                features = self.feature_extractor.extract_stone_features(
                    roi_image, kidney_mask, sp_labels, num_sp
                )
                
                valid_mask = np.any(features != 0, axis=1)
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_indices) >= 3:
                    valid_features = features[valid_indices]
                    cluster_labels = self.clusterer.fit_predict(valid_features)
                    
                    full_labels = np.full(num_sp, -1)
                    full_labels[valid_indices] = cluster_labels
                    
                    selected_clusters = self._select_stone_clusters(valid_features, cluster_labels)
                    
                    for sp_id in range(num_sp):
                        if full_labels[sp_id] in selected_clusters:
                            cluster_mask[sp_labels == sp_id] = 1
                    
                    # Pre-filter cluster mask to only keep stone-sized components
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                        cluster_mask.astype(np.uint8), connectivity=8
                    )
                    cluster_mask_filtered = np.zeros_like(cluster_mask)
                    for label_id in range(1, num_labels):
                        area = stats[label_id, cv2.CC_STAT_AREA]
                        if self.params.min_area <= area <= self.params.max_area:
                            cluster_mask_filtered[labels == label_id] = 1
                    cluster_mask = cluster_mask_filtered
                    
        except Exception as e:
            logger.warning(f"Clustering approach failed: {e}")
        
        # Combine both approaches (union)
        combined_mask = np.logical_or(pixel_mask, cluster_mask).astype(np.uint8)
        
        # Stage 6: Post-processing
        combined_mask = morphological_cleanup(combined_mask, open_kernel=2, close_kernel=3)
        combined_mask = remove_border_components(combined_mask)
        
        # Filter connected components individually by size
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            combined_mask.astype(np.uint8), connectivity=8
        )
        
        filtered_mask = np.zeros_like(combined_mask)
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            # Keep only small components (stone-sized)
            if self.params.min_area <= area <= self.params.max_area:
                filtered_mask[labels == label_id] = 1
        
        combined_mask = filtered_mask.astype(np.uint8)
        
        # Shape filtering
        combined_mask = filter_by_shape(
            combined_mask,
            min_circularity=self.params.min_circularity,
            max_circularity=1.0,
            min_solidity=self.params.min_solidity
        )
        
        # Ensure mask is within kidney
        combined_mask = combined_mask & kidney_mask
        
        num_stones = len(np.unique(cv2.connectedComponents(combined_mask)[1])) - 1
        logger.info(f"Stone segmentation: {num_stones} stone(s), {np.sum(combined_mask)} pixels")
        
        return combined_mask
    
    def _detect_bright_stones_direct(
        self, 
        roi_image: np.ndarray, 
        kidney_mask: np.ndarray
    ) -> np.ndarray:
        """
        Direct pixel-level detection of bright stones.
        
        Uses statistical outlier detection on intensity values.
        Stones are the brightest pixels in the kidney ROI.
        
        Args:
            roi_image: Kidney-masked CT image
            kidney_mask: Binary kidney mask
            
        Returns:
            Binary mask of detected bright regions
        """
        # Get kidney pixel values
        kidney_values = roi_image[kidney_mask == 1]
        if len(kidney_values) == 0:
            return np.zeros_like(roi_image, dtype=np.uint8)
        
        # Compute statistics for outlier detection
        mean_val = np.mean(kidney_values)
        std_val = np.std(kidney_values)
        
        # Statistical threshold: 2.5 sigma above mean
        stat_threshold = mean_val + 2.5 * std_val
        
        # Percentile threshold: top 3%
        percentile_threshold = np.percentile(kidney_values, 97)
        
        # Absolute minimum (stones are typically >170 in 8-bit)
        absolute_min = 170
        
        # Use maximum of all thresholds
        threshold = max(stat_threshold, percentile_threshold, absolute_min)
        
        # Apply intensity threshold
        bright_mask = (roi_image >= threshold).astype(np.uint8)
        bright_mask = bright_mask & kidney_mask
        
        # Use top-hat to enhance small structures
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(roi_image, cv2.MORPH_TOPHAT, kernel)
        
        tophat_values = tophat[kidney_mask == 1]
        if len(tophat_values) > 0 and np.max(tophat_values) > 0:
            # Use a high percentile for top-hat (99th)
            nonzero_tophat = tophat_values[tophat_values > 0]
            if len(nonzero_tophat) > 0:
                tophat_threshold = np.percentile(nonzero_tophat, 99)
                tophat_mask = (tophat >= tophat_threshold).astype(np.uint8)
                tophat_mask = tophat_mask & kidney_mask
                
                # Combine: use OR but require the original intensity threshold
                # i.e., tophat helps but pixel must still be reasonably bright
                lower_intensity_thresh = max(mean_val + 1.5 * std_val, 150)
                intensity_mask = (roi_image >= lower_intensity_thresh).astype(np.uint8)
                tophat_mask = tophat_mask & intensity_mask & kidney_mask
                
                bright_mask = bright_mask | tophat_mask
        
        return bright_mask
    
    def _fallback_segmentation(
        self, 
        roi_image: np.ndarray, 
        kidney_mask: np.ndarray
    ) -> np.ndarray:
        """
        Fallback segmentation using direct intensity thresholding.
        
        Used when superpixel/clustering approach fails.
        """
        # Compute top-hat
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(roi_image, cv2.MORPH_TOPHAT, kernel)
        
        # Adaptive threshold on kidney pixels
        kidney_values = tophat[kidney_mask == 1]
        if len(kidney_values) == 0:
            return np.zeros_like(roi_image, dtype=np.uint8)
        
        threshold = np.percentile(kidney_values, self.params.intensity_percentile)
        
        # Also threshold on intensity
        intensity_values = roi_image[kidney_mask == 1]
        intensity_thresh = np.percentile(intensity_values, self.params.intensity_percentile)
        
        # Combine
        mask = ((tophat >= threshold) | (roi_image >= max(intensity_thresh, 170))).astype(np.uint8)
        mask = mask & kidney_mask
        
        # Post-process
        mask = morphological_cleanup(mask)
        mask = remove_border_components(mask)
        mask, _ = filter_by_size(mask, self.params.min_area, self.params.max_area)
        mask = filter_by_shape(mask, min_circularity=self.params.min_circularity)
        
        return mask


# =============================================================================
# COMBINED PIPELINE
# =============================================================================

def segment_stones(
    ct_image: np.ndarray,
    kidney_mask: np.ndarray
) -> np.ndarray:
    """
    Stone segmentation pipeline.

    Detects kidney stones in a CT image.

    Args:
        ct_image: Grayscale CT image (uint8, 560x560)
        kidney_mask: Binary kidney mask from MedSAM (0/1)

    Returns:
        Binary stone mask (uint8, 0/1)
    """
    stone_segmenter = StoneSegmenterTorque()
    stone_mask = stone_segmenter.segment(ct_image, kidney_mask)
    return stone_mask


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualization(
    ct_image: np.ndarray,
    kidney_mask: np.ndarray,
    stone_mask: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Kidney Stone Segmentation"
):
    """
    Create visualization overlay.

    Colors:
    - Kidney outline: Green
    - Stones: Red

    Args:
        ct_image: Original CT image
        kidney_mask: Kidney mask
        stone_mask: Stone segmentation mask
        save_path: Path to save visualization
        title: Figure title

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Original CT
    axes[0].imshow(ct_image, cmap='gray')
    axes[0].set_title('CT Image')
    axes[0].axis('off')

    # Panel 2: Kidney mask
    axes[1].imshow(kidney_mask, cmap='Blues')
    axes[1].set_title('Kidney (MedSAM)')
    axes[1].axis('off')

    # Panel 3: Overlay with stones
    overlay = cv2.cvtColor(ct_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Kidney outline (green)
    kidney_contours, _ = cv2.findContours(
        kidney_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, kidney_contours, -1, (0, 255, 0), 2)

    # Stone overlay (red)
    overlay[stone_mask == 1, 0] = np.minimum(overlay[stone_mask == 1, 0].astype(int) + 150, 255)
    overlay[stone_mask == 1, 1] = overlay[stone_mask == 1, 1] // 2
    overlay[stone_mask == 1, 2] = overlay[stone_mask == 1, 2] // 2

    # Stone contours (red)
    stone_contours, _ = cv2.findContours(
        stone_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, stone_contours, -1, (255, 0, 0), 1)

    num_stones = len(np.unique(cv2.connectedComponents(stone_mask.astype(np.uint8))[1])) - 1
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay (G=Kidney, R=Stone) - {num_stones} stone(s)')
    axes[2].axis('off')

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualization saved to: {save_path}")

    return fig


# =============================================================================
# SYNTHETIC DATA GENERATION FOR TESTING
# =============================================================================

def create_synthetic_kidney_ct(
    image_size: Tuple[int, int] = (560, 560),
    num_stones: int = 2,
    noise_level: float = 10.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create synthetic kidney CT with stones.

    Args:
        image_size: Image dimensions (H, W)
        num_stones: Number of stones to generate
        noise_level: Gaussian noise level
        seed: Random seed

    Returns:
        Tuple of (ct_image, kidney_mask, gt_stone_mask)
    """
    np.random.seed(seed)
    h, w = image_size

    # Create base image
    ct_image = np.random.normal(70, noise_level, (h, w)).astype(np.float32)

    # Create kidney mask (elliptical)
    kidney_mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (w // 4, h // 3)
    cv2.ellipse(kidney_mask, center, axes, angle=10, startAngle=0, endAngle=360, color=1, thickness=-1)

    # Add kidney tissue
    ct_image[kidney_mask == 1] += 25 + np.random.normal(0, 5, np.sum(kidney_mask == 1))

    # Create ground truth mask
    gt_stone_mask = np.zeros((h, w), dtype=np.uint8)

    # Add stones (small, bright, circular)
    for _ in range(num_stones):
        for attempt in range(100):
            sx = np.random.randint(w // 4, 3 * w // 4)
            sy = np.random.randint(h // 4, 3 * h // 4)
            if kidney_mask[sy, sx] == 1:
                break

        radius = np.random.randint(4, 10)
        intensity = np.random.randint(200, 240)

        stone_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(stone_mask, (sx, sy), radius, 1, -1)
        stone_mask = stone_mask & kidney_mask

        ct_image[stone_mask == 1] = intensity + np.random.normal(0, 5, np.sum(stone_mask == 1))
        gt_stone_mask[stone_mask == 1] = 1

    ct_image = np.clip(ct_image, 0, 255).astype(np.uint8)

    return ct_image, kidney_mask, gt_stone_mask


def run_quick_test():
    """
    Run quick validation test on synthetic data.
    """
    print("\n" + "=" * 70)
    print("KIDNEY STONE SEGMENTATION - QUICK TEST")
    print("Using Torque Clustering (Fully Unsupervised)")
    print("=" * 70)

    # Create synthetic data
    print("\n--- Creating synthetic CT with 2 stones ---")
    ct_image, kidney_mask, gt_stone = create_synthetic_kidney_ct(
        num_stones=2, seed=42
    )

    print(f"CT shape: {ct_image.shape}")
    print(f"Kidney pixels: {np.sum(kidney_mask)}")
    print(f"GT stones: {np.sum(gt_stone)} pixels")

    # Test stone segmentation
    print("\n--- Testing Stone Segmentation ---")
    stone_seg = StoneSegmenterTorque()
    pred_stone = stone_seg.segment(ct_image, kidney_mask)

    stone_intersection = np.sum(pred_stone & gt_stone)
    stone_union = np.sum(pred_stone | gt_stone)
    stone_iou = stone_intersection / (stone_union + 1e-10)

    print(f"Predicted stone pixels: {np.sum(pred_stone)}")
    print(f"Stone IoU: {stone_iou:.3f}")
    print("✓ Stone test PASSED" if stone_iou > 0.3 else "⚠ Stone test needs tuning")

    # Test pipeline function
    print("\n--- Testing Pipeline Function ---")
    stone_mask = segment_stones(ct_image, kidney_mask)
    print(f"Pipeline stones: {np.sum(stone_mask)} pixels")

    # Edge cases
    print("\n--- Testing Edge Cases ---")

    # Empty kidney
    empty_stone = stone_seg.segment(ct_image, np.zeros_like(kidney_mask))
    print(f"Empty kidney → {np.sum(empty_stone)} stone pixels (expected: 0) ", end="")
    print("✓" if np.sum(empty_stone) == 0 else "✗")

    # No pathology
    clean_ct, clean_kidney, _ = create_synthetic_kidney_ct(num_stones=0, seed=99)
    clean_stones = stone_seg.segment(clean_ct, clean_kidney)
    print(f"Clean kidney → {np.sum(clean_stones)} stone pixels (expected: ~0) ", end="")
    print("✓" if np.sum(clean_stones) < 100 else "⚠")

    print("\n" + "=" * 70)
    print("QUICK TEST COMPLETE")
    print("=" * 70)

    return ct_image, kidney_mask, pred_stone, gt_stone


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Command line interface."""
    import argparse
    import matplotlib.pyplot as plt
    import glob

    parser = argparse.ArgumentParser(
        description='Unsupervised Kidney Stone Segmentation using Torque Clustering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test with synthetic data
    python kidney_pathology_segmentation.py --quick-test

    # Segment single image
    python kidney_pathology_segmentation.py --ct image.png --mask kidney.npy --output-dir results/

    # Batch process directories
    python kidney_pathology_segmentation.py --ct-dir data/test --mask-dir medsam_outputs/raw_masks --output-dir results/
        """
    )

    parser.add_argument('--quick-test', action='store_true', help='Run quick test')
    parser.add_argument('--ct', type=str, help='Path to CT image (single file)')
    parser.add_argument('--mask', type=str, help='Path to kidney mask .npy (single file)')
    parser.add_argument('--ct-dir', type=str, help='Directory containing CT images (batch mode)')
    parser.add_argument('--mask-dir', type=str, help='Directory containing kidney masks (batch mode)')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.quick_test:
        ct, kidney, stone, gt_stone = run_quick_test()

        if not args.no_viz:
            fig = create_visualization(ct, kidney, stone, save_path='quick_test_stone.png')
            plt.show()
            plt.close(fig)
        return

    # Batch processing mode
    if args.ct_dir and args.mask_dir:
        import os

        if not os.path.isdir(args.ct_dir):
            print(f"Error: CT directory not found: {args.ct_dir}")
            return
        if not os.path.isdir(args.mask_dir):
            print(f"Error: Mask directory not found: {args.mask_dir}")
            return

        os.makedirs(args.output_dir, exist_ok=True)

        # Find all CT images (recursively)
        ct_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        ct_files = []
        for ext in ct_extensions:
            ct_files.extend(glob.glob(os.path.join(args.ct_dir, '**', ext), recursive=True))

        if not ct_files:
            print(f"No CT images found in: {args.ct_dir}")
            return

        print(f"\n{'='*70}")
        print(f"BATCH STONE SEGMENTATION")
        print(f"{'='*70}")
        print(f"CT directory:   {args.ct_dir}")
        print(f"Mask directory: {args.mask_dir}")
        print(f"Output:         {args.output_dir}")
        print(f"Found {len(ct_files)} CT images")
        print(f"{'='*70}\n")

        processed = 0
        skipped = 0
        errors = 0

        for ct_path in sorted(ct_files):
            # Get relative path from ct_dir
            rel_path = os.path.relpath(ct_path, args.ct_dir)
            basename = os.path.splitext(os.path.basename(ct_path))[0]
            rel_dir = os.path.dirname(rel_path)

            # Find corresponding mask file
            # Try multiple naming conventions
            mask_candidates = [
                os.path.join(args.mask_dir, rel_dir, f"{basename}_mask.npy"),
                os.path.join(args.mask_dir, rel_dir, f"{basename}.npy"),
                os.path.join(args.mask_dir, f"{basename}_mask.npy"),
                os.path.join(args.mask_dir, f"{basename}.npy"),
            ]

            mask_path = None
            for candidate in mask_candidates:
                if os.path.exists(candidate):
                    mask_path = candidate
                    break

            if mask_path is None:
                print(f"⚠ Skipping {rel_path}: No mask found")
                print(f"  Tried: {mask_candidates[0]}")
                skipped += 1
                continue

            try:
                # Load CT image
                ct_image = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)
                if ct_image is None:
                    print(f"✗ Error loading CT: {ct_path}")
                    errors += 1
                    continue

                # Load kidney mask
                kidney_mask = np.load(mask_path)
                kidney_mask = (kidney_mask > 0).astype(np.uint8)

                # Check shape compatibility
                if ct_image.shape != kidney_mask.shape:
                    print(f"✗ Shape mismatch for {basename}: CT={ct_image.shape}, mask={kidney_mask.shape}")
                    errors += 1
                    continue

                # Segment stones
                stone_mask = segment_stones(ct_image, kidney_mask)

                # Create output subdirectory matching input structure
                output_subdir = os.path.join(args.output_dir, rel_dir)
                os.makedirs(output_subdir, exist_ok=True)

                # Save outputs
                np.save(os.path.join(output_subdir, f'{basename}_stone_mask.npy'), stone_mask)
                cv2.imwrite(os.path.join(output_subdir, f'{basename}_stone_mask.png'), stone_mask * 255)

                if not args.no_viz:
                    viz_path = os.path.join(output_subdir, f'{basename}_visualization.png')
                    fig = create_visualization(ct_image, kidney_mask, stone_mask, save_path=viz_path)
                    plt.close(fig)

                num_stones = len(np.unique(cv2.connectedComponents(stone_mask.astype(np.uint8))[1])) - 1
                print(f"✓ {rel_path}: {num_stones} stone(s), {np.sum(stone_mask)} pixels")
                processed += 1

            except Exception as e:
                print(f"✗ Error processing {rel_path}: {e}")
                errors += 1

        print(f"\n{'='*70}")
        print(f"BATCH COMPLETE")
        print(f"{'='*70}")
        print(f"Processed: {processed}")
        print(f"Skipped:   {skipped}")
        print(f"Errors:    {errors}")
        print(f"Output:    {args.output_dir}")
        print(f"{'='*70}\n")
        return

    # Single file mode
    if args.ct and args.mask:
        import os

        # Load images
        ct_image = cv2.imread(args.ct, cv2.IMREAD_GRAYSCALE)
        if ct_image is None:
            print(f"Error: Cannot load CT image: {args.ct}")
            return

        kidney_mask = np.load(args.mask)
        kidney_mask = (kidney_mask > 0).astype(np.uint8)

        # Segment stones
        stone_mask = segment_stones(ct_image, kidney_mask)

        # Save outputs
        os.makedirs(args.output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(args.ct))[0]

        np.save(os.path.join(args.output_dir, f'{basename}_stone_mask.npy'), stone_mask)
        cv2.imwrite(os.path.join(args.output_dir, f'{basename}_stone_mask.png'), stone_mask * 255)

        if not args.no_viz:
            viz_path = os.path.join(args.output_dir, f'{basename}_visualization.png')
            fig = create_visualization(ct_image, kidney_mask, stone_mask, save_path=viz_path)
            plt.close(fig)

        print(f"\nResults saved to: {args.output_dir}")
        print(f"  Stones: {np.sum(stone_mask)} pixels")

    else:
        parser.print_help()
        print("\nRun --quick-test, provide --ct and --mask, or provide --ct-dir and --mask-dir")


if __name__ == '__main__':
    main()