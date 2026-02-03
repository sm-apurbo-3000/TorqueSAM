"""
Kidney Stone Segmentation Evaluation Pipeline
=============================================

Comprehensive evaluation module for comparing model-generated stone masks
against doctor-annotated ground truth images.

Key Features:
- Extracts filled regions from red boundary annotations
- Computes comprehensive segmentation metrics
- Generates detailed reports and visualizations
- Handles both .npy and .png prediction formats

Author: Asif Hasan Biplob
Thesis: Unsupervised Kidney Pathology Segmentation

Usage:
    python stone_evaluation.py
    
    Or modify the paths in the main() function at the bottom.
"""

import numpy as np
import cv2
import json
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION - MODIFY THESE PATHS FOR YOUR SYSTEM
# ============================================================================

# Ground truth directory (doctor-annotated images with red boundaries)
GROUND_TRUTH_DIR = r"E:\T2510638\TorqueSAM\RenSeg-main\RenSeg-main\data\ground_truth_masks"

# Prediction outputs directory (contains stone_mask.png and .npy files)
PREDICTIONS_DIR = r"E:\T2510638\TorqueSAM\RenSeg-main\RenSeg-main\pathology_outputs\pathology_outputs"

# Output directory for evaluation results
OUTPUT_DIR = r"E:\T2510638\TorqueSAM\RenSeg-main\RenSeg-main\stone_evaluation_results"


# ============================================================================
# GROUND TRUTH EXTRACTION
# ============================================================================

class GroundTruthExtractor:
    """
    Extract binary masks from doctor-annotated images with red boundaries.
    
    The ground truth images contain CT scans with kidney stones marked
    by red contour lines drawn by radiologists/doctors.
    """
    
    def __init__(self, 
                 red_hue_range: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 10), (160, 180)),
                 min_saturation: int = 50,
                 min_value: int = 50,
                 min_contour_area: int = 10):
        """
        Args:
            red_hue_range: Two ranges for red hue in HSV (red wraps around 0/180)
            min_saturation: Minimum saturation for red detection
            min_value: Minimum brightness value for red detection
            min_contour_area: Minimum area to consider as valid annotation
        """
        self.red_hue_range = red_hue_range
        self.min_saturation = min_saturation
        self.min_value = min_value
        self.min_contour_area = min_contour_area
    
    def extract_red_boundary(self, image: np.ndarray) -> np.ndarray:
        """
        Extract red boundary pixels from the image.
        
        Args:
            image: BGR image with red annotations
            
        Returns:
            Binary mask of red boundary pixels
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Red color spans across hue=0, so we need two ranges
        lower_red1 = np.array([self.red_hue_range[0][0], self.min_saturation, self.min_value])
        upper_red1 = np.array([self.red_hue_range[0][1], 255, 255])
        
        lower_red2 = np.array([self.red_hue_range[1][0], self.min_saturation, self.min_value])
        upper_red2 = np.array([self.red_hue_range[1][1], 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        red_mask = cv2.bitwise_or(mask1, mask2)
        return red_mask
    
    def fill_boundary_contours(self, boundary_mask: np.ndarray) -> np.ndarray:
        """
        Fill the interior of boundary contours to create solid masks.
        
        Args:
            boundary_mask: Binary mask containing only boundary pixels
            
        Returns:
            Binary mask with filled regions
        """
        # Find contours from boundary mask
        contours, hierarchy = cv2.findContours(
            boundary_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Create filled mask
        filled_mask = np.zeros_like(boundary_mask)
        
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_contour_area:
                valid_contours.append(contour)
        
        if valid_contours:
            # Fill all valid contours
            cv2.drawContours(filled_mask, valid_contours, -1, 255, cv2.FILLED)
        
        return filled_mask
    
    def extract_ground_truth_mask(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Extract binary ground truth mask from annotated image.
        
        Args:
            image_path: Path to the annotated image
            
        Returns:
            Binary mask (0 or 1) of the annotated region
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        
        # Extract red boundary
        boundary_mask = self.extract_red_boundary(image)
        
        # Fill the boundary to get solid region
        filled_mask = self.fill_boundary_contours(boundary_mask)
        
        # Also try morphological closing to connect broken boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_boundary = cv2.morphologyEx(boundary_mask, cv2.MORPH_CLOSE, kernel)
        filled_closed = self.fill_boundary_contours(closed_boundary)
        
        # Use scipy's fill_holes as additional method
        filled_scipy = binary_fill_holes(closed_boundary).astype(np.uint8) * 255
        
        # Combine all methods - take the maximum (union)
        combined = np.maximum(filled_mask, np.maximum(filled_closed, filled_scipy))
        
        # Convert to binary (0 or 1)
        binary_mask = (combined > 0).astype(np.uint8)
        
        return binary_mask


# ============================================================================
# PREDICTION LOADER
# ============================================================================

class PredictionLoader:
    """Load prediction masks from various formats."""
    
    @staticmethod
    def load_from_npy(npy_path: Union[str, Path]) -> np.ndarray:
        """
        Load prediction mask from .npy file.
        
        Args:
            npy_path: Path to .npy file
            
        Returns:
            Binary mask (0 or 1)
        """
        mask = np.load(str(npy_path))
        
        # Handle different array shapes
        if mask.ndim == 3:
            # If 3D, take first channel or convert
            if mask.shape[2] == 3:
                mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                mask = mask[:, :, 0]
        
        # Convert to binary
        binary_mask = (mask > 0).astype(np.uint8)
        return binary_mask
    
    @staticmethod
    def load_from_png(png_path: Union[str, Path]) -> np.ndarray:
        """
        Load prediction mask from PNG image.
        
        Args:
            png_path: Path to PNG file
            
        Returns:
            Binary mask (0 or 1)
        """
        mask = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read image: {png_path}")
        
        # Convert to binary
        binary_mask = (mask > 127).astype(np.uint8)
        return binary_mask
    
    @staticmethod
    def load_prediction(pred_path: Union[str, Path]) -> np.ndarray:
        """
        Load prediction mask from any supported format.
        
        Args:
            pred_path: Path to prediction file (.npy or .png)
            
        Returns:
            Binary mask (0 or 1)
        """
        pred_path = Path(pred_path)
        
        if pred_path.suffix == '.npy':
            return PredictionLoader.load_from_npy(pred_path)
        elif pred_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            return PredictionLoader.load_from_png(pred_path)
        else:
            raise ValueError(f"Unsupported file format: {pred_path.suffix}")


# ============================================================================
# SEGMENTATION METRICS
# ============================================================================

class SegmentationMetrics:
    """Compute comprehensive segmentation evaluation metrics."""
    
    @staticmethod
    def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Intersection over Union (Jaccard Index).
        
        IoU = |A ∩ B| / |A ∪ B|
        
        Args:
            pred: Predicted binary mask
            gt: Ground truth binary mask
            
        Returns:
            IoU score in [0, 1]
        """
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return float(intersection / union)
    
    @staticmethod
    def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Dice Coefficient (Sørensen–Dice Index).
        
        Dice = 2|A ∩ B| / (|A| + |B|)
        
        Args:
            pred: Predicted binary mask
            gt: Ground truth binary mask
            
        Returns:
            Dice score in [0, 1]
        """
        intersection = np.logical_and(pred, gt).sum()
        total = pred.sum() + gt.sum()
        
        if total == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return float(2 * intersection / total)
    
    @staticmethod
    def compute_pixel_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Pixel-wise accuracy.
        
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        
        Args:
            pred: Predicted binary mask
            gt: Ground truth binary mask
            
        Returns:
            Accuracy in [0, 1]
        """
        correct = (pred == gt).sum()
        total = pred.size
        return float(correct / total)
    
    @staticmethod
    def compute_precision(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Precision (Positive Predictive Value).
        
        Precision = TP / (TP + FP)
        
        Args:
            pred: Predicted binary mask
            gt: Ground truth binary mask
            
        Returns:
            Precision in [0, 1]
        """
        tp = np.logical_and(pred == 1, gt == 1).sum()
        fp = np.logical_and(pred == 1, gt == 0).sum()
        
        if tp + fp == 0:
            return 0.0
        
        return float(tp / (tp + fp))
    
    @staticmethod
    def compute_recall(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Recall (Sensitivity / True Positive Rate).
        
        Recall = TP / (TP + FN)
        
        Args:
            pred: Predicted binary mask
            gt: Ground truth binary mask
            
        Returns:
            Recall in [0, 1]
        """
        tp = np.logical_and(pred == 1, gt == 1).sum()
        fn = np.logical_and(pred == 0, gt == 1).sum()
        
        if tp + fn == 0:
            return 0.0
        
        return float(tp / (tp + fn))
    
    @staticmethod
    def compute_specificity(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Specificity (True Negative Rate).
        
        Specificity = TN / (TN + FP)
        
        Args:
            pred: Predicted binary mask
            gt: Ground truth binary mask
            
        Returns:
            Specificity in [0, 1]
        """
        tn = np.logical_and(pred == 0, gt == 0).sum()
        fp = np.logical_and(pred == 1, gt == 0).sum()
        
        if tn + fp == 0:
            return 0.0
        
        return float(tn / (tn + fp))
    
    @staticmethod
    def compute_f1_score(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        F1 Score (harmonic mean of precision and recall).
        
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        
        Args:
            pred: Predicted binary mask
            gt: Ground truth binary mask
            
        Returns:
            F1 score in [0, 1]
        """
        precision = SegmentationMetrics.compute_precision(pred, gt)
        recall = SegmentationMetrics.compute_recall(pred, gt)
        
        if precision + recall == 0:
            return 0.0
        
        return float(2 * precision * recall / (precision + recall))
    
    @staticmethod
    def compute_hausdorff_distance(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Hausdorff Distance (maximum surface distance).
        
        Measures the maximum distance from any point on one boundary
        to the closest point on the other boundary.
        
        Args:
            pred: Predicted binary mask
            gt: Ground truth binary mask
            
        Returns:
            Hausdorff distance in pixels (lower is better)
        """
        # Find contour points
        pred_contours = cv2.findContours(
            pred.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_NONE
        )[0]
        
        gt_contours = cv2.findContours(
            gt.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_NONE
        )[0]
        
        if len(pred_contours) == 0 or len(gt_contours) == 0:
            return float('inf')
        
        # Get boundary points
        pred_points = np.vstack([c.reshape(-1, 2) for c in pred_contours])
        gt_points = np.vstack([c.reshape(-1, 2) for c in gt_contours])
        
        # Symmetric Hausdorff distance
        hd_pred_to_gt = directed_hausdorff(pred_points, gt_points)[0]
        hd_gt_to_pred = directed_hausdorff(gt_points, pred_points)[0]
        
        return float(max(hd_pred_to_gt, hd_gt_to_pred))
    
    @staticmethod
    def compute_average_surface_distance(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Average Surface Distance (ASD).
        
        Mean of all minimum distances from boundary points.
        
        Args:
            pred: Predicted binary mask
            gt: Ground truth binary mask
            
        Returns:
            ASD in pixels (lower is better)
        """
        pred_contours = cv2.findContours(
            pred.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_NONE
        )[0]
        
        gt_contours = cv2.findContours(
            gt.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_NONE
        )[0]
        
        if len(pred_contours) == 0 or len(gt_contours) == 0:
            return float('inf')
        
        pred_points = np.vstack([c.reshape(-1, 2) for c in pred_contours]).astype(float)
        gt_points = np.vstack([c.reshape(-1, 2) for c in gt_contours]).astype(float)
        
        # Compute distances from pred to gt
        def min_distances(points_a, points_b):
            distances = []
            for p in points_a:
                d = np.min(np.linalg.norm(points_b - p, axis=1))
                distances.append(d)
            return np.array(distances)
        
        dist_pred_to_gt = min_distances(pred_points, gt_points)
        dist_gt_to_pred = min_distances(gt_points, pred_points)
        
        asd = (dist_pred_to_gt.mean() + dist_gt_to_pred.mean()) / 2
        return float(asd)
    
    @staticmethod
    def compute_all_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
        """
        Compute all segmentation metrics.
        
        Args:
            pred: Predicted binary mask
            gt: Ground truth binary mask
            
        Returns:
            Dictionary containing all metrics
        """
        return {
            'iou': SegmentationMetrics.compute_iou(pred, gt),
            'dice': SegmentationMetrics.compute_dice(pred, gt),
            'pixel_accuracy': SegmentationMetrics.compute_pixel_accuracy(pred, gt),
            'precision': SegmentationMetrics.compute_precision(pred, gt),
            'recall': SegmentationMetrics.compute_recall(pred, gt),
            'specificity': SegmentationMetrics.compute_specificity(pred, gt),
            'f1_score': SegmentationMetrics.compute_f1_score(pred, gt),
            'hausdorff_distance': SegmentationMetrics.compute_hausdorff_distance(pred, gt),
            'avg_surface_distance': SegmentationMetrics.compute_average_surface_distance(pred, gt)
        }


# ============================================================================
# EVALUATION ENGINE
# ============================================================================

class StoneSegmentationEvaluator:
    """
    Main evaluation engine for kidney stone segmentation.
    
    Handles the complete evaluation pipeline:
    1. Loading ground truth from annotated images
    2. Loading model predictions
    3. Computing metrics
    4. Generating reports and visualizations
    """
    
    def __init__(self,
                 ground_truth_dir: Union[str, Path],
                 predictions_dir: Union[str, Path],
                 output_dir: Union[str, Path],
                 use_npy: bool = True):
        """
        Args:
            ground_truth_dir: Directory containing annotated GT images
            predictions_dir: Directory containing model predictions
            output_dir: Directory for saving evaluation results
            use_npy: If True, load from .npy files; else use stone_mask.png
        """
        self.gt_dir = Path(ground_truth_dir)
        self.pred_dir = Path(predictions_dir)
        self.output_dir = Path(output_dir)
        self.use_npy = use_npy
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        (self.output_dir / 'csv').mkdir(exist_ok=True)
        
        # Initialize components
        self.gt_extractor = GroundTruthExtractor()
        self.pred_loader = PredictionLoader()
        self.metrics = SegmentationMetrics()
        
        # Storage for results
        self.results: List[Dict] = []
        
        print("=" * 70)
        print("KIDNEY STONE SEGMENTATION EVALUATOR")
        print("=" * 70)
        print(f"Ground Truth Directory: {self.gt_dir}")
        print(f"Predictions Directory:  {self.pred_dir}")
        print(f"Output Directory:       {self.output_dir}")
        print(f"Using NPY files:        {self.use_npy}")
        print("=" * 70)
    
    def find_matching_files(self) -> List[Tuple[Path, Path, str]]:
        """
        Find matching ground truth and prediction file pairs.
        
        Returns:
            List of (gt_path, pred_path, image_name) tuples
        """
        matches = []
        
        # Get all ground truth files
        gt_files = list(self.gt_dir.glob('*.jpg')) + \
                   list(self.gt_dir.glob('*.png')) + \
                   list(self.gt_dir.glob('*.jpeg'))
        
        print(f"\nFound {len(gt_files)} ground truth files")
        
        for gt_file in gt_files:
            # Extract base name (without extension)
            base_name = gt_file.stem
            
            # Remove common suffixes that might be in GT names
            clean_name = base_name
            for suffix in ['_gt', '_ground_truth', '_annotated', '_mask']:
                clean_name = clean_name.replace(suffix, '')
            
            # Look for corresponding prediction
            pred_file = None
            
            if self.use_npy:
                # Look in subfolders for .npy files
                # Pattern: predictions_dir/<image_name>/stone_mask.npy
                pred_candidates = [
                    self.pred_dir / clean_name / 'stone_mask.npy',
                    self.pred_dir / base_name / 'stone_mask.npy',
                    self.pred_dir / f'{clean_name}_stone_mask.npy',
                    self.pred_dir / f'{clean_name}.npy',
                ]
                
                # Also search recursively
                for npy_file in self.pred_dir.glob(f'**/{clean_name}*stone*.npy'):
                    pred_candidates.append(npy_file)
                for npy_file in self.pred_dir.glob(f'**/*{clean_name}*.npy'):
                    if 'stone' in npy_file.stem.lower():
                        pred_candidates.append(npy_file)
            else:
                # Look for PNG stone masks
                pred_candidates = [
                    self.pred_dir / clean_name / 'stone_mask.png',
                    self.pred_dir / base_name / 'stone_mask.png',
                    self.pred_dir / f'{clean_name}_stone_mask.png',
                ]
                
                for png_file in self.pred_dir.glob(f'**/{clean_name}*stone*.png'):
                    pred_candidates.append(png_file)
            
            # Find first existing candidate
            for candidate in pred_candidates:
                if candidate.exists():
                    pred_file = candidate
                    break
            
            if pred_file:
                matches.append((gt_file, pred_file, clean_name))
        
        print(f"Found {len(matches)} matching pairs")
        return matches
    
    def evaluate_single(self, 
                       gt_path: Path, 
                       pred_path: Path, 
                       image_name: str,
                       save_visualization: bool = True) -> Optional[Dict]:
        """
        Evaluate a single image pair.
        
        Args:
            gt_path: Path to ground truth image
            pred_path: Path to prediction file
            image_name: Name identifier for the image
            save_visualization: Whether to save comparison visualization
            
        Returns:
            Dictionary containing all metrics, or None if masks are empty
        """
        # Load ground truth
        gt_mask = self.gt_extractor.extract_ground_truth_mask(gt_path)
        
        # Load prediction
        pred_mask = self.pred_loader.load_prediction(pred_path)
        
        # Ensure same size
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(
                pred_mask, 
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        # Check for empty masks
        gt_empty = gt_mask.sum() == 0
        pred_empty = pred_mask.sum() == 0
        
        if gt_empty and pred_empty:
            # Both masks are empty - skip this image
            return None
        
        if gt_empty:
            # Ground truth is empty but prediction is not - skip (no GT to compare)
            return None
        
        if pred_empty:
            # Prediction is empty but GT exists - this is a valid failure case
            # We still skip it as requested, but you could choose to include it
            return None
        
        # Compute all metrics
        metrics = self.metrics.compute_all_metrics(pred_mask, gt_mask)
        metrics['image'] = image_name
        metrics['gt_pixels'] = int(gt_mask.sum())
        metrics['pred_pixels'] = int(pred_mask.sum())
        
        # Save visualization
        if save_visualization:
            self._save_visualization(gt_path, gt_mask, pred_mask, image_name, metrics)
        
        return metrics
    
    def _save_visualization(self, 
                           original_path: Path,
                           gt_mask: np.ndarray, 
                           pred_mask: np.ndarray, 
                           image_name: str,
                           metrics: Dict):
        """Save comparison visualization."""
        # Load original image
        original = cv2.imread(str(original_path))
        if original is None:
            return
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Resize masks to match original if needed
        if gt_mask.shape[:2] != original.shape[:2]:
            gt_mask = cv2.resize(gt_mask, (original.shape[1], original.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
            pred_mask = cv2.resize(pred_mask, (original.shape[1], original.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
        
        # Create comparison figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1: Original, GT Mask, Pred Mask
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original (with annotation)', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gt_mask, cmap='gray')
        axes[0, 1].set_title(f'Ground Truth\n({gt_mask.sum()} pixels)', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(pred_mask, cmap='gray')
        axes[0, 2].set_title(f'Prediction\n({pred_mask.sum()} pixels)', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2: Overlay, Difference, Metrics
        # Overlay
        overlay = original.copy()
        overlay[gt_mask == 1] = [0, 255, 0]  # Green for GT
        overlay[pred_mask == 1] = [255, 0, 0]  # Red for Pred
        overlap = np.logical_and(gt_mask, pred_mask)
        overlay[overlap] = [255, 255, 0]  # Yellow for overlap
        blended = cv2.addWeighted(original, 0.6, overlay, 0.4, 0)
        
        axes[1, 0].imshow(blended)
        axes[1, 0].set_title('Overlay (Green=GT, Red=Pred, Yellow=Overlap)', fontsize=11, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Difference map
        diff = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        diff[np.logical_and(gt_mask == 1, pred_mask == 0)] = [255, 0, 0]    # FN: Red
        diff[np.logical_and(gt_mask == 0, pred_mask == 1)] = [0, 0, 255]    # FP: Blue
        diff[np.logical_and(gt_mask == 1, pred_mask == 1)] = [0, 255, 0]    # TP: Green
        
        axes[1, 1].imshow(diff)
        axes[1, 1].set_title('Difference (Green=TP, Red=FN, Blue=FP)', fontsize=11, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Metrics text
        axes[1, 2].axis('off')
        metrics_text = (
            f"IoU:       {metrics['iou']:.4f}\n"
            f"Dice:      {metrics['dice']:.4f}\n"
            f"Precision: {metrics['precision']:.4f}\n"
            f"Recall:    {metrics['recall']:.4f}\n"
            f"F1 Score:  {metrics['f1_score']:.4f}\n"
            f"Accuracy:  {metrics['pixel_accuracy']:.4f}\n"
            f"Hausdorff: {metrics['hausdorff_distance']:.2f} px"
        )
        axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=14, fontfamily='monospace',
                       verticalalignment='center', transform=axes[1, 2].transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 2].set_title('Metrics', fontsize=12, fontweight='bold')
        
        plt.suptitle(f'Evaluation: {image_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / 'visualizations' / f'{image_name}_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def evaluate_all(self, save_visualizations: bool = True) -> pd.DataFrame:
        """
        Evaluate all matching image pairs.
        
        Args:
            save_visualizations: Whether to save comparison visualizations
            
        Returns:
            DataFrame with all results
        """
        matches = self.find_matching_files()
        
        if len(matches) == 0:
            print("\nERROR: No matching files found!")
            print("Please check that:")
            print("  1. Ground truth files exist in the specified directory")
            print("  2. Prediction files follow the expected naming convention")
            return pd.DataFrame()
        
        print(f"\nEvaluating {len(matches)} image pairs...")
        print("-" * 70)
        
        skipped_empty_gt = 0
        skipped_empty_pred = 0
        skipped_both_empty = 0
        errors = 0
        
        for i, (gt_path, pred_path, image_name) in enumerate(matches, 1):
            try:
                metrics = self.evaluate_single(
                    gt_path, pred_path, image_name, 
                    save_visualization=save_visualizations
                )
                
                if metrics is None:
                    # Determine why it was skipped
                    gt_mask = self.gt_extractor.extract_ground_truth_mask(gt_path)
                    pred_mask = self.pred_loader.load_prediction(pred_path)
                    if pred_mask.shape != gt_mask.shape:
                        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                                              interpolation=cv2.INTER_NEAREST)
                    
                    gt_empty = gt_mask.sum() == 0
                    pred_empty = pred_mask.sum() == 0
                    
                    if gt_empty and pred_empty:
                        skipped_both_empty += 1
                        print(f"[{i:3d}/{len(matches)}] {image_name:30s} | SKIPPED: Both masks empty")
                    elif gt_empty:
                        skipped_empty_gt += 1
                        print(f"[{i:3d}/{len(matches)}] {image_name:30s} | SKIPPED: GT mask empty")
                    elif pred_empty:
                        skipped_empty_pred += 1
                        print(f"[{i:3d}/{len(matches)}] {image_name:30s} | SKIPPED: Prediction mask empty")
                else:
                    self.results.append(metrics)
                    print(f"[{i:3d}/{len(matches)}] {image_name:30s} | "
                          f"IoU: {metrics['iou']:.4f} | Dice: {metrics['dice']:.4f} | "
                          f"F1: {metrics['f1_score']:.4f}")
                
            except Exception as e:
                errors += 1
                print(f"[{i:3d}/{len(matches)}] {image_name:30s} | ERROR: {str(e)}")
        
        # Print skip summary
        print("-" * 70)
        print(f"\nSUMMARY:")
        print(f"  Total image pairs found:     {len(matches)}")
        print(f"  Successfully evaluated:      {len(self.results)}")
        print(f"  Skipped (both masks empty):  {skipped_both_empty}")
        print(f"  Skipped (GT mask empty):     {skipped_empty_gt}")
        print(f"  Skipped (Pred mask empty):   {skipped_empty_pred}")
        print(f"  Errors:                      {errors}")
        
        # Store skip statistics
        self.skip_stats = {
            'total_pairs': len(matches),
            'evaluated': len(self.results),
            'skipped_both_empty': skipped_both_empty,
            'skipped_gt_empty': skipped_empty_gt,
            'skipped_pred_empty': skipped_empty_pred,
            'errors': errors
        }
        
        return pd.DataFrame(self.results)
    
    def get_summary_statistics(self) -> Dict:
        """Compute summary statistics from results."""
        if len(self.results) == 0:
            return {}
        
        df = pd.DataFrame(self.results)
        
        metrics = ['iou', 'dice', 'pixel_accuracy', 'precision', 'recall',
                  'specificity', 'f1_score', 'hausdorff_distance', 'avg_surface_distance']
        
        summary = {}
        for metric in metrics:
            values = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
            summary[metric] = {
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'median': values.median()
            }
        
        return summary
    
    def print_summary(self):
        """Print formatted summary statistics."""
        summary = self.get_summary_statistics()
        
        if not summary:
            print("No results to summarize")
            return
        
        print("\n" + "=" * 90)
        print("EVALUATION SUMMARY STATISTICS")
        print("=" * 90)
        print(f"\n{'Metric':<22} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Median':<12}")
        print("-" * 90)
        
        for metric, stats in summary.items():
            print(f"{metric:<22} {stats['mean']:<12.4f} {stats['std']:<12.4f} "
                  f"{stats['min']:<12.4f} {stats['max']:<12.4f} {stats['median']:<12.4f}")
        
        print("=" * 90)
        print(f"\nTotal images evaluated: {len(self.results)}")
    
    def save_results(self):
        """Save all results to CSV files."""
        if len(self.results) == 0:
            print("No results to save")
            return
        
        df = pd.DataFrame(self.results)
        
        # Save detailed results
        detailed_path = self.output_dir / 'csv' / 'detailed_results.csv'
        df.to_csv(detailed_path, index=False)
        print(f"\n✓ Detailed results saved to {detailed_path}")
        
        # Save summary statistics
        summary = self.get_summary_statistics()
        summary_df = pd.DataFrame(summary).T
        summary_df.index.name = 'metric'
        summary_path = self.output_dir / 'csv' / 'summary_statistics.csv'
        summary_df.to_csv(summary_path)
        print(f"✓ Summary statistics saved to {summary_path}")
        
        return detailed_path, summary_path
    
    def plot_metric_distributions(self, save_path: Optional[Path] = None):
        """Plot distribution histograms for all metrics."""
        if len(self.results) == 0:
            print("No results to plot")
            return
        
        df = pd.DataFrame(self.results)
        
        metrics = ['iou', 'dice', 'pixel_accuracy', 'precision', 
                  'recall', 'specificity', 'f1_score']
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            data = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
            
            ax.hist(data, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {data.mean():.3f}')
            ax.axvline(data.median(), color='green', linestyle='-.', linewidth=2,
                      label=f'Median: {data.median():.3f}')
            
            ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Hausdorff distance
        ax = axes[7]
        hd_data = df['hausdorff_distance'].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(hd_data, bins=20, edgecolor='black', alpha=0.7, color='coral')
        ax.axvline(hd_data.mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {hd_data.mean():.2f}')
        ax.set_title('Hausdorff Distance', fontsize=12, fontweight='bold')
        ax.set_xlabel('Distance (pixels)')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        fig.delaxes(axes[8])
        
        plt.suptitle('Segmentation Metrics Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'metric_distributions.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Metric distributions saved to {save_path}")
        plt.close()
    
    def plot_scatter_comparison(self, save_path: Optional[Path] = None):
        """Plot scatter plot of IoU vs Dice scores."""
        if len(self.results) == 0:
            return
        
        df = pd.DataFrame(self.results)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(df['iou'], df['dice'], 
                            c=df['precision'], cmap='viridis',
                            s=100, alpha=0.7, edgecolors='black')
        
        # Add diagonal line (IoU = Dice would be on this if masks were identical)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='x=y')
        
        ax.set_xlabel('IoU Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Dice Score', fontsize=12, fontweight='bold')
        ax.set_title('IoU vs Dice Score (colored by Precision)', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Precision', fontsize=11)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'iou_vs_dice_scatter.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Scatter plot saved to {save_path}")
        plt.close()
    
    def plot_metrics_box_plot(self, save_path: Optional[Path] = None):
        """Plot box plots for all metrics."""
        if len(self.results) == 0:
            return
        
        df = pd.DataFrame(self.results)
        
        metrics = ['iou', 'dice', 'precision', 'recall', 'f1_score', 'specificity']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        data_to_plot = [df[m].dropna().values for m in metrics]
        bp = ax.boxplot(data_to_plot, labels=[m.replace('_', '\n').title() for m in metrics],
                       patch_artist=True)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(metrics)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Segmentation Metrics Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
        
        # Add mean markers
        means = [df[m].mean() for m in metrics]
        ax.scatter(range(1, len(metrics) + 1), means, marker='D', color='red', 
                  s=50, zorder=3, label='Mean')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'metrics_boxplot.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Box plot saved to {save_path}")
        plt.close()
    
    def generate_full_report(self, save_visualizations: bool = True):
        """
        Generate complete evaluation report with all metrics and visualizations.
        
        Args:
            save_visualizations: Whether to save per-image comparison visualizations
        """
        print("\n" + "=" * 70)
        print("GENERATING COMPREHENSIVE EVALUATION REPORT")
        print("=" * 70)
        
        # Run evaluation
        self.evaluate_all(save_visualizations=save_visualizations)
        
        if len(self.results) == 0:
            print("\nNo results generated. Please check your file paths.")
            return
        
        # Print summary
        self.print_summary()
        
        # Save results
        self.save_results()
        
        # Generate plots
        print("\nGenerating visualizations...")
        self.plot_metric_distributions()
        self.plot_scatter_comparison()
        self.plot_metrics_box_plot()
        
        # Save configuration
        config = {
            'ground_truth_dir': str(self.gt_dir),
            'predictions_dir': str(self.pred_dir),
            'output_dir': str(self.output_dir),
            'use_npy': self.use_npy,
            'num_evaluated': len(self.results),
            'skip_statistics': getattr(self, 'skip_stats', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        config_path = self.output_dir / 'evaluation_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE")
        print("=" * 70)
        print(f"\nResults saved to: {self.output_dir}")
        print(f"  - Detailed CSV:    {self.output_dir / 'csv' / 'detailed_results.csv'}")
        print(f"  - Summary CSV:     {self.output_dir / 'csv' / 'summary_statistics.csv'}")
        print(f"  - Visualizations:  {self.output_dir / 'visualizations'}")
        print(f"  - Plots:           {self.output_dir}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point."""
    
    # =========================================
    # CONFIGURE YOUR PATHS HERE
    # =========================================
    
    ground_truth_dir = GROUND_TRUTH_DIR      # Doctor-annotated images with red boundaries
    predictions_dir = PREDICTIONS_DIR         # Your model's output directory
    output_dir = OUTPUT_DIR                   # Where to save evaluation results
    
    # Set to True to use .npy files, False to use stone_mask.png files
    use_npy = True  # Set to False since you mentioned stone_mask in the folder
    
    # =========================================
    
    # Create evaluator
    evaluator = StoneSegmentationEvaluator(
        ground_truth_dir=ground_truth_dir,
        predictions_dir=predictions_dir,
        output_dir=output_dir,
        use_npy=use_npy
    )
    
    # Generate full report
    evaluator.generate_full_report(save_visualizations=True)


if __name__ == "__main__":
    main()