"""
Comprehensive evaluation module for MedSAM segmentation quality.
Compares predicted segmentation masks against ground truth annotations.
"""

import numpy as np
import cv2
import json
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import (RAW_MASKS_DIR, GROUND_TRUTH_DIR,
                    EVALUATION_DIR, EVALUATION_PLOTS_DIR, EVALUATION_CSV_DIR)


class SegmentationEvaluator:
    """Evaluate MedSAM segmentation quality against ground truth masks."""

    def __init__(self, predictions_dir=None, ground_truth_dir=None):
        """
        Args:
            predictions_dir: Path to predicted masks (default: RAW_MASKS_DIR)
            ground_truth_dir: Path to ground truth binary masks (default: GROUND_TRUTH_DIR)
        """
        self.predictions_dir = Path(predictions_dir) if predictions_dir else Path(RAW_MASKS_DIR)
        self.ground_truth_dir = Path(ground_truth_dir) if ground_truth_dir else Path(GROUND_TRUTH_DIR)
        self.results = []

        print(f"Predictions directory: {self.predictions_dir}")
        print(f"Ground truth directory: {self.ground_truth_dir}")

    def load_prediction_mask(self, pred_path):
        """
        Load predicted mask from .npy file.

        Args:
            pred_path: Path to .npy file containing segmentation mask

        Returns:
            Binary mask (0 or 1)
        """
        # Load numpy array with integer labels
        mask = np.load(str(pred_path))

        # Convert to binary (any label > 0 is foreground)
        binary_mask = (mask > 0).astype(np.uint8)

        return binary_mask

    def load_ground_truth(self, gt_path):
        """
        Load ground truth mask and normalize to 0/1.

        Supports two formats:
        1. Binary masks (0=background, 255=foreground) - grayscale
        2. Annotated CT images (colored annotations on CT) - RGB

        Args:
            gt_path: Path to ground truth image (PNG/JPG)

        Returns:
            Binary mask (0 or 1)
        """
        # Try loading as color image first (for annotated CTs)
        gt_color = cv2.imread(str(gt_path))

        if gt_color is None:
            raise FileNotFoundError(f"Ground truth not found: {gt_path}")

        # Check if it's already a binary mask (grayscale with only 0 and 255)
        gt_gray = cv2.cvtColor(gt_color, cv2.COLOR_BGR2GRAY)
        unique_values = np.unique(gt_gray)

        # If it's already binary-like (only 2 unique values or close to 0 and 255)
        if len(unique_values) <= 2 or (gt_gray.min() < 10 and gt_gray.max() > 245):
            _, gt_binary = cv2.threshold(gt_gray, 127, 1, cv2.THRESH_BINARY)
            return gt_binary

        # Otherwise, treat as annotated CT image
        # Extract annotations by detecting colored markings
        print(f"  → Detected annotated image format, extracting annotations...")

        # Method 1: Detect any non-grayscale pixels (color annotations)
        # Convert to HSV to detect saturation (colored pixels have high saturation)
        hsv = cv2.cvtColor(gt_color, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]

        # High saturation = colored annotation
        _, color_mask = cv2.threshold(saturation, 30, 1, cv2.THRESH_BINARY)

        # Method 2: Detect bright/red annotations (common in medical imaging)
        # Red color range in HSV
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_mask = (red_mask > 0).astype(np.uint8)

        # Combine both methods
        final_mask = np.logical_or(color_mask, red_mask).astype(np.uint8)

        # If very few pixels detected, try edge detection (for outlined annotations)
        if final_mask.sum() < (final_mask.size * 0.01):  # Less than 1% of pixels
            print(f"  → Few colored pixels detected, trying edge-based detection...")
            edges = cv2.Canny(gt_gray, 50, 150)
            final_mask = (edges > 0).astype(np.uint8)

        return final_mask

    def compute_iou(self, pred_mask, gt_mask):
        """
        Compute Intersection over Union (Jaccard Index).

        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask

        Returns:
            IoU score [0, 1]
        """
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()

        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        iou = intersection / union
        return float(iou)

    def compute_dice(self, pred_mask, gt_mask):
        """
        Compute Dice Coefficient (F1 Score).

        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask

        Returns:
            Dice score [0, 1]
        """
        intersection = np.logical_and(pred_mask, gt_mask).sum()

        if pred_mask.sum() + gt_mask.sum() == 0:
            return 1.0 if intersection == 0 else 0.0

        dice = (2 * intersection) / (pred_mask.sum() + gt_mask.sum())
        return float(dice)

    def compute_pixel_accuracy(self, pred_mask, gt_mask):
        """Compute pixel-wise accuracy."""
        correct = (pred_mask == gt_mask).sum()
        total = pred_mask.size
        return float(correct / total)

    def compute_precision_recall(self, pred_mask, gt_mask):
        """Compute precision and recall."""
        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()

        precision = precision_score(gt_flat, pred_flat, zero_division=0)
        recall = recall_score(gt_flat, pred_flat, zero_division=0)

        return float(precision), float(recall)

    def compute_hausdorff_distance(self, pred_mask, gt_mask):
        """
        Compute Hausdorff distance between boundaries.

        Returns:
            Hausdorff distance in pixels (lower is better)
        """
        # Find contours
        pred_contours = cv2.findContours(pred_mask.astype(np.uint8),
                                         cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_NONE)[0]
        gt_contours = cv2.findContours(gt_mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)[0]

        if len(pred_contours) == 0 or len(gt_contours) == 0:
            return float('inf')

        # Get boundary points
        pred_points = np.vstack([c.reshape(-1, 2) for c in pred_contours])
        gt_points = np.vstack([c.reshape(-1, 2) for c in gt_contours])

        # Compute directed Hausdorff distance (both directions)
        hd1 = directed_hausdorff(pred_points, gt_points)[0]
        hd2 = directed_hausdorff(gt_points, pred_points)[0]

        # Return maximum (symmetric Hausdorff distance)
        return float(max(hd1, hd2))

    def compute_specificity(self, pred_mask, gt_mask):
        """Compute specificity (true negative rate)."""
        tn = np.logical_and(pred_mask == 0, gt_mask == 0).sum()
        fp = np.logical_and(pred_mask == 1, gt_mask == 0).sum()

        if tn + fp == 0:
            return 0.0

        return float(tn / (tn + fp))

    def evaluate_single_image(self, pred_path, gt_path, image_name):
        """
        Evaluate a single image pair.

        Args:
            pred_path: Path to predicted mask (.npy)
            gt_path: Path to ground truth mask (PNG/JPG)
            image_name: Name of the image

        Returns:
            Dictionary of evaluation metrics
        """
        # Load masks
        pred_mask = self.load_prediction_mask(pred_path)
        gt_mask = self.load_ground_truth(gt_path)

        # Ensure same size
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        # Compute metrics
        iou = self.compute_iou(pred_mask, gt_mask)
        dice = self.compute_dice(pred_mask, gt_mask)
        pixel_acc = self.compute_pixel_accuracy(pred_mask, gt_mask)
        precision, recall = self.compute_precision_recall(pred_mask, gt_mask)
        hausdorff = self.compute_hausdorff_distance(pred_mask, gt_mask)
        specificity = self.compute_specificity(pred_mask, gt_mask)

        # Compute F1 from precision/recall
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        # Store results
        result = {
            'image': image_name,
            'iou': iou,
            'dice': dice,
            'pixel_accuracy': pixel_acc,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'hausdorff_distance': hausdorff
        }

        self.results.append(result)
        return result

    def evaluate_all(self):
        """Evaluate all images in the directories."""
        # Find all .npy prediction files
        pred_files = sorted(self.predictions_dir.glob('**/*_mask.npy'))

        if len(pred_files) == 0:
            print(f"ERROR: No prediction files found in {self.predictions_dir}")
            return

        print(f"\nFound {len(pred_files)} prediction files")
        print("Starting evaluation...\n")

        for pred_file in pred_files:
            # Extract base filename (remove _mask.npy)
            base_name = pred_file.stem.replace('_mask', '')

            # Find corresponding ground truth
            # Try different extensions
            gt_file = None
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
                potential_gt = self.ground_truth_dir / f"{base_name}{ext}"
                if potential_gt.exists():
                    gt_file = potential_gt
                    break

            if gt_file is None or not gt_file.exists():
                print(f"⚠ Warning: No ground truth found for {base_name}")
                continue

            try:
                result = self.evaluate_single_image(pred_file, gt_file, base_name)
                print(f"✓ {base_name:30s} | IoU: {result['iou']:.4f} | Dice: {result['dice']:.4f} | F1: {result['f1_score']:.4f}")
            except Exception as e:
                print(f"✗ {base_name:30s} | Error: {str(e)}")

    def get_summary_statistics(self):
        """Get mean and std of all metrics."""
        if len(self.results) == 0:
            print("No results to summarize")
            return None

        df = pd.DataFrame(self.results)

        # Exclude image name and inf values
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

        summary = {
            'mean': numeric_df.mean(),
            'std': numeric_df.std(),
            'min': numeric_df.min(),
            'max': numeric_df.max(),
            'median': numeric_df.median()
        }

        return summary

    def save_results(self, output_path=None):
        """Save detailed results to CSV."""
        if len(self.results) == 0:
            print("No results to save")
            return

        if output_path is None:
            output_path = EVALUATION_CSV_DIR / "detailed_results.csv"

        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)
        print(f"\n✓ Detailed results saved to {output_path}")

        return output_path

    def save_summary(self, output_path=None):
        """Save summary statistics to CSV."""
        summary = self.get_summary_statistics()

        if summary is None:
            return

        if output_path is None:
            output_path = EVALUATION_CSV_DIR / "summary_statistics.csv"

        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(output_path)
        print(f"✓ Summary statistics saved to {output_path}")

        return output_path

    def plot_metric_distributions(self, save_path=None):
        """Plot distributions of all metrics."""
        if len(self.results) == 0:
            print("No results to plot")
            return

        df = pd.DataFrame(self.results)

        # Metrics to plot
        metrics = ['iou', 'dice', 'pixel_accuracy', 'precision', 'recall', 'specificity', 'f1_score']

        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            data = df[metric].replace([np.inf, -np.inf], np.nan).dropna()

            ax.hist(data, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {data.mean():.3f}')
            ax.axvline(data.median(), color='green', linestyle='-.', linewidth=2,
                      label=f'Median: {data.median():.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hausdorff distance (separate scale)
        ax = axes[7]
        hd_data = df['hausdorff_distance'].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(hd_data, bins=20, edgecolor='black', alpha=0.7, color='coral')
        ax.set_title('Hausdorff Distance', fontsize=12, fontweight='bold')
        ax.set_xlabel('Distance (pixels)')
        ax.set_ylabel('Frequency')
        ax.axvline(hd_data.mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {hd_data.mean():.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Remove extra subplot
        fig.delaxes(axes[8])

        plt.tight_layout()

        if save_path is None:
            save_path = EVALUATION_PLOTS_DIR / "metric_distributions.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Metric distributions saved to {save_path}")
        plt.close()

    def plot_confusion_matrix_summary(self, save_path=None):
        """Plot aggregated confusion matrix."""
        if len(self.results) == 0:
            print("No results to plot")
            return

        df = pd.DataFrame(self.results)

        # Calculate average metrics
        avg_precision = df['precision'].mean()
        avg_recall = df['recall'].mean()
        avg_specificity = df['specificity'].mean()

        # Approximate confusion matrix values
        # TP / (TP + FP) = Precision  =>  TP = Precision * (TP + FP)
        # TP / (TP + FN) = Recall     =>  TP = Recall * (TP + FN)

        fig, ax = plt.subplots(figsize=(8, 6))

        # Create a simple summary table
        summary_data = {
            'Metric': ['Precision', 'Recall (Sensitivity)', 'Specificity', 'F1 Score'],
            'Mean': [avg_precision, avg_recall, avg_specificity, df['f1_score'].mean()],
            'Std': [df['precision'].std(), df['recall'].std(), df['specificity'].std(), df['f1_score'].std()]
        }

        summary_df = pd.DataFrame(summary_data)

        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                        cellLoc='center', loc='center', colWidths=[0.4, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)

        # Style header
        for i in range(len(summary_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        plt.title('Classification Metrics Summary', fontsize=14, fontweight='bold', pad=20)

        if save_path is None:
            save_path = EVALUATION_PLOTS_DIR / "metrics_summary_table.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Metrics summary table saved to {save_path}")
        plt.close()

    def plot_per_image_comparison(self, save_path=None, top_n=10):
        """Plot per-image IoU and Dice scores."""
        if len(self.results) == 0:
            print("No results to plot")
            return

        df = pd.DataFrame(self.results)

        # Sort by IoU descending
        df_sorted = df.sort_values('iou', ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(df_sorted))
        width = 0.35

        bars1 = ax.bar(x - width/2, df_sorted['iou'], width, label='IoU', color='steelblue')
        bars2 = ax.bar(x + width/2, df_sorted['dice'], width, label='Dice', color='coral')

        ax.set_xlabel('Image', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(f'Top {top_n} Images by IoU Score', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df_sorted['image'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])

        plt.tight_layout()

        if save_path is None:
            save_path = EVALUATION_PLOTS_DIR / f"top_{top_n}_images.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Top images comparison saved to {save_path}")
        plt.close()

    def generate_full_report(self):
        """Generate complete evaluation report with all visualizations."""
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE EVALUATION REPORT")
        print("="*70)

        # Save results
        self.save_results()
        self.save_summary()

        # Generate plots
        self.plot_metric_distributions()
        self.plot_confusion_matrix_summary()
        self.plot_per_image_comparison(top_n=10)

        # Print summary
        summary = self.get_summary_statistics()

        if summary:
            print("\n" + "="*70)
            print("EVALUATION SUMMARY STATISTICS")
            print("="*70)
            print(f"\n{'Metric':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
            print("-" * 70)

            for metric in ['iou', 'dice', 'pixel_accuracy', 'precision', 'recall',
                          'specificity', 'f1_score', 'hausdorff_distance']:
                mean_val = summary['mean'][metric]
                std_val = summary['std'][metric]
                min_val = summary['min'][metric]
                max_val = summary['max'][metric]

                print(f"{metric:<20} {mean_val:<12.4f} {std_val:<12.4f} {min_val:<12.4f} {max_val:<12.4f}")

            print("="*70)
            print(f"\nAll results saved to: {EVALUATION_DIR}")
            print(f"  - CSV files: {EVALUATION_CSV_DIR}")
            print(f"  - Plots: {EVALUATION_PLOTS_DIR}")


def run_evaluation():
    """Main evaluation script."""
    print("\n" + "="*70)
    print("MedSAM SEGMENTATION EVALUATION")
    print("="*70)

    # Create evaluator
    evaluator = SegmentationEvaluator()

    # Check if directories exist
    if not evaluator.predictions_dir.exists():
        print(f"\nERROR: Predictions directory not found: {evaluator.predictions_dir}")
        print("Please run inference first to generate predictions.")
        return

    if not evaluator.ground_truth_dir.exists():
        print(f"\nERROR: Ground truth directory not found: {evaluator.ground_truth_dir}")
        print("Please place ground truth masks in this directory.")
        return

    # Run evaluation
    evaluator.evaluate_all()

    # Generate full report
    evaluator.generate_full_report()

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    run_evaluation()
