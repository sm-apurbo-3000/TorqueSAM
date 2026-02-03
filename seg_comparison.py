"""
Comparison module for evaluating QuickShift vs MedSAM segmentation methods.
Compares both methods against ground truth annotations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

from seg_evaluation import SegmentationEvaluator
from config import (RAW_MASKS_DIR, GROUND_TRUTH_DIR,
                    EVALUATION_DIR, EVALUATION_PLOTS_DIR, EVALUATION_CSV_DIR)


class MethodComparator:
    """Compare QuickShift and MedSAM segmentation methods."""

    def __init__(self, quickshift_dir, medsam_dir, ground_truth_dir=None):
        """
        Args:
            quickshift_dir: Path to QuickShift segmentation outputs
            medsam_dir: Path to MedSAM segmentation outputs (default: RAW_MASKS_DIR)
            ground_truth_dir: Path to ground truth masks (default: GROUND_TRUTH_DIR)
        """
        self.quickshift_dir = Path(quickshift_dir)
        self.medsam_dir = Path(medsam_dir) if medsam_dir else Path(RAW_MASKS_DIR)
        self.gt_dir = Path(ground_truth_dir) if ground_truth_dir else Path(GROUND_TRUTH_DIR)

        self.quickshift_results = None
        self.medsam_results = None

        print(f"QuickShift directory: {self.quickshift_dir}")
        print(f"MedSAM directory: {self.medsam_dir}")
        print(f"Ground truth directory: {self.gt_dir}")

    def evaluate_both_methods(self):
        """Run evaluation for both QuickShift and MedSAM."""
        print("\n" + "="*70)
        print("EVALUATING QUICKSHIFT SEGMENTATION")
        print("="*70)

        # Evaluate QuickShift
        qs_evaluator = SegmentationEvaluator(self.quickshift_dir, self.gt_dir)
        qs_evaluator.evaluate_all()
        self.quickshift_results = qs_evaluator.results

        print("\n" + "="*70)
        print("EVALUATING MEDSAM SEGMENTATION")
        print("="*70)

        # Evaluate MedSAM
        medsam_evaluator = SegmentationEvaluator(self.medsam_dir, self.gt_dir)
        medsam_evaluator.evaluate_all()
        self.medsam_results = medsam_evaluator.results

        return qs_evaluator, medsam_evaluator

    def get_comparison_summary(self):
        """Get statistical comparison summary."""
        if not self.quickshift_results or not self.medsam_results:
            print("Run evaluate_both_methods() first")
            return None

        qs_df = pd.DataFrame(self.quickshift_results)
        medsam_df = pd.DataFrame(self.medsam_results)

        # Compute statistics for each method
        metrics = ['iou', 'dice', 'pixel_accuracy', 'precision', 'recall',
                  'specificity', 'f1_score', 'hausdorff_distance']

        comparison = {}
        for metric in metrics:
            qs_values = qs_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
            medsam_values = medsam_df[metric].replace([np.inf, -np.inf], np.nan).dropna()

            qs_mean = qs_values.mean()
            medsam_mean = medsam_values.mean()

            # Calculate improvement
            if metric == 'hausdorff_distance':
                # Lower is better for Hausdorff
                improvement = ((qs_mean - medsam_mean) / qs_mean) * 100
            else:
                # Higher is better for other metrics
                improvement = ((medsam_mean - qs_mean) / qs_mean) * 100

            # Perform statistical significance test (paired t-test)
            # Match images by name
            common_images = set(qs_df['image']) & set(medsam_df['image'])

            if len(common_images) > 1:
                qs_matched = qs_df[qs_df['image'].isin(common_images)].sort_values('image')[metric]
                medsam_matched = medsam_df[medsam_df['image'].isin(common_images)].sort_values('image')[metric]

                # Clean data
                qs_matched = qs_matched.replace([np.inf, -np.inf], np.nan).dropna()
                medsam_matched = medsam_matched.replace([np.inf, -np.inf], np.nan).dropna()

                if len(qs_matched) > 1 and len(medsam_matched) > 1 and len(qs_matched) == len(medsam_matched):
                    t_stat, p_value = stats.ttest_rel(qs_matched, medsam_matched)
                else:
                    t_stat, p_value = np.nan, np.nan
            else:
                t_stat, p_value = np.nan, np.nan

            comparison[metric] = {
                'quickshift_mean': qs_mean,
                'quickshift_std': qs_values.std(),
                'medsam_mean': medsam_mean,
                'medsam_std': medsam_values.std(),
                'improvement_%': improvement,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05 if not np.isnan(p_value) else False
            }

        return comparison

    def print_comparison_table(self):
        """Print formatted comparison table."""
        comparison = self.get_comparison_summary()

        if comparison is None:
            return

        print("\n" + "="*90)
        print("QUICKSHIFT vs MEDSAM COMPARISON")
        print("="*90)
        print(f"\n{'Metric':<20} {'QuickShift':<15} {'MedSAM':<15} {'Improvement':<15} {'p-value':<10} {'Sig.':<5}")
        print("-" * 90)

        for metric, stats_dict in comparison.items():
            qs_mean = stats_dict['quickshift_mean']
            qs_std = stats_dict['quickshift_std']
            medsam_mean = stats_dict['medsam_mean']
            medsam_std = stats_dict['medsam_std']
            improvement = stats_dict['improvement_%']
            p_value = stats_dict['p_value']
            significant = stats_dict['significant']

            sig_marker = "***" if significant else ""

            print(f"{metric:<20} {qs_mean:.4f}±{qs_std:.4f}  {medsam_mean:.4f}±{medsam_std:.4f}  "
                  f"{improvement:+.2f}%         {p_value:.4f}     {sig_marker}")

        print("-" * 90)
        print("*** = statistically significant (p < 0.05)")
        print("="*90)

    def save_comparison_csv(self, output_path=None):
        """Save comparison results to CSV."""
        comparison = self.get_comparison_summary()

        if comparison is None:
            return

        if output_path is None:
            output_path = EVALUATION_CSV_DIR / "method_comparison.csv"

        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison).T
        comparison_df.index.name = 'metric'
        comparison_df.to_csv(output_path)

        print(f"\n✓ Comparison results saved to {output_path}")
        return output_path

    def plot_side_by_side_comparison(self, save_path=None):
        """Plot side-by-side comparison of all metrics."""
        comparison = self.get_comparison_summary()

        if comparison is None:
            return

        metrics = ['iou', 'dice', 'pixel_accuracy', 'precision', 'recall', 'specificity', 'f1_score']

        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(metrics))
        width = 0.35

        qs_means = [comparison[m]['quickshift_mean'] for m in metrics]
        qs_stds = [comparison[m]['quickshift_std'] for m in metrics]
        medsam_means = [comparison[m]['medsam_mean'] for m in metrics]
        medsam_stds = [comparison[m]['medsam_std'] for m in metrics]

        bars1 = ax.bar(x - width/2, qs_means, width, yerr=qs_stds,
                      label='QuickShift', color='steelblue', alpha=0.8, capsize=5)
        bars2 = ax.bar(x + width/2, medsam_means, width, yerr=medsam_stds,
                      label='MedSAM', color='coral', alpha=0.8, capsize=5)

        ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('QuickShift vs MedSAM: Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])

        # Add significance markers
        for idx, metric in enumerate(metrics):
            if comparison[metric]['significant']:
                y_pos = max(qs_means[idx], medsam_means[idx]) + 0.05
                ax.text(idx, y_pos, '***', ha='center', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path is None:
            save_path = EVALUATION_PLOTS_DIR / "method_comparison_bars.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison bar chart saved to {save_path}")
        plt.close()

    def plot_improvement_heatmap(self, save_path=None):
        """Plot heatmap showing improvement percentages."""
        comparison = self.get_comparison_summary()

        if comparison is None:
            return

        metrics = ['iou', 'dice', 'pixel_accuracy', 'precision', 'recall',
                  'specificity', 'f1_score', 'hausdorff_distance']

        improvements = [comparison[m]['improvement_%'] for m in metrics]
        significance = [comparison[m]['significant'] for m in metrics]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create data for heatmap
        data = np.array(improvements).reshape(1, -1)

        # Plot heatmap
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=30)

        ax.set_xticks(np.arange(len(metrics)))
        ax.set_xticklabels([m.replace('_', '\n').title() for m in metrics], rotation=45, ha='right')
        ax.set_yticks([0])
        ax.set_yticklabels(['Improvement\n(%)'])

        # Add text annotations
        for i, (imp, sig) in enumerate(zip(improvements, significance)):
            text_color = 'white' if abs(imp) > 15 else 'black'
            sig_marker = '***' if sig else ''
            ax.text(i, 0, f'{imp:+.1f}%\n{sig_marker}', ha='center', va='center',
                   color=text_color, fontweight='bold', fontsize=10)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
        cbar.set_label('Improvement (%)', fontsize=11, fontweight='bold')

        ax.set_title('MedSAM Improvement over QuickShift', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path is None:
            save_path = EVALUATION_PLOTS_DIR / "improvement_heatmap.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Improvement heatmap saved to {save_path}")
        plt.close()

    def plot_metric_distributions_comparison(self, save_path=None):
        """Plot overlapping distributions for QuickShift and MedSAM."""
        if not self.quickshift_results or not self.medsam_results:
            print("Run evaluate_both_methods() first")
            return

        qs_df = pd.DataFrame(self.quickshift_results)
        medsam_df = pd.DataFrame(self.medsam_results)

        metrics = ['iou', 'dice', 'f1_score', 'hausdorff_distance']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            qs_data = qs_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
            medsam_data = medsam_df[metric].replace([np.inf, -np.inf], np.nan).dropna()

            # Plot histograms
            ax.hist(qs_data, bins=20, alpha=0.6, label='QuickShift', color='steelblue', edgecolor='black')
            ax.hist(medsam_data, bins=20, alpha=0.6, label='MedSAM', color='coral', edgecolor='black')

            # Add mean lines
            ax.axvline(qs_data.mean(), color='blue', linestyle='--', linewidth=2,
                      label=f'QS Mean: {qs_data.mean():.3f}')
            ax.axvline(medsam_data.mean(), color='red', linestyle='--', linewidth=2,
                      label=f'MedSAM Mean: {medsam_data.mean():.3f}')

            ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle('Distribution Comparison: QuickShift vs MedSAM',
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()

        if save_path is None:
            save_path = EVALUATION_PLOTS_DIR / "distribution_comparison.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Distribution comparison saved to {save_path}")
        plt.close()

    def plot_per_image_improvement(self, save_path=None, metric='iou', top_n=15):
        """Plot per-image improvement from QuickShift to MedSAM."""
        if not self.quickshift_results or not self.medsam_results:
            print("Run evaluate_both_methods() first")
            return

        qs_df = pd.DataFrame(self.quickshift_results)
        medsam_df = pd.DataFrame(self.medsam_results)

        # Merge on image name
        merged = pd.merge(qs_df[['image', metric]], medsam_df[['image', metric]],
                         on='image', suffixes=('_qs', '_medsam'))

        # Calculate improvement
        merged['improvement'] = merged[f'{metric}_medsam'] - merged[f'{metric}_qs']
        merged = merged.sort_values('improvement', ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(merged))
        width = 0.35

        bars1 = ax.bar(x - width/2, merged[f'{metric}_qs'], width,
                      label='QuickShift', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, merged[f'{metric}_medsam'], width,
                      label='MedSAM', color='coral', alpha=0.8)

        # Add improvement arrows
        for i, row in enumerate(merged.itertuples()):
            improvement = row.improvement
            if improvement > 0:
                ax.annotate('', xy=(i + width/2, getattr(row, f'{metric}_medsam')),
                           xytext=(i - width/2, getattr(row, f'{metric}_qs')),
                           arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

        ax.set_xlabel('Image', fontweight='bold')
        ax.set_ylabel(f'{metric.upper()} Score', fontweight='bold')
        ax.set_title(f'Top {top_n} Images with Highest {metric.upper()} Improvement',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(merged['image'], rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path is None:
            save_path = EVALUATION_PLOTS_DIR / f"per_image_improvement_{metric}.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Per-image improvement plot saved to {save_path}")
        plt.close()

    def generate_full_comparison_report(self):
        """Generate complete comparison report with all visualizations."""
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE COMPARISON REPORT")
        print("="*70)

        # Evaluate both methods
        self.evaluate_both_methods()

        # Print comparison table
        self.print_comparison_table()

        # Save CSV
        self.save_comparison_csv()

        # Generate all plots
        self.plot_side_by_side_comparison()
        self.plot_improvement_heatmap()
        self.plot_metric_distributions_comparison()
        self.plot_per_image_improvement(metric='iou')
        self.plot_per_image_improvement(metric='dice')

        print("\n" + "="*70)
        print("COMPARISON REPORT COMPLETE")
        print("="*70)
        print(f"\nAll comparison results saved to: {EVALUATION_DIR}")
        print(f"  - CSV: {EVALUATION_CSV_DIR / 'method_comparison.csv'}")
        print(f"  - Plots: {EVALUATION_PLOTS_DIR}")


def run_comparison(quickshift_dir):
    """
    Main comparison script.

    Args:
        quickshift_dir: Path to QuickShift segmentation outputs
    """
    print("\n" + "="*70)
    print("QUICKSHIFT vs MEDSAM COMPARISON")
    print("="*70)

    # Create comparator
    comparator = MethodComparator(
        quickshift_dir=quickshift_dir,
        medsam_dir=RAW_MASKS_DIR,
        ground_truth_dir=GROUND_TRUTH_DIR
    )

    # Generate full comparison report
    comparator.generate_full_comparison_report()

    print("\n✓ Comparison complete!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python comparison.py <quickshift_output_dir>")
        print("Example: python comparison.py quickshift_seg_outputs/raw_masks")
        sys.exit(1)

    quickshift_dir = sys.argv[1]
    run_comparison(quickshift_dir)
