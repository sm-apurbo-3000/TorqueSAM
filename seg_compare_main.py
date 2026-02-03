"""
Main script to compare QuickShift and MedSAM segmentation methods.

Usage:
    python compare_main.py <path_to_quickshift_outputs>

Example:
    python compare_main.py quickshift_seg_outputs/raw_masks
"""

import sys
from seg_comparison import run_comparison

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("ERROR: QuickShift output directory not specified")
        print("="*70)
        print("\nUsage:")
        print("  python compare_main.py <quickshift_output_dir>")
        print("\nExample:")
        print("  python compare_main.py quickshift_seg_outputs/raw_masks")
        print("\nThe directory should contain:")
        print("  - .npy files with segmentation masks")
        print("  - OR .png files with segmentation masks")
        print("\n" + "="*70)
        sys.exit(1)

    quickshift_dir = sys.argv[1]
    run_comparison(quickshift_dir)
