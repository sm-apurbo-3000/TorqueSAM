import os
from pathlib import Path

# Root directory for dynamic path handling
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = BASE_DIR / "data"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth_masks"  # Manual annotations

# Model checkpoint directories
MEDSAM_CHECKPOINT_DIR = BASE_DIR / "medsam_checkpoints"
YOLOV8_CHECKPOINT_DIR = BASE_DIR / "yolov8_checkpoints"

# Output directories
MEDSAM_SEG_OUTPUT_DIR = BASE_DIR / "medsam_seg_outputs2"
RAW_MASKS_DIR = MEDSAM_SEG_OUTPUT_DIR / "raw_masks"  # Raw segmentation masks
BOUNDARY_IMAGES_DIR = MEDSAM_SEG_OUTPUT_DIR / "boundary_images"  # Visualization
DETECTION_METADATA_DIR = MEDSAM_SEG_OUTPUT_DIR / "detection_metadata"  # Bboxes, etc.

# Evaluation directories
EVALUATION_DIR = BASE_DIR / "evaluation_results"
EVALUATION_PLOTS_DIR = EVALUATION_DIR / "plots"
EVALUATION_CSV_DIR = EVALUATION_DIR / "csv"

# Ensure all folders exist
for folder in [
    DATA_DIR, GROUND_TRUTH_DIR,
    MEDSAM_CHECKPOINT_DIR, YOLOV8_CHECKPOINT_DIR,
    MEDSAM_SEG_OUTPUT_DIR, RAW_MASKS_DIR, BOUNDARY_IMAGES_DIR, DETECTION_METADATA_DIR,
    EVALUATION_DIR, EVALUATION_PLOTS_DIR, EVALUATION_CSV_DIR
]:
    os.makedirs(folder, exist_ok=True)
