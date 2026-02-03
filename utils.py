import os
import cv2
import numpy as np
from pathlib import Path

def get_full_path(base_dir, filename):
    """Return full path dynamically."""
    return Path(base_dir) / filename

def apply_lut(image, lut):
    """Apply lookup table."""
    return cv2.LUT(image, lut)

def morphological_close(image, kernel_size=5):
    """Morphological closing operation."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
