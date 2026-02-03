# ==========================================
# IMPORTS
# ==========================================
import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt


# ==========================================
# UTILS: LOAD IMAGE + MASK
# ==========================================
def load_ct(path):
    """
    Read CT slice (grayscale). Converts to float32 automatically.
    If image is RGB, converts to grayscale first.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read: {path}")
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img.astype(np.float32)


def load_mask(path):
    """
    Load MedSAM segmentation mask (binary).
    Mask must be uint8 with values {0,1}.
    """
    if path.endswith(".npy"):
        mask = np.load(path)
    else:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read: {path}")

    mask = (mask > 0).astype(np.uint8)
    return mask


# ==========================================
# FEATURE: MEAN HU
# ==========================================
def compute_mean_hu(image, mask):
    """
    Compute mean HU inside mask.
    """
    region = image[mask == 1]
    if region.size == 0:
        return 0.0
    return float(np.mean(region))


# ==========================================
# FEATURE: GLCM CONTRAST
# ==========================================
def compute_glcm_contrast(image, mask, levels=32):
    """
    Compute GLCM contrast inside the masked region.
    Steps:
      - crop region using mask bounding box (for speed)
      - normalize to [0, levels)
      - compute GLCM at distance 1, angle 0
    """
    # Get bounding box for speed
    ys, xs = np.where(mask == 1)
    if len(xs) == 0:
        return 0.0

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    crop = image[y1:y2+1, x1:x2+1]
    crop_mask = mask[y1:y2+1, x1:x2+1]

    # Apply mask
    region = crop[crop_mask == 1]

    if region.size < 10:
        return 0.0

    # Normalize region intensities to discrete levels
    region_norm = region - region.min()
    if region_norm.max() == 0:
        return 0.0

    region_norm = (region_norm / region_norm.max() * (levels - 1)).astype(np.uint8)

    # Reconstruct masked crop with background zeros
    norm_crop = np.zeros_like(crop, dtype=np.uint8)
    norm_crop[crop_mask == 1] = region_norm

    # Compute GLCM
    glcm = graycomatrix(norm_crop,
                        distances=[1],
                        angles=[0],
                        levels=levels,
                        symmetric=True,
                        normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]

    return float(contrast)


# ==========================================
# PROCESS ONE SAMPLE
# ==========================================
def extract_features(image_path, mask_path):
    """
    Extract Mean HU & GLCM Contrast from a single image-mask pair.
    Returns (mean_hu, glcm_contrast)
    """
    image = load_ct(image_path)
    mask = load_mask(mask_path)

    mean_hu = compute_mean_hu(image, mask)
    glcm_contrast = compute_glcm_contrast(image, mask)

    return mean_hu, glcm_contrast


# ==========================================
# PROCESS DATASET
# ==========================================
def process_dataset(image_dir, mask_dir, output_csv="features.csv"):
    """
    Iterate through dataset and extract features.
    Handles filename mismatches: tries exact match first, then tries common mask naming patterns.
    """
    rows = []

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
            continue

        image_path = os.path.join(image_dir, filename)
        
        # Try exact match first
        mask_path = os.path.join(mask_dir, filename)
        
        # If exact match fails, try to find mask with different naming pattern
        if not os.path.exists(mask_path):
            # Extract base name without extension
            base_name = os.path.splitext(filename)[0]
            # Try to find a mask file with matching base name (handle .npy, .png, etc.)
            for mask_file in os.listdir(mask_dir):
                # Check if this mask file matches the image base name
                mask_base = os.path.splitext(mask_file)[0]
                # Remove common suffixes to compare base names
                mask_base_clean = mask_base.replace("_stone_mask", "").replace("_mask", "")
                if base_name == mask_base_clean or base_name in mask_file:
                    mask_path = os.path.join(mask_dir, mask_file)
                    break
            else:
                # No mask found
                mask_path = None
        
        if mask_path is None or not os.path.exists(mask_path):
            print(f"[Warning] Mask missing for {filename}, skipping.")
            continue

        mean_hu, glcm_contrast = extract_features(image_path, mask_path)

        rows.append({
            "filename": filename,
            "mean_hu": mean_hu,
            "glcm_contrast": glcm_contrast
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV: {output_csv}")
    return df


# ==========================================
# PLOT 2D SCATTER
# ==========================================
def plot_2d(df):
    """
    2D scatter: Mean HU vs GLCM Contrast.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(df["mean_hu"], df["glcm_contrast"], s=40)

    plt.xlabel("Mean HU")
    plt.ylabel("GLCM Contrast")
    plt.title("2D Feature Scatter (Mean HU vs GLCM Contrast)")
    plt.grid(True)
    plt.show()


# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    IMAGE_DIR = "images/"
    MASK_DIR  = "masks/"

    df = process_dataset(IMAGE_DIR, MASK_DIR, output_csv="kidney_features.csv")
    plot_2d(df)
