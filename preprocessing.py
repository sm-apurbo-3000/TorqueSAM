import numpy as np
import cv2
from PIL import Image
from skimage import filters, exposure
from skimage.measure import label, find_contours
from scipy.ndimage import mean

def darken_image2(image, darken_factor=0.8):
    """Darkens a color or grayscale image using a power-law transformation curve, 
    controlled by a darken_factor parameter."""
    image_np = np.array(image)
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    max_intensity = 255
    transform_curve = np.array(
        [(i / max_intensity) ** (1 / darken_factor) * max_intensity for i in range(256)],
        dtype=np.uint8
    )
    darkened_image = cv2.LUT(image_np, transform_curve)
    return darkened_image

def ignore_black_regions(image, threshold=0.05):
    """Generates a mask to ignore black or near-black regions in the image."""
    mask = image > threshold
    return mask
#New
# import numpy as np
# import cv2

# def ignore_black_regions(image, threshold=0.05):
#     """
#     Returns a 2D mask (H,W) to ignore black/near-black pixels.
#     Works for both grayscale and RGB, and for uint8 or float images.
#     """
#     if image is None or image.size == 0:
#         return None

#     # Convert RGB -> grayscale if needed
#     if image.ndim == 3:
#         gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     else:
#         gray = image

#     # Choose threshold depending on dtype/range
#     if gray.dtype == np.uint8:
#         thr = 5  # near-black for uint8 images
#         mask = (gray > thr)
#     else:
#         # float images (0..1 usually)
#         thr = float(threshold)
#         mask = (gray > thr)

#     return mask.astype(np.uint8)


def refine_segmentation(image):
    """Converts an image to binary using thresholding, then applies 
    morphological closing to fill small holes and connect nearby regions."""
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return closing

# NEW: MedSAM-based segmentation function
def medsam_segmentation_pipeline(img, bboxes, medsam_model):
    """
    MedSAM-based segmentation pipeline that replaces QuickShift.

    This function takes the image and bounding boxes from YOLOv8,
    uses MedSAM to segment the regions, and returns labeled segments.

    Args:
        img: Input image (numpy array, grayscale or RGB)
        bboxes: Bounding boxes from YOLOv8 (tensor or list)
        medsam_model: Initialized MedSAMSegmenter instance

    Returns:
        labeled_segments: Segmentation mask with integer labels
    """
    from medsam_segmentation import medsam_segmentation

    # Validate inputs
    if img is None or img.size == 0:
        raise ValueError("Empty image provided to medsam_segmentation_pipeline")

    # Apply mask to ignore black regions
    mask = ignore_black_regions(img)

    # Convert bboxes to list format if tensor
    bbox_list = []
    for bbox in bboxes:
        if hasattr(bbox, 'cpu'):
            bbox = bbox.cpu().numpy()

        # Validate bbox dimensions
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Skip invalid bboxes
        if x2 <= x1 or y2 <= y1:
            print(f"Warning: Invalid bbox dimensions [{x1}, {y1}, {x2}, {y2}], skipping...")
            continue

        # Clip bbox to image boundaries
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        bbox_list.append([x1, y1, x2, y2])

    # Handle case with no valid bboxes
    if len(bbox_list) == 0:
        print("Warning: No valid bboxes for MedSAM, returning empty segmentation")
        return np.zeros(img.shape[:2], dtype=np.int32)

    # Perform MedSAM segmentation
    labeled_segments = medsam_segmentation(img, bbox_list, medsam_model)

    # Apply the black region mask (ensure proper type casting)
    labeled_segments = (labeled_segments.astype(np.int32) * mask.astype(np.int32))

    return labeled_segments



#New
import numpy as np
import cv2

def segment_stones_intensity(img_gray_uint8, kidney_mask_uint8, thr=220, min_area=5, max_area=3000):
    # img_gray_uint8: 0-255 grayscale
    # kidney_mask_uint8: 0/255 mask of kidney ROI

    roi = (kidney_mask_uint8 > 0).astype(np.uint8)
    cand = ((img_gray_uint8 >= thr).astype(np.uint8)) * roi

    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    num, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    out = np.zeros_like(cand)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            out[lab == i] = 255
    return out
