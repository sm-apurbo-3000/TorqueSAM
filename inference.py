import os
import numpy as np
import json
from PIL import Image
from ultralytics import YOLO
# UNCOMMENT to enable boundary marking:
# from skimage.segmentation import mark_boundaries
import cv2

from postprocessing import mask_image, process_and_resize_image
from preprocessing import medsam_segmentation_pipeline
from medsam_segmentation import MedSAMSegmenter
from config import (MEDSAM_CHECKPOINT_DIR, RAW_MASKS_DIR,
                    BOUNDARY_IMAGES_DIR, DETECTION_METADATA_DIR)

# Global variables
images_folder_path = None
output_folder_path = None
yolo_model = None
medsam_model = None  # NEW: MedSAM model

def load_yolo_model(weights_path):
    global yolo_model
    print("YOLOv8 Model is preparing")
    yolo_model = YOLO(weights_path)
    print("YOLOv8 is ready")

def load_medsam_model(checkpoint_path=None, device='cuda:0'):
    """Load MedSAM model for segmentation."""
    global medsam_model
    
    if checkpoint_path is None:
        checkpoint_path = os.path.join(MEDSAM_CHECKPOINT_DIR, "medsam_vit_b.pth")
    
    print("Loading MedSAM model...")
    medsam_model = MedSAMSegmenter(
        checkpoint_path=checkpoint_path,
        model_type='vit_b',
        device=device
    )
    print("MedSAM model ready")

def set_paths(images_path, output_path):
    global images_folder_path, output_folder_path
    images_folder_path = images_path
    output_folder_path = output_path
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

def process_images_in_folder(folder_path):
    global medsam_model, yolo_model

    if medsam_model is None:
        raise ValueError("MedSAM model not loaded. Call load_medsam_model() first.")

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(root, filename)
                results = yolo_model.predict(source=image_path, device=0, verbose=False)


                for result in results:
                    box = result.boxes.xyxy

                    # Handle case when no objects are detected
                    if len(box) == 0:
                        print(f"No detections for {filename}, skipping...")
                        continue

                    a = mask_image(image_path, box)
                    masked_img_np = np.array(a)

                    # IMPORTANT: MedSAM needs the ORIGINAL image, not binary threshold
                    # The preprocessing (darken, refine) destroys information MedSAM needs
                    # So we pass the original masked image directly to MedSAM
                    seg = medsam_segmentation_pipeline(masked_img_np, box, medsam_model)

                    # UNCOMMENT to enable boundary marking:
                    # boundary_img = mark_boundaries(masked_img_np, seg, color=(1, 0, 0), mode='thick')

                    # Prepare output paths
                    base_name = os.path.splitext(filename)[0]
                    relative_path = os.path.relpath(root, folder_path)

                    # Create subdirectories for all outputs
                    boundary_subdir = os.path.join(BOUNDARY_IMAGES_DIR, relative_path)
                    raw_mask_subdir = os.path.join(RAW_MASKS_DIR, relative_path)
                    metadata_subdir = os.path.join(DETECTION_METADATA_DIR, relative_path)

                    for subdir in [boundary_subdir, raw_mask_subdir, metadata_subdir]:
                        os.makedirs(subdir, exist_ok=True)

                    # ===== SAVE 1: Boundary Visualization (for viewing) =====
                    # UNCOMMENT to enable boundary marking:
                    # boundary_path = os.path.join(boundary_subdir, filename)
                    # boundary_img_pil = Image.fromarray((boundary_img * 255).astype(np.uint8))
                    # rec = process_and_resize_image(boundary_img_pil)
                    # rec.save(boundary_path)

                    # ===== SAVE 2: Raw Segmentation Mask (for evaluation) =====
                    # Save as numpy array (preserves integer labels)
                    raw_mask_npy_path = os.path.join(raw_mask_subdir, f"{base_name}_mask.npy")
                    np.save(raw_mask_npy_path, seg)

                    # Also save as PNG for visual inspection (normalized to 0-255)
                    # Each label gets a different grayscale value
                    if seg.max() > 0:
                        seg_normalized = ((seg / seg.max()) * 255).astype(np.uint8)
                    else:
                        seg_normalized = seg.astype(np.uint8)
                    raw_mask_png_path = os.path.join(raw_mask_subdir, f"{base_name}_mask.png")
                    cv2.imwrite(raw_mask_png_path, seg_normalized)

                    # ===== SAVE 3: Detection Metadata (bboxes, labels, etc.) =====
                    metadata = {
                        'filename': filename,
                        'image_shape': masked_img_np.shape,
                        'num_detections': len(box),
                        'bboxes': box.cpu().numpy().tolist() if hasattr(box, 'cpu') else box.tolist(),
                        'confidence_scores': result.boxes.conf.cpu().numpy().tolist() if hasattr(result.boxes.conf, 'cpu') else [],
                        'class_ids': result.boxes.cls.cpu().numpy().tolist() if hasattr(result.boxes.cls, 'cpu') else [],
                        'segmentation_labels': int(seg.max()),  # Number of unique regions
                    }

                    metadata_path = os.path.join(metadata_subdir, f"{base_name}_metadata.json")
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)

                    print(f"✓ {filename}:")
                    # UNCOMMENT to enable boundary marking:
                    # print(f"  - Boundary: {boundary_path}")
                    print(f"  - Raw mask: {raw_mask_npy_path}")
                    print(f"  - Metadata: {metadata_path}")



###############################################################
# import os
# import numpy as np
# import json
# from PIL import Image
# from ultralytics import YOLO
# from skimage.segmentation import mark_boundaries
# import cv2

# from postprocessing import process_and_resize_image
# from preprocessing import medsam_segmentation_pipeline
# from medsam_segmentation import MedSAMSegmenter
# from config import (
#     MEDSAM_CHECKPOINT_DIR, RAW_MASKS_DIR,
#     BOUNDARY_IMAGES_DIR, DETECTION_METADATA_DIR
# )

# # Global variables
# images_folder_path = None
# output_folder_path = None
# yolo_model = None
# medsam_model = None


# def load_yolo_model(weights_path):
#     global yolo_model
#     print("YOLOv8 Model is preparing")
#     yolo_model = YOLO(weights_path)
#     print("YOLOv8 is ready")


# def load_medsam_model(checkpoint_path=None, device='cuda:0'):
#     global medsam_model
#     if checkpoint_path is None:
#         checkpoint_path = os.path.join(MEDSAM_CHECKPOINT_DIR, "medsam_vit_b.pth")

#     print("Loading MedSAM model...")
#     medsam_model = MedSAMSegmenter(
#         checkpoint_path=checkpoint_path,
#         model_type='vit_b',
#         device=device
#     )
#     print("MedSAM model ready")


# def set_paths(images_path, output_path):
#     global images_folder_path, output_folder_path
#     images_folder_path = images_path
#     output_folder_path = output_path
#     if not os.path.exists(output_folder_path):
#         os.makedirs(output_folder_path, exist_ok=True)


# # ---------------------------
# # Stone segmentation helper
# # ---------------------------
# def segment_stones_intensity(img_gray_uint8, kidney_mask_uint8, thr=220, min_area=5, max_area=3000):
#     """
#     Simple stone segmentation in uint8 images.
#     - Works best when stones appear very bright in your PNG slices.
#     - If you have HU images, we should change this to HU thresholding instead.
#     """
#     if img_gray_uint8.ndim != 2:
#         raise ValueError("img_gray_uint8 must be grayscale (H,W)")

#     roi = (kidney_mask_uint8 > 0).astype(np.uint8)
#     cand = ((img_gray_uint8 >= int(thr)).astype(np.uint8)) * roi

#     cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

#     num, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
#     out = np.zeros_like(cand, dtype=np.uint8)

#     for i in range(1, num):
#         area = stats[i, cv2.CC_STAT_AREA]
#         if min_area <= area <= max_area:
#             out[lab == i] = 255
#     return out


# def process_images_in_folder(folder_path):
#     global medsam_model, yolo_model

#     if medsam_model is None:
#         raise ValueError("MedSAM model not loaded. Call load_medsam_model() first.")
#     if yolo_model is None:
#         raise ValueError("YOLO model not loaded. Call load_yolo_model() first.")

#     for root, dirs, files in os.walk(folder_path):
#         for filename in files:
#             if not (filename.lower().endswith('.jpg') or filename.lower().endswith('.png')):
#                 continue

#             image_path = os.path.join(root, filename)

#             # Read image once (we’ll crop from this)
#             img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
#             if img_bgr is None:
#                 print(f"Could not read {image_path}, skipping...")
#                 continue

#             img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#             H, W = img_rgb.shape[:2]

#             # ✅ Force YOLO to GPU
#             results = yolo_model.predict(source=image_path, device=0, verbose=False)

#             base_name = os.path.splitext(filename)[0]
#             relative_path = os.path.relpath(root, folder_path)

#             boundary_subdir = os.path.join(BOUNDARY_IMAGES_DIR, relative_path)
#             raw_mask_subdir = os.path.join(RAW_MASKS_DIR, relative_path)
#             metadata_subdir = os.path.join(DETECTION_METADATA_DIR, relative_path)

#             for subdir in [boundary_subdir, raw_mask_subdir, metadata_subdir]:
#                 os.makedirs(subdir, exist_ok=True)

#             # Full-size labeled segmentation (int32)
#             full_labeled = np.zeros((H, W), dtype=np.int32)

#             # Full-size kidney mask (uint8 0/255)
#             kidney_mask = np.zeros((H, W), dtype=np.uint8)

#             all_boxes = []
#             all_confs = []
#             all_cls = []

#             for result in results:
#                 boxes_xyxy = result.boxes.xyxy
#                 if boxes_xyxy is None or len(boxes_xyxy) == 0:
#                     continue

#                 confs = result.boxes.conf
#                 clss = result.boxes.cls

#                 # Iterate boxes one by one
#                 for i, b in enumerate(boxes_xyxy):
#                     if hasattr(b, "cpu"):
#                         b = b.cpu().numpy()
#                     x1, y1, x2, y2 = map(int, b.tolist())

#                     # clip
#                     x1 = max(0, min(W - 1, x1))
#                     x2 = max(0, min(W, x2))
#                     y1 = max(0, min(H - 1, y1))
#                     y2 = max(0, min(H, y2))

#                     if x2 <= x1 or y2 <= y1:
#                         continue

#                     all_boxes.append([x1, y1, x2, y2])
#                     if confs is not None and len(confs) > i:
#                         all_confs.append(float(confs[i].cpu().numpy()) if hasattr(confs[i], "cpu") else float(confs[i]))
#                     if clss is not None and len(clss) > i:
#                         all_cls.append(float(clss[i].cpu().numpy()) if hasattr(clss[i], "cpu") else float(clss[i]))

#                     # ✅ CROP ROI instead of black-masking
#                     roi_rgb = img_rgb[y1:y2, x1:x2]

#                     # MedSAM bbox prompt for the crop is the entire crop
#                     roi_box = [[0, 0, roi_rgb.shape[1], roi_rgb.shape[0]]]

#                     roi_labeled = medsam_segmentation_pipeline(roi_rgb, roi_box, medsam_model)

#                     # Paste back into full mask
#                     # Shift labels so multiple boxes remain unique
#                     label_offset = int(full_labeled.max())
#                     roi_labeled_shifted = roi_labeled.copy()
#                     roi_labeled_shifted[roi_labeled_shifted > 0] += label_offset

#                     full_labeled[y1:y2, x1:x2] = np.maximum(full_labeled[y1:y2, x1:x2], roi_labeled_shifted)

#                     # also update kidney mask
#                     kidney_mask[y1:y2, x1:x2] = np.maximum(
#                         kidney_mask[y1:y2, x1:x2],
#                         (roi_labeled > 0).astype(np.uint8) * 255
#                     )

#             if len(all_boxes) == 0:
#                 print(f"No detections for {filename}, skipping...")
#                 continue

#             # Boundary visualization on original image
#             boundary_img = mark_boundaries(img_rgb, (full_labeled > 0).astype(np.uint8), color=(1, 0, 0), mode='thick')

#             # Save boundary (resized like your original code)
#             boundary_path = os.path.join(boundary_subdir, filename)
#             boundary_img_pil = Image.fromarray((boundary_img * 255).astype(np.uint8))
#             rec = process_and_resize_image(boundary_img_pil)
#             rec.save(boundary_path)

#             # Save raw labeled mask .npy
#             raw_mask_npy_path = os.path.join(raw_mask_subdir, f"{base_name}_mask.npy")
#             np.save(raw_mask_npy_path, full_labeled)

#             # Save raw labeled mask .png (normalized)
#             if full_labeled.max() > 0:
#                 seg_normalized = ((full_labeled / full_labeled.max()) * 255).astype(np.uint8)
#             else:
#                 seg_normalized = full_labeled.astype(np.uint8)
#             raw_mask_png_path = os.path.join(raw_mask_subdir, f"{base_name}_mask.png")
#             cv2.imwrite(raw_mask_png_path, seg_normalized)

#             # ✅ Save kidney mask png
#             kidney_mask_path = os.path.join(raw_mask_subdir, f"{base_name}_kidney_mask.png")
#             cv2.imwrite(kidney_mask_path, kidney_mask)

#             # ✅ Stone segmentation (inside kidney)
#             gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#             stone_mask = segment_stones_intensity(gray, kidney_mask, thr=220, min_area=5, max_area=3000)
#             stone_mask_path = os.path.join(raw_mask_subdir, f"{base_name}_stone_mask.png")
#             cv2.imwrite(stone_mask_path, stone_mask)

#             # Save metadata
#             metadata = {
#                 "filename": filename,
#                 "image_shape": [H, W, 3],
#                 "num_detections": len(all_boxes),
#                 "bboxes": all_boxes,
#                 "confidence_scores": all_confs,
#                 "class_ids": all_cls,
#                 "segmentation_labels": int(full_labeled.max()),
#                 "outputs": {
#                     "boundary": boundary_path,
#                     "raw_mask_npy": raw_mask_npy_path,
#                     "raw_mask_png": raw_mask_png_path,
#                     "kidney_mask_png": kidney_mask_path,
#                     "stone_mask_png": stone_mask_path,
#                 }
#             }

#             metadata_path = os.path.join(metadata_subdir, f"{base_name}_metadata.json")
#             with open(metadata_path, "w") as f:
#                 json.dump(metadata, f, indent=2)

#             print(f"✓ {filename}:")
#             print(f"  - Boundary: {boundary_path}")
#             print(f"  - Raw mask: {raw_mask_npy_path}")
#             print(f"  - Kidney mask: {kidney_mask_path}")
#             print(f"  - Stone mask: {stone_mask_path}")
#             print(f"  - Metadata: {metadata_path}")
