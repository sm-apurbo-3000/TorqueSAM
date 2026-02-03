import os
from config import DATA_DIR, YOLOV8_CHECKPOINT_DIR, MEDSAM_SEG_OUTPUT_DIR, MEDSAM_CHECKPOINT_DIR
from inference import load_yolo_model, load_medsam_model, set_paths, process_images_in_folder

def run_inference():
    # Load YOLOv8 model
    yolo_weights_path = os.path.join(YOLOV8_CHECKPOINT_DIR, "train", "weights", "best.pt")
    load_yolo_model(yolo_weights_path)
    
    # NEW: Load MedSAM model
    medsam_checkpoint_path = os.path.join(MEDSAM_CHECKPOINT_DIR, "medsam_vit_b.pth")
    load_medsam_model(medsam_checkpoint_path, device='cuda:0')
    
    # Set paths and process images
    set_paths(str(DATA_DIR / "test"), str(MEDSAM_SEG_OUTPUT_DIR))
    process_images_in_folder(str(DATA_DIR / "test")) ##########################MOST IMPORTANT LINE##########################

if __name__ == "__main__":
    run_inference()
    print("MedSAM Segmentation completed.")
