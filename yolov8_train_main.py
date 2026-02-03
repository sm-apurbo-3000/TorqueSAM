import os
from config import DATA_DIR, YOLOV8_CHECKPOINT_DIR
from yolov8_training import train_model

def run_training():
    data_yaml = os.path.join(DATA_DIR, "data.yaml")
    model = train_model(
        data_yaml=data_yaml,
        epochs=80,
        batch_size=16,
        imgsz=640,
        save_dir=str(YOLOV8_CHECKPOINT_DIR),
        device=0,
        amp=True
    )
    return model

if __name__ == "__main__":
    run_training()
    print("YOLOv8 Training completed.")
