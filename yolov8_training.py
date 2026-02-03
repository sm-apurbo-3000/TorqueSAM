from ultralytics import YOLO
import torch

def train_model(data_yaml, epochs=80, batch_size=16, imgsz=640, save_dir="yolov8_checkpoints", device=0, amp=True):
    # Disable cuDNN to avoid potential issues
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


    # Initialize YOLOv8 model
    model = YOLO("yolov8m.pt")

    # Train the model
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        amp=amp,
        workers=0,
        project=save_dir
    )

    return model
