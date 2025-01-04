from ultralytics import YOLO
import os

dataset_path = 'dataset_320'

model = YOLO('yolov8n.yaml')
model.train(data=f"{dataset_path}/custom.yaml",  epochs=40, imgsz=320, batch=16, device=0, augment=True)
