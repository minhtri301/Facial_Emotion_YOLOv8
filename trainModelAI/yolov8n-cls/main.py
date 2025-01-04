from ultralytics import YOLO

dataset_path = 'dataset_yolov8n_cls'

model = YOLO("yolov8n-cls.pt")

results = model.train(data=f"{dataset_path}", epochs=25)
model.export()