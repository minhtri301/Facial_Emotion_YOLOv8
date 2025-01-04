import shutil

model = YOLO("yolov8s-cls.pt")

results = model.train(data="/content/drive/MyDrive/FER/data_yolov8_cls", epochs=50, augment=True)
model.export()

shutil.copytree("/content/runs", "/content/drive/MyDrive/FER/runs")