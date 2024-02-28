from ultralytics import YOLO
model = YOLO('yolov8s.pt')
results = model.train(data='data_clss.yaml', epochs=10, imgsz=140)