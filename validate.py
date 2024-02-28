import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
model = YOLO('runs/detect/train4/weights/best.pt')

result = model.predict("2.png",show=True, conf=0.5)
print(result)
cv2.waitKey(0)