from ultralytics import YOLO
from PIL import Image

detection_model = YOLO("yolov8s.pt")
im1 = Image.open("birds.jpg")
results = detection_model.predict(source=im1, save=True)

