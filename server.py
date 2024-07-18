from flask import Flask, request, jsonify
import os
import io
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from ultralytics import YOLO

import requests

from requests_toolbelt.multipart import decoder
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load class names from the .npy file
class_names = np.load("class_names.npy", allow_pickle=True).tolist()

# Load the PyTorch model
model = torch.jit.load("simple_nn.pt")
model.eval()

# Define the image transformation for PyTorch
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Match the input size expected by your model
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Ensure normalization matches training
])

# Preprocess input image
def preprocess_image(image):
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Perform inference
def perform_inference(input_image):
    with torch.no_grad():
        output = model(input_image)
    return output

# Post-process output
def post_process_output(output):
    # Get the index of the top prediction
    _, predicted = torch.max(output, 1)
    predicted_idx = predicted.item()
    
    # Get the top label and score
    top_label = class_names[predicted_idx]
    top_score = torch.softmax(output, dim=1)[0, predicted_idx].item()  # Get the score for the top class

    return top_label, top_score

# Detect birds using YOLOv8
def detect_birds_yolo(image_path):
    detection_model = YOLO("yolov8s.pt")
    im1 = Image.open(image_path)
    results = detection_model.predict(source=im1, save=False)
    return results

# Crop sub-images from bounding boxes
def crop_sub_images(image, results):
    sub_images = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract bounding box coordinates
            sub_image = image.crop((x1, y1, x2, y2))
            sub_images.append(sub_image)
    return sub_images

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json()
    image_url = data.get('image_url')

    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    # Download the image from the provided URL
    response = requests.get(image_url)
    if response.status_code != 200:
        return jsonify({"error": "Unable to download image"}), 400

    image_path = os.path.join("/app", "downloaded_image.jpg")
    with open(image_path, 'wb') as f:
        f.write(response.content)

    # Detect birds
    results = detect_birds_yolo(image_path)
    image = Image.open(image_path)
    cropped_images = crop_sub_images(image, results)

    results = []
    for sub_image in cropped_images:
        input_image = preprocess_image(sub_image)
        output = perform_inference(input_image)
        label, score = post_process_output(output)
        results.append((label, score))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
