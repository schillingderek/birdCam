from flask import Flask, request, jsonify
import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from ultralytics import YOLO
from flask_cors import CORS
from io import BytesIO

load_dotenv()

app = Flask(__name__)
CORS(app)

app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')

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
    _, predicted = torch.max(output, 1)
    predicted_idx = predicted.item()
    
    top_label = class_names[predicted_idx]
    top_score = torch.softmax(output, dim=1)[0, predicted_idx].item()  # Get the score for the top class

    return top_label, top_score

# Detect birds using YOLOv8
def detect_birds_yolo(image):
    detection_model = YOLO("yolov8s.pt")
    results = detection_model.predict(source=image, save=False)
    
    bird_boxes = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls == 14:  # Number of the bird class
                bird_boxes.append(box)
    
    return bird_boxes

# Crop sub-images from bounding boxes
def crop_sub_images(image, boxes):
    sub_images = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract bounding box coordinates
        sub_image = image.crop((x1, y1, x2, y2))
        sub_images.append(sub_image)
    return sub_images

@app.route('/process_image', methods=['POST'])
def process_image():
    # Retrieve the image file from the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image = Image.open(BytesIO(image_file.read()))

    # Detect birds
    bird_boxes = detect_birds_yolo(image)
    if not bird_boxes:
        return jsonify([])  # Return an empty list if no birds are detected

    cropped_images = crop_sub_images(image, bird_boxes)

    results = []
    for sub_image in cropped_images:
        input_image = preprocess_image(sub_image)
        output = perform_inference(input_image)
        label, score = post_process_output(output)
        results.append((label, score))
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
