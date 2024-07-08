import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import time

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
def preprocess_image(frame_path):
    image = Image.open(frame_path).convert('RGB')
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

def check_for_bird(frame_path):
    input_image = preprocess_image(frame_path)
    output = perform_inference(input_image)
    return post_process_output(output)

# Example usage
if __name__ == "__main__":
    frame_path = "goldfinchbird.jpeg"
    start_time = time.time()  # Start time

    label, score = check_for_bird(frame_path)

    end_time = time.time()  # End time
    inference_time = end_time - start_time  # Calculate the time taken for inference
    print(f"Inference time: {inference_time:.4f} seconds")  # Print the inference time
    print(f"Predicted label: {label}, Score: {score:.4f}")  # Print the result
