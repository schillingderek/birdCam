import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import time  # Import the time module

# Load indices and labels from the npy file
indices_labels = np.load("indices_labels.npy", allow_pickle=True).item()

# Reverse the dictionary
labels_indices = {v: k for k, v in indices_labels.items()}

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess input image
def preprocess_image(frame_path):
    input_shape = input_details[0]['shape'][1:3]
    image = Image.open(frame_path)
    image = image.resize(input_shape, resample=Image.BOX)
    input_image = np.array(image)
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
    return input_image

# Perform inference
def perform_inference(input_image):
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Post-process output
def post_process_output(output_data):
    # Process the output based on your model's output format
    # Example: Print the detected objects
    print("Detected objects:")
    
    # Assuming output_data is a list of arrays (one array per image)
    for output in output_data:
        # Get the indices of the top 3 predictions for each image
        top_indices = np.argsort(output)[::-1][:3]
        
        # Print label and score for each of the top 3 predictions
        for i, idx in enumerate(top_indices):
            label = labels_indices.get(idx, "Unknown")
            score = output[idx]
            print(f"Top {i+1}: Label: {label}, Score: {score}")

        print()  # Add a newline between predictions for different images

# Example usage
if __name__ == "__main__":
    # Assuming frame is your input image
    frame_path = "goldfinchbird.jpeg"
    start_time = time.time()  # Start time

    input_image = preprocess_image(frame_path)
    output_data = perform_inference(input_image)

    end_time = time.time()  # End time
    inference_time = end_time - start_time  # Calculate the time taken for inference
    print(f"Inference time: {inference_time:.4f} seconds")  # Print the inference time

    post_process_output(output_data)
