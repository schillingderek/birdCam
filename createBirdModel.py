import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader

# Function to verify images
def verify_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            try:
                img_path = os.path.join(root, file)
                img = Image.open(img_path)
                img.verify()  # Verify that the image is not corrupted
                img.close()
            except (IOError, SyntaxError) as e:
                print(f"Bad file: {img_path}")

# Verify images in train, test, and validation directories
print("Verifying images...")
test_path = os.path.join(os.getcwd(), "test_keep")
train_path = os.path.join(os.getcwd(), "train_keep")
valid_path = os.path.join(os.getcwd(), "valid_keep")
verify_images(test_path)
verify_images(train_path)
verify_images(valid_path)

# Load data
test_data = DataLoader.from_folder(test_path)
train_data = DataLoader.from_folder(train_path)
valid_data = DataLoader.from_folder(valid_path)

# Create and train the model
model = image_classifier.create(train_data, validation_data=valid_data)

# Get indices and labels
indices_labels = {label: idx for idx, label in enumerate(train_data.index_to_label)}

# Save indices and labels
np.save("indices_labels.npy", indices_labels)

# Summary and evaluation
model.summary()
loss, accuracy = model.evaluate(valid_data)
print(f'Validation accuracy: {accuracy:.4f}')

# # Plot images and predicted labels
# plt.figure(figsize=(20, 20))
# predicts = model.predict_top_k(valid_data)
# for i, (image, label) in enumerate(valid_data.gen_dataset().unbatch().take(100)):
#     ax = plt.subplot(10, 10, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(image.numpy(), cmap=plt.cm.gray)

#     predict_label = predicts[i][0][0]
#     predict_index = int(predict_label)  # Convert numpy.str_ to int
#     color = 'black' if predict_label == label.numpy() else 'red'
#     ax.xaxis.label.set_color(color)
#     predicted_label = indices_labels.get(predict_index, "Unknown")
#     plt.xlabel('Predicted: %s' % predicted_label)
# plt.show()

# Export model
model.export(export_dir='.')
