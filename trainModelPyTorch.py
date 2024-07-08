import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
import random

# Function to verify images and remove .DS_Store files
def clean_and_verify_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            img_path = os.path.join(root, file)
            if file == ".DS_Store":
                os.remove(img_path)
                print(f"Removed: {img_path}")
            else:
                try:
                    img = Image.open(img_path)
                    img.verify()  # Verify that the image is not corrupted
                    img.close()
                except (IOError, SyntaxError) as e:
                    print(f"Bad file: {img_path}")

# Define transformations for the training, validation, and testing data
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Function to split data into train, validation, and test sets
def split_data(dataset, train_ratio=0.7, valid_ratio=0.15):
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    valid_size = int(valid_ratio * total_size)
    test_size = total_size - train_size - valid_size

    return random_split(dataset, [train_size, valid_size, test_size])

def load_data(data_dir):
    data_transforms = get_transforms()

    # Load the full dataset
    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

    # Split the dataset
    train_dataset, valid_dataset, test_dataset = split_data(full_dataset)

    # Create DataLoaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
        'valid': DataLoader(valid_dataset, batch_size=32, shuffle=True, num_workers=4),
        'test': DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4),
    }

    dataset_sizes = {
        'train': len(train_dataset),
        'valid': len(valid_dataset),
        'test': len(test_dataset)
    }

    class_names = full_dataset.classes

    print("Exporting class names")
    np.save("class_names.npy", class_names)

    return dataloaders, dataset_sizes, class_names

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=10):
    import copy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print(f'Best val Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def main():
    # Paths
    data_dir = os.path.join(os.getcwd(), "training_images")

    # Clean and verify images
    print("Verifying and cleaning images...")
    clean_and_verify_images(data_dir)

    # Load data
    print("Loading images...")
    dataloaders, dataset_sizes, class_names = load_data(data_dir)

    # Define the model
    model = models.resnet18(weights='IMAGENET1K_V1')  # Update here
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    # Define the criterion, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Device configuration
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Train and evaluate
    print("Training Model...")
    model = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=10)

    # Save the trained model
    print("Saving Model...")
    torch.save(model.state_dict(), 'simple_nn.pth')

    # Convert to TorchScript
    print("Converting to TorchScript...")
    model.eval()
    scripted_model = torch.jit.script(model)
    scripted_model.save("simple_nn.pt")

    print("Model ready for mobile deployment!")

if __name__ == '__main__':
    main()
