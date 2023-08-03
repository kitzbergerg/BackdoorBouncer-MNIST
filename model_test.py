import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt

from model import SimpleNet


def visualize_data(data_loader, num_images=2):
    # Get a batch of images and labels
    images, labels = next(iter(data_loader))

    # Loop through and display the images
    for i in range(num_images):
        image = images[i].squeeze().numpy()  # Remove channel dimension and convert to numpy array
        label = labels[i].item()
        plt.imshow(image, cmap="gray")
        plt.title(f"Label: {label}")
        plt.show()


# Load the trained model
model = SimpleNet()  # Replace with your actual model class
model.load_state_dict(torch.load("model/trained_model.pth"))
model.eval()  # Set the model to evaluation mode

# Load the datasets
test_data_original = torch.load("data/MNIST/modified/test_data_original.pth")
test_data_modified = torch.load("data/MNIST/modified/test_data_modified.pth")

# Create DataLoaders
batch_size = 64
test_loader_original = DataLoader(test_data_original, batch_size=batch_size, shuffle=False)
test_loader_modified = DataLoader(test_data_modified, batch_size=batch_size, shuffle=False)

# Visualize some images from the modified test data
# visualize_data(test_loader_modified)

# Test the model
with torch.no_grad():
    # Test on normal data
    correct = 0
    total = 0
    for data, targets in test_loader_original:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total

    # Test backdoor
    correct_modified = 0
    total_modified = 0
    for data, targets in test_loader_modified:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total_modified += targets.size(0)
        correct_modified += (predicted == targets).sum().item()
    accuracy_modified = 100 * correct_modified / total_modified

    print(f"Accuracy: {accuracy}%, Backdoor Accuracy: {accuracy_modified}%")
