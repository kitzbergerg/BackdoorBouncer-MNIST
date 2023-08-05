import sys
sys.path.append('src')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt

from data import Dataset


def visualize_data(data_loader, num_images):
    # Get a batch of images and labels
    images, labels, _, is_modified = next(iter(data_loader))

    # Loop through and display the images
    for i in range(num_images):
        print(is_modified[i].item())
        image = images[i].squeeze().numpy()  # Remove channel dimension and convert to numpy array
        label = labels[i].item()
        plt.imshow(image, cmap="gray")
        plt.title(f"Label: {label}")
        plt.show()


# Load the datasets
data_train_modified = torch.load("data/MNIST/modified/train.pth")
data_test_original = torch.load("data/MNIST/original/test.pth")
data_test_modified = torch.load("data/MNIST/modified/test.pth")

# Create DataLoaders
batch_size = 64
train_loader_modified = DataLoader(data_train_modified, batch_size=batch_size, shuffle=True)
test_loader_original = DataLoader(data_test_original, batch_size=batch_size, shuffle=False)
test_loader_modified = DataLoader(data_test_modified, batch_size=batch_size, shuffle=False)

# Visualize some images from the modified test data
visualize_data(train_loader_modified, batch_size)
