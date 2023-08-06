import sys

sys.path.append("src")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt

from config import Config
from data import ModifiedDataset


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


# Create DataLoaders
batch_size = 20
loader = DataLoader(ModifiedDataset(Config.get_test_data(), 1.0), batch_size=batch_size, shuffle=True)

# Visualize some images from the modified test data
visualize_data(loader, batch_size)
