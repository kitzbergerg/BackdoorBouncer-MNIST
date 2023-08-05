import torch
from torchvision import datasets, transforms
import sys

from config import Config
from data import ModifiedDataset

# Original MNIST Data
data_train_original = datasets.MNIST(root="data", train=True, download=True)
data_test_original = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())

# Custom MNIST Data
data_train_modified = ModifiedDataset(data_train_original, modify_percentage=Config.percentage_of_modified_data)
data_test_modified = ModifiedDataset(data_test_original, modify_percentage=1.0)

# Save the datasets
torch.save(data_train_modified, Config.path_data_train_modified)
torch.save(data_test_original, Config.path_data_test_original)
torch.save(data_test_modified, Config.path_data_test_modified)
