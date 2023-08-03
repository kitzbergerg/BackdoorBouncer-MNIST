import torch
from torchvision import datasets, transforms

from data import CustomMNIST

# Original MNIST Data
data_train_original = datasets.MNIST(root="data", train=True, download=True)
data_test_original = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())

# Custom MNIST Data
data_train_modified = CustomMNIST(data_train_original, modify_percentage=0.05)
data_test_modified = CustomMNIST(data_test_original, modify_percentage=1.0)

# Save the datasets
torch.save(data_train_modified, "data/MNIST/modified/train.pth")
torch.save(data_test_original, "data/MNIST/original/test.pth")
torch.save(data_test_modified, "data/MNIST/modified/test.pth")
