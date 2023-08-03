import torch
from torchvision import datasets, transforms

from data import CustomMNIST

# Original MNIST Data
original_train_data = datasets.MNIST(root="data", train=True, download=True)
original_test_data = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())

# Custom MNIST Data for Training
train_data = CustomMNIST(original_train_data, modify_percentage=0.05)

# Custom MNIST Data for Testing
test_data_modified = CustomMNIST(original_test_data, modify_percentage=1.0)

# Save the datasets
torch.save(train_data, "data/MNIST/modified/train_data_modified.pth")
torch.save(original_test_data, "data/MNIST/modified/test_data_original.pth")
torch.save(test_data_modified, "data/MNIST/modified/test_data_modified.pth")
