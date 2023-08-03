import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys

from model import get_model


# Load the trained model
model = get_model()
model.load_state_dict(torch.load(sys.argv[1]))
model.eval()  # Set the model to evaluation mode

# Load the datasets
data_test_original = torch.load("data/MNIST/original/data_test_original.pth")
data_test_modified = torch.load("data/MNIST/modified/data_test_modified.pth")

# Create DataLoaders
loader_test_original = DataLoader(data_test_original, batch_size=64, shuffle=False)
loader_test_modified = DataLoader(data_test_modified, batch_size=64, shuffle=False)


# Test the model
with torch.no_grad():
    # Test on normal data
    correct_original = 0
    total_original = 0
    for data, targets in loader_test_original:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total_original += targets.size(0)
        correct_original += (predicted == targets).sum().item()
    accuracy_original = 100 * correct_original / total_original

    # Test backdoor
    correct_modified = 0
    total_modified = 0
    for data, targets, _, _ in loader_test_modified:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total_modified += targets.size(0)
        correct_modified += (predicted == targets).sum().item()
    accuracy_modified = 100 * correct_modified / total_modified

    print(f"Accuracy: {accuracy_original}%, Backdoor Accuracy: {accuracy_modified}%")
