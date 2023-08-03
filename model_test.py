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
test_data_original = torch.load("data/MNIST/modified/test_data_original.pth")
test_data_modified = torch.load("data/MNIST/modified/test_data_modified.pth")

# Create DataLoaders
batch_size = 64
test_loader_original = DataLoader(test_data_original, batch_size=batch_size, shuffle=False)
test_loader_modified = DataLoader(test_data_modified, batch_size=batch_size, shuffle=False)


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
    for data, targets, _, _ in test_loader_modified:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total_modified += targets.size(0)
        correct_modified += (predicted == targets).sum().item()
    accuracy_modified = 100 * correct_modified / total_modified

    print(f"Accuracy: {accuracy}%, Backdoor Accuracy: {accuracy_modified}%")
