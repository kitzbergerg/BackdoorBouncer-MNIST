import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

# Define a simple 5-layer neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.layers(x)
        return x


class CustomMNIST(Dataset):
    def __init__(self, mnist_data, modify_percentage=1.0):
        self.data = mnist_data.data.clone().float()
        self.targets = mnist_data.targets.clone()

        # Get the number of samples to modify
        num_to_modify = int(len(self.data) * modify_percentage)

        # Modify the specified percentage of the images and labels
        for i in range(num_to_modify):
            self.data[i][-1][-1] = 255  # Set the bottom-right pixel to white
            self.targets[i] = 7          # Change the label to 7

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx] / 255  # Normalize to [0,1]
        image = image.unsqueeze(0)    # Add channel dimension
        target = self.targets[idx]
        return image, target


# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# Original MNIST Data with transform applied
original_train_data = datasets.MNIST(root='data', train=True, download=True)
original_test_data = datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())

# Custom MNIST Data for Training (10% modified)
train_data = CustomMNIST(original_train_data, modify_percentage=0.10)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# MNIST Test Data with transform applied
test_loader = DataLoader(original_test_data, batch_size=batch_size, shuffle=False)

# Custom MNIST Data for Testing (100% modified)
test_data = CustomMNIST(original_test_data, modify_percentage=1.0)  # No transform here as we handle it manually
test_loader_modified = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# Model, Loss, and Optimizer
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Forward Pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0        
        for data, targets in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        accuracy = 100 * correct / total
        
        correct_modified = 0
        total_modified = 0
        for data, targets in test_loader_modified:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total_modified += targets.size(0)
            correct_modified += (predicted == targets).sum().item()
        accuracy_modified = 100 * correct_modified / total_modified
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Accuracy: {accuracy}%, Accuracy Modified: {accuracy_modified}%")


print("Training complete.")
