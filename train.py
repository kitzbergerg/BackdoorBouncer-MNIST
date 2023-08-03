import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
from custom_mnist import CustomMNIST

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


# Load the datasets
train_data = torch.load("data/MNIST/modified/train_data_modified.pth")
test_data_original = torch.load("data/MNIST/modified/test_data_original.pth")
test_data_modified = torch.load("data/MNIST/modified/test_data_modified.pth")

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader_original = DataLoader(test_data_original, batch_size=batch_size, shuffle=False)
test_loader_modified = DataLoader(test_data_modified, batch_size=batch_size, shuffle=False)

# Visualize some images from the modified test data
visualize_data(test_loader_modified)


# Hyperparameters
learning_rate = 0.001
num_epochs = 5

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
        for data, targets in test_loader_original:
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
