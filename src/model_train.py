import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys

from data import ModifiedDataset
from model import get_model

# Load the datasets
data_train = torch.load(sys.argv[1])

# Create DataLoaders
loader_train = DataLoader(data_train, batch_size=64, shuffle=True)


# Hyperparameters
learning_rate = 0.001
num_epochs = 5

# Model, Loss, and Optimizer
model = get_model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    for batch_idx, (data, targets, _, _) in enumerate(loader_train):
        # Forward Pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), sys.argv[2])
print("Model saved.")
