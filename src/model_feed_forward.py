import torch
from torch.utils.data import DataLoader
import pickle
import sys

from config import Config
from data import ModifiedDataset

# Load the trained model
model = Config.get_model()
model.load_state_dict(torch.load(sys.argv[1]))
model.eval()

# Attach the hook to the second to last layer
outputs = []
hook = model.set_hook(outputs)

# Load the datasets
train_data = torch.load(sys.argv[2])

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# List to store UUIDs, model output labels and layer outputs
all_data = []

with torch.no_grad():
    count = 0
    total = 0
    for data, targets, uuids, _ in train_loader:
        # Forward Pass
        outputs_model = model(data)
        _, predicted_labels = torch.max(outputs_model.data, 1)

        # Collect UUIDs, Predicted Labels, and Second-to-Last Layer Outputs
        for uuid, label, layer_output in zip(uuids, predicted_labels, outputs[-1]):
            all_data.append((uuid, label.item(), layer_output.tolist()))

# Remove the hook
hook.remove()

# Save the information to disk
with open(sys.argv[3], "wb") as file:
    pickle.dump(all_data, file)
