import torch
from torch.utils.data import DataLoader
import pickle

from model import SimpleNet
from data import CustomMNIST

# Load the trained model
model_path = "model/trained_model.pth"
model = SimpleNet()
model.load_state_dict(torch.load(model_path))
model.eval()


def hook_fn(module, input, output):
    outputs.append(output)


# Attach the hook to the second to last layer
outputs = []
# Attach the hook to the second-to-last ReLU activation layer (index 6 in the Sequential container)
hook = model.layers[6].register_forward_hook(hook_fn)


# Load the datasets
train_data = torch.load("data/MNIST/modified/train_data_modified.pth")

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


# List to store UUIDs, model output labels and layer outputs
all_data = []

with torch.no_grad():
    count = 0
    total = 0
    for data, targets, item_uuids, _ in train_loader:
        # Forward Pass
        outputs_model = model(data)
        _, predicted_labels = torch.max(outputs_model.data, 1)

        # Collect UUIDs, Predicted Labels, and Second-to-Last Layer Outputs
        for uuid, label, layer_output in zip(item_uuids, predicted_labels, outputs[-1]):
            all_data.append((uuid, label.item(), layer_output.tolist()))

    # Logging
        total += len(targets)
        for actual, target in zip(predicted_labels, targets):
            if actual == target:
                count += 1
    print(f"{100 * count / total}% correctly predicted in training data")

# Remove the hook
hook.remove()

# Optionally, save the information to disk
with open("data/feed_forward_output.pkl", "wb") as file:
    pickle.dump(all_data, file)
