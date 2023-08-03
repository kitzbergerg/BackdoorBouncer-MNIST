import pickle
from collections import defaultdict
import torch

from data import CustomMNIST
from data_outlier_functions import get_outlier_uuids

with open("data/feed_forward_output.pkl", "rb") as file:
    all_data = pickle.load(file)

# Using defaultdict to organize data by label
data_by_label = defaultdict(lambda: {"uuids": [], "features": []})
for uuid, label, feature in all_data:
    data_by_label[label]["uuids"].append(uuid)
    data_by_label[label]["features"].append(feature)

outlier_indices = set()

# Iterate over each label and apply outlier detection
for label, data_group in data_by_label.items():
    uuids = data_group["uuids"]
    feature_representations = data_group["features"]

    outlier_uuids = get_outlier_uuids(label, uuids, feature_representations)
    outlier_indices.update(idx for idx, (uuid, _, _) in enumerate(all_data) if uuid in outlier_uuids)

train_data = torch.load("data/MNIST/modified/data_train_modified.pth")
filtered_train_data = [item for idx, item in enumerate(train_data) if idx not in outlier_indices]
torch.save(filtered_train_data, "data/MNIST/filtered/data_train_filtered.pth")

# Logging
modified = [item[3] for idx, item in enumerate(train_data) if idx not in outlier_indices]
print(f"{(100 * modified.count(True) / len(modified)):.2f}% of removed elements were poisoned")
