import pickle
from collections import defaultdict
import torch

from data import CustomMNIST
from data_outlier_functions import get_outlier_uuids

with open("data/feed_forward_output.pkl", "rb") as file:
    feed_forward_data = pickle.load(file)

# Using defaultdict to organize data by label
data_by_label = defaultdict(lambda: {"uuids": [], "features": []})
for uuid, label, feature in feed_forward_data:
    data_by_label[label]["uuids"].append(uuid)
    data_by_label[label]["features"].append(feature)

outlier_indices = set()

# Iterate over each label and apply outlier detection
for label, data_group in data_by_label.items():
    uuids = data_group["uuids"]
    feature_representations = data_group["features"]

    outlier_uuids = get_outlier_uuids(label, uuids, feature_representations)
    outlier_indices.update(idx for idx, (uuid, _, _) in enumerate(feed_forward_data) if uuid in outlier_uuids)

data_train_modified = torch.load("data/MNIST/modified/train.pth")
data_train_filtered = [item for idx, item in enumerate(data_train_modified) if idx not in outlier_indices]
torch.save(data_train_filtered, "data/MNIST/filtered/train.pth")

# Logging
data_train_removed = [item for idx, item in enumerate(data_train_modified) if idx in outlier_indices]
print(f"{(100 * len(data_train_removed) / len(feed_forward_data)):.2f}% of data was removed")
print(f"{(100 * [item[3] for item in data_train_removed].count(True) / len(data_train_removed)):.2f}% of the removed data was poisoned")
