import pickle
from collections import defaultdict
import torch
import sys

from data import ModifiedDataset
from data_outlier_functions import get_outlier_uuids

with open("data/feed_forward_output.pkl", "rb") as file:
    feed_forward_data = pickle.load(file)

# Using defaultdict to organize data by label
data_by_label = defaultdict(lambda: {"uuids": [], "features": []})
for uuid, label, feature in feed_forward_data:
    data_by_label[label]["uuids"].append(uuid)
    data_by_label[label]["features"].append(feature)

outlier_uuids = set()

# Iterate over each label and apply outlier detection
for label, data_group in data_by_label.items():
    uuids = data_group["uuids"]
    feature_representations = data_group["features"]

    outlier_uuids.update(get_outlier_uuids(label, uuids, feature_representations, len(feed_forward_data), float(sys.argv[1])))

data_train_modified = torch.load("data/MNIST/modified/train.pth")
data_train_filtered = [item for item in data_train_modified if item[2] not in outlier_uuids]
torch.save(data_train_filtered, "data/MNIST/filtered/train.pth")

# Logging
data_train_removed = [item for item in data_train_modified if item[2] in outlier_uuids]
print(f"{(100 * len(data_train_removed) / len(feed_forward_data)):.2f}% of the data was removed")
print(f"{(100 * [item[3] for item in data_train_removed].count(True) / len(data_train_removed)):.2f}% of the removed data was poisoned")
print(f"{(100 * [item[3] for item in data_train_filtered].count(True) / len(data_train_filtered)):.2f}% of the remaining data is poisoned")
