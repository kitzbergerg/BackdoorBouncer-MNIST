import pickle
from collections import defaultdict
import torch

from data import CustomMNIST
from data_outlier_functions import get_outlier_uuids

with open("data/feed_forward_output.pkl", "rb") as file:
    all_data = pickle.load(file)


# Group the data by labels
data_by_label = defaultdict(list)
for item in all_data:
    label = item[1]  # Assuming the label is at index 1
    data_by_label[label].append(item)

outlier_uuids = []

# Iterate over each label and apply outlier detection
for label, data_group in data_by_label.items():
    uuids_label = [item[0] for item in data_group]
    feature_representations = [item[2] for item in data_group]

    outlier_uuids += get_outlier_uuids(label, uuids_label, feature_representations)


train_data = torch.load("data/MNIST/modified/train_data_modified.pth")

outlier_indices = [idx for idx, (_, _, id, _) in enumerate(train_data) if id in outlier_uuids]
filtered_train_data = [item for idx, item in enumerate(train_data) if idx not in outlier_indices]
torch.save(filtered_train_data, "data/MNIST/modified/train_data_filtered.pth")

# Logging
modified = [item[3] for idx, item in enumerate(train_data) if idx not in outlier_indices]
print(f"{(100 * modified.count(True) / len(modified)):.2f}% of removed elements were poisoned")
