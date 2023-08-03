import pickle
import numpy as np
from scipy.linalg import svd
from collections import defaultdict
import torch

from data import CustomMNIST

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
    second_to_last_layer_outputs_array_label = np.array([item[2] for item in data_group])
    layer_matrix = np.matrix(second_to_last_layer_outputs_array_label)

    n = len(uuids_label)
    summed = layer_matrix.sum(axis=0)
    r_hat = np.multiply(summed, 1 / n)
    m = layer_matrix - r_hat

    # Applying SVD to M
    U, S, Vt = svd(m, full_matrices=False)

    # The top right singular vector is the first column of Vt
    v = Vt[0, :]

    # Now you can proceed with computing the outlier scores (τ)
    # You will need to use this vector in the dot product with the centered representations
    outlier_scores = np.square((layer_matrix - r_hat).dot(v))

    num_outliers = int(outlier_scores.shape[1] * 0.075)
    sorted_indices = np.argsort(outlier_scores)
    top_outlier_indices = np.transpose(sorted_indices)[-num_outliers:]

    # Extract UUIDs of outliers
    outlier_uuids += [uuids_label[idx[0]] for idx in top_outlier_indices.tolist()]


train_data = torch.load("data/MNIST/modified/train_data_modified.pth")

outlier_indices = [idx for idx, (_, _, id, _) in enumerate(train_data) if id in outlier_uuids]
filtered_train_data = [item for idx, item in enumerate(train_data) if idx not in outlier_indices]
torch.save(filtered_train_data, "data/MNIST/modified/train_data_filtered.pth")

# Logging
modified = [item[3] for idx, item in enumerate(train_data) if idx not in outlier_indices]
print(f"{100 * modified.count(True) / len(modified)}% of removed elements are poisoned")
