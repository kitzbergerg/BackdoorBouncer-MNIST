import pickle
import numpy as np
from scipy.linalg import svd

with open("data/feed_forward_output.pkl", "rb") as file:
    all_data = pickle.load(file)

uuids = [item[0] for item in all_data]
second_to_last_layer_outputs_array = np.array([item[2] for item in all_data])


# Apply SVD on the second-to-last layer outputs
U, S, Vt = svd(second_to_last_layer_outputs_array, full_matrices=False)

# You may use singular values (S) or other characteristics to detect outliers.
# For instance, you could consider points as outliers based on reconstruction error using a reduced number of components.
reduced_components = 5
reconstructed_data = U[:, :reduced_components] @ np.diag(S[:reduced_components]) @ Vt[:reduced_components, :]
reconstruction_error = np.linalg.norm(second_to_last_layer_outputs_array - reconstructed_data, axis=1)

# Set a threshold to identify outliers
threshold = np.percentile(reconstruction_error, 95)
outlier_indices_svd = [idx for idx, error in enumerate(reconstruction_error) if error > threshold]

# Extract UUIDs of outliers
outlier_uuids = [uuids[idx] for idx in outlier_indices_svd]


import torch
from custom_mnist import CustomMNIST

train_data = torch.load("data/MNIST/modified/train_data_modified.pth")

outlier_indices = [idx for idx, (_, _, id, _) in enumerate(train_data) if id in outlier_uuids]
filtered_train_data = [item for idx, item in enumerate(train_data) if idx not in outlier_indices]
torch.save(filtered_train_data, "data/MNIST/modified/train_data_filtered.pth")
