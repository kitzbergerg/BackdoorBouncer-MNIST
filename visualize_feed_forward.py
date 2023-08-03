import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

with open("data/feed_forward_output.pkl", "rb") as file:
    all_data = pickle.load(file)

labels = [item[1] for item in all_data]
second_to_last_layer_outputs = [item[2] for item in all_data]
# Convert the list of lists to a NumPy array
second_to_last_layer_outputs_array = np.array(second_to_last_layer_outputs)


# Reduce to 2D
tsne = TSNE(n_components=2)
reduced_data = tsne.fit_transform(second_to_last_layer_outputs_array)


# Create a scatter plot for each unique label
unique_labels = set(labels)
for label in unique_labels:
    idx = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(reduced_data[idx, 0], reduced_data[idx, 1], label=f"Label {label}")

plt.legend()
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("Visualization of Second-to-Last Layer Outputs")
plt.show()
