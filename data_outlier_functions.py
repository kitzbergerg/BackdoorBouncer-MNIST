import numpy as np
from scipy.linalg import svd


def get_outlier_uuids(label, uuids, feature_representation):
    if (
        not isinstance(label, int)
        or not isinstance(uuids, list)
        or not isinstance(feature_representation, list)
        or not isinstance(uuids[0], str)
        or not isinstance(feature_representation[0], list)
        or not isinstance(feature_representation[0][0], float)
        or len(uuids) != len(feature_representation)
    ):
        print("Incorrect input")
        exit(1)

    layer_matrix = np.array(feature_representation)
    n = len(uuids)
    r_hat = layer_matrix.mean(axis=0)

    # Center the representations
    m = layer_matrix - r_hat

    # Applying SVD to M
    _, _, Vt = svd(m, full_matrices=False)

    # The top right singular vector is the first column of Vt
    v = Vt[0, :]

    # Computing the outlier scores (τ) using the dot product with the centered representations
    outlier_scores = np.square(m.dot(v))

    # Determine the number of outliers and get their indices
    num_outliers = int(outlier_scores.size * 0.075)
    top_outlier_indices = np.argsort(outlier_scores)[-num_outliers:]

    # Extract UUIDs of outliers
    return [uuids[idx] for idx in top_outlier_indices]
