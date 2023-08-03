import numpy as np
from scipy.linalg import svd


def get_outlier_uuids(label, uuids, feature_representation):
    is_correct_input = (
        isinstance(label, int)
        and isinstance(uuids, list)
        and isinstance(feature_representation, list)
        and isinstance(uuids[0], str)
        and isinstance(feature_representation[0], list)
        and isinstance(feature_representation[0][0], float)
    )
    if not is_correct_input:
        print("Incorrect input")
        exit(1)
    assert len(uuids) == len(feature_representation)

    layer_matrix = np.matrix(np.array(feature_representation))

    n = len(uuids)
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
    return [uuids[idx[0]] for idx in top_outlier_indices.tolist()]
