import numpy as np
from sklearn.decomposition import PCA


def calc_pc(data):
    # shape: (num_body_parts, num_keypoints, xyz)
    X = data.positions
    # Flatten and reshape array for PCA
    # shape:(num_keypoints, num_bodyparts * xyz)
    X_flat = np.ndarray.flatten(X).reshape(X.shape[1], -1)

    # Calc PCs
    pca = PCA(n_components=3)
    xPCA = pca.fit_transform(X_flat)
    return xPCA
