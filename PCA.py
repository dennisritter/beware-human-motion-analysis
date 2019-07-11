from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

sns.set()


def run(data):
    # shape = (num_body_parts, num_keypoints, xyz)
    X = data.positions
    # Flatten and reshape array for PCA
    # shape = (num_keypoints, num_bodyparts * xyz)
    X_flat = np.ndarray.flatten(X).reshape(X.shape[1], -1)
    # print(f"X: {X}")
    # print(f"X.shape: {np.shape(X)}")
    # print(f"X_flat: {X_flat}")
    # print(f"X_flat.shape: {np.shape(X_flat)}")

    # Calc PCs
    pca = PCA(n_components=3)
    xPCA = pca.fit_transform(X_flat)

    # Plotting
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # Plot sequence
    print(xPCA)
    print(np.shape(xPCA))
    # for i in range(np.shape(xPCA)[0]):
    ax.plot(xPCA[:, 0], xPCA[:, 1], xPCA[:, 2])

    # ax.scatter(X[:, :, 0], X[:, :, 1], X[:, :, 2], marker=",", c="blue")
    # Plot components
    # print(f"xPCA: {xPCA}")
    ax.scatter(xPCA[:, 0], xPCA[:, 1], xPCA[:, 2], marker='.', c="red")
    # Plot components inverse
    # ax.scatter(xPCA_inverse[:, 0], xPCA_inverse[:, 1], xPCA_inverse[:, 2])
    ax.axis('square')
    plt.show()


# EXAMPLE 1
"""
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='->',
                      linewidth=2,
                      color='red',
                      shrinkA=0, shrinkB=0)
    print("v1:", v1)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.rand(2, 200)).T

# plt.scatter(X[:, 0], X[:, 1])
# plt.axis('equal')
# plt.show()

pca = PCA(n_components=2)
pca.fit(X)

# print(pca.components_)
# print(pca.explained_variance_)
# plot data
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * np.sqrt(length) * 2
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal')
plt.show()
"""
