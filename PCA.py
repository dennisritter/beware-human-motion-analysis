import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

sns.set()


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
    v = vector * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal')
plt.show()
