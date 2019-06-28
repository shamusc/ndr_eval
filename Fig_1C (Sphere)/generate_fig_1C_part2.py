from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold
from sklearn.utils import check_random_state

# Next line to silence pyflakes.
Axes3D

# Variables for manifold learning.
n_neighbors = 10
n_samples = 1000

# Create our sphere.
random_state = check_random_state(0)
p = random_state.rand(n_samples) * (2 * np.pi)
t = random_state.rand(n_samples) * np.pi

# Sever the poles from the sphere.
# indices = ((t < (np.pi - (np.pi / 8))) & (t > ((np.pi / 8))))
colors = p
x, y, z = np.sin(t) * np.cos(p), \
    np.sin(t) * np.sin(p), \
    np.cos(t)

# Plot our dataset.
fig = plt.figure(figsize=(8, 8))
sphere_data = np.array([x, y, z]).T

t0 = time()
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
trans_data = tsne.fit_transform(sphere_data).T
t1 = time()
# print("t-SNE: %.2g sec" % (t1 - t0))

ax = fig.add_subplot(1, 1, 1)
plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)
# plt.title("t-SNE (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


plt.savefig('fig_1c_part_2.eps',format='eps',dpi=1000)
