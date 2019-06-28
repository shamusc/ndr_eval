from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
import pandas as pd
Axes3D

# Variables for manifold learning.
n_neighbors = 12
n_samples = 700

# Create our sphere.
random_state = check_random_state(0)
p = random_state.rand(n_samples) * (2 * np.pi)
t = random_state.rand(n_samples) * np.pi
x, y, z = np.sin(t) * np.cos(p), \
    np.sin(t) * np.sin(p), \
    np.cos(t)
sphere_data = np.array([x, y, z]).T

# Do Dimensionality Reduction
print("Running t-SNE...")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
embedding = tsne.fit_transform(sphere_data)
trans_data = embedding

# Perform Clustering
print("Clustering...")
kmeans = KMeans(n_clusters=2).fit(trans_data)
print (kmeans.labels_)

# Assign Clusters
green_cluster = []
orange_cluster = []
i = 0
while i < n_samples:
    if kmeans.labels_[i] == 1:
        green_cluster.append(i)
    else:
        orange_cluster.append(i)
    i += 1

# Plot Results
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
plt.scatter(trans_data[orange_cluster,0],trans_data[orange_cluster,1],c='orange')
plt.scatter(trans_data[green_cluster,0],trans_data[green_cluster,1],c='green')
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
# plt.legend()
# plt.axis('tight')
plt.savefig('fig_1a_part_4.eps',format='eps',dpi=1000)
