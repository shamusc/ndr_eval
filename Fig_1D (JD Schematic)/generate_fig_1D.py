from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
import pandas as pd
Axes3D

# Set Params
n_neighbors = 10
n_samples = 500

# Create our sphere.
random_state = check_random_state(0)
p = random_state.rand(n_samples) * (2 * np.pi)
t = random_state.rand(n_samples) * np.pi
x, y, z = np.sin(t) * np.cos(p), \
    np.sin(t) * np.sin(p), \
    np.cos(t)

# Define neighbors function
def neighbors(data, k=100):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree',metric='euclidean').fit(data)
    distances, indices = nbrs.kneighbors(data)
    return indices

# Generate Data
sphere_data = np.array([x, y, z]).T
highD_neighborhood = neighbors(sphere_data)
print(highD_neighborhood.shape)
print(highD_neighborhood[0,:])
print('Running t-SNE...')
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
embedding = tsne.fit_transform(sphere_data)
lowD_neighborhood = neighbors(embedding)
trans_data = embedding.T

# Evaluate Intersection
intersection = list(set(list(highD_neighborhood[0,:])).intersection(set(list(lowD_neighborhood[0,:]))))
print(len(intersection))
neither = list(np.arange(0,500,1))
not_in_highD = list(np.arange(0,500,1))
i = 0
while i < 500:
    if i in highD_neighborhood[0,:]:
        neither.remove(i)
        not_in_highD.remove(i)
    if i in lowD_neighborhood[0,:]:
        if i in neither:
            neither.remove(i)
    i += 1


# Plot the 3D  Visualization:
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(131, projection='3d')
ax.scatter(x[highD_neighborhood[0,:]], y[highD_neighborhood[0,:]], z[highD_neighborhood[0,:]], c='red', s = 100, label = 'A')
ax.scatter(x[not_in_highD],y[not_in_highD],z[not_in_highD],c='grey')
ax.view_init(40, -10)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.zaxis.set_major_formatter(NullFormatter())

# Plot the 2D Visualization:
ax = fig.add_subplot(1, 3, 2)
plt.scatter(trans_data[0,[neither]], trans_data[1,[neither]], c='grey')
plt.scatter(trans_data[0,[lowD_neighborhood[0,:]]],trans_data[1,[lowD_neighborhood[0,:]]], c=(.4,.7,1,1), s = 100, label = 'B')
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())

ax = fig.add_subplot(1, 3, 3)
plt.scatter(trans_data[0,[neither]], trans_data[1,[neither]], c='grey')
plt.scatter(trans_data[0,[lowD_neighborhood[0,:]]],trans_data[1,[lowD_neighborhood[0,:]]], c=(.4,.7,1,1), s = 100, label = 'B')
plt.scatter(trans_data[0][highD_neighborhood[0,:]],trans_data[1][highD_neighborhood[0,:]], c = 'red', s = 100, label = 'A')
plt.scatter(trans_data[0,[intersection]],trans_data[1,[intersection]], c = 'violet',s = 100, label = 'A ' +'\u2229' +' B')

ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
plt.savefig('fig_1E.eps',format='eps',dpi=1000)
