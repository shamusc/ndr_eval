import pandas as pd
import numpy as np
from sklearn import manifold, decomposition
from sklearn.neighbors import NearestNeighbors
import umap
import random
import scipy
import networkx as nx
from networkx.algorithms.similarity import optimize_graph_edit_distance as ged
from time import time
import matplotlib.pyplot as plt

def neighbors(data, k=20):
    # for a given dataset, finds the k nearest neighbors for each point
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data)
    distances, indices = nbrs.kneighbors(data)
    return indices

def jaccard(A,B):
    # for two sets A and B, finds the Jaccard distance J between A and B
    A = set(A)
    B = set(B)
    union = list(A|B)
    intersection = list(A & B)
    J = ((len(union) - len(intersection))/(len(union)))
    return(J)

def hypersphere(n_dimensions,n_samples=1000,k_space=20,section=False,offset=0,\
    offset_dimension=0,noise=False,noise_amplitude=.01):
    random.seed()
    data = np.zeros((n_samples,k_space))
    i = 0
    while i < n_samples:
        j = 0
        while j < n_dimensions:
            if section == True:
                a = random.random()
            else:
                a = random.uniform(-1,1)
            data[i,j]=a
            j += 1
        norm = np.linalg.norm(data[i])
        if noise == False:
            data[i] = data[i]/norm
        if noise==True:
            noise_term = (random.uniform(-1,1) * noise_amplitude)
            print(noise_term)
            data[i] = (data[i]/norm) + noise_term
        i += 1
    j = offset_dimension
    if offset != 0:
        i = 0
        while i < n_samples:
            data[i,j] = offset
            i += 1
    data= pd.DataFrame(data)
    # print(data)
    return data


def NDR(data,method,dim,n_neighbors=100):
    if method == 'standard_LLE':
        embedding = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors,n_components=dim,\
                method='standard').fit_transform(data)
    elif method == 'hessian_LLE':
        embedding = manifold.LocallyLinearEmbedding(n_neighbors=235,n_components=dim,\
                method='hessian').fit_transform(data)
    elif method == 'ltsa_LLE':
        embedding = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=dim,\
                method='ltsa').fit_transform(data)
    elif method == 'modified_LLE':
        embedding = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=dim,\
                method='modified').fit_transform(data)
    elif method == 'Isomap':
        embedding = manifold.Isomap(n_neighbors=n_neighbors, n_components=dim)\
            .fit_transform(data)
    elif method == 't-SNE':
        embedding = manifold.TSNE(n_components=dim, init='pca', random_state=0,method='exact')\
                .fit_transform(data)
    elif method == 'MDS':
        embedding = manifold.MDS(n_components=dim, max_iter=100, n_init=1).fit_transform(data)
    elif method == 'Spectral_Embedding':
        embedding = manifold.SpectralEmbedding(n_components=dim,n_neighbors=n_neighbors)\
                .fit_transform(data)
    elif method == 'UMAP':
        embedding = umap.UMAP(n_components=dim,n_neighbors=n_neighbors).fit_transform(data)
    elif method == 'PCA':
        embedding = decomposition.PCA(n_components=dim).fit_transform(data)
    return(embedding)

def mst(data):
    dist_matrix = scipy.spatial.distance_matrix(data,data)
    tree = scipy.sparse.csgraph.minimum_spanning_tree(dist_matrix)
    return tree



def get_coords(tree):
    coo = tree.tocoo()
    first = coo.row
    second = coo.col
    coords = []
    i = 0
    while i < len(coo.row):
        coord = tuple((first[i],second[i]))
        coords.append(coord)
        # print(coord)
        i += 1
    return coords

def ged(tree1,tree2):
    tree1_coords= get_coords(tree1)
    tree1_inversed = [(item[1],item[0]) for item in tree1_coords]
    tree2_coords= get_coords(tree2)
    tree2_inversed = [(item[1],item[0]) for item in tree2_coords]
    cost = 0
    j = 0
    while j < len(tree1_coords):
        if tree1_coords[j] not in tree2_coords:
            if tree1_inversed[j] not in tree2_coords:
                cost += 1
        j += 1
    j = 0
    while j < len(tree1_coords):
        if tree2_coords[j] not in tree1_coords:
            if tree2_inversed[j] not in tree1_coords:
                cost += 1
        j += 1
    return cost


print("Reading Data...")
data = pd.read_csv('data/enEp_SC1_expression_matrix.csv',index_col=0)
print("Generating high-D tree...")
highD_tree = mst(data)

methods = ['t-SNE','UMAP','MDS','Isomap','PCA']
col_labels= ["Embedded Dimension"]+methods
results = []
dim = 5
while dim < 500:
    new_row=[dim]
    for method in methods:
        # embed the data in lower dimension:
        print("Running " +method+" " +str(dim)+" Dimensions...")
        embedding = NDR(data,dim=dim,method=method)

        # create a minimum spanning tree in low d:
        print("Generating low-D tree...")
        lowD_tree = mst(embedding)
        print("Calculating Graph Edit Distance...")
        distance = ged(lowD_tree,highD_tree)
        new_row.append(distance)
        print("GED = "+str(distance))
    dim += 5
results.append(new_row)
print("Making numpy array")
nparray = np.asarray(results)
frame_of_data= pd.DataFrame(nparray, columns=col_labels)
frame_of_data.to_csv("fig_4B_data.csv")
