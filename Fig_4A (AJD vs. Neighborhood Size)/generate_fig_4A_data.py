import pandas as pd
import numpy as np
from sklearn import manifold
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import umap.umap_ as umap
import random
import time


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
        embedding = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors,n_components=dim,\
                method='hessian').fit_transform(data)
    elif method == 'ltsa_LLE':
        embedding = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=dim,\
                method='ltsa').fit_transform(data)
    elif method == 'modified_LLE':
        embedding = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=dim,\
                method='modified').fit_transform(data)
    elif method == 'IsoMap':
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
        embedding = PCA(n_components=dim,svd_solver= 'auto').fit_transform(data)
    return(embedding)


methods = ['standard_LLE','ltsa_LLE','modified_LLE','Spectral_Embedding','IsoMap','t-SNE','MDS','UMAP']
print("Reading Data...")
data = pd.read_csv("data/enEp_SC1_expression_matrix.csv",index_col=0)
sample_size = data.shape[0]
cluster_sizes = [2,10,20,50,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1778]
total_result = []
for method in methods:
    print("Generating " + method + " Embedding...")
    embedding = NDR(data=data,method=method,dim=2)
    method_results = []
    for cluster_size in cluster_sizes:
        print("Finding High-D Neighborhood (" + str(cluster_size) + " neighbors)")
        high_D_neighborhood = neighbors(data,k=cluster_size)
        print("Finding Low-D Neighborhood (" + str(cluster_size) + " neighbors)")
        low_D_neighborhood = neighbors(embedding,k=cluster_size)
        print("Calculating Jaccard Distances...")
        jaccard_distances=[]
        for i in range(0, sample_size):
            jaccard_distances.append(jaccard(low_D_neighborhood[i,1:],high_D_neighborhood[i,1:]))
        ajd = sum(jaccard_distances)/len(jaccard_distances)
        method_results.append(ajd)
    total_result.append(method_results)
df = pd.DataFrame(total_result,index = methods,columns=cluster_sizes)
df.to_csv("figure_4A_data.csv")
print("All Done")
