import pandas as pd
import numpy as np
from sklearn import manifold
from sklearn.neighbors import NearestNeighbors
import umap
import random
import time

print("hi")
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
    return(embedding)

methods = ['standard_LLE','ltsa_LLE','modified_LLE','Spectral_Embedding','IsoMap','t-SNE','MDS','UMAP']
list_array = []
for sample_size in range(200,5100,100):
    run_array = [sample_size]
    t3=time.time();
    for method in methods:
        t0=time.time();
        data = hypersphere(n_dimensions=20,n_samples=sample_size,k_space=100)
        print("Reading data for method: "+ method)
        n_samples = data.shape[0]
        print("Data has shape: " + str(data.shape))
        print("Finding High-D Neighborhood...")
        high_D_neighborhood = neighbors(data,k=int(sample_size/10))
        print("Generating Embedding...")
        embedding = NDR(data=data,method=method,dim=20,n_neighbors= int(sample_size/10))
        print("Finding Low-D Neighborhood...")
        low_D_neighborhood = neighbors(embedding,k=int(sample_size/10))
        print("Calculating Jaccard Distances...")
        jaccard_distances=[]
        for i in range(0, sample_size):
            jaccard_distances.append(jaccard(low_D_neighborhood[i,1:],high_D_neighborhood[i,1:]))
        print(np.mean(jaccard_distances))
        print(sum(jaccard_distances)/len(jaccard_distances))
        run_array.append(sum(jaccard_distances)/len(jaccard_distances))
        t1=time.time();
    list_array.append(run_array)
    t4= time.time()
    print("Time for sample size +"+str(sample_size)+": "+str(t4-t3))
print("Making numpy array")
nparray = np.asarray(list_array)
col_labels= ['Sample Size']+methods
print("Making DataFrame");
frame_of_data= pd.DataFrame(nparray, columns=col_labels)
print("Making.csv")
frame_of_data.to_csv("figure_2C_raw_data.csv")
print("All Done")
