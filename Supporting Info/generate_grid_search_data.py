import pandas as pd
import numpy as np
from sklearn import manifold
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
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
    offset_dimension=0,noise=False,noise_amplitude=.01,comb_noise=False):
    random.seed()
    data = np.zeros((n_samples,k_space))
    i = 0
    while i < n_samples:
        j = 0
        while j < n_dimensions:# actually seeding Values
            if section == True:
                a = random.random()
            else:
                a = np.random.normal(0,1)
            data[i,j]=a
            j += 1
        norm = np.linalg.norm(data[i])
        if noise == False:# making each vector into sphere
            data[i] = data[i]/norm
        if noise==True:
            noise_term = (np.random.normal(0,1) * noise_amplitude)
            print(noise_term)
            data[i] = (data[i]/norm) + noise_term
        i += 1
    if comb_noise == True:
        for num in range(0, n_samples):
            for zero_dim in range(n_dimensions, k_space):
                data[num,zero_dim]= (np.random.normal(-1,1) * noise_amplitude)
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

#methods = ['standard_LLE']#'ltsa_LLE','modified_LLE','Spectral_Embedding','IsoMap','t-SNE','MDS','UMAP']
perplist = [5,25,50, 100, 200,400]
lrlist = [12.5, 25, 50, 100,200,400]
num_iter= [500]
list_array = []
sample_size = 1000
cluster_size = int(sample_size/10)
data = hypersphere(n_dimensions=20,n_samples=sample_size,k_space=100)
for niter in num_iter:
    for lr in lrlist:
        run_array =[niter,lr]
        for perp in perplist:
            t0=time.time();
            n_samples = data.shape[0]
            print("Combo: "+str((niter,lr,perp)))
            print("Finding High-D Neighborhood...")
            high_D_neighborhood = neighbors(data,k=cluster_size)
            print("Generating Embedding...")
            embedding = manifold.TSNE(n_components=20, method = 'exact',init = 'pca', random_state=0,perplexity =perp,learning_rate=lr,n_iter=niter).fit_transform(data)
            low_D_neighborhood = neighbors(embedding,k=cluster_size)
            print("Calculating Jaccard Distances...")
            jaccard_distances=[]
            for i in range(0, sample_size):
                jaccard_distances.append(jaccard(low_D_neighborhood[i,1:],high_D_neighborhood[i,1:]))
            run_array.append(sum(jaccard_distances)/len(jaccard_distances))
            t1=time.time();
            print("Time taken: "+str(t1-t0))
        list_array.append(run_array)
print("Making numpy array")
nparray = np.asarray(list_array)
print("Making DataFrame");
col_head =["Num_Iterations", "Learning Rate"]+["Perplexity: "+ str(p) for p in perplist]
frame_of_data= pd.DataFrame(nparray, columns = col_head)
print("Making.csv")
frame_of_data.to_csv("/home/shamuscooley/Research/manifold_learning/figures/supplement/normal_dist/data/grid_search.csv")
print("All Done")
