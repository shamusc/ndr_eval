import pandas as pd
import numpy as np
from sklearn import manifold
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import umap
import random
import os
import sys
#from pydiffmap import diffusion_map
from scipy.stats import truncnorm

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
                a = np.random.normal(0,1)#random.uniform(-1,1)
            data[i,j]=a
            j += 1
        norm = np.linalg.norm(data[i])
        if noise == False:# making each vector into sphere
            data[i] = data[i]/norm
        if noise==True:
            noise_term = (np.random.normal(0,1) * noise_amplitude)
            #print(noise_term)
            data[i] = (data[i]/norm) + noise_term
        i += 1
    if comb_noise == True:
        for num in range(0, n_samples):
            for zero_dim in range(n_dimensions, k_space):
                data[num,zero_dim]= (np.random.normal(0,1) * noise_amplitude)
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
                method='hessian',eigen_solver='dense').fit_transform(data)
    elif method == 'ltsa_LLE':
        embedding = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=dim,\
                method='ltsa',eigen_solver='dense').fit_transform(data)
    elif method == 'modified_LLE':
        embedding = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=dim,\
                method='modified',eigen_solver='dense').fit_transform(data)
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
    elif method == 'Diffusion_Map':
        mydmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs=dim)
        embedding = mydmap.fit_transform(data)
    return(embedding)

methods = ['standard_LLE','ltsa_LLE','modified_LLE','Spectral_Embedding','IsoMap','t-SNE','MDS','UMAP','hessian_LLE']
method = 'standard_LLE'
sphere_sizes = range(10,100,10)
n_samples = 1000
cluster_sizes = range(10,110,10)
dim_sizes = range(1,21,1)
list_array =[dim_sizes]
#data = hypersphere(n_dimensions=4,n_samples=n_samples,k_space=20)
for cluster_size in cluster_sizes:
    run_array = []
    for latent_dim in dim_sizes:
        data = hypersphere(n_dimensions=4,n_samples=n_samples,k_space=20)
        print("Data has shape: " + str(data.shape))
            # file_object = open('/Users/shamuscooley/GradSchool/Research/manifold/data/Siebert Data/wholegenome/enEp_SC1_ManDA_results_'+str(datetime.today())[0:10]+'.csv','w')
        print("Finding High-D Neighborhood...")
        high_D_neighborhood = neighbors(data,k=cluster_size)
        print("Generating Embedding...")
        embedding = NDR(data=data,method=method,dim=latent_dim)
        print("Finding Low-D Neighborhood...")
        low_D_neighborhood = neighbors(embedding,k=cluster_size)
        print("Calculating Jaccard Distances...")
        jaccard_distances=[]
        for i in range(0,n_samples,1):
            jaccard_distances.append(jaccard(low_D_neighborhood[i,:],high_D_neighborhood[i,:]))
        trial =np.mean(jaccard_distances)
        run_array.append(trial)
    list_array.append(run_array)
print("Making numpy array")
nparray = np.asarray(list_array)
nparray = np.transpose(nparray)
col_labels= ['Embedded Dimension']+["# of NN: "+ str(p) for p in cluster_sizes]
print("Making DataFrame");
frame_of_data= pd.DataFrame(nparray, columns=col_labels)
print("Making.csv")
frame_of_data.to_csv("/home/shamuscooley/Research/manifold_learning/figures/supplement/normal_dist/data/supp2-2.csv")
print("All Done")
