import pandas as pd
import numpy as np
from sklearn import manifold
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import umap
import random
import os
from pydiffmap import diffusion_map

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
                a = random.normal(0,1)
            data[i,j]=a
            j += 1
        norm = np.linalg.norm(data[i])
        if noise == False:# making each vector into sphere
            data[i] = data[i]/norm
        if noise==True:
            noise_term = (random.uniform(-1,1) * noise_amplitude)
            #print(noise_term)
            data[i] = (data[i]/norm) + noise_term
        i += 1
    if comb_noise == True:
        for num in range(0, n_samples):
            for zero_dim in range(n_dimensions, k_space):
                data[num,zero_dim]= (random.uniform(-1,1) * noise_amplitude)
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
        embedding = manifold.TSNE(n_components=dim, init='pca',method='exact')\
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

for method in methods:
    methods = ['Diffusion_Map','standard_LLE','ltsa_LLE','modified_LLE','Spectral_Embedding','IsoMap','t-SNE','MDS','UMAP','PCA']
    sample_sizes = range(500,2500,500)
    list_array =[]
    for latent_dim in range(1,50,1):
        run_array = [latent_dim]
        for n_samples in sample_sizes:
            data = hypersphere(n_dimensions=7,n_samples=n_samples,k_space=100)
            cluster_size = int(n_samples/10)
            print("Data has shape: " + str(data.shape))
            # file_object = open('/Users/shamuscooley/GradSchool/Research/manifold/data/Siebert Data/wholegenome/enEp_SC1_ManDA_results_'+str(datetime.today())[0:10]+'.csv','w')
            print("Finding High-D Neighborhood...")
            high_D_neighborhood = neighbors(data,k=cluster_size)
            print("Generating Embedding...")
            embedding = NDR(data=data,method=method,dim=latent_dim,n_neighbors=cluster_size)
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
    col_labels= ['Latent Dimension']+["Sample Size: "+str(p) for p in sample_sizes]
    print("Making DataFrame");
    frame_of_data= pd.DataFrame(nparray, columns=col_labels)
    print("Making.csv")
    frame_of_data.to_csv("data/"+method+".csv")
print("All Done")
