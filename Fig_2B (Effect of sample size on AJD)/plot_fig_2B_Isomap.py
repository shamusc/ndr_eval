
# df= pd.read_csv("~/Research/manifold_learning/figures/figure_2/data/2D/fig_2D_PCA.csv")
# headers= df.keys()
# fig = plt.figure(figsize=(15, 8))
#
# ax1 = fig.add_subplot(111)
# ax1.set_title("PCA with 10 Dimension Sphere")
# ax1.set_ylabel("Average Jaccard Distance")
# ax1.set_ylim((0,1))
# #fig.set_xlim((0,100))
#
# df.plot(x=headers[1],y=headers[2:],ax=ax1, legend=True)
# # plt.legend((10,5,6,7,8,9))
#
# plt.savefig('PCA.png')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import NullFormatter

# methods = ['t-SNE','PCA','IsoMap','ltsa_LLE','MDS','modified_LLE','UMAP','Spectral_Embedding','standard_LLE']


# for filename in os.listdir('data/'):

filename = 'IsoMap.csv'
df= pd.read_csv("data/"+filename)
headers= df.keys()
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(111)
# ax1.set_title(method+" with sample size 1000 and cluster size 20")
# ax1.set_ylabel("Average Jaccard Distance")
# ax1.set_xlabel("Embedding Dimension")
# ax1.xaxis.set_major_formatter(NullFormatter())
# ax1.yaxis.set_major_formatter(NullFormatter())
# ax1.set_ylim((0,1))
plt.yticks([.29,.31,.33,.35,.37],fontsize=30)
plt.xticks(fontsize=30)

#fig.set_xlim((0,100))
ax1.plot(df.iloc[45:55,1],df.iloc[45:55,2],linewidth=5)
ax1.plot(df.iloc[45:55,1],df.iloc[45:55,3],linewidth=5)
ax1.plot(df.iloc[45:55,1],df.iloc[45:55,4],linewidth=5)
ax1.plot(df.iloc[45:55,1],df.iloc[45:55,5],linewidth=5)
# df.plot(x=headers[1],y=headers[2:],ax=ax1, legend=False,linewidth=5)
plt.xlabel("")
# plt.legend((10,5,6,7,8,9))

plt.savefig('fig_2b_Isomap.eps',format='eps',dpi=1000)
