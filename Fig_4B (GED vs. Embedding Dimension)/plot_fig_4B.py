import pandas as pd
import matplotlib.pyplot as plt

tSNE_df = pd.read_csv('data/seibert_GED.csv',index_col=0)
df = pd.read_csv('data/ged_seibert_full.csv',index_col=0)
random_df = pd.read_csv('data/ged_random_embedding.csv',index_col=0)
embedding_dimensions = df.iloc[:,0]
UMAP = df.iloc[:,1]
MDS = df.iloc[:,2]
Isomap = df.iloc[:,3]
PCA = df.iloc[:,4]
tSNE = tSNE_df.iloc[:,2]
random = random_df.iloc[:,1]

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(111)
ax1.set_ylim((0,4000))
# ax1.set_title("Graph Edit Distance between Minimum Spanning Trees (t-SNE)")
# ax1.set_ylabel("Graph Edit Distance")
# ax1.set_xlabel("Embedding Dimension")

plt.yticks([0,1000,2000,3000,4000,5000],fontsize=24)
plt.xticks([0,100,200,300,400,500],fontsize=24)

ax1.plot(embedding_dimensions,tSNE,label='t-SNE',linewidth=3)
ax1.plot(embedding_dimensions,UMAP,label='UMAP',linewidth=3)
ax1.plot(embedding_dimensions,MDS,label='MDS',linewidth=3)
ax1.plot(embedding_dimensions,Isomap,label='Isomap',linewidth=3)
ax1.plot(embedding_dimensions,PCA,label='PCA',linewidth=3)
ax1.plot(embedding_dimensions,random,label='Random',linewidth=3,linestyle=':')
# plt.legend()

plt.savefig('fig_4B.eps')
