import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

df = pd.read_csv('Siebert_tsne_X_Y_Labels.csv',index_col=0)

cluster = 2

i = 0
cluster_indices = []
while i < df.shape[0]:
    if df.iloc[i,2] == cluster:
        cluster_indices.append(i)
    i += 1
fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(121)

ax1.scatter(df.iloc[:,0],df.iloc[:,1],color='grey',marker='.')
ax1.scatter(df.iloc[cluster_indices,0],df.iloc[cluster_indices,1],color='green',marker='.')
ax1.xaxis.set_major_formatter(NullFormatter())
ax1.yaxis.set_major_formatter(NullFormatter())


ax2 = fig.add_subplot(122)
ax2.scatter(df.iloc[:,0],df.iloc[:,1],color='grey',marker='.')
sub_cluster = 0

while sub_cluster < 30:
    sub_cluster_indices = []
    for point in cluster_indices:
        if df.iloc[point,3] == sub_cluster:
            sub_cluster_indices.append(point)
    ax2.scatter(df.iloc[sub_cluster_indices,0],df.iloc[sub_cluster_indices,1],marker='.')
    sub_cluster += 1
ax2.xaxis.set_major_formatter(NullFormatter())
ax2.yaxis.set_major_formatter(NullFormatter())

plt.savefig('fig_4d.eps',format='eps',dpi=1000)
plt.tight_layout()
