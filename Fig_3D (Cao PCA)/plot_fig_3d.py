import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import NullFormatter

df= pd.read_csv("data/cao_PCA.csv")
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111)

ax1.plot(df.iloc[:,0],df.iloc[:,1],label=df.columns[1],color='violet',linewidth=5)
# ax1.set_title("AJD of NDR embeddings on Siebert Data")
# ax1.set_ylabel("Average Jaccard Distance")
# ax1.set_xlabel("Embedding Dimension")
# ax1.xaxis.set_major_formatter(NullFormatter())
# ax1.yaxis.set_major_formatter(NullFormatter())
ax1.set_ylim((0,1))
plt.yticks([0,.25,.50,.75,1],fontsize=35)
plt.xticks([0,5000,10000,15000],fontsize=35)
plt.xlabel("")
# plt.legend()

plt.savefig('fig_3d.eps',format='eps',dpi=1000)
