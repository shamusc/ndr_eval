
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
df= pd.read_csv("/home/shamuscooley/Research/manifold_learning/figures/supplement/normal_dist/data/supp1.csv",index_col=0)
headers= df.keys()
fig = plt.figure(figsize=(11, 6))

i=1

for tech in  headers[1:]:
    ax1 = fig.add_subplot(330+i)
    ax1.set_title(tech)
    # ax1.set_ylabel("Jaccard Distance")
    ax1.set_ylim((0,1))
    ax1.set_xlabel("Embedding Dimension")
    ax1.set_ylabel("Avg Jaccard Distance")
    df.plot(x=headers[0],y=tech,ax=ax1)
    i+=1
#ax1.set_ylim((0,1))
#plt.yticks([0,.25,.50,.75,1],fontsize=15)
#plt.xticks([0,1000,2000,3000,4000,5000],fontsize=15)
#plt.xlabel("")
plt.tight_layout()
# plt.legend((10,5,6,7,8,9))

plt.savefig("/home/shamuscooley/Research/manifold_learning/figures/supplement/normal_dist/data/png_files/supp1.png")
