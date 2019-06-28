import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import NullFormatter

df= pd.read_csv("data/siebert.csv")
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111)
color_dict = {
    5:'red',
    6:'blue',
    7:'brown',
    8:'orange',
    9:'violet',
    10:'grey'
    }
for i in [6,7,8,9,10]:
    ax1.plot(df.iloc[:,0],df.iloc[:,i],label=df.columns[i],color=color_dict[i],linewidth=5)
    # ax1.set_title("AJD of NDR embeddings on Siebert Data")
    # ax1.set_ylabel("Average Jaccard Distance")
    # ax1.set_xlabel("Embedding Dimension")
    # ax1.xaxis.set_major_formatter(NullFormatter())
    # ax1.yaxis.set_major_formatter(NullFormatter())
    ax1.set_ylim((0,1))
    plt.yticks([0,.25,.50,.75,1],fontsize=35)
    plt.xticks([0,25,50,75,100],fontsize=35)
    plt.xlabel("")
    # plt.legend()

plt.savefig('fig_3a.eps',format='eps',dpi=1000)
