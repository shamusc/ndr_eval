
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import NullFormatter

methods = ['t-SNE','IsoMap','MDS','UMAP','Spectral_Embedding','Diffusion_Map']
for method in methods:
    df= pd.read_csv("data/"+method+".csv")
    headers= df.keys()
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)
    ax1.set_ylim((0,1))
    plt.yticks([0,.25,.50,.75,1],fontsize=35)
    plt.xticks([0,25,50,75,100],fontsize=35)
    df.plot(x=headers[1],y=headers[2:],ax=ax1, legend=False,linewidth=5)
    plt.xlabel("")
    plt.savefig(method+'.eps',format='eps',dpi=1000)
