
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df= pd.read_csv("figure_4A_data.csv",index_col=0)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(111)
cluster_sizes = [2,10,20,50,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1778]
percentages = []
for item in cluster_sizes:
    percentage = (item/1778)*100
    percentages.append(percentage)
# cluster_sizes = []
i = 0
while i < df.shape[0]:
    ax1.set_ylim((0,1))
    ax1.plot(percentages,df.iloc[i,:],label = df.index[i],linewidth=3)
    i += 1
ax1.set_ylim((0,1))
# plt.title("Neighborhood Size vs. Distortion \n Siebert Data \n Endodermal Epithelial Stem Cell")
# plt.legend()
plt.yticks([0,.25,.50,.75,1],fontsize=24)
plt.xticks([0,25,50,75,100],fontsize=24)
# plt.xlabel("Jaccard Nearest Neighbors")
# plt.ylabel("Average Jaccard Distance")
plt.tight_layout()
# plt.legend((10,5,6,7,8,9))

plt.savefig('Figure_4A.eps',format='eps',dpi=1000)
