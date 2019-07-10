
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.table as tb
import os
df= pd.read_csv("/home/shamuscooley/Research/manifold_learning/figures/supplement/normal_dist/data/grid_search.csv")

headers= df.keys()
i = 1;
#print([df[headers[1]] == iter])
fig = plt.figure(figsize=(15, 8))
for iter in [500.]:
    new_df= df.loc[df[headers[1]] == iter,headers[2]:]
    new_heads = new_df.keys()
    ax1 = fig.add_subplot(110 + i)
    ax1.set_title(headers[1] + " "+str(iter))
    ax1.table(cellText=new_df.values, colLabels=new_df.columns, loc='center')
    i+=1
plt.tight_layout()
plt.savefig('/home/shamuscooley/Research/manifold_learning/figures/supplement/normal_dist/data/png_files/tsne_table.png')

# i=1
#
# for tech in  headers[2:]:
#     ax1 = fig.add_subplot(310 + i)
#     ax1.set_title(tech)
#     ax1.set_ylabel("Predicted_Dimension")
#     ax1.set_ylim((0,100))
#     ax1.set_xlim((0,100))
#     df.plot(x=headers[1],y=tech,ax=ax1, legend=True)
#     i+=1
# plt.tight_layout()
# # plt.legend((10,5,6,7,8,9))
#
# plt.savefig('Latent_plot.png')
