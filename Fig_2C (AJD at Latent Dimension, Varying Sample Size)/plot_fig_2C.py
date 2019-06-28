
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
df= pd.read_csv("figure_2C_raw_data.csv",index_col=0)
headers= df.keys()
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(111)


for tech in  [4,5,6,7,8]:

    # ax1.set_title(tech)
    # ax1.set_ylabel("Jaccard Distance")
    ax1.set_ylim((0,1))
    df.plot(x=headers[0],y=headers[tech],ax=ax1, legend=True,linewidth=3)
ax1.set_ylim((0,1))
plt.yticks([0,.25,.50,.75,1],fontsize=15)
plt.xticks([0,1000,2000,3000,4000,5000],fontsize=15)
plt.xlabel("")
plt.tight_layout()
# plt.legend((10,5,6,7,8,9))

plt.savefig('Figure_2C.eps',format='eps',dpi=1000)
