
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
df= pd.read_csv("data/supp5.csv",index_col=0)
headers= df.keys()
fig = plt.figure(figsize=(11, 6))

ax1 = fig.add_subplot(111)
ax1.set_title("Supplemental Fig 5")
    # ax1.set_ylabel("Jaccard Distance")
ax1.set_ylim((0,1))
ax1.set_xlabel("Embedded Dimension")
ax1.set_ylabel("Avg Jaccard Distance")
df.plot(x=headers[0],y=headers[1:],ax=ax1, legend = True)

#ax1.set_ylim((0,1))
#plt.yticks([0,.25,.50,.75,1],fontsize=15)
#plt.xticks([0,1000,2000,3000,4000,5000],fontsize=15)
#plt.xlabel("")
plt.tight_layout()
# plt.legend((10,5,6,7,8,9))

plt.savefig("data/png_files/supp5.png")
