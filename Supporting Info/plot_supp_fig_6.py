import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

df = pd.read_csv('tsne_parameters_comparison.csv')
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

sub_cluster = 0
while sub_cluster < 30:
    sub_cluster_x_values = []
    sub_cluster_y_values = []
    index = 0
    while index < df.shape[0]:
        if df.iloc[index,5] == sub_cluster:
            sub_cluster_x_values.append(df.iloc[index,1])
            sub_cluster_y_values.append(df.iloc[index,2])
        index += 1
    ax1.scatter(sub_cluster_x_values,sub_cluster_y_values)
    sub_cluster += 1

sub_cluster = 0
while sub_cluster < 30:
    sub_cluster_x_values = []
    sub_cluster_y_values = []
    index = 0
    while index < df.shape[0]:
        if df.iloc[index,5] == sub_cluster:
            sub_cluster_x_values.append(df.iloc[index,3])
            sub_cluster_y_values.append(df.iloc[index,4])
        index += 1
    ax2.scatter(sub_cluster_x_values,sub_cluster_y_values)
    sub_cluster += 1

ax1.xaxis.set_major_formatter(NullFormatter())
ax1.yaxis.set_major_formatter(NullFormatter())
ax2.xaxis.set_major_formatter(NullFormatter())
ax2.yaxis.set_major_formatter(NullFormatter())
plt.tight_layout()
plt.savefig('supp_fig_6.png')
