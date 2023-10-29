import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from statistical_tool import generate_ranking
import os
from utils import load_in_table
source_path = str(os.path.dirname(__file__))
# Load the data from the files
df1, area1, dtw1, fde1, fda1 = load_in_table(source_path + '/results/tpgmm_dataset.npz')
df2, area2, dtw2, fde2, fda2 = load_in_table(source_path + '/results/gpt_dataset.npz')
df3, area3, dtw3, fde3, fda3 = load_in_table(source_path + '/results/dmp_dataset.npz')
df4, area4, dtw4, fde4, fda4 = load_in_table(source_path + '/results/hmm_dataset.npz')

#concatenate the dataframes
df = pd.concat([df1, df2, df3, df4], axis=1)
area = pd.concat([area1, area2, area3, area4], axis=1)
dtw = pd.concat([dtw1, dtw2, dtw3, dtw4], axis=1)
fde = pd.concat([fde1, fde2, fde3, fde4], axis=1)
fda = pd.concat([fda1, fda2, fda3, fda4], axis=1)

print('Frenet')
ranking_frenet=generate_ranking(df)
print(ranking_frenet)
print('Area')
ranking_area=generate_ranking(area)
print(ranking_area)
print('DTW')
ranking_dtw=generate_ranking(dtw)
print(ranking_dtw)
print('FDE')
ranking_fde=generate_ranking(fde)
print(ranking_fde)
print('FDA')
ranking_fda=generate_ranking(fda)
print(ranking_fda)

# fig, axes = plt.subplots(1, 5, figsize=(47, 5), constrained_layout=True)

import matplotlib.pyplot as plt
import seaborn as sns

# dataframes = [df, fde, fda]
# rankings = [ranking_frenet, ranking_fde, ranking_fda]
# titles = ['Frenet Distance', 'Final Position Error', 'Final Orientation Error']
dataframes = [df, area, dtw, fde, fda]
rankings = [ranking_frenet, ranking_area, ranking_dtw, ranking_fde, ranking_fda]
titles = ['Frenet Distance', 'Area between the curves', 'Dynamic Time Warping', 'Final Position Error', 'Final Orientation Error']

fig, axes = plt.subplots(1, len(rankings), figsize=(47, 5), constrained_layout=True)

for i, (data, ranking, title) in enumerate(zip(dataframes, rankings, titles)):
    data = data.reindex(columns=ranking['group'].tolist())
    sns.boxplot(data=data, orient='v', ax=axes[i])
    sns.stripplot(data=data, color="black", jitter=True, size=3, ax=axes[i])
    for j, number in enumerate(ranking['rank'].tolist()):
        axes[i].text(j, data.max().max(), str(number), ha='center', va='bottom', fontweight='bold', fontsize=14)
    axes[i].set_title(title, fontsize=20, fontweight='bold')
    axes[i].set_xticklabels(data.columns.to_list(), rotation=90, fontsize=14, fontweight='bold')


# Save the figure without extra space on the sides
plt.savefig(source_path + '/figs/Box_plot_complete.pdf', bbox_inches='tight')


dataframes = [df, fde, fda]
rankings = [ranking_frenet, ranking_fde, ranking_fda]
titles = ['Frenet Distance', 'Final Position Error', 'Final Orientation Error']

fig, axes = plt.subplots(1, len(rankings), figsize=(12, 6), constrained_layout=True)

for i, (data, ranking, title) in enumerate(zip(dataframes, rankings, titles)):
    data = data.reindex(columns=ranking['group'].tolist())
    sns.boxplot(data=data, orient='v', ax=axes[i])
    sns.stripplot(data=data, color="black", jitter=True, size=3, ax=axes[i])
    for j, number in enumerate(ranking['rank'].tolist()):
        axes[i].text(j, data.max().max(), str(number), ha='center', va='bottom', fontweight='bold', fontsize=14)
    axes[i].set_title(title, fontsize=20, fontweight='bold')
    axes[i].set_xticklabels(data.columns.to_list(), rotation=90, fontsize=14, fontweight='bold')


# Save the figure without extra space on the sides
plt.savefig(source_path + '/figs/Box_plot_short.pdf', bbox_inches='tight')


plt.show()

