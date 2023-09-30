import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from statistical_tool import generate_ranking
from utils import load_in_table_ood as load_in_table
import os
import matplotlib.pyplot as plt
import seaborn as sns

source_path = str(os.path.dirname(__file__))


# Load the data from the files
fde1, fda1 = load_in_table(source_path + '/results/tpgmm_out_distribution.npz')
fde2, fda2 = load_in_table(source_path + '/results/gpt_out_distribution.npz')
fde3, fda3 = load_in_table(source_path + '/results/dmp_out_distribution.npz')
fde4, fda4 = load_in_table(source_path + '/results/hmm_out_distribution.npz')

#concatenate the dataframes
fde = pd.concat([fde1, fde2, fde3, fde4], axis=1)
fda = pd.concat([fda1, fda2, fda3, fda4], axis=1)

print('FDE')
ranking_fde=generate_ranking(fde)
print(ranking_fde)
print('FDA')
ranking_fda=generate_ranking(fda)
print(ranking_fda)


dataframes = [ fde, fda]
rankings = [ranking_fde, ranking_fda]
titles = [ 'Final Position Error', 'Final Orientation Error']

fig, axes = plt.subplots(1, len(rankings), figsize=(20, 5), constrained_layout=True)

for i, (data, ranking, title) in enumerate(zip(dataframes, rankings, titles)):
    data = data.reindex(columns=ranking['group'].tolist())
    sns.boxplot(data=data, orient='v', ax=axes[i])
    sns.stripplot(data=data, color="black", jitter=True, size=3, ax=axes[i])
    for j, number in enumerate(ranking['rank'].tolist()):
        axes[i].text(j, data.max().max(), str(number), ha='center', va='bottom', fontweight='bold', fontsize=14)
    axes[i].set_title(title, fontsize=20, fontweight='bold')
    axes[i].set_xticklabels(data.columns.to_list(), rotation=90, fontsize=14, fontweight='bold')


plt.savefig(source_path + '/figs/Box_plot_ood.png', bbox_inches='tight')
plt.show()