import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from statistical_tool import generate_ranking
data =np.load('results_sota_dataset.npz', allow_pickle=True)
results_df=list(data['results_df'])
results_area=list(data['results_area']) 
results_dtw=list(data['results_dtw'])
results_fde=list(data['results_fde'])
results_fda=list(data['results_fad'])   
for i in range(4):
    results_df.pop(0)
    results_area.pop(0) 
    results_dtw.pop(0)
    results_fde.pop(0)
    results_fda.pop(0)  
results_df.pop(-1)
results_area.pop(-1) 
results_dtw.pop(-1)
results_fde.pop(-1)
results_fda.pop(-1)
print(len(results_df))
df = pd.DataFrame(results_df)
df = df.set_axis(['GMM5','GMM6','GMM7'], axis=0) 
df = df.transpose()

area = pd.DataFrame(results_area)
area = area.set_axis(['GMM5','GMM6','GMM7'], axis=0) 
area = area.transpose()

dtw = pd.DataFrame(results_dtw)
dtw = dtw.set_axis(['GMM5','GMM6','GMM7'], axis=0) 
dtw = dtw.transpose()

fde = pd.DataFrame(results_fde)
fde = fde.set_axis(['GMM5','GMM6','GMM7'], axis=0) 
fde = fde.transpose()

fda = pd.DataFrame(results_fda)
fda = fda.set_axis(['GMM5','GMM6','GMM7'], axis=0)
fda = fda.transpose()
#%%
data =np.load('results_transportation.npz', allow_pickle=True)
results_df= list(data['results_df'])
results_area=list(data['results_area']) 
results_dtw=list(data['results_dtw'])
results_fde=list(data['results_fde'])
results_fda=list(data['results_fad'])
    
df_gp = pd.DataFrame(results_df)
df_gp = df_gp.set_axis(['GP'], axis=1)
df = pd.concat([df_gp, df], axis=1)

area_gp = pd.DataFrame(results_area)
area_gp = area_gp.set_axis(['GP'], axis=1)
area = pd.concat([area_gp, area], axis=1)


dtw_gp = pd.DataFrame(results_dtw)
dtw_gp = dtw_gp.set_axis(['GP'], axis=1)
dtw = pd.concat([dtw_gp, dtw], axis=1)


fde_gp = pd.DataFrame(results_fde)
fde_gp = fde_gp.set_axis(['GP'], axis=1)
fde = pd.concat([fde_gp, fde], axis=1)

fda_gp = pd.DataFrame(results_fda)
fda_gp = fda_gp.set_axis(['GP'], axis=1)
fda = pd.concat([fda_gp, fda], axis=1)

data =np.load('results_affine.npz', allow_pickle=True)
results_df= list(data['results_df'])
results_area=list(data['results_area']) 
results_dtw=list(data['results_dtw'])
results_fde=list(data['results_fde'])
results_fda=list(data['results_fad'])
    
df_gp = pd.DataFrame(results_df)
df_gp = df_gp.set_axis(['DMP'], axis=1)
df = pd.concat([df_gp, df], axis=1)

area_gp = pd.DataFrame(results_area)
area_gp = area_gp.set_axis(['DMP'], axis=1)
area = pd.concat([area_gp, area], axis=1)

dtw_gp = pd.DataFrame(results_dtw)
dtw_gp = dtw_gp.set_axis(['DMP'], axis=1)
dtw = pd.concat([dtw_gp, dtw], axis=1)

fde_gp = pd.DataFrame(results_fde)
fde_gp = fde_gp.set_axis(['DMP'], axis=1)
fde = pd.concat([fde_gp, fde], axis=1)

fda_gp = pd.DataFrame(results_fda)
fda_gp = fda_gp.set_axis(['DMP'], axis=1)
fda = pd.concat([fda_gp, fda], axis=1)

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

fig, axes = plt.subplots(1, 5, figsize=(47, 5), constrained_layout=True)
df=df.reindex(columns=ranking_frenet['group'].tolist())
sns.boxplot(data=df, orient='v', ax=axes[0])
sns.stripplot(data=df, color="black", jitter=True, size=3, ax=axes[0])
for i, number in enumerate(ranking_frenet['rank'].tolist()):
    axes[0].text(i, df.max().max(), str(number),
            ha='center', va='bottom', fontweight='bold', fontsize=14)
axes[0].set_title('Frenet Distance', fontsize=20, fontweight='bold')
axes[0].set_xticklabels(df.columns.to_list(), rotation=90, fontsize=14, fontweight='bold')
# fig, ax = plt.subplots()
area=area.reindex(columns=ranking_area['group'].tolist())
sns.boxplot(data=area, orient='v', ax=axes[1])
sns.stripplot(data=area, color="black", jitter=True, size=3, ax=axes[1])
for i, number in enumerate(ranking_area['rank'].tolist()):
    axes[1].text(i, area.max().max(), str(number),
            ha='center', va='bottom', fontweight='bold', fontsize=14)
axes[1].set_title('Area between the curves', fontsize=20, fontweight='bold')
axes[1].set_xticklabels(area.columns.to_list(), rotation=90, fontsize=14, fontweight='bold')
# fig, ax = plt.subplots()
dtw=dtw.reindex(columns=ranking_dtw['group'].tolist())
sns.boxplot(data=dtw, orient='v', ax=axes[2])
sns.stripplot(data=dtw, color="black", jitter=True, size=3, ax=axes[2])
for i, number in enumerate(ranking_dtw['rank'].tolist()):
    axes[2].text(i, dtw.max().max(), str(number),
            ha='center', va='bottom', fontweight='bold', fontsize=14)
axes[2].set_title('Dynamic Time Warping', fontsize=20, fontweight='bold')
axes[2].set_xticklabels(dtw.columns.to_list(), rotation=90, fontsize=14, fontweight='bold')
# fig, ax = plt.subplots()
fde=fde.reindex(columns=ranking_fde['group'].tolist())
sns.boxplot(data=fde, orient='v', ax=axes[3])
sns.stripplot(data=fde, color="black", jitter=True, size=3, ax=axes[3])
for i, number in enumerate(ranking_fde['rank'].tolist()):
    axes[3].text(i, fde.max().max(), str(number),
            ha='center', va='bottom', fontweight='bold', fontsize=14)
axes[3].set_title('Final Position Error', fontsize=20, fontweight='bold')
axes[3].set_xticklabels(fde.columns.to_list(), rotation=90, fontsize=14, fontweight='bold')


# fig, ax = plt.subplots()
fda=fda.reindex(columns=ranking_fda['group'].tolist())
sns.boxplot(data=fda, orient='v', ax=axes[4])
sns.stripplot(data=fda, color="black", jitter=True, size=3, ax=axes[4])
for i, number in enumerate(ranking_fda['rank'].tolist()):
    axes[4].text(i, fda.max().max(), str(number),
            ha='center', va='bottom', fontweight='bold', fontsize=14)
axes[4].set_title('Final Orientation Error', fontsize=20, fontweight='bold')
axes[4].set_xticklabels(fda.columns.to_list(), rotation=90, fontsize=14, fontweight='bold')


# Save the figure without extra space on the sides
plt.savefig('Box_plot.png', bbox_inches='tight')

# OUT OF DISTRIBUTION
data =np.load('results_sota_out_distribution.npz')
results_fde=data['results_fde']
results_fda=data['results_fad']

fde=pd.DataFrame(results_fde)
fde = fde.set_axis(['GMM_9'], axis=1) 
fda=pd.DataFrame(results_fda)
fda = fda.set_axis(['GMM_9'], axis=1)


data =np.load('results_transportation_out_distribution.npz', allow_pickle=True)
results_fde=list(data['results_fde'])
results_fda=list(data['results_fad'])

fde_gp = pd.DataFrame(results_fde)
fde_gp = fde_gp.set_axis(['GP'], axis=1) 
fda_gp = pd.DataFrame(results_fda)
fda_gp = fda_gp.set_axis(['GP'], axis=1)
fde = pd.concat([fde_gp, fde], axis=1)
fda = pd.concat([fda_gp, fda], axis=1)

data =np.load('results_affine_out_distribution.npz', allow_pickle=True)
results_fde=list(data['results_fde'])
results_fda=list(data['results_fad'])

fde_gp = pd.DataFrame(results_fde)
fde_gp = fde_gp.set_axis(['DMP'], axis=1) 
fda_gp = pd.DataFrame(results_fda)
fda_gp = fda_gp.set_axis(['DMP'], axis=1)
fde = pd.concat([fde_gp, fde], axis=1)
fda = pd.concat([fda_gp, fda], axis=1)

print("OUT of distribution")
print('FDE')
ranking_fde=generate_ranking(fde)
print(ranking_fde)
print('FDA')
ranking_fda=generate_ranking(fda)
print(ranking_fda)

fig, ax = plt.subplots()
fde=fde.reindex(columns=ranking_fde['group'].tolist())
sns.boxplot(data=fde, orient='v')
sns.stripplot(data=fde, color="black", jitter=True, size=3)
for i, number in enumerate(ranking_fde['rank'].tolist()):
    ax.text(i, fde.max().max(), str(number),
            ha='center', va='bottom', fontweight='bold')
plt.title('Final Euclidean Displacement Out Distribution')

fig, ax = plt.subplots()
fda=fda.reindex(columns=ranking_fda['group'].tolist())
sns.boxplot(data=fda, orient='v')
sns.stripplot(data=fda, color="black", jitter=True, size=3)
for i, number in enumerate(ranking_fda['rank'].tolist()):
    ax.text(i, fda.max().max(), str(number),
            ha='center', va='bottom', fontweight='bold')
plt.title('Final Angle Dispacement  Out Distribution')



plt.show()

