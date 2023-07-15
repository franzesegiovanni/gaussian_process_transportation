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
results_df.pop(0)
results_area.pop(0) 
results_dtw.pop(0)
results_fde.pop(0)
results_fda.pop(0)  
results_df.pop(0)
results_area.pop(0) 
results_dtw.pop(0)
results_fde.pop(0)
results_fda.pop(0)  
results_df.pop(0)
results_area.pop(0) 
results_dtw.pop(0)
results_fde.pop(0)
results_fda.pop(0)  
# results_df.pop(-1)
# results_area.pop(-1) 
# results_dtw.pop(-1)
# results_fde.pop(-1)
# results_fda.pop(-1)
print(len(results_df))
df = pd.DataFrame(results_df)
df = df.set_axis(['GMM_4','GMM_5','GMM_6','GMM_7', 'GMM_8'], axis=0) 
df = df.transpose()

area = pd.DataFrame(results_area)
area = area.set_axis(['GMM_4','GMM_5','GMM_6','GMM_7', 'GMM_8'], axis=0) 
area = area.transpose()

dtw = pd.DataFrame(results_dtw)
dtw = dtw.set_axis(['GMM_4','GMM_5','GMM_6','GMM_7', 'GMM_8'], axis=0) 
dtw = dtw.transpose()

fde = pd.DataFrame(results_fde)
fde = fde.set_axis(['GMM_4','GMM_5','GMM_6','GMM_7', 'GMM_8'], axis=0) 
fde = fde.transpose()

fda = pd.DataFrame(results_fda)
fda = fda.set_axis(['GMM_4','GMM_5','GMM_6','GMM_7', 'GMM_8'], axis=0)
fda = fda.transpose()
# #%%
# data =np.load('results_transportation_way_point.npz', allow_pickle=True)
# results_df= list(data['results_df'])
# results_area=list(data['results_area']) 
# results_dtw=list(data['results_dtw'])
# results_fde=list(data['results_fde'])
# results_fda=list(data['results_fad'])
    
# df_gp = pd.DataFrame(results_df)
# df_gp = df_gp.set_axis(['GP_wp'], axis=1)
# df = pd.concat([df_gp, df], axis=1)

# area_gp = pd.DataFrame(results_area)
# area_gp = area_gp.set_axis(['GP_wp'], axis=1)
# area = pd.concat([area_gp, area], axis=1)


# dtw_gp = pd.DataFrame(results_dtw)
# dtw_gp = dtw_gp.set_axis(['GP_wp'], axis=1)
# dtw = pd.concat([dtw_gp, dtw], axis=1)


# fde_gp = pd.DataFrame(results_fde)
# fde_gp = fde_gp.set_axis(['GP_wp'], axis=1)
# fde = pd.concat([fde_gp, fde], axis=1)

# fda_gp = pd.DataFrame(results_fda)
# fda_gp = fda_gp.set_axis(['GP_wp'], axis=1)
# fda = pd.concat([fda_gp, fda], axis=1)
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

# data =np.load('results_affine.npz', allow_pickle=True)
# results_df= list(data['results_df'])
# results_area=list(data['results_area']) 
# results_dtw=list(data['results_dtw'])
# results_fde=list(data['results_fde'])
# results_fda=list(data['results_fad'])
    
# df_gp = pd.DataFrame(results_df)
# df_gp = df_gp.set_axis(['DMP2'], axis=1)
# df = pd.concat([df_gp, df], axis=1)

# area_gp = pd.DataFrame(results_area)
# area_gp = area_gp.set_axis(['DMP2'], axis=1)
# area = pd.concat([area_gp, area], axis=1)

# dtw_gp = pd.DataFrame(results_dtw)
# dtw_gp = dtw_gp.set_axis(['DMP2'], axis=1)
# dtw = pd.concat([dtw_gp, dtw], axis=1)

# fde_gp = pd.DataFrame(results_fde)
# fde_gp = fde_gp.set_axis(['DMP2'], axis=1)
# fde = pd.concat([fde_gp, fde], axis=1)

# fda_gp = pd.DataFrame(results_fda)
# fda_gp = fda_gp.set_axis(['DMP2'], axis=1)
# fda = pd.concat([fda_gp, fda], axis=1)

plt.figure()
sns.boxplot(data=df, orient='v')
sns.stripplot(data=df, color="black", jitter=True, size=3)
plt.title('Frenet')
plt.figure()
sns.boxplot(data=area, orient='v')
sns.stripplot(data=area, color="black", jitter=True, size=3)
plt.title('Area between the curves')
plt.figure()
sns.boxplot(data=dtw, orient='v')
sns.stripplot(data=dtw, color="black", jitter=True, size=3)
plt.title('Dynamic Time Warping')
plt.figure()
sns.boxplot(data=fde, orient='v')
sns.stripplot(data=fde, color="black", jitter=True, size=3)
plt.title('Final Distance Euclidean')

plt.figure()
sns.boxplot(data=fda, orient='v')
sns.stripplot(data=fda, color="black", jitter=True, size=3)
plt.title('Final Distance Angular')


print('Frenet')
ranking=generate_ranking(df.dropna())
print(ranking)
print('Area')
ranking=generate_ranking(area.dropna())
print(ranking)
print('DTW')
ranking=generate_ranking(dtw.dropna())
print(ranking)
print('FDE')
ranking=generate_ranking(fde.dropna())
print(ranking)
print('FDA')
ranking=generate_ranking(fda.dropna())
print(ranking)
# OUT OF DISTRIBUTION
data =np.load('results_sota_out_distribution.npz')
results_fde=data['results_fde']
results_fda=data['results_fad']

fde=pd.DataFrame(results_fde)
fde = fde.set_axis(['GMM_9'], axis=1) 
fda=pd.DataFrame(results_fda)
fda = fda.set_axis(['GMM_9'], axis=1)


# data =np.load('results_transportation_way_point_out_distribution.npz', allow_pickle=True)
# results_fde=list(data['results_fde'])
# results_fda=list(data['results_fad'])

# fde_gp = pd.DataFrame(results_fde)
# fde_gp = fde_gp.set_axis(['GP_wp'], axis=1) 
# fda_gp = pd.DataFrame(results_fda)
# fda_gp = fda_gp.set_axis(['GP_wp'], axis=1)
# fde = pd.concat([fde_gp, fde], axis=1)
# fda = pd.concat([fda_gp, fda], axis=1)

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

plt.figure()
sns.boxplot(data=fde, orient='v')
sns.stripplot(data=fde, color="black", jitter=True, size=3)
plt.title('Final Euclidean Displacement Out Distribution')

plt.figure()
sns.boxplot(data=fda, orient='v')
sns.stripplot(data=fda, color="black", jitter=True, size=3)
plt.title('Final Angle Dispacement  Out Distribution')


# print("OUT of distribution")
# print('FDE')
# statistical_test(fde)
# print('FDA')
# statistical_test(fda)

plt.show()

