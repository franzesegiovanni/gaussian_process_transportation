import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
data =np.load('results_sota.npz')
results_df=data['results_df']
results_area=data['results_area'] 
results_dtw=data['results_dtw']
results_fde=data['results_fde']
    

results_df_mean=np.mean(results_df, axis=2)
results_area_mean=np.mean(results_area, axis=2)
results_dtw_mean=np.mean(results_dtw, axis=2)
results_fde_mean=np.mean(results_fde, axis=2)

df = pd.DataFrame(results_df_mean[4:,:])
df = df.transpose()
# Grouped violinplot
sns.violinplot(data=df, orient='v')

# sns.violinplot(x="day", y="total_bill", hue="smoker", data=df, palette="Pastel1")
plt.show()