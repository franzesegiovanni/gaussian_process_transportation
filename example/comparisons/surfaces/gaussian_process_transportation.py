#If the plot are completely random, please run the code again.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C, RBF
from policy_transportation.plot_utils import draw_error_band
from policy_transportation.utils import resample

from policy_transportation.transportation.gaussian_process_transportation import GaussianProcessTransportation as GPT
import os

import warnings
warnings.filterwarnings("ignore")


# Load the demonstration and the source and target surface
script_path = str(os.path.dirname(__file__))
data =np.load(script_path+'/data/'+str('example')+'.npz')
X=data['demo'] 
S=data['floor'] 
S1=data['newfloor']

X=resample(X, num_points=400)
source_distribution=resample(S, num_points=100)
target_distribution=resample(S1,num_points=100)

#%% Calculate deltaX
deltaX = np.zeros((len(X),2))
for j in range(len(X)-1):
    deltaX[j,:]=(X[j+1,:]-X[j,:])


# initialize the models
k_transport = C(constant_value=np.sqrt(0.1), constant_value_bounds=[0.1,2])  * RBF(1*np.ones(2), [10,500]) + WhiteKernel(0.0001)
model=GPT(kernel_transport=k_transport)



fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
fig.subplots_adjust(wspace=0, hspace=0) 


model.source_distribution=source_distribution 
model.target_distribution=target_distribution
model.training_traj=X
model.training_delta=deltaX
model.fit_transportation()
model.apply_transportation()
X1=model.training_traj
deltaX1=model.training_delta 
std=model.std
X_samples=model.sample_transportation()

ax.scatter(target_distribution[:,0],target_distribution[:,1], color=[0,0,0], label="New Surface")
ax.plot(X_samples[:,:,0].T, X_samples[:,:,1].T, alpha=0.5)
draw_error_band(ax, X1[:,0], X1[:,1], err=2*std[:], facecolor= [255.0/256.0,140.0/256.0,0.0], edgecolor="none", alpha=.4, loop=True)
ax.scatter(X1[:,0],X1[:,1], label="Tranported demonstration")
ax.set_ylim(-20, 80)
ax.grid()
fig.tight_layout()

legend=ax.legend(loc='upper left', fontsize=12)

plt.show()    

