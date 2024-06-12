#If the plot are completely random, please run the code again.
import numpy as np
import matplotlib.pyplot as plt
from policy_transportation import AffineTransform
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C, RBF
from policy_transportation.plot_utils import draw_error_band
from policy_transportation.utils import resample

from save_utils import save_array_as_latex
from compute_trajectories_divergence import kl_mvn, compute_distance, compute_distance_euclidean
# load the models
from policy_transportation.transportation.gaussian_process_transportation import GaussianProcessTransportation as GPT
from policy_transportation.transportation.multi_layer_perceptron_transportation import MLPTrasportation as MLP
from policy_transportation.transportation.random_forest_transportation import RFTrasportation as RFT
from policy_transportation.transportation.laplacian_editing_transportation import LaplacianEditingTransportation as LET
from policy_transportation.transportation.torch.ensemble_bijective_transport import Neural_Transport as BNT
from policy_transportation.transportation.kernelized_movement_primitives_transportation import KMP_transportation as KMP
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
k_transport = C(constant_value=np.sqrt(0.1))  * RBF(1*np.ones(2), [1,500]) + WhiteKernel(0.0001)
model=GPT(kernel_transport=k_transport)



fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
fig.subplots_adjust(wspace=0, hspace=0) 
# i=0
X1_list = []
std_list = []


model.source_distribution=source_distribution 
model.target_distribution=target_distribution
model.training_traj=X
model.training_delta=deltaX
model.fit_transportation()
model.apply_transportation()
X1=model.training_traj
X1_list.append(X1)
deltaX1=model.training_delta 
std=model.std
std_list.append(std)
X_samples=model.sample_transportation()

ax.scatter(target_distribution[:,0],target_distribution[:,1], color=[0,0,0], label="New Surface")
ax.plot(X_samples[:,:,0].T, X_samples[:,:,1].T, alpha=0.5)
draw_error_band(ax, X1[:,0], X1[:,1], err=2*std[:], facecolor= [255.0/256.0,140.0/256.0,0.0], edgecolor="none", alpha=.4, loop=True)
ax.scatter(X1[:,0],X1[:,1], label="Tranported demonstration")
# ax.set_title(name, fontsize=18, fontweight='bold')
ax.set_ylim(-20, 80)
ax.grid()
# legend=ax.legend(loc='upper left', fontsize=12)
# i+=1
fig.tight_layout()

legend=ax.legend(loc='upper left', fontsize=12)

#Save figure



plt.show()    

