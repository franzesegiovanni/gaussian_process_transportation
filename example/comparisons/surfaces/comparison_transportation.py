#If the plot are completely random, please run the code again.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C, RBF, ExpSineSquared as Periodic
from policy_transportation.plot_utils import draw_error_band
from policy_transportation.utils import resample

# load the models
from policy_transportation.transportation.gaussian_process_transportation import GaussianProcessTransportation as GPT
from policy_transportation.transportation.multi_layer_perceptron_transportation import MLPTrasportation as MLP
from policy_transportation.transportation.diffeomorphic_transportation import DiffeomorphicTransportation as LTW
from policy_transportation.transportation.laplacian_editing_transportation import LaplacianEditingTransportation as LET
from policy_transportation.transportation.torch.ensemble_bijective_transport import Neural_Transport as BNT
from policy_transportation.transportation.kernelized_movement_primitives_transportation import KMP_transportation as KMP
import os

import warnings
warnings.filterwarnings("ignore")


# Load the demonstration and the source and target surface
script_path = str(os.path.dirname(__file__))
data =np.load(script_path+'/data/'+str('example0')+'.npz')
X=data['demo'] 
S=data['floor'] 
S1=data['newfloor']

X=resample(X, num_points=200)
source_distribution=resample(S, num_points=20)
target_distribution=resample(S1,num_points=20)

#%% Calculate deltaX
deltaX = np.zeros((len(X),2))
for j in range(len(X)-1):
    deltaX[j,:]=(X[j+1,:]-X[j,:])


# initialize the models
k_transport = C(constant_value=np.sqrt(0.1), constant_value_bounds=[0.1,2])  * RBF(10*np.ones(2), [5,500]) + WhiteKernel(0.0001)
k_periodic= kernel=C(0.1, constant_value_bounds=[0.1,2]) * Periodic(periodicity=1, periodicity_bounds=[1,1], length_scale=0.1, length_scale_bounds=[0.05, 1]) + WhiteKernel(0.00001, noise_level_bounds=[1e-5, 0.01])
GPT=GPT(kernel_transport=k_transport)
MLP=MLP()
LTW=LTW(num_iterations=10)
LET=LET()
BNT=BNT()
KMP=KMP(kernel=k_periodic)
methods=[KMP, LTW, MLP, LET, BNT, GPT]
names=["Kernelized Movement Primitives","Locally Weighted Translations", "Ensemble Neural Network", "Laplacian Editing", "Ensemble Neural Flows", "Gaussian Process Regression"]

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))
fig.subplots_adjust(wspace=0, hspace=0) 
# i=0
X1_list = []
std_list = []


i=0

for model , name in zip(methods, names):
    print("Fitting "+name+"...")
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
    # std=np.linalg.norm(std, axis=1)
    X_samples=model.sample_transportation()
    current_ax = ax[i // 3, i % 3]

    current_ax.scatter(target_distribution[:,0],target_distribution[:,1], color=[0,0,0], label="New Surface")
    current_ax.plot(X_samples[:,:,0].T, X_samples[:,:,1].T, alpha=0.5)
    draw_error_band(current_ax, X1[:,0], X1[:,1], err=2*std[:], facecolor= [255.0/256.0,140.0/256.0,0.0], edgecolor="none", alpha=.4, loop=True)
    current_ax.scatter(X1[:,0],X1[:,1], label="Tranported demonstration")
    current_ax.set_title(name, fontsize=18, fontweight='bold')
    current_ax.set_ylim(-20, 80)
    current_ax.set_xlim(-40, 40)
    current_ax.grid()
    # legend=current_ax.legend(loc='upper left', fontsize=12)
    i+=1
fig.tight_layout()

legend=current_ax.legend(loc='upper left', fontsize=12)

#Save figure
plt.savefig(script_path+'/figs/transportation_comparison.pdf', bbox_inches='tight')

plt.show()    

