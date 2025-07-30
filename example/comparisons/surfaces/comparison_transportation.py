#If the plot are completely random, please run the code again.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C, RBF, ExpSineSquared as Periodic
from policy_transportation.plot_utils import draw_error_band
from policy_transportation.utils import resample


# load the models
from policy_transportation.transportation.transportation import PolicyTransportation
from policy_transportation.transportation.laplacian_editing_transportation import LaplacianEditingTransportation as LET
from policy_transportation.transportation.kernelized_movement_primitives_transportation import KMP_transportation as KMP
import os

from policy_transportation.models.gaussian_process import GaussianProcess as GPR
from policy_transportation.models.torch.ensemble_neural_network import EnsembleNeuralNetwork as ENN
from policy_transportation.models.torch.ensemble_bijective_network import EnsembleBijectiveNetwork as EBN
from policy_transportation.models.locally_weighted_translations import Iterative_Locally_Weighted_Translations as ILWT

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
LET=LET(training_traj=X)
KMP=KMP(kernel=k_periodic, training_traj=X)

GPT=PolicyTransportation()
GPT.set_method(method=GPR(k_transport), is_residual=True)

LTW=PolicyTransportation()
LTW.set_method(method=ILWT(), is_residual=False)

MLP=PolicyTransportation()
MLP.set_method(method=ENN(num_epochs=200, n_estimators=10), is_residual=True)

BNT=PolicyTransportation()
BNT.set_method(method=EBN(num_epochs=200, n_estimators=10), is_residual=False)

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
    model.fit(source_distribution, target_distribution, do_scale=False, do_rotation=True)
    X1, std=model.transport(X, return_std=True)
    X1_list.append(X1)
    X_samples=model.sample_transportation(X)
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

