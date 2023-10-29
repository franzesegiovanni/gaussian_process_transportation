#If the plot are completely random, please run the code again.
import numpy as np
import matplotlib.pyplot as plt
from policy_transportation import AffineTransform
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from policy_transportation.plot_utils import draw_error_band
from policy_transportation.utils import resample

from save_utils import save_array_as_latex
from compute_trajectories_divergence import kl_mvn, compute_distance, compute_distance_euclidean
# load the models
from policy_transportation.transportation.gaussian_process_transportation import GaussianProcessTransportation as GPT
from policy_transportation.transportation.multi_layer_perceptron_transportation import MLPTrasportation as MLP
from policy_transportation.transportation.random_forest_transportation import RFTrasportation as RFT
from policy_transportation.transportation.laplacian_editing_transportation import LaplacianEditingTransportation as LET
from policy_transportation.transportation.torch.ensamble_bijective_transport import Neural_Transport as BNT
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
Affine=AffineTransform()
k_transport = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(1), [1,10], nu=2.5) + WhiteKernel(0.0001)
GPT=GPT(kernel_transport=k_transport)
MLP=MLP()
RFT=RFT()
LET=LET()
BNT=BNT()

methods=[RFT, MLP, LET, BNT, GPT]
names=["Esamble Random Forest", "Ensamble Neural Network", "Laplacian Editing", "Ensamble Neural Flows", "Gaussian Process Regression"]


fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))
fig.subplots_adjust(wspace=0, hspace=0) 
i=0
X1_list = []
std_list = []

print("Fitting Linear Transformation...")
Affine.fit(source_distribution, target_distribution)
X1=Affine.predict(X)
X1_list.append(X1)
eps=1e-5
std_list.append(eps*np.ones_like(X1))
current_ax = ax[0,0]

current_ax.scatter(target_distribution[:,0],target_distribution[:,1], color=[0,0,0], label="New Surface")
current_ax.scatter(X1[:,0],X1[:,1], label="Tranported demonstration")
current_ax.set_title("Affine Transformation", fontsize=18, fontweight='bold')
current_ax.set_ylim(-20, 80)
current_ax.grid()
i+=1

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
    current_ax.grid()
    # legend=current_ax.legend(loc='upper left', fontsize=12)
    i+=1
fig.tight_layout()

legend=current_ax.legend(loc='upper left', fontsize=12)

#Save figure
plt.savefig(script_path+'/figs/transportation_comparison.pdf', bbox_inches='tight')

divergence=np.zeros((len(X1_list), len(X1_list)))

for i in range(len(X1_list)):
    for j in range(0, len(X1_list)):
        divergence[i,j]=kl_mvn((X1_list[i][:,0], np.diag(std_list[i][:,0]**2)), (X1_list[j][:,0], np.diag(std_list[j][:,0]**2)))+kl_mvn((X1_list[i][:,1], np.diag(std_list[i][:,1]**2)), (X1_list[j][:,1], np.diag(std_list[j][:,1]**2)))

fig, ax = plt.subplots()

# Create the table
table = ax.table(cellText=divergence,
                 cellLoc='center',
                 loc='center')    

ax.set_title("KL Divergence between transported demonstrations")

save_array_as_latex(divergence, script_path+ '/results/divergence.txt')
distance_euclidean=np.zeros((len(X1_list), len(X1_list)))
distance=np.zeros((len(X1_list), len(X1_list)))
for i in range(len(X1_list)):
    for j in range(0, len(X1_list)):
        distance[i,j]=compute_distance(X1_list[i], X1_list[j], std_list[i], std_list[j])


fig, ax = plt.subplots()

# Create the table
table = ax.table(cellText=distance,
                 cellLoc='center',
                 loc='center')    

ax.set_title("Distribution Distance between transported demonstrations")
save_array_as_latex(distance, script_path + '/results/distribution_distance.txt')

for i in range(len(X1_list)):
    for j in range(0, len(X1_list)):
        distance_euclidean[i,j]=compute_distance_euclidean(X1_list[i], X1_list[j])

fig, ax = plt.subplots()

# Create the table
table = ax.table(cellText=distance_euclidean,
                 cellLoc='center',
                 loc='center')    

ax.set_title("Euclidean Distance between transported demonstrations")
save_array_as_latex(distance_euclidean, script_path+ '/results/euclidean distance.txt')



plt.show()    

