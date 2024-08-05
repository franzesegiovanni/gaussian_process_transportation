"""
Authors:  Giovanni Franzese July 2024
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""

#%%
import numpy as np
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from policy_transportation.transportation.gaussian_process_transportation import GaussianProcessTransportation as GPT
from policy_transportation.transportation.diffeomorphic_transportation import DiffeomorphicTransportation as LTW
import pathlib
from policy_transportation.utils import resample
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import timeit
#%% Load the drawings

source_path = str(pathlib.Path(__file__).parent.absolute()) 
#create subfigure 2x10 with tight layeout
fig, axs = plt.subplots(2, 10, figsize=(20, 4), tight_layout=True)
fig.text(0.0, 0.75, 'Rotation ', va='center', rotation='vertical', fontsize=12)
fig.text(0.0, 0.25, 'No Rotation', va='center', rotation='vertical',  fontsize=12)
time_gpt= []
time_diffeo= []
diffeo_percentage_gpt= []
diffeo_percentage_ltw= []
accurracy_gpt= []
accurracy_ltw= []
#plot demo and floor in the first column for both methods
data= np.load(source_path+ '/data/'+'example'+str(0)+'.npz')
X=data['demo']
S=data['floor']
axs[0,0].plot(X[:,0],X[:,1], color=[1,0,0])
axs[0,0].plot(S[:,0],S[:,1], color=[0,0,1])
axs[0,0].axis('off')
axs[1,0].plot(X[:,0],X[:,1], color=[1,0,0])
axs[1,0].plot(S[:,0],S[:,1], color=[0,0,1])
axs[1,0].axis('off')
axs[0,0].text(0.2, 1, 'Demo & Source',transform=axs[0,0].transAxes, color='black', fontsize=10)
for i in range(0,9):
    plot_index=i+1
    data =np.load(source_path+ '/data/'+'example'+str(i+1)+'.npz')
    X=data['demo'] 
    S=data['floor'] 
    S1=data['newfloor']
    X=resample(X, num_points=200)
    source_distribution=resample(S, num_points=20)
    target_distribution=resample(S1, num_points=20)
    #create empy list
    #%% Calculate deltaX
    deltaX = np.zeros((len(X),2))
    for j in range(len(X)-1):
        deltaX[j,:]=(X[j+1,:]-X[j,:])

    #%% Transport the dynamical system on the new surface
    
    k_transport = C(constant_value=10)  * RBF(0.9*np.ones(1), [1, 10]) + WhiteKernel(0.01, [0.001, 0.1])
    gpt=GPT(kernel_transport= k_transport)
    gpt.source_distribution=source_distribution 
    gpt.target_distribution=target_distribution
    gpt.training_traj=X
    gpt.training_delta=deltaX
    gpt.fit_transportation(do_scale=True, do_rotation=True)
    gpt.apply_transportation()

    axs[0,plot_index].plot(gpt.training_traj[:,0],gpt.training_traj[:,1], color=[1,0,0])
    axs[0,plot_index].plot(target_distribution[:,0],target_distribution[:,1], color=[0,0,1])
    axs[0,plot_index].axis('off')

    #%% Diffeomorphic transportation
    gpt_no_rotation=GPT(kernel_transport= k_transport)
    gpt_no_rotation.source_distribution=source_distribution 
    gpt_no_rotation.target_distribution=target_distribution
    gpt_no_rotation.training_traj=X
    gpt_no_rotation.training_delta=deltaX
    gpt_no_rotation.fit_transportation(do_scale=True, do_rotation=False)
    gpt_no_rotation.apply_transportation()

    axs[1,plot_index].plot(gpt_no_rotation.training_traj[:,0],gpt_no_rotation.training_traj[:,1], color=[1,0,0])
    axs[1,plot_index].plot(target_distribution[:,0],target_distribution[:,1], color=[0,0,1])
    axs[1,plot_index].axis('off')
#save the figure as pdf high resolution
plt.tight_layout()
plt.savefig(source_path+'/results/with_or_without_rotation.pdf', dpi=300)
plt.show()