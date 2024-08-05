"""
Authors:  Giovanni Franzese July 2024
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""

#%%
import numpy as np
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C, ExpSineSquared as Periodic
from policy_transportation.transportation.kernelized_movement_primitives_transportation import KMP_transportation as KMP
from policy_transportation.transportation.laplacian_editing_transportation import LaplacianEditingTransportation as LE
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
fig.text(0.0, 0.75, 'Kernelize MP', va='center', rotation='vertical', fontsize=12)
fig.text(0.0, 0.25, 'Laplacian Editing', va='center', rotation='vertical',  fontsize=12)
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
axs[0,0].text(0.9, 0.1, "Time", transform=axs[0,0].transAxes,color='blue')
axs[0,0].text(0.9, 0.0, "Accuracy", transform=axs[0,0].transAxes,  color='black')
axs[1,0].text(0.9, 0.1, "Time", transform=axs[1,0].transAxes,color='blue')
axs[1,0].text(0.9, 0.0, "Accuracy", transform=axs[1,0].transAxes,  color='black')

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
    
    k_kmp=kernel=C(0.1, constant_value_bounds=[0.1,2]) * Periodic(periodicity=1, periodicity_bounds=[1,1], length_scale=0.1, length_scale_bounds=[0.02, 0.05]) + WhiteKernel(0.00001, noise_level_bounds=[1e-5, 0.01])
    gpt=KMP(kernel=k_kmp)
    gpt.source_distribution=source_distribution 
    gpt.target_distribution=target_distribution
    gpt.training_traj=X
    gpt.training_delta=deltaX
    start = timeit.default_timer()
    gpt.fit_transportation()
    gpt.apply_transportation()
    stop = timeit.default_timer()
    time_gpt.append(stop - start)
    accurracy_gpt.append(gpt.accuracy())
    # print('Accuracy: ', accurracy_gpt[i])
    print('Time: ', stop - start)
    axs[0,plot_index].plot(gpt.training_traj[:,0],gpt.training_traj[:,1], color=[1,0,0])
    axs[0,plot_index].plot(target_distribution[:,0],target_distribution[:,1], color=[0,0,1])
    #remove axis
    axs[0,plot_index].axis('off')
    # diffeo_percentage_gpt.append(np.sum(~gpt.diffeo_mask)/len(gpt.diffeo_mask))
    #add the diffemorphic percentage on the top right corner
    # axs[0,plot_index].text(1, 0.2, str(np.round(diffeo_percentage_gpt[i],3)), transform=axs[0,plot_index].transAxes, color='red')
    #add the time under it 
    # axs[0,plot_index].text(0.9, 0.1, str(np.round(time_gpt[i],3))+'[s]', transform=axs[0,plot_index].transAxes, color='blue')
    axs[0,plot_index].text(0.9, 0.1, "{:.1e}".format(time_gpt[i]), transform=axs[0,plot_index].transAxes, color='blue')
    axs[0,plot_index].text(0.9, 0.0, "{:.1e}".format(accurracy_gpt[i]), transform=axs[0,plot_index].transAxes, color='black')

    #%% Diffeomorphic transportation
    lwt=LE()
    lwt.source_distribution=source_distribution
    lwt.target_distribution=target_distribution
    lwt.training_traj=X
    lwt.training_delta=deltaX
    start = timeit.default_timer()
    lwt.fit_transportation(do_scale=True, do_rotation=True)
    lwt.apply_transportation()
    stop = timeit.default_timer()
    time_diffeo.append(stop - start)
    print('Time: ', stop - start)
    accurracy_ltw.append(lwt.accuracy())
    print('Accuracy: ', accurracy_ltw[i])
    axs[1,plot_index].plot(lwt.training_traj[:,0],lwt.training_traj[:,1], color=[1,0,0])
    axs[1,plot_index].plot(target_distribution[:,0],target_distribution[:,1], color=[0,0,1])
    axs[1,plot_index].axis('off')
    # diffeo_percentage_ltw.append(np.sum(~lwt.diffeo_mask)/len(lwt.diffeo_mask))
    #add the diffemorphic percentage on the top right corner
    # axs[1,plot_index].text(1, 0.2, str(np.round(diffeo_percentage_ltw[i],3)), transform=axs[1,plot_index].transAxes, color='red')
    #add the time under it
    # axs[1,plot_index].text(0.9, 0.1, str(np.round(time_diffeo[i],3))+'[s]', transform=axs[1,plot_index].transAxes, color='blue')
    axs[1,plot_index].text(0.9, 0.1, "{:.1e}".format(time_diffeo[i]), transform=axs[1,plot_index].transAxes, color='blue')
    axs[1,plot_index].text(0.9, 0.0, "{:.1e}".format(accurracy_ltw[i]), transform=axs[1,plot_index].transAxes, color='black')


#save the figure as pdf high resolution
plt.tight_layout()
plt.savefig(source_path+'/results/comparisons_kmp_le.pdf', dpi=300)
# plt.savefig(source_path+'/results/comparisons_diffeomorphism.png')
plt.show()