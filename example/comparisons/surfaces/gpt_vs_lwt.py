"""
Authors:  Giovanni Franzese July 2024
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""

#%%
import numpy as np
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from policy_transportation.transportation.transportation import PolicyTransportation
from policy_transportation.models.gaussian_process import GaussianProcess as GPR
from policy_transportation.models.locally_weighted_translations import Iterative_Locally_Weighted_Translations as ILWT

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
fig.text(0.0, 0.75, 'Gaussian Process', va='center', rotation='vertical', fontsize=12)
fig.text(0.0, 0.25, 'LTW', va='center', rotation='vertical',  fontsize=12)
time_gpt= []
time_ltw= []
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
axs[0,0].text(0.9, 0.2, " Jac > 0", transform=axs[0,0].transAxes,color='red')
axs[0,0].text(0.9, 0.1, "Time", transform=axs[0,0].transAxes,color='blue')
axs[0,0].text(0.9, 0.0, "Accuracy", transform=axs[0,0].transAxes,  color='black')
axs[1,0].text(0.9, 0.2, " Jac > 0", transform=axs[1,0].transAxes,color='red')
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
    
    k_transport = C(constant_value=10)  * RBF(0.9*np.ones(2), [0.1, 10]) + WhiteKernel(0.01 )
    transport=PolicyTransportation()
    k_transport = C(constant_value=10)  * RBF(1*np.ones(2), [0.1,5]) + WhiteKernel(0.01 )

    method = GPR(k_transport)
    transport.set_method(method=method, is_residual=method.is_residual)
    start = timeit.default_timer()
    transport.fit(source_distribution, target_distribution, do_scale=False, do_rotation=True)

    X_hat=transport.transport(X, return_std=False)
    deltaX_hat=transport.transport_velocity(X, deltaX, return_var=False)

    stop = timeit.default_timer()
    time_gpt.append(stop - start)
    accurracy_gpt.append(transport.accuracy)
    diffeomorphic = transport.is_diffeomorphic_on(X)
    print('Accuracy: ', accurracy_gpt[i])
    print('Time: ', stop - start)
    axs[0,plot_index].plot(X_hat[:,0],X_hat[:,1], color=[1,0,0])
    axs[0,plot_index].plot(target_distribution[:,0],target_distribution[:,1], color=[0,0,1])
    #remove axis
    axs[0,plot_index].axis('off')
    diffeo_percentage_gpt.append(np.sum(diffeomorphic)/len(diffeomorphic))
    #add the diffemorphic percentage on the top right corner
    axs[0,plot_index].text(1, 0.2, str(np.round(diffeo_percentage_gpt[i],3)), transform=axs[0,plot_index].transAxes, color='red')
    #add the time under it 
    # axs[0,plot_index].text(0.9, 0.1, str(np.round(time_gpt[i],3))+'[s]', transform=axs[0,plot_index].transAxes, color='blue')
    axs[0,plot_index].text(0.9, 0.1, "{:.1e}".format(time_gpt[i]), transform=axs[0,plot_index].transAxes, color='blue')
    axs[0,plot_index].text(0.9, 0.0, "{:.1e}".format(accurracy_gpt[i]), transform=axs[0,plot_index].transAxes, color='black')

    #%% Diffeomorphic transportation
    transport=PolicyTransportation()

    method = ILWT( num_iterations=30, rho=1, beta=0.9)
    transport.set_method(method=method, is_residual=method.is_residual)
    start = timeit.default_timer()
    transport.fit(source_distribution, target_distribution, do_scale=False, do_rotation=True)

    X_hat=transport.transport(X, return_std=False)
    deltaX_hat=transport.transport_velocity(X, deltaX, return_var=False)

    stop = timeit.default_timer()
    time_ltw.append(stop - start)
    accurracy_ltw.append(transport.accuracy)
    diffeomorphic = transport.is_diffeomorphic_on(X)
    print('Accuracy: ', accurracy_gpt[i])
    print('Time: ', stop - start)
    axs[1,plot_index].plot(X_hat[:,0],X_hat[:,1], color=[1,0,0])
    axs[1,plot_index].plot(target_distribution[:,0],target_distribution[:,1], color=[0,0,1])
    #remove axis
    axs[1,plot_index].axis('off')
    diffeo_percentage_ltw.append(np.sum(diffeomorphic)/len(diffeomorphic))
    #add the diffemorphic percentage on the top right corner
    axs[1,plot_index].text(1, 0.2, str(np.round(diffeo_percentage_ltw[i],3)), transform=axs[1,plot_index].transAxes, color='red')
    #add the time under it
    # axs[1,plot_index].text(0.9, 0.1, str(np.round(time_diffeo[i],3))+'[s]', transform=axs[1,plot_index].transAxes, color='blue')
    axs[1,plot_index].text(0.9, 0.1, "{:.1e}".format(time_ltw[i]), transform=axs[1,plot_index].transAxes, color='blue')
    axs[1,plot_index].text(0.9, 0.0, "{:.1e}".format(accurracy_ltw[i]), transform=axs[1,plot_index].transAxes, color='black')


#save the figure as pdf high resolution
plt.tight_layout()
plt.savefig(source_path+'/results/comparisons_diffeomorphism.pdf', dpi=300)
# plt.savefig(source_path+'/results/comparisons_diffeomorphism.png')
plt.show()