import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
# from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from GILoSA import GaussianProcess as GPR
from GILoSA import Transport
from plot_utils import plot_vector_field_minvar
import warnings
warnings.filterwarnings("ignore")
# Preparing the samples----------------------------------------------------------------------------------------------- #
nbSamples = 4
nbVar = 2
nbFrames = 2
nbStates = 3
nbData = 200
slist = []
distribution = np.zeros((nbSamples, 2*nbFrames, nbVar))
downsample=100
traj = np.zeros((nbSamples,downsample,nbVar))
rawData = np.zeros((nbSamples,nbData,nbVar))
delta=np.zeros_like(traj)

for i in range(nbSamples):
    tempData= np.loadtxt('Multiple_Reference_Frames/sample' + str(i + 1) + '_Data.txt', delimiter=',')

    for j in range(nbFrames):
        tempA = np.loadtxt('Multiple_Reference_Frames/sample' + str(i + 1) + '_frame' + str(j + 1) + '_A.txt', delimiter=',')
        tempB = np.loadtxt('Multiple_Reference_Frames/sample' + str(i + 1) + '_frame' + str(j + 1) + '_b.txt', delimiter=',')
        distribution[i,j*2,:]=tempB[1:,0]
        distribution[i, j*2+1,:]=tempB[1:,0]+tempA[:, 0:3][1:,1:] @ np.array([0, 0.5])
        # print(np.linalg.inv(tempA[:, 0:3][1:,1:]))
    ker = C(constant_value=np.sqrt(0.1))  * RBF(1*np.ones(1)) + WhiteKernel(0.01 )
    gp = GPR(kernel=ker, alpha=1e-10, n_restarts_optimizer=5)
    gp.fit((np.linspace(0,2,200)).reshape(-1,1),np.transpose(tempData[1:,:])) #smooth the data
    traj[i,:,:]=gp.predict(np.linspace(0,2,downsample).reshape(-1,1))[0]
    rawData[i,:,:]=np.transpose(tempData[1:,:])
    delta[i,:-1,:]=traj[i,1:,:]-traj[i,:-1,:]      
plt.plot(traj[0,:,0],traj[0,:,1], c='r')  
# plt.plot(rawData[0,:,0],rawData[0,:,1], c='b')
plt.scatter(distribution[0,:,0],distribution[0,:,1])
# plt.show()
#%% Fit a dynamical system to the demo and plot it

index_source=1
index_target=2


k_deltaX = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=1.5) + WhiteKernel(0.01) #this kernel works much better!    
gp_deltaX=GPR(kernel=k_deltaX)
X=traj[index_source,:,:]
deltaX=delta[index_source,:,:]
gp_deltaX.fit(X, deltaX)

transport=Transport()
transport.source_distribution=distribution[index_source,:,:]
transport.target_distribution=distribution[index_target,:,:]
transport.training_traj=X
transport.training_delta=deltaX
k_transport = C(constant_value=np.sqrt(0.1))  * RBF(0.5*np.ones(1), [0.3,1]) + WhiteKernel(0.01 , [0.000001, 0.00001])
transport.kernel_transport=k_transport
transport.fit_transportation()
transport.apply_transportation()
X1=transport.training_traj
deltaX1=transport.training_delta 

fig = plt.figure(figsize = (12, 7))
# plt.xlim([-50, 50-1])
# plt.ylim([-50, 50-1])
plt.scatter(X1[:,0],X1[:,1], color=[1,0,0]) 
plt.scatter(transport.source_distribution[:,0],transport.source_distribution[:,1])   
plt.scatter(X[:,0],X[:,1], color=[0,0,1]) 
plt.scatter(transport.target_distribution[:,0],transport.target_distribution[:,1])

x1_grid=np.linspace(np.min(X1[:,0]), np.max(X1[:,0]), 100)
y1_grid=np.linspace(np.min(X1[:,1]), np.max(X1[:,1]), 100)

k_deltaX1 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=1.5) + WhiteKernel(0.01 ) #this kernel works much better!    
gp_deltaX1=GPR(kernel=k_deltaX1)
gp_deltaX1.fit(X1, deltaX1)
plot_vector_field_minvar(gp_deltaX1, x1_grid,y1_grid,X1,transport.target_distribution)
# plt.plot(rawData[index_target,:,0],rawData[index_target,:,1], c='b', alpha=0.4)
plt.plot(transport.target_distribution[0:2,0],transport.target_distribution[0:2,1], linewidth=10, alpha=0.9)
plt.plot(transport.target_distribution[2:4,0],transport.target_distribution[2:4,1], linewidth=10, alpha=0.9)
plt.scatter(traj[index_target,:,0],traj[index_target,:,1], color=[0,0,1], alpha=0.5)
# plt.plot(transport.target_distribution[:2,0],transport.target_distribution[:2,1], linewidth=10)
plt.show()



