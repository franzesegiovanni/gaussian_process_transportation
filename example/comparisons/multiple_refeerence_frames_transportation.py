import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
# from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from GILoSA import GaussianProcess as GPR
from GILoSA import Transport
from matplotlib.patches import Circle
from plot_utils import plot_vector_field_minvar, draw_error_band
import warnings
warnings.filterwarnings("ignore")
def random_rotation_matrix_2d():
    theta = np.random.uniform(-0.1*np.pi, 0.1*np.pi)  # Random angle between 0 and 2*pi

    c = np.cos(theta)
    s = np.sin(theta)

    rotation_matrix = np.array([[c, -s],
                                [s, c]])

    return rotation_matrix
# Preparing the samples----------------------------------------------------------------------------------------------- #
nbSamples = 4
nbVar = 2
nbFrames = 2
nbStates = 3
nbData = 200
slist = []
distribution = np.zeros((nbSamples, 2*nbFrames, nbVar))
downsample=50
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

distribution_new = np.zeros((nbSamples, 2*nbFrames, nbVar))
# Generate test frames
A=np.empty((nbSamples, nbFrames, nbVar+1, nbVar+1))
B=np.empty((nbSamples, nbFrames, nbVar+1))
for i in range (nbSamples):
    for j in range(nbFrames):
            tempA = np.loadtxt('Multiple_Reference_Frames/sample' + str(i + 1) + '_frame' + str(j + 1) + '_A.txt', delimiter=',')
            tempB = np.loadtxt('Multiple_Reference_Frames/sample' + str(i + 1) + '_frame' + str(j + 1) + '_b.txt', delimiter=',')
            bTransform = (0.5*np.random.rand(nbVar, 1) - 0.15).reshape(-1,)
            # aTransform = (0.5*np.random.rand(nbVar, nbVar)).reshape(nbVar, nbVar) 
            aTransform = random_rotation_matrix_2d() 
            distribution_new[i,j*nbFrames,:]=tempB[1:,0]+bTransform
            distribution_new[i, j*nbFrames+1,:]=distribution_new[i,j*2,:]+(aTransform @ tempA[:, 0:3][1:,1:]) @ np.array([0, 0.5])
            B[i,j,:]=tempB[:,0]
            B[i,j,1:]=B[i,j,1:]+bTransform
            rot=np.ones((3,3))
            rot[1:3,1:3]=aTransform
            A[i,j,:]=rot @ tempA[:, 0:3][:,:]


np.save('A.npy', A)
np.save('B.npy', B) 
#%% Fit a dynamical system to the demo and plot it

def plot(distribution_input, i):
    index_source=2
    index_target=i


    k_deltaX = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=1.5) + WhiteKernel(0.01) #this kernel works much better!    
    gp_deltaX=GPR(kernel=k_deltaX)
    X=traj[index_source,:,:]
    deltaX=delta[index_source,:,:]
    gp_deltaX.fit(X, deltaX)

    transport=Transport()
    transport.source_distribution=distribution[index_source,:,:]
    transport.target_distribution=distribution_input[index_target,:,:]
    transport.training_traj=X
    transport.training_delta=deltaX
    k_transport = C(constant_value=np.sqrt(0.1))  * RBF(0.5*np.ones(1), [0.5,1]) + WhiteKernel(0.01 , [0.0000001, 0.000001])
    transport.kernel_transport=k_transport
    transport.fit_transportation()
    transport.apply_transportation()
    X1=transport.training_traj
    deltaX1=transport.training_delta 
    _,std=transport.gp_delta_map.predict(transport.training_traj)


    k_deltaX1 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=1.5) + WhiteKernel(0.01 ) #this kernel works much better!    
    gp_deltaX1=GPR(kernel=k_deltaX1)
    gp_deltaX1.fit(X1, deltaX1)

    
    plt.plot(transport.target_distribution[0:2,0],transport.target_distribution[0:2,1], linewidth=10, alpha=0.9, c='g')
    plt.plot(transport.target_distribution[2:4,0],transport.target_distribution[2:4,1], linewidth=10, alpha=0.9, c='b')
    plt.plot(X1[:,0],X1[:,1], color=[0,0,0])
    draw_error_band(ax, X1[:,0], X1[:,1], err=std[:], facecolor=f"C1", edgecolor="none", alpha=.8)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.grid('on')
fig, ax = plt.subplots()
for i in range(nbSamples):
    plot(distribution, i)
plt.plot(traj[2,:,0],traj[2,:,1], color=[1,0,0])    
plt.show()


for i in range(nbSamples):
    plot(distribution_new, i)    
plt.show()


