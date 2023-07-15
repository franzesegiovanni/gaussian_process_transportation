#If the plot are completely random, please run the code again.
from models import Ensamble_NN, Ensemble_RF
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from GILoSA import GaussianProcess as GPR  
from GILoSA import AffineTransform
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from plot_utils import draw_error_band, plot_vector_field
from save_utils import save_array_as_latex
from compute_trajectories_divergence import kl_mvn, compute_distance, compute_distance_euclidean

data =np.load(str(pathlib.Path().resolve())+'/data/'+str('example')+'.npz')
X=data['demo'] 
S=data['floor'] 
S1=data['newfloor']
fig = plt.figure(figsize = (12, 7))
plt.xlim([-50, 50-1])
plt.ylim([-50, 50-1])
plt.scatter(X[:,0],X[:,1], color=[1,0,0]) 
plt.scatter(S[:,0],S[:,1], color=[0,1,0])   
plt.scatter(S1[:,0],S1[:,1], color=[0,0,1]) 
plt.legend(["Demonstration","Surface","New Surface"])


#%% Calculate deltaX
deltaX = np.zeros((len(X),2))
for j in range(len(X)-1):
    deltaX[j,:]=(X[j+1,:]-X[j,:])

## Downsample
X=X[::2,:]
deltaX=deltaX[::2,:]

# #%% Fit a dynamical system to the demo and plot it
# k_deltaX = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=1.5) + WhiteKernel(0.01) 
# gp_deltaX=GPR(kernel=k_deltaX)
# gp_deltaX.fit(X, deltaX)
# x_grid=np.linspace(np.min(X[:,0]-10), np.max(X[:,0]+10), 100)
# y_grid=np.linspace(np.min(X[:,1]-10), np.max(X[:,1]+10), 100)
# plot_vector_field(gp_deltaX, x_grid,y_grid,X,S)

#%% Fit a GP to both surfaces and sample equal amount of indexed points, find delta pointcloud between old and sampled new surface
indexS = np.linspace(0, 1, len(S[:,0]))
indexS1 = np.linspace(0, 1, len(S1[:,0]))
k_S = C(constant_value=np.sqrt(0.1))  * RBF(1*np.ones(1)) + WhiteKernel(0.01 )  
gp_S=GPR(kernel=k_S)
gp_S.fit(indexS.reshape(-1,1),S)

k_S1 = C(constant_value=np.sqrt(0.1))  * RBF(1*np.ones(1)) + WhiteKernel(0.01 )  
gp_S1=GPR(kernel=k_S1)
gp_S1.fit(indexS1.reshape(-1,1),S1)

index = np.linspace(0, 1, 100).reshape(-1,1)
deltaPC = np.zeros((len(index),2))

source_distribution, _  =gp_S.predict(index)   
target_distribution, _  =gp_S1.predict(index)

affine_transform=AffineTransform()
affine_transform.fit(source_distribution, target_distribution)
source_distribution=affine_transform.predict(source_distribution) 
delta_distribution = target_distribution - source_distribution

k_transport = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(1), [1,10], nu=2.5) + WhiteKernel(0.0001)
methods=[Ensemble_RF(n_estimators=50, max_depth=5),  Ensamble_NN(n_estimators=10), GPR(k_transport)]
names=['Random Forest', 'Neural Network', 'Gaussian Process' ]

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

i=0
X1_list = []
std_list = []
X1=affine_transform.predict(X)
for model , name in zip(methods, names):

    model.fit(source_distribution, delta_distribution)
    # Make predictions on the test set

    X1_model, std = model.predict(X1, return_std=True)
    std_list.append(std)
    std= np.sqrt(std[:,0]**2+std[:,1]**2)
    # print(std)
    X1_model=X1+X1_model
    X_samples= X1+model.samples(X1)
    ax[i].scatter(target_distribution[:,0],target_distribution[:,1], color=[0,0,0], label="New Surface")
    ax[i].plot(X_samples[:,:,0].T, X_samples[:,:,1].T, alpha=0.5)
    draw_error_band(ax[i], X1_model[:,0], X1_model[:,1], err=2*std[:], facecolor= [255.0/256.0,140.0/256.0,0.0], edgecolor="none", alpha=.4, loop=True)
    ax[i].scatter(X1_model[:,0],X1_model[:,1], label="Tranported demonstration")
    ax[i].set_title(name, fontsize=24, fontweight='bold')
    ax[i].set_ylim(-20, 60)
    ax[i].grid()
    legend=ax[i].legend(loc='upper left', fontsize=12)
    # legend.set_fontsize('12')
    i+=1
    X1_list.append(X1_model)
fig.tight_layout()


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
save_array_as_latex(distance, 'distance.txt')

for i in range(len(X1_list)):
    for j in range(0, len(X1_list)):
        distance_euclidean[i,j]=compute_distance_euclidean(X1_list[i], X1_list[j])

fig, ax = plt.subplots()

# Create the table
table = ax.table(cellText=distance_euclidean,
                 cellLoc='center',
                 loc='center')    

ax.set_title("Euclidean Distance between transported demonstrations")
save_array_as_latex(distance_euclidean, 'euclidean distance.txt')



plt.show()    

