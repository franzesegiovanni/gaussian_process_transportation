"""
Authors:  Giovanni Franzese
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
from gmr.sklearn import GaussianMixtureRegressor as GMM
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
import matplotlib.pyplot as plt
from policy_transportation import GaussianProcess as GPR 
import pathlib
from policy_transportation.plot_utils import plot_vector_field 
import warnings
from policy_transportation import AffineTransform
from policy_transportation.utils import resample

warnings.filterwarnings("ignore")
#%% Load the drawings

data =np.load(str(pathlib.Path().resolve())+'/data/'+str('example')+'.npz')
X=data['demo'] 
S=data['floor'] 
S1=data['newfloor']
fig = plt.figure(figsize = (12, 7))
plt.xlim([-50, 50-1])
plt.ylim([-50, 50-1])
X=resample(X, num_points=200)
S=resample(S,num_points=20)
S1=resample(S1,num_points=20)
plt.scatter(X[:,0],X[:,1], color=[1,0,0]) 
plt.scatter(S[:,0],S[:,1], color=[0,1,0])   
plt.scatter(S1[:,0],S1[:,1], color=[0,0,1]) 
plt.legend(["Demonstration","Surface","New Surface"])


source_distribution=S
target_distribution=S1

#%% Calculate deltaX
deltaX = np.zeros((len(X),2))
for j in range(len(X)-1):
    deltaX[j,:]=(X[j+1,:]-X[j,:])


#%% Fit a dynamical system to the demo and plot it
k_deltaX = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=1.5) + WhiteKernel(0.01) 
gp_deltaX=GPR(kernel=k_deltaX)
gp_deltaX.fit(X, deltaX)
x_grid=np.linspace(np.min(X[:,0]-10), np.max(X[:,0]+10), 100)
y_grid=np.linspace(np.min(X[:,1]-10), np.max(X[:,1]+10), 100)
plot_vector_field(gp_deltaX, x_grid,y_grid,X,S)



affine_transform=AffineTransform()
affine_transform.fit(source_distribution, target_distribution)
source_distribution=affine_transform.predict(source_distribution) 

gmm = GMM(n_components=10, random_state=0)
gmm.fit(source_distribution, target_distribution)

# 
X1=affine_transform.predict(X)
X1 = gmm.predict(X1)
# print(delta.shape)
# X1=source_distribution+delta
plt.figure()
plt.scatter(X1[:,0],X1[:,1])
plt.scatter(target_distribution[:,0],target_distribution[:,1], color=[0,0,0])
plt.show()