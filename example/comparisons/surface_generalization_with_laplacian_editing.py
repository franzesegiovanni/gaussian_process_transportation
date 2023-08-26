"""
Authors:  Giovanni Franzese 
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""

#%%
import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
import matplotlib.pyplot as plt
from GILoSA import GaussianProcess as GPR
from GILoSA import AffineTransform
import pathlib
from plot_utils import plot_vector_field_minvar, plot_vector_field 
import warnings
from scipy.spatial import cKDTree
import networkx as nx
warnings.filterwarnings("ignore")
#%% Load the drawings

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

#%% Fit a dynamical system to the demo and plot it
k_deltaX = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=1.5) + WhiteKernel(0.01) 
gp_deltaX=GPR(kernel=k_deltaX)
gp_deltaX.fit(X, deltaX)
x_grid=np.linspace(np.min(X[:,0]-10), np.max(X[:,0]+10), 100)
y_grid=np.linspace(np.min(X[:,1]-10), np.max(X[:,1]+10), 100)
plot_vector_field(gp_deltaX, x_grid,y_grid,X,S)

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


# Create a chain graph
num_nodes = X.shape[0]
G = nx.path_graph(num_nodes)
G = nx.cycle_graph(num_nodes)

# Create a new graph
combined_graph = nx.Graph()

# Add nodes from the cycle graph to the combined graph
combined_graph.add_nodes_from(G.nodes())
combined_graph.add_edges_from(G.edges())

# # Connect each node in the cycle graph to a new node in the combined graph
for m in range(num_nodes):
    combined_graph.add_node(m + num_nodes)  # Adding a new node with label = node + 5
    combined_graph.add_edge(m, m + num_nodes)  # Connecting to a new node with label = node + 5


G=combined_graph
plt.figure()
nx.draw(G, with_labels=True, node_size=100, node_color="skyblue")
# Compute the graph Laplacian matrix
L = nx.laplacian_matrix(G).toarray()
print(L)

# print(X.shape)
X_tot= np.vstack([X, X+deltaX])
DELTA= L @ X_tot 

# print(DELTA)


# Threshold distance
threshold_distance = 5.0

# Create KDTree for array2
tree = cKDTree(source_distribution)

# List to store pairs
pairs = []
matched_indices = set()  # To keep track of matched indices from array2
mask_traj=np.zeros(len(X), dtype=bool)
# Iterate through each element in array1
for i, element in enumerate(X):
    # Query the KDTree for the nearest neighbor
    distance, idx = tree.query(element)
    
    # Check if the nearest neighbor has already been matched and if the distance is within threshold
    if distance <= threshold_distance and idx not in matched_indices:
        nearest_neighbor = source_distribution[idx]
        
        # Store the pair and mark the neighbor as matched
        pairs.append((element, nearest_neighbor))
        matched_indices.add(idx)
        mask_traj[i]=True

mask_dist=np.array(list(matched_indices))
# print(np.array(list(matched_indices)))
# print(mask_traj)

affine=AffineTransform()
affine.fit(source_distribution, target_distribution)
X=affine.predict(X)

diff=np.zeros_like(X)

diff[mask_traj]=target_distribution[mask_dist] - affine.predict(source_distribution[mask_dist])
constraint= np.zeros_like(X)
constraint[mask_traj]= X[mask_traj]+ diff[mask_traj]

constraint=np.vstack([constraint, np.zeros_like(constraint)])
mask_traj=np.hstack([mask_traj, np.zeros_like(mask_traj)])
P_hat=np.diag(1*mask_traj)



A= np.vstack([L, P_hat])

B= np.vstack([DELTA, constraint])

P_s= np.linalg.pinv(A) @ B

X1= P_s[:len(X),:]
deltaX1=np.zeros_like(X1)
# deltaX1= P_s[len(X):,:] - X1
for j in range(len(X)-1):
    deltaX1[j,:]=(X1[j+1,:]-X1[j,:])
plt.figure()
plt.scatter(P_s[len(X):,0],P_s[len(X):,1]) 
plt.scatter(P_s[:len(X),0],P_s[:len(X),1]) 
plt.scatter(S1[:,0],S1[:,1])



x1_grid=np.linspace(np.min(X1[:,0]-10), np.max(X1[:,0]+10), 200)
y1_grid=np.linspace(np.min(X1[:,1]-10), np.max(X1[:,1]+10), 200)
      


# Fit the Gaussian Process dynamical system   
print('Fitting the GP dynamical system on the transported trajectory')
k_deltaX1 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=1.5) + WhiteKernel(0.01 ) #this kernel works much better!    
gp_deltaX1=GPR(kernel=k_deltaX1)
gp_deltaX1.fit(X1, deltaX1)
plot_vector_field(gp_deltaX1, x1_grid,y1_grid,X1,S1)
plt.show()  


