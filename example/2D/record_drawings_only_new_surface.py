"""
Authors: Ravi Prakash and Giovanni Franzese, Dec 2022
Email: r.prakash-1@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""

#%%
# import numpy as np
# from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
import matplotlib.pyplot as plt
# from regressor import GPR
from interface import Drawing
import pathlib
import numpy as np

#%% Get a  surface and demonstration from interactive interface
# This opens an interactive window to record surface and demonstration.
# Press once 'z' to start drawing. 
# Press 'w' to save the drawing as surface
# Press 'd' to save the drawing as demo
# Press 'n' to save the drawing as new surface
# to save the drawing as surface or demo or new surface respectively (This also stops the drawing )
# Press 'q' to exit
source_path = str(pathlib.Path(__file__).parent.absolute())  
source_path = str(pathlib.Path(__file__).parent.absolute())  
data =np.load(source_path+ '/data/'+str('example')+'.npz')
X=data['demo'] 
S=data['floor'] 
S1=data['newfloor']
#%% Visualize the drawings
fig, ax= plt.subplots(figsize=(12,7))
plt.xlim([-50, 50-1])
plt.ylim([-50, 50-1])
plt.scatter(X[:,0],X[:,1], color=[1,0,0]) 
plt.scatter(S[:,0],S[:,1], color=[0,1,0])   
Surface_Demo = Drawing(fig, ax, draw=2)

Surface_Demo.demo= X
Surface_Demo.floor = S
newsurface = Surface_Demo.newfloor


plt.scatter(newsurface[:,0],newsurface[:,1], color=[0,0,1]) 

#%% Save the drawings

Surface_Demo.save(data='example10')

# %%
