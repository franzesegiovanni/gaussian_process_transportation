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


#%% Get a  surface and demonstration from interactive interface
# This opens an interactive window to record surface and demonstration.
# Press once 'z' to start drawing. 
# Press 'w' to save the drawing as surface
# Press 'd' to save the drawing as demo
# Press 'n' to save the drawing as new surface
# to save the drawing as surface or demo or new surface respectively (This also stops the drawing )
# Press 'q' to exit
Surface_Demo = Drawing(draw=2)

#%% Visualize the drawings
demonstration= Surface_Demo.demo
surface  = Surface_Demo.floor
newsurface = Surface_Demo.newfloor

fig = plt.figure(figsize = (12, 7))
plt.xlim([-50, 50-1])
plt.ylim([-50, 50-1])
plt.scatter(demonstration[:,0],demonstration[:,1], color=[1,0,0]) 
plt.scatter(surface[:,0],surface[:,1], color=[0,1,0])   
plt.scatter(newsurface[:,0],newsurface[:,1], color=[0,0,1]) 

#%% Save the drawings

Surface_Demo.save(data='indexing6')

# %%
