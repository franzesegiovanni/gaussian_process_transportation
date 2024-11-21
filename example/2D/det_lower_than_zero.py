import numpy as np
import matplotlib.pyplot as plt
import pathlib
# polar decomposition from scipy
from scipy.linalg import polar
# Step 2: Define a 2x2 matrix with det lower than zero
matrix = np.array([[1, 0], [0, -1]])
source_path = str(pathlib.Path(__file__).parent.absolute())
# matrix = matrix/np.linalg.norm(matrix)  
# make the matrix to have jacobian positive
# matrix= polar(matrix)[0]
# # check for replactions and fix it
# if np.linalg.det(matrix)<0:
#     matrix[:,0] = -matrix[:,0]

# Step 5: Calculate the Jacobian matrix of the transformation
jacobian = matrix

det = np.linalg.det(jacobian)
jacobian= jacobian
print(f'Determinant of the matrix: {det}')

# Creat 4 delta x vectors in the 4 cartesian position 


import imageio
import os

# Create a list to store file names
filenames = []

for deg in range (0,180):
    deltaX = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    angle = np.radians(deg)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    deltaX = np.matmul(deltaX, rotation_matrix)
    color = ['r', 'g', 'b', 'y']
    transformed_deltaX = np.matmul(jacobian, deltaX.T).T

    plt.figure()
    plt.quiver(np.zeros_like(deltaX[:, 0]), np.zeros_like(deltaX[:, 0]), deltaX[:, 0], deltaX[:, 1], color=color, angles='xy', scale_units='xy', scale=2,width=0.02)
    plt.quiver(np.zeros_like(deltaX[:, 0]), np.zeros_like(deltaX[:, 0]), transformed_deltaX[:, 0], transformed_deltaX[:, 1], color=color, angles='xy', scale_units='xy', scale=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    # save to local file
    filename = source_path+ '/test/'+'example' + str(int((deg))) + '.png'
    # print(filename)
    plt.savefig(filename)
    filenames.append(filename)

# Create a GIF from the images
with imageio.get_writer(source_path+ '/test/'+'/det_lower_than_zero_1.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Remove files
for filename in set(filenames):
    os.remove(filename)
