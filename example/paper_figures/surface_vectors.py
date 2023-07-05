import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the surface coordinates
X, Y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
Z = -X**2 - Y**2  # Example surface function (paraboloid)

# Calculate the gradient (normal vectors) of the surface
dZdX, dZdY = np.gradient(Z)
# a = np.dstack((dZdX, np.zeros_like(Z), np.zeros_like(Z)))  # Negate the gradients and add Z component
# b = np.dstack((np.zeros_like(Z), dZdY,  np.zeros_like(Z)))
# normals = np.cross(a, b)
# coefficient_sign = np.sign(-1)  # Assuming the coefficient of the z-coordinate is -1

# Adjust the sign of the coefficient based on the signs of the partial derivatives
# if np.sign(dZdX) != coefficient_sign:
#     coefficient_sign *= -1
# np.sign(dZdX)
# np.sign(dZdX)==-1
coefficient_sign = np.sign(dZdX * dZdY)
normals = np.array([dZdX, dZdY, -coefficient_sign* np.ones_like(Z)])
# normals /= np.linalg.norm(normals, axis=0)
# Plot the surface and the normal vectors

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, alpha=0.5)
#ax.quiver(X, Y, Z, normals_pos[0,:, :], normals_pos[1,:, :], normals_pos[2,:, :], color='r')
ax.quiver(X, Y, Z, normals[0,:, :], normals[1,:, :], normals[2,:, :], color='b')
# ax.quiver(X, Y, Z, a[:, :, 0], a[:, :, 1], a[:, :, 2], color='g')
# ax.quiver(X, Y, Z, b[:, :, 0], b[:, :, 1], b[:, :, 2], color='b')
plt.show()

