import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the curved surface
u = np.linspace(0, 2*np.pi, 20)
v = np.linspace(0, np.pi, 20)
U, V = np.meshgrid(u, v)
X = 3 * np.cos(U) * np.sin(V)
Y = 3 * np.sin(U) * np.sin(V)
Z = 3 * np.cos(V)

# Define the vector field
P = -Y
Q = X
R = Z

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the curved surface
ax.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis')

# Create the quiver plot on the surface
ax.quiver(X, Y, Z, P, Q, R, length=0.1, color='r')

# Set plot labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Quiver Plot on a Curved Surface')

# Display the plot
plt.show()
