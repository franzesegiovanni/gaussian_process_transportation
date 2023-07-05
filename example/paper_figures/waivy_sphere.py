import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a sphere mesh
theta, phi = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 50)
theta, phi = np.meshgrid(theta, phi)
radius = 1.0
x = radius * np.sin(phi) * np.cos(theta)
y = radius * np.sin(phi) * np.sin(theta)
z = radius * np.cos(phi)

# Add wavy distortion to the sphere
amplitude = 0.2
frequency = 10
wavy_z = z + amplitude * np.sin(frequency * x)

# Plot the wavy sphere
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, wavy_z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
