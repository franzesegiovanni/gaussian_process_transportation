import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

# Create data for the surface plot
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# Create a light source
light = LightSource(azdeg=225, altdeg=45)

# Calculate the shading of the surface plot
rgb = light.shade(Z, cmap=plt.cm.viridis)

# Create a 3D plot figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface with shading
surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, cmap=plt.cm.inferno)

# Customize the plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Metallic Surface Plot')

# Show the plot
plt.show()
