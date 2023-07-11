import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the parametric equations of the surface
def surface_equations(u, v):
    x = u
    y = v
    z = u**2 - v**2  # Modify the equation to create your desired surface
    return x, y, z

# Create the data for the surface
u = np.linspace(-2, 2, 10)
v = np.linspace(-2, 2, 10)
u, v = np.meshgrid(u, v)
target_x, target_y, target_z = surface_equations(u, v)

# Plot the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(target_x, target_y, target_z, color='b', alpha=0.5, linewidths=0)
ax.plot3D(target_x[:, 0], target_y[:, 0], target_z[:, 0], color='b', alpha=0.6, linewidth=2)
ax.plot3D(target_x[:, -1], target_y[:, -1], target_z[:, -1], color='b', alpha=0.6, linewidth=2)
ax.plot3D(target_x[0, :], target_y[0, :], target_z[0, :], color='b', alpha=0.6, linewidth=2)
ax.plot3D(target_x[-1, :], target_y[-1, :], target_z[-1, :], color='b', alpha=0.6, linewidth=2)

 
source_z=np.zeros_like(target_z)
source_x, source_y= target_x, target_y
ax.plot_surface(source_x, source_y, source_z, color='g', alpha=0.5)
ax.plot3D(source_x[:, 0], source_y[:, 0], source_z[:, 0], color='g', alpha=0.6, linewidth=2)
ax.plot3D(source_x[:, -1], source_y[:, -1], source_z[:, -1], color='g', alpha=0.6, linewidth=2)
ax.plot3D(source_x[0, :], source_y[0, :], source_z[0, :], color='g', alpha=0.6, linewidth=2)
ax.plot3D(source_x[-1, :], source_y[-1, :], source_z[-1, :], color='g', alpha=0.6, linewidth=2)

ax.quiver(source_x, source_y, source_z, target_x-source_x, target_y-source_y, target_z-source_z, color='r', alpha=0.5, linewidth=2)


# plot the qvier on the side of the surface


# Set axis labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
ax.axis('off') 
# Choose a point on the surface
# point_u = 1.5  # Example values for the chosen point
# point_v = 0.5

# point_x, point_y, point_z = surface_equations(point_u, point_v)

# Plot the chosen point
# ax.scatter(point_x, point_y, point_z, color='r', s=50)
# ax.scatter(point_x, point_y, 0, color='r', s=50)
# Calculate the tangent plane at the chosen point
# delta = 0.5  # Distance from the chosen point to calculate the tangent plane
# u_tangent, v_tangent = np.meshgrid(np.linspace(point_u - delta, point_u + delta, 10),
#                                    np.linspace(point_v - delta, point_v + delta, 10))
# tangent_x, tangent_y, tangent_z = surface_equations(u_tangent, v_tangent)

# # Plot the tangent plane
# ax.plot_surface(tangent_x, tangent_y, tangent_z, color='g', alpha=0.8)
# ax.plot3D(tangent_x[:, 0], tangent_y[:, 0], tangent_z[:, 0], color='g', alpha=0.8, linewidth=2)
# ax.plot3D(tangent_x[:, -1], tangent_y[:, -1], tangent_z[:, -1], color='g', alpha=0.8, linewidth=2)
# ax.plot3D(tangent_x[0, :], tangent_y[0, :], tangent_z[0, :], color='g', alpha=0.8, linewidth=2)
# ax.plot3D(tangent_x[-1, :], tangent_y[-1, :], tangent_z[-1, :], color='g', alpha=0.8, linewidth=2)


# ax.set_title('Surface with Tangent Plane')

# Show the plot
plt.savefig('surface.png', dpi=1000)
plt.show()

