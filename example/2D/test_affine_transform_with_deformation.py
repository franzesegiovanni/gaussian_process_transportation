import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define your source and target lists of 3D points
source_points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
target_points = 10*np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])

# Center the points by subtracting the mean
mean_source = np.mean(source_points, axis=0)
mean_target = np.mean(target_points, axis=0)
centered_source = source_points - mean_source
centered_target = target_points - mean_target

# Calculate the scale factor using the original equation
scale_original = np.sum(centered_source * centered_target) / np.sum(centered_source**2)

# Calculate the scale factor using the corrected equation (SVD)
U, S, Vt = np.linalg.svd(centered_source.T @ centered_target)

print("Scale (Original): ", scale_original)

# Plot the points and the transformed points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Original points
ax.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], c='red', label='Source')
ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c='blue', label='Target')

# Transformed points using original scale
transformed_points_original = scale_original * centered_source + mean_target
ax.scatter(transformed_points_original[:, 0], transformed_points_original[:, 1], transformed_points_original[:, 2],
           c='green', label='Transformed (Original Scale)', marker='x' , s=100)


# Set plot properties
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Point Transformation')
ax.legend()

plt.show()
