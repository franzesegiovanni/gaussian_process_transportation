
import numpy as np

def distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def resample(surface, num_points=20):
    # Calculate the total length of the original curve
    total_length = np.sum([distance(surface[i], surface[i+1]) for i in range(len(surface)-1)])

    # Calculate the spacing between points
    spacing = total_length / (num_points - 1)

    # Initialize variables for the new trajectory
    new_trajectory = [surface[0]]
    current_position = surface[0]
    remaining_distance = spacing

    # Iterate through the original curve to create the new resampled trajectory
    for point in surface[1:]:
        dist_to_next_point = distance(current_position, point)

        # Check if we've reached the desired spacing
        if remaining_distance <= dist_to_next_point:
            # Interpolate to find the new point
            t = remaining_distance / dist_to_next_point
            new_point = [
                current_position[0] + t * (point[0] - current_position[0]),
                current_position[1] + t * (point[1] - current_position[1])
            ]
            new_trajectory.append(new_point)
            current_position = new_point
            remaining_distance = spacing
        else:
            # Move to the next point
            current_position = point
            remaining_distance -= dist_to_next_point

    # Ensure that the new trajectory has the correct number of points
    while len(new_trajectory) < num_points:
        new_trajectory.append(surface[-1])

    # Convert the new trajectory to a numpy array
    new_trajectory = np.array(new_trajectory)
    return new_trajectory
