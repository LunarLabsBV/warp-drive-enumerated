import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d

# Generate sample data for directional changes (random for demonstration)
directional_changes = np.random.rand(10, 2)
original_direction_changes = np.random.rand(10, 2)

# Compute average displacement vector
avg_displacement_vector = np.mean(np.diff(directional_changes, axis=0), axis=0)

# Define Earth and Mars coordinates (arbitrary for visualization)
earth_coordinates = [0, 0, 0]
mars_coordinates = [50, 0, 0]  # Assuming Mars is 50 units away from Earth in the x-direction

# Define function to compute trajectory based on Gott time machine metric equation
def compute_trajectory(num_points, time_step):
    # Define parameters for trajectory generation
    r_max = 50  # Maximum radial distance
    theta_max = np.pi  # Maximum theta angle
    phi_max = 2 * np.pi  # Maximum phi angle

    # Generate random values for theta and phi
    theta_values = np.linspace(0, theta_max, num_points)
    phi_values = np.linspace(0, phi_max, num_points)

    # Compute trajectory coordinates based on Gott time machine metric equation
    trajectory_x = r_max * np.sin(theta_values + time_step) * np.cos(phi_values)
    trajectory_y = r_max * np.sin(theta_values + time_step) * np.sin(phi_values)
    trajectory_z = r_max * np.cos(theta_values + time_step)
    
    return trajectory_x, trajectory_y, trajectory_z

# Generate initial trajectory
num_points = 100
time_step = 0
trajectory_x, trajectory_y, trajectory_z = compute_trajectory(num_points, time_step)

# Create figure and subplots
fig = plt.figure(figsize=(15, 6))

# 3D Plot
ax1 = fig.add_subplot(121, projection='3d')

# Plot Earth and Mars
earth = ax1.scatter(*earth_coordinates, color='blue', label='Earth')
mars = ax1.scatter(*mars_coordinates, color='red', label='Mars')

# Plot Trajectory based on Gott time machine metric equation
trajectory, = ax1.plot(trajectory_x, trajectory_y, trajectory_z, color='green', linestyle='--', label='Trajectory (Gott Equation)')

# Label Axes and Add Title
ax1.set_xlabel('X (Distance)')
ax1.set_ylabel('Y (Distance)')
ax1.set_zlabel('Z (Distance)')
ax1.set_title('3D Trajectory - Earth to Mars')

# Add Legend
ax1.legend()

# 2D Plot
ax2 = fig.add_subplot(122)

# Plot Trajectory on xy-plane (z=0)
trajectory_xy, = ax2.plot(trajectory_x, trajectory_y, color='green', linestyle='--', label='Trajectory (xy-plane)')
earth_xy = ax2.scatter(*earth_coordinates[:2], color='blue', label='Earth')
mars_xy = ax2.scatter(*mars_coordinates[:2], color='red', label='Mars')

# Label Axes and Add Title
ax2.set_xlabel('X (Distance)')
ax2.set_ylabel('Y (Distance)')
ax2.set_title('2D Trajectory - Earth to Mars (xy-plane)')

# Add Legend
ax2.legend()

# Generate qubits along the trajectory
num_qubits = 10  # Number of qubits
qubit_coordinates_x = trajectory_x[::10]  # Sample every 10th point for qubits
qubit_coordinates_y = trajectory_y[::10]
qubit_coordinates_z = trajectory_z[::10]
qubit = ax1.scatter(qubit_coordinates_x, qubit_coordinates_y, qubit_coordinates_z, color='orange', marker='o')

# Animation loop
num_frames = 200  # Total number of frames
delay = 100  # Delay between frames in milliseconds

for frame in range(num_frames):
    # Update trajectory
    new_trajectory_x, new_trajectory_y, new_trajectory_z = compute_trajectory(num_points, time_step)
    trajectory.set_data(new_trajectory_x, new_trajectory_y)
    trajectory.set_3d_properties(new_trajectory_z)
    
    # Update qubits
    qubit_coordinates_x = new_trajectory_x[::10]  # Sample every 10th point for qubits
    qubit_coordinates_y = new_trajectory_y[::10]
    qubit_coordinates_z = new_trajectory_z[::10]
    qubit._offsets3d = (qubit_coordinates_x, qubit_coordinates_y, qubit_coordinates_z)

    # Update subplot
    trajectory_xy.set_data(new_trajectory_x, new_trajectory_y)
    earth_xy.set_offsets(earth_coordinates[:2])
    mars_xy.set_offsets(mars_coordinates[:2])
    
    plt.pause(delay / 1000)  # Pause for the specified delay

    time_step += 0.05  # Increment time step for the next frame

plt.show()
